"""Standalone trainer for OpenTSLM-SP on MISATO MD trajectories.

Two variants:
  v1a (default) — faithful: pure LM loss, affinity is the "Answer: X.XX"
                  suffix of the generated rationale, parsed at eval.
  v1b           — hybrid: adds a scalar regression head on the LLM's last
                  input-position hidden state. Joint loss
                  L = L_LM + lambda * MSE(pK_pred, pK_true).

Usage on SageMaker Studio (g5.xlarge):

  export OPENTSLM_MISATO_DATA=/opt/ml/input/data        # has features_*.npz
  python train_misato.py --variant v1a --epochs 3 --batch-size 4
  python train_misato.py --variant v1b --epochs 3 --batch-size 4 --lambda-reg 0.5

Checkpoints land in /opt/ml/checkpoints (auto-synced to S3 by SageMaker) when
that directory exists, otherwise in --output-dir.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Subset
from transformers import get_linear_schedule_with_warmup
from tqdm.auto import tqdm

from opentslm.model.llm.OpenTSLM import OpenTSLM
from opentslm.model.llm.OpenTSLMSP import OpenTSLMSP
from opentslm.model_config import (
    GRAD_CLIP_NORM, LR_ENCODER, LR_PROJECTOR, WARMUP_FRAC, WEIGHT_DECAY,
)
from opentslm.time_series_datasets.misato.MISATOMDQADataset import MISATOMDQADataset
from opentslm.time_series_datasets.util import (
    extend_time_series_to_match_patch_size_and_aggregate,
)

ANSWER_RE = re.compile(r"Answer:\s*(-?\d+(?:\.\d+)?)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=["v1a", "v1b"], default="v1a")
    p.add_argument("--warm-start", default="OpenTSLM/llama-3.2-1b-tsqa-sp",
                   help="OpenTSLM HF repo to warm-start from. Maps internally to a base LLM "
                        "(e.g. meta-llama/Llama-3.2-1B) which is gated and requires HF auth.")
    p.add_argument("--cold-start-llm", default=None,
                   help="If set, skip warm-start and init OpenTSLMSP directly with this HF LLM id.")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--eval-batch-size", type=int, default=8)
    p.add_argument("--lr-lora", type=float, default=1e-4)
    p.add_argument("--lr-head", type=float, default=1e-4)
    p.add_argument("--lambda-reg", type=float, default=0.5)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--max-new-tokens", type=int, default=160)
    p.add_argument("--subset-train", type=int, default=None,
                   help="Cap training samples for smoke testing.")
    p.add_argument("--subset-eval", type=int, default=None,
                   help="Cap eval samples per split.")
    p.add_argument("--output-dir", type=Path, default=Path("runs"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--no-eval", action="store_true")
    return p.parse_args()


def select_output_dir(base: Path, variant: str) -> Path:
    """Prefer the SageMaker checkpoint dir when present, else timestamped local dir."""
    sm_ckpt = Path("/opt/ml/checkpoints")
    if sm_ckpt.exists() and os.access(sm_ckpt, os.W_OK):
        out = sm_ckpt / variant
    else:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out = base / f"{variant}_{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def collate_misato(samples: list) -> list:
    """Convert numpy time series to torch + patch-pad. Returns the list of dicts."""
    return extend_time_series_to_match_patch_size_and_aggregate(samples)


def maybe_subset(dataset: MISATOMDQADataset, n: Optional[int]):
    if n is None or n >= len(dataset):
        return dataset
    return Subset(dataset, list(range(n)))


def parse_pK_from_text(text: str) -> Optional[float]:
    m = ANSWER_RE.search(text)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def metrics(preds: np.ndarray, trues: np.ndarray) -> dict:
    mask = ~np.isnan(preds)
    if mask.sum() < 2:
        return {"n_parsed": int(mask.sum()), "rmse": float("nan"),
                "mae": float("nan"), "pearson_r": float("nan")}
    p, t = preds[mask], trues[mask]
    rmse = float(np.sqrt(np.mean((p - t) ** 2)))
    mae = float(np.mean(np.abs(p - t)))
    r = float(np.corrcoef(p, t)[0, 1]) if p.std() > 0 else float("nan")
    return {"n_parsed": int(mask.sum()), "n_total": int(len(preds)),
            "rmse": rmse, "mae": mae, "pearson_r": r}


@torch.no_grad()
def evaluate(model: OpenTSLMSP, dataset, batch_size: int, max_new_tokens: int,
             label: str) -> dict:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_misato, num_workers=0)
    model.eval()
    string_preds: list[Optional[float]] = []
    head_preds: list[Optional[float]] = []
    trues: list[float] = []
    pdbs: list[str] = []
    have_head = model.regression_enabled

    for batch in tqdm(loader, desc=f"eval[{label}]", leave=False):
        # 1) parse pK from generated text (v1a metric)
        gen = model.generate(batch, max_new_tokens=max_new_tokens, do_sample=False)
        for text in gen:
            string_preds.append(parse_pK_from_text(text))
        # 2) regression-head pK (v1b metric)
        if have_head:
            for v in model.predict_pK(batch):
                head_preds.append(float(v))
        for b in batch:
            trues.append(float(b["pK"]))
            pdbs.append(b["pdb_id"])

    trues_arr = np.array(trues)
    out = {
        "split": label,
        "n": len(trues),
        "string_parse": metrics(np.array([np.nan if x is None else x for x in string_preds]), trues_arr),
    }
    if have_head:
        out["regression_head"] = metrics(np.array(head_preds), trues_arr)
    return out


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("⚠️  CUDA not available; running on CPU will be slow.")

    out_dir = select_output_dir(args.output_dir, args.variant)
    print(f"output dir: {out_dir}")

    if args.cold_start_llm:
        print(f"cold-start: OpenTSLMSP({args.cold_start_llm})")
        model = OpenTSLMSP(llm_id=args.cold_start_llm, device=device)
        model.enable_lora(lora_r=args.lora_r)
    else:
        print(f"warm-start from {args.warm_start}")
        model = OpenTSLM.load_pretrained(args.warm_start, device=device, enable_lora=True)
    if args.variant == "v1b":
        model.enable_regression(weight=args.lambda_reg)

    print("loading MISATO splits...")
    train_ds = MISATOMDQADataset(split="train", EOS_TOKEN=model.get_eos_token())
    val_ds = MISATOMDQADataset(split="validation", EOS_TOKEN=model.get_eos_token())
    test_ds = MISATOMDQADataset(split="test", EOS_TOKEN=model.get_eos_token())
    print(f"sizes: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    train_use = maybe_subset(train_ds, args.subset_train)
    val_use = maybe_subset(val_ds, args.subset_eval)
    test_use = maybe_subset(test_ds, args.subset_eval)

    train_loader = DataLoader(train_use, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_misato, num_workers=args.num_workers,
                              pin_memory=(device == "cuda"))

    # Param groups: encoder, projector, LoRA, head.
    param_groups = [
        {"params": list(model.encoder.parameters()), "lr": LR_ENCODER},
        {"params": list(model.projector.parameters()), "lr": LR_PROJECTOR},
        {"params": model.get_lora_parameters(), "lr": args.lr_lora},
    ]
    if model.regression_enabled:
        param_groups.append({"params": model.get_regression_parameters(), "lr": args.lr_head})
    optimizer = AdamW([g for g in param_groups if g["params"]],
                      weight_decay=WEIGHT_DECAY)

    total_steps = max(1, args.epochs * math.ceil(len(train_use) / args.batch_size))
    warmup_steps = max(1, int(WARMUP_FRAC * total_steps))
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    print(f"optim: groups={len(optimizer.param_groups)}  steps={total_steps}  warmup={warmup_steps}")

    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        n_seen = 0
        pbar = tqdm(train_loader, desc=f"train ep{epoch}")
        for batch in pbar:
            loss = model.compute_loss(batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            clip_grad_norm_([p for g in optimizer.param_groups for p in g["params"]],
                            GRAD_CLIP_NORM)
            optimizer.step()
            scheduler.step()
            running += float(loss.detach()) * len(batch)
            n_seen += len(batch)
            pbar.set_postfix(loss=f"{running/max(1, n_seen):.4f}")

        train_loss = running / max(1, n_seen)
        epoch_record: dict = {"epoch": epoch, "train_loss": train_loss}

        if not args.no_eval:
            for label, ds in (("val", val_use), ("test", test_use)):
                m = evaluate(model, ds, args.eval_batch_size, args.max_new_tokens, label)
                epoch_record[label] = m
                print(f"  {label}: {json.dumps(m)}")

        history.append(epoch_record)
        ckpt_path = out_dir / f"ckpt_ep{epoch}.pt"
        model.store_to_file(str(ckpt_path))
        print(f"  saved {ckpt_path}")

        (out_dir / "history.jsonl").write_text(
            "\n".join(json.dumps(r) for r in history) + "\n"
        )

    final_path = out_dir / "ckpt_final.pt"
    model.store_to_file(str(final_path))
    print(f"done. final checkpoint: {final_path}")


if __name__ == "__main__":
    main()

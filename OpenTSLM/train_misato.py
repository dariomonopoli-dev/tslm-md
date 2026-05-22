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

W&B experiment tracking (optional — silent no-op when --wandb-project is unset):

  python train_misato.py --variant v1a --epochs 5 \
    --wandb-project misato-opentslm \
    --wandb-tags v1a baseline g5xlarge

  # offline (writes to ./wandb/, run `wandb sync` later to upload):
  python train_misato.py --variant v1a --wandb-project misato-opentslm --wandb-offline

What gets logged: per-step train loss + learning rates; per-epoch train_loss,
val/test {RMSE, MAE, Pearson R, n_parsed} for both string-parse and regression-
head metrics; a wandb.Table of per-system (pdb_id, pK_true, pK_pred) samples
each epoch; final checkpoint + history.jsonl as a versioned artifact.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import socket
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

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    _WANDB_AVAILABLE = False

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
    # ─── v2 stack: multi-task heads + ranking loss + Δ-channels ───────
    p.add_argument("--multitask", action="store_true",
                   help="Enable dissoc/drift/aux_reg multi-task heads.")
    p.add_argument("--mt-dissoc-weight", type=float, default=0.1)
    p.add_argument("--mt-drift-weight", type=float, default=0.1)
    p.add_argument("--mt-aux-reg-weight", type=float, default=0.05)
    p.add_argument("--ranking-loss-weight", type=float, default=0.0,
                   help="Margin-based pair ranking loss weight on the regression "
                        "head's output. >0 implies --variant v1b (regression head required).")
    p.add_argument("--ranking-margin", type=float, default=0.5,
                   help="Ranking-loss margin in pK units.")
    p.add_argument("--add-deltas", action="store_true",
                   help="Set OPENTSLM_MISATO_ADD_DELTAS=1: append first-difference Δ "
                        "channels alongside originals (doubles channel count).")
    # ─── W&B experiment tracking ──────────────────────────────────────
    p.add_argument("--wandb-project", default=None,
                   help="W&B project name. If unset, W&B is disabled.")
    p.add_argument("--wandb-entity", default=None,
                   help="W&B entity (team or user). Optional.")
    p.add_argument("--wandb-run-name", default=None,
                   help="Override the auto-generated run name.")
    p.add_argument("--wandb-tags", nargs="+", default=None,
                   help="Tags to attach to the W&B run (space-separated).")
    p.add_argument("--wandb-log-interval", type=int, default=20,
                   help="Log per-step train metrics every N steps.")
    p.add_argument("--wandb-log-pred-samples", type=int, default=64,
                   help="How many val/test predictions to log as a sample table per epoch.")
    p.add_argument("--wandb-offline", action="store_true",
                   help="Run W&B in offline mode (no upload until 'wandb sync').")
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
             label: str, return_predictions: bool = False) -> dict:
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
    string_arr = np.array([np.nan if x is None else x for x in string_preds])
    out = {
        "split": label,
        "n": len(trues),
        "string_parse": metrics(string_arr, trues_arr),
    }
    if have_head:
        out["regression_head"] = metrics(np.array(head_preds), trues_arr)
    if return_predictions:
        out["_pdbs"] = pdbs
        out["_string_preds"] = string_arr
        out["_head_preds"] = np.array(head_preds) if have_head else None
        out["_trues"] = trues_arr
    return out


def init_wandb(args: argparse.Namespace, out_dir: Path) -> bool:
    """Initialize a W&B run if --wandb-project is set. Returns True on success."""
    if not args.wandb_project:
        return False
    if not _WANDB_AVAILABLE:
        print("⚠️  --wandb-project set but `wandb` is not installed. `pip install wandb`.")
        return False

    cfg = vars(args).copy()
    cfg.update({
        "hostname": socket.gethostname(),
        "cuda_device": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        ),
        "cuda_vram_gb": (
            round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
            if torch.cuda.is_available() else 0
        ),
        "out_dir": str(out_dir),
        "GRAD_CLIP_NORM": GRAD_CLIP_NORM,
        "LR_ENCODER": LR_ENCODER,
        "LR_PROJECTOR": LR_PROJECTOR,
        "WARMUP_FRAC": WARMUP_FRAC,
        "WEIGHT_DECAY": WEIGHT_DECAY,
    })

    run_name = args.wandb_run_name or f"{args.variant}_{time.strftime('%Y%m%d_%H%M%S')}"
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        tags=args.wandb_tags,
        config=cfg,
        mode="offline" if args.wandb_offline else "online",
    )
    # Define metric semantics so the W&B UI groups them sensibly.
    wandb.define_metric("train/step")
    wandb.define_metric("train/loss", step_metric="train/step")
    wandb.define_metric("train/lr_*", step_metric="train/step")
    wandb.define_metric("epoch")
    wandb.define_metric("epoch/*", step_metric="epoch")
    wandb.define_metric("val/*", step_metric="epoch")
    wandb.define_metric("test/*", step_metric="epoch")
    return True


def log_epoch_to_wandb(epoch: int, train_loss: float, eval_records: dict,
                       pred_sample_limit: int) -> None:
    """Push per-epoch metrics + a sample predictions table to W&B."""
    if not (_WANDB_AVAILABLE and wandb.run):
        return
    log: dict = {"epoch": epoch, "epoch/train_loss": train_loss}
    for split_name, rec in eval_records.items():
        if rec is None:
            continue
        sp = rec.get("string_parse", {})
        for k, v in sp.items():
            log[f"{split_name}/string_{k}"] = v
        rh = rec.get("regression_head")
        if rh:
            for k, v in rh.items():
                log[f"{split_name}/head_{k}"] = v
        # Prediction sample table
        pdbs = rec.get("_pdbs")
        sp_arr = rec.get("_string_preds")
        hd_arr = rec.get("_head_preds")
        tr_arr = rec.get("_trues")
        if pdbs is not None and tr_arr is not None and len(pdbs):
            n = min(pred_sample_limit, len(pdbs))
            cols = ["epoch", "pdb_id", "pK_true", "pK_string_pred"]
            if hd_arr is not None:
                cols.append("pK_head_pred")
            rows = []
            for i in range(n):
                row = [epoch, pdbs[i], float(tr_arr[i]),
                       (float(sp_arr[i]) if sp_arr is not None and
                        not np.isnan(sp_arr[i]) else None)]
                if hd_arr is not None:
                    row.append(float(hd_arr[i]))
                rows.append(row)
            log[f"{split_name}/preds_sample"] = wandb.Table(columns=cols, data=rows)
    wandb.log(log)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("⚠️  CUDA not available; running on CPU will be slow.")

    out_dir = select_output_dir(args.output_dir, args.variant)
    print(f"output dir: {out_dir}")

    wandb_enabled = init_wandb(args, out_dir)
    if wandb_enabled:
        print(f"W&B run: {wandb.run.name} ({wandb.run.url})")

    # Δ-channels are driven by an env var inside misato_loader.py
    if args.add_deltas:
        os.environ["OPENTSLM_MISATO_ADD_DELTAS"] = "1"
        print("Δ-channels: ENABLED (first-difference appended to each channel)")

    if args.cold_start_llm:
        print(f"cold-start: OpenTSLMSP({args.cold_start_llm})")
        model = OpenTSLMSP(llm_id=args.cold_start_llm, device=device)
        model.enable_lora(lora_r=args.lora_r)
    else:
        print(f"warm-start from {args.warm_start}")
        model = OpenTSLM.load_pretrained(args.warm_start, device=device, enable_lora=True)

    # Auto-promote to v1b if ranking loss is requested (requires regression head)
    needs_regression = args.variant == "v1b" or args.ranking_loss_weight > 0
    if needs_regression:
        model.enable_regression(weight=args.lambda_reg)
    if args.ranking_loss_weight > 0:
        model.enable_ranking_loss(weight=args.ranking_loss_weight,
                                  margin=args.ranking_margin)
    if args.multitask:
        model.enable_multitask(
            dissoc_weight=args.mt_dissoc_weight,
            drift_weight=args.mt_drift_weight,
            aux_reg_weight=args.mt_aux_reg_weight,
        )

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

    # Param groups: encoder, projector, LoRA, regression head, multi-task heads.
    param_groups = [
        {"params": list(model.encoder.parameters()), "lr": LR_ENCODER, "name": "encoder"},
        {"params": list(model.projector.parameters()), "lr": LR_PROJECTOR, "name": "projector"},
        {"params": model.get_lora_parameters(), "lr": args.lr_lora, "name": "lora"},
    ]
    if model.regression_enabled:
        param_groups.append({"params": model.get_regression_parameters(),
                             "lr": args.lr_head, "name": "regression"})
    if model.multitask_enabled:
        param_groups.append({"params": model.get_multitask_parameters(),
                             "lr": args.lr_head, "name": "multitask"})
    optimizer = AdamW([g for g in param_groups if g["params"]],
                      weight_decay=WEIGHT_DECAY)

    total_steps = max(1, args.epochs * math.ceil(len(train_use) / args.batch_size))
    warmup_steps = max(1, int(WARMUP_FRAC * total_steps))
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    print(f"optim: groups={len(optimizer.param_groups)}  steps={total_steps}  warmup={warmup_steps}")

    history: list[dict] = []
    global_step = 0
    epoch_start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        n_seen = 0
        epoch_start = time.time()
        pbar = tqdm(train_loader, desc=f"train ep{epoch}")
        want_breakdown = (model.regression_enabled
                          or model.multitask_enabled
                          or model.ranking_weight > 0)
        for batch in pbar:
            if want_breakdown:
                loss, breakdown = model.compute_loss(batch, return_breakdown=True)
            else:
                loss = model.compute_loss(batch)
                breakdown = {}
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            clip_grad_norm_([p for g in optimizer.param_groups for p in g["params"]],
                            GRAD_CLIP_NORM)
            optimizer.step()
            scheduler.step()
            loss_val = float(loss.detach())
            running += loss_val * len(batch)
            n_seen += len(batch)
            global_step += 1
            pbar.set_postfix(loss=f"{running/max(1, n_seen):.4f}")

            if wandb_enabled and global_step % args.wandb_log_interval == 0:
                step_log = {
                    "train/step": global_step,
                    "train/loss": loss_val,
                    "train/loss_running_avg": running / max(1, n_seen),
                    "train/epoch_progress": (epoch - 1) + n_seen / max(1, len(train_use)),
                }
                # Per-loss-component logging
                for k, v in breakdown.items():
                    step_log[f"train/loss_{k}"] = v
                # Per-param-group LRs, named when available
                for gi, g in enumerate(optimizer.param_groups):
                    group_name = g.get("name", f"group{gi}")
                    step_log[f"train/lr_{group_name}"] = g["lr"]
                wandb.log(step_log)

        train_loss = running / max(1, n_seen)
        epoch_time = time.time() - epoch_start
        epoch_record: dict = {"epoch": epoch, "train_loss": train_loss,
                              "epoch_seconds": epoch_time}

        eval_records: dict = {"val": None, "test": None}
        if not args.no_eval:
            for label, ds in (("val", val_use), ("test", test_use)):
                m = evaluate(model, ds, args.eval_batch_size, args.max_new_tokens,
                             label, return_predictions=wandb_enabled)
                # Strip prediction arrays before serializing to history.jsonl
                public_m = {k: v for k, v in m.items() if not k.startswith("_")}
                epoch_record[label] = public_m
                eval_records[label] = m
                print(f"  {label}: {json.dumps(public_m)}")

        if wandb_enabled:
            log_epoch_to_wandb(epoch, train_loss, eval_records,
                               args.wandb_log_pred_samples)

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

    if wandb_enabled:
        # Upload the final checkpoint + history as a versioned artifact.
        try:
            art = wandb.Artifact(
                name=f"misato-opentslm-{args.variant}",
                type="model",
                description=f"OpenTSLM-SP {args.variant} on MISATO ({args.epochs} epochs).",
            )
            art.add_file(str(final_path))
            art.add_file(str(out_dir / "history.jsonl"))
            wandb.log_artifact(art)
        except Exception as e:
            print(f"⚠️  wandb artifact upload failed: {e}")
        wandb.finish()


if __name__ == "__main__":
    main()

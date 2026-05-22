"""Wiring gate: overfit a single tiny batch with the juncliu Chronos checkpoint.

Confirms the patched OpenTSLMFlamingo + featurized.h5 + targets.json wiring
actually lets gradient flow through the perceiver/adapter onto a real loss.

Mirrors tslm_md.train_stage6's forward path exactly (same collate, same loss
extraction), so a pass here means the real training run is safe to launch.

Pass criterion:
    final_loss <= initial_loss / 3   AND   final_loss < 1.5

Usage:
    python scripts/overfit_single_batch.py \
        --starting-checkpoint ~/.cache/huggingface/hub/models--juncliu--llama-3.2-1b-ecg-flamingo-epoch-35/snapshots/cfcdf8f7141b729ae50da4e1ef4e3bdc2b638674/best_model.pt
"""

from __future__ import annotations

import argparse
import contextlib
import glob
import os
import sys
from pathlib import Path

import torch
import yaml

from opentslm.model.llm.OpenTSLMFlamingo import OpenTSLMFlamingo
from opentslm.model_config import PATCH_SIZE
from opentslm.time_series_datasets.util import (
    extend_time_series_to_match_patch_size_and_aggregate,
)

from tslm_md.dataset import MDCoTQADataset


DEFAULT_CKPT_GLOB = os.path.expanduser(
    "~/.cache/huggingface/hub/models--juncliu--llama-3.2-1b-ecg-flamingo-epoch-35"
    "/snapshots/*/best_model.pt"
)


def find_default_ckpt() -> Path | None:
    matches = glob.glob(DEFAULT_CKPT_GLOB)
    return Path(matches[0]) if matches else None


def load_checkpoint(model: torch.nn.Module, ckpt_path: Path) -> None:
    print(f"loading checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"  missing={len(missing)}  unexpected={len(unexpected)}")


def main(args: argparse.Namespace) -> int:
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    model = OpenTSLMFlamingo(
        llm_id=cfg["model"]["llm_id"],
        device=device,
        cross_attn_every_n_layers=cfg["model"]["cross_attn_every_n_layers"],
        encoder_type=cfg["model"].get("encoder_type", "chronos2"),
        freeze_lm_embeddings=cfg["model"].get("freeze_lm_embeddings", False),
    )

    ckpt_path = Path(args.starting_checkpoint) if args.starting_checkpoint else find_default_ckpt()
    if ckpt_path is None:
        print("WARN: no checkpoint provided and none found at default path. "
              "Continuing without loading weights.")
    else:
        load_checkpoint(model.model, ckpt_path)

    trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.model.parameters())
    print(f"trainable params: {trainable_params:,} / {total_params:,} "
          f"({100 * trainable_params / total_params:.1f}%)")

    train_ds = MDCoTQADataset(
        split="train",
        EOS_TOKEN=model.text_tokenizer.eos_token or "</s>",
        featurized_h5=cfg["data"]["featurized_h5"],
        targets_json=cfg["data"].get("targets_json", "data/targets.json"),
        splits_dir=cfg["data"]["splits_dir"],
        max_samples=args.batch_size,
    )
    print(f"train_ds size (capped to batch_size): {len(train_ds)}")
    if len(train_ds) < args.batch_size:
        print(f"WARN: only {len(train_ds)} samples available, requested {args.batch_size}")

    raw_batch = [train_ds[i] for i in range(len(train_ds))]
    batch = extend_time_series_to_match_patch_size_and_aggregate(raw_batch, patch_size=PATCH_SIZE)
    input_ids, images, attention_mask, labels = model.pad_and_apply_batch(batch, include_labels=False)
    print(f"batch tensors: input_ids={tuple(input_ids.shape)} images={tuple(images.shape)}")

    trainable = [p for p in model.model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)

    model.model.train()
    losses: list[float] = []
    use_bf16 = args.precision == "bf16" and device == "cuda"
    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_bf16 else contextlib.nullcontext()
    )
    print(f"overfitting for {args.steps} steps at lr={args.lr} "
          f"precision={'bf16' if use_bf16 else 'fp32'}...")
    for step in range(args.steps):
        optim.zero_grad(set_to_none=True)
        with amp_ctx:
            out = model.model(
                vision_x=images, lang_x=input_ids,
                attention_mask=attention_mask, labels=labels,
            )
            loss = getattr(out, "loss", None)
            if loss is None:
                loss = out[0] if isinstance(out, tuple) else out
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optim.step()

        losses.append(loss.item())
        if step % args.log_every == 0 or step == args.steps - 1:
            print(f"  step={step:>4d}  loss={loss.item():.4f}")

    initial = sum(losses[:3]) / min(3, len(losses))
    final = sum(losses[-3:]) / min(3, len(losses))
    drop = initial / max(final, 1e-9)
    print(f"\ninitial(avg first 3): {initial:.4f}")
    print(f"final  (avg last 3) : {final:.4f}")
    print(f"reduction factor    : {drop:.2f}x")

    passed = drop >= 3.0 and final < 1.5
    if passed:
        print("\n[OK] WIRING GATE PASSED — gradient flows, adapter learns, ready for stage6.")
        return 0
    print("\n[FAIL] WIRING GATE FAILED — loss did not collapse on a 4-sample batch.")
    print("       Investigate: are too many params frozen? lr too low? labels wrong shape?")
    return 1


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/stage6_md_cot.yaml")
    p.add_argument("--starting-checkpoint", default=None,
                   help="path to juncliu best_model.pt (default: auto-detect)")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    sys.exit(main(p.parse_args()))

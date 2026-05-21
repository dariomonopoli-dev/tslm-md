"""Stage-6 training entry point: fine-tune Chronos-encoder OpenTSLMFlamingo on MISATO.

Plugs MDCoTQADataset into OpenTSLM's CurriculumTrainer as a new stage.
Initialised from juncliu/llama-3.2-1b-ecg-flamingo-epoch-35 (the Chronos-2
encoder variant directly recommended by the OpenTSLM team).

Usage:
    python -m tslm_md.train_stage6 --config configs/stage6_md_cot.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from opentslm.model.llm.OpenTSLMFlamingo import OpenTSLMFlamingo
from opentslm.time_series_datasets.util import (
    extend_time_series_to_match_patch_size_and_aggregate,
)
from opentslm.model_config import PATCH_SIZE

from tslm_md.dataset import MDCoTQADataset


def load_checkpoint_into_model(model: torch.nn.Module, ckpt_path: str | Path) -> None:
    """Load juncliu/llama-3.2-1b-ecg-flamingo-epoch-35 style checkpoint."""
    ckpt_path = Path(ckpt_path)
    print(f"Loading checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"  loaded. missing keys: {len(missing)}, unexpected: {len(unexpected)}")
    if missing:
        print(f"  example missing: {missing[:3]}")
    if unexpected:
        print(f"  example unexpected: {unexpected[:3]}")


def main(args: argparse.Namespace) -> None:
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    print(f"config:\n{yaml.dump(cfg)}")

    # 1) instantiate model
    model = OpenTSLMFlamingo(
        llm_id=cfg["model"]["llm_id"],
        device=device,
        cross_attn_every_n_layers=cfg["model"]["cross_attn_every_n_layers"],
        encoder_type=cfg["model"].get("encoder_type", "chronos2"),
        freeze_lm_embeddings=cfg["model"].get("freeze_lm_embeddings", False),
    )
    if args.starting_checkpoint:
        load_checkpoint_into_model(model.model, args.starting_checkpoint)

    # 2) dataset + loader
    train_ds = MDCoTQADataset(
        split="train",
        EOS_TOKEN=model.text_tokenizer.eos_token or "</s>",
        featurized_h5=cfg["data"]["featurized_h5"],
        targets_json=cfg["data"].get("targets_json", "data/targets.json"),
        splits_dir=cfg["data"]["splits_dir"],
    )
    val_ds = MDCoTQADataset(
        split="validation",
        EOS_TOKEN=model.text_tokenizer.eos_token or "</s>",
        featurized_h5=cfg["data"]["featurized_h5"],
        targets_json=cfg["data"].get("targets_json", "data/targets.json"),
        splits_dir=cfg["data"]["splits_dir"],
    )
    print(f"train size: {len(train_ds)}, val size: {len(val_ds)}")

    collate = lambda batch: extend_time_series_to_match_patch_size_and_aggregate(
        batch, patch_size=PATCH_SIZE
    )
    train_loader = DataLoader(
        train_ds, batch_size=cfg["training"]["batch_size"], shuffle=True, collate_fn=collate
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["training"]["batch_size"], shuffle=False, collate_fn=collate
    )

    # 3) optimiser
    trainable = [p for p in model.model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(
        trainable,
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    accum = int(cfg["training"]["grad_accum_steps"])
    max_steps = int(cfg["training"]["max_steps"])
    log_every = int(cfg["training"]["log_every_steps"])
    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_every = int(cfg["training"]["ckpt_every_steps"])

    step = 0
    optim.zero_grad(set_to_none=True)
    print("training...")
    while step < max_steps:
        for batch in train_loader:
            input_ids, images, attention_mask, labels = model.pad_and_apply_batch(
                batch, include_labels=False
            )
            out = model.model(
                vision_x=images, lang_x=input_ids,
                attention_mask=attention_mask, labels=labels,
            )
            loss = getattr(out, "loss", None) or (out[0] if isinstance(out, tuple) else out)
            (loss / accum).backward()

            if (step + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, float(cfg["training"]["grad_clip_norm"]))
                optim.step()
                optim.zero_grad(set_to_none=True)

            if step % log_every == 0:
                print(f"step={step:>6d}  loss={loss.item():.4f}")

            if step > 0 and step % ckpt_every == 0:
                p = ckpt_dir / f"step_{step}.pt"
                torch.save(model.model.state_dict(), p)
                print(f"saved {p}")

            step += 1
            if step >= max_steps:
                break

    final = ckpt_dir / "final.pt"
    torch.save(model.model.state_dict(), final)
    print(f"done. final checkpoint at {final}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/stage6_md_cot.yaml")
    p.add_argument(
        "--starting-checkpoint", default=None,
        help="path to juncliu/llama-3.2-1b-ecg-flamingo-epoch-35 best_model.pt",
    )
    main(p.parse_args())

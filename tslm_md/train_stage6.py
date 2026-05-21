"""Stage-6 training entry point: fine-tune Chronos-encoder OpenTSLMFlamingo on MISATO.

Plugs MDCoTQADataset into OpenTSLM's CurriculumTrainer as a new stage.
Initialised from juncliu/llama-3.2-1b-ecg-flamingo-epoch-35 (the Chronos-2
encoder variant directly recommended by the OpenTSLM team).

Usage:
    python -m tslm_md.train_stage6 --config configs/stage6_md_cot.yaml
"""

from __future__ import annotations

import argparse
import contextlib
import json
import time
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader

from opentslm.model.llm.OpenTSLMFlamingo import OpenTSLMFlamingo
from opentslm.time_series_datasets.util import (
    extend_time_series_to_match_patch_size_and_aggregate,
)
from opentslm.model_config import PATCH_SIZE

from tslm_md.dataset import MDCoTQADataset
from tslm_md.featurize import normalise
from tslm_md.parse import parse_answer
from tslm_md.prompts import build_prompts, channel_descriptors


_CONF_COLOR = {"high": "tab:green", "medium": "tab:orange", "low": "tab:red", None: "tab:grey"}


def _load_val_ids(cfg: dict) -> list[tuple[str, str, float]]:
    """Load (pdb_id, pdb_id_h5, truth) tuples for val — same case-insensitive
    join as MDCoTQADataset._load_splits, but pulled out so the training loop
    can run model.generate over them without depending on the tokenised view."""
    targets_path = Path(cfg["data"].get("targets_json", "data/targets.json"))
    splits_dir = Path(cfg["data"]["splits_dir"])
    featurized_h5 = Path(cfg["data"]["featurized_h5"])
    with targets_path.open() as f:
        raw = json.load(f)
    targets_lower = {k.lower(): float(v["affinity_kcal_mol"]) for k, v in raw.items()}
    with h5py.File(featurized_h5, "r") as h5:
        h5_keys_lower = {k.lower(): k for k in h5.keys()}
    with (splits_dir / "val.txt").open() as f:
        raw_ids = [l.strip() for l in f if l.strip()]
    out = []
    for pid in raw_ids:
        key = pid.lower()
        h5_key = h5_keys_lower.get(key)
        truth = targets_lower.get(key)
        if h5_key is None or truth is None:
            continue
        out.append((key, h5_key, truth))
    return out


def _load_feature_stats(path: Path) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if not path.exists():
        return None, None
    with path.open() as f:
        fs = json.load(f)
    mean = torch.tensor(fs["mean"]).reshape(-1, 1).float()
    std = torch.tensor(fs["std"]).reshape(-1, 1).float()
    return mean, std


@torch.no_grad()
def _vis_predictions_to_wandb(
    model,
    val_ids: list[tuple[str, str, float]],
    featurized_h5: Path,
    feature_stats_mean: torch.Tensor | None,
    feature_stats_std: torch.Tensor | None,
    autocast_ctx_factory,
    wb,
    step: int,
    n_samples: int,
    max_new_tokens: int,
) -> None:
    """Run generate on a fixed subset of val and log scatter + metrics to wandb."""
    model.model.eval()
    truths: list[float] = []
    preds: list[float] = []
    confs: list[str | None] = []
    n_total = min(n_samples, len(val_ids))
    n_parsed = 0
    with h5py.File(featurized_h5, "r") as h5:
        for pid, h5_key, truth in val_ids[:n_total]:
            try:
                feats = torch.from_numpy(h5[h5_key][:])
                feats_norm = normalise(feats, mean=feature_stats_mean, std=feature_stats_std)
                pre, post = build_prompts(pid)
                batch_item = {
                    "time_series": feats_norm,
                    "time_series_text": channel_descriptors(),
                    "pre_prompt": pre,
                    "post_prompt": post,
                    "answer": "",
                }
                with autocast_ctx_factory():
                    raw = model.generate([batch_item], max_new_tokens=max_new_tokens)
                raw_text = raw[0] if isinstance(raw, list) else str(raw)
                aff, conf = parse_answer(raw_text)
                if aff is not None:
                    truths.append(truth)
                    preds.append(aff)
                    confs.append(conf)
                    n_parsed += 1
            except Exception:
                continue
    model.model.train()

    payload: dict = {
        "val/n_parsed": n_parsed,
        "val/n_total": n_total,
        "val/parse_rate": n_parsed / max(n_total, 1),
        "step": step,
    }
    if n_parsed >= 3:
        t = np.asarray(truths)
        p = np.asarray(preds)
        r_p = float(pearsonr(p, t)[0])
        r_s = float(spearmanr(p, t)[0])
        mae = float(np.mean(np.abs(p - t)))
        payload.update({"val/pearson_r": r_p, "val/spearman_rho": r_s, "val/mae": mae})

        fig, ax = plt.subplots(figsize=(6, 6), dpi=120)
        for conf in ["high", "medium", "low", None]:
            mask = np.array([c == conf for c in confs])
            if not mask.any():
                continue
            ax.scatter(t[mask], p[mask], s=30, alpha=0.75,
                       color=_CONF_COLOR[conf],
                       label=conf or "unspecified",
                       edgecolor="white", linewidth=0.4)
        lo = float(min(t.min(), p.min()))
        hi = float(max(t.max(), p.max()))
        pad = 0.05 * (hi - lo + 1e-6)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", alpha=0.3, lw=1, label="y = x")
        ax.set_xlabel("Ground truth (kcal/mol)")
        ax.set_ylabel("Predicted (kcal/mol)")
        ax.set_title(f"step={step}   r={r_p:.3f}   ρ={r_s:.3f}   MAE={mae:.2f}   n={n_parsed}")
        ax.legend(loc="best", frameon=False, fontsize=9)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2)
        fig.tight_layout()
        try:
            import wandb
            payload["val/scatter"] = wandb.Image(fig)
        except ImportError:
            pass
        plt.close(fig)

    wb.log(payload)


def _maybe_init_wandb(cfg: dict, args: argparse.Namespace):
    """Initialise wandb if enabled in config and the package is importable.

    Returns the wandb module (or a no-op shim) so call sites can use
    `wb.log({...})` uniformly.
    """
    wb_cfg = cfg.get("wandb", {})
    if not wb_cfg.get("enabled", False) or args.no_wandb:
        print("wandb disabled (config or --no-wandb)")
        return _WandbNoop()
    try:
        import wandb
    except ImportError:
        print("wandb not installed — `pip install wandb` to enable. Continuing without it.")
        return _WandbNoop()
    wandb.init(
        project=wb_cfg.get("project", "tslm-md"),
        name=wb_cfg.get("run_name", "stage6_md_cot"),
        tags=wb_cfg.get("tags", []),
        config=cfg,
    )
    return wandb


class _WandbNoop:
    def log(self, *_args, **_kwargs): pass
    def finish(self): pass


@torch.no_grad()
def _val_loss(model, val_loader, autocast_ctx_factory, max_batches: int = 32) -> float:
    """Cheap val-loss probe — does NOT generate, just computes teacher-forced loss."""
    model.model.eval()
    total, n = 0.0, 0
    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        input_ids, images, attention_mask, labels = model.pad_and_apply_batch(
            batch, include_labels=False
        )
        with autocast_ctx_factory():
            out = model.model(
                vision_x=images, lang_x=input_ids,
                attention_mask=attention_mask, labels=labels,
            )
        loss = getattr(out, "loss", None) or (out[0] if isinstance(out, tuple) else out)
        total += float(loss.item())
        n += 1
    model.model.train()
    return total / max(n, 1)


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

    eval_every = int(cfg["training"].get("eval_every_steps", 0))
    vis_every = int(cfg["training"].get("vis_every_steps", 0))
    vis_n_samples = int(cfg["training"].get("vis_n_samples", 20))
    vis_max_new_tokens = int(cfg["training"].get("vis_max_new_tokens", 50))
    precision = cfg["training"].get("precision", "bf16")
    use_bf16 = precision == "bf16" and device == "cuda"
    def _autocast_ctx():
        return (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if use_bf16 else contextlib.nullcontext()
        )
    print(f"precision={'bf16' if use_bf16 else 'fp32'}")

    val_ids_for_vis: list[tuple[str, str, float]] = []
    feature_stats_mean = None
    feature_stats_std = None
    if vis_every > 0:
        val_ids_for_vis = _load_val_ids(cfg)
        feature_stats_mean, feature_stats_std = _load_feature_stats(
            Path(cfg["data"].get("feature_stats_json", "data/feature_stats.json"))
        )
        print(f"vis enabled: every {vis_every} steps on {vis_n_samples} val samples "
              f"(val pool: {len(val_ids_for_vis)})")

    wb = _maybe_init_wandb(cfg, args)

    step = 0
    optim.zero_grad(set_to_none=True)
    print("training...")
    t_start = time.time()
    while step < max_steps:
        for batch in train_loader:
            input_ids, images, attention_mask, labels = model.pad_and_apply_batch(
                batch, include_labels=False
            )
            with _autocast_ctx():
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
                elapsed = time.time() - t_start
                steps_per_sec = (step + 1) / max(elapsed, 1e-6)
                steps_left = max(max_steps - step - 1, 0)
                eta_s = steps_left / max(steps_per_sec, 1e-9)
                eta_min = eta_s / 60.0
                eta_str = (f"{eta_min:.1f}min" if eta_min < 60
                           else f"{eta_min/60:.2f}h")
                print(f"step={step:>6d}/{max_steps}  loss={loss.item():.4f}  "
                      f"sps={steps_per_sec:.2f}  elapsed={elapsed:.0f}s  "
                      f"eta={eta_str}  ({steps_left} steps left)")
                wb.log({
                    "train/loss": float(loss.item()),
                    "train/lr": optim.param_groups[0]["lr"],
                    "train/steps_per_sec": steps_per_sec,
                    "train/elapsed_s": elapsed,
                    "train/eta_min": eta_min,
                    "train/steps_left": steps_left,
                    "step": step,
                })

            if eval_every > 0 and step > 0 and step % eval_every == 0:
                vl = _val_loss(model, val_loader, _autocast_ctx)
                print(f"  [val] step={step}  val_loss={vl:.4f}")
                wb.log({"val/loss": vl, "step": step})

            if vis_every > 0 and step > 0 and step % vis_every == 0 and val_ids_for_vis:
                t0 = time.time()
                _vis_predictions_to_wandb(
                    model, val_ids_for_vis,
                    Path(cfg["data"]["featurized_h5"]),
                    feature_stats_mean, feature_stats_std,
                    _autocast_ctx, wb, step, vis_n_samples, vis_max_new_tokens,
                )
                print(f"  [vis] step={step}  scatter+metrics logged ({time.time()-t0:.1f}s)")

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
    final_val = _val_loss(model, val_loader, _autocast_ctx)
    print(f"final val_loss={final_val:.4f}")
    wb.log({"val/loss": final_val, "step": step, "final": True})
    wb.finish()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/stage6_md_cot.yaml")
    p.add_argument(
        "--starting-checkpoint", default=None,
        help="path to juncliu/llama-3.2-1b-ecg-flamingo-epoch-35 best_model.pt",
    )
    p.add_argument("--no-wandb", action="store_true",
                   help="disable wandb even if enabled in config")
    main(p.parse_args())

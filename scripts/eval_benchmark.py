"""Evaluate a trained TSLM-MD checkpoint on an external benchmark.

Generic eval driver: takes any list of PDB IDs + ground-truth affinities,
runs the trained model's generate path on each, parses the answer, and
reports regression metrics (Pearson r, Spearman rho, MAE) plus parsing/
abstention statistics.

Designed so the same script handles MDbind (PDBbind v2016 core 285),
MISATO test split, MISATO Supp Data 2, or any custom benchmark — provided
each PDB id appears in our featurized.h5.

Usage:
    # MDbind PDBbind v2016 core (after populating data/benchmarks/pdbbind_v2016_core.txt)
    python scripts/eval_benchmark.py \\
        --checkpoint results/stage6_md_cot/checkpoints/final.pt \\
        --benchmark-ids data/benchmarks/pdbbind_v2016_core.txt \\
        --label-source targets

    # MISATO official test split (uses targets.json for ground truth)
    python scripts/eval_benchmark.py \\
        --checkpoint results/stage6_md_cot/checkpoints/final.pt \\
        --benchmark-ids data/splits/test.txt \\
        --label-source targets

    # Without fine-tune — base juncliu checkpoint as baseline
    python scripts/eval_benchmark.py \\
        --checkpoint ~/.cache/.../best_model.pt \\
        --benchmark-ids data/splits/val.txt

Reported numbers can go straight into the pitch slide.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml
from scipy.stats import pearsonr, spearmanr

from opentslm.model.llm.OpenTSLMFlamingo import OpenTSLMFlamingo

from tslm_md.featurize import normalise
from tslm_md.parse import parse_answer
from tslm_md.prompts import build_prompts, channel_descriptors


def load_benchmark_ids(path: Path) -> list[str]:
    """Load PDB IDs, one per line. Comments (#) and blanks ignored."""
    with path.open() as f:
        ids = []
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            ids.append(line)
    return ids


def load_targets(targets_json: Path) -> dict[str, float]:
    """Load {pdb_id_lower: affinity_kcal_mol} from build_training_targets.py output."""
    with targets_json.open() as f:
        raw = json.load(f)
    return {k.lower(): float(v["affinity_kcal_mol"]) for k, v in raw.items()}


def load_custom_labels(csv_path: Path) -> dict[str, float]:
    """Load {pdb_id_lower: affinity_kcal_mol} from a 2-column CSV (pdb_id, affinity_kcal_mol)."""
    out: dict[str, float] = {}
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row["pdb_id"].strip().lower()
            out[pid] = float(row["affinity_kcal_mol"])
    return out


def load_model(args: argparse.Namespace, cfg: dict, device: str) -> OpenTSLMFlamingo:
    model = OpenTSLMFlamingo(
        llm_id=cfg["model"]["llm_id"],
        device=device,
        cross_attn_every_n_layers=cfg["model"]["cross_attn_every_n_layers"],
        encoder_type=cfg["model"].get("encoder_type", "chronos2"),
        freeze_lm_embeddings=cfg["model"].get("freeze_lm_embeddings", False),
    )
    print(f"loading checkpoint: {args.checkpoint}")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    missing, unexpected = model.model.load_state_dict(state, strict=False)
    print(f"  missing={len(missing)}  unexpected={len(unexpected)}")
    model.model.eval()
    return model


def predict_one(
    model: OpenTSLMFlamingo,
    pdb_id_h5: str,
    h5: h5py.File,
    stats_mean: torch.Tensor | None,
    stats_std: torch.Tensor | None,
    max_new_tokens: int,
    autocast_ctx,
) -> tuple[float | None, str | None, str]:
    """Featurise + generate + parse. Returns (affinity, confidence, raw_text)."""
    feats = torch.from_numpy(h5[pdb_id_h5][:])
    feats_norm = normalise(feats, mean=stats_mean, std=stats_std)
    pre, post = build_prompts(pdb_id_h5)
    batch_item = {
        "time_series": feats_norm,
        "time_series_text": channel_descriptors(),
        "pre_prompt": pre,
        "post_prompt": post,
        "answer": "",
    }
    with torch.no_grad(), autocast_ctx():
        raw = model.generate([batch_item], max_new_tokens=max_new_tokens)
    raw_text = raw[0] if isinstance(raw, list) else str(raw)
    aff, conf = parse_answer(raw_text)
    return aff, conf, raw_text


def main(args: argparse.Namespace) -> int:
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    benchmark_ids = load_benchmark_ids(Path(args.benchmark_ids))
    print(f"benchmark: {args.benchmark_ids}  n_ids={len(benchmark_ids)}")

    if args.labels_csv:
        targets = load_custom_labels(Path(args.labels_csv))
        print(f"labels: custom CSV  n={len(targets)}")
    else:
        targets = load_targets(Path(args.targets_json))
        print(f"labels: {args.targets_json}  n={len(targets)}")

    feature_stats_mean: torch.Tensor | None = None
    feature_stats_std: torch.Tensor | None = None
    fs_path = Path(args.feature_stats_json)
    if fs_path.exists():
        with fs_path.open() as f:
            fs = json.load(f)
        feature_stats_mean = torch.tensor(fs["mean"]).reshape(-1, 1).float()
        feature_stats_std = torch.tensor(fs["std"]).reshape(-1, 1).float()
        print(f"feature stats: {fs_path}")
    else:
        print(f"feature stats: not found — using per-sample z-score")

    model = load_model(args, cfg, device)

    use_bf16 = args.precision == "bf16" and device == "cuda"
    def _autocast_ctx():
        return (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if use_bf16 else contextlib.nullcontext()
        )

    h5p = Path(args.featurized_h5)
    print(f"featurised h5: {h5p}")

    preds: list[float] = []
    truths: list[float] = []
    rows: list[dict] = []
    n_skipped_not_in_h5 = 0
    n_skipped_not_in_targets = 0
    n_parse_fail = 0

    with h5py.File(h5p, "r") as h5:
        h5_keys_lower = {k.lower(): k for k in h5.keys()}
        ids_to_eval = benchmark_ids
        if args.max_samples:
            ids_to_eval = ids_to_eval[: args.max_samples]
        t_start = time.time()
        for i, pid in enumerate(ids_to_eval):
            key = pid.lower()
            actual_h5_pid = h5_keys_lower.get(key)
            truth = targets.get(key)
            if actual_h5_pid is None:
                n_skipped_not_in_h5 += 1
                continue
            if truth is None:
                n_skipped_not_in_targets += 1
                continue
            try:
                aff, conf, raw = predict_one(
                    model, actual_h5_pid, h5,
                    feature_stats_mean, feature_stats_std,
                    args.max_new_tokens, _autocast_ctx,
                )
            except Exception as e:
                print(f"  [{i+1}/{len(ids_to_eval)}] {pid}: predict ERROR — {e}")
                n_parse_fail += 1
                continue
            if aff is None:
                n_parse_fail += 1
                rows.append({"pdb_id": pid, "truth": truth, "pred": None,
                             "confidence": conf, "raw": raw[:200]})
                continue
            preds.append(aff)
            truths.append(truth)
            rows.append({"pdb_id": pid, "truth": truth, "pred": aff,
                         "confidence": conf, "raw": raw[:200]})
            if (i + 1) % args.log_every == 0:
                elapsed = time.time() - t_start
                print(f"  [{i+1}/{len(ids_to_eval)}] elapsed={elapsed:.0f}s  "
                      f"so far Pearson r={pearsonr(preds, truths)[0]:.3f} (n={len(preds)})")

    print("\n" + "=" * 60)
    print(f"n total ids in benchmark   = {len(benchmark_ids)}")
    print(f"n evaluated (=parsed)      = {len(preds)}")
    print(f"n skipped not in h5        = {n_skipped_not_in_h5}")
    print(f"n skipped not in targets   = {n_skipped_not_in_targets}")
    print(f"n parse failures           = {n_parse_fail}")

    if len(preds) < 3:
        print("\n[FAIL] not enough successful predictions to compute metrics")
        return 1

    p = np.array(preds)
    t = np.array(truths)
    r_p, _ = pearsonr(p, t)
    r_s, _ = spearmanr(p, t)
    mae = float(np.mean(np.abs(p - t)))
    rmse = float(np.sqrt(np.mean((p - t) ** 2)))

    print(f"\nPearson  r  = {r_p:.4f}")
    print(f"Spearman ρ  = {r_s:.4f}")
    print(f"MAE         = {mae:.4f}  kcal/mol")
    print(f"RMSE        = {rmse:.4f}  kcal/mol")

    if args.out_csv:
        out_path = Path(args.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["pdb_id", "truth", "pred", "confidence", "raw"])
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"\nper-sample predictions written to {out_path}")

    print("\nContext (published numbers for comparison):")
    print("  MISATO paper 3D-CNN + QM:  Spearman ≈ 0.64")
    print("  MDbind Videonucy:          Pearson  ≈ 0.84   (PDBbind v2016 core, MD)")
    print("  MDbind Timenucy:           Pearson  ≈ 0.78")
    print("  Pafnucy (static baseline): Pearson  ≈ 0.75")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/stage6_md_cot.yaml")
    p.add_argument("--checkpoint", required=True,
                   help="path to .pt — either trained final.pt or juncliu best_model.pt baseline")
    p.add_argument("--benchmark-ids", required=True,
                   help="txt file, one PDB id per line (e.g., data/splits/test.txt)")
    p.add_argument("--featurized-h5", default="data/featurized.h5")
    p.add_argument("--targets-json", default="data/targets.json")
    p.add_argument("--labels-csv", default=None,
                   help="optional: 2-column CSV pdb_id,affinity_kcal_mol overriding targets.json")
    p.add_argument("--feature-stats-json", default="data/feature_stats.json")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=50)
    p.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--out-csv", default=None,
                   help="optional path to write per-sample predictions CSV")
    sys.exit(main(p.parse_args()))

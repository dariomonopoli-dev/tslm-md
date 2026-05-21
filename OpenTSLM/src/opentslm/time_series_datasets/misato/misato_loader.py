"""Disk loader for MISATO MD preprocessed artifacts.

Reads features_{split}.npz + samples_{split}.jsonl + norm_stats.json from
$OPENTSLM_MISATO_DATA (defaults to ./preprocessed) and returns three HF
Dataset objects ready for `QADataset._load_splits`.

Each row contains:
  pdb_id: str
  pK: float
  rationale: str
  dissociated, unstable, multi_ligand: bool
  channels_norm: list[list[float]]   # (4, 100), train-mean/std z-scored
  channel_means: list[float]         # (4,) post-clip per-sample means
  channel_stds: list[float]          # (4,) post-clip per-sample stds

Channel order is fixed: [rmsd_ligand, interaction_energy, distance, bSASA].
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Tuple

import numpy as np
from datasets import Dataset

DEFAULT_DATA_DIR = "preprocessed"
ENV_VAR = "OPENTSLM_MISATO_DATA"


def _data_dir() -> Path:
    return Path(os.environ.get(ENV_VAR, DEFAULT_DATA_DIR))


def _load_split(data_dir: Path, split: str, train_mean: np.ndarray, train_std: np.ndarray) -> Dataset:
    npz_path = data_dir / f"features_{split}.npz"
    jsonl_path = data_dir / f"samples_{split}.jsonl"
    if not npz_path.exists() or not jsonl_path.exists():
        raise FileNotFoundError(
            f"Missing MISATO artifacts for split={split}: {npz_path} or {jsonl_path}. "
            f"Run preprocess_misato.py and set {ENV_VAR} to the output directory."
        )

    data = np.load(npz_path, allow_pickle=True)
    channels = data["channels"].astype(np.float32)  # (N, 100, 4)
    sample_means = channels.mean(axis=1)            # (N, 4) — per-system, post-clip
    sample_stds = channels.std(axis=1) + 1e-6       # (N, 4)
    channels_norm = (channels - train_mean[None, None, :]) / train_std[None, None, :]
    channels_norm_per_channel = np.transpose(channels_norm, (0, 2, 1))  # (N, 4, 100)

    samples: list[dict] = []
    with jsonl_path.open() as f:
        for line in f:
            samples.append(json.loads(line))
    if len(samples) != len(channels):
        raise RuntimeError(
            f"Length mismatch for split={split}: npz has {len(channels)} systems, "
            f"jsonl has {len(samples)}."
        )

    rows: list[dict] = []
    for i, s in enumerate(samples):
        rows.append({
            "pdb_id": s["pdb_id"],
            "pK": float(s["pK"]),
            "rationale": s["rationale"],
            "dissociated": bool(s["dissociated"]),
            "unstable": bool(s["unstable"]),
            "multi_ligand": bool(s["multi_ligand"]),
            "channels_norm": channels_norm_per_channel[i].tolist(),
            "channel_means": sample_means[i].tolist(),
            "channel_stds": sample_stds[i].tolist(),
        })
    return Dataset.from_list(rows)


def load_misato_splits() -> Tuple[Dataset, Dataset, Dataset]:
    data_dir = _data_dir()
    norm = json.loads((data_dir / "norm_stats.json").read_text())
    train_mean = np.array(norm["train_mean"], dtype=np.float32)
    train_std = np.array(norm["train_std"], dtype=np.float32)
    train = _load_split(data_dir, "train", train_mean, train_std)
    val = _load_split(data_dir, "val", train_mean, train_std)
    test = _load_split(data_dir, "test", train_mean, train_std)
    return train, val, test

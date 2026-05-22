"""Disk loader for MISATO MD preprocessed artifacts.

Reads features_{split}.npz + samples_{split}.jsonl + norm_stats.json from
$OPENTSLM_MISATO_DATA (defaults to ./preprocessed) and returns three plain
`list[dict]` splits ready for `QADataset._load_splits`.

**Important: we deliberately return plain lists, not HF `Dataset` objects.**
`Dataset.from_list` triggers a PyArrow conversion that for nested float
lists can balloon 10-50x during construction, OOMing the SageMaker
g5.xlarge (16 GB RAM). `QADataset.__init__` only does `len(...)` and
`map(format_fn, ...)` on the result, so a plain list is a drop-in
substitute that keeps peak RAM well under 1 GB.

Each row contains (channel count C ∈ {4, 12}; D ∈ {C, 2C} with deltas):
  pdb_id:        str
  pK:            float
  rationale:     str
  dissociated, unstable, multi_ligand: bool
  ligand_drift:  bool                            -- from v2 facts.summary; False on v1
  bsasa_drift:   float                           -- bSASA(last 20) - bSASA(first 20), raw units
  label_source:  str                             -- "Kd" | "Ki" | "IC50_CP" | "unknown"
  channel_order: list[str] length D              -- names of channels in row order
  channels_norm: np.ndarray (D, 100) float32     -- train-z-scored, optionally + Δ-channels
  channel_means: list[float] length D            -- per-system post-clip means (raw scale)
  channel_stds:  list[float] length D
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np

DEFAULT_DATA_DIR = "preprocessed"
ENV_VAR = "OPENTSLM_MISATO_DATA"
DELTA_ENV_VAR = "OPENTSLM_MISATO_ADD_DELTAS"   # set "1" to enable Δ-channels


def _data_dir() -> Path:
    return Path(os.environ.get(ENV_VAR, DEFAULT_DATA_DIR))


def _deltas_enabled() -> bool:
    return os.environ.get(DELTA_ENV_VAR, "0") == "1"


def _compute_aux_targets(channels_raw: np.ndarray, channel_order: List[str]
                         ) -> Tuple[float, float]:
    """Compute aux multi-task targets from raw (un-normalized) channels.

    Returns (bsasa_drift, _placeholder). bsasa_drift = mean(bSASA[-20:]) -
    mean(bSASA[:20]) — captures contact-area dynamics; ~0 for stable
    systems, large negative for ligand loss.
    """
    if "bSASA" not in channel_order:
        return 0.0, 0.0
    bi = channel_order.index("bSASA")
    bsasa = channels_raw[:, bi]                              # (T,)
    drift = float(bsasa[-20:].mean() - bsasa[:20].mean())
    return drift, 0.0


def _load_split(data_dir: Path, split: str, train_mean: np.ndarray,
                train_std: np.ndarray, channel_order: List[str],
                add_deltas: bool) -> List[dict]:
    npz_path = data_dir / f"features_{split}.npz"
    jsonl_path = data_dir / f"samples_{split}.jsonl"
    if not npz_path.exists() or not jsonl_path.exists():
        raise FileNotFoundError(
            f"Missing MISATO artifacts for split={split}: {npz_path} or {jsonl_path}. "
            f"Run preprocess_misato.py or preprocess_v2.py and set {ENV_VAR}."
        )

    data = np.load(npz_path, allow_pickle=True)
    channels = data["channels"].astype(np.float32)            # (N, T, C)
    label_sources = (data["label_source"].astype(str).tolist()
                     if "label_source" in data.files
                     else ["unknown"] * channels.shape[0])
    N, T, C = channels.shape
    if C != len(channel_order):
        raise RuntimeError(
            f"channel-count mismatch for {split}: npz has {C}, "
            f"norm_stats says {len(channel_order)} ({channel_order})"
        )

    sample_means = channels.mean(axis=1)                      # (N, C) post-clip per-system
    sample_stds = channels.std(axis=1) + 1e-6                 # (N, C)
    channels_norm = (channels - train_mean[None, None, :]) / train_std[None, None, :]
    # (N, T, C) → (N, C, T) so each channel is a 1-D time series
    channels_norm = np.transpose(channels_norm, (0, 2, 1)).astype(np.float32, copy=False)

    # Optionally append per-channel first differences as additional channels.
    # First Δ is set equal to the second to preserve length T.
    delta_channel_order: List[str] = []
    if add_deltas:
        deltas = np.diff(channels_norm, axis=2, prepend=channels_norm[..., :1])  # (N, C, T)
        channels_norm = np.concatenate([channels_norm, deltas], axis=1)           # (N, 2C, T)
        delta_channel_order = [f"delta_{name}" for name in channel_order]

    full_channel_order = list(channel_order) + delta_channel_order
    # Means/stds for the Δ-channels too, so the per-channel description text
    # the model sees stays consistent with the input.
    if add_deltas:
        delta_means = np.diff(channels, axis=1, prepend=channels[:, :1, :]).mean(axis=1)  # (N, C)
        delta_stds = np.diff(channels, axis=1, prepend=channels[:, :1, :]).std(axis=1) + 1e-6
        sample_means = np.concatenate([sample_means, delta_means], axis=1)
        sample_stds = np.concatenate([sample_stds, delta_stds], axis=1)

    samples: List[dict] = []
    with jsonl_path.open() as f:
        for line in f:
            samples.append(json.loads(line))
    if len(samples) != N:
        raise RuntimeError(
            f"Length mismatch for split={split}: npz has {N} systems, "
            f"jsonl has {len(samples)}."
        )

    rows: List[dict] = []
    means_list = sample_means.tolist()
    stds_list = sample_stds.tolist()
    for i, s in enumerate(samples):
        bsasa_drift, _ = _compute_aux_targets(channels[i], channel_order)
        # ligand_drift exists in v2 samples (facts.summary), absent in v1.
        ligand_drift = bool(s.get("facts", {})
                             .get("summary", {})
                             .get("ligand_drift", False))
        rows.append({
            "pdb_id": s["pdb_id"],
            "pK": float(s["pK"]),
            "rationale": s["rationale"],
            "dissociated": bool(s["dissociated"]),
            "unstable": bool(s["unstable"]),
            "multi_ligand": bool(s["multi_ligand"]),
            "ligand_drift": ligand_drift,
            "bsasa_drift": bsasa_drift,
            "label_source": label_sources[i],
            "channel_order": full_channel_order,
            "channels_norm": channels_norm[i],  # numpy (D, T), zero-copy slice
            "channel_means": means_list[i],
            "channel_stds": stds_list[i],
        })
    return rows


def load_misato_splits() -> Tuple[List[dict], List[dict], List[dict]]:
    data_dir = _data_dir()
    norm = json.loads((data_dir / "norm_stats.json").read_text())
    train_mean = np.array(norm["train_mean"], dtype=np.float32)
    train_std = np.array(norm["train_std"], dtype=np.float32)
    channel_order = norm.get("channel_order",
                             ["rmsd_ligand", "interaction_energy", "distance", "bSASA"])
    add_deltas = _deltas_enabled()
    train = _load_split(data_dir, "train", train_mean, train_std,
                        channel_order, add_deltas)
    val = _load_split(data_dir, "val", train_mean, train_std,
                      channel_order, add_deltas)
    test = _load_split(data_dir, "test", train_mean, train_std,
                       channel_order, add_deltas)
    return train, val, test

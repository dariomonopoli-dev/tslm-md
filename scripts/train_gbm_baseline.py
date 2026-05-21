"""R1 disproof experiment: sklearn GradientBoostingRegressor baseline.

If GBM on aggregated [mean/std/min/max] of our 6 channels achieves val Pearson r >= 0.3,
the binding signal IS in our featurisation and TSLM should at least match.
If r < 0.1, our features are too thin and need extra channels (H-bonds, contact-map entropy).

This is the cheapest informative experiment in the spec — run at hour 4.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

import argparse
import json

import h5py
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

# Import the S3 data loader
from data.s3_data_loader import get_training_input, DEFAULT_S3_URI


def aggregate(feats: np.ndarray) -> np.ndarray:
    """[6, F] -> [24] (mean, std, min, max per channel)."""
    return np.concatenate([feats.mean(1), feats.std(1), feats.min(1), feats.max(1)])


def load_split(split_file: Path) -> list[str]:
    with split_file.open() as f:
        return [l.strip() for l in f if l.strip()]


def load_xy(
    pdb_ids: list[str],
    featurized_h5: Path,
    targets_json: Path,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Case-insensitive join over splits, featurized.h5, and targets.json.

    MISATO HDF5 and Zenodo splits are uppercase; kierandidi CSV is uppercase
    but we lowercase keys in build_training_targets.py. Build a lowercase
    canonical lookup for both sides.
    """
    with targets_json.open() as f:
        targets = json.load(f)
    targets_lower = {k.lower(): v for k, v in targets.items()}

    X, y, used = [], [], []
    with h5py.File(featurized_h5, "r") as h5:
        h5_keys_lower = {k.lower(): k for k in h5.keys()}
        for pid in pdb_ids:
            actual_h5_pid = h5_keys_lower.get(pid.lower())
            target = targets_lower.get(pid.lower())
            if actual_h5_pid is None or target is None:
                continue
            X.append(aggregate(h5[actual_h5_pid][:]))
            y.append(target["affinity_kcal_mol"])
            used.append(pid)
    return np.array(X), np.array(y), used


def main(args: argparse.Namespace) -> None:
    splits_dir = Path(args.splits_dir)
    h5p = Path(args.featurized_h5)
    tjp = Path(args.targets_json)
    
    # Configure S3 data input if using SageMaker
    if args.use_s3:
        train_data = get_training_input(args.s3_uri or DEFAULT_S3_URI)
        print(f"Configured S3 data input: {args.s3_uri or DEFAULT_S3_URI}")

    print("loading train")
    X_tr, y_tr, _ = load_xy(load_split(splits_dir / "train.txt"), h5p, tjp)
    print("loading val")
    X_va, y_va, _ = load_xy(load_split(splits_dir / "val.txt"), h5p, tjp)
    print(f"train shape={X_tr.shape}  val shape={X_va.shape}")

    if X_tr.size == 0 or X_va.size == 0:
        raise SystemExit("no data — did preprocess_features.py and build_training_targets.py run?")

    model = GradientBoostingRegressor(
        n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.9, random_state=42,
    )
    print("fitting GBM...")
    model.fit(X_tr, y_tr)

    p_tr = model.predict(X_tr)
    p_va = model.predict(X_va)
    r_tr = pearsonr(p_tr, y_tr)[0]
    r_va = pearsonr(p_va, y_va)[0]
    rho_va = spearmanr(p_va, y_va)[0]
    mae_va = float(np.mean(np.abs(p_va - y_va)))

    print()
    print(f"  train Pearson r = {r_tr:.4f}")
    print(f"  val   Pearson r = {r_va:.4f}")
    print(f"  val   Spearman  = {rho_va:.4f}")
    print(f"  val   MAE       = {mae_va:.4f}  kcal/mol")
    print()
    if r_va >= 0.3:
        print("✅ R1 PASS — binding signal IS in the 6 features. TSLM has a real target.")
    elif r_va >= 0.1:
        print("⚠ R1 MARGINAL — weak signal. Consider adding ch7=H-bonds, ch8=contact-map entropy.")
    else:
        print("❌ R1 FAIL — features too thin. Add more channels before training the TSLM.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--featurized-h5", default="data/featurized.h5")
    p.add_argument("--targets-json", default="data/targets.json")
    p.add_argument("--splits-dir", default="data/splits")
    p.add_argument("--use-s3", action="store_true", help="Use S3 data input with FastFile mode")
    p.add_argument("--s3-uri", help="S3 URI for dataset (defaults to MD.hdf5)")
    main(p.parse_args())

"""Batch-featurise MISATO MD.hdf5 -> data/featurized.h5 + data/feature_stats.json.

Output schema (featurized.h5):
    /<pdb_id>           : float32 dataset of shape [6, 30]

Also writes feature_stats.json with per-channel train-set mean/std for normalisation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

from tslm_md.featurize import featurize, F_SUB, N_CHANNELS


def main(args: argparse.Namespace) -> None:
    in_path = Path(args.misato_h5)
    out_path = Path(args.out_h5)
    splits_dir = Path(args.splits_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(in_path, "r") as f:
        pdb_ids = list(f.keys())
        if args.limit:
            pdb_ids = pdb_ids[: args.limit]
        print(f"featurising {len(pdb_ids)} pdb ids from {in_path}")

        train_ids = set()
        if splits_dir.exists():
            train_file = splits_dir / "train.txt"
            if train_file.exists():
                with train_file.open() as sf:
                    train_ids = {line.strip() for line in sf if line.strip()}

        feats_for_stats: list[np.ndarray] = []  # only train-set complexes
        with h5py.File(out_path, "w") as out:
            for pid in tqdm(pdb_ids):
                try:
                    feats = featurize(f[pid]).numpy()
                except Exception as e:
                    print(f"  skip {pid}: {e}")
                    continue
                if feats.shape != (N_CHANNELS, F_SUB):
                    print(f"  skip {pid}: bad shape {feats.shape}")
                    continue
                if not np.isfinite(feats).all():
                    print(f"  skip {pid}: non-finite values")
                    continue
                out.create_dataset(pid, data=feats, dtype="float32")
                if pid in train_ids or not train_ids:
                    feats_for_stats.append(feats)

    if not feats_for_stats:
        print("warning: no train-set complexes found for stats — using all")
        with h5py.File(out_path, "r") as out:
            feats_for_stats = [out[k][:] for k in out.keys()]

    stack = np.stack(feats_for_stats, axis=0)  # [N, 6, 30]
    mean = stack.mean(axis=(0, 2)).tolist()    # per-channel mean
    std = stack.std(axis=(0, 2)).tolist()
    stats_path = out_path.parent / "feature_stats.json"
    with stats_path.open("w") as sf:
        json.dump({"mean": mean, "std": std, "n_train": len(feats_for_stats)}, sf, indent=2)
    print(f"wrote {out_path} and {stats_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--misato-h5", default="data/misato/MD.hdf5")
    p.add_argument("--out-h5", default="data/featurized.h5")
    p.add_argument("--splits-dir", default="data/splits")
    p.add_argument("--limit", type=int, default=None, help="cap pdb ids for quick test runs")
    main(p.parse_args())

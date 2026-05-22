"""Tier-1 affinity baselines for the MISATO trajectory dataset.

============================  Where the data comes from  ============================

This script reads ONLY the preprocessed artifacts produced by
`preprocess_misato.py`, located at:

    ./preprocessed/features_{train,val,test}.npz

Each .npz contains:
    channels      (N, 100, 4) float32   per-frame channel values, post-clip
                                        channel order:
                                        [rmsd_ligand, interaction_energy,
                                         distance, bSASA]
    pK            (N,) float32          target label, from misato-affinity CSV
                                        priority Kd > Ki > IC50,  pK = 9-log10(nM)
    pdb_ids       (N,) str
    dissociated, unstable, multi_ligand (N,) bool

Provenance chain:
    MD.hdf5 (124 GB, MISATO Zenodo)
      → preprocess_misato.py  (clips, joins with affinity_data.csv, splits)
      → preprocessed/features_{split}.npz   <-- WHAT THIS SCRIPT READS

Splits exactly match PROJECT_BRIEF §4.1 / DATASET.md §2:
    train 13,758  val 1,595  test 1,612

================================  What it computes  ================================

Three baselines our trained OpenTSLM-SP model has to beat:

    [1] predict_train_mean   "predict the train-set mean pK for every system"
                             Floor for any learner. DATASET.md §7 quotes test RMSE 1.93.
    [2] ols_means            OLS on per-trajectory means of the 4 channels
                             (4 features + intercept = 5 numbers). DATASET.md §8
                             quotes test RMSE 1.791  Pearson r 0.260.
    [3] mlp_engineered       2-layer MLP on 20 hand-engineered features
                             (mean / std / slope / min / max per channel).
                             Tests whether the OpenTSLM encoder learned anything
                             beyond what summary stats already give you.

Output: prints a small Markdown table and (optionally) writes the same to a JSON
file.

Usage:
    python eval_baselines.py
    python eval_baselines.py --out baselines.json
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

CHANNEL_ORDER = ["rmsd_ligand", "interaction_energy", "distance", "bSASA"]


@dataclass
class Split:
    channels: np.ndarray   # (N, 100, 4)
    pK: np.ndarray         # (N,)
    pdb_ids: np.ndarray    # (N,)


def load_split(data_dir: Path, split: str) -> Split:
    d = np.load(data_dir / f"features_{split}.npz", allow_pickle=True)
    return Split(
        channels=d["channels"].astype(np.float32),
        pK=d["pK"].astype(np.float32),
        pdb_ids=d["pdb_ids"],
    )


def metrics(pred: np.ndarray, true: np.ndarray) -> dict:
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
    mae = float(np.mean(np.abs(pred - true)))
    r = float(np.corrcoef(pred, true)[0, 1]) if pred.std() > 0 else float("nan")
    return {"n": int(len(true)), "rmse": rmse, "mae": mae, "pearson_r": r}


def trajectory_means(channels: np.ndarray) -> np.ndarray:
    return channels.mean(axis=1)  # (N, 4)


def engineered_features(channels: np.ndarray) -> np.ndarray:
    """20 features per system: mean/std/slope/min/max of each of 4 channels."""
    N, T, C = channels.shape
    frames = np.arange(T, dtype=np.float32)
    feats = []
    feats.append(channels.mean(axis=1))   # (N, C)
    feats.append(channels.std(axis=1))    # (N, C)
    # OLS slope per (sample, channel)
    f_mean = frames.mean()
    f_centered = frames - f_mean
    denom = (f_centered ** 2).sum()
    c_centered = channels - channels.mean(axis=1, keepdims=True)
    slope = (c_centered * f_centered[None, :, None]).sum(axis=1) / denom  # (N, C)
    feats.append(slope)
    feats.append(channels.min(axis=1))
    feats.append(channels.max(axis=1))
    return np.concatenate(feats, axis=1).astype(np.float32)  # (N, 5*C)


def run_baselines(data_dir: Path) -> dict:
    train = load_split(data_dir, "train")
    val = load_split(data_dir, "val")
    test = load_split(data_dir, "test")
    print(f"sizes: train={len(train.pK)}  val={len(val.pK)}  test={len(test.pK)}")
    print(f"train pK mean={train.pK.mean():.3f} std={train.pK.std():.3f}")
    print(f" test pK mean={test.pK.mean():.3f} std={test.pK.std():.3f}")

    results = {}

    # --- [1] predict train mean ---
    mu = float(train.pK.mean())
    results["predict_train_mean"] = {
        "val": metrics(np.full_like(val.pK, mu), val.pK),
        "test": metrics(np.full_like(test.pK, mu), test.pK),
        "params": {"mu": mu},
    }

    # --- [2] OLS on 4 trajectory means ---
    Xtr = trajectory_means(train.channels)
    Xva = trajectory_means(val.channels)
    Xte = trajectory_means(test.channels)
    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xva_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xva), scaler.transform(Xte)
    ols = LinearRegression().fit(Xtr_s, train.pK)
    results["ols_means"] = {
        "val": metrics(ols.predict(Xva_s), val.pK),
        "test": metrics(ols.predict(Xte_s), test.pK),
        "coefficients_zscored": dict(zip(CHANNEL_ORDER, ols.coef_.tolist())),
        "intercept": float(ols.intercept_),
    }

    # --- [3] MLP on 20 engineered features ---
    Xtr_e = engineered_features(train.channels)
    Xva_e = engineered_features(val.channels)
    Xte_e = engineered_features(test.channels)
    scaler_e = StandardScaler().fit(Xtr_e)
    Xtr_e, Xva_e, Xte_e = scaler_e.transform(Xtr_e), scaler_e.transform(Xva_e), scaler_e.transform(Xte_e)
    mlp = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        max_iter=200,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=0,
    ).fit(Xtr_e, train.pK)
    results["mlp_engineered"] = {
        "val": metrics(mlp.predict(Xva_e), val.pK),
        "test": metrics(mlp.predict(Xte_e), test.pK),
        "n_features": int(Xtr_e.shape[1]),
        "hidden": [64, 32],
        "stopped_at_iter": int(mlp.n_iter_),
    }

    return results


def print_table(results: dict) -> None:
    print()
    print("| baseline           | split |    n |  RMSE |   MAE | Pearson r |")
    print("|--------------------|-------|-----:|------:|------:|----------:|")
    for name, r in results.items():
        for split in ("val", "test"):
            m = r[split]
            print(f"| {name:<18s} | {split:<5s} | {m['n']:>4d} | {m['rmse']:.3f} | {m['mae']:.3f} | "
                  f"{m['pearson_r']:>+.3f}    |")
    print()
    print("Compare against DATASET.md §8:  ols_means test RMSE 1.791, Pearson 0.260.")
    print("Our OpenTSLM-SP runs must beat ols_means.test.rmse to claim the encoder learned anything.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("preprocessed"))
    ap.add_argument("--out", type=Path, default=None,
                    help="Optional: write results as JSON.")
    args = ap.parse_args()
    results = run_baselines(args.data_dir)
    print_table(results)
    if args.out:
        args.out.write_text(json.dumps(results, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

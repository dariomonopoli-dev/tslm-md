"""v2 audit — does the expanded 12-channel set carry more pK signal than v1's 4?

The gate question: do any of the 8 new channels beat bSASA's 0.27 Pearson on test?
"""
import json
import sys
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr

ROOT = Path(__file__).parent
PREP = ROOT / "preprocessed_v2"

meta = json.loads((PREP / "metadata.json").read_text())
order = meta["channel_order"]
print(f"v2 metadata: {meta['version']}, {meta['n_channels']} channels, "
      f"counts={meta['counts']}")
print(f"label sources: train={meta['label_source_counts_train']}, "
      f"val={meta['label_source_counts_val']}, "
      f"test={meta['label_source_counts_test']}")
print()

# Load per-split features + pK
data = {}
for split in ("train", "val", "test"):
    z = np.load(PREP / f"features_{split}.npz", allow_pickle=True)
    data[split] = {"channels": z["channels"], "pK": z["pK"],
                   "label_source": z["label_source"]}

# pK distribution on the corrected labels
print("=" * 78)
print("pK distribution AFTER Cheng-Prusoff correction (v2 labels):")
print("=" * 78)
for split in ("train", "val", "test"):
    p = data[split]["pK"]
    print(f"  {split:<5} n={len(p):>6,d}  mean={p.mean():.3f}  std={p.std():.3f}  "
          f"median={np.median(p):.3f}  min={p.min():.2f}  max={p.max():.2f}")

# Shift quantification
train_m = data["train"]["pK"].mean()
val_m = data["val"]["pK"].mean()
test_m = data["test"]["pK"].mean()
print(f"\n  train→val mean shift: {val_m - train_m:+.3f} pK "
      f"(v1 was -1.147 pK)")
print(f"  train→test mean shift: {test_m - train_m:+.3f} pK "
      f"(v1 was -1.037 pK)")
print()

# ---- THE GATE QUESTION: channel→pK Pearson per split ----
print("=" * 78)
print("CHANNEL → pK Pearson R per split (the gate)")
print("=" * 78)
print(f"{'channel':<30} {'train R':>9} {'val R':>9} {'test R':>9}  status")
print("-" * 78)
test_baselines = []
for ci, name in enumerate(order):
    rs = {}
    for split in ("train", "val", "test"):
        per_sys = data[split]["channels"][..., ci].mean(axis=1)
        rs[split] = pearsonr(per_sys, data[split]["pK"])[0]
    new = ci >= 4
    flag = ""
    if new and abs(rs["test"]) > 0.27:
        flag = "  ✓ BEATS bSASA-v1 (0.27)"
    elif new and abs(rs["test"]) > 0.20:
        flag = "  · meaningful"
    elif new and abs(rs["test"]) < 0.10:
        flag = "  × weak"
    print(f"{name:<30} {rs['train']:>+9.4f} {rs['val']:>+9.4f} {rs['test']:>+9.4f}{flag}")
    if name == "bSASA":
        test_baselines.append(("bSASA", rs["test"]))

# ---- TIME-SERIES STD per channel (does mobility predict pK?) ----
print()
print("=" * 78)
print("CHANNEL TIME-SERIES STD → pK Pearson (does in-frame variability help?)")
print("=" * 78)
print(f"{'channel':<30} {'train R':>9} {'val R':>9} {'test R':>9}")
print("-" * 78)
for ci, name in enumerate(order):
    rs = {}
    for split in ("train", "val", "test"):
        per_sys = data[split]["channels"][..., ci].std(axis=1)
        rs[split] = pearsonr(per_sys, data[split]["pK"])[0]
    print(f"{name:<30} {rs['train']:>+9.4f} {rs['val']:>+9.4f} {rs['test']:>+9.4f}")

# ---- BEST LINEAR COMBINATION (oracle ceiling estimate) ----
print()
print("=" * 78)
print("ORACLE LINEAR CEILING (OLS of per-system means on pK)")
print("=" * 78)
print("Trained on train. Reports Pearson per split. This is the best-case")
print("a linear readout on these channels can achieve.")
print()
from sklearn.linear_model import LinearRegression
X_train = data["train"]["channels"].mean(axis=1)
y_train = data["train"]["pK"]
for ci, name in enumerate(order):
    # Standardize columns so the coefficients are comparable
    pass
lr = LinearRegression().fit(X_train, y_train)
for split in ("train", "val", "test"):
    X = data[split]["channels"].mean(axis=1)
    y = data[split]["pK"]
    pred = lr.predict(X)
    r = pearsonr(pred, y)[0]
    rmse = float(np.sqrt(((pred - y) ** 2).mean()))
    print(f"  {split:<5} R={r:+.4f}  RMSE={rmse:.3f}")

# Compare to v1's 4-channel-only oracle
print()
print("Same OLS but on ORIGINAL 4 channels only (for comparison with v1 ceiling):")
X_train_v1 = X_train[:, :4]
lr_v1 = LinearRegression().fit(X_train_v1, y_train)
for split in ("train", "val", "test"):
    X = data[split]["channels"].mean(axis=1)[:, :4]
    y = data[split]["pK"]
    pred = lr_v1.predict(X)
    r = pearsonr(pred, y)[0]
    rmse = float(np.sqrt(((pred - y) ** 2).mean()))
    print(f"  {split:<5} R={r:+.4f}  RMSE={rmse:.3f}")

# Coefficient sizes — which channels are pulling weight in the 12-channel OLS?
print()
print("=" * 78)
print("STANDARDIZED LINEAR COEFFICIENTS (12-channel OLS)")
print("=" * 78)
X_std = (X_train - X_train.mean(0)) / (X_train.std(0) + 1e-9)
y_std = (y_train - y_train.mean()) / (y_train.std() + 1e-9)
lr_s = LinearRegression().fit(X_std, y_std)
order_pairs = sorted(enumerate(order),
                     key=lambda kv: -abs(lr_s.coef_[kv[0]]))
for ci, name in order_pairs:
    new = ci >= 4
    tag = "NEW" if new else "v1 "
    print(f"  [{tag}] {name:<30}  std-coef = {lr_s.coef_[ci]:+.4f}")

"""Data-quality audit for MISATO MD-affinity training set.

Answers:
  1. Kd / Ki / IC50 fraction per split (label-noise contributor)
  2. bSASA channel quality (pre-clip negatives, post-clip saturation)
  3. Per-split pK distribution (confirm train/test shift)
  4. Channel correlations on train (effective dimensionality)
  5. Cross-cut: do bSASA-broken systems cluster in any split / pK range?
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
PREP = ROOT / "preprocessed"
AFF_CSV = ROOT / "misato-affinity" / "data" / "affinity_data.csv"


def load_split_pdbs(split: str) -> list[str]:
    pdbs = []
    with open(PREP / f"samples_{split}.jsonl") as f:
        for line in f:
            pdbs.append(json.loads(line)["pdb_id"])
    return pdbs


def load_split_pks(split: str) -> np.ndarray:
    pks = []
    with open(PREP / f"samples_{split}.jsonl") as f:
        for line in f:
            pks.append(json.loads(line)["pK"])
    return np.array(pks)


def load_features(split: str) -> np.ndarray:
    z = np.load(PREP / f"features_{split}.npz")
    # channel_order = [rmsd_ligand, interaction_energy, distance, bSASA]
    return z[z.files[0]]  # shape (N, 100, 4)


# ---------------------------------------------------------------- #
# 1. Assay-type composition per split                              #
# ---------------------------------------------------------------- #
print("=" * 72)
print("1. ASSAY-TYPE COMPOSITION PER SPLIT")
print("=" * 72)
print(f"Priority rule: Kd > Ki > IC50 (first non-zero wins)\n")

df = pd.read_csv(AFF_CSV, sep=";", engine="python", on_bad_lines="skip")
df["PDBid"] = df["PDBid"].str.upper()
# Reproduce the priority assignment used during preprocessing
def assay_kind(row):
    if row["Kd (nM)"] and row["Kd (nM)"] > 0:
        return "Kd"
    if row["Ki (nM)"] and row["Ki (nM)"] > 0:
        return "Ki"
    if row["IC50 (nM)"] and row["IC50 (nM)"] > 0:
        return "IC50"
    return "none"

df["assay"] = df.apply(assay_kind, axis=1)
print(f"Overall CSV ({len(df)} PDBs): "
      f"{(df['assay'] == 'Kd').sum()} Kd, "
      f"{(df['assay'] == 'Ki').sum()} Ki, "
      f"{(df['assay'] == 'IC50').sum()} IC50, "
      f"{(df['assay'] == 'none').sum()} none")
print()

per_split = {}
for split in ["train", "val", "test"]:
    pdbs = load_split_pdbs(split)
    sub = df[df["PDBid"].isin([p.upper() for p in pdbs])].copy()
    matched = len(sub)
    counts = sub["assay"].value_counts().to_dict()
    per_split[split] = counts
    print(f"{split:<5} (N={len(pdbs)}, matched {matched} in CSV):")
    total_with_label = sum(v for k, v in counts.items() if k != "none")
    for kind in ["Kd", "Ki", "IC50", "none"]:
        n = counts.get(kind, 0)
        pct = 100 * n / matched if matched else 0
        print(f"  {kind:<5} {n:>6,d}  ({pct:5.1f}%)")
    if total_with_label:
        ic50_frac = counts.get("IC50", 0) / total_with_label
        print(f"  → IC50 fraction (of labeled): {100 * ic50_frac:.1f}%")
    print()

# ---------------------------------------------------------------- #
# 2. bSASA channel quality                                          #
# ---------------------------------------------------------------- #
print("=" * 72)
print("2. bSASA CHANNEL QUALITY (clip bounds were [0, 2500] Å²)")
print("=" * 72)
# norm_stats.json holds train-set mean/std *post-clip*. To see pre-clip
# damage we need the raw bSASA — which the preprocessed features don't have.
# But we can inspect post-clip saturation: how many systems sit at the clip bound?
norm = json.loads((PREP / "norm_stats.json").read_text())
print(f"norm_stats keys: {list(norm.keys())}\n")
print(f"norm_stats: {json.dumps(norm, indent=2)}\n")

bsasa_artifacts = {}
for split in ["train", "val", "test"]:
    feats = load_features(split)  # (N, 100, 4) — post-clip, NOT yet z-scored (let's verify)
    bsasa = feats[..., 3]  # channel 3 = bSASA per metadata
    # Per-system fraction of frames at the [0, 2500] clip bounds
    at_zero = (bsasa == 0).mean(axis=1)
    at_max = (bsasa == 2500).mean(axis=1)
    near_zero = (bsasa < 1.0).mean(axis=1)  # broader, in case features are z-scored
    near_max = (bsasa > 2499).mean(axis=1)

    n_zero_dominant = (at_zero > 0.5).sum()  # >50% frames at zero
    n_max_dominant = (at_max > 0.5).sum()    # >50% frames at max
    pdbs = load_split_pdbs(split)
    bsasa_artifacts[split] = {
        "shape": feats.shape,
        "bsasa_range": (float(bsasa.min()), float(bsasa.max())),
        "n_zero_dominant": int(n_zero_dominant),
        "n_max_dominant": int(n_max_dominant),
        "zero_dominant_pdbs": [pdbs[i] for i in np.where(at_zero > 0.5)[0][:5]],
        "max_dominant_pdbs": [pdbs[i] for i in np.where(at_max > 0.5)[0][:5]],
    }
    print(f"{split}: features shape {feats.shape}")
    print(f"  bSASA range: [{bsasa.min():.2f}, {bsasa.max():.2f}]")
    print(f"  systems with >50% frames at 0    : {n_zero_dominant:>6,d}  ({100 * n_zero_dominant / len(pdbs):.1f}%)")
    print(f"  systems with >50% frames at 2500 : {n_max_dominant:>6,d}  ({100 * n_max_dominant / len(pdbs):.1f}%)")
    if n_zero_dominant > 0:
        print(f"  zero-dominant sample PDBs: {bsasa_artifacts[split]['zero_dominant_pdbs']}")
    if n_max_dominant > 0:
        print(f"  max-dominant sample PDBs : {bsasa_artifacts[split]['max_dominant_pdbs']}")
    print()

# ---------------------------------------------------------------- #
# 3. pK distribution per split                                      #
# ---------------------------------------------------------------- #
print("=" * 72)
print("3. pK DISTRIBUTION PER SPLIT (confirm/quantify shift)")
print("=" * 72)
pks = {s: load_split_pks(s) for s in ["train", "val", "test"]}
for split, p in pks.items():
    print(f"{split:<5}  n={len(p):>6,d}  mean={p.mean():.3f}  std={p.std():.3f}  "
          f"min={p.min():.2f}  q25={np.percentile(p, 25):.2f}  "
          f"med={np.median(p):.2f}  q75={np.percentile(p, 75):.2f}  "
          f"max={p.max():.2f}")
print()
print(f"train→val mean shift: {pks['val'].mean() - pks['train'].mean():+.3f} pK")
print(f"train→test mean shift: {pks['test'].mean() - pks['train'].mean():+.3f} pK")
print()
print("Histogram (bins of 1 pK):")
bins = np.arange(0, 13, 1)
print(f"  bin     train       val      test")
for i in range(len(bins) - 1):
    lo, hi = bins[i], bins[i + 1]
    counts_line = []
    for split in ["train", "val", "test"]:
        p = pks[split]
        n = ((p >= lo) & (p < hi)).sum()
        counts_line.append(f"{n:>6,d}  ({100 * n / len(p):4.1f}%)")
    print(f"  {lo:>2.0f}-{hi:<2.0f}  " + "  ".join(counts_line))

# ---------------------------------------------------------------- #
# 4. Channel correlations on train (effective dimensionality)       #
# ---------------------------------------------------------------- #
print()
print("=" * 72)
print("4. CHANNEL CORRELATIONS ON TRAIN (per-system means)")
print("=" * 72)
feats_tr = load_features("train")
ch_names = ["rmsd_ligand", "interaction_energy", "distance", "bSASA"]
per_system = feats_tr.mean(axis=1)  # (N, 4)
corr = np.corrcoef(per_system.T)
print("Pearson R between per-system channel means:")
print(f"                  {'  '.join(f'{n[:8]:>10}' for n in ch_names)}")
for i, n in enumerate(ch_names):
    row = "  ".join(f"{corr[i, j]:>+10.3f}" for j in range(4))
    print(f"  {n[:14]:<14}  {row}")
print()
# Effective rank via SVD on standardized features
X = (per_system - per_system.mean(0)) / per_system.std(0)
s = np.linalg.svd(X, compute_uv=False)
var = s ** 2 / (s ** 2).sum()
print(f"PCA variance ratios on standardized per-system means: "
      f"{[f'{v:.3f}' for v in var]}")
print(f"Cumulative: {[f'{v:.3f}' for v in np.cumsum(var)]}")
participation_ratio = (s ** 2).sum() ** 2 / (s ** 4).sum()
print(f"Participation ratio (effective dimensionality): {participation_ratio:.2f} / 4")

# ---------------------------------------------------------------- #
# 5. Cross-cut: do bSASA-broken systems concentrate by pK?          #
# ---------------------------------------------------------------- #
print()
print("=" * 72)
print("5. CROSS-CUT: bSASA-clip-saturated systems vs pK")
print("=" * 72)
for split in ["train", "val", "test"]:
    feats = load_features(split)
    bsasa = feats[..., 3]
    at_zero = (bsasa == 0).mean(axis=1) > 0.5
    at_max = (bsasa == 2500).mean(axis=1) > 0.5
    affected = at_zero | at_max
    p = pks[split]
    if affected.sum() == 0:
        print(f"{split}: no clip-saturated systems")
        continue
    print(f"{split}: {affected.sum()} affected ({100 * affected.mean():.1f}%)  "
          f"mean pK affected={p[affected].mean():.2f}  "
          f"mean pK unaffected={p[~affected].mean():.2f}  "
          f"Δ={p[affected].mean() - p[~affected].mean():+.2f}")

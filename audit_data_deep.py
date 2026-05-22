"""Deep data audit — focuses on whether SIGNAL TRANSFERS across splits.

Aggregate statistics already told us the train/test split is non-IID.
This script asks the harder questions:

  A. Does the assay-mix shift fully explain the pK-mean shift?
  B. Does each channel's predictive relationship with pK transfer
     train → val / test? (If a channel correlates with pK on train
     but not val/test, the model learns a spurious feature.)
  C. Is there protein-family overlap between splits, or is the test
     set genuinely out-of-distribution by protein identity?
  D. What fraction of each split is flagged as dissociated / unstable
     / multi-ligand, and does the rate differ by split?
  E. For PDBs with all three assay types in the CSV, do they agree?
     (Quantifies the assay-mixing noise floor.)
  F. Channel statistics per pK bin and per split — does the conditional
     channel|pK distribution shift, or only the marginal pK distribution?
"""

import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, pearsonr, spearmanr

ROOT = Path(__file__).parent
PREP = ROOT / "preprocessed"
AFF_DIR = ROOT / "misato-affinity" / "data"
AFF_CSV = AFF_DIR / "affinity_data.csv"
UNIPROT_CLUST = AFF_DIR / "uniprot_clustering.pickle"

CHANNELS = ["rmsd_ligand", "interaction_energy", "distance", "bSASA"]


def load_samples(split):
    out = []
    with open(PREP / f"samples_{split}.jsonl") as f:
        for line in f:
            out.append(json.loads(line))
    return out


def load_features(split):
    z = np.load(PREP / f"features_{split}.npz")
    return z[z.files[0]]


def assay_kind(row):
    if row["Kd (nM)"] and row["Kd (nM)"] > 0:
        return "Kd"
    if row["Ki (nM)"] and row["Ki (nM)"] > 0:
        return "Ki"
    if row["IC50 (nM)"] and row["IC50 (nM)"] > 0:
        return "IC50"
    return "none"


print("Loading data...")
df_aff = pd.read_csv(AFF_CSV, sep=";", engine="python", on_bad_lines="skip")
df_aff["PDBid"] = df_aff["PDBid"].str.upper()
df_aff["assay"] = df_aff.apply(assay_kind, axis=1)
aff_lookup = df_aff.set_index("PDBid").to_dict("index")

samples = {s: load_samples(s) for s in ["train", "val", "test"]}
features = {s: load_features(s) for s in ["train", "val", "test"]}

# Decorate each sample with its assay type
for split, rows in samples.items():
    for r in rows:
        pdb = r["pdb_id"].upper()
        r["assay"] = aff_lookup.get(pdb, {}).get("assay", "none")
        r["uniprot"] = aff_lookup.get(pdb, {}).get("Uniprot", None)


# =================================================================
# A. Does the assay-mix shift fully explain the pK-mean shift?
# =================================================================
print()
print("=" * 76)
print("A. PER-ASSAY pK DISTRIBUTION (does assay-mix explain the pK shift?)")
print("=" * 76)
print(f"{'assay':<6} {'split':<6} {'n':>6}  {'mean':>7}  {'std':>6}  {'med':>6}")
for assay in ["Kd", "Ki", "IC50"]:
    for split in ["train", "val", "test"]:
        pks = np.array([r["pK"] for r in samples[split] if r["assay"] == assay])
        if len(pks) == 0:
            continue
        print(f"{assay:<6} {split:<6} {len(pks):>6,d}  "
              f"{pks.mean():>+7.3f}  {pks.std():>6.3f}  {np.median(pks):>6.3f}")
    print()

# Reweight train pK by val's assay-mix to "remove" the assay-mix effect.
# If reweighted train pK ≈ val pK, the shift is purely assay-mix.
val_mix = {a: 0 for a in ["Kd", "Ki", "IC50"]}
for r in samples["val"]:
    if r["assay"] in val_mix:
        val_mix[r["assay"]] += 1
val_total = sum(val_mix.values())
val_mix_frac = {a: n / val_total for a, n in val_mix.items()}

train_per_assay = defaultdict(list)
for r in samples["train"]:
    if r["assay"] in val_mix:
        train_per_assay[r["assay"]].append(r["pK"])

reweighted_mean = sum(val_mix_frac[a] * np.mean(train_per_assay[a])
                      for a in val_mix)
reweighted_std_proxy = sum(val_mix_frac[a] * np.std(train_per_assay[a])
                           for a in val_mix)

train_pk = np.array([r["pK"] for r in samples["train"]])
val_pk = np.array([r["pK"] for r in samples["val"]])
print(f"raw train pK mean:                    {train_pk.mean():.3f}")
print(f"train pK reweighted to val's assay-mix: {reweighted_mean:.3f}")
print(f"raw val pK mean:                      {val_pk.mean():.3f}")
print(f"  → shift explained by assay-mix: "
      f"{train_pk.mean() - reweighted_mean:.3f} of "
      f"{train_pk.mean() - val_pk.mean():.3f} pK total "
      f"({100 * (train_pk.mean() - reweighted_mean) / (train_pk.mean() - val_pk.mean()):.0f}%)")


# =================================================================
# B. Does each channel's predictive relationship with pK transfer?
# =================================================================
print()
print("=" * 76)
print("B. CHANNEL→pK CORRELATION PER SPLIT (does signal transfer?)")
print("=" * 76)
print("Per-system channel means correlated with pK label.")
print(f"{'channel':<22} {'train Pearson':>14} {'val Pearson':>13} {'test Pearson':>14}")

for ci, name in enumerate(CHANNELS):
    rs = {}
    for split in ["train", "val", "test"]:
        per_sys = features[split][..., ci].mean(axis=1)
        pks = np.array([r["pK"] for r in samples[split]])
        rs[split] = pearsonr(per_sys, pks)[0]
    print(f"{name:<22} {rs['train']:>+14.4f} {rs['val']:>+13.4f} {rs['test']:>+14.4f}")

print()
print("Per-system channel STDs correlated with pK label (does ligand mobility predict affinity?):")
print(f"{'channel':<22} {'train Pearson':>14} {'val Pearson':>13} {'test Pearson':>14}")
for ci, name in enumerate(CHANNELS):
    rs = {}
    for split in ["train", "val", "test"]:
        per_sys = features[split][..., ci].std(axis=1)
        pks = np.array([r["pK"] for r in samples[split]])
        rs[split] = pearsonr(per_sys, pks)[0]
    print(f"{name:<22} {rs['train']:>+14.4f} {rs['val']:>+13.4f} {rs['test']:>+14.4f}")


# =================================================================
# C. Protein overlap (Uniprot) between splits
# =================================================================
print()
print("=" * 76)
print("C. UNIPROT OVERLAP BETWEEN SPLITS")
print("=" * 76)
uniprots = {s: set(r["uniprot"] for r in samples[s] if r["uniprot"])
            for s in ["train", "val", "test"]}
for s in ["train", "val", "test"]:
    print(f"{s:<5}: {len(uniprots[s]):,d} unique Uniprot IDs "
          f"(from {len(samples[s]):,d} PDBs)")

train_val = uniprots["train"] & uniprots["val"]
train_test = uniprots["train"] & uniprots["test"]
val_test = uniprots["val"] & uniprots["test"]
print()
print(f"train ∩ val   Uniprots: {len(train_val):,d}  "
      f"({100 * len(train_val) / max(len(uniprots['val']), 1):.1f}% of val proteins seen in train)")
print(f"train ∩ test  Uniprots: {len(train_test):,d}  "
      f"({100 * len(train_test) / max(len(uniprots['test']), 1):.1f}% of test proteins seen in train)")
print(f"val ∩ test    Uniprots: {len(val_test):,d}")

# What fraction of test rows are on a protein seen in train?
test_rows_on_train_protein = sum(1 for r in samples["test"]
                                  if r["uniprot"] and r["uniprot"] in uniprots["train"])
val_rows_on_train_protein = sum(1 for r in samples["val"]
                                 if r["uniprot"] and r["uniprot"] in uniprots["train"])
print(f"\n{val_rows_on_train_protein:,d} / {len(samples['val']):,d} val rows "
      f"({100 * val_rows_on_train_protein / len(samples['val']):.1f}%) are on a Uniprot seen in train")
print(f"{test_rows_on_train_protein:,d} / {len(samples['test']):,d} test rows "
      f"({100 * test_rows_on_train_protein / len(samples['test']):.1f}%) are on a Uniprot seen in train")


# =================================================================
# D. Trajectory quality flags per split
# =================================================================
print()
print("=" * 76)
print("D. TRAJECTORY QUALITY FLAGS PER SPLIT")
print("=" * 76)
for split in ["train", "val", "test"]:
    rows = samples[split]
    n = len(rows)
    dis = sum(1 for r in rows if r.get("dissociated"))
    uns = sum(1 for r in rows if r.get("unstable"))
    mlt = sum(1 for r in rows if r.get("multi_ligand"))
    contacts = sum(1 for r in rows
                   if r.get("facts", {}).get("summary", {}).get("contacts_persistent") is False)
    drift = sum(1 for r in rows
                if r.get("facts", {}).get("summary", {}).get("ligand_drift"))
    print(f"{split:<5}: dissociated {dis:>4,d} ({100*dis/n:.1f}%)  "
          f"unstable {uns:>4,d} ({100*uns/n:.1f}%)  "
          f"multi_ligand {mlt:>4,d} ({100*mlt/n:.1f}%)  "
          f"non-persistent contacts {contacts:>4,d} ({100*contacts/n:.1f}%)  "
          f"ligand_drift {drift:>4,d} ({100*drift/n:.1f}%)")


# =================================================================
# E. Assay-cross-check: for PDBs with multiple assays, do they agree?
# =================================================================
print()
print("=" * 76)
print("E. ASSAY CROSS-CHECK (PDBs with multiple non-zero assay values)")
print("=" * 76)
deltas = {"Kd-Ki": [], "Kd-IC50": [], "Ki-IC50": []}
def pk_from_nm(nm):
    return 9 - np.log10(nm)
for _, row in df_aff.iterrows():
    kd, ki, ic50 = row["Kd (nM)"], row["Ki (nM)"], row["IC50 (nM)"]
    if kd > 0 and ki > 0:
        deltas["Kd-Ki"].append(pk_from_nm(kd) - pk_from_nm(ki))
    if kd > 0 and ic50 > 0:
        deltas["Kd-IC50"].append(pk_from_nm(kd) - pk_from_nm(ic50))
    if ki > 0 and ic50 > 0:
        deltas["Ki-IC50"].append(pk_from_nm(ki) - pk_from_nm(ic50))
for key, vals in deltas.items():
    if not vals:
        continue
    arr = np.array(vals)
    print(f"{key:<10} n={len(arr):>4,d}  "
          f"mean Δ={arr.mean():+.3f}  "
          f"std={arr.std():.3f}  "
          f"|Δ|>0.5 pK: {(np.abs(arr) > 0.5).mean()*100:.1f}%  "
          f"|Δ|>1.0 pK: {(np.abs(arr) > 1.0).mean()*100:.1f}%")


# =================================================================
# F. Conditional channel distribution per pK bin and split (KS test)
# =================================================================
print()
print("=" * 76)
print("F. CONDITIONAL CHANNEL DISTRIBUTION SHIFT (KS test, train vs test)")
print("=" * 76)
print("If the model sees the same channel|pK relationship in train and test,")
print("KS p-values should be high (no significant shift). Per channel and pK bin:")
bins = [(3, 5), (5, 7), (7, 9)]
print(f"\n{'channel':<22}  " + "  ".join(f"pK {lo}-{hi:<4}" for lo, hi in bins))
for ci, name in enumerate(CHANNELS):
    line = []
    for lo, hi in bins:
        train_sel = (np.array([r["pK"] for r in samples["train"]]) >= lo) & \
                    (np.array([r["pK"] for r in samples["train"]]) < hi)
        test_sel = (np.array([r["pK"] for r in samples["test"]]) >= lo) & \
                   (np.array([r["pK"] for r in samples["test"]]) < hi)
        if train_sel.sum() < 30 or test_sel.sum() < 30:
            line.append(f"   (n<30)")
            continue
        # Per-system channel mean
        a = features["train"][train_sel, :, ci].mean(axis=1)
        b = features["test"][test_sel, :, ci].mean(axis=1)
        ks_stat, p = ks_2samp(a, b)
        sig = "***" if p < 0.001 else "** " if p < 0.01 else "*  " if p < 0.05 else "ns "
        line.append(f"p={p:.3f}{sig}")
    print(f"{name:<22}  " + "  ".join(line))


# =================================================================
# G. SUMMARY: What this means for modeling
# =================================================================
print()
print("=" * 76)
print("G. KEY TAKEAWAYS")
print("=" * 76)

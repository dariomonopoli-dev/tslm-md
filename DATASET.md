# MISATO MD Dataset — Analysis & Preprocessing Recipe

Companion to `PROJECT_BRIEF.md`. Covers the 132.84 GB `MD.hdf5` we'll train
OpenTSLM-SP against, what's in it, what's wrong with it, and what the loader
must do before yielding samples.

All numbers in this document are derived empirically from a full pass over the
file on 2026-05-21. Scripts: `/tmp/md_integrity.py`, `/tmp/md_analyze.py`,
`/tmp/md_outliers.py`.

---

## 1. File at a glance

| | |
|---|---|
| Path | `/home/mxlk/Documents/AIproject /MD.hdf5` (folder name has trailing space) |
| Size | 132.84 GB |
| Format | HDF5, one top-level group per PDB system |
| Systems | **16,972** (matches union of MISATO train/val/test split files exactly) |
| Frames per system | **100** (uniform across the whole dataset) |
| Atoms per system | **556 – 40,798** (median 4,951) |

Per-system schema (all 16,972 systems have every key present):

```
<PDB>/
  trajectory_coordinates       (100, N_atoms, 3)  float64   # not used by OpenTSLM-SP input
  atoms_element / atoms_residue / atoms_type / atoms_number   (N_atoms,)
  molecules_begin_atom_index   (3,)
  frames_rmsd_ligand           (100,)   float64   # used as TS channel 1
  frames_interaction_energy    (100,)   float64   # used as TS channel 2
  frames_distance              (100,)   float64   # used as TS channel 3
  frames_bSASA                 (100,)   float64   # used as TS channel 4
```

OpenTSLM-SP input = the four `frames_*` channels (each shape `(100,)`, a clean
multiple of the default `PATCH_SIZE = 4` → 25 patches per channel, no padding
overhead). Trajectory coordinates stay on disk and are only opened by the
post-hoc verifier when it needs to ground a rationale.

---

## 2. Split coverage and label coverage

Cross-referenced against `misato-dataset-master/data/MD/splits/{train,val,test}_MD.txt`
and `misato-affinity/data/affinity_data.csv`:

| Split | Expected | Present in HDF5 | Labelled (after join) |
|---|---:|---:|---:|
| train | 13,765 | 13,765 (100%) | **13,758** (7 unlabelled, drop) |
| val | 1,595 | 1,595 (100%) | 1,595 |
| test | 1,612 | 1,612 (100%) | 1,612 |

**Affinity coverage: 16,965 / 16,972 = 99.96%.**

Conversion: `pK = 9 − log10(value_in_nM)`, priority `Kd > Ki > IC50` (first non-zero).

The 7 train PDBs with no entry in `affinity_data.csv` (must be filtered in
`_load_splits`):

```
4DGO, 4OTW, 4V1C, 5V8H, 5V8J, 6FIM, 6H7K
```

---

## 3. Anomalies and how the loader handles each

### 3a. `6CC9` — multi-ligand complex with mismatched channel lengths

`6CC9` has 4 molecules in `molecules_begin_atom_index` (vs. the usual 3) and
two of its frame channels are length 400 instead of 100:

```
frames_rmsd_ligand   shape=(400,)   # concatenated per ligand × 100 frames
frames_distance      shape=(400,)
frames_interaction_energy  shape=(100,)
frames_bSASA               shape=(100,)
```

OpenTSLM's padding util takes `max_len` across the four channels in a sample
and would pad the length-100 ones to 400 with **zeros**, silently corrupting
the input. Loader action: slice both 400-length channels to `[:100]` (primary
ligand) and proceed. Tag the sample with `multi_ligand=True` for the rationale
prompt.

### 3b. `bSASA` numerical bugs

The buried-surface-area channel has physically impossible values:

- 601 systems contain at least one frame with `bSASA < 0` (min observed:
  −63,673 Å²)
- 663 systems contain at least one frame with `bSASA > 2,500 Å²` (max:
  +81,907 Å²)

Buried surface area is bounded below by 0 and above by the ligand's total
solvent-accessible surface (typically < 2,500 Å²). Both extremes are MISATO
pipeline artefacts, not signal.

Loader action: **clip `bSASA` to `[0, 2500]`** before normalization. The good
signal (median 503 Å², p99 1,250 Å²) is preserved.

### 3c. Other channel outliers (kept as-is, normalized after clipping)

| Channel | Physical-sense clip | Affected systems | % | Affected frames | % |
|---|---|---:|---:|---:|---:|
| `rmsd_ligand` | `[0, 50]` Å | 135 | 0.80 | 5,492 | 0.32 |
| `interaction_energy` | `[−500, 50]` kcal/mol | 220 | 1.30 | 7,440 | 0.44 |
| `distance` | `[0, 50]` Å | 220 | 1.30 | 10,848 | 0.64 |
| `bSASA` | `[0, 2500]` Å² | 1,250 | 7.37 | 6,333 | 0.37 |

For `rmsd`, `distance`, and `IE`, the out-of-range values are *physically
real* (ligand dissociation, very strong binding) — clipping is a numerical
convenience, not a correction. They get a tag (see §4) so the rationale can
reference what actually happened.

---

## 4. Ligand-dissociation tag (kept, not dropped)

A subset of MISATO trajectories represent failed simulations where the ligand
diffuses out of the binding pocket during the 10 ns. Example signature
(`4WHY, 3ITH, 2FGH, 2GGU, 6G47, 3LM1, 3L08, 5C7E, 1OX9, 6M87`):
`rmsd_ligand > 50 Å` on every frame, `distance > 50 Å` on every frame.

**Decision: keep these systems** rather than drop them — the model should
learn to recognise a dissociated trajectory as a weak-binder signal, and the
rationale generator should be allowed to cite the dissociation.

Loader emits two booleans per sample (computed *before* clipping):

```python
dissociated = (rmsd_last20.mean() > 5.0) or (distance_last20.mean() > 30.0)
unstable    = (rmsd.max() > 10.0) and not dissociated
```

Rationale: the last 20 frames represent the post-relaxation state. >5 Å
ligand RMSD means the ligand left its starting pose; >30 Å distance means
it left the pocket. The `unstable` tag flags trajectories with a transient
excursion but a stable endpoint.

These flags get woven into `_get_pre_prompt`:

```
"...This trajectory shows ligand dissociation (mean last-20-frame RMSD
{rmsd_last20:.1f} Å). Account for this in your reasoning..."
```

Suggested thresholds are starting points — tune on val if the rationale
quality changes meaningfully.

---

## 5. Channel value distributions (all 16,971 usable systems × 100 frames)

After clipping to the bounds in §3c:

| Channel | min | p1 | p50 | mean | std | p99 | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| `rmsd_ligand` (Å) | 0.13 | 0.55 | 2.14 | 3.20 | 5.63 | 19.97 | 185.7 (clipped 50) |
| `interaction_energy` (kcal/mol) | −1319 | −295 | −28.1 | −36.0 | 50.3 | 10.2 | 183 (clipped 50) |
| `distance` (Å) | 0.03 | 2.96 | 13.5 | 15.7 | 9.0 | 45.4 | 140 (clipped 50) |
| `bSASA` (Å²) | −63,674 | 51.9 | 503 | 529 | 642 | 1,250 | 81,907 (clipped 2500) |

Normalize each channel per-channel with **train-set mean/std after clipping**.
Compute the stats once, cache to a JSON file, and apply in the loader's
`_get_text_time_series_prompt_list`. Do not z-score per-sample — channel scale
is itself information (a large interaction-energy magnitude means something).

---

## 6. Within-trajectory dynamics — the "movie" framing holds

For each system, the range and std over its 100 frames:

| Channel | systems with std=0 | median range | p10 range | p90 range | median std |
|---|---:|---:|---:|---:|---:|
| `rmsd_ligand` | 0 | 1.91 Å | 1.02 | 4.85 | 0.39 |
| `interaction_energy` | 0 | 18.6 kcal/mol | 12.4 | 35.0 | 3.69 |
| `distance` | 0 | 1.91 Å | 1.18 | 4.19 | 0.38 |
| `bSASA` | 3 | 223 Å² | 126 | 450 | 41.4 |

**Zero systems are flat on all four channels.** The frames carry real
within-trajectory information, not just a per-system constant. This validates
the "read the movie" premise at full-dataset scale.

---

## 7. Label distribution — train/val/test shift is significant

| Split | n | min | p10 | p50 | mean | p90 | max | std |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 13,758 | 0.45 | 4.05 | 6.70 | **6.59** | 8.80 | 15.22 | 1.85 |
| val | 1,594 | 0.40 | 3.22 | 5.39 | **5.44** | 7.68 | 12.00 | 1.72 |
| test | 1,612 | 0.66 | 3.57 | 5.45 | **5.55** | 7.60 | 11.30 | 1.63 |

Train mean pK is ~1.1 units higher than val/test. This is intrinsic to the
MISATO split, not a sampling artefact. Implications:

- Constant-prediction baseline gets very different RMSE on different splits
  (1.85 on train vs 1.93 on test for "predict train mean").
- Any train-vs-test metric comparison must account for this — report both.
- Calibrate against val (which matches test's distribution better than train
  does), not against train.

---

## 8. Per-channel signal and the must-beat baseline

Univariate Pearson r against pK (train only, per-system summary → pK):

| Feature | r |
|---|---:|
| `bSASA_mean` | **+0.360** |
| `bSASA_mean_last20` | +0.322 |
| `distance_last` | −0.156 |
| `distance_mean` | −0.148 |
| `rmsd_std` | −0.144 |
| `rmsd_mean` | −0.134 |
| `IE_mean` | −0.047 |

Buried surface area carries the dominant linear signal (biophysically
expected — bigger interface ≈ tighter binding). The other three channels
contribute weakly. Interaction-energy mean is essentially uncorrelated with
affinity at this granularity, which is itself worth noting: a model that
learns to use IE will be doing something non-trivial.

Inter-channel correlation of per-system means is low (max |r| = 0.44 between
`rmsd` and `distance`) — all 4 channels are worth keeping.

**Baseline OpenTSLM-SP must beat** — OLS on 4 trajectory-mean features,
train-fit, test-evaluated:

| Metric | Value |
|---|---:|
| test RMSE | **1.791** |
| test MAE | 1.468 |
| test Pearson r | 0.260 |
| z-scaled β | bSASA +0.66, dist −0.20, IE +0.08, rmsd −0.03 |

Anything that doesn't beat **RMSE 1.79** on test is not learning trajectory
structure — it's matching what a 5-parameter linear model already extracts
from per-channel means. This is the headline number to compare against in the
demo.

For context, the trivial "predict train mean" gives test RMSE 1.93.

---

## 9. Concrete loader recipe (drop-in for `MisatoCoTQADataset`)

1. Open `MD.hdf5` once per DataLoader worker (h5py is not fork-safe; cache
   the handle on first `__getitem__` per worker).
2. Build the trainable-PDB list:
   - Read split file → set of PDB IDs
   - Drop the 7 unlabelled train PDBs (§2)
   - Keep `6CC9` but mark `multi_ligand=True`
3. For each PDB:
   - Load the 4 `frames_*` channels into memory eagerly (total ~50 MB across
     all 16,972 systems — fine to pre-load).
   - If `6CC9`, slice `frames_rmsd_ligand[:100]` and `frames_distance[:100]`.
   - Compute `dissociated` / `unstable` tags from the **unclipped** rmsd and
     distance arrays (§4).
   - Clip each channel to its physical bounds (§3c).
   - Z-score with cached train mean/std (§5).
4. Look up pK from the affinity-CSV table; format answer as `"Answer: X.XX"`.
5. Return four `TextTimeSeriesPrompt` objects (one per channel) plus a
   `pre_prompt` that mentions the dissociation/multi-ligand tags when set.

---

## 10. Open questions to revisit if results disappoint

- Whether to drop the ~135 fully-dissociated systems entirely from training
  (instead of tagging) — could help if the LLM keeps echoing "ligand
  dissociated" without using the channel values.
- Whether to add a 5th channel: `interaction_energy` smoothed or first
  derivative. Raw IE is noisy and uncorrelated; a smoothed/derivative version
  might surface the binding-event signal.
- Whether to compute a per-trajectory verifier feature set (mean RMSD, slope
  of bSASA, etc.) and condition the rationale prompt on it explicitly, rather
  than letting the LLM derive it from the soft prompts alone.

These are v2 levers — do not pursue in the 20-hour v1.

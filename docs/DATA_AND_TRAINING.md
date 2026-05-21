# TSLM-MD: data and training, explained end-to-end

A guide for a teammate who hasn't touched this code yet. Read top-to-bottom; each section builds on the previous one.

## 1. What are we actually predicting?

**One number per protein-ligand complex: the binding free energy, ΔG, in kcal/mol.**

ΔG is a thermodynamic property of the *whole complex at equilibrium*. There is one Kd per complex, not one per MD frame. So:

- **The 30 MD frames are the model INPUT** — they describe how the binding pose evolves over time.
- **The model OUTPUT is a single scalar (plus a confidence tier)**, not a per-frame curve.

Concretely the model emits a string like:

```
Answer: -8.40 kcal/mol. Confidence: high.
```

A regex (`tslm_md/parse.py`) pulls the float and the tier back out. Pearson r / Spearman ρ / MAE are computed against PDBbind ground-truth ΔG.

Why not regress per-frame? Because we have no per-frame ground truth — PDBbind only gives one Kd per complex. Per-frame outputs would either be a degenerate "predict the same number 30 times" loss, or unsupervised, and wouldn't improve the affinity prediction the demo is built around.

## 2. Where the raw data comes from

**MISATO** ([Siebenmorgen et al. 2024](https://github.com/sab148/MiSaTo-dataset)) — a public dataset of molecular dynamics (MD) trajectories for protein-ligand complexes drawn from PDBbind. We use the `MD.hdf5` subset.

```
data/misato/MD.hdf5            124 GiB, 16,972 PDB ids
data/misato/MD_dummy.hdf5      2.5 MiB, tiny test fixture
```

Inspect with `python scripts/inspect_h5.py data/misato/MD.hdf5`. Each top-level key is a PDB id (`10GS`, `11GS`, ...). Inside one entry:

| Dataset | Shape | What it is |
|---|---|---|
| `trajectory_coordinates` | `(100, N_atoms, 3)` | 100 MD frames, xyz for every atom in the complex |
| `molecules_begin_atom_index` | `(3,)` | atom-range boundaries; the last entry marks where the ligand starts |
| `atoms_element`, `atoms_number`, `atoms_residue`, `atoms_type` | `(N_atoms,)` | atom metadata |
| `frames_interaction_energy` | `(100,)` | **precomputed** per-frame protein-ligand interaction energy (kcal/mol) |
| `frames_distance` | `(100,)` | precomputed per-frame protein-ligand min distance (Å) |
| `frames_rmsd_ligand` | `(100,)` | precomputed per-frame ligand RMSD (Å) |
| `frames_bSASA` | `(100,)` | precomputed per-frame buried solvent-accessible surface area (Å²) |

The four `frames_*` arrays are scalars MISATO computed for us. We use them directly — see channels 6-9 below.

## 3. Pipeline overview

```
   data/misato/MD.hdf5  (124 GiB raw MD trajectories)
                │
                ▼
   scripts/preprocess_features.py
   (calls tslm_md.featurize.featurize per complex)
                │
                ▼
   data/featurized.h5    one [10, 30] float32 tensor per PDB id
   data/feature_stats.json  per-channel mean/std over the train split
                │
                ▼
   tslm_md.dataset.MDCoTQADataset
   (zips each tensor with text descriptors and the ground-truth answer string)
                │
                ▼
   tslm_md.train_stage6 -> OpenTSLMFlamingo
   (Llama-3.2-1B + Chronos-2 time-series encoder, gated cross-attention adapter)
                │
                ▼
   "Answer: -8.40 kcal/mol. Confidence: high."
```

## 4. The 10 input channels

For each complex we compress its 100-frame MD trajectory into a `(10, 30)` matrix. The 30 frames are picked by `np.linspace(0, 99, 30)` — even sampling across the trajectory. The 10 channels:

**Six geometric channels** (computed from `trajectory_coordinates` in `tslm_md/featurize.py`):

| # | Channel | Units | Idea |
|---|---|---|---|
| 0 | min pocket-ligand distance | Å | Did the ligand stay in contact? |
| 1 | mean pocket-ligand distance under a 4 Å mask | Å | Average close-contact tightness |
| 2 | number of close contacts within 4 Å | count | Size of the contact network |
| 3 | ligand RMSD from frame 0 after Kabsch alignment on the pocket | Å | Did the ligand wander? |
| 4 | ligand radius of gyration | Å | Did the ligand compress / stretch / flex? |
| 5 | buriedness proxy: ligand atoms with ≤ 2 protein neighbours within 5 Å | count | Surface-exposed vs. deeply buried |

**Four MISATO-precomputed channels** (just subsampled at the same 30 indices):

| # | Channel | Units | Idea |
|---|---|---|---|
| 6 | `frames_interaction_energy` | kcal/mol | Direct physics-based binding-energy proxy per frame |
| 7 | `frames_distance` | Å | MISATO's reference protein-ligand min distance |
| 8 | `frames_rmsd_ligand` | Å | MISATO's reference ligand RMSD |
| 9 | `frames_bSASA` | Å² | Buried SASA — how shielded the binding interface is |

Channels 0/3/5 vs 7/8/9 overlap conceptually but use different definitions — we keep both, the model decides which is informative. Channel 6 (interaction energy) is the highest-signal addition: it's the closest single-number physical proxy for ΔG and was missing from the previous 6-channel featurizer.

The "pocket" we use for the geometric channels is the set of protein atoms within 6 Å of any ligand atom in frame 0.

Channel descriptors in `tslm_md/prompts.py:CHANNEL_DESCRIPTIONS` are the **natural-language labels** shown to the LM alongside each channel — order must match `featurize.py`.

## 5. Ground-truth labels

`data/targets.json` (~19,443 PDB ids). Built by `scripts/build_training_targets.py` from `data/misato_affinity/affinity_data.csv` (the kierandidi binding-affinity CSV):

```json
"10gs": {
  "answer": "Answer: -8.40 kcal/mol. Confidence: high.",
  "affinity_kcal_mol": -8.40,
  "pK": 6.16,
  "kind": "Kd",
  "confidence": "high",
  "ligand": "...",
  "protein": "..."
}
```

- `pK = -log10(K)` from the assay measurement (in nM, converted to M)
- `affinity_kcal_mol = -1.3642 × pK` — converts pK to ΔG via dG = -RT ln K at 298 K
- `confidence` tier comes from which measurement type was available:
  - **high** = Kd present (most direct binding measurement)
  - **medium** = Ki present (inhibition constant, well-correlated)
  - **low** = IC50 only (assay-dependent, noisier)
- `answer` is the literal string the LM is trained to emit

## 6. Splits

`data/splits/{train,val,test}.txt` — 80/10/10 random split keyed by PDB id (deterministic seed). About 13,765 / 1,595 / 1,612.

A row is only usable if it appears in **all three** of: a split file, `featurized.h5`, and `targets.json`. The dataset class does a case-insensitive join (MISATO uses uppercase ids, targets use lowercase).

## 7. What one training sample looks like

`tslm_md.dataset.MDCoTQADataset` yields a 5-key dict (the format OpenTSLM expects):

```python
{
  "time_series":      Tensor[10, 30],   # the featurized trajectory, z-scored
  "time_series_text": [str * 10],       # one descriptor per channel
  "pre_prompt":       "You are a computational chemist analysing...",
  "post_prompt":      "Output exactly one line: Answer: <x> kcal/mol. Confidence: ...",
  "answer":           "Answer: -8.40 kcal/mol. Confidence: high.",
}
```

Inside the model (`OpenTSLMFlamingo`, from the [liu-jc Chronos-2 fork of OpenTSLM](https://github.com/liu-jc/OpenTSLM/tree/add-chronos2-encoder)):

1. Each of the 10 channels is encoded **independently** by Chronos-2 into a sequence of time-series tokens.
2. The 10 token sequences are interleaved into the text prompt as 10 separate "media chunks", each preceded by its natural-language descriptor.
3. The whole thing is fed to a **frozen** Llama-3.2-1B; trainable gated cross-attention layers (every N layers) let the LM attend to the time-series tokens.
4. Loss is **standard causal-LM cross-entropy** over the answer string (teacher-forced). We're not training a regressor — we're training the LM to generate the correct sentence.

## 8. Normalisation: `feature_stats.json`

After preprocessing, `data/feature_stats.json` contains per-channel mean and std computed over the **train split only**. At dataloader time each `(10, 30)` tensor is z-scored: `(x - mean) / std`. This keeps the Chronos encoder seeing roughly unit-variance inputs across all 10 (very different-scale) channels.

## 9. End-to-end runbook

Assumes you've already done the GPU/env setup from the root `README.md` (`bash scripts/setup_gpu.sh`).

```bash
# 1. Build/refresh per-complex labels and splits (once)
python scripts/build_training_targets.py \
  --kierandidi-csv data/misato_affinity/affinity_data.csv \
  --build-splits

# 2. Featurise the full MISATO dataset -> data/featurized.h5 + data/feature_stats.json
#    Add --limit 2000 for a fast smoke test.
python scripts/preprocess_features.py \
  --misato-h5 data/misato/MD.hdf5 \
  --out-h5 data/featurized.h5 \
  --splits-dir data/splits

# 3. Train (uses configs/stage6_md_cot.yaml)
bash scripts/launch_train.sh nohup
tail -f train.log

# 4. Evaluate on the held-out test split
python scripts/eval_benchmark.py
```

Things to expect:

- Step 2 reads ~124 GiB sequentially. Plan for ~1.5–2 h wall-clock on a single disk; tqdm shows live progress.
- Step 3 wants a GPU with ≥ 24 GiB VRAM (A30 / A100 / H100). Batch size 1 with grad-accum 16; bf16; gradient checkpointing on.
- WandB logs `train/loss`, `val/loss`, and every `vis_every_steps` (default 500) it generates real predictions on 20 val samples and pushes a predicted-vs-truth scatter to wandb (`tslm_md/train_stage6.py:_vis_predictions_to_wandb`).

## 10. Quick sanity-checks while reading the code

| You're looking at... | Read this | Key things to confirm |
|---|---|---|
| One MISATO trajectory | `scripts/inspect_h5.py data/misato/MD.hdf5 --key 10GS` | 100 frames, ~6 k atoms, four `frames_*` arrays |
| The featurizer | `tslm_md/featurize.py` | Output is `[10, 30]`, channels 6-9 come straight from MISATO scalars |
| One featurized tensor | `scripts/inspect_h5.py data/featurized.h5 --key 10GS` | Shape `(10, 30)`, dtype float32 |
| One training sample | `MDCoTQADataset.__getitem__` via `train_stage6.py` | 5-key dict described in §7 |
| Training loop | `tslm_md/train_stage6.py:236` (`main`) | Causal-LM CE loss, not regression |
| Eval | `scripts/eval_benchmark.py` + `tslm_md/parse.py` | Parses the answer string, computes Pearson/Spearman/MAE |

## 11. Common pitfalls

- **Case mismatch**: MISATO uses `10GS`; targets use `10gs`. `MDCoTQADataset._load_splits` handles this — don't break the case-insensitive join.
- **Stale stats**: if you change `featurize.py`, you must rerun step 2. The `feature_stats.json` lengths/values **must** match the channel count.
- **Pocket = empty**: `featurize.py` returns zeros for geometric channels (0-5) but still fills 6-9 from MISATO precomputed arrays when available.
- **Old checkpoints**: a checkpoint trained on 6 channels is *not* drop-in compatible with the 10-channel featurizer — Chronos sees 10 media chunks now instead of 6. You need to retrain (or stub channels 6-9 to zero, but that loses the new signal).

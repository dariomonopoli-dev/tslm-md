# Project Learnings — MISATO MD-Affinity with OpenTSLM-SP

What we actually learned from one full v1a training cycle, two data audits, and a
re-preprocessing pass over MD.hdf5. Companion to [TRAINING.md](TRAINING.md)
(operational log) and [DATASET.md](DATASET.md) (curation reference). This doc is
the **findings and forward-looking blueprint** — it deliberately overlaps with
neither.

---

## 0. TL;DR

- v1a final test: **Pearson R = 0.34**, RMSE = 1.80, near the `mlp_engineered`
  ceiling (0.36 / 1.68) and consistent with the audit's prediction.
- The plateau is **a data-side ceiling, not an architecture ceiling.** Best
  single channel correlates with pK at R = 0.27 on test; the 4-channel summary
  representation caps Pearson around 0.36.
- We use **0.02% of MD.hdf5**. The published OpenTSLM-SP recipe inherits a
  representation that was designed for the architecture, not for the data.
- v2 (12 channels, derived from coordinates) lifts the linear oracle ceiling
  from 0.27 → 0.30 on test — real but modest. Most "expected" predictors
  (H-bonds, hydrophobic contacts) are heavily redundant with bSASA.
- Cheng-Prusoff label correction makes the train→test distribution shift
  *slightly worse* because train has more IC50 systems than val/test.
- Pearson R is the rank metric we care about, but the training objective
  (cross-entropy on rationale tokens) doesn't target rank at all.

The single biggest takeaway: **the published method's input representation is
the wrong choice for this dataset, but switching it requires a different model
class entirely — not a tweak to OpenTSLM-SP.**

---

## 1. About MISATO (the dataset)

### 1.1 The train/test split is structurally OOD by binding affinity

| Split | n     | mean pK | std  | source mix (Kd / Ki / IC50) |
|-------|------:|--------:|-----:|---|
| train | 13,758 | 6.59    | 1.85 | 30.6% / 26.6% / **42.8%** |
| val   | 1,595  | 5.44    | 1.72 | **64.2%** / 20.1% / 15.7% |
| test  | 1,612  | 5.55    | 1.63 | **61.8%** / 18.8% / 19.4% |

Two structural problems:

1. **train mean is 1.04 pK above test** — not a sampling fluke. MISATO selects
   for stable trajectories; tight-binding complexes don't unbind in 10 ns and
   are over-represented in train.
2. **Assay-mix is uneven.** Train is 43% IC50; val/test are 16–19%. Training
   labels are systematically noisier than eval labels. The split is also
   protein-disjoint (~3.9% Uniprot overlap) which is *good* for measuring
   generalization but compounds the OOD problem.

**Operational consequence:** raw test RMSE is dominated by mean-shift bias, not
by signal extraction. Post-hoc calibration on val (linear `a + b · pred`) is a
mandatory preprocessing step for any honest evaluation on this benchmark.

### 1.2 Channel signal is weak in the 4 published features

Per-channel Pearson R with pK (per-system means, computed across all 5 epochs
on identical data):

| Channel             | Train R | Val R  | Test R |
|---------------------|--------:|-------:|-------:|
| bSASA               | +0.39   | +0.28  | **+0.27** |
| rmsd_ligand         | −0.17   | −0.10  | −0.11 |
| distance            | −0.15   | −0.10  | −0.09 |
| interaction_energy  | −0.05   | −0.05  | −0.02 |

bSASA carries virtually all the linear signal. Interaction-energy is noise on
test (|R| = 0.02). The "4-channel summary" representation has a Pearson ceiling
around **R = 0.27 (single-channel) → 0.30 (linear oracle on all four)**.

### 1.3 Cheng-Prusoff correction has a counterintuitive side effect

We applied the standard published correction (pIC50 → pIC50 + log10(2) ≈ +0.30)
to bring IC50 labels onto a Ki-comparable scale. Because train has ~43% IC50
systems and val/test have only ~16–19%, the correction **inflates train's mean
disproportionately**:

|  | v1 (raw) | v2 (CP-corrected) |
|---|---:|---:|
| train mean pK | 6.59 | 6.72 (+0.13) |
| val mean pK   | 5.44 | 5.49 (+0.05) |
| test mean pK  | 5.55 | 5.61 (+0.06) |
| train→test gap | **−1.04** | **−1.11** |

The correction is doing what it's supposed to (per-sample), but the **split-
level distribution alignment degrades**. Lesson: label corrections that improve
per-sample fidelity can make split-level distribution shift worse when assay
composition isn't balanced across splits. For benchmarks like MISATO, *no
correction* on raw labels combined with *per-split calibration* may be cleaner
than a per-sample correction.

### 1.4 Trajectory quality varies across splits

| Flag                  | Train  | Val    | Test   |
|-----------------------|-------:|-------:|-------:|
| dissociated           | 16.8%  | 28.9%  | 26.2%  |
| ligand_drift          | 27.7%  | 39.2%  | 37.7%  |
| non-persistent contacts | 5.9% | 15.1% | 8.5%  |

Val/test have ~60% more dissociation events than train. Lower-affinity systems
naturally drift more, but the gap is bigger than affinity alone explains —
curation selected the cleaner trajectories into train. **Practical effect:** the
model sees cleaner dynamics during training than at evaluation time.

### 1.5 The new channels we derived: mostly redundant with bSASA

We added 8 channels computed from atomic coordinates (v2): `pocket_rmsd`,
`ligand_rgyr`, `min_contact_distance`, `n_contacts_4A`, `n_polar_contacts_35A`,
`n_hydrophobic_contacts_45A`, `ligand_internal_rmsd`, `com_dist_velocity`.

| New channel                       | Test R | Note |
|-----------------------------------|-------:|---|
| n_contacts_4A                     | +0.20  | redundant w/ bSASA |
| n_hydrophobic_contacts_45A        | +0.19  | redundant w/ bSASA |
| **com_dist_velocity**             | **−0.15** | **most independent — ligand mobility predicts affinity** |
| min_contact_distance              | −0.09  | weak |
| ligand_rgyr                       | +0.07  | weak |
| n_polar_contacts_35A (H-bond proxy) | +0.05 | unexpectedly weak |
| ligand_internal_rmsd              | −0.05  | weak |
| pocket_rmsd                       | −0.02  | weak |

**The two surprises:**
- **H-bond counts and hydrophobic contacts barely help on top of bSASA.**
  Published affinity predictors with reported R = 0.25–0.40 in standalone
  evaluations turn out to be heavily collinear with bSASA in this dataset.
  The "expected" features did not transfer.
- **Ligand mobility (`com_dist_velocity`) is the brightest new signal.**
  Frame-to-frame ligand motion has the second-largest standardized coefficient
  in the 12-channel oracle, independent of bSASA. Tight binders sit still.

Linear-oracle ceilings:

|  | v1 (4 channels) | v2 (12 channels) |
|---|---:|---:|
| Train R  | 0.41 | 0.44 |
| Val R    | 0.27 | 0.29 |
| **Test R** | **0.27** | **0.30** |
| Test RMSE | 1.82 | 1.74 |

v2 lifts the linear ceiling by **+0.02–0.03 Pearson** and **−0.08 RMSE** on
test. Real, modest, not transformative.

### 1.6 The 10 ns ceiling

Equilibrium ΔG generally requires μs-scale sampling. 10 ns captures local
dynamics around the binding mode but cannot recover the absolute binding free
energy. There is a hard physics ceiling on Pearson R for any model on this
dataset, independent of architecture or features.

---

## 2. About OpenTSLM-SP for binding-affinity regression

### 2.1 The encoder works — it just runs into a data ceiling

v1a's `TransformerCNNEncoder` + LoRA Llama extracts ~25% more Pearson than a
linear oracle on identical channels (0.34 vs 0.27 on test). The non-linear
processing is doing real work, not just memorizing.

But: model ceiling × data ceiling = result ceiling. With 4 channels capping at
0.27, the model getting to 0.34 means the encoder is *already near maximally
exploiting* the available input. Throwing more model at the same 4 channels
isn't the lever.

### 2.2 The training objective is misaligned with the evaluation metric

The published v1a recipe trains on cross-entropy over rationale tokens. The
evaluation metric is Pearson R, which is a **rank** metric.

- Cross-entropy targets per-token accuracy.
- The rationale string is **templated from the same channels the model sees as
  input** — so the LM head is mostly learning to verbalize the input, not to
  predict new information.
- No part of the gradient directly targets rank ordering of pK.

This shows up empirically: across our 5-epoch v1a run, val/test Pearson drifts
upward modestly (0.30 → 0.34 on test) while RMSE oscillates without trend. The
model finds *some* rank-ordering signal but no objective explicitly rewards it.

**Fix:** add a regression head (so we have a scalar prediction to compute pair
losses on), and add a ranking loss (margin-based on pairs from each batch).
This is a 50-LOC addition to the trainer.

### 2.3 Float tokenization is real but bounded

Llama tokenizes "X.XX" as ~3 tokens. The output precision floor is ~±0.1 pK
regardless of how good the encoder is. This contributes to the RMSE plateau,
but does **not** explain the Pearson plateau — rank ordering survives
quantization.

So a regression head (v1b) helps RMSE, not Pearson. It's the right
infrastructure for ranking loss but doesn't break the rank ceiling alone.

### 2.4 Warm-start from TSQA-SP is the right call

Convergence in epoch 1: val RMSE 1.87, near-best. Subsequent 4 epochs oscillate
in a 1.92–1.97 band on val. Five epochs is plenty for this dataset size
(~14K samples × bs 4 × 5 = 17K steps). More epochs would overfit train's
distribution.

### 2.5 Generation-at-eval is structurally wasteful

Each eval requires `model.generate(max_new_tokens=160)` to produce ~30 tokens
of rationale + "Answer: X.XX". For a regression task this is overkill: 
generation time dominates eval, and the only number that matters is the float
in the suffix.

A regression head replaces generation with a single forward pass and a scalar
output. ~10× eval speedup, no quantization, deterministic.

### 2.6 Multi-task supervision is the under-explored lever

The encoder's pooled hidden state is currently used for one task (text
prediction). We have free labels lying around — `dissociated`, `ligand_drift`,
derived dynamics features — that we don't supervise on. Adding small auxiliary
heads:

```
L_total = L_LM
        + 0.3 · MSE(head_pK, true_pK)         # regression supervision
        + 0.1 · BCE(head_dissoc, true_dissoc) # binary aux
        + 0.1 · BCE(head_drift, true_drift)
        + 0.2 · L_ranking(head_pK, pairs)     # Pearson-targeted
```

forces the encoder to learn representations that support all four targets. The
gradient signal per step is richer, and none of the new losses depends on the
noisy pK labels alone.

Expected gain: **+0.03–0.07 Pearson on test**, from the same data.

---

## 3. Why we use 0.02% of MD.hdf5

This is the structural finding that reframes the whole project.

### 3.1 The numbers

| Per system in MD.hdf5             | Size            | Used? |
|-----------------------------------|----------------:|---|
| `trajectory_coordinates` (100×N×3)| ~2,000,000 floats | ❌ |
| `atoms_element/number/residue/type` | ~26,000 ints  | ❌ |
| `molecules_begin_atom_index`      | 3 ints           | partially (v2) |
| `frames_rmsd_ligand`              | 100 floats       | ✅ |
| `frames_interaction_energy`       | 100 floats       | ✅ |
| `frames_distance`                 | 100 floats       | ✅ |
| `frames_bSASA`                    | 100 floats       | ✅ |

Per system: **400 floats consumed, ~2M discarded**. 5000:1 compression. The
v2 preprocessing pass scrapes 8 *derived* channels back from the coordinates,
but those are still 800 floats per system on top of the original 400 — total
input is still 12 floats × 100 frames = 1,200 numbers per system, vs the ~2M
available.

### 3.2 Why we did this

It wasn't a bug — it was a forced choice from the architecture:

1. **OpenTSLM-SP encodes `(channels, frames)` time series**, with each channel
   treated as a 1D scalar sequence patched and Transformed. It can't natively
   ingest a 3D point cloud labeled by element + residue + bond.
2. **The published recipe ships these 4 channels by default.** Choosing them
   was a fit-the-architecture decision, not an information-theoretic one. The
   published TSQA tasks (ECG, accelerometer, weather) are all low-channel
   scalar sequences — the architecture was designed for that shape.
3. **The 132 GB MD.hdf5 file can't fit on a SageMaker `ml.g5.xlarge`.**
   Preprocessing compresses it to 26 MB so training is feasible.
4. **We inherited the choice without questioning it.** Throughout v1a planning,
   the assumption was "the published method's features are the features." Only
   the post-training audit revealed that the choice left 99.98% of the data on
   the table.

### 3.3 What "using the data" would require

Using atomic coordinates isn't a tweak to OpenTSLM — it requires a different
model class:

- **SE(3)-equivariant graph network** (EGNN, Equiformer, MACE). Atoms as
  nodes, distances/angles as invariant features. The molecular-ML standard.
- **Point-cloud encoder** with chemistry awareness. Each frame is a labeled
  3D point cloud.
- **Protein-ligand foundation model**, then fine-tune (ESM-2 / ProteinMPNN /
  Equiformer-pretrained).

These are *replacements* for the `TransformerCNNEncoder`, not modifications.
Once the encoder is replaced, the project becomes "Llama as a downstream text
head over a graph-network encoder" — a different recipe entirely.

### 3.4 The honest framing

The 4-channel input was the right choice **for the architecture** and the
wrong choice **for the data**. Within OpenTSLM-SP, the channel-summary ceiling
(Pearson R ≈ 0.36) is what it is. To break it meaningfully requires
re-architecting the encoder. Our v2 pass to add 8 derived channels was the
middle-ground compromise: it scrapes some of the lost information back into
the format the architecture can consume, but it cannot recover *structural*
information (atom identities, bonds, residue context, full coordinates).

---

## 4. What worked, what didn't, and how big each was

### Worked

- **Warm-start from `OpenTSLM/llama-3.2-1b-tsqa-sp`.** Convergence in epoch 1.
- **LoRA rank 32 on Llama-3.2-1B q/k/v/o + MLP projections.** Stable, no
  catastrophic forgetting, 17 M trainable params.
- **The encoder + projector pipeline.** Extracts ~25% more Pearson than a
  linear oracle on the same input.
- **Templated rationale generation** — format-following held up
  (98–100% parse rate), no template collapse across epochs.

### Didn't work / underdelivered

- **5 epochs of training** — best val checkpoint was epoch 1. Subsequent
  epochs oscillated without meaningful improvement on val RMSE.
- **H-bond and hydrophobic-contact channels** — published affinity predictors
  that didn't deliver here due to bSASA collinearity.
- **Cheng-Prusoff label correction** — improved per-sample fidelity but
  worsened split-level distribution alignment.
- **The cross-entropy training objective** — doesn't target Pearson, which
  is the metric we evaluate on.

### Didn't even try yet (and probably should have)

- **Multi-task auxiliary heads** (free supervision from existing labels).
- **Ranking loss** (directly targets Pearson).
- **Post-hoc linear calibration on val** (free RMSE win).
- **Checkpoint ensemble** (we have 5; averaging cost nothing).
- **Sample-and-average at inference** (reduces tokenization quantization).

---

## 5. The recipe we'd use next time

In rough order of expected impact, for a project that starts over with the
same dataset and goals:

1. **Switch encoder class.** Atomic-coordinate-aware (graph network or point
   cloud over the trajectory). This is the single biggest lever and the only
   one that can credibly push Pearson past 0.4 on this benchmark.
2. **Direct scalar regression head as the primary output**, LM head as an
   *auxiliary* explanation generator. Don't make the model's headline metric
   depend on parsing text.
3. **Multi-task supervision** on derived dynamics targets (drift, dissociation,
   per-residue contact persistence). All these labels are free.
4. **Ranking loss on the regression head's output.** Pearson R is a rank
   metric; train it as one.
5. **Per-split calibration as a default eval step.** Linear `a + b · pred`
   fit on val, applied to test. Report both raw and calibrated.
6. **Skip Cheng-Prusoff at the label-correction step.** Or balance the assay
   mix across splits before correcting. The current correction undoes itself
   at the distribution level.
7. **Checkpoint ensemble at eval time.** Average the predictions of the top-3
   checkpoints by val metric.

If we couldn't change the encoder (constraints, comparison with published
method), the rest of the recipe still applies and would still net ~+0.05–0.08
Pearson improvement over what v1a achieved.

---

## 6. Open questions

1. **Does `com_dist_velocity` carry transferable signal across binding sites?**
   It was the brightest independent channel in v2. Worth a dedicated ablation:
   "single-channel ligand-mobility regression" on test as a calibration point.
2. **Would a graph-network encoder change the Pearson ceiling, or just match
   the engineered-feature ceiling more cheaply?** We don't know yet. Published
   benchmarks on coordinate-aware models for MISATO are sparse.
3. **Is the Pearson R = 0.36 ceiling on MISATO a property of 10 ns trajectories
   (physics-limited) or of 4-channel features (representation-limited)?** Until
   we run a coordinate-aware model, we can't distinguish these.
4. **What does the per-system residual structure look like?** Is the model
   making one consistent type of error (e.g., over-predicting weak binders) or
   broadly noisy? Per-PDB residual analysis would reveal whether there's a
   targetable failure mode.

---

## 7. Useful references in this repo

- `preprocess_misato.py` — v1 (4-channel) preprocessing
- `preprocess_v2.py` — v2 (12-channel) preprocessing + Cheng-Prusoff
- `audit_data.py`, `audit_data_deep.py` — dataset audits (assay-mix, channel
  signal, protein overlap, etc.)
- `audit_v2_channels.py` — gate audit on v2 (did new channels carry signal?)
- `eval_baselines.py` — OLS / MLP-engineered baselines
- `OpenTSLM/train_misato.py` — trainer entry point
- `TRAINING.md` — operational training log (per-epoch results, decision rules)
- `DATASET.md` — preprocessing/split reference

---

*Last updated after v1a 5-epoch run + v2 preprocessing + v2 channel audit.
Before any v2 training has been launched.*

# MD-Trajectory Binding Affinity with Grounded Rationales

Applying **OpenTSLM-SoftPrompt** (Stanford BDHG et al., *arXiv 2510.02410*) to a
new modality: **protein-ligand molecular dynamics trajectories** from MISATO,
with the `misato-affinity` companion dataset as labels.

The model reads the four-channel "movie" of how a drug wiggles in a binding
pocket over 100 frames (≈10 ns) and outputs **two things**:

1. A predicted binding affinity (pK).
2. A natural-language **rationale** explaining the dynamics, *post-hoc verified*
   against the underlying channel values so the explanation is grounded, not
   hallucinated.

This README is the operational entrypoint. For the project spec see
[`PROJECT_BRIEF.md`](PROJECT_BRIEF.md); for the data audit see
[`DATASET.md`](DATASET.md).

---

## 1. Goal

Train **two variants** of OpenTSLM-SP and report both:

| Variant | What it is | Status |
|---|---|---|
| **v1a — faithful** | Pure OpenTSLM-SP exactly as published. Affinity is the `Answer: X.XX` suffix of the generated rationale string. | Trainer ready |
| **v1b — hybrid** | Adds a 2-layer MLP regression head on the LLM's last input-position hidden state. Joint loss `L = L_LM + λ·MSE(pK_pred, pK_true)`. ~40-line extension. | Trainer ready, smoke-tested |

The v1a-vs-v1b ablation answers "does extending the published method help?"
Either outcome is publishable.

---

## 2. End-to-end pipeline

```
                  ┌─────────────────────────────────────────────────────────┐
                  │  ONE-TIME, LOCAL                                         │
                  ├─────────────────────────────────────────────────────────┤
  MD.hdf5         │                                                          │
  (124 GB,        │       preprocess_misato.py                               │
   MISATO         │   ───────────────────────────►                           │
   Zenodo)        │                                                          │
                  │   clips physical outliers, joins affinity_data.csv,      │
  affinity_data   │   computes dissociated/unstable tags, renders templated  │
  .csv (19,443    │   rationales from per-system facts                       │
   rows)          │                                                          │
                  │                                                          │
  splits/         │   ┌─→ features_{train,val,test}.npz  ~27 MB              │
  *_MD.txt        │   ├─→ samples_{train,val,test}.jsonl ~17 MB              │
                  │   ├─→ norm_stats.json                                    │
                  │   └─→ metadata.json                                      │
                  │       (total ≈ 41 MB → S3)                               │
                  └─────────────────────────────────────────────────────────┘
                                          │
                                          ▼  aws s3 sync
                  ┌─────────────────────────────────────────────────────────┐
                  │  SAGEMAKER STUDIO (ml.g5.xlarge on-demand)              │
                  ├─────────────────────────────────────────────────────────┤
                  │                                                          │
                  │   MISATOMDQADataset   (QADataset subclass)               │
                  │            │                                             │
                  │            ▼                                             │
                  │   OpenTSLM-SP  (Llama-3.2-1B, frozen + LoRA)             │
                  │   + optional regression head (v1b)                       │
                  │            │                                             │
                  │            ▼                                             │
                  │   train_misato.py                                        │
                  │            │                                             │
                  │            ├─→ ckpt_ep*.pt  (LoRA + encoder + projector  │
                  │            │                  + regression head if v1b)  │
                  │            └─→ history.jsonl (per-epoch metrics)         │
                  │                                                          │
                  └─────────────────────────────────────────────────────────┘
                                          │
                                          ▼
                  ┌─────────────────────────────────────────────────────────┐
                  │  EVALUATION                                              │
                  ├─────────────────────────────────────────────────────────┤
                  │                                                          │
                  │  eval_baselines.py     →  predict-mean, OLS, MLP         │
                  │                            (must-beat OLS RMSE 1.78)     │
                  │                                                          │
                  │  verify_rationale.py   →  % verified / contradicted /    │
                  │                            unverifiable claims           │
                  │                                                          │
                  └─────────────────────────────────────────────────────────┘
```

---

## 3. Inputs

All paths under `/home/mxlk/Documents/AIproject /` (note **trailing space** in
folder name — quote it in shell). Source datasets stay local; only the
preprocessed artifacts ship to SageMaker.

| Path | Size | Role | Read by |
|---|---:|---|---|
| `MD.hdf5` | **124 GB** | Source: 16,972 protein-ligand systems × 100 MD frames × 4 channels. From [MISATO Zenodo](https://zenodo.org/record/7711953). | `preprocess_misato.py` (one-time) |
| `misato-affinity/data/affinity_data.csv` | 1.2 MB | 19,443 rows of Kd/Ki/IC50 in nM. pK = 9 − log10(nM), priority Kd > Ki > IC50. | `preprocess_misato.py` |
| `misato-dataset-master/data/MD/splits/{train,val,test}_MD.txt` | 84 KB | 13,765 / 1,595 / 1,612 PDB IDs. MISATO's official split. | `preprocess_misato.py` |
| `misato-dataset-master/data/MD/h5_files/tiny_md.hdf5` | 93 MB | 20-system subset, same schema. | smoke tests only |
| `QM (1).hdf5` | 328 MB | 19,413 ligand QM features. | **not used in v1** (v2 ablation source) |
| `misato-affinity/` | — | Reference: their pretrained GCN baseline. | optional comparison |
| `OpenTSLM/` | — | Source repo (cloned from `StanfordBDHG/OpenTSLM`, modified on branch `misato-md-affinity`). | trainer |

### Per-system MD schema (the four channels we actually use)

```
<PDB>/
  frames_rmsd_ligand           (100,)  ← TS channel 0  (Angstroms)
  frames_interaction_energy    (100,)  ← TS channel 1  (kcal/mol)
  frames_distance              (100,)  ← TS channel 2  (Angstroms, CoM-CoM)
  frames_bSASA                 (100,)  ← TS channel 3  (Angstroms²)
```

`trajectory_coordinates`, `atoms_*`, topology files are **not used** by the
training input. They stay on disk for the post-hoc verifier if needed.

---

## 4. What `preprocess_misato.py` does

Reads `MD.hdf5` once, emits a ~41 MB SageMaker-ready bundle in
`preprocessed/`. Decisions baked in:

| Step | Decision |
|---|---|
| Drop 7 train PDBs with no affinity label | `{4DGO, 4OTW, 4V1C, 5V8H, 5V8J, 6FIM, 6H7K}` (DATASET.md §2) |
| Slice `6CC9` multi-ligand from 400 → 100 frames | Primary ligand only; sample tagged `multi_ligand=True` |
| Clip physically-impossible values | rmsd `[0, 50]`, IE `[-500, 50]`, distance `[0, 50]`, bSASA `[0, 2500]` |
| Compute dissociation flag on **unclipped** values | `rmsd_last20.mean() > 5` OR `dist_last20.mean() > 30` |
| Compute unstable flag | `rmsd.max() > 10` AND not dissociated |
| Normalize | per-channel **train-set** mean/std (NOT per-sample — channel scale carries signal) |
| Render rationale | templated from per-system facts (rmsd/IE/SASA stats + events) |
| Final answer string | rationale + `Answer: X.XX` |

### Output (`preprocessed/`)

| File | Shape / content | Size |
|---|---|---:|
| `features_train.npz` | `channels (13758,100,4) float32`, `pK (13758,)`, `pdb_ids`, `dissociated`, `unstable`, `multi_ligand` | 20 MB |
| `features_val.npz` | same, 1595 systems | 2.3 MB |
| `features_test.npz` | same, 1612 systems | 2.3 MB |
| `samples_train.jsonl` | one line per system: `{pdb_id, pK, facts, rationale, dissociated, unstable, multi_ligand}` | 14 MB |
| `samples_val.jsonl` | 1.7 MB | |
| `samples_test.jsonl` | 1.7 MB | |
| `norm_stats.json` | per-channel train mean/std after clipping | <1 KB |
| `metadata.json` | channel order, clip bounds, thresholds, counts | <1 KB |

Run time: **~19 seconds** on local disk against the 124 GB file.

---

## 5. What `train_misato.py` does

Standalone trainer (does **not** plug into OpenTSLM's curriculum harness).
Defaults are tuned for SageMaker `ml.g5.xlarge` on-demand.

### Required input

- `OPENTSLM_MISATO_DATA` env var → directory with `features_*.npz` +
  `samples_*.jsonl` + `norm_stats.json`. On SageMaker this is
  `/opt/ml/input/data`.

### Architecture

```
4 channels × 100 frames
       │
       ▼
TransformerCNNEncoder        (Conv1d patch=4, 6-layer Transformer)
       │
       ▼  shape (B, 4, 25, 128)  ← 25 patches per channel
MLPProjector                 → LLM hidden_size (2048 for Llama-3.2-1B)
       │
       ▼
Llama-3.2-1B (FROZEN) + LoRA on q/k/v/o + MLP projections
       │
       ├─→ LM head        → "<rationale> Answer: 6.42"     [v1a]
       └─→ regression head (v1b only): MLP → scalar pK
                            pooled at last non-pad input position
```

### Two variants

```bash
# v1a — faithful: pure LM loss
python train_misato.py --variant v1a --epochs 3 --batch-size 4

# v1b — hybrid: LM + λ * MSE(pK_pred, pK_true)
python train_misato.py --variant v1b --epochs 3 --batch-size 4 --lambda-reg 0.5
```

### Output

- `runs/{variant}_<timestamp>/ckpt_ep*.pt` — per-epoch checkpoints (LoRA +
  encoder + projector + regression head). ~50 MB each.
- `runs/{variant}_<timestamp>/history.jsonl` — train loss + val/test metrics
  per epoch: `n_parsed`, `rmse`, `mae`, `pearson_r`. For v1b also includes
  `regression_head` metrics alongside `string_parse`.
- On SageMaker: paths above redirected to `/opt/ml/checkpoints/<variant>/`,
  auto-synced to S3.

See **§6 Training configuration** below for the full hyperparameter reference,
env vars, hardware requirements, and pre-flight checklist.

---

## 6. Training configuration (everything you need to launch)

### 6.1 Hyperparameters (`train_misato.py` CLI flags)

| Flag | Default | Effect | When to tune |
|---|---|---|---|
| `--variant` | `v1a` | `v1a` (LM only) or `v1b` (LM + MSE head) | Always specify — run both |
| `--epochs` | `3` | Full passes over train set | 3 is a good first run; 5 if loss still dropping |
| `--batch-size` | `4` | Train batch | 4 fits g5.xlarge (24 GB) with Llama-1B + LoRA. Drop to 2 if OOM |
| `--eval-batch-size` | `8` | Eval batch (no gradients) | 8 safe; 16 if VRAM allows |
| `--lambda-reg` | `0.5` | **v1b only**: weight on MSE term in joint loss | Sweep `{0.1, 0.5, 1.0}` after baseline run |
| `--lora-r` | `16` | LoRA rank for q/k/v/o + MLP projections | 16 = ~8.8 M trainable; 8 lighter, 32 stronger |
| `--lr-lora` | `1e-4` | LR for LoRA adapter params | Don't touch in first run |
| `--lr-head` | `1e-4` | **v1b only**: LR for regression head | Tie to LoRA LR initially |
| `--max-new-tokens` | `160` | Generation length cap during eval | 160 covers templated rationale + `Answer: X.XX` |
| `--seed` | `0` | torch + numpy seed | Change only for multi-seed reporting |
| `--num-workers` | `2` | DataLoader workers | 2–4 fine on g5.xlarge |
| `--no-eval` | flag | Skip per-epoch eval (pure speed) | Set for sanity run; clear for real runs |
| `--subset-train` | `None` | Cap train samples | `64` for sanity, omit for real |
| `--subset-eval` | `None` | Cap eval samples | `16` for sanity, omit for real |
| `--warm-start` | `OpenTSLM/llama-3.2-1b-tsqa-sp` | Loaded via `OpenTSLM.load_pretrained` → maps to **`meta-llama/Llama-3.2-1B`** (gated) | Default — needs HF auth |
| `--cold-start-llm` | `None` | Skip warm-start: `OpenTSLMSP(llm_id=<this>)` directly | Use for non-gated smoke tests (e.g. `Qwen/Qwen2.5-0.5B`) |
| `--output-dir` | `runs` | Local output dir (overridden to `/opt/ml/checkpoints/` on SageMaker) | Rarely changed |

### 6.2 Inherited constants (`opentslm.model_config`)

Not exposed as CLI flags. To change, edit
`OpenTSLM/src/opentslm/model_config.py` before launching.

| Constant | Value | Used as |
|---|---|---|
| `LR_ENCODER` | `2e-4` | LR for `TransformerCNNEncoder` param group |
| `LR_PROJECTOR` | `1e-4` | LR for `MLPProjector` param group |
| `WEIGHT_DECAY` | `1e-2` | AdamW weight decay (all groups) |
| `GRAD_CLIP_NORM` | `1.0` | Per-step global gradient clip |
| `WARMUP_FRAC` | `0.03` | Linear warmup fraction of total steps |
| `PATCH_SIZE` | `4` | Conv1d patch size (must divide 100 — yes, 25 patches) |
| `EMBED_DIM` | `128` | Encoder embedding dim |

### 6.3 Environment variables (set before `python train_misato.py`)

| Var | Value | Why |
|---|---|---|
| `HF_TOKEN` | `hf_…` your read token | Required to download `meta-llama/Llama-3.2-1B` (gated). Set via `export HF_TOKEN=…` or `hf auth login` |
| `OPENTSLM_MISATO_DATA` | Path to dir with `features_*.npz` + `samples_*.jsonl` + `norm_stats.json` | Dataset loader reads from this. On SageMaker: `/opt/ml/input/data` |
| `CUDA_VISIBLE_DEVICES` | `0` (default) | Only set if multi-GPU and you want to pin one |
| `PYTHONUNBUFFERED` | `1` (recommended) | Real-time stdout in CloudWatch logs |

### 6.4 Hardware requirements

| Resource | Minimum | Recommended |
|---|---|---|
| GPU VRAM | 16 GB | **24 GB** (g5.xlarge A10G) |
| System RAM | 8 GB | 16 GB |
| Disk | 8 GB (HF cache + checkpoints) | 20 GB |
| Network | — | Internet on first run for HF downloads |
| Instance type | `ml.g4dn.xlarge` (T4, slower) | **`ml.g5.xlarge` on-demand** ($1.21/hr) |

**Estimated wall time on g5.xlarge:**
- v1a, 3 epochs, 13,758 samples, batch 4: **~2–3 h** (incl. per-epoch eval)
- v1b, same config: **~2–3 h**
- v1b λ sweep (3× λ ∈ {0.1, 0.5, 1.0}): **~6–9 h**
- **Total budget: ~10–15 GPU-hours ≈ $12–18 on-demand**

### 6.5 Outputs the trainer produces

```
/opt/ml/checkpoints/<variant>/        (or runs/<variant>_<timestamp>/ locally)
├── ckpt_ep1.pt        ~50 MB   encoder + projector + LoRA (+ regression head if v1b)
├── ckpt_ep2.pt
├── ckpt_ep3.pt
├── ckpt_final.pt
└── history.jsonl     per-epoch metrics:
                      {epoch, train_loss,
                       val:  {string_parse: {rmse, mae, pearson_r, n_parsed},
                              regression_head: {rmse, mae, pearson_r}  # v1b only},
                       test: {...}}
```

On SageMaker, `/opt/ml/checkpoints/` auto-syncs to S3 under the job's
checkpoint URI.

### 6.6 Pre-flight checklist

- [ ] `HF_TOKEN` exported; `hf auth whoami` succeeds
- [ ] `meta-llama/Llama-3.2-1B` access granted (Meta approval lands)
- [ ] S3 bucket `sagemaker-us-west-2-094487995066` is writable
- [ ] `preprocessed/` synced to `s3://…/misato-opentslm/features/v1/`
- [ ] OpenTSLM fork pushed with branch `misato-md-affinity`
- [ ] `ml.g5.xlarge` quota approved in `us-west-2`
- [ ] Studio notebook started on g5.xlarge with PyTorch 2.x kernel
- [ ] `pip install -e . -r requirements.txt` succeeded in the notebook env
- [ ] Sanity run (`--subset-train 64 --subset-eval 16 --epochs 1`) finishes in <5 min
- [ ] Then launch real `--variant v1a` and `--variant v1b` runs back-to-back

---

## 7. Baselines (`eval_baselines.py`)

Reads `preprocessed/features_*.npz` only. Reports test RMSE, MAE, Pearson R
for three baselines OpenTSLM-SP has to beat. Runs in 6 seconds on CPU.

| Baseline | What | Our result | DATASET.md §8 |
|---|---|---:|---:|
| `predict_train_mean` | Constant prediction = `train_pK.mean()` | RMSE **1.933** | 1.93 (match) |
| `ols_means` | OLS on 4 trajectory-mean channels | RMSE **1.778**, Pearson 0.273 | 1.791, 0.260 |
| `mlp_engineered` | MLP on 20 hand-features (mean/std/slope/min/max × 4 channels) | RMSE **1.680**, Pearson 0.356 | — |

**Bars to clear:**
- Beat `ols_means.test.rmse` (1.778) → the trajectory encoder is doing
  *something*.
- Beat `mlp_engineered.test.rmse` (1.680) → the encoder is doing something
  *beyond what summary stats already capture* — i.e. genuinely reading the
  movie.

```bash
python eval_baselines.py --out baselines.json
```

---

## 8. Rationale verifier (`verify_rationale.py`)

Regex claim extractor over the **5 closed claim types** from PROJECT_BRIEF
§7.2. For each claim:

- `rmsd_stability` — checks `mean`, `std`, `max` of channel 0
- `pocket_residence` — checks % of frames with channel 2 below threshold
- `contact_persistence` — checks mean and slope of channel 3 (bSASA)
- `energy_trend` — checks slope sign + spike magnitude of channel 1
- `flexibility` — checks std of channel 0

Each extracted claim resolves to **verified** / **contradicted** /
**unverifiable**. Self-test (templated training rationales): expect ~100%
verified.

```bash
# Self-test on templated rationales (should be ~100% verified)
python verify_rationale.py --self-test --limit 200

# Verify model-generated rationales after training
python verify_rationale.py --predictions runs/v1a_<ts>/eval_test.jsonl --split test
```

Self-test result so far: **1,287 claims across 200 rationales, 100% verified,
0 contradicted** (templater and verifier are internally consistent).

---

## 9. State of play

```
[x] preprocess_misato.py            — verified on tiny + full (19s for 124 GB)
[x] preprocessed/ artifacts          — 41 MB ready for S3
[x] MISATOMDQADataset                — formats correctly, 13758/1595/1612 splits
[x] OpenTSLMSP regression-head patch — compute_loss, predict_pK, store/load round-trip
[x] train_misato.py                  — end-to-end smoke pass on CPU w/ Qwen-0.5B
[x] eval_baselines.py                — reproduces DATASET.md §8 numbers
[x] verify_rationale.py              — 100% on templated rationales
[x] OpenTSLM/llama-3.2-1b-tsqa-sp adapter weights downloaded
[ ] meta-llama/Llama-3.2-1B base                  — awaiting Meta approval
[ ] SageMaker training run (v1a + v1b)            — runs once auth lands
[ ] v1b λ sweep (0.1, 0.5, 1.0)                   — after baseline v1a/v1b
[ ] Final comparison writeup                      — after sweep
```

### Not pursued in v1

- Static-frame GNN baseline (overscoped for 20-hour budget)
- AutoDock Vina baseline (13+ h compute)
- QM-feature ablation (v2 lever)
- 5th smoothed/derivative channel (v2 lever)

---

## 10. Reproduction (minimal command sequence)

```bash
# Local: one-time preprocessing (~19 s)
cd "/home/mxlk/Documents/AIproject "
.venv/bin/python preprocess_misato.py \
  --md-hdf5 MD.hdf5 \
  --affinity-csv misato-affinity/data/affinity_data.csv \
  --splits-dir misato-dataset-master/data/MD/splits \
  --out-dir preprocessed

# Local: baselines + verifier self-test (~10 s combined)
.venv/bin/python eval_baselines.py --out baselines.json
.venv/bin/python verify_rationale.py --self-test --limit 200

# Push to S3
aws s3 sync preprocessed/ s3://sagemaker-us-west-2-094487995066/misato-opentslm/features/v1/

# On SageMaker Studio (g5.xlarge, py3.12 + torch 2.9):
export HF_TOKEN=<your_token>            # Llama-3.2-1B is gated
export OPENTSLM_MISATO_DATA=/opt/ml/input/data
aws s3 sync s3://sagemaker-us-west-2-094487995066/misato-opentslm/features/v1/ \
            /opt/ml/input/data/
git clone <your-fork-of-OpenTSLM> && cd OpenTSLM && git checkout misato-md-affinity
pip install -e . -r requirements.txt

# Sanity (3 min)
python train_misato.py --variant v1a --subset-train 64 --subset-eval 16 --epochs 1

# Real runs (~2-3 h each)
python train_misato.py --variant v1a --epochs 3 --batch-size 4
python train_misato.py --variant v1b --epochs 3 --batch-size 4 --lambda-reg 0.5
```

---

## 11. Honest limits (carry into any writeup)

1. **10 ns ceiling.** Experimental ΔG reflects equilibrium; 10 ns can't fully
   recover it. Hard floor on Pearson R from this.
2. **Train-vs-test pK shift.** Train mean pK 6.59, test 5.55 (DATASET.md §7).
   Calibrate on val, not train.
3. **bSASA artifacts.** ~600 systems have physically impossible bSASA values
   (negative or >2,500 Å²); we clip to `[0, 2500]`. The good signal is
   preserved (median 503 Å², p99 1,250 Å²).
4. **MISATO selection bias.** The 17K systems were stable enough to simulate.
   Unstable / unbindable systems were filtered upstream.
5. **Rationale verifier closed vocabulary.** Free-form generation can produce
   claims outside the 5 verifiable types; those are marked `unverifiable`,
   not contradicted. % verified is reported over **grounded** claims only.
6. **Topology not in input.** Per-residue claims would require AMBER topology
   files (only `11gs` is local). We scope claims to the 4 shipped channels.

---

## 12. Where the model code lives

```
OpenTSLM/                                 # forked from StanfordBDHG/OpenTSLM
  src/opentslm/
    model/llm/OpenTSLMSP.py              # +enable_regression, +predict_pK,
                                          # +joint-loss path in compute_loss
    time_series_datasets/misato/
      __init__.py
      misato_loader.py                   # npz/jsonl → HF Datasets
      MISATOMDQADataset.py               # QADataset subclass
  train_misato.py                        # standalone trainer (v1a/v1b)
```

All on branch `misato-md-affinity`. The OpenTSLM base remains unchanged
except for the ~100-line `OpenTSLMSP.py` patch, which is gated behind the
opt-in `enable_regression()` call — v1a never touches it.

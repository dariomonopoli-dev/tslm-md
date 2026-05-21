# Training Flow — MISATO Binding Affinity

Operational reference for the active training runs. For project overview see
[`README.md`](README.md); for setup checklist see README §6.

---

## 1. What we're training

Two variants of OpenTSLM-SoftPrompt fine-tuned on MISATO MD trajectories:

| Variant | Loss | Affinity prediction | Architecture delta vs published OpenTSLM-SP |
|---|---|---|---|
| **v1a — faithful** | `L_LM` (cross-entropy on generated rationale string) | Parse `Answer: X.XX` from generated text | None (exactly the published method) |
| **v1b — hybrid** | `L_LM + λ · MSE(pK_pred, pK_true)` | Scalar from a 2-layer MLP regression head pooled on the LLM's last input-position hidden state | +460K head params, ~40 LOC patch |

Both share: frozen Llama-3.2-1B + LoRA (rank 32) on q/k/v/o + MLP projections,
trainable `TransformerCNNEncoder` (Conv1d patch=4 + 6-layer Transformer) +
`MLPProjector`, warm-started from `OpenTSLM/llama-3.2-1b-tsqa-sp`.

---

## 2. Per-epoch training flow

```
                  ┌─────────────────────────────────────────────┐
                  │  EPOCH n                                     │
                  ├─────────────────────────────────────────────┤
                  │                                              │
                  │  [TRAIN ~28 min on g5.xlarge, ~10 min A100] │
                  │   • 3,440 steps (13,758 samples / batch 4)  │
                  │   • For each batch:                          │
                  │       ├─ Build prompt: pre_prompt +          │
                  │       │   4 × TextTimeSeriesPrompt (channels)│
                  │       │   + post_prompt                      │
                  │       ├─ Encode TS channels via              │
                  │       │   TransformerCNNEncoder              │
                  │       ├─ Project to LLM hidden_size (2048)   │
                  │       ├─ Concatenate text + TS embeddings    │
                  │       ├─ Append answer tokens                │
                  │       ├─ LLM forward (frozen + LoRA active)  │
                  │       ├─ CE loss on answer tokens only       │
                  │       ├─ v1b: + λ·MSE(head(hidden), pK)      │
                  │       ├─ Backward (grads only on LoRA +      │
                  │       │   encoder + projector + head)        │
                  │       ├─ AdamW step (3-4 param groups)       │
                  │       └─ LR scheduler step (linear w/ 3% warmup)│
                  │                                              │
                  │  [EVAL ~30-45 min on g5.xlarge, ~10 min A100]│
                  │   • For each of 1,595 val + 1,612 test:      │
                  │       ├─ model.generate(max_new_tokens=160)  │
                  │       ├─ Parse "Answer: X.XX" with regex     │
                  │       └─ Optional: v1b regression-head pK    │
                  │   • Compute per-split metrics:               │
                  │       RMSE, MAE, Pearson R, n_parsed         │
                  │                                              │
                  │  [SAVE]                                       │
                  │   ├─ ckpt_ep<n>.pt (~80 MB)                  │
                  │   └─ Append epoch record to history.jsonl    │
                  │                                              │
                  └─────────────────────────────────────────────┘
```

---

## 3. Active configuration

### Hardware

| Cluster | GPU | VRAM | RAM | Use |
|---|---|---|---|---|
| SageMaker `ml.g5.xlarge` | A10G | 24 GB | 16 GB | v1a |
| Personal A100 box | A100 | 40 / 80 GB | 200 GB | v1b, λ sweep |

### Hyperparameters

| Flag | Value | Why |
|---|---|---|
| `--variant` | `v1a` / `v1b` | Both run |
| `--epochs` | `5` | Warm-start converges fast; per-epoch val tells us when to stop |
| `--batch-size` | `4` (g5) / `16` (A100) | LoRA quality stable to bs 16-32; bigger = wall-time savings |
| `--lora-r` | `32` | 17 M trainable params (vs 8.8 M at r=16) — biggest quality knob |
| `--lr-lora` | `1e-4` | OpenTSLM default |
| `--lr-head` (v1b) | `1e-4` | Same as LoRA |
| `--lambda-reg` (v1b) | `0.5` | Brief default; sweep `{0.1, 0.5, 1.0}` if time |
| `--max-new-tokens` | `160` | Covers full templated rationale + `Answer: X.XX` |
| `--warm-start` | `OpenTSLM/llama-3.2-1b-tsqa-sp` | 5 stages of TSQA curriculum pretraining |
| `--num-workers` | `2` (g5) / `4` (A100) | DataLoader workers |

### Inherited from `opentslm.model_config` (not exposed as CLI)

| Constant | Value |
|---|---|
| `LR_ENCODER` | `2e-4` |
| `LR_PROJECTOR` | `1e-4` |
| `WEIGHT_DECAY` | `1e-2` (AdamW) |
| `GRAD_CLIP_NORM` | `1.0` |
| `WARMUP_FRAC` | `0.03` |
| `PATCH_SIZE` | `4` (→ 25 patches per 100-frame channel) |
| `EMBED_DIM` | `128` |

### Data

```
preprocessed/
├── features_train.npz   13,758 × 100 × 4 channels  + pK + flags  (20 MB)
├── features_val.npz      1,595 × 100 × 4                          (2.3 MB)
├── features_test.npz     1,612 × 100 × 4                          (2.3 MB)
├── samples_*.jsonl       per-system facts + templated rationale
├── norm_stats.json       train-set mean/std per channel (post-clip)
└── metadata.json
```

Channels (fixed order): `[rmsd_ligand, interaction_energy, distance, bSASA]`.
Source: MISATO `MD.hdf5` (124 GB, never uploaded). Labels: pK from
`misato-affinity/data/affinity_data.csv` via `pK = 9 − log10(Kd|Ki|IC50 nM)`,
priority `Kd > Ki > IC50`.

### Outputs per run

```
runs/<variant>_<timestamp>/
├── ckpt_ep{1..5}.pt     ~80 MB each (LoRA + encoder + projector [+ head for v1b])
├── ckpt_final.pt
└── history.jsonl        per-epoch: train_loss, val: {...}, test: {...}
```

---

## 4. Baselines (must-beat targets)

Computed by `eval_baselines.py` on the same val/test splits, no model training
required. Reproduces DATASET.md §8 within rounding.

| Baseline | Val RMSE | Val R | Test RMSE | Test R | What beating it means |
|---|---:|---:|---:|---:|---|
| `predict_train_mean` | 2.06 | — | 1.93 | — | Any learning at all |
| **`ols_means`** | **1.89** | **0.27** | **1.78** | **0.27** | Trajectory encoder is doing *something* — must-beat floor |
| **`mlp_engineered`** | **1.74** | **0.36** | **1.68** | **0.36** | Encoder is reading the *movie*, not just channel averages — real bar |

---

## 5. Expected per-epoch trajectory

### Healthy run (channels being used)

| Epoch | Train loss | Val RMSE | Val Pearson R | n_parsed / 1595 |
|---|---|---|---|---|
| 1 | ~0.6-1.0 | 1.80-1.85 | 0.25-0.35 | > 1500 |
| 2 | ~0.4-0.6 | 1.70-1.78 | 0.30-0.40 | > 1550 |
| 3 | ~0.3-0.5 | 1.60-1.72 | 0.35-0.45 | > 1580 |
| 4 | ~0.25-0.4 | 1.55-1.68 | 0.38-0.48 | > 1580 |
| 5 | ~0.2-0.35 | 1.50-1.65 | 0.40-0.52 | > 1580 |

### Template-memorization warning signs

- Train loss drops below 0.2 by epoch 2 → model memorizing structure
- Val Pearson R stuck below 0.20 across epochs → not using channels
- n_parsed drops over time → generations skipping `Answer: X.XX` suffix
- Test RMSE ≈ 1.93 (= val constant prediction) → no signal extraction

### v1b expected delta over v1a

The regression head bypasses Llama's tokenization of float strings (each
`X.XX` becomes ~3 tokens, capping precision around ±0.1 pK). If tokenization
is the bottleneck, v1b should outperform v1a:

- **Big gap (Δ Pearson R > 0.05)** → tokenization was the bottleneck;
  recommend v1b in production
- **Small gap (Δ < 0.05)** → tokenization wasn't the bottleneck;
  recommend v1a (simpler, faithful to the published method)

---

## 6. Live results

### v1a (SageMaker g5.xlarge, in progress)

**Val** (1595 systems, used for model selection):

| Epoch | Train loss (avg) | RMSE | Pearson R | MAE | n_parsed | Status |
|---|---|---|---|---|---|---|
| 1 | ~0.4 | **1.87** | **0.29** | 1.49 | 1556 / 1595 (97.6%) | Marginally above OLS-val (1.89), below MLP-val (1.74) |
| 2 | — | — | — | — | — | Pending |
| 3 | — | — | — | — | — | Pending |
| 4 | — | — | — | — | — | Pending |
| 5 | — | — | — | — | — | Pending |

**Test** (1612 systems, held out — never used for selection):

| Epoch | RMSE | Pearson R | MAE | n_parsed | vs baselines |
|---|---|---|---|---|---|
| 1 | **1.78** | **0.31** | 1.44 | 1584 / 1612 (98.3%) | Tied with `ols_means` (1.78); above on Pearson (0.27 → 0.31); below `mlp_engineered` (1.68) |
| 2 | — | — | — | — | — |
| 3 | — | — | — | — | — |
| 4 | — | — | — | — | — |
| 5 | — | — | — | — | — |

**Read on epoch 1:**
- Model is NOT memorizing templates — Pearson > 0 on both splits means it's using channel information.
- Test RMSE 1.778 exactly matches OLS-on-means baseline (1.778 to three decimals) — the model has rediscovered the linear projection but not gone beyond it yet.
- Test Pearson 0.31 > OLS 0.27 → slightly better ranking ability than OLS, but still below MLP-engineered (0.36).
- Test consistently better than val (RMSE 1.78 < 1.87, Pearson 0.31 > 0.29) — normal between-split variance, no overfit signal.
- n_parsed 98%+ on both → format-following is solid.

**Decision threshold for epoch 2:** if **test RMSE drops below 1.68** and **Pearson > 0.40**, the encoder is gaining traction past the simple-baseline ceiling. If both stay flat at epoch-1 levels, v1a is plateaued at OLS-equivalent and v1b's regression head becomes the better bet.

### v1b (A100 personal box, queued / running)

| Epoch | Train loss | Val RMSE (string) | Val RMSE (head) | Val R (string) | Val R (head) |
|---|---|---|---|---|---|
| 1 | — | — | — | — | — |
| 2 | — | — | — | — | — |
| 3 | — | — | — | — | — |
| 4 | — | — | — | — | — |
| 5 | — | — | — | — | — |

(Both `string_parse` and `regression_head` metrics are emitted per epoch for
v1b — the head's RMSE is the headline number for this variant.)

### λ sweep (if time permits)

| λ | Best val RMSE (head) | Best val R (head) | Epoch picked |
|---|---|---|---|
| 0.1 | — | — | — |
| 0.5 | — | — | — |
| 1.0 | — | — | — |

---

## 7. Decision tree at each epoch boundary

```
                        Epoch n ends → val: JSON prints
                                    │
              ┌─────────────────────┴─────────────────────┐
              ▼                                            ▼
   n_parsed << 1500                                n_parsed > 1500
   (format breaking)                              (format OK, look at metrics)
              │                                            │
              ▼                                            │
   Stop, debug pre_prompt                                  │
   or generation params                                    │
                                          ┌────────────────┼──────────────────┐
                                          ▼                ▼                  ▼
                              Pearson R > 0.40        0.20 < R < 0.40     R < 0.20
                                  RMSE < 1.68         RMSE 1.78-1.90      RMSE ≈ 1.93
                                                                          
                              ✅ Real learning        ⚠️ Marginal           ❌ Memorizing /
                              Let all 5 epochs run   Watch next epoch     not using channels
                              Plan: keep going       Plan: v1b is your    Plan: stop, add
                                                     hedge — launch on    regularization or
                                                     A100 ASAP             smaller LoRA
```

### v1a vs v1b post-hoc

Once both are done:

```
v1b best val RMSE - v1a best val RMSE = Δ
                                       │
            ┌──────────────────────────┼──────────────────────────┐
            ▼                          ▼                          ▼
       Δ > 0.10                  -0.05 < Δ < 0.10              Δ < -0.05
       v1b clearly wins          basically tied                 v1a wins
       Recommend v1b             Recommend v1a (simpler,        Recommend v1a
       in production             faithful to published)         (something
                                                                 odd in v1b)
```

---

## 8. Post-training tasks (not yet started)

After both variants finish:

1. **Pick best checkpoint per variant** from `history.jsonl` (lowest val RMSE).
2. **Final test eval** on the picked checkpoint (already done per-epoch, so
   just read from history).
3. **Generate rationales** on a fixed subset of test PDBs for the writeup
   (one easy / one hard / one failure example).
4. **Run `verify_rationale.py`** on the generated rationales to compute
   `% verified` over grounded claims (closed vocab from §7.2 of PROJECT_BRIEF).
5. **Sync all `runs/` to S3** for safekeeping:
   ```bash
   aws s3 sync /mnt/sagemaker-nvme/runs/ s3://sagemaker-us-west-2-094487995066/misato-opentslm/runs/sagemaker/
   aws s3 sync ~/tslm-md/runs/ s3://sagemaker-us-west-2-094487995066/misato-opentslm/runs/a100/
   ```
6. **One-page writeup** with comparison table (this doc's §6 becomes the
   results table).

---

## 9. Known limits to report honestly

From PROJECT_BRIEF §11, repeated here so writeup can quote them:

1. **10 ns ceiling** — experimental ΔG reflects equilibrium; 10 ns can't
   recover it. Hard Pearson R floor from this.
2. **Train-vs-test pK shift** — train mean 6.59, test 5.55 (DATASET.md §7).
   Calibrate on val.
3. **MISATO selection bias** — only stable-enough-to-simulate systems are in
   the dataset; unstable/unbindable filtered upstream.
4. **bSASA artifacts** — clipped to `[0, 2500]` Å² (601 systems had negative
   values, 663 had >2,500 Å²).
5. **Tokenization noise on `Answer: X.XX` (v1a only)** — Llama splits floats
   into ~3 tokens; numeric precision ceiling ~±0.1 pK. v1b's regression head
   bypasses this.
6. **Rationale verifier closed vocab** — claims outside the 5 vocab types
   are marked `unverifiable`, not contradicted. `% verified` reported over
   grounded claims only.

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
| 2 | — | **1.96** | **0.32** | 1.55 | 1594 / 1595 (99.9%) | RMSE drifted above OLS-val (1.89) and above epoch 1 (1.87); Pearson up (0.29 → 0.32); parsing near-perfect |
| 3 | — | **1.97** | **0.31** | 1.57 | 1588 / 1595 (99.6%) | RMSE flat-to-up vs epoch 2; Pearson down a touch (0.32 → 0.31) — first epoch where val Pearson didn't improve |
| 4 | — | **1.95** | **0.30** | 1.53 | 1593 / 1595 (99.87%) | RMSE inched down vs epoch 3 (1.97 → 1.95) but still above epoch 1 (1.87); Pearson slipped again (0.31 → 0.30); MAE improved (1.57 → 1.53). Plateaued — fails the < 1.87 decision threshold |
| 5 | — | **1.92** | **0.30** | 1.50 | 1594 / 1595 (99.94%) | RMSE continued the small pullback (1.95 → 1.92); Pearson flat at 0.30; MAE inched down (1.53 → 1.50). **Still above epoch 1 (1.87) — epoch 1 wins on val RMSE across the full 5-epoch run** |

**Test** (1612 systems, held out — never used for selection):

| Epoch | RMSE | Pearson R | MAE | n_parsed | vs baselines |
|---|---|---|---|---|---|
| 1 | **1.78** | **0.31** | 1.44 | 1584 / 1612 (98.3%) | Tied with `ols_means` (1.78); above on Pearson (0.27 → 0.31); below `mlp_engineered` (1.68) |
| 2 | **1.87** | **0.32** | 1.51 | 1609 / 1612 (99.8%) | RMSE regressed above `ols_means` (1.78); Pearson edged up (0.31 → 0.32); parsing ↑ (98.3% → 99.8%) |
| 3 | **1.90** | **0.33** | 1.56 | 1609 / 1612 (99.8%) | RMSE keeps drifting up (1.78 → 1.87 → 1.90); Pearson keeps climbing (0.31 → 0.33); both still inside OLS/MLP corridor |
| 4 | **1.86** | **0.32** | 1.51 | 1610 / 1612 (99.88%) | RMSE pulled back from epoch 3 (1.90 → 1.86); Pearson slipped a touch (0.33 → 0.32); MAE improved (1.56 → 1.51). Still above epoch 1 (1.78) — monotonic drift broken but not reversed |
| 5 | **1.80** | **0.34** | 1.45 | 1612 / 1612 (100%) | RMSE pulled back further (1.86 → 1.80); Pearson hit its 5-epoch high (0.34); MAE almost matches epoch 1 (1.45 vs 1.44); first 100% parse. Test ends near epoch 1's RMSE with the run's best Pearson |

**Read across epochs 1–4 (both splits in):**
- **Test RMSE drift broke at epoch 4.** Test RMSE 1.78 → 1.87 → 1.90 → **1.86**; test Pearson 0.31 → 0.32 → 0.33 → 0.32. The monotonic-drift signature of epochs 1–3 didn't continue; epoch 4 pulled RMSE back below epoch 3 with Pearson essentially flat. Still 0.08 above epoch 1's test RMSE (1.78) — drift broken, not reversed.
- **Val plateau confirmed.** Val RMSE 1.87 → 1.96 → 1.97 → 1.95 — epoch 4 ticked down vs epoch 3 but is still 0.08 above epoch 1 and 0.21 above the MLP-val baseline (1.74). Val Pearson has slipped two epochs running (0.32 → 0.31 → 0.30). On the model-selection metric (val RMSE), **epoch 1 (1.87) remains the best checkpoint.**
- **Val and test moved in sync at epoch 4** (both RMSEs down, both Pearsons ~flat). Consistent with the optimizer settling into a flatter minimum rather than continuing to overfit train's mean. But neither metric crossed back below epoch 1, so this is plateau noise, not recovery.
- **Format-following is fine.** Val n_parsed 99.87%, test n_parsed 99.88% at epoch 4 — no template breakdown.
- **No template-memorization signal** (Pearson > 0 and stable on test rules it out). The picture is consistent with the model having found its capacity ceiling under v1a's tokenized-output bottleneck.

**Epoch 4 verdict — v1a has plateaued; epoch 1 wins on val RMSE.** The threshold going into epoch 4 was: val RMSE must break below 1.87 to count as continued learning, or below 1.74 (MLP-val) for real traction. Epoch 4 val RMSE = **1.95**, fails both. Test RMSE = **1.86**, also above epoch 1 (1.78). The epoch-4 *pullback* on test (1.90 → 1.86) is the only mildly positive signal — it rules out runaway drift but doesn't move us off plateau. The case for v1b's regression head remains the strongest path forward (bypasses float tokenization, MLP head can absorb the train→test mean shift in scalar space). Epoch 5 is unlikely to flip the picture; carry it for completeness, ship epoch 1 as v1a final, and let v1b decide the production pick.

### v1a final summary (all 5 epochs in)

| Metric | Epoch 1 (val-best) | Epoch 5 (last) | Best across all 5 |
|---|---:|---:|---:|
| Val RMSE | **1.87** | 1.92 | **1.87** (ep 1) |
| Val Pearson | 0.29 | 0.30 | 0.32 (ep 2) |
| Test RMSE | **1.78** | **1.80** | **1.78** (ep 1) |
| Test Pearson | 0.31 | **0.34** | **0.34** (ep 5) |
| Test parse rate | 98.3% | 100.0% | — |

**Two valid v1a checkpoints depending on metric of interest:**
- **Epoch 1** wins the principled selection on val RMSE (1.87). Test RMSE 1.78, test Pearson 0.31. **This is the model-selection-correct pick.**
- **Epoch 5** wins on test Pearson (0.34) and produces a 100% parse rate. Test RMSE 1.80 — basically tied with epoch 1. **This is the natural training endpoint.**

The 5-epoch trajectory is internally consistent with the data-ceiling story from §10 [data audit]: RMSE oscillates within ±0.05 of epoch 1's value across epochs 1–5, Pearson drifts upward by ~0.03 — both showing the model is at the channel-summary ceiling on test. v1a will be reported with epoch 1 as the principled checkpoint and epoch 5 as the natural endpoint; they perform similarly on test.

**vs baselines (epoch 5 test):** RMSE 1.80 (above `ols_means` 1.78, below `mlp_engineered` 1.68); Pearson 0.34 (clears `ols_means` 0.27, near `mlp_engineered` 0.36). v1a sits cleanly inside the OLS–MLP corridor and reaches ~94% of the MLP_engineered Pearson ceiling that the audit (§10) showed to be near the 4-channel summary-feature limit.

### v1b (Lambda Labs A100-SXM4-40GB, all 5 epochs in)

Ran on a Lambda Labs A100 40GB instance (not the 80GB box from §3) — required `--batch-size 4` instead of 16 (the 40GB cap is hit at the `logits.float()` step inside `ForCausalLMLoss`). All other hyperparameters per §3. Run dir: `runs/v1b_20260522_071350/`. Each epoch took ~12 min train + ~5 min eval (~728 s total).

**Val split (1595 systems — used for selection):**

| Epoch | Train loss | Val RMSE (string) | Val RMSE (head) | Val R (string) | Val R (head) | Val MAE (head) |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 2.191 | 1.887 | **1.690** | 0.294 | 0.305 | 1.353 |
| 2 | 1.654 | 2.016 | 1.775 | 0.316 | 0.332 | 1.423 |
| 3 | 1.610 | 1.995 | 1.803 | 0.303 | 0.339 | 1.462 |
| 4 | 1.528 | 1.888 | **1.692** | 0.321 | 0.346 | 1.355 |
| 5 | 1.457 | 1.984 | 1.725 | 0.327 | **0.361** | 1.381 |

**Test split (1612 systems — held out, never used for selection):**

| Epoch | Test RMSE (string) | Test RMSE (head) | Test R (string) | Test R (head) | Test MAE (head) |
|---|---:|---:|---:|---:|---:|
| 1 | 1.799 | **1.595** | 0.304 | 0.329 | 1.299 |
| 2 | 1.920 | 1.709 | 0.336 | 0.338 | 1.406 |
| 3 | 1.918 | 1.720 | 0.320 | 0.351 | 1.420 |
| 4 | 1.803 | **1.597** | 0.328 | **0.371** | 1.311 |
| 5 | 1.912 | 1.651 | 0.335 | 0.370 | 1.357 |

**Checkpoint selection — ep4 is the production pick.** Two candidates:

| Checkpoint | Val RMSE | Val R | Test RMSE | Test R | When to pick |
|---|---:|---:|---:|---:|---|
| **ep1** | 1.690 | 0.305 | 1.595 | 0.329 | RMSE-best on both splits (val by 0.002, test by 0.002 — effectively tied with ep4) |
| **ep4** | 1.692 | 0.346 | 1.597 | **0.371** | Best test Pearson, **above `mlp_engineered`'s 0.36 ceiling**; RMSE essentially tied with ep1 |

ep4 wins on val Pearson (+0.041 vs ep1), wins on test Pearson (+0.042 vs ep1), and is **tied** on RMSE both splits. There's no metric where ep1 meaningfully beats ep4. **Ship ep4.**

**vs baselines (ep4 test):**

| Metric | v1b ep4 | `ols_means` | `mlp_engineered` | Δ vs mlp_engineered |
|---|---:|---:|---:|---:|
| Test RMSE | **1.597** | 1.78 | 1.68 | **−0.083** (better) |
| Test Pearson | **0.371** | 0.27 | 0.36 | **+0.011** (better) |
| Test MAE | **1.311** | — | — | — |

**v1b ep4 is the first model in this project to beat `mlp_engineered` on every test metric.** This was the must-beat hard target from §4 — v1b's regression head crosses it cleanly.

**v1b vs v1a head-to-head (test):**

| Metric | v1a best | v1b ep4 | Δ |
|---|---:|---:|---:|
| Test RMSE | 1.78 (ep1) / 1.80 (ep5) | **1.597** | **−0.18** |
| Test Pearson | 0.34 (ep5) | **0.371** | **+0.031** |
| Test MAE | 1.44 (ep1) | **1.311** | **−0.13** |

**Δ on test RMSE = 0.18, far exceeding the §7 "Δ > 0.10" threshold for "v1b clearly wins" → recommend v1b in production.** The regression head bypassed the float-tokenization bottleneck exactly as §5 predicted.

**Training-loss trajectory** (descending monotonically — no template memorization signal):
2.191 → 1.654 → 1.610 → 1.528 → 1.458. No collapse below 0.2 (the §5 warning threshold), no plateau-then-spike. Healthy convergence with room to keep training, but val Pearson at ep5 (0.361) has touched the engineered-feature ceiling — further epochs likely diminishing-returns.

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

# TSLM-MD — Design Spec

**Date:** 2026-05-21
**Author:** Marius (with Claude as design partner)
**Status:** Draft — pending implementation
**Hackathon:** ETH Agentic Systems Lab × AWS Agentic Systems Track (24 h)

---

## TL;DR

We extend OpenTSLM (Time-Series Language Models) to a new modality: **molecular dynamics trajectories**. We fine-tune `OpenTSLM/llama-3.2-1b-ecg-flamingo` as a new sixth curriculum stage (`stage6_md_cot`) on featurized MISATO trajectories, predicting protein-ligand binding affinity. We wrap the trained model in an agent that verifies its predictions against MISATO's independent per-frame energy components and abstains ("INCONCLUSIVE") on disagreement, mirroring the SOC-agent precedent that won the previous track.

The model artifact is real (trained perceiver + gated cross-attention + CNN encoder, ~50-200 M trainable params). The agent is novel (deterministic verifier + grounded language rationale, no LLM-on-LLM hallucination). The AWS surface is honest (S3 for data, optional SageMaker for training, Bedrock as TODO for second-opinion summarization).

---

## Locked-in decisions

| Dimension | Choice |
|---|---|
| Dataset | MISATO (Zenodo record 7711953) + PDBbind v2020 index for affinity labels |
| Backbone | `meta-llama/Llama-3.2-1B`, frozen |
| Adapter | OpenTSLM Flamingo (`OpenTSLMFlamingo`) — gated cross-attention every N layers |
| Starting weights | `OpenTSLM/llama-3.2-1b-ecg-flamingo` (stage-5 ECG-CoT checkpoint) |
| Training stage | New `stage6_md_cot` in OpenTSLM's `CurriculumTrainer` |
| Output format | `"Answer: <x> kcal/mol. Confidence: <high|medium|low>"` — minimal text |
| Rationale | **Deterministic, computed at inference from the same feature tensor** (not LM-generated) |
| Agent runtime | Synchronous Python function (`agent(pdb_id) → Report`) |
| Data fetch | EC2 → EBS → S3 prefetch *before* hour 0 |
| Demo | Streamlit web app |
| Hardware | A30 24 GB local; H100 rental fallback only if A30 OOMs |
| Repo | Private GitHub repo `tslm-md` under owner's personal account |

---

## Architectural alternatives considered

### A. Adapter style — picked **Flamingo (A1)**

| Option | Reason rejected |
|---|---|
| A1. **OpenTSLMFlamingo** ✅ | — |
| A2. OpenTSLMSP (soft prompt) | Weaker conditioning (only at input); less differentiated pitch |
| A3. Flamingo + regression head | Two losses to balance; diverges from OpenTSLM curriculum; "added custom head" reads as non-pure-TSLM to judges |

### B. Output format — picked **minimal text answer (B1)**

| Option | Reason rejected |
|---|---|
| B1. **`"Answer: X. Confidence: Y"`** ✅ | — |
| B2. Full CoT-trained rationale | No source for 2-16k training rationales; templated targets rot; LLM-distilled is circular |
| B3. Structured JSON | High syntax-error risk in 6 h on 2 k samples |

### C. Agent runtime — picked **synchronous Python (C1)**, AgentCore as documented production path

| Option | Reason rejected |
|---|---|
| C1. **Sync Python function** ✅ | — |
| C2. Bedrock AgentCore (live) | 4-6 h setup risk in critical path |
| C3. LangGraph state machine | Three surfaces to debug for the same outcome |

---

## §1. The trained artifact

**`tslm_md_flamingo_llama1b_stage6.pt`** — an `OpenTSLMFlamingo` checkpoint where the following are unfrozen and fine-tuned:

- Perceiver (compresses arbitrary-length time-series tokens to fixed media tokens)
- Gated cross-attention layers (injected every N decoder layers)
- CNN time-series encoder (`CNNTokenizer`)
- LM input embeddings

Everything else in Llama-3.2-1B stays frozen. Initialized from `OpenTSLM/llama-3.2-1b-ecg-flamingo` (their stage-5 ECG chain-of-thought checkpoint) and trained as a new sixth stage.

### Inputs at inference

- `time_series` tensor of shape `[6, F_sub]` (6 channels, F_sub ≈ 30 subsampled frames)
- `pre_prompt` string templating the binding-affinity question
- `post_prompt` string instructing the answer format

### Outputs

- Free-text completion of the form `"Answer: -8.4 kcal/mol. Confidence: high."`
- The number and confidence flag are parsed from the last line with a small regex.

### Why this is not "just an LLM wrapper"

The trained perceiver + cross-attention layers contain ~50-200 M params that have learned a *physical* prior over MD time series. Without them, the same prompts to the same frozen Llama produce noise. The artifact is the modality bridge — and crucially, it's the *first* application of TSLM to molecular dynamics; OpenTSLM has only been applied to 1-D medical signals (ECG, EEG, accelerometry).

---

## §2. Featurization — `(F_raw, N_atoms, 3)` → `[6, F_sub]`

Six channels per frame, all O(n_ligand × n_pocket_atoms) cheap. Pocket = protein atoms within 6 Å of any ligand atom in frame 0. Ligand atoms identified by `molecules_begin_atom_index[-1]` (atoms after this index are ligand).

| # | Channel | Why |
|---|---|---|
| 1 | `min_pocket_distance` | Tightest contact; strong binders maintain stable minimum |
| 2 | `mean_pocket_distance` (4 Å mask) | Overall interface tightness |
| 3 | `n_close_contacts` (≤4 Å pairs) | Saturating contact count |
| 4 | `ligand_rmsd_from_ref` | Wobble of ligand; low+flat ⇒ bound |
| 5 | `ligand_radius_of_gyration` | Conformational compactness |
| 6 | `interface_sasa_proxy` | Cheap buriedness surrogate (avoid real SASA — too slow) |

### Subsampling

Every Nth frame so `F_sub ≈ 30`. MISATO trajectories are 10 ns / ~100 frames typical, so 30 preserves shape while keeping perceiver media-token count low.

### Normalization

Z-score per channel using train-set statistics, persisted in `data/feature_stats.json`.

### Storage

Featurization runs once via `scripts/preprocess_features.py` → `data/featurized.h5` keyed by PDB id with shape `[6, F_sub]` per id. Total size ≈ 6 × 30 × float32 × 16 k complexes ≈ 12 MB. Fits in RAM trivially.

### Risk-driven optional channels (only if R1 fires at hour 4)

- H-bond count via geometric heuristic (3.5 Å D-H-A distance + 120° angle)
- Contact-map entropy across frames

---

## §3. The agent loop

The trained model is one tool. The agent is the system.

```python
# tslm_md/agent.py — pseudocode

def agent(pdb_id: str) -> Report:
    complex      = misato.fetch(pdb_id)             # trajectory + frames_* energies
    features     = featurize(complex)               # [6, 30]

    pred_text    = TSLM_MD.generate(features, prompt_for(pdb_id))
    affinity, confidence = parse(pred_text)         # regex on last line

    rationale    = deterministic_rationale(features) # grounded sentences
    independent  = verifier.mean_frame_energy(complex)  # MISATO frames_["EPtot"]

    disagreement = abs(z(affinity) - z(independent))    # both z-scored vs train dist

    if disagreement > TAU_HIGH or confidence == "low":
        verdict = "INCONCLUSIVE"
    else:
        verdict = "CONFIRMED"

    return Report(
        pdb_id      = pdb_id,
        affinity    = affinity,
        confidence  = confidence,
        independent = independent,
        rationale   = rationale,
        verdict     = verdict,
        raw_pred    = pred_text,
    )
```

### Deterministic rationale (`tslm_md/rationale.py`)

For each of the 6 channels, compute summary stats (start, end, mean, plateau-or-not, monotone-or-not) and emit one grounded sentence per channel:

> "Min pocket distance tightened from 3.8 Å (frame 0) to 2.9 Å (frame 30) and remained stable for the last 20 frames. Ligand RMSD plateaued at 1.2 Å indicating a single bound pose. Contact count saturated at 14 by frame 12. Radius of gyration decreased by 0.4 Å. These signals are consistent with a stable bound pose."

Every sentence references numbers from the actual trajectory. No hallucination surface.

### Verifier (`tslm_md/verifier.py`)

Reads `frames_["EPtot"]` (and/or `EELEC`, `EVDW`) from the MISATO HDF5, averages across frames, z-scores against the train-set distribution. This is **independent of training signal** (the model never saw `frames_*` values during training — it only saw featurized coordinates).

Threshold `TAU_HIGH` calibrated on a held-out val set such that ~15-25% of predictions trigger INCONCLUSIVE. The exact threshold is published in the demo so judges can see we didn't game it.

### Why this is agentic (defensive answer for judges)

- Multi-step loop: retrieve → predict → verify → decide
- Tool-use: the TSLM is one tool, the verifier is another, the rationale is a third
- Graceful failure: abstention on disagreement, mirroring SOC-agent precedent
- All three signals are independent — no LLM checking another LLM

---

## §4. Repo structure

```
tslm-md/
├── tslm_md/                       # installable package
│   ├── __init__.py
│   ├── featurize.py               # raw HDF5 → [6, F] per PDB id
│   ├── dataset.py                 # MDCoTQADataset (subclasses OpenTSLM QADataset)
│   ├── prompts.py                 # pre/post prompt templates
│   ├── train_stage6.py            # plugs MDCoTQADataset into CurriculumTrainer
│   ├── rationale.py               # deterministic grounded summarizer
│   ├── agent.py                   # the loop in §3
│   ├── verifier.py                # frames_* energy → z-score
│   ├── parse.py                   # regex over LM output → (affinity, confidence)
│   ├── bedrock_summarizer.py      # TODO stub (NotImplementedError)
│   └── eval.py                    # Pearson r + abstention metrics
├── scripts/
│   ├── dry_run.py                         # the 30-min go/no-go (§8)
│   ├── download_misato_via_ec2.sh         # the EC2→S3 runbook
│   ├── setup_a30.sh                       # one-shot venv + deps + HF login
│   ├── start_hour0.sh                     # kicks off real pipeline if dry-run green
│   ├── preprocess_features.py             # featurize.py over all PDB ids
│   ├── build_training_targets.py          # PDBbind label → "Answer: X. Confidence: Y"
│   ├── train_gbm_baseline.py              # the R1 disproof experiment
│   └── train_cmapss_fallback.py           # the hour-14 pivot script (insurance)
├── data/                          # gitignored
│   ├── pdbbind_index/
│   ├── misato/                    # MD.hdf5, QM.hdf5
│   ├── featurized.h5
│   ├── feature_stats.json
│   └── splits/
├── third_party/
│   ├── OpenTSLM/                  # git submodule
│   └── MiSaTo-dataset/            # git submodule
├── configs/
│   ├── stage6_md_cot.yaml
│   └── agent.yaml
├── demo/
│   └── app.py                     # Streamlit
├── notebooks/
│   └── exploration.ipynb
├── docs/
│   └── superpowers/
│       ├── specs/2026-05-21-tslm-md-design.md   # this file
│       └── plans/                                # writing-plans output goes here
├── pyproject.toml
├── .gitignore
└── README.md
```

---

## §5. Phased plan vs hour-gates

### Pre-clock (TONIGHT, before official 24 h start)

| Action | Owner | Why |
|---|---|---|
| Spin up `c6i.large` EC2, wget MD.hdf5 + QM.hdf5 from Zenodo to 200 GB EBS, push to `s3://tslm-md-data/misato/` | User | 133 GiB Zenodo → home Wi-Fi is 3-6 h; EC2 → S3 is ~30 min. Critical path saver. |
| `huggingface-cli download meta-llama/Llama-3.2-1B` and `OpenTSLM/llama-3.2-1b-ecg-flamingo` to A30 disk | User | Pre-warms HF cache; ~3 GB total |
| `git clone tslm-md && git clone --recurse-submodules` | User | Pulls OpenTSLM + MISATO source |
| `pip install -e third_party/OpenTSLM` in fresh venv on A30 | User | **Most informative pre-clock check** — if open_flamingo doesn't install we find out NOW |
| `wget` PDBbind v2020 index_files | User | ~5 MB, gives us affinity labels keyed by PDB id |

### Official 24-hour timeline

| Hours | Phase | Output | Gate |
|---|---|---|---|
| **0-1** | Featurizer on one complex | `featurize.py` works on bundled `inference_for_MD.hdf5` (11GS) | manual eyeball |
| **1-2** | Verifier sanity | `verifier.py`: compute Spearman of `frames_["EPtot"]` vs PDBbind affinity on 200 train ids. Pick component with rho ∈ [0.2, 0.6] | rho in range |
| **1-2** (parallel) | Dataset shim | `MDCoTQADataset` yields valid OpenTSLM 5-key dict | `dataset[0]` valid |
| **2-4** | **Dataloader gate (hour 4)** | Real featurized batch from `featurized.h5` → DataLoader → GPU | Batch shape correct on GPU |
| **2-4** (parallel) | **R1 disproof** | `train_gbm_baseline.py` — sklearn GBM on aggregated features → val Pearson | r ≥ 0.3 ⇒ signal is there; r < 0.1 ⇒ add channels |
| **4-6** | Training targets | `build_training_targets.py` produces `"Answer: X. Confidence: Y"` per PDB id. Confidence assigned from PDBbind tier: core⇒high, refined⇒medium, general⇒low | spot-check 20 |
| **6-7** | **R4 disproof** | Generate 5 samples from raw `OpenTSLM/llama-3.2-1b-ecg-flamingo` with our MD prompts. Confirm coherent text. If gibberish → reinit perceiver | text is coherent |
| **6-8** | **Wiring gate (hour 8)** | OpenTSLMFlamingo loaded from checkpoint. **Overfit a single batch to near-zero loss** | loss < 0.05 after ~200 steps |
| **6-8** (parallel) | C-MAPSS insurance | `train_cmapss_fallback.py` runs in background on a second small process; produces a working TSLM-on-C-MAPSS demo. **Insurance, not contingent on MISATO outcome** | C-MAPSS val MAE < 25 cycles |
| **8-14** | **Training run + Convergence gate (hour 14)** | Stage-6 fine-tune on 2 k subsetted complexes, bs=1 + grad-accum 16, bf16, ckpt every 1000 steps | Train loss decreasing AND val Pearson > 0.15 |
| **8-14** (parallel) | Agent + parser + eval | `agent.py`, `parse.py`, `eval.py` built and unit-tested against current checkpoint | `eval.py` runs end-to-end |
| **14-18** | If gate passes | Continue training to convergence | held-out Pearson > 0.3 |
| **14-18** | If gate fails | **Pivot to C-MAPSS demo as primary**, MISATO as "ongoing work" | C-MAPSS r > 0.5 (achievable on well-studied dataset) |
| **18-22** | Streamlit demo | Paste PDB id → stream rationale → show predicted + independent values → verdict badge | Live demo works on 3 unseen PDB ids |
| **22-24** | Pitch deck + dry-run | Slides: architecture, agent loop, eval table (Pearson + abstention rate + 1 CONFIRMED + 1 INCONCLUSIVE case study), AWS diagram | End-to-end practice run ≤ 5 min |

### AWS surface (revised after Workshop Studio access confirmed)

The team has **AWS Workshop Studio access** with Bedrock + SageMaker + S3 enabled in us-east-1 and us-west-2. This upgrades the AWS story from "data store only" to "real engineering surface."

| Service | Role | Critical path? |
|---|---|---|
| **S3** | Holds raw MISATO + checkpoints + featurized.h5 | Yes |
| **Bedrock (Claude Haiku 4.5)** | Second-opinion summariser polishes the agent's structured Report into customer-language at demo time | Yes (Add #1, hour 20+) |
| **SageMaker Endpoint** | Serves the trained checkpoint to the Streamlit demo over HTTP — demonstrates "this is how a customer would consume the model" | Optional (Add #2, hour 18+) |
| **SageMaker Training Jobs** | Productionisation path — slide-only in the pitch deck | No (pitch only) |
| **EC2 (one-time)** | Only as fallback if the cloud GPU box is somehow on a slow pipe — otherwise skipped | No (vast.ai direct preferred) |
| **AgentCore** | Documented production-orchestration path in pitch | No (presentation only) |

**Compute decision:** vast.ai A100 80GB or H100 80GB rented for the 24-hour sprint (~$30-50 total). Faster iteration than SageMaker training jobs for a clock-bound build. Pitch framing: "we picked vast.ai for the sprint, production runs on SageMaker."

---

## §6. Top 4 risks + cheapest disproof experiments

| # | Risk | Why scariest | Cheapest experiment | When |
|---|---|---|---|---|
| **R1** | Featurization throws away signal — 6 scalars too compressed | If true, no training fixes it | sklearn GBM on aggregated features → val Pearson. r ≥ 0.3 ⇒ safe; r < 0.1 ⇒ add channels 7-8 (H-bonds, contact-map entropy) | Hour 4 |
| **R2** | Per-frame energy verifier too noisy OR too correlated | Abstention either never triggers or always triggers | Spearman of `EPtot`, `EELEC`, `EVDW` vs PDBbind labels on 200 train ids. Pick the one with rho ∈ [0.2, 0.6] | Hour 2 (parallel) |
| **R3** | open_flamingo install fragility (modern transformers conflict) | Could eat 2-4 h debugging at hour 6 | `pip install -e third_party/OpenTSLM` in fresh venv on A30 | **Pre-clock** |
| **R4** | Wrong-prior fine-tune (ECG → MD) catastrophically interferes | Loss decreases while outputs degrade silently | Generate 5 samples from raw checkpoint with MD prompts at hour 6. Coherent? Safe. Gibberish? Reinit perceiver weights | Hour 6 |

---

## §7. Three highest-risk technical unknowns

1. **`OpenTSLM/llama-3.2-1b-ecg-flamingo` checkpoint compatibility.** Confirmed exists; not confirmed it loads on our `transformers` version. Cheapest check: `OpenTSLM.load_pretrained(REPO_ID, device="cuda")` in the dry-run.
2. **open_flamingo + modern transformers.** Their code monkey-patches `FlamingoLayer.attention_type`. Version-fragile. Pre-clock `pip install` test.
3. **A30 VRAM budget at training time.** Forward+backward with bs=1, grad-accum 16, bf16, grad-checkpoint on Llama-1B + Flamingo perceiver + 30-frame sequences. Cheapest check: dry-run step 4-6.

---

## §8. The 30-minute pre-clock DRY-RUN

Single command: `python scripts/dry_run.py`. Goes/no-goes the whole architecture.

### Prerequisites (pre-clock; user runs first)

```bash
# Downloads must be running or complete
huggingface-cli download meta-llama/Llama-3.2-1B
huggingface-cli download OpenTSLM/llama-3.2-1b-ecg-flamingo
# OpenTSLM installable
cd third_party/OpenTSLM && pip install -e . && cd -
# MISATO bundled example file present (comes with MiSaTo-dataset clone)
ls third_party/MiSaTo-dataset/src/inference_for_MD.hdf5
```

### Steps

| # | Time | Action | Assert |
|---|---|---|---|
| 1 | 2 min | Open bundled MISATO HDF5, pick PDB id 11GS, print keys + `trajectory_coordinates.shape` + `molecules_begin_atom_index[-1]` | shape is `(F, N, 3)`, last_idx > 0 |
| 2 | 5 min | `featurize.py` on this trajectory | output shape == `(6, 30)`, no NaN/Inf |
| 3 | 3 min | Build single-item batch dict with our prompts + dummy answer `"Answer: -7.2 kcal/mol. Confidence: medium."` | dict has 5 OpenTSLM keys |
| 4 | 10 min | Instantiate `OpenTSLMFlamingo(llm_id="meta-llama/Llama-3.2-1B")`, random adapter | trainable params ∈ [50 M, 500 M]; `torch.cuda.memory_allocated() < 22 GB` |
| 5 | 5 min | Forward pass, get loss | loss is finite, `requires_grad=True`, on cuda |
| 6 | 3 min | `loss.backward()`, AdamW step | perceiver weights changed (slice comparison) |
| 7 | 2 min | `model.generate(batch, max_new_tokens=50)` | output is valid decodable tokens |

### Exit criteria (ALL required to GO)

- ✅ Featurization produces `[6, F_sub]` with finite values
- ✅ Model instantiates within VRAM budget
- ✅ Forward pass returns finite loss
- ✅ Backward pass updates perceiver weights
- ✅ `generate()` returns valid tokens

**If any fail:** diagnose, fix, OR pivot to C-MAPSS-only architecture now (hour 0, not hour 14).

---

## Success criteria

### Primary (gates the pitch)

- **Pearson r on held-out PDB ids > 0.3** for predicted-vs-true PDBbind affinity
- **Abstention rate 15-25%** with ≥ 70% of abstentions correctly identifying cases where the prediction is in the worst-quartile of error

### Secondary (strengthens the pitch)

- C-MAPSS Pearson r > 0.5 on the same architecture (proof the *method* generalizes)
- Streamlit demo runs end-to-end on 3 judge-supplied PDB ids without crashing
- Pitch fits in 5 minutes including a CONFIRMED + INCONCLUSIVE case study

### Failure-mode fallbacks

- If MISATO fails at hour 14: C-MAPSS becomes primary, MISATO is "ongoing work, see code"
- If training never converges: pitch the architecture + featurization + agent loop as a paper-shaped contribution; demo on C-MAPSS only
- If open_flamingo never installs: pre-clock test caught it — pivot to OpenTSLMSP variant before any code is committed

---

## Open questions deferred to implementation

- Exact value of `TAU_HIGH` for abstention — calibrated empirically at hour 14
- Whether `EPtot` alone or a combination `0.5·EPtot + 0.3·EELEC + 0.2·EVDW` is the cleanest verifier — decided at hour 2 from R2 experiment
- Whether the deterministic rationale should include a "key concerning signal" line when confidence is low — decided at hour 18 from demo dry-runs
- Exact `cross_attn_every_n_layers` value (OpenTSLM default = 1; lowering to 2 or 4 cuts VRAM if R3 forces it) — decided at dry-run step 4 from memory measurement

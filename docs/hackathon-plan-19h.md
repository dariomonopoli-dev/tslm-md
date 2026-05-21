# TSLM-MD — 19-Hour Execution Plan

> Written at T-19h. Updated as gates are hit.

## Team & roles

| Person | Role | Owns |
|---|---|---|
| **A — "Trainer"** (Dario, on noctua A30) | GPU + training + ckpts | `tslm_md/train_stage6.py`, GPU job, checkpoints, hour gates |
| **B — "Data"** | Featurization + baselines | `scripts/preprocess_features.py`, `scripts/train_gbm_baseline.py`, `data/targets.json` sanity |
| **C — "Demo / agent integrator"** | Streamlit + agent wiring | `demo/app.py`, `tslm_md/agent.py` integration tests |
| **D — "Pitch + AWS"** | Slides + Bedrock + backup video | `docs/pitch/`, AWS Bedrock, final rehearsal |

If you only have 3 people: C and D merge (Demo + Pitch).
If you only have 2 people: A handles training only; the other handles B + C + D in parallel.

## Communication protocol

- **Telegram group** — all chatter
- **GitHub repo** — all code, commit + push frequently (no PRs needed for hackathon speed)
- **Every 2 hours each person posts:** *currently doing X, next is Y, blocker = none|...*

---

## Hour-by-hour timeline

### **Hour 0 (now → +50 min): MD.hdf5 download, parallel setup**

| Person | Task |
|---|---|
| A | Watch MD.hdf5 download. Run `ls -lh data/misato/MD.hdf5` every ~10 min. DO NOT keep retrying dry-run — h5py refuses truncated files. |
| B | Verify label distribution: `python -c "import json; d=json.load(open('data/targets.json')); vals=sorted(v['affinity_kcal_mol'] for v in d.values()); print(f'min={vals[0]:.2f} p10={vals[len(vals)//10]:.2f} median={vals[len(vals)//2]:.2f} p90={vals[len(vals)*9//10]:.2f} max={vals[-1]:.2f}')"`. Median should be ~-6 to -8 kcal/mol. |
| C | `git clone` repo locally + `pip install streamlit torch numpy`. Run `streamlit run demo/app.py` — mock data loads. Iterate on layout/visual polish. Push improvements. |
| D | Open AWS Workshop Studio. Enable Claude Haiku 4.5 in Bedrock console (us-east-1). Draft pitch slides 1-3: Problem, Architecture, Agent loop. |

### **Hour 1 (T+50min → T+1h30min): MD.hdf5 lands → dry-run**

| Person | Task |
|---|---|
| A | When `MD.hdf5 = 132G`: run `python scripts/dry_run.py --misato-h5 data/misato/MD.hdf5 --pdb-id 5WIJ`. Report PASS / FAIL with output. If PASS: kick off preprocessing in background: `nohup python scripts/preprocess_features.py --misato-h5 data/misato/MD.hdf5 --limit 2000 > preprocess.log 2>&1 &` |
| B | Once `featurized.h5` exists (~20-30 min after preprocess starts): `python scripts/train_gbm_baseline.py`. **This is the R1 gate.** Report Pearson r. |
| C | Continue demo polish. Add a "loading…" state so the UI works during model inference. |
| D | Pitch slides 4-5: Why OpenTSLM (judge's own model) + Why Chronos (the encoder the OpenTSLM team recommended). |

### **Hour 2 (T+1h30min → T+3h): wiring stage**

| Person | Task |
|---|---|
| A | Test loading juncliu checkpoint into the freshly-instantiated model: `python -c "import torch; from opentslm.model.llm.OpenTSLMFlamingo import OpenTSLMFlamingo; m = OpenTSLMFlamingo(device='cuda', llm_id='meta-llama/Llama-3.2-1B', encoder_type='chronos2'); s = torch.load('/home/user/dmonopoli/.cache/huggingface/hub/models--juncliu--llama-3.2-1b-ecg-flamingo-epoch-35/snapshots/cfcdf8f7141b729ae50da4e1ef4e3bdc2b638674/best_model.pt', map_location='cpu', weights_only=False); m.model.load_state_dict(s, strict=False); print('loaded')"` |
| B | If R1 PASS (r ≥ 0.3): green-light A to start training. If r < 0.1: flag for adding more channels. |
| C | Wire a stub `agent.run_agent()` call into Streamlit (replace `mock_report()`). Will be slow without trained checkpoint but the path is exercised. |
| D | Pitch slide 6: Eval methodology (Pearson r + abstention precision/recall). |

### **Hour 3-4 (T+3h → T+5h): WIRING GATE — overfit single batch**

| Person | Task |
|---|---|
| A | **Critical:** Overfit a single batch to near-zero loss. Use `tslm_md/train_stage6.py` config with `max_steps: 200`. If loss < 0.05 at step 200 → architecture is sound. If not → debug 1-2 h max, then fall back to `--encoder-type cnn`. |
| B | Prepare `configs/stage6_md_cot.yaml` for the real training run: `batch_size: 1`, `grad_accum_steps: 16`, `max_steps: 3000`, `ckpt_every_steps: 500`. |
| C | Test demo with the half-overfit checkpoint — should produce semi-sensible outputs. |
| D | Pitch slides 7-8: Business case (CHF 2-3B/drug, MD signal discarded) + AWS architecture diagram. |

### **Hour 4-12 (T+5h → T+13h): TRAINING (8 hours)**

| Person | Task |
|---|---|
| A | Launch real training: `python tslm_md/train_stage6.py --config configs/stage6_md_cot.yaml --starting-checkpoint <juncliu_path>/best_model.pt`. **Watch the loss curve.** Save ckpt every 500 steps. Sleep / stay productive — don't babysit. |
| B | Build the evaluation script. Test it against random predictions first. Confirm Pearson computation is correct. |
| C | Wire the *latest* checkpoint into the demo every 30 min: pull, restart Streamlit, verify outputs improve. |
| D | Pitch slides 9-10: Future work + closing. Run through full deck twice for timing (target 4-5 min). |

### **Hour 12 (T+13h): CONVERGENCE GATE**

| | |
|---|---|
| If val loss decreasing AND val Pearson > 0.15 | ✅ **Continue training another 2 h, then stop and evaluate**. Proceed to hour 14 plan. |
| Loss flat or val Pearson ≈ 0 | ❌ **STOP**. Switch demo to use the pretrained juncliu checkpoint with our prompts (no fine-tune). Pitch the architecture, not the convergence. |
| NaN / diverging | ❌ Reduce LR 5×, restart from juncliu checkpoint, lose ~30 min. If still diverging by hour 13 → fall back as above. |

### **Hour 14-17 (T+15h → T+18h): final eval + polish**

| Person | Task |
|---|---|
| A | Run final eval on held-out test set. Save numbers: Pearson r, MAE, abstention rate, abstention precision/recall. |
| B | Pick the **CONFIRMED case study** (low disagreement, accurate prediction) and the **INCONCLUSIVE case study** (high disagreement, ambiguous binding) — these are the demo highlights. |
| C | Final demo testing on 5 unseen PDB ids. Fix any UI bugs. Confirm the abstention badge shows correctly. |
| D | Record a backup demo video (~3 min) in case live demo breaks. Insert eval numbers + case studies into slides. |

### **Hour 17-19 (T+18h → T+19h): rehearsal**

- **All four:** rehearse the pitch end-to-end ≥ 2 times. Aim for 4:30 spoken (so you have 30 sec slack on a 5-min pitch).
- Run order: Problem (D) → Architecture (D) → Live demo (C) → Results (D) → AWS productionization (D) → Vision (D)
- One person mans the demo laptop, one person speaks, two people answer Q&A.

---

## Critical decision points

| Hour | Gate | Pass criterion | Fail action |
|---|---|---|---|
| 2 | Dry-run + R1 baseline | dry-run ✅ + GBM Pearson r ≥ 0.3 | r < 0.1: add ch7=H-bonds, ch8=contact entropy; lose 1 h |
| 4 | Wiring | Single batch overfits to loss < 0.05 | Try `--encoder-type cnn`; if still broken: pitch C-MAPSS instead |
| 12 | Convergence | Val Pearson > 0.15 AND loss decreasing | Stop, use juncliu pretrained ckpt + our prompts |
| 16 | Backup video | Live demo works on 3 unseen ids | Record video, plan to play it if live fails |

---

## What the demo MUST do (the moment of truth)

**Story arc:** *"A computational chemist pastes a PDB id and gets back a verified affinity prediction in seconds."*

1. **Sidebar:** PDB id input + Analyse button (already in `demo/app.py`)
2. **Top row:** 6 sparkline charts of the feature trajectories
3. **Center:** Predicted affinity (kcal/mol) + confidence badge + deterministic rationale paragraph
4. **Right:** Verifier comparison: predicted vs independent physics + CONFIRMED / INCONCLUSIVE badge
5. **One CONFIRMED demo** and **one INCONCLUSIVE demo** are the highlight moments

---

## Pitch outline (5 slides, ~5 minutes)

| Slide | Owner | Content |
|---|---|---|
| 1. Hook + Problem | D | "Binding affinity is the drug-discovery bottleneck. Current ML throws away the MD trajectory — the very signal that distinguishes a stable bound pose from an unbinding event." |
| 2. Architecture | D | One diagram: PDB id → trajectory → 6-channel time-series → Chronos → Llama → answer + rationale + verifier → CONFIRMED/INCONCLUSIVE. "First TSLM applied to molecular dynamics." |
| 3. Live demo | C | Paste PDB id #1 → CONFIRMED. Paste PDB id #2 → INCONCLUSIVE with reason. "It knows when not to trust itself." |
| 4. Results + judge alignment | D | Pearson r table + abstention precision/recall. "We extended your own OpenTSLM with stage 6, used your team's recommended Chronos checkpoint, mirrored the SOC-agent precedent." |
| 5. AWS productionization + vision | D | "Compute on vast.ai for the sprint; production runs on SageMaker training jobs + SageMaker Endpoint + Bedrock for second-opinion summarization. Scales linearly per complex." |

---

## What we will say IF the live demo breaks

Have the backup video ready. **Don't apologize twice.** One smooth pivot:
> *"Live network is being a network — here's our recorded run on three judge-selected PDB ids from yesterday."*

Play 30 seconds of video, return to slides.

---

## What NOT to do in the next 19 hours

- ❌ Don't try to use the full 16k complex training set. 2000 is enough.
- ❌ Don't tune hyperparameters. Start with our config; only change if loss diverges.
- ❌ Don't add features beyond the spec. No bonus risk.
- ❌ Don't pitch the Pearson r number first — lead with the agent + abstention behavior.
- ❌ Don't reinstall anything that already works (Llama, Chronos, juncliu, OpenTSLM).
- ❌ Don't burn time getting AWS perfect — Bedrock summarizer is the only AWS-critical wiring.

---

## Tracking artifact

This file lives at `docs/hackathon-plan-19h.md`. Update the "status" column below as you hit gates.

| Hour | Gate | Owner | Status |
|---|---|---|---|
| 0 | MD.hdf5 done | A | ⏳ |
| 1 | Dry-run pass | A | ⏳ |
| 2 | R1 baseline pass | B | ⏳ |
| 4 | Wiring gate pass | A | ⏳ |
| 12 | Convergence gate | A | ⏳ |
| 16 | Demo works on 3 unseen ids | C | ⏳ |
| 17 | Backup video recorded | D | ⏳ |
| 19 | Pitch rehearsed end-to-end | All | ⏳ |

# Idea Evaluation — TSLM-MD

> Use this file to evaluate this idea against your team's three other ideas.
> A blank scoring template for the other ideas lives at `evaluation-template.md`.

---

## 1. The 30-second pitch

We extend OpenTSLM — the ETH Agentic Systems Lab's own flagship time-series LLM — to a brand-new modality: **molecular dynamics trajectories**. We fine-tune a small frozen LLM (Llama-3.2-1B) with a trainable Flamingo-style cross-modal adapter so it can reason over MD trajectories and predict protein-ligand **binding affinity** — the central quantitative endpoint of structure-based drug discovery. We wrap the trained model in an agent that verifies its prediction against MISATO's independent per-frame physics-based energies and **abstains ("INCONCLUSIVE") on disagreement**, mirroring the SOC-agent precedent that won the previous track.

**One sentence:** *The first TSLM that reasons over molecules, with a verifier-driven agent that knows when not to trust itself.*

---

## 2. What we'd actually train (the artifact)

| | |
|---|---|
| **Artifact** | `tslm_md_flamingo_llama1b_stage6.pt` — a fine-tuned OpenTSLMFlamingo checkpoint |
| **Trainable params** | ~50-200 M (perceiver + gated cross-attention + CNN time-series encoder + LM input embeddings) |
| **Frozen** | All ~1 B params of Llama-3.2-1B |
| **Starting point** | `OpenTSLM/llama-3.2-1b-ecg-flamingo` (their stage-5 ECG-CoT checkpoint), added as new sixth curriculum stage |
| **Input** | `[6 features × ~30 frames]` tensor per protein-ligand complex |
| **Output** | `"Answer: -8.4 kcal/mol. Confidence: high."` (parsed by regex) |
| **Eval metric** | Pearson r between predicted and PDBbind ground-truth affinity on held-out PDB ids |

**This is not an LLM wrapper.** Without the trained adapter, the same prompts to the same Llama produce noise. The artifact is the modality bridge.

---

## 3. What makes it agentic (the loop)

```
agent(pdb_id):
    1. Retrieve trajectory + features for the complex
    2. Call trained TSLM-MD → predicted affinity + confidence
    3. Call deterministic rationale generator → grounded language explanation
    4. Call verifier → independent physics-based energy estimate
    5. Compare → if z-score disagreement above threshold → INCONCLUSIVE
    6. Emit structured Report
```

- **Multi-step + tool-use:** trained TSLM is one tool, verifier is another, rationale generator is a third — all independent
- **Graceful failure:** abstention triggers on real disagreement, not a fake confidence score
- **No LLM-on-LLM hallucination:** verifier is deterministic physics, rationale sentences are computed from numbers in the trajectory
- **Evaluation built in:** abstention rate is itself a measurable demo metric

---

## 4. Mapping to the hackathon's hard requirements

| Requirement | How this idea satisfies it |
|---|---|
| **NOT an "LLM wrapper" — must have a real trained model** | ✅ ~50-200 M trainable params in the cross-modal adapter, fine-tuned on MISATO. Verifiable artifact on disk. |
| **Novel infrastructure, hard to copy, "not seen before"** | ✅ First application of TSLM to molecular dynamics. OpenTSLM has only been used on 1-D medical signals (ECG/EEG/accel). The feature pipeline + stage-6 curriculum + agent verifier is the novel infra. |
| **Open-source dataset, used responsibly** | ✅ MISATO (Zenodo 7711953, Nature Computational Science 2024) + PDBbind v2020 index. Both fully open, properly cited. We use the data as intended (binding-affinity prediction is its primary purpose). |
| **Clear business impact + target user** | ✅ Target user: computational chemists at biotech/pharma. Problem: static-structure affinity models throw away the dynamic signal in MD trajectories that the chemists already ran. We use it. Reduces wet-lab false positives → cheaper lead optimization. |
| **Agentic engineering with real scalability** | ✅ Synchronous Python agent loop with SageMaker-ready training, S3-backed storage, AgentCore documented as the production path. Pipeline is repeatable across complexes, not hand-crafted. |
| **Deliberate AWS integration** | ✅ S3 for MISATO data + checkpoints (critical path). EC2 for the one-time data prefetch. SageMaker as training contingency. Bedrock Claude as optional second-opinion summarizer. AgentCore as documented productionization path. |

---

## 5. Resonance with the judges (qualitative)

| Judge signal | This idea |
|---|---|
| Judges' flagship work is **OpenTSLM** | We literally extend it. Framing: "we built stage 6 of your curriculum." |
| Judges' identity: "design, develop, and **EVALUATE** agentic systems" | Pearson r + abstention rate + CONFIRMED/INCONCLUSIVE case studies are real eval, not vibes |
| Previous winner: **SOC agent with INCONCLUSIVE verdict** | Direct structural parallel — our verifier-driven abstention is the same pattern in a new domain |
| **AWS sponsorship** wants serious infra use | S3 + EC2 + SageMaker + Bedrock + AgentCore all have a clear role |

This idea has the strongest possible resonance with the judges *if* it works.

---

## 6. Business case

| | |
|---|---|
| **Who pays** | Mid-to-large biotechs and pharma R&D departments running structure-based drug design pipelines |
| **What's broken** | They run expensive MD simulations (hours-to-days of GPU time per complex) and then collapse the trajectory back to a single docking score, discarding the dynamic signal. Static-structure ML models (Pafnucy, OnionNet, etc.) are state-of-the-art but plateau because they can't see motion. |
| **Why now** | MISATO (2024) is the first large open MD-with-affinity dataset. OpenTSLM (2024-2025) is the first time-series-native LLM. The intersection didn't exist 12 months ago. |
| **Realistic deployment** | Plugin into existing MD pipelines (GROMACS/AMBER output → our agent). Inference cost is negligible (~seconds per complex on commodity GPU once trained). |
| **Why the agent matters commercially** | Pharma reviewers do NOT trust black-box predictions. An agent that says "INCONCLUSIVE — model and physics disagree" is *more* useful to a chemist than one that confidently outputs the wrong number. |

---

## 7. Implementation plan (compressed)

### Pre-clock (tonight)
- EC2 → S3 prefetch MISATO (133 GiB → ~30 min on AWS pipe vs hours on home Wi-Fi)
- HuggingFace download Llama-3.2-1B + OpenTSLM stage-5 checkpoint
- `pip install -e` OpenTSLM in fresh venv on A30 (the most informative pre-clock check — catches dep hell now, not at hour 6)

### Official 24 h

| Hour | Output | Gate |
|---|---|---|
| 0-1 | Featurizer on one trajectory | shape `[6, 30]`, no NaN |
| 1-2 | Verifier sanity (Spearman of energy components vs PDBbind on 200 ids) | rho ∈ [0.2, 0.6] |
| 2-4 | Dataloader + GBM baseline (R1 disproof) | GBM Pearson r ≥ 0.3 |
| 4-6 | Training targets + stage-6 wiring | 20 spot-checks pass |
| 6-8 | **Wiring gate**: overfit single batch | loss < 0.05 |
| 6-8 (parallel) | C-MAPSS insurance training | val MAE < 25 cycles |
| 8-14 | Stage-6 fine-tune on 2 k complexes | **convergence gate** — val Pearson > 0.15 at hour 14 |
| 8-14 (parallel) | Agent + verifier + parser + eval scripts | end-to-end pipeline runs |
| 14-18 | Continue training (or pivot to C-MAPSS if gate failed) | held-out Pearson > 0.3 |
| 18-22 | Streamlit demo | works on 3 unseen PDB ids |
| 22-24 | Pitch deck + dry-run | ≤ 5 min pitch |

**Full hour-by-hour plan + risk table:** see `docs/superpowers/specs/2026-05-21-tslm-md-design.md`.

---

## 8. Risks (honest)

| Risk | Mitigation | Residual P(failure) |
|---|---|---|
| MD.hdf5 (133 GiB) download blows the budget | EC2→S3 prefetch tonight | ~5% if prefetch done |
| `open_flamingo` dep hell vs modern transformers | Pre-clock install test on A30 | ~10% (catch early or pivot to SP variant) |
| Stage-6 fine-tune doesn't converge in 6 h | C-MAPSS architecture insurance running in parallel | ~30% for MISATO Pearson > 0.3; ~5% for *some* working demo |
| Trained model's natural-language outputs look template-y | We sidestep this — deterministic rationale generator is auditable, model only emits the number | low |
| Pearson r underwhelming vs sklearn baseline | Pitch leans on agent loop + abstention + first-TSLM-on-molecules, not on beating SOTA | low (story holds even at modest r) |

**Honest feasibility estimate: 75-85% chance of a working end-to-end demo; 50-60% chance of Pearson r > 0.3 on MISATO; ~99% chance of *some* compelling pitch (worst case = C-MAPSS demo + MISATO-as-ongoing-work).**

---

## 9. Self-score (1-5 per axis — apply same rubric to other 3 ideas)

| Criterion | Score | Justification |
|---|---|---|
| Trained model artifact (not wrapper) | **5** | Real adapter with ~100 M trainable params, fine-tuned from a real pretrained checkpoint |
| Novelty of infrastructure | **5** | First TSLM applied to molecules; novel featurization pipeline + agent verifier |
| Open-source dataset, responsible use | **5** | MISATO + PDBbind, properly cited, used as intended |
| Business impact + clear user | **4** | Strong drug-discovery story; not "we'll change healthcare" hype, but a real pipeline problem |
| Real engineering / scalability | **4** | Pipeline is reproducible across complexes; depends on training actually converging |
| AWS integration depth | **4** | S3 + EC2 + optional SageMaker + Bedrock + AgentCore documented; not all on critical path |
| Agentic behavior (multi-step + abstention) | **5** | Verifier-driven INCONCLUSIVE is the SOC-agent precedent in a new domain |
| Resonance with judges | **5** | We extend their own flagship work; we mirror the prior winner's pattern |
| 24 h feasibility | **3** | Borderline. Achievable with the simplifications in §7-§8, but ML training in 24 h always carries variance |
| Demo wow-factor | **4** | "Paste a PDB id, watch the agent reason and abstain" is visually crisp; would be 5 if we had a 3D viewer (out of scope) |
| **TOTAL (out of 50)** | **44** | |

---

## 10. What would make us pick THIS over another idea

Pick this if:
- The team has at least one person comfortable with PyTorch fine-tuning + HF Transformers
- We're willing to accept ~25% probability the MISATO training underperforms (insured by C-MAPSS)
- We want the pitch to land most strongly with *these specific judges*

Don't pick this if:
- No one on the team has shipped a fine-tuning run before (the wiring is the riskiest part)
- We want a demo that's pure-frontend / pure-API (this is back-end heavy)
- We have a different idea that more naturally lives in pure-AWS-managed services

---

## 11. Open questions for the team

- Who on the team has actually loaded an HF model + trained it to convergence before? (gates feasibility)
- Do we want to commit to a Streamlit demo or have someone capable of a richer UI?
- Are we willing to spend $5-20 of AWS credit pre-clock on the EC2 prefetch + (optionally) a SageMaker training fallback?
- Do we want the pitch to lean on "we extended their work" (safe, judge-flattering) or "we built something they hadn't thought of" (riskier, higher ceiling)?

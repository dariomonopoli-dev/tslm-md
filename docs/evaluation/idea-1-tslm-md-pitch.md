# Project Summary: "TSLM-MD" — Agentic Binding-Affinity Copilot for Drug Discovery
*(working name — swap freely)*

## One-line pitch
An agentic copilot that ingests a protein-ligand molecular-dynamics trajectory, predicts binding affinity using a time-series language model we fine-tune ourselves from OpenTSLM, generates a grounded natural-language explanation of the binding behaviour, verifies its own prediction against independent physics-based energy components — and abstains with "INCONCLUSIVE" instead of guessing when prediction and physics disagree.

## The problem (with evidence)
Bringing a new drug to market costs roughly USD 2-3 billion and takes 10-15 years; over 90% of clinical candidates fail. A meaningful chunk of that risk is set in the preclinical *lead-optimisation* phase, where chemists pick which molecules to take forward based on **binding affinity** — how strongly a candidate holds onto its target protein. State-of-the-art ML predictors (Pafnucy, OnionNet, equivariant GNNs, AlphaFold-derived scorers) all collapse a complex to a **single static 3D snapshot** and discard the molecular-dynamics trajectory chemists *already ran*. That trajectory is the exact signal that distinguishes a stably bound pose from a fluttering, unbinding one. The gap: **no model reasons over the time-series of how a complex actually moves.**

## Target user & buyer
**Primary user:** computational chemist / molecular modeller on a medicinal-chemistry team at a biotech or pharma. **Buyer / GTM:** head of computational chemistry or VP of drug discovery — concentrated, paying audience (~200-500 mid-to-large pharma R&D groups + ~2 000 biotechs globally). **Distribution wedge:** drop-in plugin to existing MD pipelines (GROMACS / AMBER / OpenMM). The trajectories already exist as a by-product of standard SBDD workflows; we just read them.

## What we build (the core)
A code-orchestrated single agent (deterministic Python loop, not model-driven flow):
upload PDB id → fetch trajectory + per-frame energies from S3 → featurize each frame to a small multivariate vector → classify (fine-tuned **OpenTSLMFlamingo** on top of Llama-3.2-1B) → explain (deterministic grounded summariser reading the same feature tensor) → verify (independent per-frame energy components from MISATO) → abstain-or-answer (z-score disagreement gate) → cited verdict / escalate to human.

## The novelty (our differentiator)
**First time-series language model applied to molecules.** OpenTSLM has only ever been applied to 1-D medical signals (ECG, EEG, accelerometry). We extend it with a new sixth curriculum stage — *molecular dynamics* — and add a verifier-driven abstention layer grounded in independent physics, not in another LLM's opinion.

Tagline: **"Static models guess. Dynamic models reason. Ours knows when not to trust itself."**

## Why OpenTSLM (the moat)
OpenTSLM is the **ETH Agentic Systems Lab's own open-source flagship** — the only architecture explicitly designed to treat time-series as a *native* LLM modality (Flamingo-style gated cross-attention + perceiver + trainable CNN tokenizer). No US foundation model handles continuous time-series this way; the alternative ("dump numbers as text into the prompt") is provably worse on every published benchmark. Starting from OpenTSLM's pretrained ECG-CoT checkpoint and adding stage 6 means we inherit a model that already knows how to interpret time-series tokens — a capability competitors structurally cannot replicate without first rebuilding the cross-modal adapter from scratch. And **it is the judges' own model**.

## How it maps to the judging criteria

| Criterion | How we hit it |
|---|---|
| **Clear problem + target user** | Computational chemists at biotech/pharma; problem cited with industry numbers ($2-3B/drug, 90% failure rate, MD signal currently discarded) |
| **Meaningful fine-tune / RAG / agentic** | Direct fine-tune of OpenTSLMFlamingo (~50-200 M trainable params: perceiver + gated cross-attention + CNN encoder + LM input embeddings) + orchestrated multi-tool agent — two of three "excellence verbs" |
| **Responsible open data** | MISATO (Apache-2.0, Zenodo 7711953, Nature Computational Science 2024) + PDBbind v2020 — both cited, properly attributed, used for their intended purpose |
| **Working demo** | Deterministic agent loop + verifier + rationale + Streamlit runs on the pretrained OpenTSLM checkpoint from hour 6; fine-tune is the upgrade, never on critical path |
| **Technical creativity + feasibility** | Verifier-driven abstention + deterministic grounded rationale (novel, auditable, no LLM-on-LLM hallucination); first TSLM on molecules; ETH's own model on a new modality |
| **Evaluation / evidence** | Held-out Pearson r vs PDBbind ground-truth + abstention precision/recall + 1 CONFIRMED + 1 INCONCLUSIVE case study → a number, a chart, two stories |
| **AWS used + scale** | S3 (data + checkpoints) → EC2 (one-time prefetch) → SageMaker (training fallback) → Bedrock (Claude as optional second-opinion summariser); per-complex inference scales linearly |

## AWS lifecycle (concrete)
S3 stores the raw MISATO MD/QM HDF5 files + the featurized tensor cache + all checkpoints → EC2 used once pre-clock to pull the 133 GiB Zenodo dataset onto AWS's gigabit pipe and push to S3 (~30 min vs hours on home Wi-Fi) → SageMaker available as spot-train fallback if the local A30 OOMs → fine-tuned model served from S3 to the A30 (or as a SageMaker endpoint) → Bedrock runs Claude as an optional second-opinion summariser at demo time → AgentCore documented as the productionisation path in the pitch deck.

## Tech stack
OpenTSLMFlamingo (Llama-3.2-1B backbone, frozen) + CNNTokenizer + Perceiver + gated cross-attention via `open_flamingo` + HuggingFace `transformers` + h5py + MISATO HDF5 + PDBbind v2020 index + Kabsch alignment for RMSD + sklearn GradientBoostingRegressor for the baseline-disproof experiment + Streamlit web UI with a live "feature sparklines + agent reasoning + verifier verdict + confidence" panel.

## Demo script (the memorable moment)
A computational chemist pastes a real PDB id (say **1A4K**, a kinase inhibitor) → the agent fetches the trajectory and streams the six per-frame feature traces as small sparklines → emits *"Min pocket distance tightened from 3.8 Å to 2.9 Å with a plateau at frame 12. Ligand RMSD stable at 1.2 Å. Contact count saturated at 14. **Answer: -8.4 kcal/mol. Confidence: high.**"* → verifier panel shows independent physics energy agrees → green **CONFIRMED** badge → switch to a deliberately hard PDB id where TSLM and physics disagree → red **INCONCLUSIVE** badge with *"Routed to human reviewer."* → closing slide: *"Held-out Pearson r = X.XX on unseen complexes; abstention catches Y% of worst-quartile errors."*

## Risk register
- **#1 risk:** 133 GiB MISATO download blows the time budget → mitigate via the **EC2 → S3 prefetch tonight**, pre-clock.
- **#2 risk:** `open_flamingo` dep hell vs modern `transformers` (their code monkey-patches `FlamingoLayer.attention_type`) → mitigate via **pre-clock `pip install -e .` test on the A30** — the single most informative thing we can do tonight.
- **#3 risk:** Stage-6 fine-tune fails to converge in 6 h on a wrong-prior (ECG → MD) checkpoint → mitigate with **NASA C-MAPSS architecture-insurance training running in parallel from hour 6**; the demo pipeline runs on the pretrained checkpoint regardless of fine-tune outcome.

## Effort / 24 h feasibility
**Medium-high.** Featurization + dataset wiring + agent loop + verifier + parser + Streamlit demo = the bulk (one focused engineer). Stage-6 fine-tune runs in parallel (one ML person). Abstention layer is pure logic, no infra. C-MAPSS insurance is an isolated script. Achievable with a 3-4 person team **if pre-clock prep (data prefetch + HF download + dep test) happens tonight.**

## What would make us LOSE
- Spending hours 6-10 fighting `open_flamingo` install issues because we didn't test it pre-clock.
- Letting the 133 GiB MISATO download eat hours 0-6 because we didn't pre-fetch via EC2.
- Pitching the Pearson r number *first* instead of the **abstention behaviour** — the judges already know ML on novel domains in 6 h is mediocre; what they care about is graceful failure.
- Building five features at 70 % instead of three at 100 %.
- A flat pitch that doesn't show the live **INCONCLUSIVE** moment with a real chemist's question.

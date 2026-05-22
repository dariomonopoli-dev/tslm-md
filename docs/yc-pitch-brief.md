# TSLM-MD — Y-Combinator-Style Pitch Brief

> One-pager for slide drafting. Copy any section into the deck verbatim.

---

## One-line description

> *We turn the molecular-dynamics simulations pharmas already pay for into binding-affinity predictions a chemist can trust — by teaching a language model to read motion as time-series.*

---

## The hook (open with this)

> Every new drug costs $2-3 billion and takes 10-15 years. 90% of candidates fail. The single biggest preclinical bet — *which molecules to move forward* — comes down to one number: **binding affinity**. And every state-of-the-art model predicting that number throws away the most informative data you already have.

---

## The problem

Computational chemists run **molecular-dynamics (MD) simulations** — frame-by-frame physics videos of how a drug candidate moves while bound to its target protein. Hours-to-days of GPU time per complex. Critical signal: *does the ligand stay put, or does it wander off?*

But the entire SOTA — Pafnucy, OnionNet, equivariant GNNs, AlphaFold-derived scorers — collapses the trajectory into **one static snapshot** before predicting. Throwing away the time dimension is throwing away the answer.

**The MD signal that distinguishes a stable bound pose from an unbinding event is currently discarded.**

---

## The solution

**TSLM-MD** is the first Time-Series Language Model applied to molecular dynamics.

We featurize each MD trajectory into a small multivariate time-series (6 physically-meaningful channels × 30 frames), feed it to Amazon's **Chronos-2** time-series foundation model, and let a fine-tuned **Llama-3.2-1B with a Flamingo-style cross-modal adapter** reason over the trajectory and predict binding affinity.

Then we wrap it in an **agent** that:
1. Generates the prediction with the trained model
2. Generates a deterministic, fully-auditable rationale from the same trajectory data
3. **Independently verifies** the prediction against MISATO's per-frame physics energies (which the model never saw at training)
4. **Abstains** with "INCONCLUSIVE" if model and physics disagree

> *Static models guess. Dynamic models reason. Ours knows when not to trust itself.*

---

## Who we serve

| | |
|---|---|
| **Primary user** | Computational chemists in medicinal-chemistry teams |
| **Buyer** | Head of Computational Chemistry / VP of Drug Discovery |
| **Where it lives** | Plug-in to existing MD pipelines (GROMACS / AMBER / OpenMM) — they already produce the trajectories |
| **Trigger to adopt** | Wet-lab waste: 4 out of 5 lab-tested compounds fail. Cutting that by 25% via better in-silico triage is the prize. |

---

## Market

- **~200-500** mid-to-large pharma R&D groups globally
- **~2,000** biotechs
- Drug-discovery software market: **$5 B today, growing 11% CAGR**
- Per-customer ACV: $50K-$500K depending on team size and integration depth
- TAM if pricing scales with simulation-hours saved: **$1-3 B addressable**

Bottom-up signal: structure-based drug discovery teams currently burn **10,000-100,000 GPU-hours per project** on MD. Cost per project: ~$50K-$500K in compute alone, before scientist time.

---

## Why now

Three things turned this from "interesting" to "buildable" in the last 18 months:

1. **MISATO** (Nature Computational Science, 2024) — the first large open MD-with-affinity dataset. 16,972 protein-ligand complexes with full trajectories + per-frame energies. Didn't exist 12 months ago.
2. **OpenTSLM** (Stanford BDHG + ETH, 2024-2025) — the first time-series-native LLM. Lets us treat trajectories as a first-class modality.
3. **Chronos-2** (Amazon, 2025) — a pretrained time-series foundation model. Removes the need to train a from-scratch encoder.

These three pieces only became combinable in 2025. We're the first to combine them.

---

## Secret sauce (the moat)

1. **First TSLM applied to molecules.** OpenTSLM has only been applied to ECG, EEG, accelerometry — 1-D medical signals. Bringing it to molecular dynamics is a one-shot first-mover.
2. **The verifier-driven abstention layer.** Other ML agents in this space confidently predict and sometimes hallucinate. Ours measures its own confidence by comparing to *independent physics* it never trained on, and routes uncertain cases to humans. Pharma trusts a model that says *"I'm not sure"* far more than one that's always certain.
3. **Deterministic grounded rationale.** Every sentence the user sees about a prediction references a real number from the actual trajectory. Zero hallucination surface. Auditable. Defensible for regulatory submissions later.
4. **Endorsed by the OpenTSLM team itself.** Their lead authors confirmed the approach is novel and pointed us at the best checkpoint and fork. We're extending their work with their blessing.

---

## Competition

| Competitor | What they do | Why we win |
|---|---|---|
| **Pafnucy / OnionNet / equivariant GNN-based scorers** | Predict affinity from a single static structure | They discard the MD signal. We use it. |
| **AlphaFold-based binding predictors (e.g., AlphaFold3)** | Predict structure + binding from sequence | Different layer of the stack. We complement them — they pick a candidate pose, we score how stable it is over time. |
| **Hand-rolled physics scoring (MM/PBSA, FEP)** | Compute free energies from MD trajectories with classical mechanics | Slow (hours per complex), narrow domain. We're seconds per complex and generalise across target families. |
| **Generic LLM agents on chemistry data (ChemCrow, etc.)** | LLM with chemistry tools | They're LLM wrappers. We have a real trained model. |
| **In-house enterprise tools (Schrödinger, OpenEye)** | Closed, expensive, structure-only | We work on top of their pipelines, augment rather than replace |

**Critical:** no one is doing TSLM-on-MD. We checked.

---

## What we built in 24 hours (technical traction for the demo)

- **Trained-model artifact:** fine-tuned OpenTSLMFlamingo (Chronos-2 + Llama-3.2-1B) — ~50-200M trainable parameters
- **Built on the OpenTSLM team's recommended fork** (liu-jc/OpenTSLM, `add-chronos2-encoder` branch) with their recommended pretrained checkpoint (`juncliu/llama-3.2-1b-ecg-flamingo-epoch-35`)
- **MISATO dataset** (16,972 complexes), official train/val/test splits, kierandidi-affinity labels
- **6-feature MD featurization** computed per frame: min pocket distance, mean pocket distance, close-contact count, ligand RMSD, radius of gyration, interface buriedness
- **Synchronous agent loop** with deterministic grounded rationale + physics-based verifier + abstention
- **Live Streamlit demo** — paste a PDB id, watch the model reason
- **AWS-native production path:** S3 for data + checkpoints, Bedrock Claude for second-opinion summarisation, SageMaker Endpoints for serving

---

## Business model (if asked)

- **SaaS tier** for individual labs: $10K-$30K / seat / year
- **Enterprise tier** for pharma R&D: $200K-$500K / team / year, includes private model fine-tunes on the customer's proprietary trajectories
- **API tier** for biotech: $0.10 / complex scored, volume-discounted

Gross margin > 80% once trained — inference is seconds-per-complex on commodity GPU.

---

## Team

(Fill in your 4 teammates' names + 1-line backgrounds)

---

## The ask (closing slide)

> *We extended your own OpenTSLM into a brand-new modality, used your team's recommended checkpoint, and built an agent that mirrors the SOC-agent pattern that won the previous track. We'd love your vote — and your introduction to a computational-chemistry team at a pharma where this could land in week one.*

---

## Suggested names (Fiorenzo can pick / iterate)

| Name | Why |
|---|---|
| **Helix** | Time + structure, alpha-helix nod |
| **Tempo** | Time / rhythm — the trajectory dimension we use |
| **Flux** | Movement; reads as "in flux" — the binding event |
| **Trajeq** | Trajectory + equilibrium |
| **Atlas Bind** | Maps the dynamic binding landscape |
| **Molecule.run** | Runs the molecule's trajectory through the model |
| **PoseProof** | Proves a binding pose is stable |
| **In Silico Stable** | What chemists actually care about — "is the pose stable?" |
| **Kinopsis** | Greek "view of motion" |
| **PhaseSpace** | The physics term for the dynamic state space we live in |

Personal favourites: **Helix**, **PoseProof**, **Tempo**.

---

## TL;DR Fiorenzo can paste into the first slide

> **TSLM-MD: The first time-series language model for molecular dynamics.**
> *Pharmas already run hours of MD per drug candidate, then throw away the time dimension. We don't. We turn the trajectory into a time-series, feed it to a fine-tuned LLM, generate a grounded rationale, verify against independent physics, and abstain when uncertain. First-mover on a $5B-and-growing drug-discovery software market, built on the judges' own open-source flagship.*

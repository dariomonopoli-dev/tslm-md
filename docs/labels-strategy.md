# Rationale-labels strategy

> Direct answer to the OpenTSLM team's question: *"It probably comes down to how you can generate rational labels for the molecules."*

## The short version

**We do not generate fake rationales for training.** That sidesteps the labels problem entirely.

- **Training targets** are minimal: a single line per complex, `"Answer: -8.4 kcal/mol. Confidence: high."` (~15 tokens). The LLM learns to map from the Chronos-encoded trajectory tokens to the numeric answer + tier, nothing else.
- **Rationale at inference** comes from `tslm_md/rationale.py` — a deterministic function that reads the SAME [6, 30] feature tensor the model consumed and emits sentences whose every claim references a number actually in the trajectory. Example:
  > "Minimum pocket-ligand distance moved from 3.82 Å to 2.91 Å and is decreasing across frames. Ligand RMSD from frame 0 averages 1.18 Å and is stable — low and tight. Close-contact count peaks at 14 pairs and is stable. Ligand radius of gyration changed by -0.34 Å over the trajectory. Mean buriedness proxy: 6.7 buried ligand atoms. Overall these time-series signals are consistent with a stable bound pose."
- **Independent verifier** is the MISATO `frames_EPtot` (or a weighted combination of EPtot/EELEC/EVDW) — values the model NEVER saw at training. Compared via z-score against train-set distribution. Disagreement above threshold triggers `INCONCLUSIVE`.

Result: every piece of language the user sees is either (a) deterministic and auditable, or (b) routed through the abstain-and-escalate gate.

## Why this is honest

The alternatives all have known failure modes for a 24-hour hackathon budget:

| Alternative | Failure mode |
|---|---|
| **Templated CoT rationales** (e.g., "stable binding because RMSD is X") fed as training targets | Model memorises the template. Outputs look formulaic at inference. Judges notice. |
| **LLM-generated rationales** (Claude writes CoT given features → use as labels) | Circular: training one LLM on another LLM's hallucinations. Inherits Claude's mistakes. Costs hours of API time. |
| **Human-annotated rationales** | We do not have ~2-16 k expert-chemist annotations. |

Skipping rationales-as-targets gives us:
- A clean training signal (number + tier, both objectively verifiable from PDBbind)
- Zero hallucination surface at inference (rationale is deterministic)
- A pitch that survives scrutiny ("the model predicts; the rationale is grounded; the verifier abstains")

## What the OpenTSLM team gets

We're directly extending the work they pointed us at:
- Fork: `liu-jc/OpenTSLM` branch `add-chronos2-encoder`
- Encoder: `amazon/chronos-2` (their current best)
- Starting weights: `juncliu/llama-3.2-1b-ecg-flamingo-epoch-35` (recommended by Patrick + team)
- New `stage6_md_cot` in their `CurriculumTrainer`
- Data: MISATO MD trajectories (Nature Comp Sci 2024)

We are **not** the team that has to make CoT rationales work on molecules — we are the team that ships a working agentic system without that piece, by routing the language through a deterministic path.

## What we will say back to them

Suggested reply (short, technical, grateful):

> Thanks both! That's hugely helpful. We pivoted to the `add-chronos2-encoder` branch and the juncliu epoch-35 checkpoint today. On the rationale-labels question: rather than synthesise CoT targets, we're training only on `"Answer: <kcal/mol>. Confidence: <tier>."` strings (tier from PDBbind core/refined/general) and producing the language rationale at inference from a deterministic summariser over the same feature tensor. The abstention is driven by an independent physics verifier (MISATO `frames_EPtot`) the model never sees during training — disagreement above z-threshold → INCONCLUSIVE. That sidesteps the rationale-data bottleneck and gives us auditability for the demo. Will share results.

# Idea Evaluation Template

> Copy this file once per idea (`idea-2-<slug>.md`, `idea-3-...`, `idea-4-...`) and fill it in.
> Score on the same 10 axes so the four ideas are directly comparable.

---

## 1. The 30-second pitch

*One paragraph. End with a one-sentence summary.*

---

## 2. What we'd actually train (the artifact)

| | |
|---|---|
| **Artifact** | *e.g. fine-tuned checkpoint, distilled model, custom-trained adapter, RL policy* |
| **Trainable params** | *order-of-magnitude* |
| **Frozen** | *which parts are not trained* |
| **Starting point** | *pretrained checkpoint? from scratch?* |
| **Input** | *shape + modality* |
| **Output** | *shape + modality* |
| **Eval metric** | *the single number that determines whether it worked* |

**Why this is not just an LLM wrapper:** *one sentence — if you can't answer this, the idea fails the hackathon's hard constraint.*

---

## 3. What makes it agentic (the loop)

*Describe the multi-step loop in pseudocode. Highlight: tool-use, evaluation/verification step, graceful failure / abstention behavior, structured output.*

---

## 4. Mapping to the hackathon's hard requirements

| Requirement | How this idea satisfies it |
|---|---|
| NOT an "LLM wrapper" — must have a real trained model | |
| Novel infrastructure, hard to copy, "not seen before" | |
| Open-source dataset, used responsibly | |
| Clear business impact + target user | |
| Agentic engineering with real scalability | |
| Deliberate AWS integration | |

---

## 5. Resonance with the judges (qualitative)

| Judge signal | How this idea aligns |
|---|---|
| Judges' flagship work is **OpenTSLM** (time-series as native LLM modality) | |
| Judges' identity: "design, develop, and **EVALUATE** agentic systems" | |
| Previous winner: **SOC agent with INCONCLUSIVE verdict / graceful failure** | |
| **AWS sponsorship** wants serious infra use | |

---

## 6. Business case

| | |
|---|---|
| **Who pays** | |
| **What's broken today** | |
| **Why now (what changed in the last 12 months)** | |
| **Realistic deployment path** | |
| **Why the agentic behavior matters commercially** | |

---

## 7. Implementation plan (compressed)

### Pre-clock (tonight)

*What absolutely has to happen before hour 0?*

### Official 24 h

| Hour | Output | Gate |
|---|---|---|
| 0-4 | | |
| 4-8 | | |
| 8-14 | | |
| 14-18 | | |
| 18-22 | | |
| 22-24 | | |

---

## 8. Risks (honest)

| Risk | Mitigation | Residual P(failure) |
|---|---|---|
| | | |
| | | |
| | | |

**Honest feasibility estimate:** *P(working demo) | P(hitting primary metric) | P(any compelling pitch)*

---

## 9. Self-score (1-5 per axis)

| Criterion | Score | Justification |
|---|---|---|
| Trained model artifact (not wrapper) | / 5 | |
| Novelty of infrastructure | / 5 | |
| Open-source dataset, responsible use | / 5 | |
| Business impact + clear user | / 5 | |
| Real engineering / scalability | / 5 | |
| AWS integration depth | / 5 | |
| Agentic behavior (multi-step + abstention) | / 5 | |
| Resonance with judges | / 5 | |
| 24 h feasibility | / 5 | |
| Demo wow-factor | / 5 | |
| **TOTAL** | / 50 | |

---

## 10. What would make us pick THIS over another idea

Pick this if:
-
-
-

Don't pick this if:
-
-
-

---

## 11. Open questions for the team

-
-
-

---

# Scoring rubric — what each 1-5 means

> Apply consistently across all four ideas.

| Axis | 1 = Poor | 3 = Okay | 5 = Excellent |
|---|---|---|---|
| **Trained model artifact** | Pure API/wrapper, no training | LoRA / small adapter trained on small data | Substantial trained component (≥10 M params or non-trivial architecture) with measurable artifact |
| **Novelty of infrastructure** | Tutorial-recreatable | Notable combination of existing tools | New combination + new architecture + not seen before in this domain |
| **Open-source dataset** | Closed/synthetic data | Standard public dataset, used as intended | Notable public dataset, novel use, properly cited |
| **Business impact** | Toy / unclear user | Plausible but not differentiated | Specific user, specific broken workflow, specific dollar savings |
| **Real engineering** | Notebook-only, single-PDB-id hack | Pipeline runs over a small set | Reproducible across the dataset, handles failure cases |
| **AWS integration** | None or token | One service used meaningfully | Multiple services with clear, non-cosmetic roles |
| **Agentic behavior** | Single-turn LLM call | Multi-step but no verification | Multi-step + tool-use + verification + graceful failure |
| **Resonance with judges** | Off-topic | Generally aligned | Directly extends judges' flagship work and/or mirrors prior winner's pattern |
| **24 h feasibility** | Needs days | Tight but possible with team's stack | Comfortable margin, multiple safety nets |
| **Demo wow-factor** | Static screenshots / CLI only | Live web demo works | Live demo + visceral "this is doing something real" moment |

---

# Four-way comparison sheet (fill in after all four ideas scored)

| Axis | Idea 1: TSLM-MD | Idea 2: ___ | Idea 3: ___ | Idea 4: ___ |
|---|---|---|---|---|
| Trained model artifact | 5 | | | |
| Novelty of infrastructure | 5 | | | |
| Open-source dataset | 5 | | | |
| Business impact | 4 | | | |
| Real engineering | 4 | | | |
| AWS integration | 4 | | | |
| Agentic behavior | 5 | | | |
| Resonance with judges | 5 | | | |
| 24 h feasibility | 3 | | | |
| Demo wow-factor | 4 | | | |
| **TOTAL** | **44** | | | |

---

# Decision questions to discuss as a team after scoring

1. Which idea has the highest **ceiling** (best possible outcome if everything works)?
2. Which idea has the highest **floor** (least embarrassing outcome if things go wrong)?
3. Which idea best matches the **specific skills already on the team**?
4. Which idea would each of us be **most excited to demo** at hour 24?
5. If two ideas are within 3 points: which one would we regret *not* doing more?

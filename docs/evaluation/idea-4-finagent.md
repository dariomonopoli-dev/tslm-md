# Idea Evaluation — FinAgent (the chosen idea)

> Scored on the same 10-axis rubric as the other ideas, then a strategic assessment of how to maximise win probability.

---

## Scoring

| Criterion | Score | Justification |
|---|---|---|
| Trained model artifact (not wrapper) | **5** | Genuine fine-tune of a VLM (Llama 3.2 Vision / Qwen2.5 VL 3B) with SFT + RLVR + DPO. Central artifact, not an afterthought. Continuous DPO loop is production-grade ML thinking. |
| Novelty of infrastructure | **4** | VLM + Swiss financial documents + continuous DPO-from-corrections loop is well-differentiated. Multi-agent Bedrock orchestration itself is standard. |
| Open-source dataset, responsible use | **3** | "Swiss invoice datasets" not specifically named. "500 labeled Swiss SME documents" — origin unclear. Kontenrahmen KMU + MWSTG are public reference texts, not labelled training data. **The dataset question is the single biggest unknown.** |
| Business impact + clear user | **5** | CHF 3-15k/year per SME, 600k SMEs in CH, 2025 FTA digital-mandate tailwind. Quantified, cited, concrete. |
| Real engineering / scalability | **5** | Continuous learning loop, SageMaker Pipelines, clear CH → DACH → EU expansion path with per-country VAT-agent variants. |
| AWS integration depth | **5** | Textract + SageMaker + Bedrock + Bedrock KB + S3 + Lambda + EventBridge + EC2 + API Gateway. Most extensive AWS surface of any idea evaluated. |
| Agentic behaviour (multi-step + abstention) | **4** | Supervisor + 4 specialist agents = real multi-step + tool-use. **No explicit abstention/escalation mechanism described** — the SOC-agent prior-winner pattern is missing. Easy 1-paragraph fix. |
| Resonance with judges | **3** | Hits the technical excellence verbs (fine-tune + RAG + multi-agent + benchmark). Doesn't extend judges' flagship (OpenTSLM not used), doesn't use ETH ecosystem (no Apertus). Swiss focus is the only judge-adjacent angle. |
| 24 h feasibility | **3** | Multimodal VLM fine-tune (SFT + RLVR + DPO) + 4 agents + RAG per agent + Textract pipeline + Bedrock orchestration + benchmark in 24 h is genuinely ambitious. VLM fine-tuning specifically has higher memory + variance than text-only. |
| Demo wow-factor | **5** | Swiss restaurant receipt → *Repräsentationsaufwand* 6840 → 7.7% VAT → 50% deductible → booking entry, in <10s. Visceral, Swiss-specific, judges can imagine using it personally. |
| **TOTAL** | **42 / 50** | |

---

## Updated four-way matrix

| Criterion | TSLM-MD | Helvetia | Trade Compliance | **FinAgent** |
|---|---|---|---|---|
| Trained model artifact | 5 | 3 | 1 | **5** |
| Novelty of infrastructure | 5 | 4 | 2 | **4** |
| Open-source dataset | 5 | 5 | 3 | **3** |
| Business impact + user | 4 | 5 | 4 | **5** |
| Real engineering | 4 | 4 | 4 | **5** |
| AWS integration | 4 | 5 | 5 | **5** |
| Agentic behaviour | 5 | 5 | 4 | **4** |
| Resonance with judges | 5 | 4 | 2 | **3** |
| 24 h feasibility | 3 | 4 | 4 | **3** |
| Demo wow-factor | 4 | 5 | 3 | **5** |
| **TOTAL** | **44** | **44** | **32** | **42** |

FinAgent ranks #3 of 4 by raw score, but the gap to the leaders is small (-2) and the strengths-on-different-axes pattern is real. The two scores I'd flag as *most fixable to lift before the pitch* are **agentic behaviour (4 → 5)** and **judge resonance (3 → 4)** — both addressable with pitch-deck additions, not code changes.

---

## Can we win? Honest assessment

**Yes — probably 25-40% chance, depending on execution.** The idea is genuinely competitive but not the strongest possible pitch *as currently written*. Three things determine outcome:

1. **Does the fine-tune actually converge meaningfully in 24 h?** VLM fine-tuning is harder than text-only. If the benchmark slide says "fine-tuned VLM beats Claude 3.5 Sonnet by X% on field extraction" — you win the technical-creativity criterion. If it says "≈ Claude 3.5" — the pitch shifts to "look at our Swiss-specific reasoning" which is softer.
2. **Does the demo land in under 10 s, live, on a judge-supplied document?** Pre-recorded backup is a safety net but live > recorded by a wide margin on wow-factor.
3. **Do you add abstention?** Currently the biggest gap vs prior-winner pattern. See action #1 below.

The path-to-win exists. The execution risk is real. The pitch can be lifted significantly with 4-5 specific additions described below — all of which are *pitch-deck level* (1-2 hours of work each), not code-level.

---

## Five concrete actions to maximise win probability

### Action #1 — Add abstention. Single highest-leverage change.

The previous track's winning project was the SOC agent that returned **INCONCLUSIVE** instead of guessing. The judges have publicly signalled they reward graceful failure. FinAgent currently doesn't have this.

**Fix (one paragraph in the pitch + ~30 lines of code):**
> "FinAgent computes a confidence score from (a) VLM logit entropy on the extracted fields, (b) RAG-retrieval agreement across top-k passages, and (c) a deterministic rule-based VAT calculator's agreement with the VLM output. If confidence is below threshold, FinAgent **abstains and escalates to a Treuhänder** with a structured summary of what it's uncertain about. Default: ~15-20% of edge cases get escalated — measurable and tuneable."

This lifts the agentic-behaviour score from 4 → 5, and the judge-resonance score from 3 → 4. **Do this even if you don't change anything else.**

### Action #2 — Frame the evaluation centrally, not as a closing bullet.

The judges' lab identity is literally "design, develop, and **evaluate** agentic systems." Right now the benchmark vs Claude is mentioned once. It should be:
- The **opening slide** of the technical section ("here's our held-out test set, here's our metric, here's our scoreboard")
- The **closing number** of the demo ("on 500 labelled docs, fine-tuned FinAgent achieves X% field-extraction accuracy vs Y% for Claude 3.5 Sonnet baseline — and abstains on Z% with N% recall of true errors")
- A **chart on the deck** — not a sentence

Same data, three placements, dramatically higher signal.

### Action #3 — Acknowledge OpenTSLM in the pitch.

You can't ethically retrofit OpenTSLM into FinAgent's stack at this stage, but you *can* tip the cap. One slide late in the deck:
> "The Agentic Systems Lab's own OpenTSLM has shown that fine-tuning a small frozen LLM with a domain-specific adapter beats prompting a frontier model in time-series reasoning. FinAgent applies the same principle to multimodal Swiss financial documents — fine-tuned 3B VLM beats Claude 3.5 Sonnet on the field-extraction benchmark."

This is honest *and* judge-flattering. Lifts judge resonance from 3 → 4 with zero code work.

### Action #4 — Name the dataset (or own that you built it).

The "500 labelled Swiss SME documents" is the single weakest cell in the rubric. Options:
- If it's a public dataset → cite it (turns 3 → 5 on the dataset axis)
- If it's hand-labelled by the team → say so explicitly ("we curated 500 labelled documents from anonymised Swiss SME invoices, available under CC-BY-4.0 with this submission") → turns the weakness into *another* contribution
- If it's synthetic / generated → say so + note the validation methodology

The worst answer is "trust us, it exists" — judges will probe.

### Action #5 — Pitch the 4-agent architecture as "built 2, designed 4" *positively*.

"Built fully: Bookkeeping + VAT Agent" can read as "you only built half." Re-frame:
> "We made an architectural decision to build the two highest-revenue-impact agents *to production quality* with full fine-tuning, RAG, and evaluation — rather than four stubs. The Cash Flow and Fiduciary agents are architected with the same pattern and would slot in directly. We're shipping a working system, not a sketch."

Same words, opposite signal.

---

## Risks to actively manage in the next 24 h

| Risk | Mitigation | When |
|---|---|---|
| VLM fine-tune doesn't converge in 24 h on a multimodal task | **Pipeline-first**: have the full system working on the BASE VLM by hour 8. Fine-tune is an *upgrade*, never on critical path. If at hour 18 fine-tuning is still not converging, ship with base + spend the time on the abstention layer and the eval chart. | Continuous |
| Dataset labels are wrong or sparse | Spend hour 0-2 verifying labels by hand on a 20-doc sample. Bad labels = bad fine-tune = wasted training time. | Hour 0-2 |
| Textract + VLM handoff fails on Swiss-specific layouts (QR-bills, ESR slips) | Test specifically on a Swiss QR-bill in hour 1. If it breaks, you'd rather know now than at hour 20. | Hour 1 |
| Multi-agent Bedrock orchestration is more setup than expected | Get Supervisor + ONE specialist agent end-to-end by hour 6. Adding the second specialist is then a copy-paste. | Hour 6 gate |
| Demo fails live on a judge-supplied document | Record a backup demo video at hour 22. Have it ready to play. Live first, recorded as parachute. | Hour 22 |
| The "10 seconds" claim is bigger than reality | Time the actual end-to-end latency at hour 18. If it's 30s, change the slide. Don't let a judge time you and prove you wrong. | Hour 18 |

---

## Where FinAgent is genuinely strong (lean into these)

1. **The continuous-learning loop is the most novel part of the system.** SFT → RLVR → DPO from user corrections, retrained weekly, is production ML — not hackathon code. Most teams won't have this. Make it a slide.
2. **Most extensive AWS surface of any submission.** Textract + SageMaker + Bedrock + KB + S3 + Lambda + EventBridge + API Gateway. AWS sponsors notice.
3. **2025 FTA digital-mandate is a real regulatory tailwind**, not invented "why now." Cite the actual law/date in the deck.
4. **The demo is visceral and Swiss-specific.** Receipt → *Repräsentationsaufwand* with the actual VAT rate is the kind of moment a non-technical judge can grasp instantly.
5. **Business case is the strongest of the four ideas.** 600k SMEs × CHF 3-15k/yr is a real TAM, not a hand-wave.

---

## What I'd be most worried about if I were on the team

- **Hour 0-2 has to confirm the dataset is real and usable.** If it's not, the whole "fine-tuned VLM beats Claude" pitch evaporates.
- **The fine-tune must produce a number better than Claude on the held-out set.** Even by 2-3 percentage points. The pitch is *built* on this delta. If it's negative or zero, the technical claim collapses.
- **The "no abstention" gap.** Add it. Today. Pitch + 30 lines of code.

---

## Net verdict

**FinAgent is a credible winning candidate** — well-architected, business-relevant, technically substantive, AWS-deep, demo-friendly. It's not the *most-aligned-with-this-specific-judging-panel* option I would have ranked first, but it's well above the disqualification line and competitive on substance.

**The path-to-win is execution + 4-5 specific pitch upgrades, not a rebuild.** Add abstention. Centre the evaluation. Tip the cap to OpenTSLM. Name the dataset. Re-frame "stubbed" as "scope-disciplined." Then ship a working demo at hour 24.

If those happen → **40% chance of winning**. If only execution happens → **25%**. If execution slips → **10%**.

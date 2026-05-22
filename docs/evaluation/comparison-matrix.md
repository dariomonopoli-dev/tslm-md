# Three-Way Idea Comparison

Scoring all three pitches on the same 10-axis rubric (`evaluation-template.md`). Same rubric I applied to my own idea — no home-team bias. The fourth idea will slot in once your teammate writes it.

## TL;DR

| | TSLM-MD | Helvetia | Trade Compliance |
|---|---|---|---|
| **Total / 50** | **44** | **44** | **32** |
| **One-line characterisation** | Highest ceiling, hardest to build | Best business case, lowest risk | Most extensive AWS, fails hackathon's #1 constraint |

**TSLM-MD and Helvetia tie at 44 with opposite strength profiles.** Trade Compliance trails by 12 points primarily because of one structural issue called out below. Read the per-axis justifications before reacting.

---

## Side-by-side scoring matrix

| Criterion | TSLM-MD | Helvetia | Trade Compliance |
|---|---|---|---|
| Trained model artifact (not wrapper) | **5** | **3** | **1** ⚠ |
| Novelty of infrastructure | **5** | **4** | **2** |
| Open-source dataset, responsible use | **5** | **5** | **3** |
| Business impact + clear user | **4** | **5** | **4** |
| Real engineering / scalability | **4** | **4** | **4** |
| AWS integration depth | **4** | **5** | **5** |
| Agentic behaviour (multi-step + abstention) | **5** | **5** | **4** |
| Resonance with judges | **5** | **4** | **2** |
| 24 h feasibility | **3** | **4** | **4** |
| Demo wow-factor | **4** | **5** | **3** |
| **TOTAL** | **44** | **44** | **32** |

---

## Per-axis justifications

### 1. Trained model artifact (not wrapper)
- **TSLM-MD = 5.** Adapter fine-tune is the *central* artifact (~50-200 M trainable params: perceiver + gated cross-attention + CNN encoder + LM input embeddings). The artifact IS the project.
- **Helvetia = 3.** A LoRA fine-tune of Apertus-8B *is* a real trained component (small adapter, but real params trained on real labels), but the team itself writes "fine-tune is an upgrade, never on critical path" — meaning the demo can run on the base model. At demo time this could read as a wrapper if the fine-tune fails. Score reflects the *worst-case* shape.
- **Trade Compliance = 1.** ⚠ Read the proposal carefully — **there is no trained model anywhere**. It uses Claude-3.5-Sonnet (pretrained), function-calling, OpenSearch RAG, and EC2 scrapers. Every model is off-the-shelf. The hackathon's first hard constraint, quoted from the brief: *"NOT an 'LLM wrapper'. There must be a model artifact we actually trained."* As written, this proposal fails that gate. Fix is non-trivial in 24 h — would need to add e.g. a fine-tuned HS-code classifier or a distilled regulatory-domain embedding model, which weren't scoped.

### 2. Novelty of infrastructure
- **TSLM-MD = 5.** First TSLM on molecules; OpenTSLM has only ever been applied to 1-D medical signals. Novel featurization pipeline + stage-6 curriculum extension + verifier-driven abstention layer.
- **Helvetia = 4.** Apertus + Swiss-German is genuinely hard for any non-Swiss team to replicate (US foundation models are weak on Swiss-German). The abstention layer is novel-ish in the agent space (calibration is well-studied; agent-level abstention less so). Combination is differentiated.
- **Trade Compliance = 2.** Bedrock Agents + multi-agent supervisor + OpenSearch RAG is essentially the standard AWS reference architecture for "agentic enterprise app". Tutorial-recreatable. There are also commercial products in this exact space (Descartes, Avalara, SAP GTS) — the "moat" framing is weaker.

### 3. Open-source dataset, responsible use
- **TSLM-MD = 5.** MISATO (Nature Computational Science 2024, Apache-2.0) + PDBbind v2020 — both cited, properly attributed, used for their intended purpose.
- **Helvetia = 5.** Fedlex + ESTV + opendata.swiss — all official Swiss government open data, plus Apache-2.0 Apertus. Strong on this axis.
- **Trade Compliance = 3.** Mentions "bilateral free-trade agreements and ESG regulatory frameworks" but doesn't name a specific dataset. Public government data is fine but unnamed = weaker.

### 4. Business impact + clear user
- **TSLM-MD = 4.** Real drug-discovery problem ($2-3B/drug, 90% clinical failure rate, MD signal discarded), specific user (computational chemists). Distribution channel exists (plug into MD pipelines). One point short of 5 because "we'll improve drug discovery" is a long, soft impact chain.
- **Helvetia = 5.** Strongest of all three. Cited numbers (55 h/month per SME, CHF 6 B/yr, VAT is the #1 ranked ask). Clear primary user (SME owner) AND clear buyer (Treuhänder fiduciary). Concentrated paying distribution channel. This is venture-quality target-user thinking.
- **Trade Compliance = 4.** Real user (compliance officers at industrial firms), real pain. Less evidence-backed than Helvetia ("massive bottlenecks", "weeks instead of minutes" — qualitative not quantitative).

### 5. Real engineering / scalability
- **TSLM-MD = 4.** Reproducible pipeline across the dataset; multi-tenant friendly (per-complex inference). Depends on training converging.
- **Helvetia = 4.** Multi-tenant SaaS architecture; per-SME serving via Bedrock + Qdrant scales linearly. Honest "code-orchestrated single agent" framing.
- **Trade Compliance = 4.** Proper VPC + KMS + serverless, multi-agent decomposition, audit-ready PDF output. Enterprise-grade if a bit heavy.

### 6. AWS integration depth
- **TSLM-MD = 4.** S3 + EC2 (one-time) + SageMaker (fallback) + Bedrock (optional Claude) + AgentCore (documented). Multiple services with clear roles, but not all on critical path.
- **Helvetia = 5.** S3 (data) → SageMaker (LoRA training, *on critical path*) → Bedrock (Apertus serving + Claude orchestration). Three services, all genuinely load-bearing. Full lifecycle.
- **Trade Compliance = 5.** S3 + EC2 + OpenSearch Serverless + Bedrock + Bedrock Agents + KMS + VPC. Most extensive AWS surface of any idea. However — this also reads as *AWS compensating for thin model novelty*. Judges may notice that.

### 7. Agentic behaviour (multi-step + abstention)
- **TSLM-MD = 5.** Multi-step loop, three independent tools (TSLM, deterministic rationale, physics verifier), graceful failure via INCONCLUSIVE. Direct structural parallel to the SOC-agent prior winner.
- **Helvetia = 5.** Multi-step (extract → classify → retrieve → verify → abstain), Claude-checks-grounding verification step, explicit abstention with human-escalation. Also mirrors the prior winner's pattern.
- **Trade Compliance = 4.** Four genuinely specialised agents with a Supervisor — strong multi-step structure. But: **no abstention mechanism described.** The closest is "traffic-light risk rating" which is a classification, not a refusal-to-answer. Misses the prior-winner pattern.

### 8. Resonance with judges (ETH Agentic Systems Lab)
- **TSLM-MD = 5.** Extends *exactly* the lab's own flagship work (OpenTSLM). Literally adds stage 6 to their curriculum. Maximum possible alignment.
- **Helvetia = 4.** Uses Apertus (ETH/EPFL/CSCS), which is ETH-adjacent and Swiss-sovereign — strong. One point short of 5 because **Apertus is not the Agentic Systems Lab's flagship; OpenTSLM is.** The judges built and care about OpenTSLM more than Apertus.
- **Trade Compliance = 2.** No use of the lab's ecosystem at all. No abstention pattern. Generic enterprise-AWS application. Could have been pitched at any AWS hackathon. The judges' specific identity is not reflected.

### 9. 24 h feasibility
- **TSLM-MD = 3.** Borderline. Fine-tuning + new dataset + new featurisation + agent + demo in 24 h is achievable with the C-MAPSS insurance net, but variance is high. Already negotiated honestly in the spec.
- **Helvetia = 4.** "Pipeline-first on base model, fine-tune is the upgrade" is the right risk posture. Most stack pieces are well-trodden (RAG, FastAPI, LoRA on SageMaker). Comfortable margin for a 3-4 person team.
- **Trade Compliance = 4.** No convergence risk (no training). Standard AWS services. The risky bit is OpenSearch Serverless setup + tuning embeddings in 6 h, but it's bounded engineering, not unbounded research.

### 10. Demo wow-factor
- **TSLM-MD = 4.** "Paste a PDB id, watch features stream, see the agent abstain on a hard case" is crisp but requires audience domain literacy. Would be 5 with a 3D molecular viewer (out of scope).
- **Helvetia = 5.** Café-owner uploads real receipt → live cite of actual ESTV article → catches deductibility error → abstains on ambiguous one. Visceral, relatable, judges-can-imagine-using-it.
- **Trade Compliance = 3.** "Upload BOM, get compliance PDF, traffic-light rating" is functional but not visceral. PDF output is the weakest demo medium.

---

## Profile comparison: where each idea is strongest

**TSLM-MD wins on:** trained artifact, novelty, judge resonance, agentic graceful failure.
**Helvetia wins on:** business case, AWS lifecycle integration, demo memorability, 24 h feasibility, dataset quality (tied).
**Trade Compliance wins on:** raw AWS surface area, breadth of agent decomposition.

```
       Trained  Novelty  Data  Bus.  Eng.  AWS  Agent  Judge  Feas.  Demo
TSLM     ★★★★★   ★★★★★   ★★★★★ ★★★★  ★★★★  ★★★★ ★★★★★  ★★★★★  ★★★    ★★★★
Helv     ★★★     ★★★★    ★★★★★ ★★★★★ ★★★★  ★★★★★★★★★★ ★★★★   ★★★★   ★★★★★
Trade    ★       ★★      ★★★   ★★★★  ★★★★  ★★★★★★★★★  ★★     ★★★★   ★★★
```

The TSLM-MD vs Helvetia tie is real. The Trade Compliance gap is real and primarily *one issue*.

---

## The Trade Compliance issue (called out clearly)

The hackathon brief is explicit: *"NOT an 'LLM wrapper'. There must be a model artifact we actually trained."*

The Trade Compliance proposal as written has no trained component. Every model in the pipeline is pretrained off-the-shelf (Claude-3.5-Sonnet, generic embedding models in OpenSearch). The four "agents" are role-played Claude calls with tool access. This is the exact architectural pattern the brief warns against.

This isn't a minor scoring quibble — it's the hackathon's **first hard constraint**. Judges will either:
- (a) Disqualify the project outright on the wrapper criterion, or
- (b) Score it heavily down on technical creativity ("nothing built, only assembled")

To pass the wrapper gate, Trade Compliance would need to add a trained component — realistic options:
- Fine-tune a small classifier on HS-code → category mapping (real labeled data exists)
- Train a domain-specific embedding model on trade regulations (small T5 or sentence-transformer fine-tune)
- Distill Claude's compliance-reasoning into a smaller model

Any of these would lift the trained-artifact score from 1 → 3+ and the total from 32 → 36+. **This conversation with the Trade Compliance author should happen before final selection** — the idea may be salvageable with a 1-paragraph addition, but as written it has a structural problem.

---

## Where would you regret picking each?

- **TSLM-MD:** if the team has no one who's shipped a fine-tuning run before. The wiring + open_flamingo dep + convergence risk all compound on one person.
- **Helvetia:** if the team doesn't have someone fluent in Swiss-German or in Swiss VAT law. The pitch's whole moat ("nothing else handles Swiss-German") collapses if the model output reads as obviously-non-native to judges or to a Swiss-German-speaking demo audience.
- **Trade Compliance:** if no one adds a trained component, judges will dismiss the project on the explicit hackathon constraint. As written, this is the safe-feeling option that has the highest chance of an embarrassing failure mode for *non-technical* reasons.

---

## Highest-ceiling vs highest-floor

|  | Highest ceiling (best possible outcome) | Highest floor (least bad outcome) |
|---|---|---|
| **Winner** | **TSLM-MD** — "first TSLM on molecules, ETH's own model extended" is a publishable contribution if it lands | **Helvetia** — even a half-working demo on base Apertus is a compelling SME story |
| **Runner-up** | Helvetia | Trade Compliance (no training risk) |

If the team is risk-averse and wants the safest pitch with the most polished demo → **Helvetia**.
If the team has the ML horsepower and wants the highest-impact pitch → **TSLM-MD**.
**Trade Compliance is the third option only after the trained-artifact issue is resolved.**

---

## Hybrid worth discussing

The TSLM-MD and Helvetia structures are surprisingly similar — both are "fine-tune a small model + deterministic agent loop + verifier + abstain". The patterns are interchangeable. If the team has the right people, a **chassis swap is possible**:

- Helvetia chassis + TSLM-MD's deterministic-grounded-rationale rigor → strongest version of Helvetia
- TSLM-MD chassis + Helvetia's business-case framing in the pitch deck → strongest version of TSLM-MD

Not a fourth idea, but a conversation worth having: *which idea's bones, dressed in which idea's clothes, makes the best pitch?*

---

## What's missing: the 4th idea

Score it on the same 10 axes when the teammate writes it. If it scores below 35, the team's choice is effectively between TSLM-MD and Helvetia. If it scores 40+, it joins the real contest.

## Decision questions for the team meeting

1. Of the four ideas, which one would each of us be *most excited to demo* at hour 24? (Excitement is a real predictor of finishing strong)
2. Which one matches the actual skills *already on the team* — not the skills we wish we had?
3. Is the Trade Compliance author willing to add a trained component? If yes, re-score. If no, drop from contention.
4. For the two tied ideas (TSLM-MD, Helvetia): is anyone strongly opposed to either? Strong opposition matters more than weak preference.
5. Is the "hybrid" chassis-swap conversation worth 15 minutes before voting?

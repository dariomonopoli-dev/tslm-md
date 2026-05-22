# Frontend v2 — MD-Trajectory Binding Affinity Demo

**A custom Astro site that lets a visitor pick a PDB, see the model's prediction and rationale, watch an independent tool-using agent audit that prediction against orthogonal evidence, and triage candidates at batch scale.**

One-line pitch: the model deliverable is a checkpoint; the *credibility* deliverable is a public site where the prediction comes with a structured audit trail — physics, structure, literature, all cited — produced by an agent that uses information the model could not have memorized.

Supersedes the Gradio-based plan in `FRONTEND.md` (kept frozen as v1 for history). Companion to [PROJECT_BRIEF.md](./PROJECT_BRIEF.md) — read that first.

---

## 1. Goal

A public-facing demo with five properties. The first three carry over from v1; the last two are new and motivate the v2 rewrite.

1. **Shows the "movie" framing visually.** Frame slider drives a synchronized 3D viewer and time-series cursor, so the dynamics are not abstract.
2. **Surfaces the v1a vs v1b ablation directly.** A toggle swaps which checkpoint generated the prediction and rationale.
3. **Makes rationale grounding verifiable in the UI.** Each rationale claim is color-coded (verified / contradicted / unverifiable) against the underlying channel data — the deterministic regex verifier from brief §7.2.
4. **Surfaces an independent agent verdict alongside the prediction.** A tool-using agent (Claude Opus 4.7) re-checks each prediction against information the trained model never saw (raw coordinates, external physics, label-filtered literature) and emits a structured verdict with citations.
5. **Demonstrates auditable triage at batch scale.** A dedicated tab ranks dozens of candidates by their *defensible* predicted pK, not just the raw model output, so a chemist sees the agent's recommendation in their workflow context.

The frontend is **demo-grade**, not product-grade. No auth, no usage tracking, no multi-user state. Static site + one inference API + one Anthropic API dependency.

---

## 2. Audience and deployment target

| Decision | Value |
|---|---|
| Primary audience | External — paper supplement, LinkedIn link, recruiter messages |
| Frontend deployment | **Astro static site** under the existing `website/` project, hosted on Vercel / Netlify / Cloudflare Pages |
| Inference deployment | **FastAPI service** on SageMaker async endpoint (or Modal as a fallback) |
| Agent dependency | Anthropic API (Claude Opus 4.7) called from the inference service |
| Inference mode | Live — checkpoint loaded at API boot, forward pass on demand |
| Auth | None — public frontend |
| URL shape | `<your-domain>/demo/{single, batch, failure-modes, about}` |
| Polish bar | Inherits the existing site's brand styling; recruiter-facing — not pixel-perfect, but clearly not a Gradio default theme |

---

## 3. Architecture — three services

```
┌───────────────────────────────────────────────────────────────────────────┐
│                              USER BROWSER                                 │
│   Astro static site: /demo/{single, batch, failure-modes, about}          │
└─────────────────────────────────┬─────────────────────────────────────────┘
                                  │  HTTPS  (fetch + CORS)
                                  ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                       FastAPI inference service                           │
│                                                                           │
│   /predict, /predict/batch, /pdb_string   ── trained TSLM (v1a + v1b)     │
│   /evaluate                               ── RAG + LLM judge (fast)       │
│   /evaluate/agent                         ── RAG + LLM + tools loop       │
│                                                                           │
│   Tool registry: 13 orthogonal tools                                      │
│   RAG corpus: label-filtered chunks                                       │
│   Regex verifier: deterministic, in-proc                                  │
│                                                                           │
└────────────────────────────────────────────────────┬──────────────────────┘
                                                     │ Anthropic API
                                                     ▼
                                          ┌──────────────────────┐
                                          │  Claude Opus 4.7     │
                                          │  (agent reasoning,   │
                                          │   tool-use protocol) │
                                          └──────────────────────┘
```

The frontend is free (static). The GPU and Anthropic costs are concentrated in the inference service. The Anthropic API is the only external paid dependency at request time.

---

## 4. The four use cases

| # | Use case | Audience | Tab that demonstrates it |
|---|---|---|---|
| 1 | **Auditable triage** — rank candidates by *defensible* predicted pK; only "trust"-marked rows go forward | Medchem team | Batch |
| 2 | **Disagreement triage** — find systems where model and agent disagree; informs v2 dataset curation | ML researcher | Failure modes |
| 3 | **Hypothesis generation** — agent triangulation surfaces specific residues / interactions to test | Structural biologist | Single |
| 4 | **Auditable AI demonstration** — a six-step argument with sources, not a black-box prediction | Recruiter / reviewer | About → Single → Failure modes |

Each use case maps to one tab as the primary entry point, but all four use cases ultimately flow through Single for full inspection.

---

## 5. UI structure — four tabs

```
┌──────────────────────────────────────────────────────────────────────────┐
│  MD-Trajectory Affinity        Single │ Batch │ Failure modes │ About    │
└──────────────────────────────────────────────────────────────────────────┘
```

| Tab | Route | Primary use case | Contents |
|---|---|---|---|
| Single | `/demo/single` | Hypothesis generation (3) + recruiter demo (4) | PDB picker, prediction card, rationale with regex badges, 3D viewer, channel plot, agent verdict panel (expandable), agent trace, citations |
| Batch | `/demo/batch` | Auditable triage (1) | Multi-select PDBs, batch run, sortable/filterable table with recommendation column |
| Failure modes | `/demo/failure-modes` | Disagreement triage (2) | Precomputed top-10 disagreements, aggregate failure-pattern analysis |
| About | `/demo/about` | Recruiter demo (4) | Project intro, worked-example shortcuts, what-this-is-not list, independence guarantees |

---

## 6. Mockups

### 6.1 Single tab — after Predict and Run deep evaluation

```
╔════════════════════════════════════════════════════════════════════════════╗
║  MD-Trajectory Affinity        ▌Single│ Batch │ Failure modes │ About      ║
╠════════════════════════════════════════════════════════════════════════════╣
║  PDB ID  [1A1B  ▼]    Variant  [v1a │▌v1b]    [Predict]                    ║
║                                                                            ║
║  ┌──────────────────────────────────┬─────────────────────────────────┐    ║
║  │  ─── Prediction ───              │  3D pocket view                 │    ║
║  │  Predicted pK     6.42           │  ┌───────────────────────────┐  │    ║
║  │  Actual pK        6.31           │  │  protein cartoon (gray)   │  │    ║
║  │  |Δ|              0.11           │  │  ligand sticks (orange)   │  │    ║
║  │  Variant          v1b (hybrid)   │  │  Lys-145 highlighted      │  │    ║
║  │  Model version    v1b-2026-05-22 │  │  frame 47 / 99            │  │    ║
║  │                                  │  └───────────────────────────┘  │    ║
║  │  ─── Rationale (regex verified)──│  ◀ ●━━━━━━━━━━━━━━ ▶  47/99     │    ║
║  │  Interaction energy averages     │  [▶ play]  [⟳ loop]             │    ║
║  │  -37.2 kcal/mol  ✓               ├─────────────────────────────────┤    ║
║  │  stabilises after frame 20. ✓    │  Per-frame channels             │    ║
║  │  Ligand RMSD under 2.5 Å  ✗      │  ┌───────────────────────────┐  │    ║
║  │  bSASA above 500 Å²  ✓           │  │ RMSD     ╱╲      ╱╲       │  │    ║
║  │  Pose is stable.  ?              │  │ energy ╱  ╲╱╲╱    ╲       │  │    ║
║  │                                  │  │ dist   ─────────────      │  │    ║
║  │  Regex verified: 3/4 (75%)       │  │ bSASA  ▔▔▔▔▔▔▔▔▔▔         │  │    ║
║  └──────────────────────────────────┴─────────────────────────────────┘    ║
║                                                                            ║
║  ┌─ Independent agent verdict ────────────────────────────────────────┐    ║
║  │  Recommendation:  ✓ TRUST                                          │    ║
║  │                                                                    │    ║
║  │  Structural   █████████░ 0.85   (cluster_poses: 1 dominant pose)   │    ║
║  │  Physical     ███████░░░ 0.72   (vina: -7.2, within 0.3 pK)        │    ║
║  │  Literature   █████████░ 0.90   (PDBbind: kinase family)           │    ║
║  │  Chemical     ██████░░░░ 0.65   (LE=0.40, plausible)               │    ║
║  │                                                                    │    ║
║  │  ▾ Agent trace (6 steps, 18 s, $0.22)                              │    ║
║  │    1. lookup_split           → "test"                              │    ║
║  │    2. rag_query              → 6 chunks, 2 leak-filtered           │    ║
║  │    3. cluster_poses(k=3)     → 84/11/5 → 1 dominant cluster        │    ║
║  │    4. vina_rescore(frame=47) → -7.2 kcal/mol (~ pK 5.3)            │    ║
║  │    5. hbond_persistence      → Lys-145↔ligand-N1 in 94% of frames  │    ║
║  │    6. ligand_descriptors     → MW=283, LogP=2.1, LE=0.40           │    ║
║  │                                                                    │    ║
║  │  ▾ Citations                                                       │    ║
║  │    [PDBbind_1A1B_binding_mode]   kinase ATP pocket, Lys-145        │    ║
║  │    [uniprot_P00734]              serine/threonine kinase family    │    ║
║  │                                                                    │    ║
║  │  Caveats:                                                          │    ║
║  │  • Train/test split: test ✓                                        │    ║
║  │  • Prior-knowledge probe: Claude returned "unknown" without RAG ✓  │    ║
║  │  • RAG label-filter: 2 chunks containing Kd excluded ✓             │    ║
║  │                                                                    │    ║
║  │  Hypothesis surfaced by agent:                                     │    ║
║  │    "Lys-145 is load-bearing. K145A should reduce affinity ≥1.5 pK" │    ║
║  └────────────────────────────────────────────────────────────────────┘    ║
║                                                                            ║
║  [ Show baselines ▾ ]   [ Compare to v1a ⇄ ]   [ Export trace as JSON ]    ║
╚════════════════════════════════════════════════════════════════════════════╝
```

### 6.2 Batch tab

```
╔════════════════════════════════════════════════════════════════════════════╗
║  Batch triage                                       Variant: [v1b ▼]       ║
║  Rank a set of test-split PDBs by their defensible predicted pK.           ║
║                                                                            ║
║  Selected (12): [1A1B] [1A28] [1A30] [1B6H] [1F0R] [1F0S] [1G2K] [1KE5]    ║
║                 [1NHU] [1QPE] [2BR1] [2X3K]              [+ Add] [Clear]   ║
║                                                                            ║
║  [+ Pick 20 random]  [Import CSV]                                          ║
║  ☑ Include agent evaluation (~$0.20 each, ~20 s each)                      ║
║                                                                            ║
║  [ Run batch ]                                  Est cost: $2.40, ~4 min    ║
║                                                                            ║
║  Sort: [recommendation ▼]   Filter: ☑trust ☑review ☐discard   12/12 shown  ║
║                                                                            ║
║  ┌──────┬───────┬──────┬──────────┬──────────┬───────────────────────────┐ ║
║  │ PDB  │ pred  │ |Δ|  │ regex    │ agent    │ supporting evidence       │ ║
║  ├──────┼───────┼──────┼──────────┼──────────┼───────────────────────────┤ ║
║  │ 1A28 │ 8.91  │ 0.12 │ 4/4 ✓✓✓✓ │ ✓ trust  │ vina ✓, hbond Lys145 94%  │ ║
║  │ 1F0R │ 7.84  │ 0.21 │ 4/4 ✓✓✓✓ │ ✓ trust  │ vina ✓, lit confirms      │ ║
║  │ 1A1B │ 6.42  │ 0.11 │ 3/4 ✓✓✓✗ │ ✓ trust  │ vina ✓, 1 contradiction   │ ║
║  │ 1B6H │ 7.21  │ 0.59 │ 4/4 ✓✓✓✓ │ ⚠ review │ vina disagrees by 1.4     │ ║
║  │ 1G2K │ 5.40  │ 0.31 │ 2/4 ✓✗✓✗ │ ⚠ review │ pose unstable (3 clust.)  │ ║
║  │ 2X3K │ 7.50  │ 1.50 │ 4/4 ✓✓✓✓ │ ✗ discard│ LE implausible, no lit.   │ ║
║  │ 1QPE │ 4.80  │ 0.42 │ 3/4 ✓✓✗✓ │ ✓ trust  │ weak binder, agent agrees │ ║
║  └──────┴───────┴──────┴──────────┴──────────┴───────────────────────────┘ ║
║                                                                            ║
║  ↑ 2X3K passes regex but agent rejects. Click row for full trace.          ║
║  [ Export selected as CSV ]      [ Send only "trust" to assay queue ]      ║
╚════════════════════════════════════════════════════════════════════════════╝
```

### 6.3 Failure modes tab

```
╔════════════════════════════════════════════════════════════════════════════╗
║  Where the model fails                                                     ║
║  Predictions the model made confidently, where the independent agent found ║
║  contradicting evidence. The 10 most informative test-set systems.         ║
║                                                                            ║
║  Sort: [|model − mmgbsa| ▼]    Variant: [v1b ▼]                            ║
║                                                                            ║
║  ┌──────┬──────┬──────┬─────────┬─────────┬─────────────────────────────┐  ║
║  │ PDB  │model │vina  │mm-gbsa  │agent    │ why the model is wrong      │  ║
║  ├──────┼──────┼──────┼─────────┼─────────┼─────────────────────────────┤  ║
║  │ 2X3K │ 7.5  │ 5.1  │ 5.3     │✗discard │ LE 0.71 — implausible for   │  ║
║  │      │      │      │         │         │ 320 Da ligand               │  ║
║  │ 4HHB │ 8.2  │ 6.7  │ 6.5     │⚠review  │ 1 outlier frame dominates;  │  ║
║  │      │      │      │         │         │ clash_check flagged frm 73  │  ║
║  │ 1RPE │ 7.0  │ 5.2  │ 5.4     │✗discard │ Pose splits into 3 clusters │  ║
║  └──────┴──────┴──────┴─────────┴─────────┴─────────────────────────────┘  ║
║                                                                            ║
║  Click any row → opens prediction + agent trace in Single tab.             ║
║                                                                            ║
║  Aggregate failure pattern analysis                                        ║
║   Failure cluster                      Count   Affected systems            ║
║   Implausible ligand efficiency          3     2X3K, 1ZZB, 2P2N            ║
║   Pose unstable (>2 clusters)            4     1RPE, 1ABE, 1B6H, 1F8B      ║
║   Single-frame outlier dominates         2     4HHB, 1MZK                  ║
║   Literature contradicts binding mode    1     2X3K                        ║
║   → v2 suggestion: add pose-stability auxiliary supervision.               ║
╚════════════════════════════════════════════════════════════════════════════╝
```

### 6.4 About tab

```
╔════════════════════════════════════════════════════════════════════════════╗
║  MD-Trajectory Binding Affinity                                            ║
║  "We read the molecular movie — not the snapshot — and explain it."        ║
║                                                                            ║
║  ┌─ Try these worked examples ───────────────────────────────────────────┐ ║
║  │  [1A1B — easy, trust]    Stable binder, all 4 sources agree           │ ║
║  │  [4QZL — hard but right] Model beats Vina by 1.8 pK, lit confirms     │ ║
║  │  [2X3K — failure caught] Model overshoots; agent catches the gap      │ ║
║  └───────────────────────────────────────────────────────────────────────┘ ║
║                                                                            ║
║  How it works                                                              ║
║   MISATO MD → trained TSLM → regex verifier → independent agent            ║
║   4 channels   pK +           rationale vs.   physics + structure +        ║
║                rationale      channels        label-filtered literature    ║
║                                                                            ║
║  What this is              What this is NOT                                ║
║   ✓ OpenTSLM-SP applied     ✗ A production drug-discovery tool             ║
║     to a new modality       ✗ A replacement for wet-lab assays             ║
║   ✓ Grounded rationales     ✗ A regulatory/clinical decision tool          ║
║   ✓ Auditable agent         ✗ Higher precision than 10 ns MD allows        ║
║                                                                            ║
║  Independence guarantees                                                   ║
║   • Agent tools operate only on data the model did not see                 ║
║   • RAG corpus is label-filtered per evaluated PDB                         ║
║   • Every agent claim must cite a tool output or evidence chunk            ║
╚════════════════════════════════════════════════════════════════════════════╝
```

### 6.5 Shared visual conventions

| Element | Style |
|---|---|
| Recommendation pill | `✓ trust` green; `⚠ review` amber; `✗ discard` red |
| Score bars | Always 4 bars in fixed order: structural / physical / literature / chemical |
| Regex marks | Inline `✓` / `✗` / `?` immediately after each claim sentence |
| Citations | Square-bracketed chunk IDs `[PDBbind_1A1B]`, clickable → side drawer with the chunk text |
| Agent trace step | Numbered, tool name + args + 1-line result |
| Caveats | Always include train/test split, prior-knowledge probe status, label-filter status |
| Currency / latency | Plain text: "18 s, $0.22" — honest about cost |

---

## 7. Tech stack

| Layer | Choice | Rationale |
|---|---|---|
| Frontend framework | **Astro 5.x** (already in `website/`) | Static-first, supports React/Svelte islands for interactivity, reuses the existing site's brand styling |
| Styling | **Tailwind 4** (already in `website/`) | Inherited; consistent with the rest of the site |
| Interactive islands | **Svelte** | Smallest bundle, simplest reactive state for the frame-slider-driven pattern |
| Plots | **Plotly** via CDN inside an island | Native zoom/pan, frame cursor sync |
| 3D viewer | **3Dmol.js** via CDN | Multi-MODEL PDB strings with built-in animation; auto-bonds by distance — no topology required |
| API client | `fetch` in `lib/api.ts` | One source of truth for API base URL and auth headers |
| Inference backend | **FastAPI** | Native async; integrates cleanly with Anthropic SDK tool use |
| Inference deployment | SageMaker async endpoint, fallback Modal | Reuses training infra; pay-per-call for the agent loop |
| Static deployment | Vercel / Netlify / Cloudflare Pages | Free; instant global CDN |

Pin matrix (matches training env where overlapping):

```
# Frontend
astro@^5         tailwindcss@^4    svelte@^5

# Backend
python>=3.12     torch>=2.9        transformers>=4.57
peft>=0.18       fastapi>=0.115    anthropic>=0.40
chromadb>=0.5    mdanalysis>=2.7   rdkit-pypi>=2024.3
h5py>=3.11       numpy>=2.0
```

---

## 8. API contract

### 8.1 Inference

```http
POST /predict
{ "pdb_id": "1A1B", "variant": "v1b" }

→ 200 OK
{
  "pdb_id": "1A1B",
  "variant": "v1b",
  "pK": 6.42,
  "rationale": "During the trajectory ... Answer: 6.42",
  "hidden_pK": 6.41,
  "regex_verifier": {
    "verified": 3, "contradicted": 1, "unverifiable": 0, "total": 4,
    "claims": [{"text": "...", "status": "verified", "evidence": "mean=-37.21"}, ...]
  },
  "latency_ms": 1843,
  "model_version": "v1b-2026-05-22-a1b2c3d"
}
```

```http
POST /predict/batch
{ "pdb_ids": ["1A1B", "1A28", ...], "variant": "v1b" }
→ { "results": [...], "failed": [...] }
```

```http
GET /pdb_string/{pdb_id}?stride=5&drop_water=true
→ 200 OK   text/plain   (multi-MODEL PDB)

GET /pdb_ids
→ ["1A1B", "1A28", ...]   (test split only, 1612 entries)

GET /health
→ { "status": "ready", "variants_loaded": ["v1a", "v1b"], "warm_since": "..." }
```

### 8.2 Evaluation

```http
POST /evaluate              # RAG + LLM judge, no tools, ~3-5 s, ~$0.02
POST /evaluate/agent        # RAG + LLM + tool loop, ~20 s, ~$0.20
```

Both return the structured verdict schema in §9.4 below.

### 8.3 Hard constraints

- **Deterministic.** `temperature=0`. Same input → byte-identical output.
- **Whitelist.** Reject PDB IDs not in `test_MD.txt`. Returns 404, not a wrong prediction.
- **Bounded batch.** Max 50 IDs per `/predict/batch`; 20 per `/evaluate/agent` batch. Larger → 413.
- **Versioned.** `model_version`, `rag_corpus_version`, `judge_model` mandatory in every response.

---

## 9. The agent — tools, loop, guidance

### 9.1 Tool catalog (orthogonal only)

Every tool operates on information the TSLM did **not** see during training (raw atomic coordinates, external physics, external chemistry, or external literature). Tools that just re-derive the four training channels are deliberately excluded — that's the regex verifier's job, not the agent's.

| Tool | Independence source | Cost | Typical use |
|---|---|---|---|
| `lookup_split(pdb_id)` | Metadata | <1 ms | First call always — flags if PDB is train/val/test |
| `actual_pK_lookup(pdb_id)` | Ground truth | <1 ms | Context only; redacted on non-test splits |
| `cluster_poses(pdb_id, k)` | Raw heavy-atom coords | <2 s | Verify "pose stable" vs "ligand drifts" |
| `clash_check(pdb_id, frame_idx)` | Raw coords | <1 s | Rule out broken-frame artifacts |
| `radius_of_gyration(pdb_id, scope)` | Raw coords | <1 s | Pocket-collapse / unfolding claims |
| `hbond_persistence(pdb_id)` | Raw coords + heuristic typing | 2–5 s | Per-bond persistence; surface missing claims |
| `per_residue_contacts(pdb_id)` | Raw coords | 1–3 s | Map ligand atoms → nearest residues |
| `pocket_volume(pdb_id, frame_idx)` | Raw coords (fpocket) | 3–8 s | Sanity check on pocket fit |
| `ligand_descriptors(pdb_id)` | RDKit on SMILES | <1 s | MW, LogP, LE — chemistry the model didn't see |
| `vina_rescore(pdb_id, frame_idx)` | External force field (subprocess) | 5–15 s | Independent per-frame pK estimate |
| `compare_to_static_gnn(pdb_id)` | Orthogonal ML baseline | 1–2 s | Another estimator that didn't see MD |
| `rag_query(query, pdb_id)` | External literature (label-filtered) | <100 ms | Pull knowledge chunks |
| `mmgbsa_estimate(pdb_id, frames)` | External physics (stretch — v3) | 1–3 min | Physics-based ΔG estimate |

### 9.2 The loop

```python
def evaluate_agent(pdb_id, model_pK, rationale, max_steps=8):
    split = lookup_split(pdb_id)
    initial_rag = rag_query(f"binding mode of {pdb_id}",
                            pdb_id=pdb_id, top_k=6)
    system = build_system_prompt()
    user = build_user_prompt(pdb_id, model_pK, rationale, split, initial_rag)
    messages = [{"role": "user", "content": user}]
    trace = []

    for step in range(max_steps):
        resp = client.messages.create(
            model="claude-opus-4-7",
            system=system,
            tools=TOOL_SCHEMAS,
            max_tokens=2048,
            messages=messages,
            extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"},
        )
        messages.append({"role": "assistant", "content": resp.content})

        if resp.stop_reason == "end_turn":
            return parse_verdict(resp), trace

        tool_results = []
        for block in resp.content:
            if block.type != "tool_use":
                continue
            try:
                result = TOOL_REGISTRY[block.name](**block.input)
            except Exception as e:
                result = {"error": str(e)}
            trace.append({"step": step, "tool": block.name,
                          "input": block.input, "result": result})
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": json.dumps(result),
            })
        messages.append({"role": "user", "content": tool_results})

    return {"error": "max_steps_exceeded"}, trace
```

Choices to note: pre-flight RAG before the agent's first turn; step cap at 8; structured trace for the UI; prompt caching on the system prompt + tool schemas.

### 9.3 System prompt (the guidance)

```
ROLE
You evaluate predictions from a trained time-series language model that
predicts protein-ligand binding affinity (pK) from 10 ns MD trajectories.
The model saw four aggregated per-frame channels during training:
ligand RMSD, interaction energy, ligand-protein distance, buried SASA.
It did NOT see raw atomic coordinates, ligand SMILES, or any external
chemistry knowledge.

OBJECTIVE
Decide whether the prediction is defensible from sources the model
could not have used. You are not grading the prediction against ground
truth. You are checking that, given orthogonal evidence, the prediction
and its rationale are consistent.

INDEPENDENCE RULES (hard)
1. Use only the retrieved RAG chunks and the tool outputs in this
   session. Do not use prior knowledge of this PDB, ligand, or target.
2. Every factual claim must cite a tool output or chunk id.
   Uncited claims are discarded.
3. If evidence is insufficient, say "insufficient evidence" — do not guess.
4. If actual_pK_lookup returns "[redacted]", the PDB is not in the test
   split — flag this and note that low error is uninformative.

PROCESS
Plan first. Your first message must list:
  (a) claims extracted from the rationale,
  (b) which tool you will call for each, and why,
  (c) what RAG queries you will make.
Then execute. Max 8 tool calls total. Do not duplicate work the regex
verifier already does.

OUTPUT FORMAT
End with a single JSON object matching the schema in §9.4.
```

### 9.4 Output schema

```json
{
  "scores": {
    "structural_consistency": 0.85,
    "physical_consistency":   0.72,
    "literature_consistency": 0.90,
    "chemical_plausibility":  0.65
  },
  "verified_claims":     [{"claim": "...", "evidence": "..."}],
  "contradicted_claims": [{"claim": "...", "contradicting_evidence": "..."}],
  "missing_claims":      [{"evidence": "...", "why_relevant": "..."}],
  "recommendation":      "trust",
  "citations":           [{"chunk_id": "...", "score": 0.91}],
  "independence_caveats": ["train/test split: test", "prior-knowledge probe: unknown", "..."],
  "judge_model":           "claude-opus-4-7",
  "rag_corpus_version":    "v1-2026-05-22",
  "tool_versions":         {"vina": "1.2.5"},
  "agent_trace": {
    "tool_calls": 6, "latency_ms": 18430,
    "input_tokens": 8421, "output_tokens": 1108
  }
}
```

---

## 10. RAG — corpus and label-filtering

### 10.1 Corpus

| Source | Approx chunks | Why included |
|---|---|---|
| PDBbind per-system binding-mode annotations | ~20k | System-specific evidence the regex can't represent |
| UniProt summaries for target proteins | ~2k | Target-family context |
| MISATO paper (chunked) | ~50 | Dataset-level constraints |
| OpenTSLM paper (chunked) | ~50 | Method-level expectations |
| PubMed abstracts mentioning the PDB | ~5k | Literature comparison |
| `PROJECT_BRIEF.md` + `FRONTEND_v2.md` | ~30 | The agent knows the 10 ns ceiling, the 5-claim vocab, etc. |
| **Total** | **~27k chunks** | Fits in Chroma / Qdrant / pgvector |

### 10.2 Three rules

1. **Label tagging at ingest.** Every chunk that mentions a numerical Kd / Ki / IC50 / pK is tagged `contains_label: true` plus the PDB-IDs the value pertains to.
2. **Label filtering at query time.** `rag_query(pdb_id="X")` excludes chunks where `contains_label=true AND X ∈ chunk.pdb_ids`. Other label-bearing chunks (about *different* PDBs) remain — those are comparators, not leaks.
3. **PDB-ID-first retrieval.** Top-k=8 with priority: chunks tagged with the query PDB first, then target-family chunks, then general. Falls through if nothing system-specific exists.

### 10.3 Retrieval entry point

```python
def rag_query(query: str, pdb_id: str, top_k: int = 8) -> list[Chunk]:
    candidates = vector_store.search(query, top_k * 3)
    filtered = [c for c in candidates
                if not (c.contains_label and pdb_id in c.pdb_ids)]
    return rerank(filtered, by_pdb_id=pdb_id)[:top_k]
```

Same signature as any other tool — RAG retrievals appear in the agent trace alongside physics tool calls.

---

## 11. Independence guarantees

Independence is engineered, not assumed. Five concrete mechanisms:

| Risk | Mechanism |
|---|---|
| Agent uses training-channel tools (circular) | Such tools removed from the registry entirely |
| Agent looks up the answer in RAG | `contains_label` filter strips Kd/Ki/IC50 chunks for the system under test |
| Agent uses Claude's prior knowledge | System prompt forbids it; every claim must cite a chunk or tool result |
| Agent doesn't realize this PDB is in train | `lookup_split` mandatory; `actual_pK_lookup` redacts on non-test |
| Agent rationalizes post-hoc | "Plan first" rule forces claim enumeration before tool results arrive |

These are the **per-prediction** independence guarantees. They do not establish *generalization* — that requires dataset-level experiments (train/test gap, out-of-target split, permutation test on rationales, human spot-check) reported in the writeup, not enforced in the UI.

The Caveats block in the Single-tab agent panel exposes the status of each mechanism per call, so visitors see independence as an audit trail, not a brand promise.

---

## 12. HDF5 → PDB reconstruction (moved to backend)

The logic from v1 §6 is unchanged, but moved from the UI process to a `GET /pdb_string/{pdb_id}` endpoint on the inference service.

```
trajectory_coordinates      (100, N_atoms, 3)
atoms_element               (N_atoms,)
atoms_residue               (N_atoms,)
molecules_begin_atom_index  (3,)
```

Synthesize a multi-MODEL PDB:

```
ATOM  {serial:5d}  {name:<4s}{resname:>3s} {chain:1s}{resseq:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2s}
```

- chain `"A"` for protein (atoms < `molecules_begin_atom_index[1]`), `"L"` for ligand (between `[1]` and `[2]`), `"W"` for water (>= `[2]`, usually dropped).
- `resname = "UNK"` if not available; fallback to `line`/`stick` rendering if `cartoon` fails.
- Drop waters; stride every 5th frame; coord precision 2 decimals — keeps payload < 5 MB per system.

Reference implementation in `inference-service/hdf5_to_pdb.py` — same code as Appendix A of v1.

---

## 13. Component breakdown

```
website/src/                       (Astro + Tailwind, existing project)
  pages/demo/
    single.astro
    batch.astro
    failure-modes.astro
    about.astro
  components/demo/
    PdbPicker.svelte
    VariantToggle.svelte
    PredictionCard.svelte
    RationaleWithBadges.svelte
    AgentVerdictPanel.svelte
    AgentTraceExpander.svelte
    ScoreBars.svelte
    RecommendationPill.svelte
    StructureViewer.astro
    ChannelPlot.svelte
    FrameSlider.svelte
    BatchTriageTable.svelte
    FailureModesTable.svelte
    CitationDrawer.svelte
  lib/
    api.ts                         # fetch wrapper, base URL, retries
    pdb_ids.json                   # built from test_MD.txt at site-build
    failure_modes.json             # precomputed, static
  data/
    worked_examples.json           # 3 precomputed agent traces

inference-service/                 (FastAPI, separate repo or subfolder)
  app.py                           # FastAPI routes
  inference.py                     # trained model load + predict()
  hdf5_to_pdb.py                   # /pdb_string endpoint
  verifier.py                      # regex verifier
  orchestrator.py                  # agent loop
  prompts/
    system.md                      # the §9.3 system prompt
    user_template.md
  tools/
    __init__.py                    # registry
    splits.py
    coords.py                      # cluster_poses, clash_check, ...
    physics.py                     # vina_rescore, mmgbsa_estimate
    chemistry.py                   # ligand_descriptors
    rag.py                         # rag_query
  rag/
    corpus_v1/                     # PDBbind, UniProt, PubMed, papers
    ingest.py                      # chunking + label tagging
    store.py                       # Chroma wrapper
  Dockerfile
  requirements.txt
```

The reusable Svelte components (`ScoreBars`, `RecommendationPill`, `AgentTraceExpander`, `CitationDrawer`) appear on every tab — they're the brand.

---

## 14. Schedule (phased)

The original 20-hour sprint cannot fit custom-UI + API + RAG + agent. Phased plan:

| Phase | Hours | Ships |
|---|---|---|
| **Phase 1** — sprint, H10–H20 (within original 20 h budget) | 10 | Astro `demo/single` page; deployed FastAPI service; `/predict`, `/predict/batch`, `/pdb_string`, `/health`; regex verifier; 3D viewer; channel plot; 3 worked examples linked from About |
| **Phase 2** — post-sprint week 1 | ~12 | RAG corpus (v1) + label filter + `/evaluate` endpoint; `AgentVerdictPanel` (no tools yet, judge only); precomputed evaluations for ~50 representative systems baked into the static site |
| **Phase 3** — post-sprint week 2 | ~14 | Agentic tools (6 in-process + Vina) + `/evaluate/agent`; Batch tab; Failure modes tab; agent trace JSON for the 10 failure-mode entries baked in; rate limiting + daily Anthropic cap |
| **Phase 4** — paper-grade extensions | ~20 | MM-GBSA; out-of-target retrain validation; permutation-test on rationales; human spot-check protocol with 20 chemist-graded systems |

Phase 1 hits the original 20-hour brief deliverable. Each later phase ships independently and the demo stays live throughout.

### 14.1 Phase 1 detail (replaces v1 §8 schedule)

| Hours | Work |
|---|---|
| H10–H12 | `demo/single.astro` page + Tailwind layout + `PdbPicker` + `VariantToggle` + `api.ts` |
| H12–H15 | `PredictionCard` + `RationaleWithBadges` against mocked API |
| H15–H17 | `StructureViewer` (3Dmol embed) + `ChannelPlot` + `FrameSlider` syncing both |
| H17–H19 | Deploy FastAPI inference service with real checkpoints, swap mock for live API |
| H19–H20 | Deploy Astro site, smoke test, record screen captures |

---

## 15. Risks and limits

1. **3Dmol fails on malformed PDB.** Same as v1 §10.1 — unit-test `hdf5_to_pdb` on `tiny_md.hdf5`; fallback to static frame-0/50/99 triptych.
2. **Cold-start time on the inference API.** ~30 s on t4 to load Llama-3.2-1B + LoRA + encoder + both variants. Mitigation: visible "loading model…" splash; warm-keep instance during demo windows.
3. **Anthropic API cost runaway.** Each `/evaluate/agent` call ~$0.20–0.50. Mitigation: precompute the 3 worked examples + 10 failure-mode entries + ~50 representative systems so most clicks hit cached JSON. Live agent button gated behind a "this will spend ~$0.30" modal *or* daily cap.
4. **RAG label-filter failure.** If a PDBbind chunk with a leaked Kd slips through, the agent "validates" by reading the answer. Mitigation: unit-test the filter on 50 hand-curated leaky chunks before deploy; CI assertion blocks regression.
5. **CORS / origin pinning.** Static site origin ≠ API origin. Lock API's CORS allowlist to production site only.
6. **Out-of-vocabulary PDB IDs.** Users can only pick from the test split; no upload-your-own-MD endpoint.
7. **AI evaluating AI bias.** Claude judging a TSLM may share training-distribution biases. Mitigation: §11 mechanisms + the human spot-check planned for Phase 4. Without that anchor, the agent's "trust" is suggestive, not authoritative.
8. **Phase 2/3 slippage.** Each post-sprint phase is independent — the Phase 1 site stays live regardless. The risk is the writeup waiting on Phase 3; mitigate by writing the 3 worked-example narrative around the Phase 2 capabilities first.

---

## 16. What this UI does NOT do

Scoped out deliberately:

- No user accounts, no saved prediction history.
- No upload-your-own-MD endpoint (needs queued AMBER prep — out of scope).
- No per-residue claim highlighting in the 3D viewer (5-claim vocabulary is whole-trajectory; per-residue is Phase 4).
- No mobile-optimized layout. 3D viewer is desktop-first; phones get a "best viewed on desktop" notice.
- No fine-tuning UI, no "retrain on my data" button. Read-only.
- No generalization claims in the UI — those are in the writeup, anchored on dataset-level experiments.
- No human-in-the-loop review UI for the chemist spot-check; that protocol runs offline in Phase 4.
- The agent does not estimate ground-truth pK itself — it would just be another LLM guessing.

---

## 17. Open decisions

1. **Hosting org for the static site.** Personal subdomain vs new team org? Recruiter-facing means it should look institutional.
2. **Caching strategy for agent results.** In-memory (resets on API restart) vs persistent JSON sidecar vs Redis. Recommend persistent JSON to disk — deterministic predictions justify it.
3. **One service hosting both v1a and v1b, or two?** Llama-3.2-1B + LoRA fits easily in 16 GB; recommend one process with both checkpoints loaded.
4. **Auth on the inference API.** Public unauthed (matches the public site) or bearer token? Public is fine for the demo; a token is cheap insurance.
5. **Telemetry.** Anonymous click counts on which PDB IDs / which agent runs visitors trigger would inform Phase 4. Conflicts with the "no tracking" stance. Default: none.
6. **MM-GBSA in Phase 3 or Phase 4.** Phase 3 ships the agentic harness without it; Phase 4 adds MM-GBSA once the harness is validated.

---

## 18. Deliverables

**Phase 1 (H20):**
- Live Astro `/demo/single` page talking to a deployed FastAPI service.
- Inference on the 1,612-PDB test split, regex-verified rationales rendered with badges.
- Three worked examples linked from About tab.
- Three short screen recordings: frame slider driving 3D + plot; v1a vs v1b toggle; regex catching a contradicted claim.

**Phase 2 (week 1 post-sprint):**
- RAG corpus v1 deployed, label-filter verified by CI.
- `AgentVerdictPanel` live on Single tab.
- `/evaluate` precomputed for ~50 representative systems, baked into the static site.

**Phase 3 (week 2 post-sprint):**
- Six in-process tools + `vina_rescore` available; `/evaluate/agent` live.
- Batch and Failure modes tabs live.
- Agent trace JSON for the 10 failure-mode entries baked in.
- Anthropic spend cap + rate limiting.

**Phase 4 (paper):**
- MM-GBSA tool added.
- Out-of-target retrain validation.
- Permutation-test on rationales.
- Human spot-check on 20 systems with chemist grading, used to calibrate the agent's `recommendation` thresholds.

---

## Appendix A — Tool JSON schema example

```python
@register({
    "name": "cluster_poses",
    "description": "Cluster the 100 trajectory frames in heavy-atom coordinate "
                   "space; returns cluster sizes and inter-cluster RMSD.",
    "input_schema": {
        "type": "object",
        "properties": {
            "pdb_id": {"type": "string"},
            "k":      {"type": "integer", "default": 3, "minimum": 2, "maximum": 5},
        },
        "required": ["pdb_id"],
    },
})
def cluster_poses(pdb_id: str, k: int = 3) -> dict:
    coords = load_heavy_atom_coords(pdb_id)        # (100, N_heavy, 3)
    flat = coords.reshape(100, -1)
    labels = KMeans(n_clusters=k, n_init=10).fit_predict(flat)
    sizes = np.bincount(labels, minlength=k).tolist()
    medoids = compute_medoids(coords, labels)
    rmsd_matrix = pairwise_rmsd(medoids)
    return {
        "cluster_sizes":      sizes,
        "dominant_cluster":   int(sizes.index(max(sizes))),
        "rmsd_between_medoids": rmsd_matrix.tolist(),
    }
```

The same registration pattern applies to all 13 tools — JSON schema first, function body second, registered into `TOOL_REGISTRY` and `TOOL_SCHEMAS` at import time.

## Appendix B — Links

- `PROJECT_BRIEF.md` (parent) — `./PROJECT_BRIEF.md`
- `FRONTEND.md` (v1, Gradio plan, frozen) — `./FRONTEND.md`
- Astro docs — <https://docs.astro.build>
- Anthropic tool use — <https://docs.claude.com/en/docs/build-with-claude/tool-use>
- 3Dmol.js — <https://3dmol.csb.pitt.edu/>
- MISATO repo — <https://github.com/t7morgen/misato-dataset>
- OpenTSLM repo — <https://github.com/StanfordBDHG/OpenTSLM>

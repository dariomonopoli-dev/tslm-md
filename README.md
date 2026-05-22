# Trajecta

**Reading the movie of how a drug wiggles in a binding pocket, then auditing what the model claims about it.**

Submission for **Colosseum Idearum 2026**. A demo-grade artifact that applies
[OpenTSLM-SoftPrompt](https://arxiv.org/abs/2510.02410) (Stanford BDHG et al., ICML 2026)
to a modality it has not been tried on: protein–ligand molecular dynamics trajectories
from [MISATO](https://zenodo.org/record/7711953), labelled with the `misato-affinity`
companion dataset.

The model reads the four-channel, 100-frame trajectory of a ligand inside its pocket
and outputs two things:

1. A predicted binding affinity (pK).
2. A natural-language **rationale** that explains the dynamics it saw, with every
   factual claim verifiable against either the raw input channels or an independent
   evidence source.

An autonomous agent (Claude Opus 4.7 via OpenRouter) then audits each rationale before
it ships, using orthogonal tools the trained model never touched: raw atomic
coordinates, AutoDock Vina rescoring, ligand chemistry descriptors, and a
label-filtered RAG corpus.

> **Honest scope.** This is a hackathon artifact, not a production drug-discovery
> tool. 10 ns of MD limits affinity resolution to roughly ±0.3 pK. The agent does
> not replace wet-lab assays.

---

## What you get

### 1. A trained TSLM, two variants

| Variant | Description | Status |
|---|---|---|
| **v1a (faithful)** | Pure OpenTSLM-SP exactly as published. Affinity is the `Answer: X.XX` suffix of the generated rationale string. | Trainer ready |
| **v1b (hybrid)** | Adds a 2-layer MLP regression head on the LLM's last input-position hidden state. Joint loss `L = L_LM + λ · MSE(pK_pred, pK_true)`. | Trainer ready, smoke-tested |

The v1a vs v1b ablation answers the obvious reviewer question: *does extending the
published method actually help?* Either outcome is publishable.

### 2. An auditing agent with three independence guarantees

1. Uses only tools that operate on data the trained model did not see (raw atomic
   coordinates, external force fields, label-filtered RAG).
2. Cannot look up the answer. RAG chunks containing the system's experimental Kd
   are excluded at retrieval time for the system under test.
3. Refuses to use prior knowledge. Every factual claim must cite either a tool
   output or a retrieved evidence chunk.

### 3. An interactive UI to inspect, triage, and stress-test

Five views:
- **Overview**: editorial landing with three worked examples.
- **Inspect**: pick a PDB, see the prediction, the rationale, the verifier marks,
  the 3D structure animation, and the per-channel charts.
- **Triage**: batch evaluation over many systems with CSV export.
- **Failure modes**: precomputed cases where the model is confidently wrong, with
  the agent's catch.
- **Knowledge**: drag-and-drop RAG upload, source list.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                          USER BROWSER                            │
│   nginx → React 19 + Vite 6 + Tailwind v4 bundle on :3000        │
└──────────────────────────────┬───────────────────────────────────┘
                               │  /api/* (same-origin proxy)
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│                  FastAPI (inference service) :8000               │
│                                                                  │
│   /predict, /predict/batch, /pdb_string, /pdb_ids, /health       │
│   /evaluate, /evaluate/agent, /failure_modes                     │
│                                                                  │
│   Components:                                                    │
│     · TSLM v1a + v1b loader (OpenTSLMSP)                         │
│     · Deterministic regex rationale verifier                     │
│     · MISATO HDF5 to multi-MODEL PDB reconstruction              │
│     · Agent orchestrator (Claude Opus 4.7, 8-step loop)          │
│     · 9 tools: splits, coords, chemistry, physics (Vina), rag    │
│     · Embedded ChromaDB + OpenAI text-embedding-3-small          │
│     · Persistent eval cache + daily USD cap                      │
└──────────────────────────────┬───────────────────────────────────┘
                               │  (mode B only)
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│              SageMaker endpoint (optional)                       │
│   v1a + v1b checkpoints behind boto3 invoke_endpoint             │
│   ml.g5.xlarge realtime, or async with scale-to-zero             │
└──────────────────────────────────────────────────────────────────┘
```

Switching between local-GPU and SageMaker is one env var change. See
[STACK.md](STACK.md) for the operating guide.

---

## Stack

### Frontend (`trajecta/`)

React 19 + Vite 6 + Tailwind v4. Talks to the FastAPI service at `/api`
(same-origin proxied by nginx in prod, by Vite in dev).

```
trajecta/src/
  App.tsx                  five-tab router with lifted PDB + variant state
  index.css                Tailwind v4 + design tokens + animations
  types.ts                 wire types (mirror Pydantic schemas)
  lib/
    api.ts                 typed fetch client, AbortController, ApiResult union
    utils.ts               cn() helper (clsx + tailwind-merge)
  components/
    ui.tsx                 RecommendationPill, ScoreBar, VerifierMark, Citation, AgentTrace
    StructureViewer.tsx    3Dmol.js multi-MODEL animation (auto-rotates)
    Brand.tsx              animated Trajecta wordmark + atom glyph
    BackgroundFX.tsx       drifting gradient mesh + dot grid + grain
    AnimatedNumber.tsx     smooth count-up tween
    MoleculeHero.tsx       cinematic auto-rotating hero molecule
    GlowCard.tsx           surface card with gradient border + bloom
  views/
    SingleView.tsx         picker + predict + agent panel + 3D viewer + channels
    BatchView.tsx          multi-select + parallel /evaluate fan-out + CSV export
    FailureModesView.tsx   precomputed JSON view
    AboutView.tsx          editorial landing with bento composition
    KnowledgeView.tsx      drag-drop RAG upload + source list
```

### Inference service (`inference-service/`)

FastAPI orchestrator. Serves the TSLM, runs the regex verifier, reconstructs PDB
trajectories from MISATO HDF5, and runs the independent agent loop.

```
inference-service/
  app.py                   FastAPI routes
  inference.py             TSLM loader + predict()
  verifier.py              Deterministic regex rationale verifier
  hdf5_to_pdb.py           MISATO HDF5 to multi-MODEL PDB
  orchestrator.py          Agent loop (Claude Opus 4.7 via OpenRouter)
  tools/
    splits.py              lookup_split, actual_pK_lookup
    coords.py              cluster_poses, clash_check, hbond_persistence, per_residue_contacts
    chemistry.py           ligand_descriptors
    physics.py             vina_rescore
  rag/
    ingest.py              One-shot corpus build
    store.py               rag_query with label filter
  llm/
    openrouter.py          Anthropic-compatible OpenRouter client
    pricing.py             Local spend computation
  prompts/
    system.md              Agent system prompt
    user_template.md       per-call template
```

### Model + data pipeline

```
preprocess_misato.py       MISATO HDF5 + affinity CSV → 41 MB SageMaker bundle
train_misato.py            standalone trainer (v1a + v1b)
eval_baselines.py          predict-mean, OLS, MLP (must-beat OLS RMSE 1.78)
verify_rationale.py        offline regex verifier over generated rationales
sagemaker-deploy/          model.tar.gz builder + endpoint deploy
OpenTSLM/                  forked from StanfordBDHG, branch misato-md-affinity
configs/                   training and inference config files
preprocessed/              SageMaker-ready features + samples + norm stats
```

---

## How it works, in one picture

```
MD trajectory (100 frames per system)
        │
        ▼  load 4 precomputed per-frame channels from MISATO HDF5:
            · frames_rmsd_ligand        (100,)  Å
            · frames_interaction_energy (100,)  kcal/mol
            · frames_distance           (100,)  Å (CoM-CoM)
            · frames_bSASA              (100,)  Å²
        │
        ▼  per-channel normalized univariate time-series
        │
   OpenTSLM-SoftPrompt
   (Llama-3.2-1B, frozen + LoRA on q/k/v/o + MLP projections)
        │
        ▼  generates: "<rationale> Answer: 6.42"
        │
   ┌────┴───────────────────────────┐
   ▼                                ▼
 affinity (parse Answer:)      rationale → regex verifier
                                    │
                                    ▼
                               agent loop (Claude Opus 4.7)
                                    │
                                    ▼  uses orthogonal tools only:
                               splits · coords · chemistry · Vina · RAG
                                    │
                                    ▼
                          verdict: trust / unsure / reject (with citations)
```

No new architecture. The contribution is the **modality** (MD trajectories), the
**grounded rationale loop** (every claim checkable against the input or against
an independent source), and the **regression-head ablation** (v1a vs v1b).

---

## Quickstart

> Two deploy modes share everything except where `/predict` runs the model.
> See [STACK.md](STACK.md) for full details and switching commands.

### Mode A: local GPU

```bash
# 1. Configure
cp .env.example .env
# Fill in OPENROUTER_API_KEY, OPENAI_API_KEY, HUGGING_FACE_HUB_TOKEN

# 2. Make sure the inference container can see:
#    ./MD.hdf5                                       124 GB MISATO trajectory
#    ./misato-affinity/data/affinity_data.csv        labels
#    ./preprocessed/features_test.npz                training features
#    ./preprocessed/samples_test.jsonl               per-PDB facts (verifier)
#    ./checkpoints/v1a/ckpt_ep1.pt                   trained TSLM (v1a)
#    ./checkpoints/v1b/ckpt_final.pt                 trained TSLM (v1b)
# Anything missing puts the inference service into degraded mode (visible in /health).

# 3. Boot the stack
make up
make logs
make ps

# 4. One-time RAG corpus build (~$0.50 in OpenAI embeddings, ~3 min)
make ingest

# 5. Optional: precompute worked examples + failure modes (~$15, ~30 min)
make precompute

# 6. Open http://localhost:3000
```

### Mode B: SageMaker (recommended for demos)

See [`sagemaker-deploy/README.md`](sagemaker-deploy/README.md). Three commands in
Code Editor, then flip `INFERENCE_BACKEND=sagemaker` in `.env` and `make restart`.

### Frontend dev only

```bash
cd trajecta
npm install
npm run dev          # serves on :3000, proxies /api → http://localhost:8000
npm run lint         # tsc --noEmit
npm run build        # vite build → dist/
```

If the backend is elsewhere:

```bash
DEV_API_URL=http://192.168.1.10:8000 npm run dev
# or for a prod build:
VITE_API_BASE_URL=https://api.trajecta.dev npm run build
```

---

## Environment variables

Create a `.env` in the project root:

```
# LLM judge (Claude Opus 4.7 via OpenRouter)
OPENROUTER_API_KEY=sk-or-...

# RAG embeddings (OpenAI text-embedding-3-small)
OPENAI_API_KEY=sk-...

# Hugging Face (Llama-3.2-1B is gated)
HUGGING_FACE_HUB_TOKEN=hf_...

# Where to run /predict
INFERENCE_BACKEND=local            # or "sagemaker"

# Mode B only:
SAGEMAKER_ENDPOINT_NAME=trajecta-tslm
SAGEMAKER_REGION=us-west-2
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
```

OpenRouter routes to Claude Opus 4.7 using the Anthropic Messages API shape (clean
tool-use round-trips). OpenAI's `text-embedding-3-small` is the cheapest competent
embedding model for the RAG corpus. The two are not interchangeable; do not wire
OpenAI as the judge.

---

## Data

| Source | Size | Role |
|---|---:|---|
| [MISATO](https://zenodo.org/record/7711953) `MD.hdf5` | 124 GB | 16,972 protein–ligand systems × 100 MD frames × 4 channels |
| `misato-affinity/data/affinity_data.csv` | 1.2 MB | 19,443 rows of Kd, Ki, IC50 (nM). `pK = 9 - log10(nM)`. Priority Kd > Ki > IC50. |
| Official splits | 84 KB | 13,765 train / 1,595 val / 1,612 test PDB IDs |

After preprocessing the full 124 GB file ships only a 41 MB SageMaker-ready bundle:
features (clipped, normalized), templated rationales, per-system facts, splits.

See [DATASET.md](DATASET.md) for the full data audit and [TRAINING.md](TRAINING.md)
for what `train_misato.py` does end-to-end.

---

## Baselines that must be beaten

Computed by `eval_baselines.py` (6 s on CPU) from `preprocessed/features_*.npz`.

| Baseline | What it does | Test RMSE | Bar |
|---|---|---:|---|
| `predict_train_mean` | Constant = `train_pK.mean()` | 1.933 | reference floor |
| `ols_means` | OLS on 4 trajectory-mean channels | **1.778** | the trajectory encoder must do *something* |
| `mlp_engineered` | MLP on 20 hand-features per channel (mean/std/slope/min/max × 4) | **1.680** | the encoder must do something *beyond summary stats* (i.e. actually read the movie) |

---

## Documentation index

The detailed operational docs live alongside this README:

| File | What's in it |
|---|---|
| [PROJECT_BRIEF.md](PROJECT_BRIEF.md) | Full project spec, method, channels, ablations |
| [DATASET.md](DATASET.md) | Data audit: coverage, distributions, outlier handling |
| [TRAINING.md](TRAINING.md) | Step-by-step training reference (CLI flags, env vars, hardware, pre-flight) |
| [STACK.md](STACK.md) | Two deploy modes, switching, troubleshooting |
| [FRONTEND.md](FRONTEND.md) | Frontend wire contract + view-by-view spec |
| [LEARNINGS.md](LEARNINGS.md) | Things we got wrong and what we learned |
| [trajecta/README.md](trajecta/README.md) | Frontend dev setup |
| [inference-service/README.md](inference-service/README.md) | Backend dev setup |
| [sagemaker-deploy/README.md](sagemaker-deploy/README.md) | SageMaker deploy walkthrough |
| [docs/yc-pitch-brief.md](docs/yc-pitch-brief.md) | 1-page pitch summary |
| [docs/hackathon-plan-19h.md](docs/hackathon-plan-19h.md) | The 19-hour build plan |

---

## Honest limits

1. **10 ns ceiling.** Experimental ΔG reflects equilibrium; 10 ns cannot fully
   recover it. Hard floor on Pearson R from this.
2. **Train-vs-test pK shift.** Train mean pK 6.59, test 5.55. Calibrate on val,
   not train.
3. **bSASA artifacts.** Around 600 systems have physically impossible bSASA values
   (negative or > 2,500 Å²); we clip to `[0, 2500]`. Median 503 Å², p99 1,250 Å².
4. **MISATO selection bias.** The 17 K systems were stable enough to simulate.
   Unstable or unbindable systems were filtered upstream.
5. **Rationale verifier has a closed vocabulary.** Free-form generation can produce
   claims outside the 5 verifiable types; those are marked `unverifiable`, not
   contradicted. Percent verified is reported over **grounded** claims only.
6. **Topology not in input.** Per-residue claims would require AMBER topology
   files (only `11gs` is local). Claims are scoped to the 4 shipped channels.

---

## Disclaimer

For informational, demonstrative, and research purposes only. Not a regulatory or
clinical decision tool. Not a replacement for wet-lab assays. Treat it as a triage
layer.

---

## Credits

- **OpenTSLM-SoftPrompt:** Stanford BDHG et al., [arXiv:2510.02410](https://arxiv.org/abs/2510.02410), ICML 2026.
- **MISATO:** [Siebenmorgen et al., 2024](https://zenodo.org/record/7711953).
- **Llama-3.2-1B:** Meta (gated, requires approval).
- **Claude Opus 4.7:** Anthropic, routed via OpenRouter.
- Built for **Colosseum Idearum 2026**.

# Trajecta — frontend

React 19 + Vite 6 + Tailwind v4. Talks to the FastAPI inference service at
`/api` (same-origin proxied by nginx in prod, by Vite in dev).

## Dev

```bash
npm install
npm run dev          # serves on :3000, proxies /api → http://localhost:8000
npm run lint         # tsc --noEmit
npm run build        # vite build, emits dist/
```

If your backend is elsewhere:

```bash
DEV_API_URL=http://192.168.1.10:8000 npm run dev
# or in prod-build mode:
VITE_API_BASE_URL=https://api.trajecta.dev npm run build
```

## Layout

```
src/
  App.tsx                       five-tab router with lifted PDB + variant state
  main.tsx                      React 19 entry
  index.css                     Tailwind v4 + dark design system + animations
  vite-env.d.ts                 import.meta.env typing
  types.ts                      §8 / §9.4 wire types (matches backend exactly)
  data.ts                       MOCK_CHART_DATA + ALPHABET_PDBS (mock, kept for chart placeholder)
  lib/
    api.ts                      typed fetch client, AbortController, ApiResult union
    utils.ts                    cn() helper (clsx + tailwind-merge)
  components/
    ui.tsx                      RecommendationPill, ScoreBar, VerifierMark, Citation, AgentTrace
    StructureViewer.tsx         3Dmol.js multi-MODEL animation (auto-rotates)
    Brand.tsx                   animated Trajecta wordmark + atom glyph
    BackgroundFX.tsx            drifting gradient mesh + dot grid + grain
    AnimatedNumber.tsx          smooth count-up tween for predicted values
    MoleculeHero.tsx            cinematic auto-rotating hero molecule
    GlowCard.tsx                surface card with gradient border + bloom
  views/
    SingleView.tsx              picker + predict + agent panel + 3D viewer + channels
    BatchView.tsx               multi-select + parallel /evaluate fan-out + CSV export
    FailureModesView.tsx        precomputed JSON view
    AboutView.tsx               cinematic landing hero + bento "how it works"
    KnowledgeView.tsx           drag-drop RAG upload + source list
```

## Backend contract

See the inference-service README — every type in `src/types.ts` mirrors a
Pydantic schema in `inference-service/app.py` and `orchestrator.py`.

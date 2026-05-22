# Trajecta stack — operating guide

## Two deploy modes for the TSLM

Both modes share the FastAPI orchestrator + agent loop + RAG + frontend; they
differ only in where `/predict` actually runs the trained model.

| Mode | `INFERENCE_BACKEND` | Model runs on | Setup complexity | Cost when idle | UX |
|---|---|---|---|---|---|
| **A — local** (default) | `local` | The inference container (needs a GPU host + mounted `./checkpoints/`) | Low — just `make up` | $0 (your own GPU) | ~2-3 s predict |
| **B — SageMaker** (recommended for demos) | `sagemaker` | A SageMaker endpoint deployed via `sagemaker-deploy/` | Medium — 3 commands in Code Editor | ~$0/hr async, ~$1.21/hr realtime g5.xlarge | ~2-3 s realtime; async adds polling |

The local FastAPI container is identical in both modes — same image, same
routes, same agent loop. The only thing that changes is what `/predict`
does internally: load + forward locally, or `boto3.invoke_endpoint` against
the SageMaker endpoint.

**Switching modes is one env-var change + restart:**

```bash
# Mode A → Mode B
sed -i 's/INFERENCE_BACKEND=local/INFERENCE_BACKEND=sagemaker/' .env
echo SAGEMAKER_ENDPOINT_NAME=trajecta-tslm >> .env
make restart
```

`/health` reports `inference_backend` + endpoint info so the frontend can
warn if the SM endpoint is misconfigured.

## Architecture

The demo is two Docker services:

```
┌──────────────────────────────────────────────────────────────────┐
│                       USER BROWSER                              │
│   nginx (trajecta/frontend) → static React bundle on :3000    │
└──────────────────────────┬──────────────────────────────────────┘
                           │  /api/* (same-origin proxy)
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│              FastAPI (trajecta/inference) :8000               │
│                                                                  │
│   /predict, /predict/batch, /pdb_string, /pdb_ids, /health      │
│   /evaluate, /evaluate/agent, /failure_modes                    │
│                                                                  │
│   Components:                                                    │
│     - TSLM v1a + v1b loader (OpenTSLMSP)                        │
│     - Regex verifier (vendored verify_rationale.py)             │
│     - HDF5 → multi-MODEL PDB reconstruction                     │
│     - Agent orchestrator (Claude via OpenRouter, 8 step loop)   │
│     - 9 tools: splits, coords, chemistry, physics (Vina), rag   │
│     - Embedded ChromaDB + OpenAI text-embedding-3-small         │
│     - Persistent eval cache + daily USD cap                     │
└──────────────────────────────────────────────────────────────────┘
```

## First-time setup (Mode A — local checkpoints)

1. Copy `.env.example` to `.env` and fill in real keys:
   ```bash
   cp .env.example .env
   # edit OPENROUTER_API_KEY, OPENAI_API_KEY, HUGGING_FACE_HUB_TOKEN
   ```

2. Make sure the assets the inference container mounts actually exist locally:
   ```
   ./MD.hdf5                              124 GB MISATO trajectory
   ./misato-affinity/data/Maps/           atoms_*_map.pickle files
   ./misato-affinity/data/affinity_data.csv
   ./preprocessed/features_test.npz       per-channel training features
   ./preprocessed/samples_test.jsonl      per-PDB facts for the verifier
   ./checkpoints/v1a/ckpt_ep1.pt          (or ckpt_final.pt) — the trained TSLM
   ./checkpoints/v1b/ckpt_final.pt
   ```

   Anything missing will cause the inference container to boot in `degraded`
   mode (visible in `/health`) — endpoints that need the missing data return
   503 with a clear message; the rest of the stack keeps running.

3. Boot the stack:
   ```bash
   make up                # builds both images, starts in background
   make logs              # tail logs
   make ps                # confirm both services healthy
   ```

4. One-time RAG corpus build (cost: ~$0.50 in OpenAI embeddings, ~3 min):
   ```bash
   make ingest
   ```

5. Optional: precompute worked examples + failure modes so the live UI mostly
   serves cached responses (cost: ~$15, ~30 min):
   ```bash
   make precompute
   ```

6. Open the frontend at <http://localhost:3000>.

## First-time setup (Mode B — SageMaker endpoint)

Full step-by-step in **`sagemaker-deploy/README.md`**. TL;DR from a Code
Editor terminal in SageMaker Studio:

```bash
cd sagemaker-deploy
python build_model_tarball.py \
    --v1a-ckpt /opt/ml/checkpoints/v1a/ckpt_ep1.pt \
    --v1b-ckpt /opt/ml/checkpoints/v1b/ckpt_final.pt \
    --preprocessed /home/sagemaker-user/preprocessed \
    --s3-uri s3://<your-bucket>/trajecta/model.tar.gz

python deploy.py \
    --model-data s3://<your-bucket>/trajecta/model.tar.gz \
    --endpoint-name trajecta-tslm \
    --mode realtime
```

Then on the machine running this stack:

```bash
# .env additions
INFERENCE_BACKEND=sagemaker
SAGEMAKER_ENDPOINT_NAME=trajecta-tslm
SAGEMAKER_REGION=us-west-2
# Either provide explicit AWS keys OR run the host with an instance role that
# has 'sagemaker:InvokeEndpoint' permission on the endpoint ARN.
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...

# Then:
make restart inference
make smoketest      # verifies /predict round-trips to SageMaker
```

The local container does NOT need OpenTSLM, a GPU, or the checkpoint mount in
SageMaker mode — those concerns move into the SM container. The MD.hdf5
mount is still required because `/pdb_string` (the 3D viewer trajectory)
runs locally.

When you're done:

```bash
cd sagemaker-deploy
python deploy.py --endpoint-name trajecta-tslm --delete
```

## Health check + smoke test

```bash
make smoketest           # 7-check end-to-end test, see scripts/smoketest.py
curl http://localhost:8000/health
```

A green smoke test means:
- both variants loaded
- predict is deterministic
- /pdb_string parses
- /evaluate + /evaluate/agent return valid verdicts

## Common operations

| Command | What it does |
|---|---|
| `make up` | Build + start the stack |
| `make down` | Stop containers (keeps the `trajecta_inference_data` volume) |
| `make restart` | Restart both services |
| `make logs` | Tail logs from both services |
| `make shell-inference` | Drop into a bash inside the inference container |
| `make ingest` | Run the RAG ingest pipeline |
| `make precompute` | Run the worked-example + failure-mode precompute script |
| `make test` | Run the inference-service pytest suite (label-filter regression, etc.) |
| `make clean` | `make down` + delete the persistent data volume |

## Dev mode without Docker

Run the backend and frontend in separate terminals:

```bash
# terminal 1 — backend
cd inference-service
pip install -r requirements.txt
pip install -e ../OpenTSLM
export $(grep -v '^#' ../.env | xargs)
uvicorn app:app --reload --port 8000

# terminal 2 — frontend
cd trajecta
npm install
npm run dev
```

Vite proxies `/api/*` to `http://localhost:8000`, so the same `api.ts` client
works in both dev and prod. Override the proxy target with `DEV_API_URL=...`.

## Money + safety

- The agent loop calls OpenRouter (Claude Opus 4.7). One run ≈ $0.20–0.50.
- `OPENROUTER_DAILY_USD_CAP` (default $20) is enforced in-process; once
  exhausted, `/evaluate/agent` returns HTTP 429.
- Predict responses are deterministic (`temperature=0`, fixed seeds) and free
  — call them as much as you want.
- Cached agent verdicts are returned for free; the frontend shows a
  "(cached)" pill when this happens.

## Troubleshooting

| Symptom | Probable cause | Fix |
|---|---|---|
| `/health` returns `status: degraded` | no checkpoint in `./checkpoints/v{1a,1b}/` | drop a `ckpt_*.pt` in there and `make restart` |
| `/predict` returns 404 "not in test split" | the PDB isn't in `preprocessed/features_test.npz` | use one of the PDBs from `/pdb_ids` |
| `/evaluate/agent` returns 429 | daily cap reached | wait for midnight UTC reset, or raise `OPENROUTER_DAILY_USD_CAP` in `.env` |
| `/evaluate/agent` returns "OPENROUTER_API_KEY not set" | missing env | edit `.env`, `make restart` |
| 3D viewer empty | `/pdb_string` 503 — HDF5 not mounted | check `./MD.hdf5` exists and `docker compose config` shows it mounted |
| Failure-modes tab shows "no precomputed" | `make precompute` not run | run it; the JSON lands in the `trajecta_inference_data` volume |

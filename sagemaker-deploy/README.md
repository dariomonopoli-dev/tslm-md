# Deploy the Trajecta TSLM to a SageMaker endpoint

The trained TSLM (v1a + v1b, ~80 MB each) runs on a SageMaker GPU endpoint;
the rest of the stack (agent loop, RAG, tools, FastAPI orchestrator) stays
wherever you run it and calls the endpoint via `boto3 invoke_endpoint`.

Two-step deploy, both run from the **SageMaker Studio Code Editor** (or any
terminal with AWS creds + the SageMaker SDK):

```
1. build_model_tarball.py   → packages checkpoints + handler → model.tar.gz → S3
2. deploy.py                → creates SageMaker model + endpoint config + endpoint
```

After that, `invoke_test.py` confirms it works, and the local FastAPI service
points `INFERENCE_BACKEND=sagemaker` (next section) to route `/predict` calls
through it.

## Step 0 — prerequisites in Code Editor

```bash
pip install --upgrade sagemaker boto3
# Hugging Face token: Llama-3.2-1B is gated
export HUGGING_FACE_HUB_TOKEN=hf_REPLACE_ME
```

Confirm you're in the right region and have an exec role:

```bash
python -c "import sagemaker; print(sagemaker.Session().boto_region_name, sagemaker.get_execution_role())"
```

## Step 1 — build model.tar.gz

From within Code Editor, with access to the checkpoint files (either on the
training instance's filesystem, or after `aws s3 cp` from your S3 sync target
in TRAINING.md §8):

```bash
cd sagemaker-deploy
python build_model_tarball.py \
    --v1a-ckpt /opt/ml/checkpoints/v1a/ckpt_ep1.pt \
    --v1b-ckpt /opt/ml/checkpoints/v1b/ckpt_final.pt \
    --preprocessed /home/sagemaker-user/preprocessed \
    --code-dir ./code \
    --out model.tar.gz \
    --s3-uri s3://<your-bucket>/trajecta/model.tar.gz
```

Either `--v1a-ckpt` or `--v1b-ckpt` is enough — pass both to serve both.

The script bundles:
```
model.tar.gz
├── v1a/ckpt_final.pt
├── v1b/ckpt_final.pt
├── preprocessed/
│   ├── features_test.npz       MISATOMDQADataset reads from here
│   └── samples_test.jsonl
└── code/
    ├── inference.py            SageMaker handler (model_fn / input_fn / predict_fn / output_fn)
    └── requirements.txt        installs OpenTSLM + transformers + peft into the DLC at boot
```

Expected size: ~200 MB (checkpoints + features_test.npz dominate). No raw
MD.hdf5 trajectory — `/pdb_string` (which needs the 124 GB HDF5) stays in
the local FastAPI service, not on SageMaker.

## Step 2 — deploy

```bash
python deploy.py \
    --model-data s3://<your-bucket>/trajecta/model.tar.gz \
    --endpoint-name trajecta-tslm \
    --instance-type ml.g5.xlarge \
    --mode realtime
```

First boot takes ~3–5 min (DLC pull, `pip install -r code/requirements.txt`,
Llama-3.2-1B download from HF, checkpoint load). `deploy.py` blocks until
the endpoint reaches `InService`.

To swap to async (scale-to-zero, request → S3 → response → S3):

```bash
python deploy.py \
    --model-data s3://<your-bucket>/trajecta/model.tar.gz \
    --endpoint-name trajecta-tslm \
    --mode async \
    --async-output-s3 s3://<your-bucket>/trajecta/async-out/
```

## Step 3 — sanity check

```bash
python invoke_test.py --endpoint-name trajecta-tslm --pdb 1A1B --variant v1a
```

Expected response:

```json
{
  "pdb_id": "1A1B",
  "variant": "v1a",
  "pK": 6.42,
  "rationale": "During the trajectory the interaction energy ... Answer: 6.42",
  "head_pK": null,
  "hidden_pK": 6.31,
  "latency_ms": 1840,
  "model_version": "v1a-ckpt_final"
}
```

## Step 4 — point the local FastAPI service at the endpoint

In `.env` at the repo root:

```
INFERENCE_BACKEND=sagemaker
SAGEMAKER_ENDPOINT_NAME=trajecta-tslm
SAGEMAKER_REGION=us-west-2
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
# or, if running on EC2 with an instance role, omit the keys
```

Then `make restart inference`. `/predict` and `/predict/batch` now proxy to
SageMaker; everything else (agent, RAG, /pdb_string, /evaluate) keeps
running locally.

## Step 5 — tear down

When you're done demoing:

```bash
python deploy.py --endpoint-name trajecta-tslm --delete
```

Deletes the endpoint + endpoint-config + model. The `model.tar.gz` in S3
stays so you can redeploy without rebuilding.

## Common issues

| Symptom | Probable cause | Fix |
|---|---|---|
| "401 Client Error: Unauthorized" during boot | HF token missing/expired | Re-export `HUGGING_FACE_HUB_TOKEN`, re-run `deploy.py` |
| `model_fn` errors with "no checkpoints loaded" | tarball missing `v1a/` or `v1b/` directory | Re-run `build_model_tarball.py` with the right `--*-ckpt` paths |
| InService but invoke returns 500 | check CloudWatch `/aws/sagemaker/Endpoints/<name>` logs | `aws logs tail /aws/sagemaker/Endpoints/trajecta-tslm --follow` |
| Cold start > 5 min | torch version mismatch forces wheel rebuild | Match `--framework-version` to your training torch (default 2.4.0) |
| First invocation slow (~30 s) | OS-level model load + JIT warmup | Send one warmup call after deploy; subsequent calls are ~2-3 s |

"""FastAPI entry point for the MoleMotion inference service.

Routes:
  /health, /pdb_ids                     — task #7 (live)
  /predict, /predict/batch              — task #7 (live)
  /pdb_string/{pdb_id}                  — task #9 (stub)
  /evaluate, /evaluate/agent            — task #18 (stub)
  /failure_modes                        — task #6 (stub)
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

load_dotenv()

import inference  # noqa: E402
from llm import spend as spend_mod  # noqa: E402
from middleware import limiter, AGENT_LIMIT, PREDICT_LIMIT  # noqa: E402
from slowapi.errors import RateLimitExceeded  # noqa: E402
from slowapi import _rate_limit_exceeded_handler  # noqa: E402


@asynccontextmanager
async def lifespan(app: FastAPI):
    ckpt = os.getenv("CHECKPOINT_DIR", "/app/checkpoints")
    split = os.getenv("TEST_SPLIT_PATH", "/app/data/test_MD.txt")
    try:
        inference.warm_up(ckpt, split)
    except Exception as e:
        print(f"[startup] inference warm_up failed: {e}")
        # Still boot — /predict will 503, but /health surfaces the failure
        # so the frontend renders a clear "model not loaded" state.
    yield


app = FastAPI(title="MoleMotion inference service", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(GZipMiddleware, minimum_size=1024)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:3000").split(","),
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: str
    variants_loaded: list[str]
    warm_since: str | None = None
    rag_corpus_version: str
    judge_model: str
    spend_today_usd: float
    remaining_cap_usd: float
    inference_backend: str
    sagemaker: dict | None = None


class PredictRequest(BaseModel):
    pdb_id: str
    variant: str = Field(pattern="^v1[ab]$")


class BatchPredictRequest(BaseModel):
    pdb_ids: list[str]
    variant: str = Field(pattern="^v1[ab]$")


class EvaluateRequest(BaseModel):
    pdb_id: str
    variant: str = Field(pattern="^v1[ab]$")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    loaded = inference.variants_loaded()
    backend = inference.backend()
    return HealthResponse(
        status="ready" if loaded else "degraded",
        variants_loaded=loaded,
        rag_corpus_version=os.getenv("RAG_CORPUS_VERSION", "v1-unset"),
        judge_model=os.getenv("ANTHROPIC_MODEL", "anthropic/claude-opus-4-7"),
        spend_today_usd=spend_mod.spend_today_usd(),
        remaining_cap_usd=spend_mod.remaining_cap_usd(),
        inference_backend=backend,
        sagemaker=inference.sm_endpoint_info() if backend == "sagemaker" else None,
    )


@app.get("/pdb_ids")
async def pdb_ids() -> list[str]:
    ids = inference.list_pdb_ids()
    if not ids:
        raise HTTPException(status_code=503, detail="test-split index not built")
    return ids


def _build_predict_response(result: dict) -> dict:
    """Wrap inference.predict() output with the v8.1 envelope + verifier result.

    The regex verifier is wired in task #8 — until then, attach an empty
    placeholder so the frontend can render the response shape stably.
    """
    try:
        import verifier  # task #8
        regex = verifier.verify(result["rationale"], result["pdb_id"])
    except (NotImplementedError, ImportError):
        regex = {
            "verified": 0, "contradicted": 0, "unverifiable": 0, "total": 0,
            "claims": [],
        }
    return {
        **result,
        "regex_verifier": regex,
        "rag_corpus_version": os.getenv("RAG_CORPUS_VERSION", "v1-unset"),
        "judge_model": os.getenv("ANTHROPIC_MODEL", "anthropic/claude-opus-4-7"),
    }


@app.post("/predict")
async def predict(req: PredictRequest):
    if req.variant not in inference.variants_loaded():
        raise HTTPException(status_code=503, detail=f"variant {req.variant} not loaded")
    if not inference.is_in_test_split(req.pdb_id):
        raise HTTPException(status_code=404, detail=f"{req.pdb_id} not in test split")
    try:
        result = inference.predict(req.pdb_id, req.variant)  # type: ignore[arg-type]
    except KeyError:
        raise HTTPException(status_code=404, detail=f"{req.pdb_id} not in MISATO index")
    return _build_predict_response(result)


@app.post("/predict/batch")
async def predict_batch(req: BatchPredictRequest):
    if len(req.pdb_ids) > 50:
        raise HTTPException(status_code=413, detail="max 50 ids per batch")
    if req.variant not in inference.variants_loaded():
        raise HTTPException(status_code=503, detail=f"variant {req.variant} not loaded")

    results, failed = [], []
    for pid in req.pdb_ids:
        if not inference.is_in_test_split(pid):
            failed.append({"pdb_id": pid, "error": "not in test split"})
            continue
        try:
            r = inference.predict(pid, req.variant)  # type: ignore[arg-type]
            results.append(_build_predict_response(r))
        except Exception as e:
            failed.append({"pdb_id": pid, "error": str(e)})
    return {"results": results, "failed": failed}


# ---------------------------------------------------------------------------
# Routes still owned by other tasks — surface as 501 with task pointer.
# ---------------------------------------------------------------------------


@app.get("/pdb_string/{pdb_id}", response_class=PlainTextResponse)
async def pdb_string(pdb_id: str, stride: int = 5, drop_water: bool = True):
    try:
        from hdf5_to_pdb import hdf5_to_pdb
        return hdf5_to_pdb(pdb_id, stride=stride, drop_water=drop_water)
    except NotImplementedError:
        raise HTTPException(status_code=501, detail="task #9 not landed")


async def _run_predict_for_eval(req: EvaluateRequest) -> dict:
    """Fetch (or generate) the prediction we're evaluating against.

    Predict results are not cached separately (they're cheap, deterministic);
    only the evaluate-verdict layer is cached.
    """
    if req.variant not in inference.variants_loaded():
        raise HTTPException(status_code=503, detail=f"variant {req.variant} not loaded")
    if not inference.is_in_test_split(req.pdb_id):
        raise HTTPException(status_code=404, detail=f"{req.pdb_id} not in test split")
    try:
        return inference.predict(req.pdb_id, req.variant)  # type: ignore[arg-type]
    except KeyError:
        raise HTTPException(status_code=404, detail=f"{req.pdb_id} not in MISATO index")


def _versions(predict_result: dict) -> tuple[str, str, str]:
    return (
        predict_result.get("model_version", "unknown"),
        os.getenv("RAG_CORPUS_VERSION", "v1-unset"),
        os.getenv("ANTHROPIC_MODEL", "anthropic/claude-opus-4-7"),
    )


@app.post("/evaluate")
@limiter.limit(PREDICT_LIMIT)
async def evaluate(request: Request, req: EvaluateRequest, force: bool = False):
    from orchestrator import evaluate_fast
    import eval_cache

    pred = await _run_predict_for_eval(req)
    mv, rcv, jm = _versions(pred)

    if not force:
        cached = eval_cache.get(req.pdb_id, req.variant, "fast", mv, rcv, jm)
        if cached:
            cached = {**cached, "cached": True}
            return cached

    verdict = await evaluate_fast(
        pdb_id=req.pdb_id, model_pK=pred["pK"],
        rationale=pred["rationale"], variant=req.variant,
    )
    spend_mod.record({
        "pdb_id": req.pdb_id, "variant": req.variant, "model": jm,
        "input_tokens": verdict["agent_trace"]["input_tokens"],
        "output_tokens": verdict["agent_trace"]["output_tokens"],
        "cache_read_input_tokens": verdict["agent_trace"].get("cache_read_input_tokens", 0),
        "usd": _verdict_cost_usd(verdict, jm),
        "cached": False, "tool_calls": 0,
    })
    eval_cache.put(req.pdb_id, req.variant, "fast", mv, rcv, jm, verdict)
    return {**verdict, "cached": False}


@app.post("/evaluate/agent")
@limiter.limit(AGENT_LIMIT)
async def evaluate_agent_route(request: Request, req: EvaluateRequest, force: bool = False):
    from orchestrator import evaluate_agent as run_agent
    import eval_cache

    pred = await _run_predict_for_eval(req)
    mv, rcv, jm = _versions(pred)

    if not force:
        cached = eval_cache.get(req.pdb_id, req.variant, "agent", mv, rcv, jm)
        if cached:
            return {**cached, "cached": True}

    verdict, trace = await run_agent(
        pdb_id=req.pdb_id, model_pK=pred["pK"],
        rationale=pred["rationale"], variant=req.variant,
    )
    payload = {"verdict": verdict, "trace": trace, "cached": False}
    spend_mod.record({
        "pdb_id": req.pdb_id, "variant": req.variant, "model": jm,
        "input_tokens": verdict["agent_trace"]["input_tokens"],
        "output_tokens": verdict["agent_trace"]["output_tokens"],
        "cache_read_input_tokens": verdict["agent_trace"].get("cache_read_input_tokens", 0),
        "usd": _verdict_cost_usd(verdict, jm),
        "cached": False, "tool_calls": verdict["agent_trace"]["tool_calls"],
    })
    eval_cache.put(req.pdb_id, req.variant, "agent", mv, rcv, jm, payload)
    return payload


def _verdict_cost_usd(verdict: dict, model: str) -> float:
    from llm.pricing import compute_usd
    t = verdict.get("agent_trace", {})
    return compute_usd(
        model,
        int(t.get("input_tokens", 0)),
        int(t.get("output_tokens", 0)),
        int(t.get("cache_read_input_tokens", 0)),
    )


@app.get("/failure_modes")
async def failure_modes(variant: str = "v1b"):
    """Static-file backed; populated by task #20 precompute script."""
    path = f"data/failure_modes_{variant}.json"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"no precomputed failure modes for {variant}")
    import json
    with open(path) as f:
        return json.load(f)

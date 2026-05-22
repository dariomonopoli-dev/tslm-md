"""TSLM v1a + v1b loader and predict() — task #7 + #27.

Two backends, picked at startup via `INFERENCE_BACKEND` env var:
  - local      (default): load OpenTSLMSP checkpoints in-process. Needs GPU
                          + OpenTSLM installed + ckpt_*.pt mounted.
  - sagemaker            : boto3.invoke_endpoint() against a SageMaker
                          endpoint deployed via sagemaker-deploy/. The
                          local FastAPI container becomes a thin proxy +
                          regex verifier + agent orchestrator.

Architecture (from TRAINING.md, applies to both backends):
  - Frozen Llama-3.2-1B + LoRA (rank 32) on q/k/v/o + MLP projections
  - TransformerCNNEncoder (Conv1d patch=4 + 6-layer Transformer)
  - MLPProjector → LLM hidden_size
  - v1b: + 2-layer MLP regression head on last input-position hidden state
"""

from __future__ import annotations

import hashlib
import os
import random
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypedDict


Variant = Literal["v1a", "v1b"]
ANSWER_RE = re.compile(r"Answer:\s*(-?\d+(?:\.\d+)?)")


class PredictResult(TypedDict):
    pdb_id: str
    variant: Variant
    pK: float
    rationale: str
    hidden_pK: float | None
    head_pK: float | None
    latency_ms: int
    model_version: str


@dataclass
class LoadedVariant:
    """One loaded checkpoint + everything needed to call .generate() on it."""
    model: Any              # OpenTSLMSP instance
    ckpt_path: Path
    version: str            # "{variant}-{ckpt_basename}-{git_sha[:7]}"


_LOADED: dict[Variant, LoadedVariant] = {}
_PDB_INDEX: dict[str, dict] = {}
_TEST_SPLIT: set[str] = set()

# Backend state (set by warm_up).
_BACKEND: str = "local"
_SM_ENDPOINT_NAME: str = ""
_SM_REGION: str = ""
_SM_VARIANTS: list[str] = []
_SM_ASYNC: bool = False
_SM_RUNTIME = None  # lazy boto3 client


# --------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------


def _determinism(seed: int = 0) -> None:
    """Clamp every source of float drift. Call once at startup."""
    os.environ.setdefault("PYTHONHASHSEED", "0")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass


# --------------------------------------------------------------------------
# Version stamping
# --------------------------------------------------------------------------


def _git_sha() -> str:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short=7", "HEAD"],
            capture_output=True, text=True, check=True, timeout=5,
        ).stdout.strip()
        return sha or "nogit"
    except Exception:
        return "nogit"


def _version(variant: Variant, ckpt: Path) -> str:
    base = ckpt.stem  # e.g. "ckpt_ep1"
    stamp = time.strftime("%Y-%m-%d")
    return f"{variant}-{stamp}-{base}-{_git_sha()}"


# --------------------------------------------------------------------------
# Test-split whitelist + per-PDB sample index
# --------------------------------------------------------------------------


def _load_test_split(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def _build_pdb_index() -> dict[str, dict]:
    """Map pdb_id → sample dict (channels, pK, rationale facts) from MISATO test split.

    The OpenTSLM MISATOMDQADataset returns one dict per system; we cache them
    in memory at startup so predict() is an O(1) lookup.
    """
    try:
        from opentslm.time_series_datasets.misato.MISATOMDQADataset import (
            MISATOMDQADataset,
        )
    except ImportError:
        return {}

    # EOS_TOKEN is filled in once the model is loaded — for indexing alone we
    # can use an empty string; the dict carries the same fields either way.
    ds = MISATOMDQADataset(split="test", EOS_TOKEN="")
    index: dict[str, dict] = {}
    for sample in ds:
        pdb = sample.get("pdb_id")
        if pdb:
            index[str(pdb)] = sample
    return index


# --------------------------------------------------------------------------
# Loader
# --------------------------------------------------------------------------


def load_variants(checkpoint_dir: str) -> dict[Variant, LoadedVariant]:
    """Eager-load both variants at startup. Returns {variant: LoadedVariant}.

    Looks for two checkpoint files inside ``checkpoint_dir``:
      v1a -> $CHECKPOINT_DIR/v1a/ckpt_final.pt  (or ckpt_ep1.pt — best per TRAINING.md)
      v1b -> $CHECKPOINT_DIR/v1b/ckpt_final.pt

    Falls through silently for missing variants — the service still boots
    so /health and routes that don't need the model still work.
    """
    _determinism()

    try:
        from opentslm.model.llm.OpenTSLMSP import OpenTSLMSP
    except ImportError as e:
        raise RuntimeError(
            "OpenTSLM not installed. `pip install -e /path/to/OpenTSLM` first."
        ) from e

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_llm = os.getenv("BASE_LLM_ID", "meta-llama/Llama-3.2-1B")
    lora_r = int(os.getenv("LORA_R", "32"))
    lambda_reg = float(os.getenv("LAMBDA_REG", "0.5"))

    candidates: dict[Variant, list[Path]] = {
        "v1a": [
            Path(checkpoint_dir) / "v1a" / "ckpt_final.pt",
            Path(checkpoint_dir) / "v1a" / "ckpt_ep1.pt",
        ],
        "v1b": [
            Path(checkpoint_dir) / "v1b" / "ckpt_final.pt",
            Path(checkpoint_dir) / "v1b" / "ckpt_ep5.pt",
        ],
    }

    out: dict[Variant, LoadedVariant] = {}
    for variant, paths in candidates.items():
        ckpt = next((p for p in paths if p.exists()), None)
        if ckpt is None:
            print(f"[inference] {variant}: no checkpoint found at {paths[0]}")
            continue

        print(f"[inference] {variant}: loading {ckpt}")
        model = OpenTSLMSP(llm_id=base_llm, device=device)
        model.enable_lora(lora_r=lora_r)
        if variant == "v1b":
            model.enable_regression(weight=lambda_reg)
        model.load_from_file(str(ckpt))
        model.eval()

        out[variant] = LoadedVariant(
            model=model,
            ckpt_path=ckpt,
            version=_version(variant, ckpt),
        )
        print(f"[inference] {variant}: ready, version={out[variant].version}")
    return out


def warm_up(checkpoint_dir: str, test_split_path: str) -> None:
    """Called once from app.py lifespan. Dispatches to the chosen backend."""
    global _LOADED, _PDB_INDEX, _TEST_SPLIT, _BACKEND
    global _SM_ENDPOINT_NAME, _SM_REGION, _SM_VARIANTS, _SM_ASYNC

    _BACKEND = os.getenv("INFERENCE_BACKEND", "local").lower()
    _TEST_SPLIT = _resolve_test_split(test_split_path)
    print(f"[inference] backend={_BACKEND}  test_split_size={len(_TEST_SPLIT)}")

    if _BACKEND == "sagemaker":
        _SM_ENDPOINT_NAME = os.getenv("SAGEMAKER_ENDPOINT_NAME", "")
        _SM_REGION = os.getenv("SAGEMAKER_REGION", os.getenv("AWS_REGION", "us-west-2"))
        _SM_VARIANTS = [v.strip() for v in os.getenv("SAGEMAKER_VARIANTS", "v1a,v1b").split(",") if v.strip()]
        _SM_ASYNC = os.getenv("SAGEMAKER_ASYNC", "0") == "1"
        if not _SM_ENDPOINT_NAME:
            print("[inference] SAGEMAKER_ENDPOINT_NAME unset — /predict will 503")
            return
        try:
            _check_sm_endpoint()
        except Exception as e:
            print(f"[inference] sagemaker endpoint check failed: {e}")
        return

    # Local backend: load checkpoints + build per-PDB index.
    _LOADED = load_variants(checkpoint_dir)
    _PDB_INDEX = _build_pdb_index()
    if not _TEST_SPLIT and _PDB_INDEX:
        _TEST_SPLIT = set(_PDB_INDEX.keys())


def _resolve_test_split(test_split_path: str) -> set[str]:
    """Try text-file first, then preprocessed/features_test.npz, then npz dir."""
    path = Path(test_split_path)
    if path.exists():
        return _load_test_split(path)
    for npz in (
        Path("/app/data/preprocessed/features_test.npz"),
        Path("preprocessed/features_test.npz"),
        Path("../preprocessed/features_test.npz"),
    ):
        if npz.exists():
            import numpy as np
            arr = np.load(npz, allow_pickle=True)["pdb_ids"]
            return {str(p) for p in arr}
    return set()


def _check_sm_endpoint() -> None:
    """Probe describe_endpoint to confirm InService. Raises on failure."""
    import boto3
    sm = boto3.client("sagemaker", region_name=_SM_REGION)
    desc = sm.describe_endpoint(EndpointName=_SM_ENDPOINT_NAME)
    status = desc.get("EndpointStatus")
    print(f"[inference] sagemaker endpoint {_SM_ENDPOINT_NAME}: {status}")
    if status != "InService":
        raise RuntimeError(f"endpoint {_SM_ENDPOINT_NAME} status={status}")


def _sm_runtime():
    global _SM_RUNTIME
    if _SM_RUNTIME is None:
        import boto3
        client = "sagemaker-runtime"
        _SM_RUNTIME = boto3.client(client, region_name=_SM_REGION)
    return _SM_RUNTIME


def variants_loaded() -> list[str]:
    if _BACKEND == "sagemaker":
        return list(_SM_VARIANTS) if _SM_ENDPOINT_NAME else []
    return list(_LOADED.keys())


def backend() -> str:
    return _BACKEND


def sm_endpoint_info() -> dict:
    """Surfaced via /health when backend=sagemaker."""
    return {"endpoint_name": _SM_ENDPOINT_NAME, "region": _SM_REGION, "async": _SM_ASYNC}


def is_in_test_split(pdb_id: str) -> bool:
    return pdb_id in _TEST_SPLIT


def list_pdb_ids() -> list[str]:
    return sorted(_TEST_SPLIT)


# --------------------------------------------------------------------------
# Predict
# --------------------------------------------------------------------------


def _build_batch(pdb_id: str, eos_token: str) -> list[dict]:
    """Materialize a single-item batch for the loaded model.

    The MISATOMDQADataset cached samples are missing the model's EOS token
    (we built the index before a model existed). We restore it here.
    """
    from opentslm.time_series_datasets.util import (
        extend_time_series_to_match_patch_size_and_aggregate,
    )

    sample = _PDB_INDEX.get(pdb_id)
    if sample is None:
        raise KeyError(f"pdb_id {pdb_id} not in MISATO test index")

    # Inject the model's EOS so the rendered prompt template matches training.
    sample = dict(sample)
    sample["EOS_TOKEN"] = eos_token
    return extend_time_series_to_match_patch_size_and_aggregate([sample])


def predict(pdb_id: str, variant: Variant, max_new_tokens: int = 160) -> PredictResult:
    """Dispatch to the configured backend. Deterministic in both."""
    if _BACKEND == "sagemaker":
        return _predict_sagemaker(pdb_id, variant, max_new_tokens)
    return _predict_local(pdb_id, variant, max_new_tokens)


def _predict_local(pdb_id: str, variant: Variant, max_new_tokens: int) -> PredictResult:
    """In-process forward pass — `INFERENCE_BACKEND=local`."""
    import torch

    loaded = _LOADED.get(variant)
    if loaded is None:
        raise RuntimeError(f"variant {variant} not loaded")

    sample = _PDB_INDEX.get(pdb_id)
    if sample is None:
        raise KeyError(pdb_id)

    eos = loaded.model.get_eos_token()
    batch = _build_batch(pdb_id, eos)

    t0 = time.monotonic()
    with torch.no_grad():
        generated = loaded.model.generate(batch, max_new_tokens=max_new_tokens, do_sample=False)
        head_pK: float | None = None
        if getattr(loaded.model, "regression_enabled", False):
            head_pK = float(loaded.model.predict_pK(batch)[0])
    latency_ms = int((time.monotonic() - t0) * 1000)

    rationale = generated[0]
    parsed = _parse_pK(rationale)

    # For v1b prefer the regression head; for v1a use the parsed string.
    if variant == "v1b" and head_pK is not None:
        pK = head_pK
    elif parsed is not None:
        pK = parsed
    else:
        # Failed parse: surface zero-rationale state for the UI to flag.
        pK = float("nan")

    return PredictResult(
        pdb_id=pdb_id,
        variant=variant,
        pK=pK,
        rationale=rationale,
        hidden_pK=float(sample.get("pK")) if "pK" in sample else None,
        head_pK=head_pK,
        latency_ms=latency_ms,
        model_version=loaded.version,
    )


def _predict_sagemaker(pdb_id: str, variant: Variant, max_new_tokens: int) -> PredictResult:
    """boto3 invoke_endpoint against the SageMaker TSLM endpoint."""
    import json
    if not _SM_ENDPOINT_NAME:
        raise RuntimeError("SAGEMAKER_ENDPOINT_NAME not configured")
    if variant not in _SM_VARIANTS:
        raise RuntimeError(f"variant {variant} not in SAGEMAKER_VARIANTS={_SM_VARIANTS}")

    payload = json.dumps({
        "pdb_id": pdb_id,
        "variant": variant,
        "max_new_tokens": max_new_tokens,
    }).encode()

    t0 = time.monotonic()
    if _SM_ASYNC:
        raise NotImplementedError(
            "async mode requires an input S3 location — wire from app.py + poll the OutputLocation"
        )
    resp = _sm_runtime().invoke_endpoint(
        EndpointName=_SM_ENDPOINT_NAME,
        ContentType="application/json",
        Accept="application/json",
        Body=payload,
    )
    body = resp["Body"].read().decode()
    parsed = json.loads(body)
    rtt_ms = int((time.monotonic() - t0) * 1000)

    if "error" in parsed:
        raise RuntimeError(f"sagemaker: {parsed['error']}")

    return PredictResult(
        pdb_id=parsed["pdb_id"],
        variant=parsed["variant"],
        pK=float(parsed["pK"]) if parsed.get("pK") is not None else float("nan"),
        rationale=parsed.get("rationale", ""),
        hidden_pK=parsed.get("hidden_pK"),
        head_pK=parsed.get("head_pK"),
        latency_ms=int(parsed.get("latency_ms", rtt_ms)),
        model_version=parsed.get("model_version", f"sm-{_SM_ENDPOINT_NAME}"),
    )


def predict_batch(pdb_ids: list[str], variant: Variant) -> list[PredictResult]:
    """Sequential fan-out. SageMaker mode benefits from threadpool concurrency
    (see app.py); the in-process backend uses one GPU so concurrency = 1.
    """
    if _BACKEND == "sagemaker":
        import concurrent.futures as cf
        max_workers = int(os.getenv("SAGEMAKER_PARALLEL", "4"))
        with cf.ThreadPoolExecutor(max_workers=max_workers) as pool:
            return list(pool.map(lambda pid: predict(pid, variant), pdb_ids))
    return [predict(pid, variant) for pid in pdb_ids]


def _parse_pK(text: str) -> float | None:
    m = ANSWER_RE.search(text)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


# --------------------------------------------------------------------------
# Helpers for downstream tasks
# --------------------------------------------------------------------------


def get_sample(pdb_id: str) -> dict | None:
    """Used by the verifier (task #8) + tools (task #15) for raw channel data."""
    return _PDB_INDEX.get(pdb_id)


def determinism_signature(pdb_id: str, variant: Variant) -> str:
    """Hash used for cache keys — invalidates when checkpoint or input changes."""
    loaded = _LOADED.get(variant)
    if loaded is None:
        return ""
    h = hashlib.sha256()
    h.update(loaded.version.encode())
    h.update(pdb_id.encode())
    return h.hexdigest()[:16]

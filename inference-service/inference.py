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


class PredictResult(TypedDict, total=False):
    # `total=False` lets the tunnel backend stash extra fields (`_tunnel_raw`)
    # without breaking strict typing on the common keys below.
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

# Mock backend state — read from preprocessed/samples_test.jsonl.
# Lets us exercise the full agent loop without a trained model.
_MOCK_SAMPLES: dict[str, dict] = {}
_MOCK_NOISE: str = "lossy"  # "clean" or "lossy"

# Tunnel backend state — calls a remote /predict endpoint (e.g. Cloudflare
# tunnel pointing at a training-instance FastAPI). The remote owns model
# loading; we just translate schemas.
_TUNNEL_URL: str = ""
_TUNNEL_TAU_HIGH: float = 1.5
_TUNNEL_PDB_IDS: list[str] = []
_TUNNEL_HEALTH: dict = {}  # cached /health response: {ok, backbone, model_path, n_pdb_ids, ...}
_GROUND_TRUTH: dict[str, float] = {}  # pdb_id → pK, populated from samples_test.jsonl at warm-up


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

    if _BACKEND == "mock":
        _load_mock_samples()
        if not _TEST_SPLIT and _MOCK_SAMPLES:
            _TEST_SPLIT = set(_MOCK_SAMPLES.keys())
        print(f"[inference] mock backend: {len(_MOCK_SAMPLES)} samples, noise={_MOCK_NOISE}")
        return

    if _BACKEND == "tunnel":
        global _TUNNEL_URL, _TUNNEL_TAU_HIGH, _TUNNEL_PDB_IDS
        _TUNNEL_URL = os.getenv("TUNNEL_URL", "").rstrip("/")
        _TUNNEL_TAU_HIGH = float(os.getenv("TUNNEL_TAU_HIGH", "1.5"))
        if not _TUNNEL_URL:
            print("[inference] TUNNEL_URL unset — /predict will 503")
            return
        try:
            _tunnel_warm_up()
        except Exception as e:
            print(f"[inference] tunnel warm_up failed: {e}")
        # Load ground-truth so predictions on test-split PDBs show actual pK.
        _load_ground_truth()
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


def _load_mock_samples() -> None:
    """Build pdb_id → sample dict from preprocessed/samples_test.jsonl.

    Each sample has {pdb_id, pK, rationale, facts}. We treat the
    templated rationale as the mock model's output. Optionally perturb in
    `lossy` mode to make the agent's verifier panel show contradictions.
    """
    import json
    global _MOCK_SAMPLES, _MOCK_NOISE
    _MOCK_NOISE = os.getenv("MOCK_NOISE", "lossy").lower()
    if _MOCK_NOISE not in ("clean", "lossy"):
        _MOCK_NOISE = "lossy"
    candidates = [
        Path(os.getenv("MISATO_SAMPLES_TEST", "")),
        Path("/app/data/preprocessed/samples_test.jsonl"),
        Path("preprocessed/samples_test.jsonl"),
        Path("../preprocessed/samples_test.jsonl"),
    ]
    path = next((p for p in candidates if p and p.exists()), None)
    if path is None:
        print("[inference] mock backend: no samples_test.jsonl found — /predict will 404")
        return
    out: dict[str, dict] = {}
    with path.open() as f:
        for line in f:
            try:
                s = json.loads(line)
            except Exception:
                continue
            pid = s.get("pdb_id")
            if pid:
                out[str(pid)] = s
    _MOCK_SAMPLES = out


def _tunnel_warm_up() -> None:
    """Probe the tunnel /health. PDB enumeration is on-demand via /ids.

    Trying to enumerate the full 16k-PDB universe at startup is brittle (the
    /ids endpoint is autocomplete-style, prefix-required). Instead we trust
    the tunnel to validate pdb_ids at predict time (it 404s if unknown), and
    use it as an on-demand autocomplete from the frontend.

    Also caches the /health response so we can surface the actual model name
    (backbone + checkpoint path) instead of a generic "tunnel" label.
    """
    import urllib.request, json
    global _TUNNEL_PDB_IDS, _TUNNEL_HEALTH

    with urllib.request.urlopen(f"{_TUNNEL_URL}/health", timeout=10) as r:
        h = json.loads(r.read())
    _TUNNEL_HEALTH = h
    n_ids = h.get("n_pdb_ids", 0)
    print(f"[inference] tunnel /health: backbone={h.get('backbone') or h.get('checkpoint')} "
          f"device={h.get('device')} n_pdb_ids={n_ids}")

    # Mark backend ready by stashing a non-empty placeholder so variants_loaded() returns true.
    _TUNNEL_PDB_IDS = [f"<tunnel-{n_ids}-ids>"]  # sentinel — never returned to user


def _load_ground_truth() -> None:
    """pdb_id → pK lookup. Two sources, merged:

      1. preprocessed/samples_test.jsonl — ~1612 MISATO test PDBs (precomputed pK)
      2. misato-affinity/data/affinity_data.csv — ~16k PDBs with raw Kd/Ki/IC50;
         pK computed as 9 - log10(nM) per training preprocessor convention.

    Same priority as training: Kd > Ki > IC50.
    """
    import csv as _csv
    import json
    import math
    global _GROUND_TRUTH
    out: dict[str, float] = {}

    # Source 1: samples_test.jsonl (MISATO test split with precomputed pK)
    candidates = [
        Path(os.getenv("MISATO_SAMPLES_TEST", "")),
        Path("/app/data/preprocessed/samples_test.jsonl"),
        Path("preprocessed/samples_test.jsonl"),
        Path("../preprocessed/samples_test.jsonl"),
    ]
    samples_path = next((p for p in candidates if p and p.exists()), None)
    if samples_path:
        with samples_path.open() as f:
            for line in f:
                try:
                    s = json.loads(line)
                except Exception:
                    continue
                pid = s.get("pdb_id")
                pk = s.get("pK")
                if pid and pk is not None:
                    out[str(pid)] = float(pk)

    # Source 2: affinity_data.csv (the full PDBbind-derived set used by training).
    aff_candidates = [
        Path(os.getenv("AFFINITY_CSV_PATH", "")),
        Path("/app/data/affinity_src/affinity_data.csv"),
        Path("misato-affinity/data/affinity_data.csv"),
        Path("../misato-affinity/data/affinity_data.csv"),
    ]
    aff_path = next((p for p in aff_candidates if p and p.exists()), None)
    if aff_path:
        with aff_path.open() as f:
            for row in _csv.DictReader(f, delimiter=";"):
                pid = (row.get("PDBid") or "").strip()
                if not pid or pid in out:
                    continue  # samples_test.jsonl took priority
                for col in ("Kd (nM)", "Ki (nM)", "IC50 (nM)"):
                    val = (row.get(col) or "").strip()
                    if val and val not in ("nan", "NA", "0.0", "0"):
                        try:
                            nM = float(val)
                            if nM > 0:
                                out[pid] = round(9.0 - math.log10(nM), 3)
                                break
                        except ValueError:
                            continue

    _GROUND_TRUTH = out
    n_samples = sum(1 for _ in (samples_path.open() if samples_path else []))
    print(f"[inference] ground-truth lookup: {len(_GROUND_TRUTH)} pdb_ids "
          f"({n_samples} from samples_test, rest from affinity_data.csv)")


def tunnel_health() -> dict:
    return _TUNNEL_HEALTH


def ground_truth_pK(pdb_id: str) -> float | None:
    return _GROUND_TRUTH.get(pdb_id)


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
    if _BACKEND == "mock":
        return ["v1a", "v1b"] if _MOCK_SAMPLES else []
    if _BACKEND == "tunnel":
        # Tunnel exposes one checkpoint at a time; we present both variant
        # labels so the UI still works, both routing to the same call.
        return ["v1a", "v1b"] if _TUNNEL_URL and _TUNNEL_PDB_IDS else []
    return list(_LOADED.keys())


def backend() -> str:
    return _BACKEND


def sm_endpoint_info() -> dict:
    """Surfaced via /health when backend=sagemaker."""
    return {"endpoint_name": _SM_ENDPOINT_NAME, "region": _SM_REGION, "async": _SM_ASYNC}


def is_in_test_split(pdb_id: str) -> bool:
    # In tunnel mode, defer validation to the tunnel (its /predict 404s on unknown ids).
    if _BACKEND == "tunnel":
        return True
    return pdb_id in _TEST_SPLIT


def list_pdb_ids() -> list[str]:
    # Tunnel mode: defer to tunnel_search_ids() — the universe is 16k+ items
    # so we don't return the full list, we expose autocomplete via /pdb_ids?q=.
    if _BACKEND == "tunnel":
        return []   # frontend calls /pdb_ids?q= for autocomplete instead
    return sorted(_TEST_SPLIT)


def tunnel_search_ids(query: str, limit: int = 20) -> list[str]:
    """Proxy the tunnel's /ids?q=&limit= autocomplete."""
    import urllib.request, urllib.parse, json
    if not _TUNNEL_URL:
        return []
    url = f"{_TUNNEL_URL}/ids?q={urllib.parse.quote(query)}&limit={limit}"
    try:
        with urllib.request.urlopen(url, timeout=8) as r:
            return (json.loads(r.read()) or {}).get("ids", []) or []
    except Exception as e:
        print(f"[inference] tunnel /ids failed: {e}")
        return []


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
    """Dispatch to the configured backend. Deterministic in all four."""
    if _BACKEND == "sagemaker":
        return _predict_sagemaker(pdb_id, variant, max_new_tokens)
    if _BACKEND == "mock":
        return _predict_mock(pdb_id, variant)
    if _BACKEND == "tunnel":
        return _predict_tunnel(pdb_id, variant)
    return _predict_local(pdb_id, variant, max_new_tokens)


# kcal/mol → pK at 298 K. ΔG ≈ -RT ln Kd; pK = -log10(Kd) = -ΔG / (RT ln10).
# RT ln 10 ≈ 1.3633 kcal/mol at 298 K.
_RT_LN10_KCAL = 1.3633


def _predict_tunnel(pdb_id: str, variant: Variant) -> PredictResult:
    """Call the remote /predict endpoint over HTTPS.

    Translates the tunnel's response (affinity in kcal/mol, verdict +
    rationale + channel_summary) into our internal PredictResult shape.
    The full tunnel payload is preserved as `_tunnel_raw` so the FastAPI
    route handler can pass extras through to the frontend.
    """
    import urllib.request, urllib.error, json
    if not _TUNNEL_URL:
        raise RuntimeError("TUNNEL_URL not configured")

    body = json.dumps({"pdb_id": pdb_id, "tau_high": _TUNNEL_TAU_HIGH}).encode()
    req = urllib.request.Request(
        f"{_TUNNEL_URL}/predict",
        data=body, method="POST",
        headers={"Content-Type": "application/json"},
    )

    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            data = json.loads(r.read())
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise KeyError(pdb_id) from e
        raise RuntimeError(f"tunnel /predict HTTP {e.code}: {e.read()[:200]}") from e
    rtt_ms = int((time.monotonic() - t0) * 1000)

    affinity = float(data.get("affinity", 0.0))
    pK = -affinity / _RT_LN10_KCAL if affinity else float("nan")
    rationale = data.get("rationale", "")
    if not rationale and data.get("raw_pred"):
        rationale = str(data["raw_pred"]).strip()

    # Build a stable model_version from tunnel /health, not from per-PDB verdict.
    h = _TUNNEL_HEALTH or {}
    backbone = h.get("backbone") or h.get("checkpoint") or "tunnel"
    model_path = h.get("model_path") or ""
    if model_path:
        model_version = f"{backbone}@{model_path}"
    else:
        model_version = f"tunnel-{backbone}"

    result: PredictResult = PredictResult(
        pdb_id=pdb_id,
        variant=variant,
        pK=pK,
        rationale=rationale,
        hidden_pK=_GROUND_TRUTH.get(pdb_id),   # local samples_test.jsonl lookup
        head_pK=None,
        latency_ms=rtt_ms,
        model_version=model_version,
    )
    # Stash the full tunnel payload for app.py to surface in the response envelope.
    result["_tunnel_raw"] = data  # type: ignore[typeddict-unknown-key]
    return result


def tunnel_url() -> str:
    return _TUNNEL_URL


def tunnel_pdb_ids() -> list[str]:
    return list(_TUNNEL_PDB_IDS)


def _predict_mock(pdb_id: str, variant: Variant) -> PredictResult:
    """Synthetic prediction from preprocessed/samples_test.jsonl.

    `clean` mode: returns the templated rationale verbatim — regex
                  verifier should mark every claim as `verified`.
    `lossy` mode: adds ±0.3 pK noise + perturbs the FIRST numeric value
                  in the rationale by +1.5 — verifier shows one
                  `contradicted` claim, exercising the failure path the UI
                  is designed to surface.
    """
    import re
    sample = _MOCK_SAMPLES.get(pdb_id)
    if sample is None:
        raise KeyError(pdb_id)

    base_pk = float(sample.get("pK"))
    rationale = str(sample.get("rationale", ""))

    if _MOCK_NOISE == "lossy":
        # Deterministic per-PDB noise via stable hash.
        seed = sum(ord(c) for c in pdb_id)
        pK = round(base_pk + (((seed * 7) % 7) - 3) * 0.1, 2)  # ±0.3
        # Perturb the first numeric token so one claim flips to contradicted.
        rationale = re.sub(
            r"(\d+\.\d+)",
            lambda m: f"{float(m.group(1)) + 1.5:.2f}",
            rationale, count=1,
        )
    else:
        pK = round(base_pk, 2)

    return PredictResult(
        pdb_id=pdb_id,
        variant=variant,
        pK=pK,
        rationale=rationale,
        hidden_pK=base_pk,
        head_pK=pK if variant == "v1b" else None,
        latency_ms=0,
        model_version=f"mock-{variant}-{_MOCK_NOISE}",
    )


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

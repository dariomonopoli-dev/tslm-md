"""SageMaker PyTorch container entry-point for the Trajecta TSLM.

Loads both v1a + v1b checkpoints once at startup, then serves one prediction
per invocation. The SageMaker PyTorch inference container (DLC) calls the
four functions below — model_fn at boot, input/predict/output_fn per request.

Request format (application/json):
    {"pdb_id": "1A1B", "variant": "v1a", "max_new_tokens": 160}

Response format (application/json):
    {
      "pdb_id":        "1A1B",
      "variant":       "v1a",
      "pK":            6.42,
      "rationale":     "During the trajectory ... Answer: 6.42",
      "head_pK":       null,                     // v1b only
      "hidden_pK":     6.31,                     // truth, only on test split
      "latency_ms":    1840,
      "model_version": "v1a-ckpt_final"
    }
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

import torch

# OpenTSLM installed via code/requirements.txt at container start.
from opentslm.model.llm.OpenTSLMSP import OpenTSLMSP
from opentslm.time_series_datasets.misato.MISATOMDQADataset import MISATOMDQADataset
from opentslm.time_series_datasets.util import (
    extend_time_series_to_match_patch_size_and_aggregate,
)


ANSWER_RE = re.compile(r"Answer:\s*(-?\d+(?:\.\d+)?)")

BASE_LLM_ID = os.getenv("BASE_LLM_ID", "meta-llama/Llama-3.2-1B")
LORA_R = int(os.getenv("LORA_R", "32"))
LAMBDA_REG = float(os.getenv("LAMBDA_REG", "0.5"))


# --------------------------------------------------------------------------
# model_fn — called ONCE when the container boots.
# --------------------------------------------------------------------------


def model_fn(model_dir: str) -> dict[str, Any]:
    """Load both variants if their checkpoints exist; build the per-PDB index.

    Expected layout inside model.tar.gz:
        v1a/ckpt_final.pt   (or ckpt_ep1.pt — best per TRAINING.md)
        v1b/ckpt_final.pt
        preprocessed/features_test.npz
        preprocessed/samples_test.jsonl
        code/inference.py
        code/requirements.txt
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[model_fn] device={device} model_dir={model_dir}", flush=True)
    _stamp_determinism()

    # MISATOMDQADataset reads from OPENTSLM_MISATO_DATA — point it at the bundle.
    preprocessed = Path(model_dir) / "preprocessed"
    if preprocessed.exists():
        os.environ["OPENTSLM_MISATO_DATA"] = str(preprocessed)

    state: dict[str, Any] = {"device": device, "models": {}, "pdb_index": {}}

    for variant in ("v1a", "v1b"):
        ckpt = _find_ckpt(Path(model_dir) / variant)
        if ckpt is None:
            print(f"[model_fn] {variant}: no checkpoint, skipping", flush=True)
            continue

        print(f"[model_fn] {variant}: loading {ckpt}", flush=True)
        model = OpenTSLMSP(llm_id=BASE_LLM_ID, device=device)
        model.enable_lora(lora_r=LORA_R)
        if variant == "v1b":
            model.enable_regression(weight=LAMBDA_REG)
        model.load_from_file(str(ckpt))
        model.eval()
        state["models"][variant] = {
            "model": model,
            "ckpt": ckpt.name,
            "version": f"{variant}-{ckpt.stem}",
        }
        print(f"[model_fn] {variant}: ready", flush=True)

    # Build pdb_id → sample index once. EOS_TOKEN is per-model — we re-inject
    # at predict time from whichever variant was requested.
    ds = MISATOMDQADataset(split="test", EOS_TOKEN="")
    state["pdb_index"] = {str(s["pdb_id"]): s for s in ds if s.get("pdb_id")}
    print(f"[model_fn] indexed {len(state['pdb_index'])} test PDBs", flush=True)

    if not state["models"]:
        raise RuntimeError("no checkpoints loaded — model.tar.gz incomplete")
    return state


def _find_ckpt(d: Path) -> Path | None:
    if not d.exists():
        return None
    for name in ("ckpt_final.pt", "ckpt_ep5.pt", "ckpt_ep4.pt", "ckpt_ep1.pt"):
        if (d / name).exists():
            return d / name
    pts = sorted(d.glob("ckpt_*.pt"))
    return pts[-1] if pts else None


def _stamp_determinism(seed: int = 0) -> None:
    """Same seed config as inference.py in the local FastAPI."""
    os.environ.setdefault("PYTHONHASHSEED", "0")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


# --------------------------------------------------------------------------
# input_fn — parses the incoming HTTP body.
# --------------------------------------------------------------------------


def input_fn(request_body: bytes | str, content_type: str = "application/json") -> dict:
    if content_type != "application/json":
        raise ValueError(f"unsupported content_type: {content_type}")
    raw = request_body.decode() if isinstance(request_body, (bytes, bytearray)) else request_body
    payload = json.loads(raw)
    if "pdb_id" not in payload or "variant" not in payload:
        raise ValueError("request must include {pdb_id, variant}")
    payload.setdefault("max_new_tokens", 160)
    return payload


# --------------------------------------------------------------------------
# predict_fn — one inference.
# --------------------------------------------------------------------------


def predict_fn(payload: dict, state: dict) -> dict:
    pdb_id = payload["pdb_id"]
    variant = payload["variant"]
    if variant not in state["models"]:
        return {"error": f"variant {variant} not loaded", "loaded": list(state["models"])}

    sample = state["pdb_index"].get(pdb_id)
    if sample is None:
        return {"error": f"{pdb_id} not in MISATO test index"}

    entry = state["models"][variant]
    model = entry["model"]

    # Inject the model's EOS token before patch-padding the batch.
    sample = dict(sample)
    sample["EOS_TOKEN"] = model.get_eos_token()
    batch = extend_time_series_to_match_patch_size_and_aggregate([sample])

    t0 = time.monotonic()
    with torch.no_grad():
        gen = model.generate(batch, max_new_tokens=payload["max_new_tokens"], do_sample=False)
        head_pK = None
        if getattr(model, "regression_enabled", False):
            head_pK = float(model.predict_pK(batch)[0])
    latency_ms = int((time.monotonic() - t0) * 1000)

    rationale = gen[0]
    parsed = _parse_pK(rationale)
    pK = head_pK if (variant == "v1b" and head_pK is not None) else parsed
    if pK is None:
        pK = float("nan")

    return {
        "pdb_id": pdb_id,
        "variant": variant,
        "pK": pK,
        "rationale": rationale,
        "head_pK": head_pK,
        "hidden_pK": float(sample.get("pK")) if "pK" in sample else None,
        "latency_ms": latency_ms,
        "model_version": entry["version"],
    }


def _parse_pK(text: str) -> float | None:
    m = ANSWER_RE.search(text)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


# --------------------------------------------------------------------------
# output_fn — serializes the response.
# --------------------------------------------------------------------------


def output_fn(prediction: dict, accept: str = "application/json") -> tuple[str, str]:
    # Replace NaN with None so the JSON parses cleanly client-side.
    if isinstance(prediction.get("pK"), float) and prediction["pK"] != prediction["pK"]:
        prediction = {**prediction, "pK": None}
    return json.dumps(prediction), "application/json"

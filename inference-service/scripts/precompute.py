"""Precompute worked examples + failure modes — task #20.

Generates static JSON the demo serves without spending OpenRouter $$ per click.

Inputs (env or defaults):
  WORKED_EXAMPLES   "1A1B,4QZL,2X3K"
  N_REPRESENTATIVE  50    — agent-eval this many random test systems
  N_FAILURE_MODES   10    — top-N by |model − vina| disagreement

Outputs (under data/):
  worked_examples.json
  failure_modes_v1a.json
  failure_modes_v1b.json
  eval_cache.jsonl entries (so live UI clicks hit the cache)

Run inside the inference container:
  make precompute
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

# Allow `python scripts/precompute.py` from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import inference
import eval_cache
from orchestrator import evaluate_agent
from tools import TOOL_REGISTRY


DATA = Path(os.getenv("PRECOMPUTE_OUT", "data"))
DATA.mkdir(parents=True, exist_ok=True)


def _versions(predict_result: dict) -> tuple[str, str, str]:
    return (
        predict_result.get("model_version", "unknown"),
        os.getenv("RAG_CORPUS_VERSION", "v1-unset"),
        os.getenv("ANTHROPIC_MODEL", "anthropic/claude-opus-4-7"),
    )


async def _agent_for(pdb_id: str, variant: str) -> dict[str, Any]:
    pred = inference.predict(pdb_id, variant)  # type: ignore[arg-type]
    mv, rcv, jm = _versions(pred)
    cached = eval_cache.get(pdb_id, variant, "agent", mv, rcv, jm)
    if cached:
        print(f"  [{pdb_id}] cache hit, skipping")
        return cached
    verdict, trace = await evaluate_agent(
        pdb_id=pdb_id, model_pK=pred["pK"],
        rationale=pred["rationale"], variant=variant,
    )
    payload = {"verdict": verdict, "trace": trace, "cached": False, "prediction": pred}
    eval_cache.put(pdb_id, variant, "agent", mv, rcv, jm, payload)
    return payload


async def precompute_worked_examples(variant: str) -> None:
    raw = os.getenv("WORKED_EXAMPLES", "1A1B,4QZL,2X3K")
    pdbs = [p.strip() for p in raw.split(",") if p.strip()]
    out = []
    for pid in pdbs:
        if not inference.is_in_test_split(pid):
            print(f"  [{pid}] skipped: not in test split")
            continue
        print(f"  worked-example: {pid} ({variant})")
        try:
            out.append({"pdb_id": pid, "variant": variant, **(await _agent_for(pid, variant))})
        except Exception as e:
            print(f"    failed: {e}")
    path = DATA / f"worked_examples_{variant}.json"
    path.write_text(json.dumps(out, indent=2, sort_keys=True))
    print(f"  wrote {path}")


async def precompute_failure_modes(variant: str) -> None:
    """Compute |model_pK − vina_approx_pK| over a sample of the test split, take top N."""
    n_sample = int(os.getenv("FAILURE_MODE_SAMPLE", "60"))
    n_take = int(os.getenv("N_FAILURE_MODES", "10"))
    all_ids = inference.list_pdb_ids()
    random.seed(42)
    sample = random.sample(all_ids, min(n_sample, len(all_ids)))

    scored: list[dict] = []
    for pid in sample:
        try:
            pred = inference.predict(pid, variant)  # type: ignore[arg-type]
        except Exception as e:
            print(f"  [{pid}] predict failed: {e}")
            continue
        vina = TOOL_REGISTRY["vina_rescore"](pdb_id=pid, frame_idx=50)
        if "error" in vina:
            print(f"  [{pid}] vina skipped: {vina['error']}")
            continue
        delta = abs(pred["pK"] - vina["approx_pK"])
        scored.append({
            "pdb": pid, "model": pred["pK"], "vina": vina["vina_kcal_mol"],
            "approx_pK": vina["approx_pK"], "delta": delta,
        })

    scored.sort(key=lambda r: r["delta"], reverse=True)
    top = scored[:n_take]
    print(f"  computed {len(scored)} disagreements; taking top {len(top)}")

    # Run agent on each
    rows = []
    for r in top:
        pid = r["pdb"]
        print(f"  agent on {pid} (delta={r['delta']:.2f})")
        try:
            agent = await _agent_for(pid, variant)
        except Exception as e:
            print(f"    failed: {e}")
            continue
        rec = agent["verdict"].get("recommendation", "review")
        reasons = agent["verdict"].get("contradicted_claims") or agent["verdict"].get("missing_claims") or []
        reason_text = ""
        if reasons:
            first = reasons[0]
            reason_text = first.get("contradicting_evidence") or first.get("why_relevant") or ""
        rows.append({
            "pdb": pid,
            "model": round(r["model"], 1),
            "vina": round(-r["vina"] / 1.36, 1),  # vina kcal/mol → pK
            "mmgbsa": round((-r["vina"] / 1.36 + r["model"]) / 2, 1),  # placeholder until MM-GBSA tool
            "agent": rec,
            "reason": reason_text[:200],
        })

    # Aggregate failure patterns — cluster reasons by simple keyword bucket
    patterns = _bucket_patterns(rows)

    payload = {
        "variant": variant,
        "rows": rows,
        "patterns": patterns,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    path = DATA / f"failure_modes_{variant}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"  wrote {path}")


def _bucket_patterns(rows: list[dict]) -> list[dict]:
    buckets: dict[str, list[str]] = {
        "Implausible ligand efficiency": [],
        "Pose unstable (>2 clusters)": [],
        "Single-frame outlier dominates": [],
        "Literature contradicts binding mode": [],
        "Other": [],
    }
    for r in rows:
        reason = (r.get("reason") or "").lower()
        if "ligand efficien" in reason or "le " in reason or "implausible" in reason:
            buckets["Implausible ligand efficiency"].append(r["pdb"])
        elif "pose" in reason and ("unstable" in reason or "cluster" in reason or "split" in reason):
            buckets["Pose unstable (>2 clusters)"].append(r["pdb"])
        elif "outlier" in reason or "single frame" in reason or "clash" in reason:
            buckets["Single-frame outlier dominates"].append(r["pdb"])
        elif "literature" in reason or "contradict" in reason:
            buckets["Literature contradicts binding mode"].append(r["pdb"])
        else:
            buckets["Other"].append(r["pdb"])
    return [
        {"cluster": k, "count": len(v), "systems": ", ".join(v) or "—"}
        for k, v in buckets.items()
        if v
    ]


async def precompute_representative(variant: str) -> None:
    n = int(os.getenv("N_REPRESENTATIVE", "50"))
    all_ids = inference.list_pdb_ids()
    random.seed(0)
    sample = random.sample(all_ids, min(n, len(all_ids)))
    for i, pid in enumerate(sample, 1):
        print(f"  [{i}/{len(sample)}] representative agent eval: {pid}")
        try:
            await _agent_for(pid, variant)
        except Exception as e:
            print(f"    failed: {e}")


async def main() -> None:
    # Warm up the model loader the same way the FastAPI lifespan does.
    inference.warm_up(
        os.getenv("CHECKPOINT_DIR", "/app/checkpoints"),
        os.getenv("TEST_SPLIT_PATH", "/app/data/test_MD.txt"),
    )
    loaded = inference.variants_loaded()
    if not loaded:
        raise SystemExit("no model variants loaded; mount checkpoints and try again")

    for variant in loaded:
        print(f"\n=== variant {variant} ===")
        await precompute_worked_examples(variant)
        await precompute_failure_modes(variant)
        if os.getenv("PRECOMPUTE_REPRESENTATIVE", "0") == "1":
            await precompute_representative(variant)


if __name__ == "__main__":
    asyncio.run(main())

"""Persistent JSON cache for /evaluate and /evaluate/agent — task #18.

Cache key includes EVERY version that could change the verdict:
  (pdb_id, variant, model_version, rag_corpus_version, judge_model, mode)

So a model retrain, RAG re-ingest, prompt edit, or judge change invalidates
cleanly. Backing: data/eval_cache.jsonl (append-only) + in-memory dict loaded
at startup. Easy to nuke entries by `jq` filtering.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_LOCK = threading.Lock()
_CACHE: dict[str, dict[str, Any]] = {}
_LOADED = False


def _path() -> Path:
    p = Path(os.getenv("EVAL_CACHE_PATH", "data/eval_cache.jsonl"))
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _key(
    pdb_id: str, variant: str, mode: str,
    model_version: str, rag_corpus_version: str, judge_model: str,
) -> str:
    raw = f"{pdb_id}|{variant}|{mode}|{model_version}|{rag_corpus_version}|{judge_model}"
    return hashlib.sha256(raw.encode()).hexdigest()


def _load() -> None:
    global _LOADED
    if _LOADED:
        return
    path = _path()
    if path.exists():
        with path.open() as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    _CACHE[entry["key"]] = entry
                except Exception:
                    continue
    _LOADED = True


def get(
    pdb_id: str, variant: str, mode: str,
    model_version: str, rag_corpus_version: str, judge_model: str,
) -> dict[str, Any] | None:
    _load()
    k = _key(pdb_id, variant, mode, model_version, rag_corpus_version, judge_model)
    entry = _CACHE.get(k)
    if entry is None:
        return None
    return entry["value"]


def put(
    pdb_id: str, variant: str, mode: str,
    model_version: str, rag_corpus_version: str, judge_model: str,
    value: dict[str, Any],
) -> None:
    _load()
    k = _key(pdb_id, variant, mode, model_version, rag_corpus_version, judge_model)
    entry = {
        "key": k,
        "pdb_id": pdb_id,
        "variant": variant,
        "mode": mode,
        "model_version": model_version,
        "rag_corpus_version": rag_corpus_version,
        "judge_model": judge_model,
        "ts": datetime.now(timezone.utc).isoformat(),
        "value": value,
    }
    with _LOCK:
        _CACHE[k] = entry
        with _path().open("a") as f:
            f.write(json.dumps(entry) + "\n")

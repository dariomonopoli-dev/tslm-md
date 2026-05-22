"""Spend tracking + daily cap — task #19.

OpenRouter does NOT return $ amounts. We compute spend locally from token
counts × the price table in `pricing.py`. Persisted append-only to
`data/spend_log.jsonl` so the cap survives container restarts.

Reset boundary: midnight UTC.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict


_LOG_LOCK = threading.Lock()


class SpendEvent(TypedDict, total=False):
    ts: str
    pdb_id: str
    variant: str
    model: str
    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int
    usd: float
    cached: bool
    tool_calls: int


def _log_path() -> Path:
    p = Path(os.getenv("SPEND_LOG_PATH", "data/spend_log.jsonl"))
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _daily_cap_usd() -> float:
    return float(os.getenv("OPENROUTER_DAILY_USD_CAP", "20"))


def record(event: SpendEvent) -> None:
    """Append a spend event (one JSON object per line)."""
    event.setdefault("ts", datetime.now(timezone.utc).isoformat())
    with _LOG_LOCK:
        with _log_path().open("a") as f:
            f.write(json.dumps(event) + "\n")


def _today_utc_date() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def spend_today_usd() -> float:
    """Sum of usd in spend_log.jsonl for the current UTC day."""
    path = _log_path()
    if not path.exists():
        return 0.0
    today = _today_utc_date()
    total = 0.0
    with path.open() as f:
        for line in f:
            try:
                e = json.loads(line)
            except Exception:
                continue
            ts = str(e.get("ts", ""))
            if ts.startswith(today):
                total += float(e.get("usd", 0.0))
    return total


def remaining_cap_usd() -> float:
    return max(0.0, _daily_cap_usd() - spend_today_usd())


def check_or_429() -> None:
    """Raise HTTPException 429 if the daily cap is exhausted.

    Called from inside route handlers BEFORE the LLM call. Cached responses
    skip this check (no spend incurred).
    """
    if remaining_cap_usd() <= 0.0:
        from fastapi import HTTPException
        midnight = (
            datetime.now(timezone.utc)
            .replace(hour=23, minute=59, second=59, microsecond=0)
            .isoformat()
        )
        raise HTTPException(
            status_code=429,
            detail={
                "error": "daily OpenRouter spend cap reached",
                "cap_usd": _daily_cap_usd(),
                "spend_today_usd": spend_today_usd(),
                "reset_at": midnight,
            },
        )

"""Per-IP rate limiting for /evaluate/agent — task #19.

slowapi (FastAPI-compatible flask-limiter port). Default: 10
/evaluate/agent calls per hour per IP. Override via env var.

A bearer token in `DEMO_TOKEN` bypasses the IP limit — useful for screen
recordings without losing the public rate limit.
"""

from __future__ import annotations

import os

from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address


def _key_func(request: Request) -> str:
    auth = request.headers.get("authorization", "")
    demo_token = os.getenv("DEMO_TOKEN", "")
    if demo_token and auth == f"Bearer {demo_token}":
        return "__demo_token__"  # all demo-token requests share one bucket
    return get_remote_address(request)


limiter = Limiter(
    key_func=_key_func,
    default_limits=[],  # we apply limits per-route via decorator
)


AGENT_LIMIT = os.getenv("EVAL_AGENT_PER_IP_HOURLY", "10") + "/hour"
PREDICT_LIMIT = os.getenv("PREDICT_PER_IP_HOURLY", "120") + "/hour"

"""OpenRouter Anthropic-compatible client — task #14.

Thin async wrapper around POST /api/v1/messages. We use OpenRouter's
Anthropic adapter (cleaner tool-use shape than /chat/completions).

Key behaviours:
  - `transforms: []` disables OpenRouter's automatic prompt rewriting so
    tool-use order is preserved (determinism matters for the demo).
  - stop_reason normalized: OpenRouter sometimes returns "stop" instead of
    "end_turn"; we map both to a single internal enum.
  - Usage tracked (input/output/cache-read tokens) so the spend-cap middleware
    (task #19) can compute $ from llm.pricing.PRICES.

Auth via OPENROUTER_API_KEY env var. The HTTP-Referer + X-Title headers are
OpenRouter convention — they show up in your dashboard analytics.
"""

from __future__ import annotations

import os
from typing import Any, Literal, TypedDict

import httpx
from tenacity import (
    AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential,
)

from .pricing import compute_usd


_BASE_URL = "https://openrouter.ai/api/v1"
_DEFAULT_TIMEOUT_S = 60.0


StopReason = Literal["end_turn", "tool_use", "max_tokens", "stop_sequence", "other"]


class Usage(TypedDict):
    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int
    cost_usd: float


class MessageResponse(TypedDict):
    content: list[dict[str, Any]]  # [{type: 'text'|'tool_use', ...}, ...]
    stop_reason: StopReason
    usage: Usage
    raw: dict[str, Any]


class OpenRouterError(RuntimeError):
    """Raised on non-2xx responses; carries status + parsed body."""

    def __init__(self, status: int, body: Any, message: str | None = None):
        super().__init__(message or f"OpenRouter {status}: {body}")
        self.status = status
        self.body = body


def _normalize_stop(raw: str | None) -> StopReason:
    if raw in ("end_turn", "tool_use", "max_tokens", "stop_sequence"):
        return raw  # type: ignore[return-value]
    if raw in ("stop", "end_sequence", None):
        return "end_turn"
    if raw == "length":
        return "max_tokens"
    return "other"


def _headers() -> dict[str, str]:
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise OpenRouterError(0, None, "OPENROUTER_API_KEY not set")
    return {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": os.getenv("OPENROUTER_REFERER", "http://localhost:3000"),
        "X-Title": os.getenv("OPENROUTER_TITLE", "MoleMotion"),
    }


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPError):
        return True
    if isinstance(exc, OpenRouterError):
        return 500 <= exc.status < 600
    return False


async def messages_create(
    *,
    system: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    max_tokens: int = 2048,
    temperature: float = 0.0,
    model: str | None = None,
    extra_body: dict[str, Any] | None = None,
    timeout_s: float = _DEFAULT_TIMEOUT_S,
) -> MessageResponse:
    """Call OpenRouter's Anthropic-compatible /messages endpoint.

    Returns the full content block list so the caller (orchestrator) can
    iterate tool_use entries directly. Determinism: temperature=0 by default.
    """
    model_id = model or os.getenv("ANTHROPIC_MODEL", "anthropic/claude-opus-4-7")

    body: dict[str, Any] = {
        "model": model_id,
        "system": system,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "transforms": [],  # disable OpenRouter prompt rewriting
    }
    if tools:
        body["tools"] = tools
    if extra_body:
        body.update(extra_body)

    async def _do_call() -> MessageResponse:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            resp = await client.post(
                f"{_BASE_URL}/messages",
                headers=_headers(),
                json=body,
            )
            try:
                parsed = resp.json()
            except Exception:
                parsed = {"raw_text": resp.text}

            if resp.status_code >= 400:
                raise OpenRouterError(resp.status_code, parsed)

            usage_raw = parsed.get("usage", {}) or {}
            input_tokens = int(usage_raw.get("input_tokens", 0))
            output_tokens = int(usage_raw.get("output_tokens", 0))
            cache_read = int(usage_raw.get("cache_read_input_tokens", 0))
            cost = compute_usd(model_id, input_tokens, output_tokens, cache_read)

            return MessageResponse(
                content=parsed.get("content", []) or [],
                stop_reason=_normalize_stop(parsed.get("stop_reason")),
                usage=Usage(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cache_read_input_tokens=cache_read,
                    cost_usd=cost,
                ),
                raw=parsed,
            )

    last_exc: BaseException | None = None
    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(2),  # one retry on 5xx / network
        wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
        retry=retry_if_exception_type((httpx.HTTPError, OpenRouterError)),
        reraise=True,
    ):
        with attempt:
            try:
                return await _do_call()
            except OpenRouterError as e:
                last_exc = e
                if not _is_retryable(e):
                    raise

    # unreachable in practice — AsyncRetrying with reraise=True propagates the last error
    raise last_exc or RuntimeError("messages_create exhausted retries")

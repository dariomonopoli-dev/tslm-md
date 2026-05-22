"""OpenRouter spend computation — task #19.

OpenRouter does NOT return $ amounts. We compute spend locally from token
counts × the price table below. Update this table when prices change.
Prices are USD per 1M tokens.
"""

from __future__ import annotations


PRICES: dict[str, dict[str, float]] = {
    # Verify against https://openrouter.ai/models before relying on these.
    "anthropic/claude-opus-4-7": {
        "input": 15.00,
        "output": 75.00,
        "cache_read": 1.50,
    },
    "openai/text-embedding-3-small": {
        "input": 0.02,
        "output": 0.0,
        "cache_read": 0.0,
    },
}


def compute_usd(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_input_tokens: int = 0,
) -> float:
    price = PRICES[model]
    return (
        (input_tokens - cache_read_input_tokens) * price["input"] / 1_000_000
        + cache_read_input_tokens * price["cache_read"] / 1_000_000
        + output_tokens * price["output"] / 1_000_000
    )

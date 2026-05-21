"""Parse LM-generated output into structured fields."""

from __future__ import annotations

import re
from typing import Optional

_PATTERN = re.compile(
    r"Answer:\s*(-?\d+(?:\.\d+)?)\s*kcal\s*/\s*mol[\.\s,]*Confidence:\s*(high|medium|low)",
    re.IGNORECASE | re.DOTALL,
)


def parse_answer(text: str) -> tuple[Optional[float], Optional[str]]:
    """Extract (affinity_kcal_mol, confidence) from a model completion.

    Returns (None, None) if parsing fails.

    Handles minor model whitespace/punctuation variation but expects the
    template established in prompts.POST_PROMPT.
    """
    m = _PATTERN.search(text or "")
    if not m:
        return None, None
    try:
        return float(m.group(1)), m.group(2).lower()
    except (TypeError, ValueError):
        return None, None

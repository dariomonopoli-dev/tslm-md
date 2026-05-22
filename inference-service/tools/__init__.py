"""Agent tool registry.

Tools register themselves into TOOL_REGISTRY (name → callable) and
TOOL_SCHEMAS (list of JSON schemas for the LLM) at import time via the
@register decorator. See tools/splits.py for the reference pattern.
"""

from __future__ import annotations

from typing import Any, Callable


TOOL_REGISTRY: dict[str, Callable[..., Any]] = {}
TOOL_SCHEMAS: list[dict[str, Any]] = []


def register(schema: dict[str, Any]):
    """Decorator: registers a function as a tool with its JSON schema."""

    def _wrap(fn: Callable[..., Any]) -> Callable[..., Any]:
        name = schema["name"]
        if name in TOOL_REGISTRY:
            raise ValueError(f"duplicate tool: {name}")
        TOOL_REGISTRY[name] = fn
        TOOL_SCHEMAS.append(schema)
        return fn

    return _wrap


# Import side effects: register all tools.
from . import splits  # noqa: E402, F401
from . import coords  # noqa: E402, F401
from . import chemistry  # noqa: E402, F401
from . import physics  # noqa: E402, F401

# rag.store registers a `rag_query` tool. Import here so the tool catalog is
# complete after `import tools`. Circular-safe: `register` is already defined
# above by the time rag.store imports it from us.
try:
    from rag import store  # noqa: E402, F401
except Exception as e:  # pragma: no cover — chromadb may be missing in test runs
    import sys
    print(f"[tools] rag.store import skipped: {e}", file=sys.stderr)

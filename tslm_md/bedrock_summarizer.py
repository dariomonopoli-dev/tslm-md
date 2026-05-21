"""Optional second-opinion summariser via AWS Bedrock Claude.

Wired into the agent loop only at hour 20+ if time permits.
Never on critical path.
"""

from __future__ import annotations

from typing import Optional


def summarise_via_bedrock(
    rationale: str,
    affinity: Optional[float],
    confidence: Optional[str],
    verdict: str,
    region: str = "us-east-1",
    model_id: str = "anthropic.claude-haiku-4-5-20251001-v1:0",
) -> str:
    """Hand the agent's Report to Claude on Bedrock for a polished one-paragraph summary."""
    raise NotImplementedError(
        "Wire boto3 bedrock-runtime client at hour 20+ if time permits."
    )

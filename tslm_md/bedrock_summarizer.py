"""Optional second-opinion summariser via AWS Bedrock Claude.

Hands the agent's structured Report to Claude on Bedrock for a polished
one-paragraph natural-language summary at demo time.

This is intentionally optional and demo-safe: if Bedrock is unavailable
(no creds, wrong region, model not enabled), the function falls back to
returning the raw deterministic rationale so the demo never crashes.

Wire into agent.py at hour 20+ if time permits.
"""

from __future__ import annotations

import json
from typing import Optional

import boto3
import botocore

DEFAULT_MODEL_ID = "anthropic.claude-haiku-4-5-20251001-v1:0"
DEFAULT_REGION = "us-east-1"

SYSTEM_PROMPT = """You are an assistant helping a computational chemist read the output of a binding-affinity prediction agent.

You will receive:
  - A predicted binding affinity (kcal/mol)
  - The model's stated confidence (high / medium / low)
  - An independent physics-based energy estimate
  - The agent's verdict (CONFIRMED or INCONCLUSIVE)
  - A grounded summary of the molecular dynamics trajectory

Write ONE short paragraph (50-80 words) that:
  1. States the predicted affinity and whether it indicates a strong, moderate, or weak binder
  2. Notes whether the physics-based verifier agrees with the model
  3. If INCONCLUSIVE, plainly tells the chemist what to do next (longer MD? re-dock? human review?)

No lists. No headings. Plain professional paragraph. No hedging."""


def summarise_via_bedrock(
    rationale: str,
    affinity: Optional[float],
    confidence: Optional[str],
    independent_energy: Optional[float],
    verdict: str,
    region: str = DEFAULT_REGION,
    model_id: str = DEFAULT_MODEL_ID,
    max_tokens: int = 200,
) -> str:
    """Return a one-paragraph Claude summary of the agent's Report.

    Falls back to the raw rationale if Bedrock is unreachable so the demo
    never crashes.
    """
    user_msg = (
        f"Predicted affinity: {affinity if affinity is not None else 'unknown'} kcal/mol\n"
        f"Confidence: {confidence or 'unknown'}\n"
        f"Independent physics energy: "
        f"{independent_energy if independent_energy is not None else 'unknown'}\n"
        f"Verdict: {verdict}\n\n"
        f"Grounded trajectory summary:\n{rationale}"
    )

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": user_msg}],
    }

    try:
        client = boto3.client("bedrock-runtime", region_name=region)
        response = client.invoke_model(
            modelId=model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(body),
        )
        payload = json.loads(response["body"].read())
        # Claude Messages API returns content as a list of typed blocks
        for block in payload.get("content", []):
            if block.get("type") == "text":
                return block["text"].strip()
        return rationale  # malformed response
    except (
        botocore.exceptions.BotoCoreError,
        botocore.exceptions.ClientError,
        KeyError,
        IndexError,
        ValueError,
    ) as exc:
        return f"{rationale}\n\n[Bedrock summariser unavailable: {type(exc).__name__}]"

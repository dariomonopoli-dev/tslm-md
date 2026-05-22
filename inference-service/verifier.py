"""Deterministic regex rationale verifier — task #8.

Thin wrapper around verify_rationale.py (vendored from project root). Adds:
  - per-PDB facts lookup from preprocessed/samples_test.jsonl
  - response-shape conversion to the §8.1 PredictResponse.regex_verifier schema
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal, TypedDict

import verify_rationale as vr  # vendored


ClaimStatus = Literal["verified", "contradicted", "unverifiable"]


class Claim(TypedDict):
    text: str
    status: ClaimStatus
    evidence: str


class VerifierResult(TypedDict):
    verified: int
    contradicted: int
    unverifiable: int
    total: int
    claims: list[Claim]


_FACTS: dict[str, dict] = {}


def _load_facts() -> dict[str, dict]:
    """Build a pdb_id → facts lookup from preprocessed/samples_test.jsonl.

    Cached after first call; populated lazily so app boot does not require
    the preprocessed corpus to exist (the verifier silently degrades to
    'unverifiable' for unknown PDBs).
    """
    global _FACTS
    if _FACTS:
        return _FACTS

    candidates = [
        Path(os.getenv("MISATO_SAMPLES_TEST", "")),
        Path("/app/data/samples_test.jsonl"),
        Path("preprocessed/samples_test.jsonl"),
        Path("../preprocessed/samples_test.jsonl"),
    ]
    for path in candidates:
        if path and path.exists():
            out: dict[str, dict] = {}
            with path.open() as f:
                for line in f:
                    s = json.loads(line)
                    facts = s.get("facts", {})
                    facts.setdefault("pdb_id", s.get("pdb_id"))
                    if s.get("pdb_id"):
                        out[s["pdb_id"]] = facts
            _FACTS = out
            return _FACTS
    return {}


def verify(rationale: str, pdb_id: str) -> VerifierResult:
    facts = _load_facts().get(pdb_id)
    if facts is None:
        # No facts → cannot grade. Frontend renders an "unverifiable" stripe
        # without misleading users into thinking the model was contradicted.
        return VerifierResult(
            verified=0, contradicted=0, unverifiable=0, total=0, claims=[],
        )

    report = vr.verify_rationale(rationale, facts)
    counts = report.counts()
    claims: list[Claim] = [
        {
            "text": f"{v.claim_type}.{v.field}: claimed={v.claimed}, actual={v.actual}",
            "status": v.status,  # type: ignore[typeddict-item]
            "evidence": v.detail,
        }
        for v in report.verdicts
    ]
    return VerifierResult(
        verified=counts["verified"],
        contradicted=counts["contradicted"],
        unverifiable=counts["unverifiable"],
        total=len(report.verdicts),
        claims=claims,
    )

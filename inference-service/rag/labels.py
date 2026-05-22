"""Label-tagging regex — the SAFETY CRITICAL piece of the RAG pipeline.

Every chunk that mentions a numerical binding-affinity value is tagged with
`contains_label=True` and the set of PDB IDs the value pertains to. At
retrieval time, chunks where (contains_label AND pdb_id_under_test ∈ pdb_ids)
are excluded — otherwise the agent can read the ground-truth answer.

Tested by tests/rag/test_label_filter.py against a 50-leaky-chunk fixture
(task #13). Do NOT loosen the patterns without re-running that suite.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


# Match a 4-character PDB code, anchored on word boundaries.
PDB_ID_RE = re.compile(r"\b([0-9][A-Za-z0-9]{3})\b")

# Numeric label patterns. We deliberately over-match (units in nM/μM/M, scientific
# notation, pK/pKi/pIC50) — false positives are acceptable; false negatives are not.
# The metric-name → number window is bounded at ~60 chars so we don't accidentally
# pair "Kd" with an unrelated number further down the paragraph.
LABEL_PATTERNS = [
    # "Kd = 12 nM", "Kd of 3.2 µM", "Kd value was determined as 100 pM"
    re.compile(
        r"\b(Kd|Ki|IC50|EC50|Kb)\b[^.;\n]{0,60}?"
        r"(\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*"
        r"(nM|µM|uM|μM|pM|mM|M\b)",
        re.IGNORECASE,
    ),
    # "pK = 6.4", "pKi 7.1", "pIC50: 8.0"
    re.compile(
        r"\b(pK[adib]?|pIC50|pEC50)\b\s*(?:=|of|:|is|was)?\s*"
        r"(-?\d+(?:\.\d+)?)",
        re.IGNORECASE,
    ),
    # "binding affinity of 50 nM" — common in PubMed abstracts
    re.compile(
        r"\bbinding\s+affinit(?:y|ies)\s+of\s+\d+(?:\.\d+)?\s*"
        r"(?:nM|µM|uM|μM|pM|mM)",
        re.IGNORECASE,
    ),
]


@dataclass
class LabelTag:
    contains_label: bool
    pdb_ids: list[str]


def _pdb_set(text: str) -> set[str]:
    return {m.group(1).upper() for m in PDB_ID_RE.finditer(text)}


def tag_chunk(text: str, explicit_pdb_ids: list[str] | None = None) -> LabelTag:
    """Inspect chunk text; return (contains_label, pdb_ids it pertains to).

    `explicit_pdb_ids` — if the chunk's source row already names PDBs (e.g.
    a CSV with PDBid column), pass them; we union with whatever the regex
    finds in the text.
    """
    contains = any(p.search(text) for p in LABEL_PATTERNS)
    found_pdbs = _pdb_set(text)
    if explicit_pdb_ids:
        found_pdbs.update(p.upper() for p in explicit_pdb_ids)
    return LabelTag(contains_label=contains, pdb_ids=sorted(found_pdbs))


def is_leak_for(chunk_tag: LabelTag, query_pdb_id: str) -> bool:
    """Strict filter predicate used by rag_query."""
    return chunk_tag.contains_label and query_pdb_id.upper() in chunk_tag.pdb_ids

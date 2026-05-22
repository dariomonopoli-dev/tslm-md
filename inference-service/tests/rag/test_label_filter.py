"""Regression test: label filter must NEVER leak a Kd/Ki/IC50 for the system
under test, AND must NOT over-filter chunks that don't contain a numeric label.

Per FRONTEND_v2.md §15 risk 4 — BLOCKING: /evaluate/agent cannot ship without
this test passing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag.labels import tag_chunk, is_leak_for


FIXTURE_DIR = Path(__file__).parent


def _load(name: str) -> list[dict]:
    path = FIXTURE_DIR / name
    if not path.exists():
        pytest.skip(f"{path} missing")
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


LEAKY = _load("leaky_chunks.jsonl")
SAFE = _load("safe_chunks.jsonl")


@pytest.mark.parametrize("entry", LEAKY, ids=lambda e: f"{e['pdb_id']}_{e['kind']}")
def test_leak_detected(entry):
    """Every leaky chunk must be flagged as contains_label AND name its pdb_id."""
    tag = tag_chunk(entry["text"], explicit_pdb_ids=[entry["pdb_id"]])
    assert tag.contains_label, f"missed label in: {entry['text']}"
    assert entry["pdb_id"].upper() in tag.pdb_ids, (
        f"pdb_id {entry['pdb_id']} not associated with the chunk"
    )


@pytest.mark.parametrize("entry", LEAKY, ids=lambda e: f"{e['pdb_id']}_{e['kind']}")
def test_leak_filtered_for_target(entry):
    """is_leak_for(chunk, target_pdb) must be True for the named PDB."""
    tag = tag_chunk(entry["text"], explicit_pdb_ids=[entry["pdb_id"]])
    assert is_leak_for(tag, entry["pdb_id"])


@pytest.mark.parametrize("entry", LEAKY, ids=lambda e: f"{e['pdb_id']}_{e['kind']}")
def test_leak_not_overfiltered_for_unrelated(entry):
    """is_leak_for(chunk, OTHER_pdb) must be False — comparators stay visible."""
    tag = tag_chunk(entry["text"], explicit_pdb_ids=[entry["pdb_id"]])
    # An obviously unrelated PDB (never appears in the fixture).
    assert not is_leak_for(tag, "ZZZZ")


@pytest.mark.parametrize("entry", SAFE, ids=lambda e: f"{e['pdb_id']}_{e['kind']}")
def test_safe_not_flagged_as_leak(entry):
    """Chunks WITHOUT a numeric label must NOT trip contains_label."""
    tag = tag_chunk(entry["text"], explicit_pdb_ids=[entry["pdb_id"]])
    assert not tag.contains_label, (
        f"false positive label on safe chunk: {entry['text']}"
    )


def test_co_mention_filters_both_pdbs():
    """A chunk mentioning two PDBs alongside a Kd should filter for both."""
    text = "1A1B mentioned alongside 2X3K with Kd = 320 nM."
    tag = tag_chunk(text)
    assert tag.contains_label
    assert "1A1B" in tag.pdb_ids and "2X3K" in tag.pdb_ids
    assert is_leak_for(tag, "1A1B")
    assert is_leak_for(tag, "2X3K")
    assert not is_leak_for(tag, "4QZL")

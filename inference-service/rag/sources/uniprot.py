"""Fetch UniProt protein descriptions for the RAG corpus.

Pulls function + family + similarity comments per unique UniProt ID in
affinity_data.csv. Writes `data/rag_sources/uniprot.jsonl`, one line per
UniProt, with the PDB IDs that map to that protein.

The `rag/ingest.py` pipeline picks these up alongside the existing
markdown + affinity-row sources.

Usage:
    python -m rag.sources.uniprot
    python -m rag.sources.uniprot --limit 50
"""

from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path


UNIPROT_BASE = "https://rest.uniprot.org/uniprotkb"
RATE_LIMIT_S = 0.1


def _collect_uniprots(affinity_csv: Path) -> dict[str, list[str]]:
    """Return uniprot → [pdb_id, ...]."""
    out: dict[str, list[str]] = defaultdict(list)
    with affinity_csv.open() as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            uni = (row.get("Uniprot") or "").strip()
            pid = (row.get("PDBid") or "").strip()
            if uni and pid and uni.upper() not in ("NAN", "NA", ""):
                out[uni].append(pid)
    return dict(out)


def _fetch_one(uni: str) -> dict | None:
    url = f"{UNIPROT_BASE}/{uni}.json"
    try:
        with urllib.request.urlopen(url, timeout=20) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        print(f"  uniprot {uni}: HTTP {e.code}")
        return None
    except Exception as e:
        print(f"  uniprot {uni}: {e}")
        return None


def _extract_text(data: dict) -> str:
    """Build a compact natural-language summary the embedder + agent can use."""
    name = ((data.get("proteinDescription") or {}).get("recommendedName") or {}).get("fullName", {}).get("value", "")
    organism = (data.get("organism") or {}).get("scientificName", "")
    families = []
    for c in data.get("comments", []) or []:
        if c.get("commentType") == "SIMILARITY":
            for t in c.get("texts", []) or []:
                families.append(t.get("value", ""))
    functions = []
    for c in data.get("comments", []) or []:
        if c.get("commentType") == "FUNCTION":
            for t in c.get("texts", []) or []:
                functions.append(t.get("value", ""))
    parts = [f"{name} ({organism})."] if name else []
    if families:
        parts.append("Family: " + "; ".join(families))
    if functions:
        parts.append("Function: " + " ".join(functions))
    return " ".join(parts).strip()


def fetch_all(affinity_csv: Path, out_jsonl: Path, limit: int | None = None) -> None:
    uniprot_to_pdbs = _collect_uniprots(affinity_csv)
    if limit:
        uniprot_to_pdbs = dict(list(uniprot_to_pdbs.items())[:limit])
    print(f"[uniprot] {len(uniprot_to_pdbs)} unique uniprot IDs")

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out_jsonl.open("w") as f:
        for i, (uni, pdbs) in enumerate(uniprot_to_pdbs.items(), 1):
            data = _fetch_one(uni)
            time.sleep(RATE_LIMIT_S)
            if not data:
                continue
            text = _extract_text(data)
            if not text:
                continue
            f.write(json.dumps({
                "kind": "uniprot",
                "uniprot": uni,
                "pdb_ids": sorted(set(pdbs)),
                "text": text,
            }) + "\n")
            written += 1
            if i % 50 == 0:
                print(f"  {i}/{len(uniprot_to_pdbs)} … {written} written")
    print(f"[uniprot] wrote {written} entries → {out_jsonl}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--affinity-csv", type=Path,
                   default=Path("../misato-affinity/data/affinity_data.csv"))
    p.add_argument("--out", type=Path, default=Path("data/rag_sources/uniprot.jsonl"))
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()
    if not args.affinity_csv.exists():
        raise SystemExit(f"affinity csv not found: {args.affinity_csv}")
    fetch_all(args.affinity_csv, args.out, args.limit)


if __name__ == "__main__":
    main()

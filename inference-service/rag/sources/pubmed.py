"""Fetch the primary-citation PubMed abstract for each PDB.

Two-step lookup that gives the AUTHORITATIVE paper (the one published with
the structure), not random PubMed hits matching the 4-character PDB code:

  1. RCSB GraphQL → entry(entry_id: PDB).rcsb_primary_citation.pdbx_database_id_PubMed
     → returns PMID per PDB (batched, N PDBs per call)
  2. NCBI efetch → abstract text for those PMIDs

Writes data/rag_sources/pubmed.jsonl with one line per unique abstract,
including the list of PDB IDs the paper covers (so the label-filter
regex in rag.labels correctly redacts leaks per query PDB).

Usage:
    python -m rag.sources.pubmed
    python -m rag.sources.pubmed --limit 50
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path


RCSB_GRAPHQL = "https://data.rcsb.org/graphql"
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
RCSB_BATCH = 50      # PDBs per RCSB GraphQL request
EFETCH_BATCH = 50    # PMIDs per efetch request
RATE_LIMIT_S = 0.35  # NCBI: 3 req/s without API key


def _api_key_qs() -> str:
    key = os.getenv("NCBI_API_KEY", "")
    return f"&api_key={key}" if key else ""


def _read_pdb_ids(affinity_csv: Path) -> list[str]:
    ids: list[str] = []
    with affinity_csv.open() as f:
        for row in csv.DictReader(f, delimiter=";"):
            pid = (row.get("PDBid") or "").strip()
            if pid:
                ids.append(pid)
    return ids


def _rcsb_pmids(pdb_ids: list[str]) -> dict[str, str | None]:
    """Returns {pdb_id: pmid|None} via one batched GraphQL request."""
    aliases = ", ".join(
        f'e{i}: entry(entry_id: "{pid}") {{ '
        f'rcsb_primary_citation {{ pdbx_database_id_PubMed title }} '
        f'}}'
        for i, pid in enumerate(pdb_ids)
    )
    query = f"{{ {aliases} }}"
    req = urllib.request.Request(
        RCSB_GRAPHQL,
        data=json.dumps({"query": query}).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.loads(r.read())
    except Exception as e:
        print(f"  RCSB graphql: {e}")
        return {pid: None for pid in pdb_ids}

    out: dict[str, str | None] = {}
    payload = data.get("data") or {}
    for i, pid in enumerate(pdb_ids):
        entry = payload.get(f"e{i}")
        if not entry:
            out[pid] = None
            continue
        cite = entry.get("rcsb_primary_citation") or {}
        pmid = cite.get("pdbx_database_id_PubMed")
        out[pid] = str(pmid) if pmid else None
    return out


def _efetch_abstracts(pmids: list[str]) -> dict[str, dict]:
    if not pmids:
        return {}
    url = (
        f"{EUTILS}/efetch.fcgi?db=pubmed&id={','.join(pmids)}"
        f"&rettype=abstract&retmode=xml{_api_key_qs()}"
    )
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            xml = r.read()
    except Exception as e:
        print(f"  efetch: {e}")
        return {}

    out: dict[str, dict] = {}
    try:
        root = ET.fromstring(xml)
    except ET.ParseError as e:
        print(f"  xml parse: {e}")
        return {}

    for article in root.findall(".//PubmedArticle"):
        # Note: ElementTree elements that are FOUND but childless are falsy
        # in Python 3.12. Use `is None` for existence checks.
        pmid_el = article.find(".//PMID")
        title_el = article.find(".//ArticleTitle")
        abstract_parts = []
        for ab in article.findall(".//Abstract/AbstractText"):
            label = ab.get("Label") or ""
            text = "".join(ab.itertext()).strip()
            if not text:
                continue
            abstract_parts.append(f"{label}: {text}" if label else text)
        if pmid_el is None:
            continue
        pmid = pmid_el.text or ""
        title = "".join(title_el.itertext()).strip() if title_el is not None else ""
        out[pmid] = {
            "pmid": pmid,
            "title": title,
            "abstract": " ".join(abstract_parts),
        }
    return out


def fetch_all(affinity_csv: Path, out_jsonl: Path, limit: int | None = None) -> None:
    pdb_ids = _read_pdb_ids(affinity_csv)
    if limit:
        pdb_ids = pdb_ids[:limit]
    print(f"[pubmed] primary-citation lookup for {len(pdb_ids)} PDBs")

    # Step 1: PDB → PMID via RCSB
    pmid_for_pdb: dict[str, str | None] = {}
    for i in range(0, len(pdb_ids), RCSB_BATCH):
        batch = pdb_ids[i:i + RCSB_BATCH]
        pmid_for_pdb.update(_rcsb_pmids(batch))
        ok = sum(1 for v in pmid_for_pdb.values() if v)
        print(f"  RCSB {i + len(batch)}/{len(pdb_ids)} … {ok} citations")
        time.sleep(0.1)

    pdbs_per_pmid: dict[str, set[str]] = defaultdict(set)
    for pid, pmid in pmid_for_pdb.items():
        if pmid:
            pdbs_per_pmid[pmid].add(pid)

    unique_pmids = sorted(pdbs_per_pmid.keys())
    print(f"  → {len(unique_pmids)} unique primary citations")

    # Step 2: efetch abstracts for those PMIDs
    abstracts: dict[str, dict] = {}
    for i in range(0, len(unique_pmids), EFETCH_BATCH):
        batch = unique_pmids[i:i + EFETCH_BATCH]
        abstracts.update(_efetch_abstracts(batch))
        time.sleep(RATE_LIMIT_S)
        print(f"  efetch {min(i + EFETCH_BATCH, len(unique_pmids))}/{len(unique_pmids)} "
              f"… {sum(1 for v in abstracts.values() if v.get('abstract'))} with text")

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out_jsonl.open("w") as f:
        for pmid, payload in abstracts.items():
            if not payload.get("abstract"):
                continue
            pdbs = sorted(pdbs_per_pmid.get(pmid, set()))
            text = f"{payload['title']}\n\n{payload['abstract']}"
            f.write(json.dumps({
                "kind": "pubmed_primary_citation",
                "pmid": pmid,
                "title": payload["title"],
                "pdb_ids": pdbs,
                "text": text,
            }) + "\n")
            written += 1
    print(f"[pubmed] wrote {written} primary citations → {out_jsonl}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--affinity-csv", type=Path,
                   default=Path("../misato-affinity/data/affinity_data.csv"))
    p.add_argument("--out", type=Path, default=Path("data/rag_sources/pubmed.jsonl"))
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()
    if not args.affinity_csv.exists():
        raise SystemExit(f"affinity csv not found: {args.affinity_csv}")
    fetch_all(args.affinity_csv, args.out, args.limit)


if __name__ == "__main__":
    main()

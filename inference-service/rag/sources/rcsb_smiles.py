"""Fetch ligand SMILES from the RCSB PDB Chemical Component Dictionary.

For each unique ligand 3-letter code in misato-affinity/data/affinity_data.csv,
query RCSB's GraphQL endpoint and extract the canonical SMILES (with stereo
when available). Writes `data/ligand_smiles.csv` with columns:

    pdb_id,ligand_code,smiles

The `ligand_descriptors` tool reads this file to compute MW / LogP / LE
via RDKit. No API key needed; RCSB is free + unauthenticated.

Usage:
    python -m rag.sources.rcsb_smiles                  # all PDBs in affinity_data.csv
    python -m rag.sources.rcsb_smiles --limit 50       # sample first 50
    python -m rag.sources.rcsb_smiles --out data/ligand_smiles.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


RCSB_GRAPHQL = "https://data.rcsb.org/graphql"
BATCH_SIZE = 50      # GraphQL allows multiple comp_ids per query
RATE_LIMIT_S = 0.1   # ~10 req/s; RCSB is forgiving but be polite


def _load_pdb_to_ligand(affinity_csv: Path) -> dict[str, list[str]]:
    """Return pdb_id → [ligand_code, ...]. Splits 'A&B' multi-ligand on '&'."""
    out: dict[str, list[str]] = {}
    with affinity_csv.open() as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            pid = (row.get("PDBid") or "").strip()
            lig = (row.get("ligand") or "").strip()
            if not pid or not lig:
                continue
            out[pid] = [c.strip() for c in lig.split("&") if c.strip()]
    return out


def _graphql_smiles(comp_ids: list[str]) -> dict[str, str | None]:
    """One GraphQL request returns SMILES for many comp_ids."""
    # RCSB GraphQL fields are case-sensitive: SMILES + SMILES_stereo (UPPER).
    aliases = ", ".join(
        f'c{i}: chem_comp(comp_id: "{cid}") '
        f'{{ chem_comp {{ id name }} '
        f'rcsb_chem_comp_descriptor {{ SMILES SMILES_stereo }} }}'
        for i, cid in enumerate(comp_ids)
    )
    query = f"{{ {aliases} }}"
    body = json.dumps({"query": query}).encode()

    req = urllib.request.Request(
        RCSB_GRAPHQL, data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.loads(r.read())
    except urllib.error.HTTPError as e:
        print(f"  RCSB HTTP {e.code}: {e.reason}")
        return {cid: None for cid in comp_ids}

    out: dict[str, str | None] = {}
    payload = data.get("data") or {}
    for i, cid in enumerate(comp_ids):
        entry = payload.get(f"c{i}")
        if not entry:
            out[cid] = None
            continue
        d = entry.get("rcsb_chem_comp_descriptor") or {}
        # Prefer stereo SMILES when available, fall back to canonical.
        out[cid] = d.get("SMILES_stereo") or d.get("SMILES")
    return out


def fetch_all(affinity_csv: Path, out_csv: Path, limit: int | None = None) -> None:
    pdb_to_ligs = _load_pdb_to_ligand(affinity_csv)
    if limit:
        pdb_to_ligs = dict(list(pdb_to_ligs.items())[:limit])

    unique_codes = sorted({c for codes in pdb_to_ligs.values() for c in codes})
    print(f"[rcsb] {len(pdb_to_ligs)} PDBs, {len(unique_codes)} unique ligand codes")

    code_to_smiles: dict[str, str | None] = {}
    for i in range(0, len(unique_codes), BATCH_SIZE):
        batch = unique_codes[i:i + BATCH_SIZE]
        batch_map = _graphql_smiles(batch)
        code_to_smiles.update(batch_map)
        ok = sum(1 for v in batch_map.values() if v)
        print(f"  batch {i // BATCH_SIZE + 1}/{(len(unique_codes) + BATCH_SIZE - 1) // BATCH_SIZE}: {ok}/{len(batch)} smiles")
        time.sleep(RATE_LIMIT_S)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pdb_id", "ligand_code", "smiles"])
        for pid, codes in pdb_to_ligs.items():
            for c in codes:
                smi = code_to_smiles.get(c)
                if smi:
                    w.writerow([pid, c, smi])
                    written += 1
                else:
                    skipped += 1
    print(f"[rcsb] wrote {written} rows, skipped {skipped} (no SMILES) → {out_csv}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--affinity-csv", type=Path,
                   default=Path("../misato-affinity/data/affinity_data.csv"))
    p.add_argument("--out", type=Path, default=Path("data/ligand_smiles.csv"))
    p.add_argument("--limit", type=int, default=None,
                   help="Only process first N PDBs (for quick test)")
    args = p.parse_args()
    if not args.affinity_csv.exists():
        raise SystemExit(f"affinity csv not found: {args.affinity_csv}")
    fetch_all(args.affinity_csv, args.out, args.limit)


if __name__ == "__main__":
    main()

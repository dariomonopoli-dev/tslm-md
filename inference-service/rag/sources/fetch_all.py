"""Run all three RAG-source fetchers in sequence.

Usage:
    python -m rag.sources.fetch_all                 # full pull (~15 min, all 1612 PDBs)
    python -m rag.sources.fetch_all --limit 50      # sample (~3 min)
"""

from __future__ import annotations

import argparse
from pathlib import Path

from . import rcsb_smiles, uniprot, pubmed


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--affinity-csv", type=Path,
                   default=Path("../misato-affinity/data/affinity_data.csv"))
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--limit", type=int, default=None,
                   help="Cap on PDBs across all fetchers (None = all)")
    args = p.parse_args()

    print("== 1. RCSB SMILES ==")
    rcsb_smiles.fetch_all(
        args.affinity_csv, args.data_dir / "ligand_smiles.csv", args.limit,
    )

    print("\n== 2. UniProt descriptions ==")
    uniprot.fetch_all(
        args.affinity_csv, args.data_dir / "rag_sources" / "uniprot.jsonl", args.limit,
    )

    print("\n== 3. PubMed primary citations ==")
    pubmed.fetch_all(
        args.affinity_csv, args.data_dir / "rag_sources" / "pubmed.jsonl", args.limit,
    )

    print("\nNext: pip install rdkit-pypi  &&  python -m rag.ingest")


if __name__ == "__main__":
    main()

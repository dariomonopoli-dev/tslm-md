"""RAG corpus ingest pipeline — task #11.

Reads source documents, chunks them, runs the leak-tagging regex from
rag.labels, embeds with OpenAI text-embedding-3-small, writes to ChromaDB.

Run as a one-shot: `python -m rag.ingest`. Idempotent — chunk IDs are
sha256(source + chunk_index), and the embedding cache lives at
$EMBEDDING_CACHE_DIR (default data/embedding_cache/).

Sources (minimum viable):
  - misato-affinity/data/affinity_data.csv  → 1 chunk per PDB with Kd/Ki/IC50
    context. THESE ARE THE LEAK SOURCE — tag_chunk will set
    contains_label=True with explicit_pdb_ids=[row.PDBid].
  - Project markdown docs (PROJECT_BRIEF, FRONTEND_v2, TRAINING, DATASET)
    → ~512-token chunks with 50-token overlap. No labels.

Add UniProt / PubMed when network access is available — same pattern.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

from .labels import tag_chunk, LabelTag


CHUNK_SIZE = 800        # chars (rough proxy for ~200 tokens — we don't tokenize here)
CHUNK_OVERLAP = 100
BATCH_SIZE = 64
EMBED_DIM = 1536        # text-embedding-3-small


@dataclass
class Chunk:
    chunk_id: str
    text: str
    source: str
    contains_label: bool
    pdb_ids: list[str]
    metadata: dict


def _chunk_id(source: str, index: int) -> str:
    h = hashlib.sha256(f"{source}|{index}".encode()).hexdigest()[:16]
    return f"{source.split('/')[-1]}_{index}_{h}"


# --------------------------------------------------------------------------
# Source readers
# --------------------------------------------------------------------------


def _read_affinity_csv(path: Path) -> Iterable[Chunk]:
    """One chunk per PDB row. These are the canonical leak source."""
    if not path.exists():
        print(f"[ingest] skipping (missing): {path}")
        return
    with path.open() as f:
        reader = csv.DictReader(f, delimiter=";")
        for i, row in enumerate(reader):
            pid = (row.get("PDBid") or "").strip()
            if not pid:
                continue
            uniprot = (row.get("Uniprot") or "").strip()
            protein = (row.get("Protein") or "").strip()
            ligand = (row.get("ligand") or "").strip()
            kd = (row.get("Kd (nM)") or "").strip()
            ki = (row.get("Ki (nM)") or "").strip()
            ic50 = (row.get("IC50 (nM)") or "").strip()
            kind = (row.get("type") or "").strip()
            parts = [f"PDB {pid}: target {protein} ({uniprot}); ligand {ligand}; classification {kind}."]
            if kd and kd not in ("nan", "NA"):
                parts.append(f"Kd = {kd} nM.")
            if ki and ki not in ("nan", "NA"):
                parts.append(f"Ki = {ki} nM.")
            if ic50 and ic50 not in ("nan", "NA"):
                parts.append(f"IC50 = {ic50} nM.")
            text = " ".join(parts)
            tag = tag_chunk(text, explicit_pdb_ids=[pid])
            yield Chunk(
                chunk_id=_chunk_id(str(path), i),
                text=text,
                source=str(path),
                contains_label=tag.contains_label,
                pdb_ids=tag.pdb_ids,
                metadata={
                    "kind": "affinity_row",
                    "primary_pdb": pid,
                    "uniprot": uniprot,
                },
            )


def _read_markdown(path: Path) -> Iterable[Chunk]:
    if not path.exists():
        return
    text = path.read_text()
    i = 0
    chunk_i = 0
    while i < len(text):
        piece = text[i:i + CHUNK_SIZE]
        tag = tag_chunk(piece)
        yield Chunk(
            chunk_id=_chunk_id(str(path), chunk_i),
            text=piece,
            source=str(path),
            contains_label=tag.contains_label,
            pdb_ids=tag.pdb_ids,
            metadata={"kind": "markdown", "title": path.stem},
        )
        i += CHUNK_SIZE - CHUNK_OVERLAP
        chunk_i += 1


def discover_sources() -> list[tuple[str, Path]]:
    """Returns (source-name, path) tuples for each input we know how to read."""
    root = Path(os.getenv("INGEST_ROOT", "."))
    sources = [
        ("affinity_csv", root / "misato-affinity" / "data" / "affinity_data.csv"),
        ("md_brief", root / "PROJECT_BRIEF.md"),
        ("md_frontend_v2", root / "FRONTEND_v2.md"),
        ("md_training", root / "TRAINING.md"),
        ("md_dataset", root / "DATASET.md"),
        ("md_readme", root / "README.md"),
    ]
    return [(name, p) for name, p in sources if p.exists()]


# --------------------------------------------------------------------------
# Embeddings (OpenAI text-embedding-3-small)
# --------------------------------------------------------------------------


def _embed_cache_path(text: str) -> Path:
    h = hashlib.sha256(text.encode()).hexdigest()
    cache_dir = Path(os.getenv("EMBEDDING_CACHE_DIR", "data/embedding_cache"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{h}.json"


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts; per-text disk cache; one OpenAI call per uncached batch."""
    cached: dict[int, list[float]] = {}
    uncached_idx: list[int] = []
    uncached_texts: list[str] = []
    for i, t in enumerate(texts):
        p = _embed_cache_path(t)
        if p.exists():
            cached[i] = json.loads(p.read_text())
        else:
            uncached_idx.append(i)
            uncached_texts.append(t)

    if uncached_texts:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        resp = client.embeddings.create(model=model, input=uncached_texts)
        for j, item in enumerate(resp.data):
            i = uncached_idx[j]
            vec = list(item.embedding)
            cached[i] = vec
            _embed_cache_path(uncached_texts[j]).write_text(json.dumps(vec))

    return [cached[i] for i in range(len(texts))]


# --------------------------------------------------------------------------
# Chroma upsert
# --------------------------------------------------------------------------


def _chroma_collection():
    import chromadb
    client = chromadb.PersistentClient(path=os.getenv("CHROMA_PATH", "/app/data/chroma"))
    return client.get_or_create_collection(
        name="molemotion",
        metadata={"hnsw:space": "cosine"},
    )


def upsert(chunks: list[Chunk], embeddings: list[list[float]]) -> None:
    col = _chroma_collection()
    metadatas = []
    for c in chunks:
        meta = dict(c.metadata)
        meta["contains_label"] = bool(c.contains_label)
        meta["pdb_ids_csv"] = ",".join(c.pdb_ids)   # Chroma metadata can't be lists
        meta["source"] = c.source
        metadatas.append(meta)
    col.upsert(
        ids=[c.chunk_id for c in chunks],
        documents=[c.text for c in chunks],
        embeddings=embeddings,
        metadatas=metadatas,
    )


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY not set")

    sources = discover_sources()
    if not sources:
        raise SystemExit("no ingest sources found (set INGEST_ROOT)")

    print(f"[ingest] sources: {[s[0] for s in sources]}")

    all_chunks: list[Chunk] = []
    for name, path in sources:
        if name == "affinity_csv":
            all_chunks.extend(_read_affinity_csv(path))
        else:
            all_chunks.extend(_read_markdown(path))

    print(f"[ingest] {len(all_chunks)} chunks total; "
          f"{sum(1 for c in all_chunks if c.contains_label)} contain labels")

    # Batched embedding + upsert
    total = len(all_chunks)
    for i in range(0, total, BATCH_SIZE):
        batch = all_chunks[i:i + BATCH_SIZE]
        t0 = time.monotonic()
        embeddings = embed_batch([c.text for c in batch])
        upsert(batch, embeddings)
        dt = time.monotonic() - t0
        print(f"[ingest] {i + len(batch)}/{total} in {dt:.1f}s")
        time.sleep(0.1)  # gentle on the rate limit

    # Stamp version
    version = os.getenv("RAG_CORPUS_VERSION", f"v1-{time.strftime('%Y-%m-%d')}")
    Path(os.getenv("CHROMA_PATH", "/app/data/chroma") + "/version.json").write_text(
        json.dumps({"rag_corpus_version": version, "n_chunks": total}, indent=2)
    )
    print(f"[ingest] done. rag_corpus_version={version}")


if __name__ == "__main__":
    main()

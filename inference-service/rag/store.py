"""RAG retrieval with label filtering — task #12.

Exposes rag_query() consumed by the orchestrator's pre-flight step AND
registered as the `rag_query` agent tool (visible in agent trace).
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import TypedDict

from .labels import LabelTag, is_leak_for
from tools import register


class Chunk(TypedDict):
    chunk_id: str
    text: str
    score: float
    contains_label: bool
    pdb_ids: list[str]
    source: str


@lru_cache(maxsize=1)
def _collection():
    import chromadb
    client = chromadb.PersistentClient(path=os.getenv("CHROMA_PATH", "/app/data/chroma"))
    return client.get_or_create_collection(name="trajecta", metadata={"hnsw:space": "cosine"})


def _embed(text: str) -> list[float]:
    """Embed query at lookup time. Cached on disk via ingest's helper."""
    from .ingest import embed_batch
    return embed_batch([text])[0]


def _parse_pdb_ids(meta: dict) -> list[str]:
    raw = meta.get("pdb_ids_csv", "") or ""
    return [p for p in raw.split(",") if p]


@register({
    "name": "rag_query",
    "description": (
        "Retrieve up to top_k chunks from the label-filtered RAG corpus. "
        "Chunks containing the binding label for the given pdb_id are "
        "filtered out at retrieval time — you cannot read the answer."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "pdb_id": {"type": "string"},
            "top_k": {"type": "integer", "default": 6, "minimum": 1, "maximum": 12},
        },
        "required": ["query", "pdb_id"],
    },
})
def rag_query(query: str, pdb_id: str, top_k: int = 6) -> dict:
    """Vector search → label filter → rerank → top_k.

    Returns a dict (not bare list) so the agent trace renders nicely:
    `{"chunks": [...], "filtered_n": N}`. Degrades gracefully to
    `{chunks: [], error|note: "..."}` on missing key / empty corpus —
    the orchestrator handles the empty case without crashing.
    """
    import os
    if not (os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")):
        return {
            "query": query, "pdb_id": pdb_id,
            "filtered_n": 0, "returned_n": 0, "chunks": [],
            "error": "no embeddings API key set — RAG disabled",
        }

    try:
        col = _collection()
        if col.count() == 0:
            return {
                "query": query, "pdb_id": pdb_id,
                "filtered_n": 0, "returned_n": 0, "chunks": [],
                "note": "RAG corpus not ingested (`make ingest` to populate)",
            }
        oversample = top_k * 3
        qvec = _embed(query)
        raw = col.query(
            query_embeddings=[qvec],
            n_results=oversample,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as e:
        return {
            "query": query, "pdb_id": pdb_id,
            "filtered_n": 0, "returned_n": 0, "chunks": [],
            "error": f"{type(e).__name__}: {e}",
        }

    docs = raw["documents"][0] if raw["documents"] else []
    metas = raw["metadatas"][0] if raw["metadatas"] else []
    dists = raw["distances"][0] if raw["distances"] else []
    ids = raw["ids"][0] if raw["ids"] else []

    candidates: list[Chunk] = []
    filtered = 0
    for cid, doc, meta, dist in zip(ids, docs, metas, dists):
        pdb_ids = _parse_pdb_ids(meta)
        tag = LabelTag(
            contains_label=bool(meta.get("contains_label", False)),
            pdb_ids=pdb_ids,
        )
        if is_leak_for(tag, pdb_id):
            filtered += 1
            continue
        # cosine distance → similarity score
        score = float(1.0 - dist)
        # rerank boost when this chunk explicitly mentions our pdb_id
        if pdb_id.upper() in pdb_ids:
            score += 0.3
        candidates.append(Chunk(
            chunk_id=cid,
            text=doc,
            score=score,
            contains_label=tag.contains_label,
            pdb_ids=pdb_ids,
            source=str(meta.get("source", "")),
        ))

    candidates.sort(key=lambda c: c["score"], reverse=True)
    return {
        "query": query,
        "pdb_id": pdb_id,
        "filtered_n": filtered,
        "returned_n": min(top_k, len(candidates)),
        "chunks": candidates[:top_k],
    }

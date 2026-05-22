"""Knowledge upload pipeline — feeds user-uploaded content into ChromaDB.

Uses docling to convert any supported file (PDF, DOCX, PPTX, XLSX, HTML,
images via OCR) into structured Markdown. Then runs the existing chunker,
label-tagger, OpenRouter embeddings, and Chroma upsert.

The label-filter regex still runs on uploaded text — anything mentioning
a Kd/Ki/IC50 for a specific PDB is auto-tagged contains_label so the
/evaluate/agent leak filter excludes it when that PDB is the query.
"""

from __future__ import annotations

import hashlib
import tempfile
import time
from pathlib import Path
from typing import Iterable

from .ingest import (
    BATCH_SIZE, CHUNK_OVERLAP, CHUNK_SIZE, Chunk,
    _chunk_id, embed_batch, upsert,
)
from .labels import tag_chunk


_DOCLING_OK_EXTS = {
    ".pdf", ".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls",
    ".html", ".htm", ".md", ".markdown", ".txt", ".text",
    ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp",
}


# --------------------------------------------------------------------------
# Extractor — docling everywhere, falls back to plain decode for trivial cases.
# --------------------------------------------------------------------------


def extract(filename: str, data: bytes) -> tuple[str, str]:
    """Return (kind, markdown). Heavy lifting via docling."""
    suffix = Path(filename).suffix.lower()
    if suffix not in _DOCLING_OK_EXTS:
        raise ValueError(
            f"unsupported file type: {filename} (extensions: {sorted(_DOCLING_OK_EXTS)})"
        )

    # Cheap fast path for already-text files — skip docling startup cost.
    if suffix in {".md", ".markdown", ".txt", ".text"}:
        return "markdown" if suffix.startswith(".m") else "text", data.decode("utf-8", errors="replace")

    from docling.document_converter import DocumentConverter

    # docling needs a path on disk for most converters; write to temp.
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tf:
        tf.write(data)
        tmp_path = Path(tf.name)
    try:
        converter = DocumentConverter()
        result = converter.convert(str(tmp_path))
        markdown = result.document.export_to_markdown()
    finally:
        tmp_path.unlink(missing_ok=True)

    kind = suffix.lstrip(".")
    return kind, markdown


# --------------------------------------------------------------------------
# Chunk + embed + upsert
# --------------------------------------------------------------------------


def _chunks_from_text(
    text: str, source_id: str, kind: str, title: str,
    explicit_pdb_ids: list[str],
) -> Iterable[Chunk]:
    """Sliding-window chunker matching the existing rag.ingest pattern."""
    if not text.strip():
        return
    uploaded_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    i = 0
    chunk_i = 0
    while i < len(text):
        piece = text[i:i + CHUNK_SIZE]
        if piece.strip():
            tag = tag_chunk(piece, explicit_pdb_ids=explicit_pdb_ids)
            yield Chunk(
                chunk_id=_chunk_id(source_id, chunk_i),
                text=piece,
                source=source_id,
                contains_label=tag.contains_label,
                pdb_ids=tag.pdb_ids,
                metadata={
                    "kind": f"upload/{kind}",
                    "title": title,
                    "source_id": source_id,
                    "uploaded_at": uploaded_at,
                },
            )
        i += CHUNK_SIZE - CHUNK_OVERLAP
        chunk_i += 1


def _source_id(filename: str, content_bytes: bytes) -> str:
    """sha256(filename + content) — stable across re-uploads of the same file."""
    h = hashlib.sha256()
    h.update(filename.encode())
    h.update(b"|")
    h.update(content_bytes)
    return f"upload_{h.hexdigest()[:16]}"


def ingest_upload(
    filename: str, content: bytes,
    title: str, pdb_ids: list[str],
) -> dict:
    """End-to-end: extract → chunk → tag → embed → upsert. Returns summary."""
    kind, text = extract(filename, content)
    source_id = _source_id(filename, content)
    chunks = list(_chunks_from_text(text, source_id, kind, title, pdb_ids))
    if not chunks:
        return {
            "source_id": source_id, "title": title, "kind": kind,
            "chunks_added": 0, "warning": "no extractable text",
        }

    for j in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[j:j + BATCH_SIZE]
        embeddings = embed_batch([c["text"] for c in batch])
        upsert(batch, embeddings)

    return {
        "source_id": source_id,
        "title": title,
        "kind": kind,
        "filename": filename,
        "chunks_added": len(chunks),
        "n_label_chunks": sum(1 for c in chunks if c["contains_label"]),
        "pdb_ids_tagged": sorted({p for c in chunks for p in c["pdb_ids"]}),
        "text_length_chars": len(text),
    }


# --------------------------------------------------------------------------
# List / delete uploaded sources
# --------------------------------------------------------------------------


def list_uploaded_sources() -> list[dict]:
    """Group Chroma entries by source_id, return summary per source."""
    from .store import _collection
    try:
        col = _collection()
    except Exception:
        return []
    try:
        raw = col.get(where={"kind": {"$contains": "upload/"}}, include=["metadatas"])
    except Exception:
        raw = col.get(include=["metadatas"])
        raw["metadatas"] = [
            m for m in (raw.get("metadatas") or [])
            if str(m.get("kind", "")).startswith("upload/")
        ]

    grouped: dict[str, dict] = {}
    for meta in raw.get("metadatas", []) or []:
        sid = meta.get("source_id") or "?"
        if sid not in grouped:
            grouped[sid] = {
                "source_id": sid,
                "title": meta.get("title", "(no title)"),
                "kind": meta.get("kind", "upload/?"),
                "uploaded_at": meta.get("uploaded_at", ""),
                "n_chunks": 0,
            }
        grouped[sid]["n_chunks"] += 1
    return sorted(grouped.values(), key=lambda s: s.get("uploaded_at", ""), reverse=True)


def delete_uploaded_source(source_id: str) -> int:
    """Remove all chunks for a given source_id. Returns the count removed."""
    from .store import _collection
    col = _collection()
    raw = col.get(where={"source_id": source_id}, include=[])
    ids = raw.get("ids", []) or []
    if ids:
        col.delete(ids=ids)
    return len(ids)

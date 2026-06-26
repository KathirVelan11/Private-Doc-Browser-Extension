"""Ingestion pipeline: file -> units -> chunks -> embeddings -> store.

This is the glue that turns an uploaded document into searchable chunks in
the hybrid retriever.
"""
import hashlib
import os
from typing import List

from . import config_bridge as cfg
from .chunking import chunk_text
from .loaders import load_document


def _chunk_id(source: str, page: int, index: int, text: str) -> str:
    h = hashlib.sha1(f"{source}|{page}|{index}|{text}".encode("utf-8")).hexdigest()
    return h[:16]


def ingest_file(path: str, retriever) -> dict:
    """Load, chunk, embed and index a single file. Returns a small summary."""
    source = os.path.basename(path)
    units = load_document(path)

    documents: List[str] = []
    metadatas: List[dict] = []
    ids: List[str] = []

    for text, page in units:
        chunks = chunk_text(text, cfg.CHUNK_SIZE, cfg.CHUNK_OVERLAP)
        for i, chunk in enumerate(chunks):
            cid = _chunk_id(source, page, i, chunk)
            documents.append(chunk)
            metadatas.append({"source": source, "page": page})
            ids.append(cid)

    if not documents:
        return {"source": source, "chunks": 0, "pages": len(units)}

    embeddings = retriever.embedder.encode(documents)
    retriever.add_chunks(ids, embeddings, documents, metadatas)

    return {"source": source, "chunks": len(documents), "pages": len(units)}

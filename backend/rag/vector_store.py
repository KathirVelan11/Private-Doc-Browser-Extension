"""Persistent vector store backed by ChromaDB (cosine space).

Chroma is the single source of truth for chunks: documents, metadata and
embeddings all live here and survive restarts. The BM25 sparse index is
rebuilt in memory from this store on startup.
"""
import os
from typing import Dict, List

# Chroma's telemetry is noisy and offline-unfriendly; disable before import.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import chromadb


class VectorStore:
    def __init__(self, persist_dir: str, collection_name: str):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, ids, embeddings, documents, metadatas):
        self.collection.add(
            ids=ids,
            embeddings=[e.tolist() for e in embeddings],
            documents=documents,
            metadatas=metadatas,
        )

    def query(self, embedding, n_results: int):
        res = self.collection.query(
            query_embeddings=[embedding.tolist()],
            n_results=n_results,
        )
        out = []
        if not res["ids"] or not res["ids"][0]:
            return out
        for i, cid in enumerate(res["ids"][0]):
            out.append(
                {
                    "id": cid,
                    "text": res["documents"][0][i],
                    "metadata": res["metadatas"][0][i],
                    # cosine distance -> similarity
                    "score": 1.0 - res["distances"][0][i],
                }
            )
        return out

    def get_all(self) -> Dict[str, List]:
        """Return every chunk (id, text, metadata) for rebuilding BM25."""
        return self.collection.get(include=["documents", "metadatas"])

    def count(self) -> int:
        return self.collection.count()

    def sources(self) -> Dict[str, int]:
        data = self.collection.get(include=["metadatas"])
        counts: Dict[str, int] = {}
        for md in data.get("metadatas", []) or []:
            src = md.get("source", "unknown")
            counts[src] = counts.get(src, 0) + 1
        return counts

    def reset(self):
        name = self.collection.name
        self.client.delete_collection(name)
        self.collection = self.client.get_or_create_collection(
            name=name, metadata={"hnsw:space": "cosine"}
        )

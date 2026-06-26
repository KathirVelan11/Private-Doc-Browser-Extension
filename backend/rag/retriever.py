"""Hybrid retriever: dense (vector) + sparse (BM25) fused with Reciprocal
Rank Fusion (RRF).

RRF is a simple, robust way to combine rankings from retrievers whose scores
are on different scales. Each result gets ``sum(1 / (RRF_K + rank))`` across
the lists it appears in. It consistently beats either retriever alone and
needs no score normalization or tuning -- ideal for a "no-LLM, just great
retrieval" system.
"""
import time
from typing import Dict, List

from . import config_bridge as cfg
from .bm25_index import BM25Index
from .embeddings import Embedder
from .vector_store import VectorStore


class HybridRetriever:
    def __init__(self, embedder: Embedder, store: VectorStore):
        self.embedder = embedder
        self.store = store
        self.bm25 = BM25Index()
        self._id_to_chunk: Dict[str, dict] = {}
        self.rebuild_sparse()

    # -- indexing ---------------------------------------------------------
    def rebuild_sparse(self):
        data = self.store.get_all()
        ids = data.get("ids", []) or []
        docs = data.get("documents", []) or []
        metas = data.get("metadatas", []) or []
        self.bm25.build(ids, docs)
        self._id_to_chunk = {
            cid: {"id": cid, "text": docs[i], "metadata": metas[i]}
            for i, cid in enumerate(ids)
        }

    def add_chunks(self, ids, embeddings, documents, metadatas):
        self.store.add(ids, embeddings, documents, metadatas)
        self.rebuild_sparse()

    # -- search -----------------------------------------------------------
    def search(self, query: str, top_k: int = None) -> dict:
        top_k = top_k or cfg.TOP_K
        t0 = time.perf_counter()

        q_emb = self.embedder.encode([query])[0]
        dense = self.store.query(q_emb, cfg.CANDIDATE_POOL)
        sparse = self.bm25.search(query, cfg.CANDIDATE_POOL)

        fused = self._rrf(dense, sparse)
        results = []
        for cid, info in fused[:top_k]:
            chunk = self._id_to_chunk.get(cid)
            if not chunk:
                continue
            md = chunk["metadata"]
            results.append(
                {
                    "text": chunk["text"],
                    "source": md.get("source", "unknown"),
                    "page": md.get("page", 1),
                    "rrf_score": round(info["rrf"], 5),
                    "dense_score": round(info["dense"], 4) if info["dense"] is not None else None,
                    "bm25_score": round(info["bm25"], 4) if info["bm25"] is not None else None,
                    "matched_by": info["matched_by"],
                }
            )
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        return {"query": query, "results": results, "elapsed_ms": elapsed_ms}

    def _rrf(self, dense: List[dict], sparse):
        k = cfg.RRF_K
        agg: Dict[str, dict] = {}

        for rank, item in enumerate(dense):
            cid = item["id"]
            entry = agg.setdefault(
                cid, {"rrf": 0.0, "dense": None, "bm25": None, "matched_by": []}
            )
            entry["rrf"] += 1.0 / (k + rank + 1)
            entry["dense"] = item["score"]
            entry["matched_by"].append("semantic")

        for rank, (cid, score) in enumerate(sparse):
            entry = agg.setdefault(
                cid, {"rrf": 0.0, "dense": None, "bm25": None, "matched_by": []}
            )
            entry["rrf"] += 1.0 / (k + rank + 1)
            entry["bm25"] = score
            entry["matched_by"].append("keyword")

        return sorted(agg.items(), key=lambda x: x[1]["rrf"], reverse=True)

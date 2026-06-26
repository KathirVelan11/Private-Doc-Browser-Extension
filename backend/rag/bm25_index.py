"""In-memory BM25 sparse index for keyword/lexical retrieval.

Dense embeddings are great at meaning but can miss exact technical tokens
(alloy formulas like "MoNbTaW", "BCC", specific temperatures). BM25 nails
those. Combining both is what makes the hybrid retriever strong.
"""
import re
from typing import List

from rank_bm25 import BM25Okapi

_TOKEN = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> List[str]:
    return _TOKEN.findall(text.lower())


class BM25Index:
    def __init__(self):
        self.ids: List[str] = []
        self._bm25 = None

    def build(self, ids: List[str], texts: List[str]):
        self.ids = list(ids)
        corpus = [tokenize(t) for t in texts]
        self._bm25 = BM25Okapi(corpus) if corpus else None

    def search(self, query: str, n: int):
        """Return [(id, score), ...] best-first, length <= n."""
        if self._bm25 is None or not self.ids:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(zip(self.ids, scores), key=lambda x: x[1], reverse=True)
        return [(cid, float(s)) for cid, s in ranked[:n] if s > 0]

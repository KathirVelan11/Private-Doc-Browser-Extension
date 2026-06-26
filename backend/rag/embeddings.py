"""Embedding model wrapper.

Uses ChromaDB's bundled ONNX build of ``all-MiniLM-L6-v2`` (runs on
``onnxruntime`` — no PyTorch, ~80 MB). The model is loaded lazily on first
use so the Flask app starts instantly and only pays the (one-time) model-load
cost when documents are first indexed or the first question is asked.
"""
from typing import List

import numpy as np


class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._ef = None

    @property
    def ef(self):
        if self._ef is None:
            from chromadb.utils import embedding_functions

            self._ef = embedding_functions.ONNXMiniLM_L6_V2()
        return self._ef

    def encode(self, texts: List[str]) -> np.ndarray:
        """Return L2-normalized embeddings (so dot product == cosine sim)."""
        vecs = np.asarray(self.ef(texts), dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms

"""Re-exports top-level ``config`` so modules inside the ``rag`` package can
import settings with a stable name (``from . import config_bridge as cfg``).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (  # noqa: E402,F401
    CANDIDATE_POOL,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    RRF_K,
    TOP_K,
)

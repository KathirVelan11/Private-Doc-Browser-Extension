"""Central configuration for the Private Doc backend.

Everything is overridable through environment variables. Defaults are tuned
for a laptop running Ollama locally — fully offline, nothing leaves the
machine.
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _env(name, default):
    return os.environ.get(name, default)


# --- Storage -------------------------------------------------------------
DATA_DIR = _env("PDB_DATA_DIR", os.path.join(BASE_DIR, "data"))
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
CHROMA_DIR = os.path.join(DATA_DIR, "chroma")
COLLECTION_NAME = _env("PDB_COLLECTION", "private_documents")

# --- Embedding model -----------------------------------------------------
# ChromaDB's bundled ONNX all-MiniLM-L6-v2 (~80 MB, no PyTorch, CPU-only).
EMBEDDING_MODEL = _env("PDB_EMBED_MODEL", "all-MiniLM-L6-v2")

# --- Chunking ------------------------------------------------------------
CHUNK_SIZE = int(_env("PDB_CHUNK_SIZE", "800"))        # characters
CHUNK_OVERLAP = int(_env("PDB_CHUNK_OVERLAP", "150"))  # characters

# --- Retrieval -----------------------------------------------------------
TOP_K = int(_env("PDB_TOP_K", "5"))
RRF_K = int(_env("PDB_RRF_K", "60"))
CANDIDATE_POOL = int(_env("PDB_CANDIDATE_POOL", "20"))

# --- Upload limits -------------------------------------------------------
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx", ".pptx", ".html", ".htm", ".csv"}
MAX_CONTENT_LENGTH = int(_env("PDB_MAX_MB", "64")) * 1024 * 1024

# --- Ollama (local LLM) --------------------------------------------------
OLLAMA_URL = _env("PDB_OLLAMA_URL", "http://127.0.0.1:11434")
DEFAULT_MODEL = _env("PDB_MODEL", "qwen2.5:7b")
# If the chosen model can't load (e.g. not enough free RAM), fall back to this
# small model so chat still works on constrained machines.
FALLBACK_MODEL = _env("PDB_FALLBACK_MODEL", "qwen2.5:0.5b")
# Max characters of retrieved context fed to the model.
MAX_CONTEXT_CHARS = int(_env("PDB_MAX_CONTEXT_CHARS", "6000"))

# --- Bookmark import -----------------------------------------------------
# Whether to fetch each bookmarked page and index its text (best-effort).
BOOKMARK_FETCH_TIMEOUT = int(_env("PDB_BOOKMARK_TIMEOUT", "8"))  # seconds


def ensure_dirs():
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(CHROMA_DIR, exist_ok=True)

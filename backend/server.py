"""Private Doc backend — Flask API for the browser extension.

A fully-offline personal assistant:

* /chat   - ask a question; answered by a LOCAL Ollama model, grounded in your
            uploaded documents + imported bookmarks via hybrid RAG retrieval.
            Streams tokens back as Server-Sent Events.
* /upload - add documents (pdf, docx, pptx, txt, md, html, csv) to the index.
* /import_bookmarks - index your browser bookmarks (title/url, optionally page
            text) so the assistant can "learn" from what you save.
* /query  - raw hybrid retrieval (no LLM), returns matching passages.
* /documents, /reset, /models, /health - housekeeping.

Nothing ever leaves the machine: retrieval is local (ChromaDB ONNX embeddings
+ BM25) and generation is local (Ollama).
"""
import json
import os
import time
import traceback
import urllib.error
import urllib.request

from flask import Flask, Response, jsonify, request
from werkzeug.utils import secure_filename

import config
from rag.chunking import chunk_text
from rag.embeddings import Embedder
from rag.ingest import ingest_file
from rag.loaders import is_supported
from rag.retriever import HybridRetriever
from rag.vector_store import VectorStore

config.ensure_dirs()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH

# Built once and shared across requests (single-process dev server).
_embedder = Embedder(config.EMBEDDING_MODEL)
_store = VectorStore(config.CHROMA_DIR, config.COLLECTION_NAME)
_retriever = HybridRetriever(_embedder, _store)


# --------------------------------------------------------------------------
# CORS — the extension's side panel runs from a chrome-extension:// origin and
# must be allowed to call this localhost server.
# --------------------------------------------------------------------------
@app.after_request
def add_cors(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return resp


@app.route("/<path:_any>", methods=["OPTIONS"])
@app.route("/", methods=["OPTIONS"])
def cors_preflight(_any=None):
    return ("", 204)


# --------------------------------------------------------------------------
# Ollama helpers
# --------------------------------------------------------------------------
def _ollama_models():
    """List locally available Ollama models."""
    try:
        req = urllib.request.Request(config.OLLAMA_URL + "/api/tags")
        with urllib.request.urlopen(req, timeout=5) as r:
            data = json.loads(r.read().decode("utf-8"))
        return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


def _build_context(question: str, top_k: int):
    """Retrieve grounding passages for a question. Returns (context, sources)."""
    if _store.count() == 0:
        return "", []
    result = _retriever.search(question, top_k=top_k)
    passages, sources, used = [], [], 0
    for i, r in enumerate(result["results"], start=1):
        snippet = r["text"].strip()
        block = f"[{i}] (source: {r['source']}, page {r['page']})\n{snippet}"
        if used + len(block) > config.MAX_CONTEXT_CHARS:
            break
        passages.append(block)
        sources.append(
            {
                "n": i,
                "source": r["source"],
                "page": r["page"],
                "matched_by": r["matched_by"],
                "snippet": snippet[:240],
            }
        )
        used += len(block)
    return "\n\n".join(passages), sources


SYSTEM_PROMPT = (
    "You are a private, fully-offline personal assistant running on the user's "
    "own laptop. Answer clearly and concisely. When CONTEXT passages are "
    "provided, prefer them and cite the sources you used inline like [1], [2]. "
    "If the context does not contain the answer, say so and answer from your "
    "own knowledge, making clear it is not from the user's documents."
)


def _ollama_chat_stream(model, messages):
    """Yield text tokens from Ollama's streaming chat API."""
    payload = json.dumps({"model": model, "messages": messages, "stream": True})
    req = urllib.request.Request(
        config.OLLAMA_URL + "/api/chat",
        data=payload.encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        for raw in resp:
            line = raw.decode("utf-8").strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("done"):
                break
            token = obj.get("message", {}).get("content", "")
            if token:
                yield token


def _sse(event, data):
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.route("/health")
def health():
    return jsonify(
        {
            "ok": True,
            "total_chunks": _store.count(),
            "ollama_models": _ollama_models(),
            "default_model": config.DEFAULT_MODEL,
        }
    )


@app.route("/models")
def models():
    return jsonify({"models": _ollama_models(), "default": config.DEFAULT_MODEL})


@app.route("/documents")
def documents():
    return jsonify(
        {
            "documents": _store.sources(),
            "total_chunks": _store.count(),
            "embedding_model": config.EMBEDDING_MODEL,
        }
    )


@app.route("/upload", methods=["POST"])
def upload():
    files = request.files.getlist("files")
    if not files or all(f.filename == "" for f in files):
        return jsonify({"error": "No files provided."}), 400

    summaries, skipped = [], []
    t0 = time.perf_counter()
    for f in files:
        if not f.filename:
            continue
        if not is_supported(f.filename):
            skipped.append(f.filename)
            continue
        filename = secure_filename(f.filename)
        dest = os.path.join(config.UPLOAD_DIR, filename)
        f.save(dest)
        try:
            summaries.append(ingest_file(dest, _retriever))
        except Exception as exc:
            traceback.print_exc()
            skipped.append(f"{f.filename} ({exc})")

    return jsonify(
        {
            "indexed": summaries,
            "skipped": skipped,
            "total_chunks": _store.count(),
            "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
        }
    )


@app.route("/query", methods=["POST"])
def query():
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "Empty question."}), 400
    if _store.count() == 0:
        return jsonify({"error": "No documents indexed yet."}), 400
    top_k = int(data.get("top_k") or config.TOP_K)
    return jsonify(_retriever.search(question, top_k=top_k))


@app.route("/chat", methods=["POST"])
def chat():
    """Streamed, RAG-grounded chat with a local Ollama model (SSE)."""
    data = request.get_json(silent=True) or {}
    history = data.get("messages") or []
    if not history or history[-1].get("role") != "user":
        return jsonify({"error": "Last message must be from the user."}), 400

    model = data.get("model") or config.DEFAULT_MODEL
    use_rag = data.get("use_rag", True)
    top_k = int(data.get("top_k") or config.TOP_K)
    question = (history[-1].get("content") or "").strip()

    context, sources = ("", [])
    if use_rag:
        context, sources = _build_context(question, top_k)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    # Keep the recent conversation; inject context just before the last turn.
    messages.extend(history[:-1])
    if context:
        user_block = (
            f"CONTEXT (from the user's private documents & bookmarks):\n{context}"
            f"\n\n---\nQUESTION: {question}"
        )
    else:
        user_block = question
    messages.append({"role": "user", "content": user_block})

    def _looks_like_oom(exc):
        msg = str(exc).lower()
        return any(
            s in msg
            for s in ("allocate", "terminated", "out of memory", "buffer", "500")
        )

    def generate():
        yield _sse("sources", {"sources": sources, "model": model})
        active = model
        try:
            try:
                for token in _ollama_chat_stream(active, messages):
                    yield _sse("token", {"t": token})
            except urllib.error.HTTPError as exc:
                # Most commonly the model can't allocate enough RAM. Retry once
                # with the small fallback model so the user still gets an answer.
                if _looks_like_oom(exc) and active != config.FALLBACK_MODEL:
                    active = config.FALLBACK_MODEL
                    yield _sse(
                        "notice",
                        {
                            "message": f"{model} could not load (low memory); "
                            f"using {active} instead."
                        },
                    )
                    for token in _ollama_chat_stream(active, messages):
                        yield _sse("token", {"t": token})
                else:
                    raise
        except urllib.error.URLError as exc:
            yield _sse(
                "error",
                {"message": f"Could not reach Ollama at {config.OLLAMA_URL}: {exc}"},
            )
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            yield _sse("error", {"message": str(exc)})
        yield _sse("done", {"ok": True})

    return Response(generate(), mimetype="text/event-stream")


@app.route("/import_bookmarks", methods=["POST"])
def import_bookmarks():
    """Index browser bookmarks. Each item: {title, url}.

    If ``fetch`` is true, best-effort download each page and index its text;
    otherwise just index 'title — url' so the assistant knows what you saved.
    """
    data = request.get_json(silent=True) or {}
    items = data.get("bookmarks") or []
    fetch = bool(data.get("fetch", False))
    if not items:
        return jsonify({"error": "No bookmarks provided."}), 400

    indexed, fetched, failed = 0, 0, 0
    t0 = time.perf_counter()
    for bm in items:
        title = (bm.get("title") or "").strip()
        url = (bm.get("url") or "").strip()
        if not url:
            continue
        page_text = ""
        if fetch and url.startswith(("http://", "https://")):
            page_text = _fetch_page_text(url)
            if page_text:
                fetched += 1
            else:
                failed += 1
        body = f"{title}\n{url}"
        if page_text:
            body += "\n\n" + page_text
        try:
            _index_text(body, source=title or url, origin="bookmark", url=url)
            indexed += 1
        except Exception:
            traceback.print_exc()
            failed += 1

    _retriever.rebuild_sparse()
    return jsonify(
        {
            "indexed": indexed,
            "fetched_pages": fetched,
            "failed": failed,
            "total_chunks": _store.count(),
            "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
        }
    )


@app.route("/reset", methods=["POST"])
def reset():
    _store.reset()
    _retriever.rebuild_sparse()
    return jsonify({"ok": True, "total_chunks": _store.count()})


# --------------------------------------------------------------------------
# Indexing helpers for non-file text (bookmarks, raw selections)
# --------------------------------------------------------------------------
import hashlib  # noqa: E402


def _index_text(text, source, origin="text", url=""):
    """Chunk, embed and add raw text to the store under a logical source."""
    chunks = chunk_text(text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    if not chunks:
        return 0
    ids, docs, metas = [], [], []
    for i, ch in enumerate(chunks):
        cid = hashlib.sha1(f"{source}|{i}|{ch}".encode("utf-8")).hexdigest()[:16]
        ids.append(cid)
        docs.append(ch)
        metas.append({"source": source, "page": 1, "origin": origin, "url": url})
    embeddings = _retriever.embedder.encode(docs)
    # add directly to the store; caller rebuilds sparse index once at the end.
    _store.add(ids, embeddings, docs, metas)
    return len(chunks)


def _fetch_page_text(url):
    """Best-effort fetch + strip HTML to readable text. Returns '' on failure."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=config.BOOKMARK_FETCH_TIMEOUT) as r:
            ctype = r.headers.get("Content-Type", "")
            if "html" not in ctype and "text" not in ctype:
                return ""
            raw = r.read(2_000_000).decode("utf-8", errors="ignore")
    except Exception:
        return ""
    from rag.loaders import _TextExtractor

    parser = _TextExtractor()
    try:
        parser.feed(raw)
    except Exception:
        return ""
    text = "\n".join(parser.chunks)
    return text[:50_000]


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    print(f" * Private Doc backend on http://127.0.0.1:{port}")
    print(f" * Ollama: {config.OLLAMA_URL}  models: {_ollama_models()}")
    app.run(host="127.0.0.1", port=port, threaded=True, debug=False)

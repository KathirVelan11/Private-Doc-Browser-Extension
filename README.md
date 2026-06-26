# Private Doc — Local AI Browser Assistant

A **fully-offline, fully-private** browser assistant. Select text on any web
page, right-click, and ask a **local Ollama model** about it. Answers are
grounded in **your own documents and bookmarks** using hybrid RAG retrieval.
Nothing ever leaves your machine — no API keys, no cloud.

> Select text → right-click → *“Ask Private AI”* → a side panel opens and a
> local model answers, citing your documents.

## Features

- 🔒 **100% local & private** — retrieval (ChromaDB ONNX embeddings + BM25) and
  generation (Ollama) both run on your laptop.
- 🖱️ **Right-click on a selection** → side panel chat about that text.
- 💬 **Streaming chat** with any local Ollama model (picker in the header).
- 📄 **Upload documents** (PDF, DOCX, PPTX, TXT, MD, HTML, CSV) — they become
  searchable knowledge.
- 🔖 **Import your browser bookmarks** so the assistant learns from what you
  save (optionally fetching and reading each page).
- 🧠 **Hybrid RAG** — dense (semantic) + sparse (BM25 keyword) retrieval fused
  with Reciprocal Rank Fusion. Reuses the engine from the
  *High-Refractory-Entropy-Alloy-RAG* project.
- ♻️ **Auto-fallback** — if your chosen model can't fit in RAM, the backend
  retries with a small model so chat never hard-fails.

## Architecture

```
Browser extension (MV3)                 Local backend (Flask, 127.0.0.1:5000)
┌───────────────────────────┐           ┌──────────────────────────────────────┐
│ background.js              │  context  │ /chat   RAG + Ollama, streamed (SSE)  │
│   right-click menu         │  menu →   │ /upload ingest documents              │
│ sidepanel.html/js/css      │  open     │ /import_bookmarks  index bookmarks    │
│   chat · upload · bookmarks│  panel    │ /query  raw hybrid retrieval          │
│                            │ ───────►  │ /documents /reset /models /health     │
└───────────────────────────┘  fetch    └──────────────┬───────────────────────┘
                                                        │
                                   ┌────────────────────┴───────────────┐
                                   │ rag/  (hybrid retriever)            │  Ollama
                                   │  ChromaDB ONNX embeddings + BM25    │  127.0.0.1:11434
                                   └─────────────────────────────────────┘
```

## Setup

### 1. Prerequisites

You need three things. Only the middle one is pure Python.

**a) Python 3.10+** — then install the backend dependencies (Flask, ChromaDB,
pypdf, etc.). A virtual environment is recommended:
```
cd backend
python -m venv .venv
# Windows:        .venv\Scripts\activate
# macOS / Linux:  source .venv/bin/activate
pip install -r requirements.txt
```
> First run downloads the embedding model once (ChromaDB's bundled ONNX
> all-MiniLM-L6-v2, ~80 MB). No PyTorch, no GPU, no API keys.

**b) Ollama** — this is a separate app, *not* a pip package. Install it from
<https://ollama.com/download>, then pull at least one model:
```
ollama serve                # usually already running after install
ollama pull qwen2.5:7b      # good default (needs ~5 GB free RAM)
ollama pull qwen2.5:0.5b    # tiny fallback for low-memory machines
```

**c) A Chromium browser** — Chrome or Edge, to load the extension (step 3).

### 2. Start the backend
```
backend\run.ps1      (PowerShell)   — or —   backend\run.bat
```
It listens on `http://127.0.0.1:5000`. Leave it running while you browse.

### 3. Load the extension
1. Open `chrome://extensions` (or `edge://extensions`).
2. Enable **Developer mode**.
3. Click **Load unpacked** and select the `extension/` folder.
4. Pin the extension and click its icon (or right-click a text selection) to
   open the side panel.

## Usage

- **Ask about a selection:** highlight text on any page → right-click →
  *“Ask Private AI about …”*. The side panel opens with the text prefilled.
- **Chat:** type in the composer. Toggle *“use my documents”* to ground answers
  in your indexed knowledge (on by default).
- **Add knowledge:** *Knowledge* tab → upload files, or **Import all bookmarks**
  (tick *fetch & read page text* to also index each bookmarked page's content).
- **Pick a model:** the header dropdown lists your installed Ollama models.

## Notes on memory

This machine has 16 GB RAM; 7B models need ~5 GB free. If a model can't load,
the backend automatically falls back to `qwen2.5:0.5b` and tells you in the
chat. Close other apps for best quality, or just select a smaller model.

## Configuration (env vars, optional)

| Variable | Default | Meaning |
|---|---|---|
| `PDB_MODEL` | `qwen2.5:7b` | default chat model |
| `PDB_FALLBACK_MODEL` | `qwen2.5:0.5b` | model used when the default won't load |
| `PDB_OLLAMA_URL` | `http://127.0.0.1:11434` | Ollama endpoint |
| `PDB_TOP_K` | `5` | passages retrieved per question |
| `PDB_DATA_DIR` | `backend/data` | where the index + uploads live |
| `PDB_MAX_MB` | `64` | max upload size |

## Privacy

Every component is local. The only outbound network requests are **optional**:
when you tick *fetch & read page text* during bookmark import, the backend
downloads those bookmarked pages (over your own connection) to index their
text. Disable it to stay fully air-gapped.

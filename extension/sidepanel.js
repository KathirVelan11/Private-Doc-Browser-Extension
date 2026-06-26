// Side panel UI. Talks ONLY to the local backend (http://127.0.0.1:5000),
// which in turn uses local Ollama + local RAG. No remote calls.

const BACKEND = "http://127.0.0.1:5000";

const $ = (id) => document.getElementById(id);
const messagesEl = $("messages");
const inputEl = $("input");
const modelEl = $("model");
const statusDot = $("status-dot");

// Conversation history sent to the model (system prompt is added server-side).
let history = [];
let streaming = false;

// ---------------------------------------------------------------- backend
async function checkHealth() {
  try {
    const r = await fetch(`${BACKEND}/health`);
    const d = await r.json();
    statusDot.classList.toggle("ok", true);
    statusDot.classList.toggle("bad", false);
    statusDot.title = `connected · ${d.total_chunks} chunks indexed`;
    populateModels(d.ollama_models, d.default_model);
  } catch (e) {
    statusDot.classList.toggle("ok", false);
    statusDot.classList.toggle("bad", true);
    statusDot.title = "backend offline — run backend/run.ps1";
    modelEl.innerHTML = '<option>backend offline</option>';
  }
}

function populateModels(models, def) {
  if (!models || !models.length) {
    modelEl.innerHTML = '<option value="">no models</option>';
    return;
  }
  const saved = localStorage.getItem("pdb_model");
  modelEl.innerHTML = models
    .map((m) => `<option value="${m}">${m}</option>`)
    .join("");
  modelEl.value = saved && models.includes(saved) ? saved : def || models[0];
}

modelEl.addEventListener("change", () =>
  localStorage.setItem("pdb_model", modelEl.value)
);

// ---------------------------------------------------------------- chat UI
function addMessage(role, text = "") {
  const hint = $("empty-hint");
  if (hint) hint.remove();
  const el = document.createElement("div");
  el.className = `msg ${role}`;
  const body = document.createElement("span");
  body.className = "body";
  body.textContent = text;
  el.appendChild(body);
  el._body = body; // dedicated text node so a sources box can follow it
  messagesEl.appendChild(el);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return el;
}

function renderSources(el, sources) {
  if (!sources || !sources.length) return;
  const box = document.createElement("div");
  box.className = "sources";
  box.innerHTML =
    "<div>Sources:</div>" +
    sources
      .map(
        (s) =>
          `<div class="src"><b>[${s.n}]</b> ${escapeHtml(s.source)} · p${s.page} · ${s.matched_by.join("+")}</div>`
      )
      .join("");
  el.appendChild(box);
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}

async function send() {
  if (streaming) return;
  const text = inputEl.value.trim();
  if (!text) return;
  inputEl.value = "";
  addMessage("user", text);
  history.push({ role: "user", content: text });

  const botEl = addMessage("assistant", "");
  botEl.classList.add("typing");
  streaming = true;
  $("send").disabled = true;

  let acc = "";
  try {
    const resp = await fetch(`${BACKEND}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        messages: history,
        model: modelEl.value,
        use_rag: $("use-rag").checked,
      }),
    });
    if (!resp.ok || !resp.body) throw new Error(`HTTP ${resp.status}`);

    await readSSE(resp.body, (event, data) => {
      if (event === "sources") {
        renderSources(botEl, data.sources);
      } else if (event === "notice") {
        acc += `(${data.message})\n\n`;
        botEl._body.textContent = acc;
      } else if (event === "token") {
        acc += data.t;
        botEl._body.textContent = acc;
        messagesEl.scrollTop = messagesEl.scrollHeight;
      } else if (event === "error") {
        botEl.classList.add("error");
        acc += `\n[error] ${data.message}`;
        botEl._body.textContent = acc;
      }
    });
    history.push({ role: "assistant", content: acc });
  } catch (e) {
    botEl.classList.add("error");
    botEl.textContent =
      "Could not reach the backend. Start it with backend/run.ps1 and make sure Ollama is running.";
  } finally {
    botEl.classList.remove("typing");
    streaming = false;
    $("send").disabled = false;
  }
}

// Minimal SSE reader over fetch's ReadableStream.
async function readSSE(stream, onEvent) {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let idx;
    while ((idx = buf.indexOf("\n\n")) >= 0) {
      const block = buf.slice(0, idx);
      buf = buf.slice(idx + 2);
      let event = "message";
      let dataStr = "";
      for (const line of block.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) dataStr += line.slice(5).trim();
      }
      if (dataStr) {
        try {
          onEvent(event, JSON.parse(dataStr));
        } catch {}
      }
    }
  }
}

$("send").addEventListener("click", send);
inputEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    send();
  }
});

// ---------------------------------------------------------------- selection
function applySelection(payload) {
  if (!payload || !payload.text) return;
  // Prefill the composer with the selection and a hint of where it came from.
  const src = payload.title ? `\n\n(from: ${payload.title})` : "";
  inputEl.value = `About this text:\n"""\n${payload.text.trim()}\n"""${src}\n\nExplain / answer:`;
  inputEl.focus();
  // place cursor at end
  inputEl.selectionStart = inputEl.selectionEnd = inputEl.value.length;
}

chrome.storage.session.get("pendingSelection").then((d) => {
  if (d.pendingSelection) {
    applySelection(d.pendingSelection);
    chrome.storage.session.remove("pendingSelection");
  }
});

chrome.runtime.onMessage.addListener((msg) => {
  if (msg && msg.type === "selection") applySelection(msg.payload);
});

// ---------------------------------------------------------------- tabs
document.querySelectorAll(".tab").forEach((t) =>
  t.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((x) => x.classList.remove("active"));
    t.classList.add("active");
    const which = t.dataset.tab;
    $("tab-chat").classList.toggle("hidden", which !== "chat");
    $("tab-knowledge").classList.toggle("hidden", which !== "knowledge");
    if (which === "knowledge") refreshDocs();
  })
);

// ---------------------------------------------------------------- knowledge
async function refreshDocs() {
  try {
    const r = await fetch(`${BACKEND}/documents`);
    const d = await r.json();
    const entries = Object.entries(d.documents || {});
    $("doc-list").innerHTML = entries.length
      ? entries
          .map(
            ([name, n]) =>
              `<div class="d"><span>${escapeHtml(name)}</span><span>${n}</span></div>`
          )
          .join("")
      : "Nothing indexed yet.";
  } catch {
    $("doc-list").textContent = "backend offline";
  }
}

$("upload").addEventListener("click", async () => {
  const files = $("files").files;
  if (!files.length) {
    $("upload-status").textContent = "Pick at least one file.";
    return;
  }
  const fd = new FormData();
  for (const f of files) fd.append("files", f);
  $("upload-status").textContent = "Indexing…";
  try {
    const r = await fetch(`${BACKEND}/upload`, { method: "POST", body: fd });
    const d = await r.json();
    const ok = (d.indexed || []).map((x) => `${x.source} (${x.chunks} chunks)`);
    $("upload-status").textContent =
      `Indexed: ${ok.join(", ") || "none"}` +
      (d.skipped && d.skipped.length ? `\nSkipped: ${d.skipped.join(", ")}` : "");
    refreshDocs();
    checkHealth();
  } catch {
    $("upload-status").textContent = "Upload failed — is the backend running?";
  }
});

$("import-bm").addEventListener("click", async () => {
  $("bm-status").textContent = "Reading bookmarks…";
  let bookmarks;
  try {
    bookmarks = await collectBookmarks();
  } catch {
    $("bm-status").textContent = "Could not read bookmarks (permission?).";
    return;
  }
  if (!bookmarks.length) {
    $("bm-status").textContent = "No bookmarks found.";
    return;
  }
  const fetchPages = $("fetch-pages").checked;
  $("bm-status").textContent = `Importing ${bookmarks.length} bookmarks${
    fetchPages ? " (fetching pages, this can take a while)…" : "…"
  }`;
  try {
    const r = await fetch(`${BACKEND}/import_bookmarks`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ bookmarks, fetch: fetchPages }),
    });
    const d = await r.json();
    $("bm-status").textContent = `Imported ${d.indexed} bookmarks${
      fetchPages ? `, read ${d.fetched_pages} pages` : ""
    }. Total chunks: ${d.total_chunks}.`;
    refreshDocs();
    checkHealth();
  } catch {
    $("bm-status").textContent = "Import failed — is the backend running?";
  }
});

function collectBookmarks() {
  return new Promise((resolve, reject) => {
    if (!chrome.bookmarks) return reject(new Error("no bookmarks api"));
    chrome.bookmarks.getTree((tree) => {
      const out = [];
      const walk = (nodes) => {
        for (const n of nodes) {
          if (n.url) out.push({ title: n.title || n.url, url: n.url });
          if (n.children) walk(n.children);
        }
      };
      walk(tree);
      resolve(out);
    });
  });
}

$("reset").addEventListener("click", async () => {
  if (!confirm("Delete ALL indexed documents and bookmarks?")) return;
  try {
    await fetch(`${BACKEND}/reset`, { method: "POST" });
    refreshDocs();
    checkHealth();
  } catch {}
});

// ---------------------------------------------------------------- init
checkHealth();
setInterval(checkHealth, 15000);

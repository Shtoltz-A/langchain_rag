# Roadmap

The single source of truth for "where are we and what's next". Update this file when a step is completed or re-planned (rule 3 in [`../AGENTS.md`](../AGENTS.md)).

Status markers: ✅ done · ⏭ current/next · ⬜ planned · 💡 backlog idea (unordered, not committed)

Each step follows the template: **Goal / Scope / Done when**.

---

## Step 1 — Baseline ✅

**Goal:** a working end-to-end naive RAG loop with proper foundations.

**Delivered:**
- Configuration layer (`config.py`): pydantic-settings + `.env`, cached `get_settings()`.
- Logging/console layer (`console.py`): colored, timestamped `Console` with debug/verbose support.
- Embeddings via Ollama (`embeddinggemma`), generation via Ollama chat model (`llama3.2`).
- Indexing: word-based recursive splitting → Chroma persistence (full rebuild per run).
- Querying: top-k similarity search → context assembly with char cap → anti-injection system prompt.
- Typer CLI: `app index`, `app query`, Ollama reachability check.

**Done when:** `uv run app index` + `uv run app query` works end-to-end on `data/raw/*.txt`. ✅

---

## Step 2 — Multi-Format Ingestion ⏭

**Goal:** ingest more than flat `.txt` files with clean, context-rich chunks. Replace the naive read-and-split in `app.index()` with a dedicated `ingestion/` package.

### 2.1 — Create the `ingestion` package ✅

```
src/langchain_rag/ingestion/
├── __init__.py     # exports load_documents()
├── loaders.py      # DocumentLoader base + txt/md/pdf loaders + registry
├── cleaning.py     # stub (Step 2.4)
└── chunking.py     # stub (Step 2.3)
```

### 2.2 — Loaders ✅

- Build a **Loader base class**; derive type-related loader classes from it. ✅ `DocumentLoader` extends `langchain_core.document_loaders.base.BaseLoader` (ctor takes `path`, declares `supported_suffixes`, implements `supports()`; subclasses implement `lazy_load()`).
- First loaders: `.txt`, `.md`, `.pdf`. ✅ `TextFileLoader`, `MarkdownLoader` (`.md`/`.markdown`, loaded as plain text for now), `PdfLoader` (raw `pypdf`, one Document per non-empty page with 0-based `page` metadata).
- `app.index()` dispatches through `load_documents()` ✅; unknown types skipped with a warning ✅, per-file errors reported and skipped ✅.

**Done:** verified 2026-10-29 — mixed txt/md/pdf (incl. uppercase suffixes) indexed end-to-end, PDF page metadata stored in Chroma and retrievable via `query`, corrupt/bad-encoding files skipped without aborting, double-index idempotent, zero-loadable run leaves old index intact with a warning.

### 2.3 — Smart chunking

- Chunking strategies per content type in `chunking.py` (replaces the inline `RecursiveCharacterTextSplitter` usage in `app.py`).
- Different treatment for plain text/markdown vs code (respect structure: headings/sections for text, functions/blocks for code).

### 2.4 — Cleaning

- **Normalize:** remove excessive `\n`, fix hyphenation from PDFs, remove repeated headers/footers.
- **Context prefix per chunk**, e.g.:
  - `[File: auth/login.py | function: login_user]\n...`
  - `[Doc: Policy.pdf p.12]\n...`
- **Optional (very powerful) — Contextual Retrieval:** use a small LLM to generate a 1-sentence summary for each chunk before embedding (e.g. *"This chunk implements JWT refresh token in auth.py"*), stored with the chunk.

**Done when:**
- A directory mixing `.txt`, `.md`, and `.pdf` files indexes successfully end-to-end.
- Chunks carry source metadata and human-readable context prefixes.
- Cleaning passes are covered by small, focused tests or verifiable sample fixtures.
- `architecture.md`, `configuration.md`, and `rag-decisions.md` updated for the new package and any new settings/decisions.

**Known implications to handle during implementation:**
- ~~New dependency for PDF extraction~~ → resolved in 2.2: raw `pypdf`, recorded in [rag-decisions.md](rag-decisions.md).
- Chunk settings may split into per-type settings once `chunking.py` lands — update `configuration.md` accordingly.

---

## Backlog (💡 unordered, TBD)

Capture future stage ideas here; promote into a numbered Step (with Goal/Scope/Done-when) when committing to them.

- 💡 Metadata filtering / scoped queries (per-collection or per-directory "all-purpose" separation)
- 💡 Hybrid retrieval (BM25 + vector) with RRF fusion
- 💡 Reranking of retrieved chunks
- 💡 Citations in answers (source references back to files/pages)
- 💡 Evaluation harness (small golden QA set, retrieval/answer quality metrics)
- 💡 Conversational/multi-turn querying
- 💡 API/server interface (beyond the CLI)
- 💡 Incremental indexing (only re-embed changed files) instead of full rebuild

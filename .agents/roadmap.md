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

### 2.3 — Smart chunking ✅

*Scope expanded during planning (user-approved): added `.py`/`.json` loaders and Markdown code-fence handling alongside the originally planned text/md splitting.*

- `chunking.py` mirrors the loaders pattern: `ChunkingStrategy` base (+`supported_suffixes`/`supports()`), `CHUNKING_STRATEGIES` tuple, `get_chunker(path)` lookup, public `split_documents(documents)` exported from `ingestion/__init__.py`. `app.index()` now calls `load_documents() → split_documents()` — the inline `RecursiveCharacterTextSplitter` is gone.
- Strategies: `TextChunker` (recursive, word-based), `MarkdownChunker` (own fence-aware line scanner: headings only outside ``` / `~~~` fences, h1–h3 metadata + heading lines kept in content, stub-section merging, fences atomic, oversized fences re-wrapped per fragment), `PdfChunker` (per-page recursive split, `source`/`page` propagate), `CodeChunker` (`from_language(Language.PYTHON)` → prefers def/class boundaries), `JsonChunker` (one chunk per top-level key + `json_key` metadata; oversized subtrees and non-dict roots fall back to text splitting).
- Settings split per type (9 fields, e.g. `chunk_size_markdown_words=300`/`chunk_overlap_markdown_words=50`); old `chunk_size_words`/`chunk_overlap_words` removed. **Unit fix:** chunk sizes are now genuinely word-based (`length_function=len(text.split())`) — previously `chunk_size_words=500` actually meant ~500 chars. See [rag-decisions.md](rag-decisions.md).
- New loaders: `PythonLoader` (.py), `JsonLoader` (.json, `json.loads` validation rejects malformed files via per-file tolerance).
- Per-document error tolerance in `split_documents()`: a strategy that raises is reported via `Console.error` and skipped; zero-chunk files warned.

**Done:** verified 2026-10-29 — 22 pytest unit tests (`tests/ingestion/`) cover dispatch, metadata survival (source/page/h1–h3/json_key), fence atomicity + `#`-in-fence not-a-heading, stub merging, size caps, JSON key splitting, and broken-doc tolerance; e2e `app index` + `app query` over mixed txt/md/pdf/py/json fixtures in a scratch dir.

### 2.4 — Cleaning ✅

- **Normalize:** remove excessive `\n`, fix hyphenation from PDFs, remove repeated headers/footers. ✅ `cleaning.normalize_documents()` — collapse blank-line runs + trim per-line trailing whitespace + normalize CRLF for all types; PDF-only: rejoin hyphen-broken words (`configu-\nration` → `configuration`) and drop repeated headers/footers (a line recurring on ≥3 pages AND ≥60% of a file's pages). Boilerplate detection runs on the post-hyphen text so joined content isn't mistaken for chrome.
- **Context prefix per chunk**, e.g.: ✅ `cleaning.apply_context_prefixes()` — one line at the top of each chunk's `page_content` (embeddings see it), metadata kept intact:
  - `[Doc: policy.pdf p.12]` (PDF, 1-based display from 0-based `page`)
  - `[File: runbook.md > Deployment > Rollback]` (Markdown heading chain from h1–h3)
  - `[File: config.json · key: database]` (JSON top-level key)
  - `[File: math.py]` / `[File: note.txt]` (code / text)
  - basename-only, idempotent (no double-prefix).
- **Optional — Contextual Retrieval:** deferred to a new numbered step (see 2.5) — deterministic cleaning proven first; index stays fast.

Pipeline order in `app.index()`: `load → normalize → split → prefix → store` (four explicit calls; normalize is document-level, prefix is chunk-level).

**Done when:** ✅ met 2026-10-29 — scratch dir of `.txt`/`.md`/`.pdf`(with repeated header+footer + hyphen-wrapped word)/`.py`/`.json` indexed end-to-end (reportlab-built PDF); every chunk carries a context prefix, PDF chrome dropped + hyphenation rejoined verified in debug dump; `query` works with prefixes in context; broken JSON still skipped. 43 pytest unit tests (`test_cleaning.py` + existing) green. Docs updated below.

**New settings:** `cleaning_enabled` (bool, default `True`) — kill-switch to compare retrieval with/without cleaning; both helpers are no-ops when `False`.

**Addendum (2026-10-29):** corpus grouped into `data/raw/<group>/` subfolders exposed a gap — the loader only read top-level files. Now `load_documents()` is recursive (hidden `.`-paths skipped) and every chunk prefix shows the folder trail via new `relative_path` metadata (`[Doc: pdf/geology_isidis.pdf p.1]`). See [rag-decisions.md](rag-decisions.md).

### 2.5 — Contextual Retrieval ⬜

- Use a small LLM to generate a 1-sentence summary for each chunk before embedding (e.g. *"This chunk implements JWT refresh token in auth.py"*), stored with the chunk and/or prepended to its embedded text.
- Cost/latency: one LLM call per chunk per index run — gate behind a setting; cache by chunk hash to keep re-indexes cheap.

**Done when:** summaries are generated and stored, retrieval quality vs. 2.4 measured on a small golden set, and the cost is acceptable or mitigated by caching.

**Known implications to handle during implementation:**
- ~~New dependency for PDF extraction~~ → resolved in 2.2: raw `pypdf`, recorded in [rag-decisions.md](rag-decisions.md).
- Chunk settings may split into per-type settings once `chunking.py` lands → resolved in 2.3: 9 per-type word-based fields in `Settings` (see [configuration.md](configuration.md)).

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

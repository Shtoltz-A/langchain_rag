# Roadmap

The single source of truth for "where are we and what's next" (rule 3 in [`../AGENTS.md`](../AGENTS.md)). Update on completion or re-planning.
Markers: ✅ done · ⬜ planned · 💡 backlog idea. Future steps follow the **Goal / Scope / Done when** template; design rationale lives in [rag-decisions.md](rag-decisions.md).

---

## Steps 1–3 ✅ (delivered)

**Step 1 — Baseline ✅:** config layer (`config.py` + `.env`, cached `get_settings()`) · `Console` · Ollama embeddings (`embeddinggemma`) + chat generation (`llama3.2`) · Chroma-persisted index · top-k query with char-capped context and anti-injection prompt · Typer CLI (`app index`, `app query`) with Ollama reachability check.

**Step 2 — Multi-Format Ingestion ✅:** `ingestion/` package replacing naive read-and-split:
- Loaders (`DocumentLoader` hierarchy): `.txt` `.md` `.markdown` `.pdf` `.py` `.json`; recursive walk, hidden-path skip, per-file error tolerance; PDFs → one Document per non-empty page.
- Chunking (`ChunkingStrategy` registry, dispatched by source suffix): word-based sizes per type (genuinely words, not chars), fence-aware Markdown splitter with h1–h3 metadata, per-page PDF, Python code, JSON top-level keys.
- Cleaning (gated by `cleaning_enabled`): document-level normalize (whitespace/CRLF; PDF hyphenation rejoin + header/footer dedup) before chunking; context prefixes (`[File: …]` / `[Doc: … p.N]`) after chunking.
- `relative_path` metadata (POSIX folder trail from index root) after the corpus moved to `data/raw/<group>/` subfolders.

**Step 3 — Storage Layer & Incremental Indexing ✅:** `storage/` package extracted from `app.py` (`VectorStore` singleton class + pure `manifest.py`). Manifest-driven incremental indexing: per-file SHA-256 diff re-embeds only added/changed files, purges vectors of deleted ones; full rebuild auto-triggered by fingerprint / root / manifest-vs-store count mismatch or `--full`. Per-file entry points (`iter_supported_files()`, `process_file()`) with batch-parity tests.

**Step 3.1 — Multi-Root Indexing ✅ (2026-10-30):** supersedes Step 3's root-change rebuild trigger and `--full`. The manifest keys per-root source records under normcase `roots` — a new `-d` folder is *added* to the collection (other roots untouched), a tracked folder diffs only its own records; `--full` becomes `--clear` as pure teardown (wipes the collection, deletes the manifest file, exits without indexing). Whole-store wipe (collection + all root records, loud warning lists dropped roots) only on unusable manifest, fingerprint change, or manifest-vs-store count mismatch — after one fires, only the requested folder is rebuilt, so re-index every folder again. Nested/overlapping roots are rejected with an error. The manifest carries no schema version: structure is the contract, an unrecognized file triggers a rebuild; format changes are handled by `--clear` (no migration code).

---

## Step 4 — Contextual Retrieval ⬜

**Goal:** better retrieval via per-chunk LLM context.

- Generate a 1-sentence summary per chunk before embedding (e.g. *"This chunk implements JWT refresh token in auth.py"*), stored with the chunk and/or prepended to its embedded text.
- Cost: one LLM call per chunk per index run — gate behind a setting; cache by chunk hash (Step 3 manifest hashing/diff is directly reusable).

**Done when:** summaries generated and stored, retrieval quality vs the Step 2.4 baseline measured on a small golden set, and cost acceptable or mitigated by caching.

---

## Backlog 💡 (unordered, TBD)

Promote into a numbered Step (with Goal/Scope/Done-when) when committing.

- 💡 Metadata filtering / scoped queries (per-collection or per-directory separation)
- 💡 Hybrid retrieval (BM25 + vector) with RRF fusion
- 💡 Reranking of retrieved chunks
- 💡 Citations in answers (source references back to files/pages)
- 💡 Evaluation harness (golden QA set, retrieval/answer quality metrics)
- 💡 Conversational/multi-turn querying
- 💡 API/server interface (beyond the CLI)
- 💡 `app status` command: manifest summary (files indexed, last index time, drift)

# Architecture

Module responsibilities, data flow, and directory layout. Keep updated when structure changes; *why* behind choices lives in [rag-decisions.md](rag-decisions.md).

## Directory layout

```
langchain_rag/
├── AGENTS.md                  # Hub: stage, docs map, always-on rules
├── .agents/                   # Detailed documentation (this directory)
├── .env                       # Local settings overrides (git-ignored)
├── data/
│   ├── raw/                   # Input documents to index
│   ├── chroma/                # Persisted Chroma collection
│   └── index_manifest.json    # Tracked sources + content hashes (incremental index)
├── tests/                     # Unit tests (project root, outside src/)
└── src/langchain_rag/
    ├── cli.py                 # Typer commands: index (--clear teardown), query
    ├── app.py                 # Core logic: index(), clear_index(), answer()
    ├── config.py              # Settings (pydantic-settings) + get_settings()
    ├── console.py             # Console: colored, timestamped terminal output
    ├── ingestion/             # File → Documents pipeline (Step 2)
    │   ├── __init__.py        # load/normalize/split/prefix + process_file
    │   ├── loaders.py         # DocumentLoader base + txt/md/pdf/py/json + registry
    │   ├── cleaning.py        # normalize_documents() + apply_context_prefixes()
    │   └── chunking.py        # ChunkingStrategy base + per-type strategies
    └── storage/               # Persistence layer (Step 3)
        ├── vectorstore.py     # VectorStore singleton class (Chroma + embeddings)
        └── manifest.py        # Multi-root manifest: hashes, fingerprint, diff (pure)
```

## Module responsibilities

- **`cli.py`** — the only UI layer: Typer commands, prerequisite checks (`_ensure_ollama_running()`), reporting via `Console`. Business logic never lives here (rule 6).
- **`app.py`** — `index(directory)`: `directory.resolve()` + `root_key()` (normcase) is the canonical root — manifest root keys, Chroma `source` metadata, and delete filters must all share this form. A never-seen directory is *added* alongside other roots' vectors; a tracked one updates via sha256 diff scoped to its own manifest records (→ delete stale vectors → `process_file()` + re-embed added/changed → skip unchanged). Only a whole-store wipe (no usable manifest / fingerprint change / manifest-vs-store count mismatch) clears the collection first, warning about dropped roots; overlapping roots raise `ValueError` (CLI: error + exit 1). Returns `IndexResult`.
  `clear_index()` is the deliberate teardown entry (CLI: `index --clear`): `store.clear()` then `delete_manifest()` — collection gone, manifest file deleted, nothing indexed; either crash order self-heals via the count-mismatch / no-manifest rebuild gates. Returns the removed-chunk count.
  `answer(question)`: `top_k` search → context joined and capped at `max_context_chars` → LLM with anti-injection system prompt.
- **`storage/vectorstore.py`** — only seam to `langchain_chroma`/`OllamaEmbeddings`. `VectorStore` process singleton (`__new__` memoizes; Chroma handle built on first instantiation — never instantiate at module level so imports stay disk-free). Methods `clear` / `add_documents` / `delete_by_source` / `search` / `count`; `chroma` property is the escape hatch.
- **`storage/manifest.py`** — pure (stdlib + settings): manifest grouping source records under normcase `roots` keys; `compute_fingerprint()` (global: pipeline version + embedding model + `chunk_*`/cleaning settings), `hash_file()` (SHA-256), atomic `load_manifest()`/`save_manifest()` (no schema version by design — structure is the contract: corrupt or unrecognized → `None` → whole-store rebuild; format changes handled by wiping, no migration code), `delete_manifest()` (tolerant unlink, used by the `--clear` teardown), `diff()` (one root's records), `matches()` (fingerprint only), `root_key()` / `overlapping_root()`.
- **`config.py`** — `Settings` read once via cached `get_settings()`; see [configuration.md](configuration.md).
- **`console.py`** — all terminal output; warning/error → stderr, debug only in verbose mode.

## Data flow

```
INDEX:  data/raw/* ──load_documents()──▶ Documents ──normalize_documents()──▶
        split_documents() ──apply_context_prefixes()──▶ chunks
        ──manifest gate──▶ whole-store wipe (clear collection + every root's records)
                        └▶ per-root update: sha256 diff on this root's files only —
                           added/changed: delete_by_source + process_file + add
                           deleted: delete_by_source · unchanged: skipped
        ▶ data/chroma/ (all roots in one collection) + data/index_manifest.json (roots-keyed)

QUERY:  question ─▶ VectorStore.search(top_k) ─▶ context (capped) ─▶ ChatOllama ─▶ answer

CLEAR:  index --clear ─▶ clear_index(): store.clear() + delete_manifest() ─▶ exit (nothing indexed)
```

## Ingestion contracts

- **Per-file tolerance everywhere:** broken or unknown files are reported (`Console.error`) and skipped, never fatal — `load_documents()`, `process_file()`, and `split_documents()` all follow this. A whole-store wipe clears the collection before scanning, so a run that loads nothing afterwards leaves the index empty.
- **Chunk metadata contract:** every chunk carries `source` (full path) and `relative_path` (POSIX, relative to index root); PDF adds `page` (0-based int), Markdown adds `h1`–`h3`, JSON adds `json_key`. Values are always strings/ints — Chroma rejects lists/None.
- **Cleaning** (`cleaning_enabled` gate; pure, input never mutated): `normalize_documents()` before chunking (blank-run/trailing-ws/CRLF all types; PDF-only hyphenation rejoin + header/footer dedup), `apply_context_prefixes()` after chunking (idempotent `[File: …]` / `[Doc: … p.N]` provenance line prepended to `page_content`).
- **Per-file entry points** (for incremental indexing): `iter_supported_files(directory)` (same walk rules as `load_documents()`, unsupported files excluded silently); `process_file(path, root)` = load → normalize → split → prefix for one file, chunk-for-chunk parity with the batch chain (parity-tested).

## Key structural facts

- All paths derive from `PROJECT_ROOT`; commands work from any CWD.
- Indexing is **incremental by design** — unchanged files are skipped by content hash, diffed per root (triggers + rationale: [rag-decisions.md](rag-decisions.md)).
- One Chroma collection (`"documents"`) shared by all index roots (per-root manifest sections keep their records separate); multi-collection scoping is roadmap backlog.

## Testing

`tests/` (run `uv run pytest`) covers pure logic only: chunking, cleaning, loaders (`tmp_path`), manifest hashing/diff/fingerprint, `VectorStore` singleton via fakes — never touches real Ollama or Chroma.

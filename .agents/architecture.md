# Architecture

Module responsibilities, data flow, and directory layout. The *why* behind choices lives in [rag-decisions.md](rag-decisions.md).

## Directory layout

```
langchain_rag/
├── AGENTS.md                  # Hub: current state, docs map, always-on rules
├── .agents/                   # Detailed documentation (this directory)
├── .env                       # Local settings overrides (git-ignored)
├── data/
│   ├── raw/                   # Input documents to index
│   ├── chroma/                # Persisted Chroma collection
│   └── index_manifest.json    # Tracked sources + content hashes per root
├── tests/                     # Offline unit tests (project root, outside src/)
└── src/langchain_rag/
    ├── cli.py                 # Typer commands: index (--clear teardown), query; global --verbose flag
    ├── app.py                 # Core logic: index(), clear_index(), answer()
    ├── config.py              # Settings (pydantic-settings) + get_settings()
    ├── console.py             # Console: colored, timestamped output; shared instance via get_console()
    ├── generation/            # Chat-model construction + invoke tracing + answer prompts
    │   ├── __init__.py        # Re-exports only: factories, trace seam, prompt, context helpers
    │   ├── llm.py             # build_chat_model(provider) + path factories + traced_invoke() — ONLY places LLM clients are built / calls logged
    │   ├── prompts.py         # ANSWER_SYSTEM_PROMPT (grounding + anti-injection)
    │   └── context.py         # format_context() + apply_budget() (pure)
    ├── ingestion/             # File → Documents pipeline
    │   ├── __init__.py        # load/normalize/split/prefix + process_file
    │   ├── loaders.py         # DocumentLoader base + txt/md/pdf/py/json + registry
    │   ├── cleaning.py        # normalize_documents() + apply_context_prefixes()
    │   └── chunking.py        # ChunkingStrategy base + per-type strategies
    ├── storage/               # Persistence layer
    │   ├── vectorstore.py     # VectorStore singleton class (Chroma + embeddings)
    │   └── manifest.py        # Multi-root manifest: hashes, fingerprint, diff (pure)
    └── retrieval/             # Query pipeline
        ├── __init__.py        # Re-exports only: retrieve, rerank, rrf_fuse
        ├── pipeline.py        # retrieve(): mode routing, fusion call, rerank, top-k cut; per-stage verbose traces
        ├── retrievers.py      # BM25Index + vector_search() + pure rrf_fuse()
        ├── display.py         # doc_label() / doc_preview(): chunk rendering for the verbose tables (pure)
        └── reranking.py       # Listwise LLM reranking, fail-open (opt-in); raw-score + result-order debug
```

## Module responsibilities

- **`cli.py`** — the only UI layer: Typer commands, prerequisite checks (`_ensure_ollama_running()`), reporting via `Console`. Business logic never lives here (rule 6).
- **`app.py`** — `index(directory)`: `directory.resolve()` + `root_key()` (normcase) is the canonical root — manifest keys, Chroma `source` metadata, and delete filters must all share this form. A never-seen directory is *added* alongside other roots; a tracked one updates via sha256 diff scoped to its own manifest records (delete stale vectors → `process_file()` + re-embed → skip unchanged). Only a whole-store wipe (no usable manifest / fingerprint change / manifest-vs-store count mismatch) clears the collection first; overlapping roots raise `ValueError`. Returns `IndexResult`.
  `clear_index()` is the deliberate teardown (CLI: `index --clear`): `store.clear()` + `delete_manifest()`, nothing indexed; either crash order self-heals via the rebuild gates. Returns the removed-chunk count.
  `answer(question)`: `retrieval.retrieve()` → `format_context()` + `apply_budget(max_context_chars)` → `traced_invoke("answer", …)` on `get_answer_llm()` with `ANSWER_SYSTEM_PROMPT`.
- **`storage/vectorstore.py`** — the only seam to `langchain_chroma`/`OllamaEmbeddings`. Process singleton (`__new__` memoizes; Chroma handle built on first instantiation — never at module level, so imports stay disk-free). Methods `clear` / `add_documents` / `delete_by_source` / `search` / `all_documents` / `count`; `chroma` property is the escape hatch.
- **`storage/manifest.py`** — pure (stdlib + settings): source records grouped under normcase `roots` keys; `compute_fingerprint()` (pipeline version + embedding model + `chunk_*`/cleaning), `hash_file()` (SHA-256), atomic `load_manifest()`/`save_manifest()` (no schema version — corrupt or unrecognized → `None` → whole-store rebuild; format changes handled by wiping), `delete_manifest()` (tolerant unlink), `diff()`, `matches()`, `root_key()` / `overlapping_root()`.
- **`generation/`** — the only LLM seam: `build_chat_model()` dispatches `ollama` (`ChatOllama`) vs `openai` (`ChatOpenAI` at `openai_base_url`, `SecretStr` key, SDK retries disabled where latency-bounded); path factories `get_answer_llm()` (cached, no temperature set) and `get_rerank_chat_model()` (temp 0, `rerank_num_ctx` on Ollama, `rerank_timeout`, `max_retries=0`); model falls back per provider (`OLLAMA_LLM_MODEL` / `openai_model`). `traced_invoke()` wraps every chat/structured call (backend, prompt size, duration, error class — verbose-gated, transparent passthrough so test fakes stay the seam). Missing gateway config raises `ValueError` — rerank turns it into a fail-open warning naming the missing `.env` key; `query` exits cleanly. The rerank prompt and score schema stay stage-private in `retrieval/reranking.py`.
- **`retrieval/`** — `retrieve(question)`: guards the empty store, fetches `top_k` (× rerank multiplier) candidates by `retrieval_mode` — `vector_search()` goes through the store facade; keyword/hybrid build one `BM25Index` per run from a single `all_documents()` load (above-zero scores only) — hybrid fuses via pure `rrf_fuse()` (dedupe by `doc_key`, `top_n` cutoff), then the optional listwise rerank (fail-open) returns ≤ `top_k` chunks. Verbose mode narrates every stage: durations, ranked `label :: preview` tables per retriever *before* fusion plus the fused list, raw rerank scores (pre-validation) and the keep/DROP order via `display.py`.
- **`config.py`** — `Settings` read once via cached `get_settings()`; see [configuration.md](configuration.md).
- **`console.py`** — all terminal output; warning/error → stderr, debug only in verbose mode. Shared process-wide instance from `get_console()` (verbose from the `verbose` setting; the CLI `--verbose` flag overrides).

## Data flow

```
INDEX:  data/raw/* ──load_documents()──▶ Documents ──normalize_documents()──▶
        split_documents() ──apply_context_prefixes()──▶ chunks
        ──manifest gate──▶ whole-store wipe (clear collection + every root's records)
                        └▶ per-root update: sha256 diff on this root's files only —
                           added/changed: delete_by_source + process_file + add
                           deleted: delete_by_source · unchanged: skipped
        ▶ data/chroma/ (all roots in one collection) + data/index_manifest.json (roots-keyed)

QUERY:  question ─▶ retrieval.retrieve(): VectorStore.search ∥ BM25 (by mode)
                  ─▶ RRF fuse ─▶ [LLM rerank? provider from settings] ─▶ top_k
                  ─▶ context (join + budget) ─▶ generation answer LLM (ollama | openai) ─▶ answer

CLEAR:  index --clear ─▶ clear_index(): store.clear() + delete_manifest() ─▶ exit (nothing indexed)
```

## Ingestion contracts

- **Per-file tolerance everywhere:** broken or unknown files are reported (`Console.error`) and skipped, never fatal — `load_documents()`, `process_file()`, and `split_documents()` all follow this. A whole-store wipe clears the collection before scanning, so a run that loads nothing afterwards leaves the index empty.
- **Chunk metadata contract:** every chunk carries `source` (full path) and `relative_path` (POSIX, relative to index root); PDF adds `page` (0-based int), Markdown adds `h1`–`h3`, JSON adds `json_key`. Values are always strings/ints — Chroma rejects lists/None.
- **Cleaning** (`cleaning_enabled` gate; pure, input never mutated): `normalize_documents()` before chunking (blank-run/trailing-ws/CRLF all types; PDF-only hyphenation rejoin + header/footer dedup), `apply_context_prefixes()` after chunking (idempotent `[File: …]` / `[Doc: … p.N]` provenance line prepended to `page_content`).
- **Per-file entry points** (for incremental indexing): `iter_supported_files(directory)` (same walk rules as `load_documents()`); `process_file(path, root)` = load → normalize → split → prefix for one file, chunk-for-chunk parity with the batch chain (parity-tested).

## Key structural facts

- All paths derive from `PROJECT_ROOT`; commands work from any CWD.
- Indexing is **incremental by design** — unchanged files are skipped by content hash, diffed per root (triggers + rationale: [rag-decisions.md](rag-decisions.md)).
- One Chroma collection (`"documents"`) shared by all index roots (per-root manifest sections keep their records separate); multi-collection scoping is roadmap backlog.
- Query-side modes and reranking never enter the index fingerprint: switching retrieval needs no re-index (deliberate, [rag-decisions.md](rag-decisions.md)).
- Chat clients, their providers, and the invoke trace (`traced_invoke`) live only in `generation/llm.py`; switching a backend is one `.env` line per path.
- Verbose mode is the project's whole observability layer: every stage self-reports via `Console.debug`, tables are built only when `Console.verbose` — no separate logging framework by design (rule 5).

## Testing

`tests/` (run `uv run pytest`) covers pure logic only: chunking, cleaning, loaders (`tmp_path`), manifest hashing/diff/fingerprint, `VectorStore` singleton via fakes, RRF arithmetic, mode routing and fail-open rerank (fake LLM), provider dispatch + invoke tracing + context assembly + `answer()` wiring + verbose-table rendering — never touches real Ollama, Chroma or the gateway. A global autouse fixture in `tests/conftest.py` clears every process-wide memoization (`get_settings`, cached LLM factories) around each test.

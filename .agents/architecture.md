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
│   ├── index_manifest.json    # Tracked sources + content hashes per root
│   └── chunk_summaries.json   # Content-keyed per-chunk LLM summaries (contextual retrieval)
├── tests/                     # Offline unit tests (project root, outside src/)
└── src/langchain_rag/
    ├── cli.py                 # Typer commands: index (--clear teardown, keeps the summary cache), query; global --verbose flag
    ├── app.py                 # Thin facade: re-exports services/ for CLI/tests (no logic)
    ├── config.py              # Settings (pydantic-settings) + get_settings()
    ├── console.py             # Console: colored, timestamped output (log/debug + verbose trace stage lines); shared instance via get_console()
    ├── services/              # Workflow coordinators (import leaf packages only)
    │   ├── __init__.py        # Re-exports only: index, clear_index, answer, IndexResult, ClearResult
    │   ├── models.py          # IndexResult / ClearResult (frozen Pydantic value objects)
    │   ├── indexing.py        # index()/clear_index(): invalidation gate, per-root diff, per-file pipeline
    │   └── answering.py       # answer(): retrieve → context budget → answer LLM
    ├── generation/            # Chat-model construction + invoke tracing + answer prompts
    │   ├── __init__.py        # Re-exports only: factories, trace seam, prompt, context helpers
    │   ├── llm.py             # build_chat_model(provider) + path factories + traced_invoke() — ONLY places LLM clients are built / calls logged
    │   ├── prompts.py         # ANSWER_SYSTEM_PROMPT (grounding + anti-injection)
    │   └── context.py         # format_context() + apply_budget() + build_budgeted_context() (pure)
    ├── ingestion/             # File → Documents pipeline
    │   ├── __init__.py        # load/normalize/split/prefix + process_file
    │   ├── loaders.py         # DocumentLoader base + txt/md/pdf/py/json + registry
    │   ├── cleaning.py        # normalize_documents() + apply_context_prefixes()
    │   ├── chunking.py        # ChunkingStrategy base + per-type strategies
    │   └── contextual.py      # apply_contextual_summaries(): per-chunk LLM context sentence (opt-in; the only LLM-touching module here)
    ├── storage/               # Persistence layer
    │   ├── vectorstore.py     # VectorStore singleton class (Chroma + embeddings)
    │   ├── manifest.py        # Multi-root manifest: hashes, fingerprint, diff (pure)
    │   └── summary_cache.py   # Chunk-summary cache: content-keyed, atomic, survives --clear (pure)
    └── retrieval/             # Query pipeline
        ├── __init__.py        # Re-exports only: retrieve, rerank, rrf_fuse
        ├── pipeline.py        # retrieve(): mode routing, fusion call, rerank, top-k cut; per-stage verbose traces
        ├── retrievers.py      # BM25Index + vector_search() + pure rrf_fuse()
        ├── display.py         # doc_label() / doc_preview(): chunk rendering for the verbose tables (pure)
        └── reranking.py       # Listwise LLM reranking, fail-open (opt-in); raw-score + result-order debug
```

## Module responsibilities

- **`cli.py`** — the only UI layer: Typer commands, prerequisite checks (`_ensure_ollama_running()`), reporting via `Console`. Business logic never lives here (rule 6).
- **`app.py`** — thin facade: re-exports `index` / `clear_index` / `answer` and `IndexResult` / `ClearResult` from `services/`, giving `cli.py` and tests one stable import surface. No logic lives here.
- **`services/models.py`** — `IndexResult` / `ClearResult` (frozen Pydantic models): the workflows' CLI-reporting value objects; immutability + JSON readiness for the future API surface.
- **`services/indexing.py`** — `index(directory)`: `directory.resolve()` + `root_key()` (normcase) is the canonical root — manifest keys, Chroma `source` metadata, and delete filters must all share this form. A never-seen directory is *added* alongside other roots; a tracked one updates via sha256 diff scoped to its own manifest records (delete stale vectors → `process_file()` → contextual stage → `add_documents()`; skip unchanged). With contextual retrieval enabled, the summarizer client is probed up front — a misconfigured `.env` aborts before any whole-store wipe — and each added/changed file runs through `apply_contextual_summaries()` between `process_file()` and the store; a `ContextualGenerationError` leaves that file out of the manifest (reported, retried whole on the next run). Only a whole-store wipe (no usable manifest / fingerprint change / manifest-vs-store count mismatch) clears the collection first; overlapping roots raise `ValueError`. Returns `IndexResult`.
  Splits: `_invalidation_reason()` is the pure wipe-vs-update gate; `_rebuild_store()` performs the wipe and returns the fresh manifest (persists nothing — `_update_root` owns every `save_manifest`); `_check_overlap()` refuses nesting and is consulted *only* when no rebuild fires (a wipe scenario re-indexes cleanly instead); `_update_root()` runs the diff + deletes + per-file loop + manifest saves; `_index_file()` is one file's process → summarize → store → record → trace pass (returns chunks stored, `None` on a summarizer failure so the file's manifest save is skipped).
  `clear_index()` is the deliberate teardown (CLI: `index --clear`): `store.clear()` + `delete_manifest()`, nothing indexed; either crash order self-heals via the rebuild gates. Returns `ClearResult` (removed chunks + stale segment folders pruned).
- **`services/answering.py`** — `answer(question)`: `retrieval.retrieve()` → `generation.build_budgeted_context(max_context_chars)` (traces the raw/budget lengths under verbose) → `traced_invoke("answer", …)` on `get_answer_llm()` with `ANSWER_SYSTEM_PROMPT`.
- **`storage/vectorstore.py`** — the only seam to `langchain_chroma`/`OllamaEmbeddings`. Process singleton (`__new__` memoizes; Chroma handle built on first instantiation — never at module level, so imports stay disk-free). Methods `clear` / `add_documents` / `delete_by_source` / `search` / `all_documents` / `count`; `chroma` property is the escape hatch. `clear()` also runs `prune_orphan_segment_dirs()` — chromadb's reset leaks the previous vector-segment dirs into `data/chroma/`, so the sweep (stdlib sqlite read of the `segments`/`collections` id tables; deletes only unreferenced UUID-named directories, fail-safe on any DB/OSError trouble) lives at the single wipe seam (rag-decisions.md).
- **`storage/manifest.py`** — pure (stdlib + settings): source records grouped under normcase `roots` keys; `compute_fingerprint()` (pipeline version + embedding model + `chunk_*`/cleaning settings, plus the contextual-retrieval block **only while enabled**), `hash_file()` (SHA-256), atomic `load_manifest()`/`save_manifest()` (no schema version — corrupt or unrecognized → `None` → whole-store rebuild; format changes handled by wiping), `delete_manifest()` (tolerant unlink), `diff()`, `matches()`, `root_key()` / `overlapping_root()`.
- **`storage/summary_cache.py`** — pure (stdlib + config): the chunk-summary cache at `data/chunk_summaries.json`, keyed by SHA-256 of chunk text (post-prefix, pre-summary) under a dependency key over prompt version + summarizer provider/model + source cap. Unreadable / corrupt / wrong-key ⇒ empty cache (everything is simply re-summarized: a stale cache is cost, never correctness — no migration code), atomic `save_summaries()`; deliberately survives `index --clear`, keeping rebuilds cheap.
- **`generation/`** — the only LLM seam: `build_chat_model()` dispatches `ollama` (`ChatOllama`) vs `openai` (`ChatOpenAI` at `openai_base_url`, `SecretStr` key, SDK retries disabled where latency-bounded); path factories `get_answer_llm()` (cached, no temperature set), `get_rerank_chat_model()` (temp 0, `rerank_num_ctx` on Ollama, `rerank_timeout`, `max_retries=0`) and `get_contextual_chat_model()` (cached, temp 0, `contextual_timeout`, `max_retries=0` — no structured-output wrapper on top, so the factory itself carries the one cache layer); model falls back per provider (`ollama_llm_model` / `openai_model`) via the single source `config.fallback_chat_model`, shared with the index fingerprint so both hash and dispatch the same model. `traced_invoke()` wraps every chat/structured call (backend, prompt size, duration, error class — verbose-gated, transparent passthrough so test fakes stay the seam). Missing gateway config raises `ValueError` — rerank turns it into a fail-open warning naming the missing `.env` key; `query` exits cleanly. The rerank prompt and score schema stay stage-private in `retrieval/reranking.py`.
- **`retrieval/`** — `retrieve(question)`: guards the empty store, fetches `top_k` (× rerank multiplier) candidates by `retrieval_mode` — `vector_search()` goes through the store facade; keyword/hybrid build one `BM25Index` per run from a single `all_documents()` load (above-zero scores only) — hybrid fuses via pure `rrf_fuse()` (dedupe by `doc_key`, `top_n` cutoff), then the optional listwise rerank (fail-open) returns ≤ `top_k` chunks. Verbose mode narrates every stage: durations, ranked `label :: preview` tables per retriever *before* fusion plus the fused list, raw rerank scores (pre-validation) and the keep/DROP order via `display.py`.
- **`config.py`** — `Settings` read once via cached `get_settings()`; see [configuration.md](configuration.md).
- **`console.py`** — all terminal output; warning/error → stderr; `debug`/`debug_block` and the styled `trace` stage summary (`›`/`✓` icon, bold label, ` · `-joined plain parts, verbose-only) gate on verbose mode. Styling lives here, never at call sites. Shared process-wide instance from `get_console()` (verbose from the `verbose` setting; the CLI `--verbose` flag overrides).

## Data flow

```
INDEX:  data/raw/* ──load_documents()──▶ Documents ──normalize_documents()──▶
        split_documents() ──apply_context_prefixes()──▶ chunks
        ──manifest gate──▶ whole-store wipe (clear collection + every root's records)
                        └▶ per-root update: sha256 diff on this root's files only —
                           added/changed: delete_by_source + process_file
                             ──[contextual on? cached summary, else one LLM call;
                                failed file → skipped now, retried next run]──▶ add
                           deleted: delete_by_source · unchanged: skipped
        ▶ data/chroma/ (all roots in one collection) + data/index_manifest.json (roots-keyed)

QUERY:  question ─▶ retrieval.retrieve(): VectorStore.search ∥ BM25 (by mode)
                  ─▶ RRF fuse ─▶ [LLM rerank? provider from settings] ─▶ top_k
                  ─▶ context (join + budget) ─▶ generation answer LLM (ollama | openai) ─▶ answer

CLEAR:  index --clear ─▶ clear_index(): store.clear() (+ segment-dir sweep) + delete_manifest() ─▶ exit (nothing indexed; chunk-summary cache kept)
```

## Ingestion contracts

- **Per-file tolerance everywhere:** broken or unknown files are reported (`Console.error`) and skipped, never fatal — `load_documents()`, `process_file()`, and `split_documents()` all follow this. A whole-store wipe clears the collection before scanning, so a run that loads nothing afterwards leaves the index empty.
- **Chunk metadata contract:** every chunk carries `source` (full path) and `relative_path` (POSIX, relative to index root); PDF adds `page` (0-based int), Markdown adds `h1`–`h3`, JSON adds `json_key`; with contextual retrieval enabled each summarized chunk also carries `context_summary` (one-line string; a chunk the summarizer returned empty for is stored without one, warned). Values are always strings/ints — Chroma rejects lists/None.
- **Cleaning** (`cleaning_enabled` gate; pure, input never mutated): `normalize_documents()` before chunking (blank-run/trailing-ws/CRLF all types; PDF-only hyphenation rejoin + header/footer dedup), `apply_context_prefixes()` after chunking (idempotent `[File: …]` / `[Doc: … p.N]` provenance line prepended to `page_content`).
- **Contextual summaries** (`contextual_retrieval_enabled` gate, off by default): `ingestion/contextual.py` prepends a `[Context: …]` one-sentence LLM summary to each chunk's text (idempotent) and copies it to `metadata["context_summary"]` — the line is part of the *stored* chunk text, so embedding, BM25 corpus, and answer context all see it. Applied by `services/indexing` between `process_file()` and the store via the `storage/summary_cache.py` cache, one LLM call per unseen chunk content; a failed summarize call fails the whole file (left out of the manifest, retried next run) while an empty model output indexes that chunk bare, warned. This is the only LLM-touching module in `ingestion/`; `process_file()` stays LLM-free so batch-chain parity holds.
- **Per-file entry points** (for incremental indexing): `iter_supported_files(directory)` (same walk rules as `load_documents()`); `process_file(path, root)` = load → normalize → split → prefix for one file, chunk-for-chunk parity with the batch chain (parity-tested).

## Key structural facts

- All paths derive from `PROJECT_ROOT`; commands work from any CWD.
- Indexing is **incremental by design** — unchanged files are skipped by content hash, diffed per root (triggers + rationale: [rag-decisions.md](rag-decisions.md)).
- One Chroma collection (`"documents"`) shared by all index roots (per-root manifest sections keep their records separate); multi-collection scoping is roadmap backlog.
- Query-side modes and reranking never enter the index fingerprint: switching retrieval needs no re-index (deliberate, [rag-decisions.md](rag-decisions.md)). Contextual-retrieval settings join the fingerprint **only while enabled** — with the feature off the payload is byte-identical to plain indexing (merging the feature never invalidates an existing index); enabling it *is* the one deliberate whole-store rebuild.
- Chat clients, their providers, and the invoke trace (`traced_invoke`) live only in `generation/llm.py`; switching a backend is one `.env` line per path.
- Verbose mode is the project's whole observability layer: every stage self-reports via `Console.debug`, tables are built only when `Console.verbose` — no separate logging framework by design (rule 5).

## Testing

`tests/` (run `uv run pytest`) covers pure logic only: chunking, cleaning, loaders (`tmp_path`), manifest hashing/diff/fingerprint, `VectorStore` singleton via fakes, RRF arithmetic, mode routing and fail-open rerank (fake LLM), provider dispatch + invoke tracing + context assembly + `answer()` wiring + verbose-table rendering — never touches real Ollama, Chroma or the gateway. A global autouse fixture in `tests/conftest.py` clears every process-wide memoization (`get_settings`, cached LLM factories) around each test.

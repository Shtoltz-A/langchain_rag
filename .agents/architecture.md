# Architecture

Module responsibilities, data flow, and directory layout. Keep this file updated when structure changes.

## Directory layout

```
langchain_rag/
├── AGENTS.md                  # Hub: mission, stage, docs map, always-on rules
├── .agents/                   # Detailed documentation (this directory)
├── pyproject.toml             # uv project: deps, `app` console script, uv_build backend
├── .env                       # Local overrides for settings (git-ignored)
├── data/
│   ├── raw/                   # Input documents to index
│   └── chroma/                # Persisted Chroma collection(s)
└── src/langchain_rag/
    ├── cli.py                 # Typer commands: index, query
    ├── app.py                 # Core logic: index(), answer()
    ├── config.py              # Settings (pydantic-settings) + get_settings()
    ├── console.py             # Console: colored, timestamped terminal output
    └── ingestion/             # File → Document pipeline (Step 2)
        ├── __init__.py        # Exports load/normalize/split/prefix functions
        ├── loaders.py         # DocumentLoader base + txt/md/pdf/py/json + registry
        ├── cleaning.py        # normalize_documents() + apply_context_prefixes()
        └── chunking.py        # ChunkingStrategy base + per-type strategies
```

> Tests live at the **project root** `tests/` (not inside `src/`); the tree above is the `src/langchain_rag/` package only.

## Module responsibilities

### `cli.py` — the only UI layer
- Defines the Typer app and commands (`index`, `query`); exposed as the `app` console script.
- Checks prerequisites before doing work (`_ensure_ollama_running()` pings the Ollama API).
- Reports results through `Console` and delegates all real work to `app.py`.
- **Rule:** command/argument parsing and user-facing flow live here; business logic never does.

### `app.py` — core RAG logic
- `index(directory)`: loads documents via `ingestion.load_documents()` → normalizes via `ingestion.normalize_documents()` → chunks via `ingestion.split_documents()` → prefixes via `ingestion.apply_context_prefixes()` → **resets** the collection → embeds and stores chunks in Chroma. Full rebuild, no incremental append.
- `answer(question)`: similarity search (`top_k`) → joins chunks into a context block → truncates to `max_context_chars` → invokes the chat LLM with a system prompt that treats context as data, not instructions.
- Owns the lazy `_get_vector_store()` singleton (Chroma, collection `"documents"`, persisted to `data/chroma/`) and a module-level `ChatOllama` instance.

### `config.py` — settings
- `Settings(BaseSettings)` with defaults, read once and cached by `get_settings()`; `.env` at project root overrides defaults, unknown keys ignored.
- See [configuration.md](configuration.md) for the full list of settings.

### `console.py` — output
- `Console` class: timestamped, level-colored messages (`info/warning/success/error/debug`), headings (`banner/section`), bordered multiline output (`block/debug_block`) with truncation unless verbose.
- `WARNING`/`ERROR` go to stderr; `DEBUG` output only in verbose mode.

## Data flow (current, Step 1)

```
INDEX                                     QUERY
─────                                     ─────
data/raw/* files                          question (CLI arg)
   │ ingestion.load_documents()              │
   │  recursive walk, hidden paths skipped;  │
   │  dispatch by suffix (case-insensitive): │
   │  .txt/.md/.py/.json → whole file        │
   │  .pdf → one Doc per non-empty page      │
   ▼                                         │
Document(page_content,                       │
  {source, relative_path[, page]})           │
   │  unknown types / broken files:          │
   │  warned + skipped                      │
   ▼                                         │
ingestion.normalize_documents()              │
   │  blank-run/trailing-ws trim (all);      │
   │  hyphenation + header/footer dedup (pdf)│
   ▼                                         │
ingestion.split_documents()              Chroma.similarity_search(k=top_k)
   │ per-type ChunkingStrategy,             │  Ollama embeddings embed the query
   │ routed by source suffix:               ▼
   │  md → fence-aware sections (h1–h3)   context = "\n\n".join(chunks)
   │  txt/pdf → recursive (words)           │ truncated to max_context_chars
   │  py → PYTHON from_language             ▼
   │  json → per top-level key            messages = [system, human(context, question)]
   ▼                                         │
ingestion.apply_context_prefixes()           ▼
   │ prepend "[File…]/[Doc…]" line        ChatOllama.invoke()
   ▼                                         │
Chroma.reset_collection()                    ▼
   │ add_documents(chunks)                answer printed via Console
   ▼
data/chroma/ persisted collection
```

## Key structural facts

- Chroma persist path is derived from `PROJECT_ROOT` (`src/langchain_rag/../..`), so commands work from any CWD.
- The vector store is a lazily-created module singleton; embeddings model comes from settings.
- Indexing is destructive by design: every `index` run replaces the whole collection (see [rag-decisions.md](rag-decisions.md)).
- There is currently one collection (`"documents"`). Multi-collection support for "all-purpose" behavior is a future-stage concern, tracked in the roadmap backlog.

## Ingestion pipeline (Step 2)

- `ingestion/loaders.py` defines `DocumentLoader` (extends LangChain `BaseLoader`): constructor takes a `Path`, `supported_suffixes` is a lowercase `frozenset`, subclasses implement `lazy_load()`.
- `load_documents(directory)` walks the tree recursively (`rglob`), skipping hidden paths (any component starting with `.`), dispatches via `get_loader()`/`SUPPORTED_LOADERS`, and consumes each loader **inside** a per-file `try` (lazy_load generators raise at consumption time), so broken files warn-and-skip instead of aborting the run.
- PDFs yield one `Document` per non-empty page with 0-based `page` metadata (LangChain convention; format as `p.{page+1}` at display time). `.json` files are validated with `json.loads` at load time; malformed ones are skipped per the same tolerance rule.
- `ingestion/chunking.py` mirrors the loader pattern: `ChunkingStrategy` base (`supported_suffixes`/`supports()`), `CHUNKING_STRATEGIES` tuple, `get_chunker(path)` lookup with `TextChunker` fallback. `split_documents()` groups documents by `metadata["source"]` suffix, chunks each group per type, and guards each group with try/except (`Console.error` + skip; zero-chunk groups warned).
- Chunk metadata contract: every chunk carries `source` (full path) and `relative_path` (POSIX path relative to the index root, e.g. `pdf/report.pdf`); PDF chunks add `page`; Markdown chunks add `h1`–`h3` (heading line stays in content too); JSON chunks add `json_key`. Values are always strings/ints (Chroma rejects lists/None).
- `ingestion/cleaning.py` runs two pure stages: `normalize_documents()` **before** chunking (blank-line collapse + per-line trailing-whitespace trim + CRLF normalization for all types; PDF-only hyphenation rejoin and header/footer dedup, detected on post-hyphen text) and `apply_context_prefixes()` **after** chunking (a `[File: …]` / `[Doc: … p.N]` provenance line prepended to `page_content`, idempotent, metadata preserved; the file part includes the folder trail from `relative_path`, e.g. `[Doc: pdf/report.pdf p.12]`). Both are no-ops when `cleaning_enabled=False`, and neither mutates its input documents.
- `cleaning.py` implements Step 2.4 (normalize + context prefixes). Contextual retrieval (LLM per-chunk summaries) is deferred to Step 2.5.

## Testing

- Unit tests live in `tests/` (run with `uv run pytest`), cover pure logic only (chunking strategies, cleaning passes, loaders with `tmp_path`) — no Ollama or Chroma access. Dev dependency: `pytest` (`[dependency-groups] dev`).

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
        ├── __init__.py        # Exports load_documents()
        ├── loaders.py         # DocumentLoader base + txt/md/pdf + registry
        ├── cleaning.py        # stub — content normalization (Step 2.4)
        └── chunking.py        # stub — per-type chunking (Step 2.3)
```

## Module responsibilities

### `cli.py` — the only UI layer
- Defines the Typer app and commands (`index`, `query`); exposed as the `app` console script.
- Checks prerequisites before doing work (`_ensure_ollama_running()` pings the Ollama API).
- Reports results through `Console` and delegates all real work to `app.py`.
- **Rule:** command/argument parsing and user-facing flow live here; business logic never does.

### `app.py` — core RAG logic
- `index(directory)`: loads documents via `ingestion.load_documents()` → splits into chunks → **resets** the collection → embeds and stores chunks in Chroma. Full rebuild, no incremental append.
- `answer(question)`: similarity search (`top_k`) → joins chunks into a context block → truncates to `max_context_chars` → invokes the chat LLM with a system prompt that treats context as data, not instructions.
- Owns the lazy `_get_vector_store()` singleton (Chroma, collection `"documents"`, persisted to `data/chroma/`) and a module-level `ChatOllama` instance.
- **Planned (Step 2.3):** chunking migrates from the inline splitter in `index()` to `ingestion/chunking.py`.

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
   │  dispatch by suffix (case-insensitive): │
   │  .txt/.md → whole file (utf-8)          │
   │  .pdf → one Doc per non-empty page      │
   ▼                                         │
Document(page_content, {source[, page]})     │
   │  unknown types / broken files:          │
   │  warned + skipped                      │
   ▼                                         ▼
RecursiveCharacterTextSplitter           Chroma.similarity_search(k=top_k)
(chunk_size/overlap in words)              │  Ollama embeddings embed the query
   │ chunks                                ▼
   ▼                                     context = "\n\n".join(chunks)
Chroma.reset_collection()                  │ truncated to max_context_chars
   │ add_documents(chunks)                 ▼
   ▼                                     messages = [system, human(context, question)]
data/chroma/ persisted collection          │
                                           ▼
                                         ChatOllama.invoke()
                                           │
                                           ▼
                                         answer printed via Console
```

## Key structural facts

- Chroma persist path is derived from `PROJECT_ROOT` (`src/langchain_rag/../..`), so commands work from any CWD.
- The vector store is a lazily-created module singleton; embeddings model comes from settings.
- Indexing is destructive by design: every `index` run replaces the whole collection (see [rag-decisions.md](rag-decisions.md)).
- There is currently one collection (`"documents"`). Multi-collection support for "all-purpose" behavior is a future-stage concern, tracked in the roadmap backlog.

## Ingestion pipeline (Step 2)

- `ingestion/loaders.py` defines `DocumentLoader` (extends LangChain `BaseLoader`): constructor takes a `Path`, `supported_suffixes` is a lowercase `frozenset`, subclasses implement `lazy_load()`.
- `load_documents(directory)` iterates the top-level files (not recursive), dispatches via `get_loader()`/`SUPPORTED_LOADERS`, and consumes each loader **inside** a per-file `try` (lazy_load generators raise at consumption time), so broken files warn-and-skip instead of aborting the run.
- PDFs yield one `Document` per non-empty page with 0-based `page` metadata (LangChain convention; format as `p.{page+1}` at display time).
- `cleaning.py` and `chunking.py` are docstring-only stubs until Steps 2.4/2.3.

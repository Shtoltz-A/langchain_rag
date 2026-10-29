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
    └── console.py             # Console: colored, timestamped terminal output
```

## Module responsibilities

### `cli.py` — the only UI layer
- Defines the Typer app and commands (`index`, `query`); exposed as the `app` console script.
- Checks prerequisites before doing work (`_ensure_ollama_running()` pings the Ollama API).
- Reports results through `Console` and delegates all real work to `app.py`.
- **Rule:** command/argument parsing and user-facing flow live here; business logic never does.

### `app.py` — core RAG logic
- `index(directory)`: reads files from a directory → splits into chunks → **resets** the collection → embeds and stores chunks in Chroma. Full rebuild, no incremental append.
- `answer(question)`: similarity search (`top_k`) → joins chunks into a context block → truncates to `max_context_chars` → invokes the chat LLM with a system prompt that treats context as data, not instructions.
- Owns the lazy `_get_vector_store()` singleton (Chroma, collection `"documents"`, persisted to `data/chroma/`) and a module-level `ChatOllama` instance.
- **Planned (Step 2):** `index()` will delegate loading/cleaning/chunking to the `ingestion/` package instead of reading text files and splitting inline.

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
   │ read_text (utf-8, whole file)           │
   ▼                                         │
Document(page_content, {source})             │
   │                                         │
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

## Planned structure after Step 2

```
src/langchain_rag/
└── ingestion/
    ├── __init__.py
    ├── loaders.py     # Loader base class + type-specific loaders (.txt, .md, .pdf)
    ├── cleaning.py    # Normalize extracted content
    └── chunking.py    # Chunking strategies per content type (text vs code)
```

`app.index()` will call into `ingestion/` instead of doing file-reading and splitting itself.

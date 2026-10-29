# Commands

Everything needed to run, index, and query. Run all commands from the project root unless noted.

## Prerequisites

- Python 3.12 (pinned via `.python-version`), [`uv`](https://docs.astral.sh/uv/) installed.
- [Ollama](https://ollama.com) running locally:

```powershell
ollama serve                      # if not already running as a service
ollama pull llama3.2              # chat model  (OLLAMA_LLM_MODEL)
ollama pull embeddinggemma        # embedding model (OLLAMA_EMBEDDING_MODEL)
ollama list                       # verify
```

## Setup

```powershell
uv sync                           # create/refresh .venv from uv.lock
```

## Add a dependency

```powershell
uv add <package>                  # runtime dependency
uv add --dev <package>            # development-only dependency
```

Never `pip install` directly; commit the updated `pyproject.toml` and `uv.lock` together.

## Usage

The Typer app is exposed as the console script `app` (see `[project.scripts]`).

```powershell
# Index documents (default directory: data/raw)
uv run app index
uv run app index --directory path/to/docs     # or -d

# Ask a question against the index
uv run app query "What does the policy say about X?"

# Help
uv run app --help
uv run app index --help
```

### Behavior notes

- `index` **replaces** the whole collection on every run (reset + add). Indexing an empty directory leaves the store empty.
- `index` reads every file in the directory as UTF-8 text; non-text/binary formats fail until Step 2 adds real loaders.
- Both commands first check that Ollama is reachable and exit with a clear message if not.
- Verbose diagnostics (chunk dumps, retrieved context) are controlled by the `Console(verbose=...)` flag inside `app.py` — there is no `--verbose` CLI flag yet.

## Quick smoke test

```powershell
# 1. Put a .txt file into data/raw
# 2. Index it
uv run app index
# 3. Query something only that file knows
uv run app query "What is in the test document?"
```

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `Ollama is not reachable` | Start `ollama serve`; check `OLLAMA_HOST` |
| Model-not-found error from Ollama | `ollama pull <model>` matching the setting name |
| Answers claim nothing is indexed | Re-run `uv run app index` (store may have been reset or embedding model changed) |
| `UnicodeDecodeError` during index | File is not UTF-8 text — remove/convert it (real loaders arrive in Step 2) |

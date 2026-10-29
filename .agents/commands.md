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

- `index` **replaces** the whole collection on every run (reset + add). If nothing loadable is found, the run stops with a warning and the existing index stays untouched.
- Supported input formats: `.txt`, `.md`/`.markdown`, `.pdf` (one chunk-source document per non-empty PDF page), `.py`, `.json` (validated at load). Unknown file types and unreadable files (bad encoding, corrupt PDF, malformed JSON) are reported and skipped, never fatal.
- Both commands first check that Ollama is reachable and exit with a clear message if not.
- Verbose diagnostics (chunk dumps with metadata, retrieved context) are controlled by the `Console(verbose=...)` flag inside `app.py` — there is no `--verbose` CLI flag yet.

## Tests

```powershell
uv run pytest                     # unit tests (tests/): chunking + loaders, no Ollama needed
```

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
| `Failed to load <file>` during index | File skipped on purpose (bad encoding, corrupt PDF); other files still indexed |
| `UnicodeEncodeError` (`charmap`) when piping output | Known console issue: box-drawing characters fail when stdout is redirected (cp1252). Run in a normal terminal or set `$env:PYTHONIOENCODING='utf-8'` |

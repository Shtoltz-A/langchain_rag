# Commands

Run everything from the project root. Windows note: when piping CLI output, set `$env:PYTHONIOENCODING='utf-8'` first (box-drawing chars fail on cp1252).

## Prerequisites & setup

Python 3.12 (`uv`-managed) and Ollama running locally with the models from `configuration.md`:

```powershell
ollama serve                      # if not already a service
ollama pull llama3.2              # OLLAMA_LLM_MODEL
ollama pull embeddinggemma        # OLLAMA_EMBEDDING_MODEL
uv sync                           # create/refresh .venv from uv.lock
```

Dependencies: `uv add <pkg>` / `uv add --dev <pkg>` only — never `pip install`; commit `pyproject.toml` + `uv.lock` together.

## Usage

The Typer app is exposed as the console script `app`.

```powershell
uv run app index                              # incremental: re-embeds only new/changed files
uv run app index -d path/to/docs              # add this folder to the index (other folders untouched)
uv run app index --clear                      # delete ALL indexed data (vector store + manifest file) and exit — nothing is indexed, -d ignored
uv run app query "What does the policy say about X?"
uv run app --help
uv run pytest                                 # unit tests — no Ollama/Chroma needed
```

## Behavior notes

- **Incremental `index`** (Steps 3/3.1): diffs per-file SHA-256 against `data/index_manifest.json`; deleted files' vectors are purged. Indexing different `-d` folders repeatedly accumulates one combined index — a new folder is *added*, tracked folders diff only their own records; a folder nested inside (or around) an already-indexed root errors out, index the outer folder instead. Whole-store rebuild (wipes **every** indexed folder) on missing/unusable manifest, fingerprint change (embedding model, chunking/cleaning settings, pipeline version), or manifest-vs-store count disagreement. `--clear` is a separate pure teardown, not a rebuild trigger: it deletes ALL indexed data (vector store + manifest file) and exits without indexing; the next plain `index` starts from scratch on whatever `-d` folder you give it. Formats: `.txt` `.md` `.markdown` `.pdf` `.py` `.json`; unknown/broken files are reported and skipped, never fatal. Both commands exit early with a clear message if Ollama is unreachable. Verbose chunk/context dumps are behind `Console(verbose=...)` in `app.py` (no CLI flag yet).
- Quick smoke test: put a `.txt` in `data/raw` → `uv run app index` → `uv run app query "what's in the test doc?"`.

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `Ollama is not reachable` | Start `ollama serve`; check `OLLAMA_HOST` |
| Model-not-found from Ollama | `ollama pull <model>` matching the setting name |
| Answers claim nothing is indexed | Re-run `uv run app index` — wiped store or changed model triggers an auto whole-store wipe (debug line `Whole-index rebuild: <reason>`); afterwards re-index **every** folder again, the other roots were dropped |
| `Failed to load <file>` during index | Skipped on purpose (bad encoding, corrupt PDF); other files still indexed |
| `UnicodeEncodeError` (`charmap`) when piping output | Set `$env:PYTHONIOENCODING='utf-8'` or run in a normal terminal |

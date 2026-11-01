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
uv run app --verbose query "..."   # or -v: full pipeline debug output; also via `verbose=true` in `.env`
uv run app --help
uv run pytest                                 # unit tests — no Ollama/Chroma needed
```

Optional gateway reranking (what the local `.env` enables): paste your key into `OPENAI_API_KEY` in `.env` — `OPENAI_BASE_URL` / `OPENAI_MODEL` already point at it. Without a key, rerank warns once per query and keeps retrieval order; answering and embeddings work regardless.

## Behavior notes

- **Incremental `index`:** per-file SHA-256 diff against `data/index_manifest.json`, scoped per root; deleted files' vectors are purged. Indexing different `-d` folders repeatedly accumulates one combined index — a new folder is *added*, tracked folders diff only their own records; a folder nested inside (or around) an indexed root errors out (index the outer folder instead). A whole-store rebuild (wipes **every** indexed folder) fires only on missing/unusable manifest, fingerprint change (embedding model, chunking/cleaning settings, pipeline version), or manifest-vs-store count disagreement — afterwards re-index **every** folder. `--clear` is a pure teardown, not a rebuild trigger. Formats: `.txt` `.md` `.markdown` `.pdf` `.py` `.json`; unknown/broken files are reported and skipped, never fatal. Both commands exit early with a clear message if Ollama is unreachable.
- **Query modes:** `RETRIEVAL_MODE=vector|keyword|hybrid` picks candidate generation; hybrid fuses vector + BM25 rankings with RRF (`RRF_C`, `HYBRID_VECTOR_WEIGHT`, `HYBRID_BM25_WEIGHT`). `RERANK_ENABLED=true` fetches `top_k × RERANK_CANDIDATE_MULTIPLIER` candidates and reorders them with one extra LLM call per query — noticeably slower on local hardware; on any LLM trouble it warns and keeps fusion order. Switching modes never needs a re-index. BM25 only returns chunks sharing a weighted term with the query, so `keyword` mode can yield fewer than `top_k` — or none on a non-empty store ("No relevant chunks found").
- **Verbose output (`app --verbose` / `-v`, or `VERBOSE=true`):** every pipeline step reports itself via the shared `Console` — per-stage durations (vector search incl. query embedding, BM25 build+search), RRF parameters, ranked `#rank label :: preview` tables per retriever *before* fusion plus the fused list; `LLM[answer]`/`LLM[rerank]` lines per call (backend, prompt size, duration, error class); `rerank scores (raw)` before validation plus a keep/DROP result table; context size vs `max_context_chars`, answer length, per-file `Loaded → Split → Indexed … in X.Xs`. Final summary: `Retrieval: mode=…, candidates=…, kept=…, rerank=on(provider/model)|off`. The tables embed chunk previews — don't pipe verbose output of private corpora into shared logs.
- Quick smoke test: put a `.txt` in `data/raw` → `uv run app index` → `uv run app query "what's in the test doc?"`.

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `Ollama is not reachable` | Start `ollama serve`; check `OLLAMA_HOST` |
| Model-not-found from Ollama | `ollama pull <model>` matching the setting name |
| Answers claim nothing is indexed | Re-run `uv run app index` — wiped store or changed model triggers an auto whole-store wipe (debug line `Whole-index rebuild: <reason>`); afterwards re-index **every** folder again, the other roots were dropped |
| `Failed to load <file>` during index | Skipped on purpose (bad encoding, corrupt PDF); other files still indexed |
| `UnicodeEncodeError` (`charmap`) when piping output | Set `$env:PYTHONIOENCODING='utf-8'` or run in a normal terminal |
| Queries much slower after enabling rerank | Expected — one extra LLM call per query; set `RERANK_ENABLED=false`, or `RERANK_PROVIDER=ollama`, or point `RERANK_MODEL`/`OPENAI_MODEL` at a smaller model |
| `Rerank failed (…OPENAI_API_KEY is not set in .env)` | Paste the gateway key into `.env` — or set `RERANK_PROVIDER=ollama` to rank locally; rerank is fail-open, answers still work |
| `Rerank failed (scored 0 of N documents)` | The model answered with empty scores — the N-document listwise prompt is too much for it (3B-class always fails at N=15); use a bigger `OPENAI_MODEL`/`RERANK_MODEL` |
| `Rerank failed (…ConnectionError/timeout)` with the openai provider | The gateway needs intranet/VPN; check `OPENAI_BASE_URL`. Fail-open by design — queries keep retrieval order; set `RERANK_PROVIDER=ollama` to silence |

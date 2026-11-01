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
uv run app index --clear                      # delete ALL indexed data (vector store + manifest file + leaked Chroma segment folders) and exit — nothing is indexed, -d ignored; the chunk-summary cache is kept
uv run app query "What does the policy say about X?"
uv run app --verbose query "..."   # or -v: full pipeline debug output; also via `verbose=true` in `.env`
uv run app --help
uv run pytest                                 # unit tests — no Ollama/Chroma needed
uv run ruff check .                           # lint (ruff is a pinned dev dep; config in pyproject [tool.ruff])
uv run ruff format --check .                  # formatting verification (drop --check to apply)
```

Optional gateway reranking (what the local `.env` enables): paste your key into `OPENAI_API_KEY` in `.env` — `OPENAI_BASE_URL` / `OPENAI_MODEL` already point at it. Without a key, rerank warns once per query and keeps retrieval order; answering and embeddings work regardless.

Contextual retrieval (off by default): set `CONTEXTUAL_RETRIEVAL_ENABLED=true` in `.env` — or per run, `$env:CONTEXTUAL_RETRIEVAL_ENABLED='true'; uv run app index …`. The summarizer is a third chat path: `CONTEXTUAL_PROVIDER` (default `ollama`) and `CONTEXTUAL_MODEL` (falls back to `OLLAMA_LLM_MODEL` / `OPENAI_MODEL`), with `CONTEXTUAL_SOURCE_MAX_CHARS` capping what each prompt sends and `CONTEXTUAL_TIMEOUT` bounding each call.

## Behavior notes

- **Incremental `index`:** per-file SHA-256 diff against `data/index_manifest.json`, scoped per root; deleted files' vectors and manifest records are purged together, so a re-index that fails mid-file retries as an added file next run instead of tripping the count gate; files that cannot be hashed (locked/vanished) keep their record and stay unchanged. Indexing different `-d` folders repeatedly accumulates one combined index — a new folder is *added*, tracked folders diff only their own records; a folder nested inside (or around) an indexed root errors out (index the outer folder instead). A whole-store rebuild (wipes **every** indexed folder) fires only on missing/unusable manifest, fingerprint change (embedding model, chunking/cleaning settings, pipeline version, or enabling contextual retrieval / changing its provider, resolved model, or source cap), or manifest-vs-store count disagreement — afterwards re-index **every** folder. `--clear` is a pure teardown, not a rebuild trigger. Every collection wipe (`--clear` or a whole-store rebuild) also sweeps stale HNSW segment folders Chroma leaks into `data/chroma/` (rag-decisions.md). Formats: `.txt` `.md` `.markdown` `.pdf` `.py` `.json`; unknown/broken files are reported and skipped, never fatal. Both commands exit early with a clear message if Ollama is unreachable.
- **Query modes:** `RETRIEVAL_MODE=vector|keyword|hybrid` picks candidate generation; hybrid fuses vector + BM25 rankings with RRF (`RRF_C`, `HYBRID_VECTOR_WEIGHT`, `HYBRID_BM25_WEIGHT`). `RERANK_ENABLED=true` fetches `top_k × RERANK_CANDIDATE_MULTIPLIER` candidates and reorders them with one extra LLM call per query — noticeably slower on local hardware; on any LLM trouble it warns and keeps fusion order. Switching modes never needs a re-index. BM25 only returns chunks sharing a weighted term with the query, so `keyword` mode can yield fewer than `top_k` — or none on a non-empty store ("No relevant chunks found").
- **Contextual retrieval (`CONTEXTUAL_RETRIEVAL_ENABLED=true`):** indexing spends one LLM call per *unseen chunk content* to prepend a `[Context: …]` sentence (visible in embeddings, BM25, and answer context). The first enable is one whole-store rebuild; afterwards summaries are cached by chunk content in `data/chunk_summaries.json` — `index --clear` keeps that cache, so rebuilds re-summarize nothing; it re-summarizes everything only after a provider/model/prompt/source-cap change. If the summarizer is unreachable mid-file, that file is reported, NOT indexed, and retried whole on the next `index` run (already-generated summaries are cached, so the retry is partial); a chunk whose model output is empty is indexed without a summary, warned. A summarizer misconfiguration (e.g. missing gateway keys) aborts before the first-enable whole-store wipe — your index survives a broken `.env`.
- **Verbose output (`app --verbose` / `-v`, or `VERBOSE=true`):** every pipeline step reports itself once, through the shared `Console`. Indexing: one `✓ relative\path · N chunks · X.Xs` trace line per indexed file (diff counts, whole-rebuild reasons and segment-dir pruning stay gray debug lines; the summary cache counts ride on the per-file `› summaries · N new · M cached` line when contextual retrieval is on, followed by the generated summaries themselves). Query: `› query …` (store size), `› vector|keyword|rrf … · N/M candidates · X.Xs` per stage, one `› LLM answer|rerank|summarize · backend · N → M chars · X.Xs` line per call (error class instead on failure), `› context …` budget line, final `✓ kept N/M · rerank on(provider/model)|off`. The ranked `#rank label :: preview` tables per retriever *before* fusion, the fused list, `rerank scores (raw)` (pre-validation) and the keep/DROP table stay as bordered blocks. The tables embed chunk previews — don't pipe verbose output of private corpora into shared logs.
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
| `Contextual summaries failed for <file> … File NOT indexed` | Summarizer unreachable/too slow — fix the backend settings or raise `CONTEXTUAL_TIMEOUT`, then re-run `uv run app index`: the file is retried whole and its already-cached summaries are reused |
| Index aborts: `contextual summaries uses the openai provider but … is not set in .env` | Summarizer misconfiguration — set the gateway keys or use `CONTEXTUAL_PROVIDER=ollama`; the probe runs before any whole-store wipe, so the existing index is untouched |

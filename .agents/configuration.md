# Configuration

All runtime settings live in `src/langchain_rag/config.py` as one pydantic-settings model. Nothing reads the environment outside this module (rule 4 in [`../AGENTS.md`](../AGENTS.md)).

`Settings` field names are lower `snake_case`; `.env` keys use SCREAMING_SNAKE_CASE and mirror them (`LLM_PROVIDER` → `llm_provider`) — matching is case-insensitive, and that split is the documentation convention too: `.env`/environment keys uppercase, Python fields lowercase. `.env` ships every everyday setting (providers, models, retrieval mode, feature toggles); tuning-only settings (chunk sizes, fusion constants, prompt caps, timeouts) keep their code defaults in `config.py` and join `.env` only on deliberate override. Block and field order in `config.py` mirror `.env`.

## Settings reference

“In `.env” marks the settings the shipped `.env` sets a key for (✓, uppercase); “no” means tuning-only, the default lives in `config.py`. Table rows list the snake_case field names the code uses.

| Setting | Type | Default | In `.env` | Purpose |
|---|---|---|---|---|
| `llm_provider` | `Literal` | `ollama` | ✓ | Answer-LLM backend: `ollama` or `openai` (any OpenAI-compatible gateway) |
| `llm_model` | `str \| None` | *(none)* | no | Answer model override; falls back to `ollama_llm_model` / `openai_model` per provider |
| `rerank_provider` | `Literal` | `ollama` | ✓ | Reranker backend; the local `.env` flips it to the gateway (small local models return empty score lists on big listwise prompts) |
| `rerank_model` | `str \| None` | *(none)* | no | Rerank model override; falls back to `ollama_llm_model` (ollama) or `openai_model` (openai) |
| `ollama_llm_model` | `str` | `llama3.2` | ✓ | Ollama chat model for every ollama-provider path (must be pulled in Ollama) |
| `ollama_embedding_model` | `str` | `embeddinggemma` | ✓ | Embedding model for indexing and query embedding |
| `openai_model` | `str` | *(empty)* | ✓ | Model name on the gateway, used by every path whose provider is `openai` |
| `openai_base_url` | `str` | *(empty)* | ✓ | Gateway API root URL (`…/v1`); required when a path selects `openai` |
| `openai_api_key` | `SecretStr \| None` | *(none)* | ✓ | Gateway key; an empty `openai_api_key=` counts as unset; never logged |
| `retrieval_mode` | `Literal` | `vector` | ✓ | Candidate source: `vector`, `keyword` (BM25) or `hybrid` (RRF-fused) |
| `top_k` | `int` | `5` (min 1) | ✓ | Number of chunks retrieved per query |
| `rerank_enabled` | `bool` | `false` | ✓ | Reorder over-fetched candidates with one listwise LLM call; fail-open on trouble |
| `max_context_chars` | `int` | `12000` (min 1) | no | Hard cap on the assembled context string before prompting |
| `rrf_c` | `int` | `60` | no | RRF damping constant: `score += weight / (rrf_c + rank)` |
| `hybrid_vector_weight` | `float` | `0.5` | no | Influence of the vector ranking in hybrid fusion |
| `hybrid_bm25_weight` | `float` | `0.5` | no | Influence of the BM25 ranking in hybrid fusion |
| `rerank_candidate_multiplier` | `int` | `3` | no | Candidates fetched (`top_k` × this) before reranking |
| `rerank_doc_max_chars` | `int` | `2000` | no | Per-candidate character cap in the rerank prompt |
| `rerank_num_ctx` | `int` | `16384` | no | Context window for the listwise rerank prompt (Ollama only; gateways manage context server-side) |
| `rerank_timeout` | `float` | `120` | no | Seconds before the rerank call is abandoned (fail-open) |
| `contextual_retrieval_enabled` | `bool` | `false` | ✓ | Indexing-side gate: prepend a one-sentence LLM context summary to each new chunk before it is embedded (cached by chunk content); enabling re-indexes the store once |
| `contextual_provider` | `Literal` | `ollama` | no | Summarizer backend (used only while contextual retrieval is enabled) |
| `contextual_model` | `str \| None` | *(none)* | no | Summary model override; falls back to `ollama_llm_model` (ollama) or `openai_model` (openai) |
| `contextual_source_max_chars` | `int` | `3000` | no | Per-chunk character cap in the summarize prompt |
| `contextual_timeout` | `float` | `60` | no | Seconds before a summarize call is abandoned (the file is then retried on the next index run) |
| `cleaning_enabled` | `bool` | `true` | ✓ | Master switch for `ingestion/cleaning.py`; when `false` both stages pass documents through unchanged |
| `chunk_size_text_words` | `int` | `150` | no | Chunk size for plain-text documents, in words |
| `chunk_overlap_text_words` | `int` | `40` | no | Chunk overlap for plain text, in words |
| `chunk_size_markdown_words` | `int` | `300` | no | Chunk size for Markdown sections, in words |
| `chunk_overlap_markdown_words` | `int` | `50` | no | Chunk overlap within oversized Markdown sections, in words |
| `chunk_size_pdf_words` | `int` | `150` | no | Chunk size per PDF page, in words |
| `chunk_overlap_pdf_words` | `int` | `40` | no | Chunk overlap per PDF page, in words |
| `chunk_size_code_words` | `int` | `400` | no | Chunk size for source code, in words |
| `chunk_overlap_code_words` | `int` | `100` | no | Chunk overlap for source code, in words |
| `chunk_size_json_words` | `int` | `400` | no | Size cap for a JSON top-level-key chunk, in words |
| `verbose` | `bool` | `true` | ✓ | Debug output everywhere; the CLI `--verbose`/`-v` flag overrides per run |

Chunk sizes are genuinely word-based (`length_function=len(text.split())`).

## Precedence & access

Environment variables > `.env` at project root (UTF-8) > field defaults. Unknown keys are ignored (`extra="ignore"`); key matching is case-insensitive.

```python
from .config import get_settings
_settings = get_settings()   # cached, parsed once per process
```

Never instantiate `Settings()` in feature code and never add `os.getenv` calls.

## Adding a new setting

Add a typed `snake_case` field with a default to `Settings`, in the block matching its subsystem and ordered everyday-knobs-first (model-facing fields get `Field(description=...)`), read it via the cached settings object, and update the table above — plus, if it is an everyday setting, the matching block in `.env` (uppercase key) — in the same change.

## Notes

- Model names must match what Ollama has pulled (`ollama list`); the CLI checks connectivity, not models.
- **Backend switching is one `.env` line per path:** `LLM_PROVIDER` (answers) and `RERANK_PROVIDER` (reranker) flip between `ollama` and `openai`; each provider keeps its own model setting, so a flip needs no model-name edits. Per-run override: `$env:LLM_PROVIDER='openai'; uv run app query …`. Embeddings are always Ollama.
- Answer/rerank gateway settings are query-side only — flipping those providers never triggers a re-index, and while contextual retrieval is off, changing `OLLAMA_LLM_MODEL` invalidates nothing stored. With contextual retrieval **enabled** that stops holding for the summarizer: its provider and *resolved* model are fingerprinted (see the contextual notes below). `OPENAI_API_KEY` lives in the git-ignored `.env`; the empty placeholder means "unset" and makes openai-provider paths fail loudly (rerank: warning + retrieval order kept).
- Changing `OLLAMA_EMBEDDING_MODEL` invalidates stored vectors — no manual action needed: the model name is in the index fingerprint, so the next `app index` rebuilds automatically. (The fingerprint hashes values, not field names — renaming a setting never rebuilds the index.)
- Enabling contextual retrieval re-indexes the whole store once: the contextual block (provider + resolved summary model + source cap) joins the fingerprint **only while enabled** — disabled leaves every existing fingerprint byte-identical, while enabled flips of `CONTEXTUAL_PROVIDER` / `CONTEXTUAL_MODEL` (or of `OLLAMA_LLM_MODEL` with no override) each trigger one rebuild.
- The chunk-summary cache `data/chunk_summaries.json` (git-ignored) survives `index --clear` — entries are keyed by chunk content, so a rebuild reuses them — and self-invalidates wholesale on a summarizer provider/model, summary-prompt, or `CONTEXTUAL_SOURCE_MAX_CHARS` change (key mismatch ⇒ silent re-summarize; a stale cache costs, it never misleads).
- Retrieval knobs are deliberately excluded from the index fingerprint (it hashes only pipeline version, embedding model, `chunk_*` and cleaning settings, plus contextual settings while enabled) — flipping modes/rerank takes effect immediately with no re-index; conversely, never prefix a new *indexing* setting with `chunk_` by accident. The `contextual_*` settings are the deliberate counterpart: the fingerprint collects them explicitly and only while enabled, so a query-side `contextual_*` read can never trigger a needless rebuild.

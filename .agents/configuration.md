# Configuration

All runtime settings live in `src/langchain_rag/config.py` as one pydantic-settings model. Nothing reads the environment outside this module (rule 4 in [`../AGENTS.md`](../AGENTS.md)).

## Settings reference

| Setting | Type | Default | Purpose |
|---|---|---|---|
| `OLLAMA_LLM_MODEL` | `str` | `llama3.2` | Chat model used for generation (must be pulled in Ollama) |
| `OLLAMA_EMBEDDING_MODEL` | `str` | `embeddinggemma` | Embedding model for indexing and query embedding |
| `verbose` | `bool` | `true` | Debug output everywhere; the CLI `--verbose`/`-v` flag overrides per run |
| `chunk_size_text_words` | `int` | `300` | Chunk size for plain-text documents, in words |
| `chunk_overlap_text_words` | `int` | `50` | Chunk overlap for plain text, in words |
| `chunk_size_markdown_words` | `int` | `300` | Chunk size for Markdown sections, in words |
| `chunk_overlap_markdown_words` | `int` | `50` | Chunk overlap within oversized Markdown sections, in words |
| `chunk_size_pdf_words` | `int` | `300` | Chunk size per PDF page, in words |
| `chunk_overlap_pdf_words` | `int` | `50` | Chunk overlap per PDF page, in words |
| `chunk_size_code_words` | `int` | `400` | Chunk size for source code, in words |
| `chunk_overlap_code_words` | `int` | `100` | Chunk overlap for source code, in words |
| `chunk_size_json_words` | `int` | `400` | Size cap for a JSON top-level-key chunk, in words |
| `top_k` | `int` | `5` | Number of chunks retrieved per query |
| `max_context_chars` | `int` | `12000` | Hard cap on the assembled context string before prompting |
| `retrieval_mode` | `Literal` | `vector` | Candidate source: `vector`, `keyword` (BM25) or `hybrid` (RRF-fused) |
| `rrf_c` | `int` | `60` | RRF damping constant: `score += weight / (rrf_c + rank)` |
| `hybrid_vector_weight` | `float` | `0.5` | Influence of the vector ranking in hybrid fusion |
| `hybrid_bm25_weight` | `float` | `0.5` | Influence of the BM25 ranking in hybrid fusion |
| `rerank_enabled` | `bool` | `false` | Reorder over-fetched candidates with one listwise LLM call; fail-open on trouble |
| `rerank_candidate_multiplier` | `int` | `3` | Candidates fetched (`top_k` × this) before reranking |
| `rerank_model` | `str \| None` | *(none)* | Rerank model override; falls back to `OLLAMA_LLM_MODEL` (ollama) or `openai_model` (openai) |
| `rerank_doc_max_chars` | `int` | `2000` | Per-candidate character cap in the rerank prompt |
| `rerank_num_ctx` | `int` | `16384` | Context window for the listwise rerank prompt (Ollama only; gateways manage context server-side) |
| `rerank_timeout` | `float` | `120` | Seconds before the rerank call is abandoned (fail-open) |
| `llm_provider` | `Literal` | `ollama` | Answer-LLM backend: `ollama` or `openai` (any OpenAI-compatible gateway) |
| `llm_model` | `str \| None` | *(none)* | Answer model override; falls back to `OLLAMA_LLM_MODEL` / `openai_model` per provider |
| `rerank_provider` | `Literal` | `ollama` | Reranker backend; the local `.env` flips it to the gateway (small local models return empty score lists on big listwise prompts) |
| `openai_model` | `str` | *(empty)* | Model name on the gateway, used by every path whose provider is `openai` |
| `openai_base_url` | `str` | *(empty)* | Gateway API root URL (`…/v1`); required when a path selects `openai` |
| `openai_api_key` | `SecretStr \| None` | *(none)* | Gateway key; an empty `OPENAI_API_KEY=` counts as unset; never logged |
| `cleaning_enabled` | `bool` | `true` | Master switch for `ingestion/cleaning.py`; when `false` both stages pass documents through unchanged |

Chunk sizes are genuinely word-based (`length_function=len(text.split())`).

## Precedence & access

Environment variables > `.env` at project root (UTF-8) > field defaults. Unknown keys are ignored (`extra="ignore"`).

```python
from .config import get_settings
_settings = get_settings()   # cached, parsed once per process
```

Never instantiate `Settings()` in feature code and never add `os.getenv` calls.

## Adding a new setting

Add a typed field with a default to `Settings` (model-facing fields get `Field(description=...)`), read it via the cached settings object, and update the table above in the same change.

## Notes

- Model names must match what Ollama has pulled (`ollama list`); the CLI checks connectivity, not models.
- **Backend switching is one `.env` line per path:** `LLM_PROVIDER` (answers) and `RERANK_PROVIDER` (reranker) flip between `ollama` and `openai`; each provider keeps its own model setting, so a flip needs no model-name edits. Per-run override: `$env:LLM_PROVIDER='openai'; uv run app query …`. Embeddings are always Ollama.
- Gateway settings are query-side only — flipping providers never triggers a re-index. `OPENAI_API_KEY` lives in the git-ignored `.env`; the empty placeholder means "unset" and makes openai-provider paths fail loudly (rerank: warning + retrieval order kept).
- Changing `OLLAMA_EMBEDDING_MODEL` invalidates stored vectors — no manual action needed: the model name is in the index fingerprint, so the next `app index` rebuilds automatically.
- Retrieval knobs are deliberately excluded from the index fingerprint (it hashes only pipeline version, embedding model, `chunk_*` and cleaning settings) — flipping modes/rerank takes effect immediately with no re-index; conversely, never prefix a new *indexing* setting with `chunk_` by accident.

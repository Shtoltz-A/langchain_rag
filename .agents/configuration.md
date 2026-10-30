# Configuration

All runtime settings live in `src/langchain_rag/config.py` as one pydantic-settings model. Nothing reads the environment outside this module (rule 4 in [`../AGENTS.md`](../AGENTS.md)).

## Settings reference

| Setting | Type | Default | Purpose |
|---|---|---|---|
| `OLLAMA_LLM_MODEL` | `str` | `llama3.2` | Chat model used for generation (must be pulled in Ollama) |
| `OLLAMA_EMBEDDING_MODEL` | `str` | `embeddinggemma` | Embedding model for indexing and query embedding |
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
| `cleaning_enabled` | `bool` | `true` | Master switch for `ingestion/cleaning.py`; when `false` both stages pass documents through unchanged |

Chunk sizes are genuinely word-based (`length_function=len(text.split())`). Pre-2.3 keys (`chunk_size_words`, `chunk_overlap_words`) no longer exist and are silently ignored if left in `.env` — replace them with the per-type fields above.

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
- Changing `OLLAMA_EMBEDDING_MODEL` invalidates stored vectors — no manual action needed: the model name is in the index fingerprint, so the next `app index` rebuilds automatically.

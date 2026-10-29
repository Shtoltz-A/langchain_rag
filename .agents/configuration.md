# Configuration

All runtime settings live in `src/langchain_rag/config.py` as a single pydantic-settings model. See rule 4 in [`../AGENTS.md`](../AGENTS.md): nothing reads the environment outside this module.

## Settings reference

| Setting | Type | Default | Purpose |
|---|---|---|---|
| `OLLAMA_LLM_MODEL` | `str` | `llama3.2` | Chat model used for generation (must be pulled in Ollama) |
| `OLLAMA_EMBEDDING_MODEL` | `str` | `embeddinggemma` | Embedding model used for indexing and query embedding |
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
| `cleaning_enabled` | `bool` | `true` | Master switch for `ingestion/cleaning.py` (normalize + context prefixes); when `false` both stages pass documents through unchanged |

## Precedence

1. Environment variables (highest).
2. `.env` file at the project root (`PROJECT_ROOT / ".env"`, UTF-8).
3. Field defaults in `Settings`.

Unknown keys in the environment or `.env` are ignored (`extra="ignore"`), so `.env` may hold unrelated variables.

## Access pattern

```python
from .config import get_settings

_settings = get_settings()   # cached with lru_cache(maxsize=1)
```

`get_settings()` reads and parses `.env` once per process. Do not instantiate `Settings()` directly in feature code, and do not add module-level `os.getenv` calls.

## Adding a new setting

1. Add a typed field with a sensible default to `Settings` in `config.py`; give model-facing fields a `Field(description=...)` and group related fields under a comment (`# Chunking settings`, `# Retrieval settings`, ...).
2. Read it where needed via the cached settings object — never re-read the file.
3. Update the table above in the same change.

## Notes

- Model names must match what Ollama actually has pulled (`ollama list`). The CLI checks connectivity but not model availability.
- Changing `OLLAMA_EMBEDDING_MODEL` invalidates the existing index: stored vectors were produced by the old model. Always re-run `app index` after changing it.
- Chunk settings are genuinely **word-based** since Step 2.3 (`length_function=len(text.split())`). Before 2.3 the `chunk_size_words=500` setting was misnamed — `length_function` defaulted to `len` (characters), so effective chunks were ~500 chars. The old `chunk_size_words`/`chunk_overlap_words` keys no longer exist; if they linger in a local `.env` they are silently ignored (`extra="ignore"`) — replace them with the per-type fields above.

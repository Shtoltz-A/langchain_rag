# Configuration

All runtime settings live in `src/langchain_rag/config.py` as a single pydantic-settings model. See rule 4 in [`../AGENTS.md`](../AGENTS.md): nothing reads the environment outside this module.

## Settings reference

| Setting | Type | Default | Purpose |
|---|---|---|---|
| `OLLAMA_LLM_MODEL` | `str` | `llama3.2` | Chat model used for generation (must be pulled in Ollama) |
| `OLLAMA_EMBEDDING_MODEL` | `str` | `embeddinggemma` | Embedding model used for indexing and query embedding |
| `chunk_size_words` | `int` | `500` | Chunk size for the text splitter, in words |
| `chunk_overlap_words` | `int` | `100` | Overlap between consecutive chunks, in words |
| `top_k` | `int` | `5` | Number of chunks retrieved per query |
| `max_context_chars` | `int` | `12000` | Hard cap on the assembled context string before prompting |

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
- Chunking settings are word-based (`length_function` defaults to `len`, so values approximate tokens). Step 2 will move chunking into `ingestion/chunking.py`; settings will be extended there rather than replaced silently.

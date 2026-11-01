# langchain_rag

An all-purpose RAG (Retrieval-Augmented Generation) toolkit: ingest documents
of many formats, keep an incremental Chroma index across multiple folders, and
answer questions with grounded, cited answers — all from one CLI.

## Features

- **Multi-format ingestion** — text, Markdown, PDF, code, JSON and more, with
  per-file-type chunking strategies, optional text cleaning and context
  prefixes on every chunk.
- **Incremental multi-root indexing** — a content-hash manifest
  (`data/index_manifest.json`) lets you index several folders; repeat runs only
  re-embed new/changed files and drop deleted ones from the store.
- **Hybrid retrieval** — `vector`, `keyword` (BM25) or `hybrid` mode with RRF
  fusion, plus opt-in listwise LLM reranking of over-fetched candidates.
- **Opt-in contextual retrieval** — a one-sentence LLM summary per chunk is
  prepended before embedding (cached by chunk content hash).
- **Switchable chat backends** — every LLM path (answers, rerank, summaries)
  runs on local Ollama or any OpenAI-compatible gateway, chosen per path in
  `.env`; embeddings always run on Ollama.
- **Verbose tracing** — `--verbose` adds one trace line per pipeline stage and
  file, plus retrieval and context diagnostics.

## Prerequisites

- Python 3.12 and [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com) running locally, with the chat and embedding
  models pulled (defaults: `llama3.2` and `embeddinggemma` — check
  `ollama list`; the CLI checks connectivity, not model names)
- Optional: an OpenAI-compatible gateway if you point the answer/rerank/summary
  paths at it (`LLM_PROVIDER=openai` etc.)

## Quickstart

```powershell
# 1. Install dependencies
uv sync

# 2. Configure: create .env at the project root (it is git-ignored).
#    Every key and its default is documented in .agents/configuration.md.

# 3. Index documents (default folder: data/raw; use -d for any other folder,
#    and repeat per folder — indexing is incremental and multi-root)
uv run app index
uv run app index --directory C:\path\to\docs

# 4. Ask questions against the index
uv run app query "How does the chunking strategy work?"

# 5. Start over: delete the vector store and manifest (the chunk-summary
#    cache is kept)
uv run app index --clear
```

Add `--verbose` (or `-v`) before the subcommand for stage-by-stage tracing:

```powershell
uv run app --verbose query "Why is my context truncated?"
```

## Documentation

This README covers end-user usage only. Agent-facing and architecture
documentation lives in [`AGENTS.md`](AGENTS.md) and the
[`.agents/`](.agents/) directory (architecture, conventions, configuration,
commands, roadmap and design decisions).

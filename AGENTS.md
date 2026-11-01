# langchain_rag

A powerful, all-purpose RAG (Retrieval-Augmented Generation) application built with LangChain.

## Current state

Implemented and covered by offline tests: multi-format ingestion (loaders, per-type chunking, cleaning, context prefixes) · manifest-driven incremental multi-root indexing in Chroma · opt-in contextual retrieval (per-chunk LLM summary prepended before embedding, content-hash cached) · vector/keyword/hybrid retrieval (RRF) with opt-in listwise LLM reranking · Ollama ⇄ OpenAI-compatible gateway chat backends switchable per path via `generation/` · `--verbose` pipeline tracing.

**Next:** no numbered step committed — the evaluation harness (backlog) is the natural candidate, since it unblocks measuring every other quality change. See [`.agents/roadmap.md`](.agents/roadmap.md) for open threads and the improvement backlog.

## Tech stack

| Concern | Choice |
|---|---|
| Language / runtime | Python 3.12, managed with `uv` |
| LLM & embeddings | Embeddings: local Ollama (`langchain-ollama`); chat: Ollama **or** OpenAI-compatible gateway (`langchain-openai`), provider per path via `generation/` |
| Vector store | Chroma, persisted at `data/chroma/` (`langchain-chroma`), behind `storage/vectorstore.py` |
| Incremental indexing | Multi-root content-hash manifest at `data/index_manifest.json` (`storage/manifest.py`) |
| Contextual summaries | Opt-in per-chunk LLM context sentence prepended before embedding (`ingestion/contextual.py`), content-hash cache at `data/chunk_summaries.json` (`storage/summary_cache.py`) |
| Splitting | `ingestion/chunking.py` (per-type strategies); `langchain-text-splitters` used as building blocks |
| Retrieval | `retrieval/` modes `vector`/`keyword`/`hybrid` (RRF fusion, `rank-bm25`) + opt-in LLM reranking, behind `retrieval.retrieve()` |
| CLI | Typer (entry point: `app`) |
| Settings | pydantic-settings + `.env` |

## Docs map

All detailed project documentation lives in [`.agents/`](.agents/). Start here, then follow the link relevant to your task:

| Document | What it covers |
|---|---|
| [`.agents/architecture.md`](.agents/architecture.md) | Module responsibilities, data flow, directory layout |
| [`.agents/conventions.md`](.agents/conventions.md) | Code style and structural rules for this codebase |
| [`.agents/configuration.md`](.agents/configuration.md) | All settings, `.env` handling, how to add a setting |
| [`.agents/commands.md`](.agents/commands.md) | Dev workflow: setup, indexing, querying, prerequisites, troubleshooting |
| [`.agents/roadmap.md`](.agents/roadmap.md) | What's next + improvement backlog (forward-looking only, no history) |
| [`.agents/rag-decisions.md`](.agents/rag-decisions.md) | Design decisions with rationale and Revisit (= improvement) notes |

## Always-on rules

1. **Read the roadmap first.** Before starting any work, check [`.agents/roadmap.md`](.agents/roadmap.md) to know the current step and its scope. Stay within the current step unless told otherwise.
2. **Keep docs in sync.** Any merged change that affects architecture, settings, commands, or conventions must update the relevant `.agents/*.md` file in the same change.
3. **Roadmap stays forward-looking.** When a step lands, remove it from `roadmap.md` (promote unfinished ideas to backlog); its durable result belongs in `architecture.md`/`rag-decisions.md` and the code — the roadmap never accumulates history.
4. **Settings only via `get_settings()`.** No new environment reads or hardcoded tunables scattered through the code. See [configuration.md](.agents/configuration.md).
5. **Terminal output only via `Console`.** No raw `print()` in application code.
6. **Core logic stays CLI-agnostic.** `services/` (and `ingestion/`) must not import Typer or exit the process; commands belong in `cli.py`, and `app.py` stays a re-export facade.
7. **Record real decisions.** Non-obvious RAG design choices (chunking, retrieval, prompting) get an entry in [`.agents/rag-decisions.md`](.agents/rag-decisions.md) with rationale and revisit conditions.

# langchain_rag

A powerful, all-purpose RAG (Retrieval-Augmented Generation) application built with LangChain — developed step by step, one solid stage at a time.

## Current stage

- **Step 1 — Baseline** ✅ done: configuration, console logging, Ollama embeddings, LLM generation, naive index + query CLI.
- **Step 2 — Multi-Format Ingestion** ✅ done: loaders, smart chunking, cleaning + context prefixes (2.1–2.4). Contextual retrieval renumbered to Step 4.
- **Step 3 — Storage Layer & Incremental Indexing** ✅ done: `storage/` package, manifest-driven incremental indexing (only changed files re-embedded), multi-root indexing (3.1).
- **Step 4 — Contextual Retrieval** ⬜ next: per-chunk LLM summaries before embedding.

See [`.agents/roadmap.md`](.agents/roadmap.md) for the full staged plan.

## Tech stack

| Concern | Choice |
|---|---|
| Language / runtime | Python 3.12, managed with `uv` |
| LLM & embeddings | Ollama (`langchain-ollama`) — local models |
| Vector store | Chroma, persisted at `data/chroma/` (`langchain-chroma`), behind `storage/vectorstore.py` |
| Incremental indexing | Multi-root content-hash manifest at `data/index_manifest.json` (`storage/manifest.py`, since Step 3/3.1) |
| Splitting | `ingestion/chunking.py` (per-type strategies, since Step 2.3); `langchain-text-splitters` used as building blocks |
| CLI | Typer (entry point: `app`) |
| Settings | pydantic-settings + `.env` |

## Docs map

All detailed project documentation lives in [`.agents/`](.agents/). Start here, then follow the link relevant to your task:

| Document | What it covers |
|---|---|
| [`.agents/architecture.md`](.agents/architecture.md) | Module responsibilities, data flow, directory layout |
| [`.agents/conventions.md`](.agents/conventions.md) | Code style and structural rules for this codebase |
| [`.agents/configuration.md`](.agents/configuration.md) | All settings, `.env` handling, how to add a setting |
| [`.agents/commands.md`](.agents/commands.md) | Dev workflow: setup, indexing, querying, prerequisites |
| [`.agents/roadmap.md`](.agents/roadmap.md) | Staged plan: what is done, what is next, done-criteria |
| [`.agents/rag-decisions.md`](.agents/rag-decisions.md) | RAG design decisions, trade-offs, where to revisit them |

## Always-on rules

1. **Read the roadmap first.** Before starting any work, check [`.agents/roadmap.md`](.agents/roadmap.md) to know the current stage and its scope. Stay within the current stage unless told otherwise.
2. **Keep docs in sync.** Any merged change that affects architecture, settings, commands, or conventions must update the relevant `.agents/*.md` file in the same change.
3. **Mark progress.** When a stage or sub-step is completed, update `roadmap.md` (status markers and done-notes) — the roadmap is the single source of truth for "where are we".
4. **Settings only via `get_settings()`.** No new environment reads or hardcoded tunables scattered through the code. See [configuration.md](.agents/configuration.md).
5. **Terminal output only via `Console`.** No raw `print()` in application code.
6. **Core logic stays CLI-agnostic.** `app.py` (and later `ingestion/`) must not import Typer or exit the process; commands belong in `cli.py`.
7. **Record real decisions.** Non-obvious RAG design choices (chunking, retrieval, prompting) get an entry in [`.agents/rag-decisions.md`](.agents/rag-decisions.md) with rationale and revisit conditions.

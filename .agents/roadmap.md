# Roadmap

Forward-looking content only: the next step, open threads, and the backlog. Finished work leaves this file when it lands — its durable result lives in [architecture.md](architecture.md), [rag-decisions.md](rag-decisions.md), and the code itself. Per-decision improvement threads stay as **Revisit** notes in [rag-decisions.md](rag-decisions.md).

## Next — Step 5: Contextual Retrieval ⬜

**Goal:** better retrieval via per-chunk LLM context.

- Generate a 1-sentence summary per chunk before embedding (e.g. *"This chunk implements JWT refresh token in auth.py"*), stored with the chunk and/or prepended to its embedded text.
- Cost: one LLM call per chunk per index run — gate behind a setting; cache by chunk hash (the manifest's hashing/diff is directly reusable).

**Done when:** summaries generated and stored, retrieval quality measured on a small golden set against the current baseline, and cost acceptable or mitigated by caching.

## Open threads

- Hybrid retrieval and reranking are checked only ad hoc (before/after queries with `RETRIEVAL_MODE=hybrid` / `RERANK_ENABLED=true`); a formal before/after comparison waits for the evaluation harness below.

## Backlog 💡 (unordered)

Promote into a numbered step (with Goal/Scope/Done-when) when committing.

- 💡 Evaluation harness (golden QA set, retrieval/answer quality metrics) — unblocks measuring every other quality change
- 💡 Metadata filtering / scoped queries (per-collection or per-directory separation)
- 💡 Citations in answers (source references back to files/pages)
- 💡 Conversational/multi-turn querying
- 💡 API/server interface (beyond the CLI)
- 💡 `app status` command: manifest summary (files indexed, last index time, drift)

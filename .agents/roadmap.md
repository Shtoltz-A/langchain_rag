# Roadmap

Forward-looking content only: what comes next, open threads, and the backlog. Finished work leaves this file when it lands — its durable result lives in [architecture.md](architecture.md), [rag-decisions.md](rag-decisions.md), and the code itself. Per-decision improvement threads stay as **Revisit** notes in [rag-decisions.md](rag-decisions.md).

## Next

No step is committed. Natural candidate: the evaluation harness (backlog) — it unblocks measuring every other quality change. Committing it means giving it Goal/Scope/Done-when here.

## Open threads

- Hybrid retrieval and reranking are checked only ad hoc (before/after queries with `RETRIEVAL_MODE=hybrid` / `RERANK_ENABLED=true`); a formal before/after comparison waits for the evaluation harness below.
- Contextual retrieval (`CONTEXTUAL_RETRIEVAL_ENABLED=true`) is likewise measured only by manual before/after queries against the summary-less baseline so far; the golden-set comparison also awaits the evaluation harness.

## Backlog 💡 (unordered)

Promote into a numbered step (with Goal/Scope/Done-when) when committing.

- 💡 Evaluation harness (golden QA set, retrieval/answer quality metrics) — unblocks measuring every other quality change
- 💡 Metadata filtering / scoped queries (per-collection or per-directory separation)
- 💡 Citations in answers (source references back to files/pages)
- 💡 Conversational/multi-turn querying
- 💡 API/server interface (beyond the CLI)
- 💡 `app status` command: manifest summary (files indexed, last index time, drift)

"""``retrieve()``: mode routing → candidate generation → RRF fusion → rerank.

The query pipeline behind ``services/answering.answer()``; mode and reranker
are pure
settings decisions, so nothing here knows about the CLI. In verbose mode every
stage reports itself once as a styled ``Console.trace`` line (mode, counts and
duration in one line — no per-stage debug chatter), with ranked results as
numbered ``label :: preview`` tables.
"""

from __future__ import annotations

import time

from ..config import get_settings
from ..console import get_console
from ..generation import describe_model
from ..storage import VectorStore
from .display import doc_label, doc_preview
from .reranking import rerank
from .retrievers import BM25Index, Ranking, rrf_fuse, vector_search

c = get_console()


def _results_block(title: str, docs: Ranking) -> None:
    """Numbered ``#rank label :: preview`` table, built only when it prints."""
    if not c.verbose:
        return
    rows = [
        f"#{rank:<2} {doc_label(doc)} :: {doc_preview(doc)}"
        for rank, doc in enumerate(docs, start=1)
    ]
    c.debug_block(title, "\n".join(rows) or "<no results>")


def retrieve(question: str) -> Ranking:
    """Return the chunks to feed the answer prompt, per current settings.

    Pipeline: empty-store guard → ``top_k`` (× ``rerank_candidate_multiplier``
    when reranking) candidates from the configured ``retrieval_mode`` —
    keyword/hybrid build one ``BM25Index`` from a single corpus load, hybrid
    fuses its ranking with the vector one via RRF — → optional LLM rerank →
    at most ``top_k`` chunks.

    Returns ``[]`` on an empty store so the caller keeps its "index first"
    message.

    Raises:
        ValueError: ``retrieval_mode`` is not one of the three known modes —
            a misconfiguration must fail loudly, not silently reroute (the
            ``Settings`` ``Literal`` normally rejects this already; this guard
            covers injected or unvalidated settings objects).
    """
    settings = get_settings()
    store = VectorStore()
    chunks_in_store = store.count()
    if chunks_in_store == 0:
        c.debug("Vector store is empty — nothing to retrieve")
        return []

    n_candidates = settings.top_k * (
        settings.rerank_candidate_multiplier if settings.rerank_enabled else 1
    )
    c.trace("query", repr(question), f"{chunks_in_store} chunks indexed")
    mode = settings.retrieval_mode

    if mode == "vector":
        start = time.perf_counter()
        candidates = vector_search(store, question, n_candidates)
        c.trace(
            "vector",
            f"{len(candidates)}/{n_candidates} candidates",
            f"{time.perf_counter() - start:.1f}s (incl. query embedding)",
        )
        _results_block("vector results", candidates)
    elif mode == "keyword":
        # Lazy: the keyword corpus is loaded only by the modes that need it.
        corpus = store.all_documents()
        start = time.perf_counter()
        keyword = BM25Index(corpus)
        candidates = keyword.search(question, n_candidates)
        c.trace(
            "keyword",
            f"{len(candidates)}/{n_candidates} candidates",
            f"BM25 over {len(corpus)} chunks",
            f"{time.perf_counter() - start:.1f}s",
        )
        _results_block("keyword results", candidates)
    elif mode == "hybrid":
        corpus = store.all_documents()
        start = time.perf_counter()
        vector_ranking = vector_search(store, question, n_candidates)
        vector_secs = time.perf_counter() - start
        start = time.perf_counter()
        keyword = BM25Index(corpus)
        keyword_ranking = keyword.search(question, n_candidates)
        keyword_secs = time.perf_counter() - start
        # Pre-fusion lists stay visible: post-fusion alone, nothing explains
        # which retriever buried a chunk that RRF then ranked low.
        c.trace(
            "vector",
            f"{len(vector_ranking)}/{n_candidates} candidates",
            f"{vector_secs:.1f}s (incl. query embedding)",
        )
        _results_block("vector results", vector_ranking)
        c.trace(
            "keyword",
            f"{len(keyword_ranking)}/{n_candidates} candidates",
            f"BM25 over {len(corpus)} chunks",
            f"{keyword_secs:.1f}s",
        )
        _results_block("keyword results", keyword_ranking)
        candidates = rrf_fuse(
            rankings=[vector_ranking, keyword_ranking],
            weights=[settings.hybrid_vector_weight, settings.hybrid_bm25_weight],
            c=settings.rrf_c,
            top_n=n_candidates,
        )
        c.trace(
            "rrf",
            f"{len(vector_ranking)} + {len(keyword_ranking)} → {len(candidates)} "
            "candidates",
            f"weights {settings.hybrid_vector_weight}/{settings.hybrid_bm25_weight}",
            f"c={settings.rrf_c}",
        )
        _results_block("fused candidates", candidates)
    else:
        raise ValueError(
            f"Unknown retrieval_mode {mode!r}: expected vector, keyword or hybrid"
        )

    if settings.rerank_enabled:
        final = rerank(question, candidates, k=settings.top_k)
        backend = (
            f"rerank on({settings.rerank_provider}/"
            f"{describe_model(settings.rerank_provider, settings.rerank_model, settings)})"
        )
    else:
        final = candidates[: settings.top_k]
        backend = "rerank off"
    c.trace("kept", f"{len(final)}/{len(candidates)} chunks", backend, ok=True)
    return final

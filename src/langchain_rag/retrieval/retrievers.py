"""Candidate generation: dense vector search and BM25 keyword retrieval.

The mode is a settings decision (``retrieval_mode``); this module exposes the
plain-callable candidate generators and the in-house reciprocal rank fusion —
rank-based merging is ~20 lines of pure Python and keeps
``langchain-community``'s dependency tree out of the project (rag-decisions.md).
"""

from __future__ import annotations

import heapq
import re
import unicodedata
from collections import defaultdict
from hashlib import sha1

import numpy as np
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from ..storage import VectorStore

Ranking = list[Document]

_TOKEN_PATTERN = re.compile(r"\w+")

# Metadata fields that locate a chunk inside its source. They join the fusion
# key so identical boilerplate at two positions stays distinct; fields the
# pipeline does not emit yet simply contribute nothing (rag-decisions.md).
_POSITION_FIELDS = ("chunk_index", "offset", "page")


def _tokenize(text: str) -> list[str]:
    """Lowercased word tokens with accents folded (été → ete), NFKD-based.

    Deliberately naive: no stemming, stopwords or elision handling — those
    need per-language decisions (rag-decisions.md). Identifiers like ERR_5021
    survive intact.
    """
    folded = "".join(
        ch
        for ch in unicodedata.normalize("NFKD", text.lower())
        if not unicodedata.combining(ch)
    )
    return _TOKEN_PATTERN.findall(folded)


def doc_key(doc: Document) -> str:
    """Stable identity for fusion dedupe: explicit id, else source + position + content hash.

    The same chunk surfaces from the vector and keyword retrievers as
    different instances whose key must still match — that holds because both
    ``VectorStore`` read paths return the identical stored metadata and neither
    injects an id (pinned in tests/storage/test_vectorstore.py); a path that
    started adding metadata the other lacks would silently break the dedupe.
    """
    explicit = doc.metadata.get("id")
    if explicit is not None:
        return f"id:{explicit}"
    digest = sha1(usedforsecurity=False)
    digest.update(str(doc.metadata.get("source", "")).encode())
    for field in _POSITION_FIELDS:
        value = doc.metadata.get(field)
        if value is not None:
            digest.update(f"\0{field}={value}".encode())
    digest.update(b"\0")
    digest.update(doc.page_content.encode())
    return f"sha1:{digest.hexdigest()}"


def vector_search(store: VectorStore, query: str, k: int) -> Ranking:
    """Dense top-k search through a ``VectorStore`` facade (the one Chroma seam)."""
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    return store.search(query, k=k)


class BM25Index:
    """BM25 keyword search over an in-memory chunk corpus.

    Built explicitly by the pipeline once per query run — no module-level
    cache: the one-shot CLI process makes the object fresh by construction,
    and sole ownership removes the shared-mutable-state questions. A
    chat/server mode would instead reuse one index keyed on store changes
    (revisit condition, rag-decisions.md).
    """

    def __init__(self, corpus: list[Document]) -> None:
        self._corpus = corpus
        self._bm25 = (
            BM25Okapi([_tokenize(doc.page_content) for doc in corpus])
            if corpus
            else None
        )

    def search(self, query: str, k: int) -> Ranking:
        """Top-k chunks scoring above zero, best first; ties keep corpus order.

        Zero scorers share no term with the query and would only inject noise
        into fusion. (Raw BM25 IDF goes negative for corpus-wide terms, but
        rank_bm25 floors it at epsilon × average IDF, so surviving negatives
        are mostly a very-small-corpus artefact.)
        """
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        positive = np.flatnonzero(scores > 0)  # vectorised beats a Python scan
        # nlargest keeps selection O(n log k) instead of a full O(n log n)
        # sort; ties resolve like a stable descending sort (corpus order).
        top = heapq.nlargest(k, positive, key=scores.__getitem__)
        return [self._corpus[i] for i in top]


def rrf_fuse(
    rankings: list[Ranking],
    weights: list[float],
    *,
    c: int,
    top_n: int | None = None,
) -> Ranking:
    """Fuse ranked lists into one order by reciprocal rank fusion.

    ``score(doc) = Σ weight_i / (c + rank_i)`` over every list containing the
    document. Only ranks enter the score, which is what makes the fusion safe
    across incomparable scales (BM25 is unbounded, cosine is 0–1).

    Args:
        rankings: one ranked document list per retriever, best first.
        weights: per-list influence, parallel to ``rankings``.
        c: damping constant; larger ``c`` flattens rank differences.
        top_n: optional cutoff for the fused list (fusing two k-lists can
            otherwise return up to 2k documents).

    Raises:
        ValueError: ``c < 1``, ``top_n < 0``, or ranking/weight length mismatch.

    Documents are deduplicated by ``doc_key`` and the first occurrence's
    instance wins; ties keep the earlier list's order.
    """
    if c < 1:
        raise ValueError(f"c must be >= 1, got {c}")
    if top_n is not None and top_n < 0:
        raise ValueError(f"top_n must be >= 0, got {top_n}")
    scores: dict[str, float] = defaultdict(float)
    seen: dict[str, Document] = {}
    for ranked, weight in zip(rankings, weights, strict=True):
        for rank, doc in enumerate(ranked):
            key = doc_key(doc)
            seen.setdefault(key, doc)
            scores[key] += weight / (c + rank + 1)
    order = sorted(scores, key=scores.__getitem__, reverse=True)
    if top_n is not None:
        order = order[:top_n]
    return [seen[key] for key in order]

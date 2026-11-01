"""Unit tests for retrieval candidate generation and RRF fusion (pure fakes)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.documents import Document

from langchain_rag.retrieval import pipeline, retrieve
from langchain_rag.retrieval.retrievers import (
    BM25Index,
    _tokenize,
    doc_key,
    rrf_fuse,
    vector_search,
)


def _doc(text: str, **metadata: Any) -> Document:
    return Document(page_content=text, metadata={"source": "x.txt", **metadata})


class _FakeVectorStore:
    """Stands in for VectorStore: canned corpus + vector hits, records calls."""

    def __init__(
        self,
        *,
        count: int,
        corpus: list[Document] | None = None,
        hits: list[Document] | None = None,
    ) -> None:
        self._count = count
        self._corpus = corpus or []
        self._hits = hits or []
        self.calls: list[tuple] = []

    def count(self) -> int:
        return self._count

    def all_documents(self) -> list[Document]:
        self.calls.append(("all_documents",))
        return list(self._corpus)

    def search(self, query: str, k: int) -> list[Document]:
        self.calls.append(("search", query, k))
        return list(self._hits[:k])


def _settings(
    *,
    mode: str = "vector",
    top_k: int = 2,
    rerank: bool = False,
    multiplier: int = 3,
    vector_weight: float = 0.5,
    bm25_weight: float = 0.5,
    rrf_c: int = 60,
) -> SimpleNamespace:
    """Settings surface retrieval.retrieve() actually touches."""
    return SimpleNamespace(
        retrieval_mode=mode,
        top_k=top_k,
        rerank_enabled=rerank,
        rerank_candidate_multiplier=multiplier,
        hybrid_vector_weight=vector_weight,
        hybrid_bm25_weight=bm25_weight,
        rrf_c=rrf_c,
        # Backend display for the debug line (generation.describe_model):
        ollama_llm_model="llama-test",
        rerank_provider="ollama",
        rerank_model=None,
        openai_model="",
    )


def _install(
    monkeypatch: pytest.MonkeyPatch,
    store: _FakeVectorStore,
    settings: SimpleNamespace,
) -> None:
    """Point every collaborator retrieve() reaches for at fakes.

    retrieve() lives in the pipeline module, so that module's globals — not
    the package's — are what its name lookups resolve against.
    """
    monkeypatch.setattr(pipeline, "VectorStore", lambda: store)
    monkeypatch.setattr(pipeline, "get_settings", lambda: settings)


class TestTokenize:
    def test_lowercases_splits_and_folds_accents(self) -> None:
        assert _tokenize("Failed! see ERR_5021") == ["failed", "see", "err_5021"]
        # NFKD + combining-mark strip: accented and bare spellings collide.
        assert _tokenize("Café l'été") == ["cafe", "l", "ete"]
        assert _tokenize("Ошибка ERR_5021") == ["ошибка", "err_5021"]


class TestDocKey:
    def test_identity_rules(self) -> None:
        # Explicit id wins over content: same id, same chunk — different text.
        one = Document(page_content="text a", metadata={"id": 7, "source": "a.txt"})
        other = Document(page_content="text b", metadata={"id": 7, "source": "b.txt"})
        assert doc_key(one) == doc_key(other)
        # Fallback hash merges the same chunk seen as different instances.
        assert doc_key(_doc("same text")) == doc_key(_doc("same text"))
        # Position fields keep repeated boilerplate distinct (known limit:
        # duplicates inside one source stay merged while no position metadata
        # distinguishes the two chunks).
        assert doc_key(_doc("boilerplate", page=1)) != doc_key(
            _doc("boilerplate", page=2)
        )
        assert doc_key(_doc("boilerplate", chunk_index=0)) != doc_key(
            _doc("boilerplate", chunk_index=1)
        )
        # Identical text from different sources stays apart.
        assert doc_key(_doc("same text", source="a.txt")) != doc_key(
            _doc("same text", source="b.txt")
        )


class TestRrfFuse:
    def test_orders_by_weighted_rank_dedupes_and_cuts(self) -> None:
        a, b, other = _doc("a"), _doc("b"), _doc("c")
        # b agrees on both lists (rank 1 + rank 0) and outranks single-list
        # tops. Hand check: b = 0.5/62 + 0.5/61 ≈ 0.016235 beats
        # a = 0.5/61 ≈ 0.008197; c only appears in list 2.
        fused = rrf_fuse([[a, b], [b, other]], [0.5, 0.5], c=60)
        assert fused[0] is b
        assert fused[1] is a
        assert [d.page_content for d in fused] == ["b", "a", "c"]

        # Chroma hands the same stored chunk to both retrievers as different
        # instances; same (source, position, content) means one document —
        # so the distinguishing field must not be a position field — and the
        # first-seen instance wins.
        first = _doc("same text", marker="first")
        second = _doc("same text", marker="second")
        fused = rrf_fuse([[first], [second]], [0.5, 0.5], c=60)
        assert len(fused) == 1
        assert fused[0].metadata["marker"] == "first"
        # Identical text from different sources stays distinct.
        fused = rrf_fuse(
            [[_doc("same text", source="a.txt")], [_doc("same text", source="b.txt")]],
            [0.5, 0.5],
            c=60,
        )
        assert len(fused) == 2

        # A zero weight excludes a list even when it says the opposite.
        a, b = _doc("a"), _doc("b")
        fused = rrf_fuse([[a, b], [b, a]], [1.0, 0.0], c=60)
        assert [d.page_content for d in fused] == ["a", "b"]
        # Ties keep the earlier list's order.
        fused = rrf_fuse([[a, b], [a, b]], [0.5, 0.5], c=60)
        assert [d.page_content for d in fused] == ["a", "b"]

        # Empty rankings fuse away; top_n cuts the fused list (two 2-lists
        # fuse to up to 2k documents without a cutoff).
        a = _doc("a")
        assert rrf_fuse([], [], c=60) == []
        assert rrf_fuse([[], [a]], [0.5, 0.5], c=60) == [a]
        docs = [_doc(t) for t in "abcd"]
        rankings, weights = [[docs[0], docs[1]], [docs[2], docs[3]]], [0.5, 0.5]
        assert len(rrf_fuse(rankings, weights, c=60)) == 4
        # Ranks: a, c at 0.5/61 each, b, d at 0.5/62; ties keep first-seen.
        fused = rrf_fuse(rankings, weights, c=60, top_n=3)
        assert [d.page_content for d in fused] == ["a", "c", "b"]


class TestBM25Index:
    def test_ranks_filters_and_respects_k_and_corpus_order(self) -> None:
        # Exact identifier beats a paraphrase; no term overlap = no evidence,
        # so even a generous k must not resurrect zero-score padding docs —
        # they would only add fusion noise.
        corpus = [
            _doc("an error occurred while connecting"),
            _doc("raises ERR_5021 when the token expires"),
            _doc("banana bread recipe"),
        ]
        exact = corpus[1]
        assert BM25Index(corpus).search("ERR_5021", k=2)[0] is exact
        assert BM25Index(corpus).search("ERR_5021", k=5) == [exact]
        # A query without any overlap returns nothing.
        corpus = [_doc("alpha one"), _doc("alpha two"), _doc("beta three")]
        assert BM25Index(corpus).search("zzz_missing_token", k=3) == []
        # k truncates. 3 of 7 docs match: rank_bm25 keeps IDF positive only
        # while the term stays rare, so the non-matching filler docs are
        # load-bearing here.
        corpus = [
            _doc("alpha one"),
            _doc("alpha two"),
            _doc("alpha three"),
            _doc("beta four"),
            _doc("gamma five"),
            _doc("delta six"),
            _doc("epsilon seven"),
        ]
        found = BM25Index(corpus).search("alpha", k=2)
        assert len(found) == 2
        assert all("alpha" in d.page_content for d in found)
        # Ties keep corpus order.
        corpus = [
            _doc("alpha x"),
            _doc("alpha y"),
            _doc("beta one"),
            _doc("beta two"),
            _doc("beta three"),
        ]
        assert BM25Index(corpus).search("alpha", k=3) == corpus[:2]
        # rank_bm25 itself dies on an empty corpus; the index must not.
        assert BM25Index([]).search("anything", k=3) == []


class TestRetrievePipeline:
    def test_routes_modes_and_fuses_hybrid(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Empty store short-circuits before any retriever is touched.
        store = _FakeVectorStore(count=0)
        _install(monkeypatch, store, _settings(mode="hybrid"))
        assert retrieve("anything") == []
        assert store.calls == []

        # vector: fetches n_candidates and cuts to top_k.
        hits = [_doc(f"h{i}") for i in range(5)]
        store = _FakeVectorStore(count=5, hits=hits)
        _install(monkeypatch, store, _settings(mode="vector", top_k=2))
        found = retrieve("q")
        assert [d.page_content for d in found] == ["h0", "h1"]
        assert store.calls == [("search", "q", 2)]

        # keyword: never touches vector search, and zero-score noise docs no
        # longer pad the result up to top_k. 3 docs: on a 2-doc corpus
        # rank_bm25's IDF for a term in exactly one doc is 0 (corpus-wide
        # terms get floored), so the term-free doc is needed to keep the
        # identifier's weight positive.
        exact = _doc("raises ERR_5021 when the token expires")
        store = _FakeVectorStore(
            count=3,
            corpus=[_doc("an error occurred"), exact, _doc("banana bread")],
            hits=[_doc("wrong")],
        )
        _install(monkeypatch, store, _settings(mode="keyword", top_k=1))
        assert retrieve("ERR_5021") == [exact]
        _install(monkeypatch, store, _settings(mode="keyword", top_k=5))
        assert retrieve("ERR_5021") == [exact]
        assert all(call[0] != "search" for call in store.calls)

        # hybrid: BM25 puts ``shared`` first where the vector list ranks it
        # second, so RRF must promote it; fusion dedupes the chunk both
        # retrievers returned by doc_key, and the corpus loads exactly once.
        shared = _doc("shared unique_token passage")
        store = _FakeVectorStore(
            count=3,
            corpus=[shared, _doc("unrelated"), _doc("banana bread")],
            hits=[_doc("vector-only hit"), _doc(shared.page_content)],
        )
        _install(monkeypatch, store, _settings(mode="hybrid", top_k=2))
        found = retrieve("unique_token")
        assert found[0].page_content == shared.page_content
        assert found[1].page_content == "vector-only hit"
        assert store.calls.count(("all_documents",)) == 1  # corpus loaded once
        assert ("search", "unique_token", 2) in store.calls

    def test_rerank_overfetches_then_cuts_to_top_k(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        hits = [_doc(f"h{i}") for i in range(4)]
        store = _FakeVectorStore(count=4, hits=hits)
        seen: list[tuple[int, int]] = []

        def fake_rerank(
            question: str, candidates: list[Document], k: int
        ) -> list[Document]:
            seen.append((len(candidates), k))
            return list(reversed(candidates))[:k]

        monkeypatch.setattr(pipeline, "rerank", fake_rerank)
        _install(
            monkeypatch,
            store,
            _settings(mode="vector", top_k=2, rerank=True, multiplier=3),
        )

        found = retrieve("q")
        # Candidates fetched = top_k × multiplier, even though hits ran short.
        assert store.calls == [("search", "q", 6)]
        assert seen == [(4, 2)]
        assert [d.page_content for d in found] == ["h3", "h2"]


class TestInputValidation:
    """Misconfiguration must fail loudly, not silently return odd rankings."""

    def test_guards_reject_invalid_arguments(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        with pytest.raises(ValueError, match="k must be"):
            BM25Index([_doc("a")]).search("a", k=0)
        with pytest.raises(ValueError, match="k must be"):
            vector_search(_FakeVectorStore(count=1), "q", k=0)
        with pytest.raises(ValueError, match="c must be"):
            rrf_fuse([], [], c=0)
        with pytest.raises(ValueError, match="top_n"):
            rrf_fuse([], [], c=60, top_n=-1)
        with pytest.raises(ValueError):
            rrf_fuse([[_doc("a")]], [], c=60)
        # Explicit dispatch: a typo'd mode must not silently run as hybrid.
        store = _FakeVectorStore(count=1, corpus=[_doc("a")])
        _install(monkeypatch, store, _settings(mode="semantic"))
        with pytest.raises(ValueError, match="retrieval_mode"):
            retrieve("q")
        assert store.calls == []

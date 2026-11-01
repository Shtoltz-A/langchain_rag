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
        OLLAMA_LLM_MODEL="llama-test",
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
    def test_lowercases_and_splits_punctuation(self) -> None:
        assert _tokenize("Failed! see ERR_5021") == ["failed", "see", "err_5021"]

    def test_folds_accents(self) -> None:
        # NFKD + combining-mark strip: accented and bare spellings collide.
        assert _tokenize("Café l'été") == ["cafe", "l", "ete"]

    def test_keeps_cyrillic(self) -> None:
        assert _tokenize("Ошибка ERR_5021") == ["ошибка", "err_5021"]


class TestDocKey:
    def test_explicit_id_wins_over_content(self) -> None:
        one = Document(page_content="text a", metadata={"id": 7, "source": "a.txt"})
        other = Document(page_content="text b", metadata={"id": 7, "source": "b.txt"})
        assert doc_key(one) == doc_key(other)

    def test_fallback_hash_merges_same_chunk_across_instances(self) -> None:
        # Known limit: duplicated boilerplate inside one source stays merged
        # while no position metadata distinguishes the two chunks.
        assert doc_key(_doc("same text")) == doc_key(_doc("same text"))

    def test_position_fields_keep_repeated_boilerplate_distinct(self) -> None:
        assert doc_key(_doc("boilerplate", page=1)) != doc_key(
            _doc("boilerplate", page=2)
        )
        assert doc_key(_doc("boilerplate", chunk_index=0)) != doc_key(
            _doc("boilerplate", chunk_index=1)
        )

    def test_fallback_hash_keeps_same_text_from_different_sources_apart(self) -> None:
        a = _doc("same text", source="a.txt")
        b = _doc("same text", source="b.txt")
        assert doc_key(a) != doc_key(b)


class TestRrfFuse:
    def test_document_agreeing_on_both_lists_outranks_single_top(self) -> None:
        a, b, other = _doc("a"), _doc("b"), _doc("c")
        # b: rank 1 in list 1 + rank 0 in list 2; a: rank 0 only; c: rank 1 only.
        fused = rrf_fuse([[a, b], [b, other]], [0.5, 0.5], c=60)
        assert [d.page_content for d in fused] == ["b", "a", "c"]

    def test_scores_match_hand_computed_rrf(self) -> None:
        a, b, other = _doc("a"), _doc("b"), _doc("c")
        # Hand check: b = 0.5/62 + 0.5/61 ≈ 0.016235 must beat a = 0.5/61 ≈ 0.008197.
        fused = rrf_fuse([[a, b], [b, other]], [0.5, 0.5], c=60)
        assert fused[0] is b
        assert fused[1] is a

    def test_merges_same_chunk_seen_by_both_retrievers(self) -> None:
        # Chroma hands the same stored chunk to both retrievers as different
        # instances; same (source, position, content) means one document —
        # so the distinguishing field must not be a position field.
        first = _doc("same text", marker="first")
        second = _doc("same text", marker="second")
        fused = rrf_fuse([[first], [second]], [0.5, 0.5], c=60)
        assert len(fused) == 1
        assert fused[0].metadata["marker"] == "first"  # first instance wins

    def test_identical_text_from_different_sources_stays_distinct(self) -> None:
        a = _doc("same text", source="a.txt")
        b = _doc("same text", source="b.txt")
        fused = rrf_fuse([[a], [b]], [0.5, 0.5], c=60)
        assert len(fused) == 2

    def test_zero_weight_excludes_a_list(self) -> None:
        a, b = _doc("a"), _doc("b")
        # List 2 says the opposite but carries no weight.
        fused = rrf_fuse([[a, b], [b, a]], [1.0, 0.0], c=60)
        assert [d.page_content for d in fused] == ["a", "b"]

    def test_ties_keep_earlier_list_order(self) -> None:
        a, b = _doc("a"), _doc("b")
        fused = rrf_fuse([[a, b], [a, b]], [0.5, 0.5], c=60)
        assert [d.page_content for d in fused] == ["a", "b"]

    def test_empty_rankings(self) -> None:
        a = _doc("a")
        assert rrf_fuse([], [], c=60) == []
        assert rrf_fuse([[], [a]], [0.5, 0.5], c=60) == [a]

    def test_top_n_truncates_fused_list(self) -> None:
        docs = [_doc(t) for t in "abcd"]
        rankings, weights = [[docs[0], docs[1]], [docs[2], docs[3]]], [0.5, 0.5]
        # Two 2-lists fuse to up to 2k documents without a cutoff.
        assert len(rrf_fuse(rankings, weights, c=60)) == 4
        # Ranks: a, c at 0.5/61 each, b, d at 0.5/62; ties keep first-seen.
        fused = rrf_fuse(rankings, weights, c=60, top_n=3)
        assert [d.page_content for d in fused] == ["a", "c", "b"]


class TestBM25Index:
    def test_exact_identifier_beats_paraphrase(self) -> None:
        corpus = [
            _doc("an error occurred while connecting"),
            _doc("raises ERR_5021 when the token expires"),
            _doc("banana bread recipe"),
        ]
        found = BM25Index(corpus).search("ERR_5021", k=2)
        assert found[0] is corpus[1]

    def test_zero_score_documents_are_excluded(self) -> None:
        # No term overlap = no evidence; padding with them only adds fusion
        # noise, so even a generous k must not resurrect them.
        exact = _doc("raises ERR_5021 when the token expires")
        corpus = [exact, _doc("an error occurred"), _doc("banana bread recipe")]
        assert BM25Index(corpus).search("ERR_5021", k=5) == [exact]

    def test_query_without_overlap_returns_empty(self) -> None:
        corpus = [_doc("alpha one"), _doc("alpha two"), _doc("beta three")]
        assert BM25Index(corpus).search("zzz_missing_token", k=3) == []

    def test_k_truncates(self) -> None:
        # 3 of 7 docs match: rank_bm25 keeps IDF positive only while the term
        # stays rare, so the non-matching filler docs are load-bearing here.
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

    def test_ties_keep_corpus_order(self) -> None:
        corpus = [
            _doc("alpha x"),
            _doc("alpha y"),
            _doc("beta one"),
            _doc("beta two"),
            _doc("beta three"),
        ]
        assert BM25Index(corpus).search("alpha", k=3) == corpus[:2]

    def test_empty_corpus_returns_empty_without_crashing(self) -> None:
        # rank_bm25 itself dies on an empty corpus; the index must not.
        assert BM25Index([]).search("anything", k=3) == []


class TestRetrievePipeline:
    def test_empty_store_short_circuits_every_retriever(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store = _FakeVectorStore(count=0)
        _install(monkeypatch, store, _settings(mode="hybrid"))
        assert retrieve("anything") == []
        assert store.calls == []

    def test_vector_mode_fetches_and_cuts_top_k(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        hits = [_doc(f"h{i}") for i in range(5)]
        store = _FakeVectorStore(count=5, hits=hits)
        _install(monkeypatch, store, _settings(mode="vector", top_k=2))
        found = retrieve("q")
        assert [d.page_content for d in found] == ["h0", "h1"]
        assert store.calls == [("search", "q", 2)]

    def test_keyword_mode_never_touches_vector_search(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        exact = _doc("raises ERR_5021 when the token expires")
        store = _FakeVectorStore(
            count=3,
            # 3 docs: on a 2-doc corpus rank_bm25's IDF for a term in exactly
            # one doc is 0 (corpus-wide terms get floored), so the term-free
            # doc is needed to keep the identifier's weight positive.
            corpus=[_doc("an error occurred"), exact, _doc("banana bread")],
            hits=[_doc("wrong")],
        )
        _install(monkeypatch, store, _settings(mode="keyword", top_k=1))
        found = retrieve("ERR_5021")
        assert found == [exact]
        assert all(call[0] != "search" for call in store.calls)

    def test_keyword_mode_can_return_fewer_than_top_k(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Zero-score noise documents no longer pad the result up to top_k.
        exact = _doc("raises ERR_5021 when the token expires")
        store = _FakeVectorStore(
            count=3,
            corpus=[exact, _doc("an error occurred"), _doc("banana bread")],
        )
        _install(monkeypatch, store, _settings(mode="keyword", top_k=5))
        assert retrieve("ERR_5021") == [exact]

    def test_hybrid_mode_fuses_both_rankings(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        shared = _doc("shared unique_token passage")
        store = _FakeVectorStore(
            count=3,
            corpus=[shared, _doc("unrelated"), _doc("banana bread")],
            # Vector ranks a plain hit first; shared only second — BM25 puts
            # shared first, so RRF must promote it to the top.
            hits=[_doc("vector-only hit"), _doc(shared.page_content)],
        )
        _install(monkeypatch, store, _settings(mode="hybrid", top_k=2))
        found = retrieve("unique_token")
        # Fusion dedupes by doc_key, so the shared chunk wins on rank sum
        # even though neither retriever ranked it #1 twice.
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

    def test_bm25_search_rejects_nonpositive_k(self) -> None:
        with pytest.raises(ValueError, match="k must be"):
            BM25Index([_doc("a")]).search("a", k=0)

    def test_vector_search_rejects_nonpositive_k(self) -> None:
        with pytest.raises(ValueError, match="k must be"):
            vector_search(_FakeVectorStore(count=1), "q", k=0)

    def test_fuse_rejects_invalid_c_and_top_n(self) -> None:
        with pytest.raises(ValueError, match="c must be"):
            rrf_fuse([], [], c=0)
        with pytest.raises(ValueError, match="top_n"):
            rrf_fuse([], [], c=60, top_n=-1)

    def test_fuse_rejects_length_mismatch(self) -> None:
        with pytest.raises(ValueError):
            rrf_fuse([[_doc("a")]], [], c=60)

    def test_retrieve_rejects_unknown_mode(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Explicit dispatch: a typo'd mode must not silently run as hybrid.
        store = _FakeVectorStore(count=1, corpus=[_doc("a")])
        _install(monkeypatch, store, _settings(mode="semantic"))
        with pytest.raises(ValueError, match="retrieval_mode"):
            retrieve("q")
        assert store.calls == []

"""Unit tests for storage.vectorstore (VectorStore singleton with fakes)."""

from __future__ import annotations

import itertools
from typing import Any, ClassVar

import pytest
from langchain_core.documents import Document

from langchain_rag.config import PROJECT_ROOT, Settings
from langchain_rag.retrieval.retrievers import doc_key
from langchain_rag.storage import vectorstore as vectorstore_mod
from langchain_rag.storage.vectorstore import VectorStore


class _FakeEmbeddings:
    """Stands in for OllamaEmbeddings: records the model name only."""

    def __init__(self, model: str) -> None:
        self.model = model


class _FakeCollection:
    """Stands in for the private chromadb collection (count() only)."""

    def count(self) -> int:
        return 7


class _FakeChroma:
    """Stands in for the LangChain Chroma wrapper; records every call."""

    # Canned stored chunks shared by both read paths (search and get).
    _STORE: ClassVar[list[tuple[str, dict]]] = [
        ("chunk one", {"source": r"C:\docs\a.txt"}),
        ("chunk two", {"source": r"C:\docs\b.pdf", "page": 2}),
    ]

    def __init__(
        self, collection_name: str, embedding_function: Any, persist_directory: str
    ) -> None:
        self.collection_name = collection_name
        self.embedding_function = embedding_function
        self.persist_directory = persist_directory
        self.calls: list[tuple] = []
        self._collection = _FakeCollection()

    def reset_collection(self) -> None:
        self.calls.append(("reset_collection",))

    def add_documents(self, documents: list[Any]) -> None:
        self.calls.append(("add_documents", documents))

    def delete(self, where: dict) -> None:
        self.calls.append(("delete", where))

    def similarity_search(self, query: str, k: int) -> list[Document]:
        self.calls.append(("similarity_search", query, k))
        return [
            Document(page_content=text, metadata=dict(meta))
            for text, meta in itertools.islice(itertools.cycle(self._STORE), k)
        ]

    def get(self, include: list[str]) -> dict:
        # ids stay top-level (like chromadb): neither read path may leak them
        # into Document metadata — retrieval key consistency depends on it.
        self.calls.append(("get", tuple(include)))
        return {
            "ids": [f"id-{i}" for i, _ in enumerate(self._STORE)],
            "documents": [text for text, _ in self._STORE],
            "metadatas": [dict(meta) for _, meta in self._STORE],
        }


@pytest.fixture(autouse=True)
def fake_chroma(
    monkeypatch: pytest.MonkeyPatch,
) -> list[_FakeChroma]:
    """Atomically fake Chroma/OllamaEmbeddings and reset the singleton.

    Patching happens before any test body runs, so a real Chroma handle on
    ``data/chroma/`` can never be built by this suite; every ``VectorStore()``
    in the suite gets the fake below.
    """
    created: list[_FakeChroma] = []

    def factory(**kwargs: Any) -> _FakeChroma:
        store = _FakeChroma(**kwargs)
        created.append(store)
        return store

    monkeypatch.setattr(vectorstore_mod, "Chroma", factory)
    monkeypatch.setattr(vectorstore_mod, "OllamaEmbeddings", _FakeEmbeddings)
    monkeypatch.setattr(VectorStore, "_instance", None)
    return created


def _patch_settings(monkeypatch: pytest.MonkeyPatch, **updates: Any) -> None:
    """Point the module's get_settings at a real Settings with overrides."""
    settings = Settings().model_copy(update=updates)
    monkeypatch.setattr(vectorstore_mod, "get_settings", lambda: settings)


class TestSingleton:
    def test_returns_same_instance(self, fake_chroma: list[_FakeChroma]) -> None:
        assert VectorStore() is VectorStore()

    def test_constructs_underlying_store_exactly_once(
        self, fake_chroma: list[_FakeChroma]
    ) -> None:
        for _ in range(3):
            VectorStore()
        assert len(fake_chroma) == 1

    def test_construction_kwargs_pinned(
        self, fake_chroma: list[_FakeChroma], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_settings(monkeypatch, OLLAMA_EMBEDDING_MODEL="test-embedder")
        VectorStore()
        fake = fake_chroma[0]
        assert fake.collection_name == "documents"
        assert fake.persist_directory == str(PROJECT_ROOT / "data" / "chroma")
        assert fake.embedding_function.model == "test-embedder"

    def test_chroma_property_exposes_the_single_handle(
        self, fake_chroma: list[_FakeChroma]
    ) -> None:
        assert VectorStore().chroma is VectorStore().chroma is fake_chroma[0]


class TestPrimitives:
    def test_clear_resets_collection(self, fake_chroma: list[_FakeChroma]) -> None:
        VectorStore().clear()
        assert ("reset_collection",) in fake_chroma[0].calls

    def test_add_documents_forwards(self, fake_chroma: list[_FakeChroma]) -> None:
        documents = ["chunk-a", "chunk-b"]  # opaque payloads suffice
        VectorStore().add_documents(documents)
        assert ("add_documents", documents) in fake_chroma[0].calls

    def test_delete_by_source_uses_where_filter(
        self, fake_chroma: list[_FakeChroma]
    ) -> None:
        source = r"C:\index\sub\doc.md"
        VectorStore().delete_by_source(source)
        assert ("delete", {"source": source}) in fake_chroma[0].calls

    def test_search_forwards_query_and_k(self, fake_chroma: list[_FakeChroma]) -> None:
        hits = VectorStore().search("what?", k=4)
        assert ("similarity_search", "what?", 4) in fake_chroma[0].calls
        assert len(hits) == 4

    def test_count_reads_underlying_collection(
        self, fake_chroma: list[_FakeChroma]
    ) -> None:
        assert VectorStore().count() == 7


class TestReadPathKeyConsistency:
    """RRF dedupe compares doc_keys across the two read paths, so the same
    stored chunk must yield the same key via search() and via
    all_documents() — an id injected by one path only would split every key
    into id:… vs sha1:… and silently disable the dedupe (rag-decisions.md)."""

    def test_search_and_all_documents_agree_on_doc_key(
        self, fake_chroma: list[_FakeChroma]
    ) -> None:
        via_search = VectorStore().search("q", k=1)[0]
        via_get = VectorStore().all_documents()[0]
        assert doc_key(via_search) == doc_key(via_get)

    def test_neither_read_path_injects_an_id(
        self, fake_chroma: list[_FakeChroma]
    ) -> None:
        assert "id" not in VectorStore().search("q", k=1)[0].metadata
        assert all("id" not in doc.metadata for doc in VectorStore().all_documents())

    def test_position_metadata_survives_both_paths(
        self, fake_chroma: list[_FakeChroma]
    ) -> None:
        # The PDF chunk's page must distinguish it on either path, not just
        # one — otherwise position-aware dedupe diverges between the paths.
        paged_search = next(
            doc for doc in VectorStore().search("q", k=2) if "page" in doc.metadata
        )
        paged_get = next(
            doc for doc in VectorStore().all_documents() if "page" in doc.metadata
        )
        assert doc_key(paged_search) == doc_key(paged_get)

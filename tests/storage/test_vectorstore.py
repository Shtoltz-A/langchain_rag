"""Unit tests for storage.vectorstore (VectorStore singleton with fakes)."""

from __future__ import annotations

import itertools
import sqlite3
from pathlib import Path
from typing import Any, ClassVar
from uuid import uuid4

import pytest
from langchain_core.documents import Document

from langchain_rag.config import PROJECT_ROOT, Settings
from langchain_rag.retrieval.retrievers import doc_key
from langchain_rag.storage import vectorstore as vectorstore_mod
from langchain_rag.storage.vectorstore import VectorStore, prune_orphan_segment_dirs


def _make_chroma_db(
    directory: Path, segment_ids: list[str], collection_ids: list[str] | None = None
) -> None:
    """Minimal stand-in for ``chroma.sqlite3``: the two id tables the sweep reads."""
    connection = sqlite3.connect(directory / "chroma.sqlite3")
    with connection:
        connection.execute("create table segments (id text primary key)")
        connection.execute("create table collections (id text primary key)")
        connection.executemany(
            "insert into segments values (?)", [(i,) for i in segment_ids]
        )
        connection.executemany(
            "insert into collections values (?)", [(i,) for i in collection_ids or []]
        )
    connection.close()


def _make_db_without_expected_tables(directory: Path) -> None:
    """A DB that opens cleanly but lacks the segments/collections tables."""
    connection = sqlite3.connect(directory / "chroma.sqlite3")
    with connection:
        connection.execute("create table migrations (version integer)")
    connection.close()


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


def test_singleton_builds_one_store_with_pinned_kwargs(
    fake_chroma: list[_FakeChroma], monkeypatch: pytest.MonkeyPatch
) -> None:
    """One memoized instance per process; the real Chroma is built exactly once,
    on first call, with pinned collection/persist-dir and the configured model."""
    _patch_settings(monkeypatch, ollama_embedding_model="test-embedder")

    store = VectorStore()

    assert VectorStore() is store  # same instance handed back every call
    assert store.chroma is VectorStore().chroma
    for _ in range(3):
        VectorStore()
    assert len(fake_chroma) == 1  # underlying store constructed only once
    fake = fake_chroma[0]
    assert store.chroma is fake  # the single handle the property exposes
    assert fake.collection_name == "documents"
    assert fake.persist_directory == str(PROJECT_ROOT / "data" / "chroma")
    assert fake.embedding_function.model == "test-embedder"


def test_primitives_forward_to_chroma(fake_chroma: list[_FakeChroma]) -> None:
    """add/delete/search/count translate to the exact Chroma call expected."""
    store = VectorStore()
    documents = ["chunk-a", "chunk-b"]  # opaque payloads suffice
    store.add_documents(documents)
    assert ("add_documents", documents) in fake_chroma[0].calls

    source = r"C:\index\sub\doc.md"
    store.delete_by_source(source)
    assert ("delete", {"source": source}) in fake_chroma[0].calls

    hits = store.search("what?", k=4)
    assert ("similarity_search", "what?", 4) in fake_chroma[0].calls
    assert len(hits) == 4

    assert store.count() == 7  # reads the private _collection.count()


def test_clear_resets_collection_and_sweeps_orphans(
    fake_chroma: list[_FakeChroma],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """clear() resets the collection then sweeps leaked segment dirs."""
    # Redirect the sweep at tmp_path: the suite may never touch the real
    # data/chroma. The DB holds the fresh post-reset segment id; the orphan dir
    # matches nothing and must die with the wipe.
    sweep_dir = tmp_path / "chroma"
    sweep_dir.mkdir()
    monkeypatch.setattr(vectorstore_mod, "PERSIST_SUBDIR", sweep_dir)
    _make_chroma_db(sweep_dir, [str(uuid4())])
    orphan = sweep_dir / str(uuid4())
    orphan.mkdir()

    assert VectorStore().clear() == 1
    assert ("reset_collection",) in fake_chroma[0].calls
    assert not orphan.exists()


def test_read_paths_agree_on_metadata_and_doc_keys(
    fake_chroma: list[_FakeChroma],
) -> None:
    """RRF dedupe compares doc_keys across the two read paths, so a chunk must
    yield the same key via search() and all_documents(): no id may be injected by
    one path, and position metadata (page) must survive both."""
    store = VectorStore()

    via_search = store.search("q", k=1)[0]
    via_get = store.all_documents()[0]
    assert doc_key(via_search) == doc_key(via_get)

    assert "id" not in via_search.metadata
    assert all("id" not in doc.metadata for doc in store.all_documents())

    # the PDF chunk's page must distinguish it on either path, not just one
    paged_search = next(doc for doc in store.search("q", k=2) if "page" in doc.metadata)
    paged_get = next(doc for doc in store.all_documents() if "page" in doc.metadata)
    assert doc_key(paged_search) == doc_key(paged_get)


def test_prune_orphan_segment_dirs(tmp_path: Path) -> None:
    """The janitor: only UUID dirs the DB forgot may die; every other case — a
    case-mismatched live id or an unreadable DB — is a fail-safe no-op."""
    # (1) prunes only the orphan uuid dir; live/collection/non-uuid dirs, the
    # sqlite file and a transient -journal all survive
    root = tmp_path / "healthy"
    root.mkdir()
    live_segment, live_collection, orphan = str(uuid4()), str(uuid4()), str(uuid4())
    _make_chroma_db(root, [live_segment], [live_collection])
    keepers = [root / live_segment, root / live_collection, root / "not-a-uuid"]
    for directory in [*keepers, root / orphan]:
        directory.mkdir()
    (root / "chroma.sqlite3-journal").write_bytes(b"")

    assert prune_orphan_segment_dirs(root) == 1
    assert not (root / orphan).exists()
    assert all(directory.exists() for directory in keepers)
    assert (root / "chroma.sqlite3-journal").exists()

    # (2) NTFS is case-insensitive: a dir may present different casing than the
    # DB id yet still be live
    cased = tmp_path / "cased"
    cased.mkdir()
    live = str(uuid4())
    _make_chroma_db(cased, [live.upper()])
    (cased / live).mkdir()
    assert prune_orphan_segment_dirs(cased) == 0
    assert (cased / live).exists()

    # (3) missing / corrupt / tableless DB: orphan status is unprovable -> 0
    for name, build in (
        ("missing", lambda d: None),
        ("corrupt", lambda d: (d / "chroma.sqlite3").write_bytes(b"not a database")),
        ("no-tables", _make_db_without_expected_tables),
    ):
        case = tmp_path / name
        case.mkdir()
        build(case)
        stale = case / str(uuid4())
        stale.mkdir()
        assert prune_orphan_segment_dirs(case) == 0, name
        assert stale.exists(), name

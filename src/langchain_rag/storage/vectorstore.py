"""Vector store access: Chroma persistence, embedding model construction, and
the janitor that sweeps segment directories a collection reset leaves behind.

The only place that builds the embeddings model or touches ``langchain_chroma``
— or the on-disk layout behind it. `services/indexing.py` orchestrates the
pipeline; this
module owns the store as a singleton class — one persisted collection per
process, built lazily on first use.
"""

from __future__ import annotations

import shutil
import sqlite3
import uuid
from pathlib import Path
from typing import ClassVar, Self

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

from ..config import PROJECT_ROOT, get_settings
from ..console import get_console

COLLECTION_NAME = "documents"
PERSIST_SUBDIR = "data/chroma"
SQLITE_FILENAME = "chroma.sqlite3"


def prune_orphan_segment_dirs(chroma_dir: Path | None = None) -> int:
    """Delete persisted segment directories the Chroma database no longer knows.

    ``reset_collection()`` assigns fresh segment ids but chromadb 1.5.9 never
    deletes the old HNSW directories from the persist dir (rag-decisions.md),
    so every collection wipe leaks one folder per vector segment. This janitor
    removes only what the database provably no longer references: direct
    sub-directories whose names are UUIDs absent from both the ``segments``
    and ``collections`` id columns. Files (the sqlite DB itself, a transient
    ``-journal``) and non-UUID directories are never touched, and every
    failure path fails safe (skip, never raise) — janitorial work must not
    break the wipe it follows.

    Args:
        chroma_dir: persist directory to sweep; defaults to the app's Chroma
            dir, resolved at call time so tests can redirect it.

    Returns:
        Number of orphan directories deleted.
    """
    c = get_console()
    directory = PROJECT_ROOT / PERSIST_SUBDIR if chroma_dir is None else chroma_dir
    db = directory / SQLITE_FILENAME
    live: set[str] = set()
    try:
        # Read-only connection: chroma's local DB runs in rollback-journal
        # mode, so the freshly committed post-reset state is immediately
        # visible and readers can never block the (same-thread) writer.
        connection = sqlite3.connect(f"{db.as_uri()}?mode=ro", uri=True)
        try:
            for table in ("segments", "collections"):
                live.update(
                    str(row_id).lower()
                    for (row_id,) in connection.execute(f"select id from {table}")
                )
        finally:
            connection.close()
    except sqlite3.Error as exc:
        # Missing/corrupt/locked DB: orphan status is unprovable — touch nothing.
        c.debug(f"Segment-dir sweep skipped: {exc}")
        return 0

    pruned = 0
    try:
        children = list(directory.iterdir())
    except OSError as exc:
        c.debug(f"Segment-dir sweep skipped: {exc}")
        return 0
    for child in children:
        if not child.is_dir() or child.name.lower() in live:
            continue  # files and live dirs are untouchable by design
        try:
            uuid.UUID(child.name)
        except ValueError:
            continue  # not a Chroma segment directory — not ours to judge
        try:
            shutil.rmtree(child)
        except OSError as exc:  # e.g. a Windows lock — leave it for the next wipe
            c.debug(f"Could not prune stale segment dir {child.name}: {exc}")
            continue
        pruned += 1
    return pruned


class VectorStore:
    """Singleton facade over the persisted Chroma collection.

    Exactly one store may exist per process (one collection, one persist
    directory — a second Chroma handle on the same collection is a footgun),
    so ``__new__`` memoizes the single instance and the Chroma handle is
    built on first instantiation, keeping imports free of disk side effects.
    Constructing a subclass before the base class is a footgun of its own
    (see rag-decisions.md); settings are frozen at first construction for
    the life of the process.

    Attributes:
        _chroma: the underlying Chroma wrapper, set on the instance in
            ``__new__`` during the one and only construction.
    """

    _instance: ClassVar[VectorStore | None] = None

    _chroma: Chroma

    def __new__(cls) -> Self:
        """Return the shared instance, building Chroma on first call.

        Deliberately no ``__init__``: every later ``VectorStore()`` call
        re-runs nothing and hands back the memoized instance unchanged.
        """
        if cls._instance is None:
            instance = super().__new__(cls)
            instance._chroma = Chroma(
                collection_name=COLLECTION_NAME,
                embedding_function=OllamaEmbeddings(
                    model=get_settings().ollama_embedding_model
                ),
                persist_directory=str(PROJECT_ROOT / PERSIST_SUBDIR),
            )
            cls._instance = instance
        return cls._instance

    @property
    def chroma(self) -> Chroma:
        """The underlying LangChain Chroma wrapper (escape hatch)."""
        return self._chroma

    def clear(self) -> int:
        """Delete and recreate the collection, then sweep leaked segment dirs.

        chromadb never removes the old HNSW directories on a collection reset
        (rag-decisions.md), so the janitor runs here — the module's only wipe
        seam — rather than at each call site that could forget it.

        Returns:
            Number of stale segment directories pruned along the way.
        """
        self._chroma.reset_collection()
        return prune_orphan_segment_dirs()

    def add_documents(self, documents: list[Document]) -> None:
        """Embed and store chunks (append — delete first when replacing a file)."""
        self._chroma.add_documents(documents)

    def delete_by_source(self, source: str) -> None:
        """Delete every stored chunk whose ``source`` metadata equals this path.

        ``delete(where=...)`` reaches chromadb through the LangChain wrapper's
        untyped ``**kwargs`` passthrough (verified against langchain-chroma
        1.1.0), so the call is kept in this one place.
        """
        self._chroma.delete(where={"source": source})

    def search(self, question: str, k: int) -> list[Document]:
        """Return the k most similar chunks for the question."""
        return self._chroma.similarity_search(question, k=k)

    def all_documents(self) -> list[Document]:
        """Every stored chunk as ``Document``s (BM25 corpus building).

        Vectors are excluded via ``include`` — they are the bulk of the
        response and keyword retrieval has no use for them.
        """
        raw = self._chroma.get(include=["documents", "metadatas"])
        return [
            Document(page_content=text or "", metadata=metadata or {})
            for text, metadata in zip(
                raw["documents"] or [], raw["metadatas"] or [], strict=True
            )
        ]

    def count(self) -> int:
        """Number of chunks currently stored (manifest desync guard).

        The LangChain wrapper exposes no public count, so this pokes the
        private ``_collection`` — deliberately the only place that does.
        """
        return self._chroma._collection.count()

"""Vector store access: Chroma persistence and embedding model construction.

The only place that builds the embeddings model or touches ``langchain_chroma``.
`app.py` orchestrates the pipeline; this module owns the store as a singleton
class — one persisted collection per process, built lazily on first use.
"""

from __future__ import annotations

from typing import ClassVar

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

from ..config import PROJECT_ROOT, get_settings

COLLECTION_NAME = "documents"
PERSIST_SUBDIR = "data/chroma"


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

    def __new__(cls) -> VectorStore:
        """Return the shared instance, building Chroma on first call.

        Deliberately no ``__init__``: every later ``VectorStore()`` call
        re-runs nothing and hands back the memoized instance unchanged.
        """
        if cls._instance is None:
            instance = super().__new__(cls)
            instance._chroma = Chroma(
                collection_name=COLLECTION_NAME,
                embedding_function=OllamaEmbeddings(
                    model=get_settings().OLLAMA_EMBEDDING_MODEL
                ),
                persist_directory=str(PROJECT_ROOT / PERSIST_SUBDIR),
            )
            cls._instance = instance
        return cls._instance

    @property
    def chroma(self) -> Chroma:
        """The underlying LangChain Chroma wrapper (escape hatch)."""
        return self._chroma

    def clear(self) -> None:
        """Delete and recreate the collection (full-rebuild entry point)."""
        self._chroma.reset_collection()

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

    def count(self) -> int:
        """Number of chunks currently stored (manifest desync guard).

        The LangChain wrapper exposes no public count, so this pokes the
        private ``_collection`` — deliberately the only place that does.
        """
        return self._chroma._collection.count()

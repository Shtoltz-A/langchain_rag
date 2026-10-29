"""File loaders: turn files on disk into LangChain Documents.

Defines a DocumentLoader base class and one subclass per supported file
type. Use load_documents() to load a whole directory with per-file error
tolerance; unknown file types are skipped with a warning.
"""

from __future__ import annotations

import json
from abc import abstractmethod
from collections.abc import Iterator
from pathlib import Path
from typing import ClassVar

from langchain_core.document_loaders.base import BaseLoader
from langchain_core.documents import Document
from pypdf import PdfReader

from ..console import Console

c = Console()


def _is_hidden(path: Path, root: Path) -> bool:
    """True if the path or any part below ``root`` starts with ``.``."""
    return any(part.startswith(".") for part in path.relative_to(root).parts)


def file_metadata(path: Path, root: Path) -> dict[str, object]:
    """Base metadata for every loader: full source + index-root-relative path.

    ``relative_path`` is POSIX-style (e.g. "pdf/report.pdf") so context
    prefixes can show the folder grouping without OS-specific separators.
    """
    return {
        "source": str(path),
        "relative_path": path.relative_to(root).as_posix(),
    }


class DocumentLoader(BaseLoader):
    """Base class for file-based document loaders.

    Subclasses declare which file suffixes they handle and implement
    lazy_load(), yielding Documents with at least a ``source`` metadata
    entry identifying the origin file. The ``root`` constructor argument
    is the directory being indexed; it anchors the ``relative_path``
    metadata and defaults to the file's own directory.

    Use cases:
        - Dispatch a file to the right loader via the registry below.
        - Add a new file format by subclassing and registering it.
    """

    supported_suffixes: ClassVar[frozenset[str]] = frozenset()

    def __init__(self, path: Path, root: Path | None = None) -> None:
        self.path = path
        self.root = root if root is not None else path.parent

    @property
    def base_metadata(self) -> dict[str, object]:
        """Metadata common to every document this loader yields."""
        return file_metadata(self.path, self.root)

    @classmethod
    def supports(cls, path: Path) -> bool:
        """Return True if this loader handles the given file's suffix."""
        return path.suffix.lower() in cls.supported_suffixes

    @abstractmethod
    def lazy_load(self) -> Iterator[Document]:
        """Yield documents produced from the file."""


class TextFileLoader(DocumentLoader):
    """Load a plain-text file as a single document."""

    supported_suffixes = frozenset({".txt"})

    def lazy_load(self) -> Iterator[Document]:
        content = self.path.read_text(encoding="utf-8")
        yield Document(page_content=content, metadata=dict(self.base_metadata))


class MarkdownLoader(DocumentLoader):
    """Load a Markdown file as a single document.

    Content is kept as-is; structure-aware chunking arrives in Step 2.3.
    """

    supported_suffixes = frozenset({".md", ".markdown"})

    def lazy_load(self) -> Iterator[Document]:
        content = self.path.read_text(encoding="utf-8")
        yield Document(page_content=content, metadata=dict(self.base_metadata))


class PdfLoader(DocumentLoader):
    """Load a PDF as one document per non-empty page.

    Page numbers are 0-based, following the LangChain metadata convention;
    format them at display time (e.g. "p.{page + 1}").
    """

    supported_suffixes = frozenset({".pdf"})

    def lazy_load(self) -> Iterator[Document]:
        reader = PdfReader(str(self.path))
        for page_number, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if not text.strip():
                continue
            yield Document(
                page_content=text,
                metadata={**self.base_metadata, "page": page_number},
            )


class PythonLoader(DocumentLoader):
    """Load a Python source file as a single document."""

    supported_suffixes = frozenset({".py"})

    def lazy_load(self) -> Iterator[Document]:
        content = self.path.read_text(encoding="utf-8")
        yield Document(page_content=content, metadata=dict(self.base_metadata))


class JsonLoader(DocumentLoader):
    """Load a JSON file as a single document.

    Content is parsed with json.loads() to reject malformed files early
    (per-file error tolerance skips them); the raw text is kept as page
    content so structure-aware chunking can re-serialize per key.
    """

    supported_suffixes = frozenset({".json"})

    def lazy_load(self) -> Iterator[Document]:
        content = self.path.read_text(encoding="utf-8")
        json.loads(content)  # validation only; raises on malformed JSON
        yield Document(page_content=content, metadata=dict(self.base_metadata))


SUPPORTED_LOADERS: tuple[type[DocumentLoader], ...] = (
    TextFileLoader,
    MarkdownLoader,
    PdfLoader,
    PythonLoader,
    JsonLoader,
)


def get_loader(path: Path) -> type[DocumentLoader] | None:
    """Return the loader class registered for the file's suffix, if any."""
    for loader in SUPPORTED_LOADERS:
        if loader.supports(path):
            return loader
    return None


def load_documents(directory: Path) -> list[Document]:
    """Load all supported files under the directory into documents (recursive).

    Args:
        directory: Root directory; subdirectories are walked to any depth so
            files grouped in folders ("repositories") are all ingested.
            Hidden paths (any component starting with ``.``, e.g. ``.git/``)
            are skipped silently.

    Use cases:
        - Feed an ingestion-ready document list into chunking and indexing.

    Unknown file types are skipped with a warning. A file that fails to
    load (bad encoding, corrupt PDF, ...) is reported and skipped so one
    broken file cannot abort the whole run. Every document carries
    ``source`` (full path) and ``relative_path`` (POSIX path from
    ``directory``) metadata.
    """
    documents: list[Document] = []
    for file in sorted(directory.rglob("*")):
        if not file.is_file() or _is_hidden(file, directory):
            continue
        loader = get_loader(file)
        if loader is None:
            c.warning(f"Unsupported file type, skipping: {file.relative_to(directory)}")
            continue
        try:
            # Consume inside the guard: lazy_load is a generator, so file
            # errors (encoding, corrupt PDF) surface at consumption time.
            documents.extend(loader(file, root=directory).load())
        except Exception as exc:
            c.error(f"Failed to load {file.relative_to(directory)}: {exc}")
    return documents

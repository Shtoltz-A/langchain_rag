"""Chunking strategies per content type.

Mirrors the loaders pattern: ChunkingStrategy base + one subclass per
content family (text, markdown, pdf, code, json), a registry, and a
get_chunker() lookup. Documents are routed to a strategy by the suffix of
their metadata["source"] path; unknown or missing sources fall back to
TextChunker.

MarkdownChunker and JsonChunker are custom implementations (fence-aware
section parsing and top-level-key splitting respectively); the rest wrap
langchain_text_splitters.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar

from langchain_core.documents import Document
from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter,
)

from ..config import Settings, get_settings
from ..console import Console

c = Console()

_settings: Settings = get_settings()


def _word_length(text: str) -> int:
    """Count words (whitespace-separated tokens) instead of characters."""
    return len(text.split())


class ChunkingStrategy(ABC):
    """Base class for per-content-type chunking strategies.

    Subclasses declare the file suffixes they chunk and implement
    split_documents(), turning loader-produced Documents into retrieval
    chunks. Every strategy preserves the parent metadata on each chunk.
    """

    supported_suffixes: ClassVar[frozenset[str]] = frozenset()

    @classmethod
    def supports(cls, path: Path) -> bool:
        """Return True if this strategy chunks files with the given path's suffix."""
        return path.suffix.lower() in cls.supported_suffixes

    @abstractmethod
    def split_documents(self, documents: list[Document]) -> list[Document]:
        """Split loader documents into retrieval chunks, preserving metadata."""


class TextChunker(ChunkingStrategy):
    """Recursive character splitting for plain-text documents."""

    supported_suffixes = frozenset({".txt"})

    def __init__(self) -> None:
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=_settings.chunk_size_text_words,
            chunk_overlap=_settings.chunk_overlap_text_words,
            length_function=_word_length,
            keep_separator=False,
        )

    def split_documents(self, documents: list[Document]) -> list[Document]:
        return self._splitter.split_documents(documents)


class MarkdownChunker(ChunkingStrategy):
    """Fence-aware structure splitting for Markdown documents.

    A line scanner tracks ```/~~~ fenced regions so ATX headings are only
    recognized outside code. Sections are built at `#`..`###` boundaries
    (stored as metadata h1/h2/h3), heading-stub sections are merged, fenced
    blocks stay atomic (line-split only if a fence alone exceeds the cap),
    and oversized prose sections go through a recursive splitter.
    """

    supported_suffixes = frozenset({".md", ".markdown"})

    #: Heading stubs shorter than this many words merge into an adjacent section.
    _MIN_SECTION_WORDS: ClassVar[int] = 15

    def __init__(self) -> None:
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=_settings.chunk_size_markdown_words,
            chunk_overlap=_settings.chunk_overlap_markdown_words,
            length_function=_word_length,
            keep_separator=False,
        )

    def split_documents(self, documents: list[Document]) -> list[Document]:
        chunks: list[Document] = []
        for doc in documents:
            for text, headers in self._parse_sections(doc.page_content):
                metadata = {**doc.metadata, **headers}
                chunks.extend(self._enforce_cap(text, metadata))
        return [chunk for chunk in chunks if chunk.page_content.strip()]

    @classmethod
    def _parse_sections(cls, text: str) -> list[tuple[str, dict[str, str]]]:
        """Split text into (section_text, header_metadata) pairs.

        Use cases:
            - Section extraction used by split_documents(); exposed for tests.

        Fenced code lines never count as headings and are kept verbatim in
        their section. Heading lines stay in the section body (for context)
        *and* become metadata keys (h1..h3); levels deeper than h3 stay in
        the body only.
        """
        sections: list[tuple[str, dict[str, str]]] = []
        headers: dict[str, str] = {}
        fence: str | None = None  # active fence marker ("```" or "~~~"), if any
        buf: list[str] = []

        def flush() -> None:
            body = "\n".join(buf).strip("\n")
            if body.strip() or headers:
                sections.append((body, dict(headers)))
            buf.clear()

        for line in text.splitlines():
            stripped = line.strip()
            if fence is not None:
                buf.append(line)
                if stripped.startswith(fence):
                    fence = None
                continue
            if stripped.startswith("```") or stripped.startswith("~~~"):
                fence = stripped[:3]
                buf.append(line)
                continue
            if stripped.startswith("#"):
                level = len(stripped) - len(stripped.lstrip("#"))
                heading_text = stripped[level:].strip()
                if 1 <= level <= 3 and heading_text:
                    flush()
                    headers = {k: v for k, v in headers.items() if int(k[1]) < level}
                    headers[f"h{level}"] = heading_text
                    buf.append(line)  # keep the heading line in section content
                    continue
            buf.append(line)
        flush()

        return cls._merge_stubs(sections)

    @classmethod
    def _merge_stubs(
        cls, sections: list[tuple[str, dict[str, str]]]
    ) -> list[tuple[str, dict[str, str]]]:
        """Merge heading-stub sections into the previous or next section.

        A stub is a section with body shorter than _MIN_SECTION_WORDS words
        (e.g. a bare heading acting as a group label); keeping it separate
        pollutes top_k retrieval with 12-char chunks.
        """
        if not sections:
            return sections
        merged: list[tuple[str, dict[str, str]]] = []
        pending: tuple[str, dict[str, str]] | None = None
        for body, headers in sections:
            if pending is not None:
                body = f"{pending[0]}\n\n{body}".strip()
                headers = {**pending[1], **headers}
                pending = None
            if _word_length(body) < cls._MIN_SECTION_WORDS:
                pending = (body, headers)
                continue
            merged.append((body, headers))
        if pending is not None:
            if merged:  # trailing stub joins the previous section
                last_body, last_headers = merged[-1]
                merged[-1] = (f"{last_body}\n\n{pending[0]}".strip(), last_headers)
            else:  # the whole document is one stub
                merged.append(pending)
        return merged

    def _enforce_cap(self, text: str, metadata: dict) -> list[Document]:
        """Split a section to the size cap while keeping fenced blocks atomic.

        Lines accumulate outside fences; when the buffer exceeds the cap it
        is flushed through the recursive splitter. Fenced blocks are only
        appended once closed, so a chunk never starts or ends mid-fence.
        A single fence larger than the cap goes to _split_oversized_fence.
        """
        documents: list[Document] = []
        buf: list[str] = []  # prose lines pending a flush
        fence: list[str] | None = None  # lines of the currently open fence
        for line in text.splitlines():
            stripped = line.strip()
            if fence is not None:
                fence.append(line)
                if stripped and stripped.startswith(fence[0].strip()[:3]):
                    block = "\n".join(fence)
                    if _word_length(block) > _settings.chunk_size_markdown_words:
                        documents.extend(
                            self._flush(buf, metadata)
                        )  # emit prose before the fence fragment
                        buf = []
                        for piece in self._split_oversized_fence(block):
                            documents.append(
                                Document(page_content=piece, metadata=metadata)
                            )
                    else:
                        buf.append(block)
                    fence = None
                continue
            if stripped.startswith("```") or stripped.startswith("~~~"):
                if _word_length("\n".join(buf)) > _settings.chunk_size_markdown_words:
                    documents.extend(self._flush(buf, metadata))
                    buf = []
                fence = [line]
                continue
            buf.append(line)
            if _word_length("\n".join(buf)) > _settings.chunk_size_markdown_words:
                documents.extend(self._flush(buf, metadata))
                buf = []
        if fence is not None:  # unterminated fence at EOF
            buf.extend(fence)
        documents.extend(self._flush(buf, metadata))
        return documents

    def _flush(self, lines: list[str], metadata: dict) -> list[Document]:
        """Split buffered lines to the cap, preserving fenced blocks verbatim."""
        text = "\n".join(lines).strip("\n")
        if not text.strip():
            return []
        if _word_length(text) <= _settings.chunk_size_markdown_words:
            return [Document(page_content=text, metadata=metadata)]
        if "```" not in text and "~~~" not in text:
            return self._splitter.split_documents(
                [Document(page_content=text, metadata=metadata)]
            )
        # Mixed prose+fences within the cap: split prose, re-emit fences whole.
        documents: list[Document] = []
        prose: list[str] = []
        in_fence_lines: list[str] = []
        in_fence = False
        for line in text.splitlines():
            stripped = line.strip()
            if not in_fence and (
                stripped.startswith("```") or stripped.startswith("~~~")
            ):
                in_fence = True
                in_fence_lines.append(line)
                continue
            if in_fence:
                in_fence_lines.append(line)
                if stripped and stripped.startswith(in_fence_lines[0].strip()[:3]):
                    if prose:
                        documents.extend(
                            self._splitter.split_documents(
                                [
                                    Document(
                                        page_content="\n".join(prose), metadata=metadata
                                    )
                                ]
                            )
                        )
                        prose = []
                    documents.append(
                        Document(
                            page_content="\n".join(in_fence_lines), metadata=metadata
                        )
                    )
                    in_fence_lines = []
                    in_fence = False
                continue
            prose.append(line)
            if _word_length("\n".join(prose)) > _settings.chunk_size_markdown_words:
                documents.extend(
                    self._splitter.split_documents(
                        [Document(page_content="\n".join(prose), metadata=metadata)]
                    )
                )
                prose = []
        if prose:
            documents.extend(
                self._splitter.split_documents(
                    [Document(page_content="\n".join(prose), metadata=metadata)]
                )
            )
        if in_fence_lines:  # unterminated: emit as-is
            documents.append(
                Document(page_content="\n".join(in_fence_lines), metadata=metadata)
            )
        return documents

    def _split_oversized_fence(self, block: str) -> list[str]:
        """Chop an oversized fenced block into cap-sized line groups.

        Every fragment re-opens (and closes) the fence so each chunk stays
        syntactically valid Markdown and reads standalone.
        """
        lines = block.splitlines()
        open_marker = lines[0].strip() if lines else "```"
        fence_marker = open_marker[:3]
        has_close = bool(lines) and lines[-1].strip().startswith(fence_marker)
        close_marker = lines[-1].strip() if has_close else fence_marker
        body = lines[1:-1] if has_close else lines[1:]

        pieces: list[str] = []
        current: list[str] = []
        words = 0
        for line in body:
            line_words = _word_length(line)
            if current and words + line_words > _settings.chunk_size_markdown_words:
                pieces.append("\n".join(current))
                current, words = [], 0
            current.append(line)
            words += line_words
        if current:
            pieces.append("\n".join(current))
        return [f"{open_marker}\n{piece}\n{close_marker}" for piece in pieces]


class PdfChunker(ChunkingStrategy):
    """Recursive splitting applied per page-document (one Document per page)."""

    supported_suffixes = frozenset({".pdf"})

    def __init__(self) -> None:
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=_settings.chunk_size_pdf_words,
            chunk_overlap=_settings.chunk_overlap_pdf_words,
            length_function=_word_length,
            keep_separator=False,
        )

    def split_documents(self, documents: list[Document]) -> list[Document]:
        return self._splitter.split_documents(documents)


class CodeChunker(ChunkingStrategy):
    """Language-aware splitting for source files (prefers def/class boundaries)."""

    supported_suffixes = frozenset({".py"})

    def __init__(self) -> None:
        self._splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON,
            chunk_size=_settings.chunk_size_code_words,
            chunk_overlap=_settings.chunk_overlap_code_words,
            length_function=_word_length,
        )

    def split_documents(self, documents: list[Document]) -> list[Document]:
        return self._splitter.split_documents(documents)


class JsonChunker(ChunkingStrategy):
    """One chunk per top-level JSON key; oversized subtrees get split further."""

    supported_suffixes = frozenset({".json"})

    def __init__(self) -> None:
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=_settings.chunk_size_json_words,
            chunk_overlap=0,
            length_function=_word_length,
            separators=["\n\n", "\n", ", ", " "],
        )

    def split_documents(self, documents: list[Document]) -> list[Document]:
        chunks: list[Document] = []
        for doc in documents:
            data = json.loads(doc.page_content)
            if isinstance(data, dict) and data:
                for key, value in data.items():
                    text = json.dumps({key: value}, indent=2, ensure_ascii=False)
                    metadata = {**doc.metadata, "json_key": key}
                    if _word_length(text) > _settings.chunk_size_json_words:
                        chunks.extend(
                            self._splitter.create_documents(
                                [text], metadatas=[metadata]
                            )
                        )
                    else:
                        chunks.append(Document(page_content=text, metadata=metadata))
            else:
                # Non-dict root (array/scalar) or empty object: text splitting.
                chunks.extend(self._splitter.split_documents([doc]))
        return chunks


CHUNKING_STRATEGIES: tuple[type[ChunkingStrategy], ...] = (
    MarkdownChunker,
    PdfChunker,
    CodeChunker,
    JsonChunker,
    TextChunker,
)


def get_chunker(path: Path) -> ChunkingStrategy:
    """Return a chunking strategy instance registered for the file's suffix.

    Unknown suffixes fall back to TextChunker.
    """
    for strategy in CHUNKING_STRATEGIES:
        if strategy.supports(path):
            return strategy()
    return TextChunker()


def _source_of(document: Document) -> str:
    """Return the source path recorded on the document, or an empty string."""
    source = document.metadata.get("source", "")
    return str(source) if source else ""


def split_documents(documents: list[Document]) -> list[Document]:
    """Split loader documents into retrieval chunks using per-type strategies.

    Args:
        documents: Documents produced by load_documents(), each carrying a
            metadata["source"] path used for strategy dispatch.

    Use cases:
        - Replace a generic splitter between load_documents() and the vector store.

    Documents whose source maps to the same suffix are chunked together. A
    document that fails to chunk is reported and skipped so one broken file
    cannot abort the whole run.
    """
    by_suffix: dict[str, tuple[ChunkingStrategy, list[Document]]] = {}
    for document in documents:
        source = _source_of(document)
        suffix = Path(source).suffix.lower()
        if suffix not in by_suffix:
            # Path() of a bare suffix would lose it (".md" parses as a name),
            # so dispatch on the document's real source path.
            by_suffix[suffix] = (get_chunker(Path(source) if source else Path(".")), [])
        by_suffix[suffix][1].append(document)

    chunks: list[Document] = []
    for strategy, docs in by_suffix.values():
        source = Path(_source_of(docs[0])).name if docs else "<unknown>"
        try:
            produced = strategy.split_documents(docs)
        except Exception as exc:
            c.error(f"Failed to chunk {source}: {exc}")
            continue
        if not produced:
            c.warning(f"No chunks produced from {source}")
        chunks.extend(produced)
    return chunks

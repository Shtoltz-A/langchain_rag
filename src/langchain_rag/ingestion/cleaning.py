"""Content cleaning: normalization and per-chunk context prefixes.

Two pure, unit-testable stages wrap the loader→chunking pipeline:

    normalize_documents()   — document-level text hygiene (before chunking):
        collapse blank-line runs, trim per-line trailing whitespace, rejoin
        hyphen-wrapped words, and drop repeated headers/footers (PDF only).
    apply_context_prefixes() — chunk-level provenance line (after chunking):
        one prefix derived from metadata, e.g. "[Doc: pdf/Policy.pdf p.12]"
        or "[File: ops/runbook.md > Deployment]" (folder trail included).

Neither mutates its input Documents; both return new lists. Contextual
retrieval (LLM per-chunk summaries) is deliberately out of scope here
(roadmap Step 2.5).
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import ClassVar

from langchain_core.documents import Document

from ..config import Settings, get_settings

_settings: Settings = get_settings()

_TRAILING_WS = re.compile(r"[ \t]+$", re.MULTILINE)
_BLANK_RUN = re.compile(r"\n{3,}")
_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
_PAGE_SUFFIXES = frozenset({".pdf"})


def _source_suffix(document: Document) -> str:
    """Lowercased file suffix of the document's source path ("" if absent)."""
    source = document.metadata.get("source", "")
    return Path(str(source)).suffix.lower() if source else ""


def _basename(document: Document) -> str:
    """Folder-trail + basename of the source, relative to the index root.

    Prefers the loader's POSIX ``relative_path`` (e.g. "pdf/report.pdf") so
    the grouping folders are visible in context prefixes; falls back to the
    bare basename when the metadata is absent.
    """
    meta = document.metadata
    relative = str(meta.get("relative_path", ""))
    if relative:
        return relative
    source = meta.get("source", "")
    return Path(str(source)).name if source else "unknown"


def _is_pdf(document: Document) -> bool:
    return _source_suffix(document) in _PAGE_SUFFIXES


class _PdfLineFilter:
    """Detects header/footer lines repeated across PDF page-documents."""

    #: Min distinct pages a line must recur on to count as boilerplate.
    min_count: ClassVar[int] = 3
    #: Min fraction of pages a line must appear on within one source.
    min_ratio: ClassVar[float] = 0.6

    def __init__(self, texts: list[str]) -> None:
        # texts are the already-normalized, hyphen-joined page bodies so the
        # detector sees the lines that will actually be kept, not fragments.
        self._drop: set[str] = set()
        pages = len(texts)
        if pages < self.min_count:
            return
        counts: Counter[str] = Counter()
        for text in texts:
            seen = {ln.strip() for ln in text.splitlines() if ln.strip()}
            counts.update(seen)
        for line, count in counts.items():
            if count >= self.min_count and count / pages >= self.min_ratio:
                self._drop.add(line.lower())

    def filter_page(self, text: str) -> str:
        """Return page content with boilerplate lines removed."""
        if not self._drop:
            return text
        kept = [ln for ln in text.splitlines() if ln.strip().lower() not in self._drop]
        return "\n".join(kept)

    def has_rules(self) -> bool:
        return bool(self._drop)


def _normalize_text(text: str) -> str:
    """Per-document text hygiene shared by all types (no hyphenation fix)."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _TRAILING_WS.sub("", text)
    text = _BLANK_RUN.sub("\n\n", text)
    return text.strip()


def normalize_documents(documents: list[Document]) -> list[Document]:
    """Normalize extracted text in place for each document (before chunking).

    Args:
        documents: Loader-produced Documents.

    Use cases:
        - Clean PDF/page artifacts before splitting into chunks.

    Collapses blank-line runs and trailing whitespace for every document;
    rejoins hyphen-broken words and drops repeated headers/footers only for
    PDF sources (guarded so markdown `code-block` hyphens aren't mangled).
    """
    if not documents:
        return []
    if not _settings.cleaning_enabled:
        return list(documents)

    # First pass: per-document hygiene (+ PDF hyphenation). Boilerplate
    # detection needs the post-hyphen texts, so it runs per PDF source after.
    texts: list[str] = []
    for doc in documents:
        text = _normalize_text(doc.page_content)
        if _is_pdf(doc):
            text = _HYPHEN_BREAK.sub(r"\1\2", text)
        texts.append(text)

    # Second pass: PDF-only header/footer dedup, per source file.
    filters: dict[str, _PdfLineFilter] = {}
    by_source: dict[str, list[int]] = {}
    for idx, doc in enumerate(documents):
        if _is_pdf(doc):
            by_source.setdefault(str(doc.metadata.get("source", "")), []).append(idx)
    for source, indices in by_source.items():
        filters[source] = _PdfLineFilter([texts[i] for i in indices])

    cleaned: list[Document] = []
    for doc, text in zip(documents, texts):
        if _is_pdf(doc):
            pdf_filter = filters[str(doc.metadata.get("source", ""))]
            if pdf_filter.has_rules():
                text = pdf_filter.filter_page(text)
        cleaned.append(Document(page_content=text, metadata=dict(doc.metadata)))
    return cleaned


def _context_prefix(document: Document) -> str:
    """Build the one-line context prefix for a chunk from its metadata.

    The file part includes the folder trail relative to the index root
    ("pdf/report.pdf"), so directory grouping is visible in every chunk.
    """
    name = _basename(document)
    suffix = _source_suffix(document)
    meta = document.metadata
    if suffix == ".pdf":
        page = meta.get("page")
        page_str = f" p.{int(page) + 1}" if page is not None else ""
        return f"[Doc: {name}{page_str}]"
    if suffix in {".md", ".markdown"}:
        trail = " > ".join(str(meta[k]) for k in ("h1", "h2", "h3") if meta.get(k))
        return f"[File: {name} > {trail}]" if trail else f"[File: {name}]"
    if suffix == ".json":
        key = meta.get("json_key")
        return f"[File: {name} \u00b7 key: {key}]" if key else f"[File: {name}]"
    return f"[File: {name}]"


def apply_context_prefixes(chunks: list[Document]) -> list[Document]:
    """Prepend a provenance line to each chunk (after chunking).

    Args:
        chunks: Retrieval chunks carrying source/heading/page/key metadata.

    Use cases:
        - Make each chunk self-describing in its embedded text.

    Idempotent: a chunk already starting with its own prefix is returned
    unchanged (no double-prefixing). Structured metadata is left intact
    alongside the prefix. Disabled when cleaning_enabled is False.
    """
    if not chunks or not _settings.cleaning_enabled:
        return list(chunks)
    result: list[Document] = []
    for chunk in chunks:
        prefix = _context_prefix(chunk)
        content = chunk.page_content
        if content.startswith(f"{prefix}\n"):
            result.append(chunk)
            continue
        result.append(
            Document(page_content=f"{prefix}\n{content}", metadata=dict(chunk.metadata))
        )
    return result

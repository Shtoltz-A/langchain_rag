"""Unit tests for ingestion.cleaning (pure logic: no Ollama, no Chroma)."""

from pathlib import Path

import pytest
from langchain_core.documents import Document

from langchain_rag.config import Settings
from langchain_rag.ingestion import cleaning
from langchain_rag.ingestion.cleaning import (
    apply_context_prefixes,
    normalize_documents,
)


def _doc(text: str, suffix: str, **metadata: object) -> Document:
    return Document(
        page_content=text, metadata={"source": f"some/dir/file{suffix}", **metadata}
    )


def _rel(text: str, rel: str, **metadata: object) -> Document:
    """Document as the loaders produce them: POSIX path relative to the root."""
    return Document(
        page_content=text,
        metadata={"source": f"some/dir/{rel}", "relative_path": rel, **metadata},
    )


def _prefixed(document: Document) -> str:
    """Content of a single chunk after the context-prefix stage."""
    return apply_context_prefixes([document])[0].page_content


def test_normalize_cleans_text_and_joins_pdf_hyphens() -> None:
    """Blank runs/whitespace for every type; hyphen-joining guarded to PDFs."""
    assert normalize_documents([]) == []

    doc = _doc("line1   \n\n\n\nline2\n\n\n", ".txt")
    assert normalize_documents([doc])[0].page_content == "line1\n\nline2"
    assert doc.page_content == "line1   \n\n\n\nline2\n\n\n"  # input not mutated

    crlf = _doc("a\r\nb\r\n\r\n\r\nc", ".txt")
    assert normalize_documents([crlf])[0].page_content == "a\nb\n\nc"

    pdf = _doc("this is a configu-\nration example", ".pdf", page=0)
    assert "configuration" in normalize_documents([pdf])[0].page_content

    # Markdown keeps `code-block` hyphens intact.
    md = _doc("a code-\nblock reference", ".md")
    assert "code-\nblock" in normalize_documents([md])[0].page_content


def test_pdf_boilerplate_lines_are_dropped_per_source() -> None:
    """Repeated headers/footers go; rare lines and short documents stay."""
    pages = [
        _doc(
            f"Confidential\n\npage {i} body text unique {i}\n\nPage 1 of 9",
            ".pdf",
            page=i,
        )
        for i in range(5)
    ]
    out = normalize_documents(pages)
    assert all("Confidential" not in d.page_content for d in out)
    assert all("Page 1 of 9" not in d.page_content for d in out)
    assert any("page 2 body text unique 2" in d.page_content for d in out)

    # "Intro line" is on 2 of 5 pages: below min_count and min_ratio, so it is
    # content and must be kept.
    pages = [
        _doc(
            f"Intro line\n\npage {i} body content different {i}"
            if i < 2
            else f"page {i} body content different {i}",
            ".pdf",
            page=i,
        )
        for i in range(5)
    ]
    assert any("Intro line" in d.page_content for d in normalize_documents(pages))

    # Below min_count pages the detector does not run at all.
    pages = [_doc(f"Shared Header\n\nbody {i}", ".pdf", page=i) for i in range(2)]
    assert all("Shared Header" in d.page_content for d in normalize_documents(pages))


def test_disabled_cleaning_passes_documents_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """cleaning_enabled=False: neither stage touches the documents."""
    settings = Settings().model_copy(update={"cleaning_enabled": False})
    monkeypatch.setattr(cleaning, "get_settings", lambda: settings)
    assert (
        normalize_documents([_doc("a\n\n\n\nb   ", ".pdf", page=0)])[0].page_content
        == "a\n\n\n\nb   "
    )
    assert _prefixed(_doc("body", ".txt")) == "body"


def test_stages_reread_settings_on_every_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Settings come from get_settings() at call time, not from import time."""
    doc = _doc("a\n\n\n\nb   ", ".txt")
    assert normalize_documents([doc])[0].page_content == "a\n\nb"

    settings = Settings().model_copy(update={"cleaning_enabled": False})
    monkeypatch.setattr(cleaning, "get_settings", lambda: settings)
    assert normalize_documents([doc])[0].page_content == "a\n\n\n\nb   "


def test_context_prefix_follows_the_file_type() -> None:
    """One provenance line per kind of source, metadata kept, paths shortened."""
    assert apply_context_prefixes([]) == []

    assert _prefixed(_doc("body", ".txt")) == "[File: file.txt]\nbody"
    assert _prefixed(_doc("body", ".py")) == "[File: file.py]\nbody"
    # PDFs use [Doc: …] and show the 1-based page number.
    assert _prefixed(_doc("body", ".pdf", page=11)) == "[Doc: file.pdf p.12]\nbody"
    # Markdown appends the heading chain; JSON the top-level key.
    assert (
        _prefixed(_doc("body", ".md", h1="Runbook", h2="Deployment"))
        == "[File: file.md > Runbook > Deployment]\nbody"
    )
    assert _prefixed(_doc("body", ".md")) == "[File: file.md]\nbody"
    assert (
        _prefixed(_doc("body", ".json", json_key="database"))
        == "[File: file.json \u00b7 key: database]\nbody"
    )
    assert (
        _prefixed(Document(page_content="body", metadata={})) == "[File: unknown]\nbody"
    )

    # Only the basename of the source path ever shows up.
    deep = Document(
        page_content="body",
        metadata={"source": str(Path("/deep/nested/dir/notes.txt"))},
    )
    assert _prefixed(deep) == "[File: notes.txt]\nbody"

    # Structured metadata stays intact next to the prefix.
    chunk = apply_context_prefixes([_doc("body", ".pdf", page=3)])[0]
    assert chunk.metadata["page"] == 3
    assert "source" in chunk.metadata


def test_prefix_shows_the_folder_trail_of_relative_path() -> None:
    """relative_path wins over the basename, so folder grouping stays visible."""
    assert (
        _prefixed(_rel("body", "pdf/report.pdf", page=0))
        == "[Doc: pdf/report.pdf p.1]\nbody"
    )
    assert (
        _prefixed(_rel("body", "md/runbook.md", h1="Runbook", h2="Deployment"))
        == "[File: md/runbook.md > Runbook > Deployment]\nbody"
    )
    assert _prefixed(_rel("body", "json/config.json", json_key="database")).startswith(
        "[File: json/config.json \u00b7 key: database]\n"
    )
    assert _prefixed(_rel("body", "repo/docs/deep.txt")) == (
        "[File: repo/docs/deep.txt]\nbody"
    )
    # Without relative_path the prefix falls back to the bare basename.
    assert _prefixed(_doc("body", ".txt")) == "[File: file.txt]\nbody"


def test_prefixes_are_idempotent() -> None:
    """A chunk that already carries its own prefix is returned unchanged."""
    for chunk in (_doc("body", ".txt"), _rel("body", "pdf/report.pdf", page=4)):
        once = apply_context_prefixes([chunk])[0]
        assert apply_context_prefixes([once])[0].page_content == once.page_content

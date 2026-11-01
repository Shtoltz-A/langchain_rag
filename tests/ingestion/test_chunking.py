"""Unit tests for ingestion.chunking (pure logic: no Ollama, no Chroma)."""

import json
from pathlib import Path

import pytest
from langchain_core.documents import Document

from langchain_rag.config import get_settings
from langchain_rag.ingestion.chunking import (
    ChunkingStrategy,
    CodeChunker,
    JsonChunker,
    MarkdownChunker,
    PdfChunker,
    TextChunker,
    _word_length,
    get_chunker,
    split_documents,
)

SOURCE = str(Path("fake/dir/example"))


def _doc(text: str, suffix: str, **metadata: object) -> Document:
    return Document(
        page_content=text, metadata={"source": f"{SOURCE}{suffix}", **metadata}
    )


def test_get_chunker_routes_by_suffix_with_text_fallback() -> None:
    """Registry dispatch on the suffix, TextChunker as the fallback strategy."""
    assert isinstance(get_chunker(Path("a/x.md")), MarkdownChunker)
    assert isinstance(get_chunker(Path("a/x.markdown")), MarkdownChunker)
    assert isinstance(get_chunker(Path("a/x.pdf")), PdfChunker)
    assert isinstance(get_chunker(Path("a/x.py")), CodeChunker)
    assert isinstance(get_chunker(Path("a/x.JSON")), JsonChunker)  # case-insensitive
    assert isinstance(get_chunker(Path("a/x.txt")), TextChunker)
    assert isinstance(get_chunker(Path("a/x.rst")), TextChunker)  # unknown suffix

    # A document with no source at all still chunks as plain text.
    chunks = split_documents([Document(page_content="hello world", metadata={})])
    assert [chunk.page_content for chunk in chunks] == ["hello world"]

    with pytest.raises(TypeError):
        ChunkingStrategy()  # type: ignore[abstract]


def test_every_strategy_preserves_parent_metadata() -> None:
    """Chunks carry the source through; per-type extras (page/h1-h3/json_key)."""
    docs = [
        _doc("some plain text here", ".txt"),
        _doc("# Title\n\nbody text", ".md"),
        _doc("page content words", ".pdf", page=3),
        _doc("def f():\n    return 1\n", ".py"),
        _doc('{"key": "value"}', ".json"),
    ]
    for doc in docs:
        chunks = get_chunker(Path(doc.metadata["source"])).split_documents([doc])
        assert chunks, doc.metadata["source"]
        assert all(
            chunk.metadata.get("source") == doc.metadata["source"] for chunk in chunks
        )

    # The 0-based page survives on every chunk of a page-document.
    page_chunks = PdfChunker().split_documents(
        [_doc("page with a decent amount of words to chunk", ".pdf", page=7)]
    )
    assert all(chunk.metadata["page"] == 7 for chunk in page_chunks)

    # Headings become h1..h3 metadata on the sections below them.
    md = "# Book\nintro\n\n## Chapter\nstory\n\n### Scene\naction words\n"
    scene = next(
        chunk
        for chunk in MarkdownChunker().split_documents([_doc(md, ".md")])
        if "action" in chunk.page_content
    )
    assert scene.metadata["h1"] == "Book"
    assert scene.metadata["h2"] == "Chapter"
    assert scene.metadata["h3"] == "Scene"

    # One chunk per top-level key, tagged with it.
    json_chunks = JsonChunker().split_documents(
        [_doc('{"alpha": {"x": 1}, "beta": [1, 2]}', ".json")]
    )
    assert [chunk.metadata["json_key"] for chunk in json_chunks] == ["alpha", "beta"]


def test_markdown_chunker_respects_fences_headings_and_cap() -> None:
    """Fence-aware headings, atomic fenced blocks, stub merging, prose size cap."""
    # A '#' inside a fenced block is code, never a heading.
    md = "# Real\n\ntext before\n\n```python\n# not a heading\nx = 1\n```\n\ntail\n"
    chunks = MarkdownChunker().split_documents([_doc(md, ".md")])
    assert "# not a heading" in "\n".join(c.page_content for c in chunks)
    assert all(c.metadata.get("h1") == "Real" for c in chunks)
    assert all("h2" not in c.metadata for c in chunks)

    # An oversized fence is chopped up, but no chunk ends inside a fence it did
    # not open, and every line of the original block survives.
    code = "\n".join(f"value_{i} = {i}" for i in range(400))
    chunks = MarkdownChunker().split_documents(
        [_doc(f"# Big\n\n```python\n{code}\n```\n", ".md")]
    )
    joined = "\n".join(c.page_content for c in chunks)
    assert all(f"value_{i} = {i}" in joined for i in range(400))
    for chunk in chunks:
        text = chunk.page_content
        assert not text.lstrip().startswith("```") or text.rstrip().endswith("```")

    # A heading stub joins a neighbouring section instead of chunking alone.
    md = "# Guide\n\n" + "word " * 50 + "\n\n## API\n\n## Commands\n\n" + "run " * 50
    chunks = MarkdownChunker().split_documents([_doc(md, ".md")])
    assert all(_word_length(c.page_content) >= 10 for c in chunks)
    assert "## API" in "\n".join(c.page_content for c in chunks)

    # Oversized prose is split down to the configured word cap.
    cap = get_settings().chunk_size_markdown_words
    chunks = MarkdownChunker().split_documents(
        [_doc("# Long\n\n" + " ".join(f"w{i}" for i in range(cap * 3)), ".md")]
    )
    assert len(chunks) > 1
    assert all(_word_length(c.page_content) <= cap for c in chunks)


def test_code_and_json_strategies_split_on_structure() -> None:
    """Python prefers def boundaries; JSON splits per key and per oversized key."""
    src = "\n\n".join(
        f"def func_{i}(x):\n    'Docstring number {i}.'\n    y = x + {i}\n    return y\n"
        for i in range(40)
    )
    chunks = CodeChunker().split_documents([_doc(src, ".py")])
    assert len(chunks) > 1
    joined = "\n".join(c.page_content for c in chunks)
    assert all(f"def func_{i}" in joined for i in range(40))
    assert any(c.page_content.lstrip().startswith("def ") for c in chunks)

    # An oversized subtree is split further; every fragment keeps its key.
    cap = get_settings().chunk_size_json_words
    big = _doc(json.dumps({"big": [f"item_{i}" for i in range(cap * 2)]}), ".json")
    chunks = JsonChunker().split_documents([big])
    assert len(chunks) > 1
    assert all(c.metadata["json_key"] == "big" for c in chunks)

    # A non-dict root (array/scalar/empty object) falls back to text splitting.
    chunks = JsonChunker().split_documents([_doc('["a", "b", "c"]', ".json")])
    assert chunks
    assert all("json_key" not in c.metadata for c in chunks)


def test_empty_and_broken_documents_never_abort_the_run() -> None:
    """Whitespace-only input chunks to nothing; one bad file cannot sink the rest."""
    assert TextChunker().split_documents([_doc("   \n\n  ", ".txt")]) == []

    good = _doc("fine plain text content", ".txt")
    broken = _doc("{ not valid json", ".json")
    chunks = split_documents([broken, good])
    assert chunks  # the good document survived the broken one
    assert all(_word_length(c.page_content) > 0 for c in chunks)

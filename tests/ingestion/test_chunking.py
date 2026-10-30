"""Unit tests for ingestion.chunking (pure logic: no Ollama, no Chroma)."""

from pathlib import Path

from langchain_core.documents import Document

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


def _doc(text: str, suffix: str, **metadata) -> Document:
    return Document(
        page_content=text, metadata={"source": f"{SOURCE}{suffix}", **metadata}
    )


class TestDispatch:
    def test_suffix_routes_to_strategy(self):
        assert isinstance(get_chunker(Path("a/x.md")), MarkdownChunker)
        assert isinstance(get_chunker(Path("a/x.markdown")), MarkdownChunker)
        assert isinstance(get_chunker(Path("a/x.pdf")), PdfChunker)
        assert isinstance(get_chunker(Path("a/x.py")), CodeChunker)
        assert isinstance(
            get_chunker(Path("a/x.JSON")), JsonChunker
        )  # case-insensitive
        assert isinstance(get_chunker(Path("a/x.txt")), TextChunker)

    def test_unknown_suffix_falls_back_to_text(self):
        assert isinstance(get_chunker(Path("a/x.rst")), TextChunker)

    def test_missing_source_falls_back_to_text(self):
        chunks = split_documents([Document(page_content="hello world", metadata={})])
        assert len(chunks) == 1
        assert chunks[0].page_content == "hello world"


class TestMetadata:
    def test_source_survives_every_strategy(self):
        docs = [
            _doc("some plain text here", ".txt"),
            _doc("# Title\n\nbody text", ".md"),
            _doc("page content words", ".pdf", page=3),
            _doc("def f():\n    return 1\n", ".py"),
            _doc('{"key": "value"}', ".json"),
        ]
        for doc in docs:
            strategy = get_chunker(Path(doc.metadata["source"]))
            chunks = strategy.split_documents([doc])
            assert chunks, doc.metadata["source"]
            assert all(
                chunk.metadata.get("source") == doc.metadata["source"]
                for chunk in chunks
            )

    def test_pdf_page_metadata_propagates(self):
        doc = _doc("page with a decent amount of words to chunk", ".pdf", page=7)
        chunks = PdfChunker().split_documents([doc])
        assert all(chunk.metadata["page"] == 7 for chunk in chunks)

    def test_markdown_headings_become_h1_h3_metadata(self):
        md = "# Book\nintro\n\n## Chapter\nstory\n\n### Scene\naction words\n"
        chunks = MarkdownChunker().split_documents([_doc(md, ".md")])
        scene = [c for c in chunks if "action" in c.page_content][0]
        assert scene.metadata["h1"] == "Book"
        assert scene.metadata["h2"] == "Chapter"
        assert scene.metadata["h3"] == "Scene"

    def test_json_key_metadata(self):
        doc = _doc('{"alpha": {"x": 1}, "beta": [1, 2]}', ".json")
        chunks = JsonChunker().split_documents([doc])
        assert [c.metadata["json_key"] for c in chunks] == ["alpha", "beta"]
        assert all("source" in c.metadata for c in chunks)


class TestMarkdownFences:
    def test_hash_inside_fence_is_not_a_heading(self):
        md = "# Real\n\ntext before\n\n```python\n# not a heading\nx = 1\n```\n\ntail\n"
        chunks = MarkdownChunker().split_documents([_doc(md, ".md")])
        joined = "\n".join(c.page_content for c in chunks)
        assert "# not a heading" in joined
        assert all(c.metadata.get("h1") == "Real" for c in chunks)
        assert all("h2" not in c.metadata for c in chunks)

    def test_fence_never_split_mid_block(self):
        code = "\n".join(f"value_{i} = {i}" for i in range(400))
        md = f"# Big\n\n```python\n{code}\n```\n"
        chunks = MarkdownChunker().split_documents([_doc(md, ".md")])
        starts = [c.page_content.lstrip().startswith("```") for c in chunks]
        ends = [c.page_content.rstrip().endswith("```") for c in chunks]
        # Each fragment carved out of the oversized fence keeps every line.
        joined = "\n".join(c.page_content for c in chunks)
        for i in range(400):
            assert f"value_{i} = {i}" in joined
        # No chunk ends inside a fence it started outside of.
        for started, ended in zip(starts, ends):
            assert started == ended or not started

    def test_heading_stub_merged_not_emitted_alone(self):
        md = (
            "# Guide\n\n" + "word " * 50 + "\n\n## API\n\n## Commands\n\n" + "run " * 50
        )
        chunks = MarkdownChunker().split_documents([_doc(md, ".md")])
        assert all(_word_length(c.page_content) >= 10 for c in chunks)
        joined = "\n".join(c.page_content for c in chunks)
        assert "## API" in joined  # stub heading text preserved, just not alone

    def test_size_cap_respected_for_prose(self):
        from langchain_rag.config import get_settings

        cap = get_settings().chunk_size_markdown_words
        md = "# Long\n\n" + " ".join(f"w{i}" for i in range(cap * 3))
        chunks = MarkdownChunker().split_documents([_doc(md, ".md")])
        assert len(chunks) > 1
        assert all(_word_length(c.page_content) <= cap for c in chunks)


class TestCodeChunker:
    def test_splits_on_def_boundaries(self):
        src = "\n\n".join(
            f"def func_{i}(x):\n    'Docstring number {i}.'\n    y = x + {i}\n    return y\n"
            for i in range(40)
        )
        chunks = CodeChunker().split_documents([_doc(src, ".py")])
        assert len(chunks) > 1
        joined = "\n".join(c.page_content for c in chunks)
        for i in range(40):
            assert f"def func_{i}" in joined
        # At least some chunks start at a function boundary.
        assert any(c.page_content.lstrip().startswith("def ") for c in chunks)


class TestJsonChunker:
    def test_oversized_subtree_split_further(self):
        from langchain_rag.config import get_settings

        cap = get_settings().chunk_size_json_words
        big = {"big": [f"item_{i}" for i in range(cap * 2)]}
        import json as _json

        doc = _doc(_json.dumps(big), ".json")
        chunks = JsonChunker().split_documents([doc])
        assert len(chunks) > 1
        assert all(c.metadata["json_key"] == "big" for c in chunks)

    def test_non_dict_root_falls_back_to_text_split(self):
        doc = _doc('["a", "b", "c"]', ".json")
        chunks = JsonChunker().split_documents([doc])
        assert chunks
        assert all("json_key" not in c.metadata for c in chunks)


class TestTolerance:
    def test_empty_document_yields_no_chunks_without_raising(self):
        assert TextChunker().split_documents([_doc("   \n\n  ", ".txt")]) == []

    def test_broken_document_skipped_in_split_documents(self):
        good = _doc("fine plain text content", ".txt")
        broken = _doc("{ not valid json", ".json")
        chunks = split_documents([broken, good])
        assert chunks  # good one survived
        assert all(_word_length(c.page_content) > 0 for c in chunks)

    def test_strategy_base_is_abstract(self):
        import pytest

        with pytest.raises(TypeError):
            ChunkingStrategy()  # type: ignore[abstract]

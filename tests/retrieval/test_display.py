"""Unit tests for the compact chunk rendering behind the verbose tables."""

from __future__ import annotations

from langchain_core.documents import Document

from langchain_rag.retrieval.display import PREVIEW_WIDTH, doc_label, doc_preview


def _doc(text: str = "body", **meta: object) -> Document:
    return Document(page_content=text, metadata=dict(meta))


class TestDocLabel:
    def test_prefers_the_root_relative_path(self) -> None:
        doc = _doc(relative_path="pdf/report.pdf", source=r"C:\indexed\pdf\report.pdf")
        assert doc_label(doc) == "pdf/report.pdf"

    def test_falls_back_to_the_source_basename(self) -> None:
        assert doc_label(_doc(source=r"C:\docs\x.txt")) == "x.txt"

    def test_appends_the_one_based_page_when_present(self) -> None:
        # Matches the cleaning prefix: 0-based metadata page 2 shows as p.3.
        assert doc_label(_doc(source="r.pdf", page=2)) == "r.pdf p.3"

    def test_document_without_any_source_metadata(self) -> None:
        assert doc_label(_doc()) == "unknown"


class TestDocPreview:
    def test_collapses_whitespace(self) -> None:
        assert doc_preview(_doc("a\n\n  b\t c")) == "a b c"

    def test_skips_the_context_prefix_line(self) -> None:
        doc = _doc("[File: x.txt · section: intro]\nActual body.")
        assert doc_preview(doc) == "Actual body."

    def test_brackets_later_in_the_text_are_kept(self) -> None:
        assert doc_preview(_doc("Body [note] end")) == "Body [note] end"

    def test_truncates_with_an_ellipsis(self) -> None:
        preview = doc_preview(_doc("x" * 200), width=50)
        assert len(preview) == 50
        assert preview.endswith("…")

    def test_default_width_is_the_named_constant(self) -> None:
        assert len(doc_preview(_doc("y" * 500))) == PREVIEW_WIDTH

    def test_whitespace_only_content_shows_the_marker(self) -> None:
        assert doc_preview(_doc("   \n ")) == "<empty>"

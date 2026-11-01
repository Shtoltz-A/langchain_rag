"""Unit tests for the compact chunk rendering behind the verbose tables."""

from __future__ import annotations

from langchain_core.documents import Document

from langchain_rag.retrieval.display import PREVIEW_WIDTH, doc_label, doc_preview


def _doc(text: str = "body", **meta: object) -> Document:
    return Document(page_content=text, metadata=dict(meta))


class TestDocLabel:
    def test_prefers_relative_path_then_source_and_appends_page(self) -> None:
        doc = _doc(relative_path="pdf/report.pdf", source=r"C:\indexed\pdf\report.pdf")
        assert doc_label(doc) == "pdf/report.pdf"
        # No relative_path: fall back to the source basename …
        assert doc_label(_doc(source=r"C:\docs\x.txt")) == "x.txt"
        # … and append the 1-based page (matches the cleaning prefix:
        # 0-based metadata page 2 shows as p.3).
        assert doc_label(_doc(source="r.pdf", page=2)) == "r.pdf p.3"
        # No source metadata at all: a fixed placeholder.
        assert doc_label(_doc()) == "unknown"


class TestDocPreview:
    def test_skips_every_leading_bracket_line(self) -> None:
        assert doc_preview(_doc("[File: x.txt · section: intro]\nActual body.")) == (
            "Actual body."
        )
        assert (
            doc_preview(
                _doc("[Context: Greenhouse CO2 readings of Sol 512.]\nActual body.")
            )
            == "Actual body."
        )
        # Step 5 stacks its [Context: …] line on top of the cleaning prefix:
        # the preview must strip the whole run of leading bracket lines.
        doc = _doc(
            "[Context: Greenhouse CO2 readings of Sol 512.]\n"
            "[File: x.txt · section: intro]\nActual body."
        )
        assert doc_preview(doc) == "Actual body."
        # Brackets later in the text are kept.
        assert doc_preview(_doc("Body [note] end")) == "Body [note] end"

    def test_collapses_truncates_and_marks_empty(self) -> None:
        assert doc_preview(_doc("a\n\n  b\t c")) == "a b c"
        # Cut to width with an ellipsis …
        preview = doc_preview(_doc("x" * 200), width=50)
        assert len(preview) == 50
        assert preview.endswith("…")
        # … by default at the named constant …
        assert len(doc_preview(_doc("y" * 500))) == PREVIEW_WIDTH
        # … and whitespace-only content shows the marker.
        assert doc_preview(_doc("   \n ")) == "<empty>"

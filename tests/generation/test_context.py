"""Unit tests for answer-context assembly (pure functions)."""

from __future__ import annotations

from langchain_core.documents import Document

from langchain_rag.generation import apply_budget, format_context


def _doc(text: str) -> Document:
    return Document(page_content=text)


class TestFormatContext:
    def test_joins_chunks_with_blank_lines(self) -> None:
        assert format_context([_doc("a"), _doc("b")]) == "a\n\nb"

    def test_empty_chunk_list_gives_empty_context(self) -> None:
        assert format_context([]) == ""


class TestApplyBudget:
    def test_short_context_passes_through(self) -> None:
        assert apply_budget("short", 100) == "short"

    def test_long_context_is_cut_to_budget(self) -> None:
        assert apply_budget("x" * 300, 120) == "x" * 120

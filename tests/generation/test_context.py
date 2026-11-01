"""Unit tests for answer-context assembly (pure functions)."""

from __future__ import annotations

from langchain_core.documents import Document

from langchain_rag.generation import (
    apply_budget,
    build_budgeted_context,
    format_context,
)


def _doc(text: str) -> Document:
    return Document(page_content=text)


def test_format_context_joins_chunks_with_blank_lines() -> None:
    """Chunk contents are joined for the prompt; no chunks give empty context."""
    assert format_context([_doc("a"), _doc("b")]) == "a\n\nb"
    assert format_context([]) == ""


def test_apply_budget_cuts_context_to_the_char_budget() -> None:
    """Short contexts pass through; longer ones get a best-effort hard cut."""
    assert apply_budget("short", 100) == "short"
    assert apply_budget("x" * 300, 120) == "x" * 120


def test_build_budgeted_context_reports_un_cut_length() -> None:
    """Fitted context + raw length come back, so the caller can trace the cut."""
    docs = [_doc("a" * 100), _doc("b" * 100)]  # joined: 100 + 2 + 100 chars
    joined = "a" * 100 + "\n\n" + "b" * 100

    context, raw_chars = build_budgeted_context(docs, 12_000)
    assert context == joined
    assert raw_chars == len(joined)

    context, raw_chars = build_budgeted_context(docs, 120)
    assert context == joined[:120]
    assert raw_chars == len(joined)  # budget hit: raw length outlives the cut

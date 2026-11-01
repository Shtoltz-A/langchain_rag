"""Context assembly for the answer prompt: join chunks, fit the char budget.

Pure string functions — the console stays out of here;
``services/answering.answer()`` gets the un-cut length back from
``build_budgeted_context`` and traces the budget itself under ``--verbose``.
"""

from __future__ import annotations

from langchain_core.documents import Document


def format_context(chunks: list[Document]) -> str:
    """Join chunk contents with blank lines (the shape the prompt expects)."""
    return "\n\n".join(chunk.page_content for chunk in chunks)


def apply_budget(context: str, max_chars: int) -> str:
    """Cut the context to the character budget (best-effort hard cut)."""
    return context[:max_chars]


def build_budgeted_context(chunks: list[Document], max_chars: int) -> tuple[str, int]:
    """Join chunk contents and cut the result to the character budget.

    Use cases:
        - ``services/answering.answer()``: one call yields the prompt
          context; the *un-cut* length comes back too so the verbose trace
          can report ``raw → budget`` without re-joining — and the console
          stays out of this module.
    """
    context = format_context(chunks)
    return apply_budget(context, max_chars), len(context)

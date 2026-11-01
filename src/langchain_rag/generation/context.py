"""Context assembly for the answer prompt: join chunks, fit the char budget.

Pure string functions — the console stays out of here so ``app.answer()`` can
dump the *full* context under ``--verbose`` before the budget cut.
"""

from __future__ import annotations

from langchain_core.documents import Document


def format_context(chunks: list[Document]) -> str:
    """Join chunk contents with blank lines (the shape the prompt expects)."""
    return "\n\n".join(chunk.page_content for chunk in chunks)


def apply_budget(context: str, max_chars: int) -> str:
    """Cut the context to the character budget (best-effort hard cut)."""
    return context[:max_chars]

"""Answering workflow: retrieve → budgeted context → grounded answer LLM.

The query-side coordinator: retrieval lives in ``retrieval/``, the prompt and
the LLM in ``generation/`` — this module only wires them together and traces
the context budget. No Typer, no process exits (rule 6).
"""

from ..config import get_settings
from ..console import get_console
from ..generation import (
    ANSWER_SYSTEM_PROMPT,
    build_budgeted_context,
    describe_answer_backend,
    get_answer_llm,
    traced_invoke,
)
from ..retrieval import retrieve

c = get_console()


def answer(question: str) -> str:
    """Retrieve relevant chunks and answer using their contents."""
    relevant = retrieve(question)

    if not relevant:
        # Empty store *or* e.g. BM25 finding no term overlap with the query.
        return (
            "No relevant chunks found. If the index is empty, run the "
            "index command first."
        )

    budget = get_settings().max_context_chars
    context, raw_chars = build_budgeted_context(relevant, budget)
    if raw_chars > budget:
        c.trace("context", f"{raw_chars:,} → {budget:,} chars", "cut to budget")
    else:
        c.trace("context", f"{raw_chars:,} chars", f"budget {budget:,}")

    messages = [
        ("system", ANSWER_SYSTEM_PROMPT),
        # Delimiters mark the untrusted region so indexed text cannot forge a
        # "Question:" line and pose as the real query (anti-injection rule).
        ("human", f"<context>\n{context}\n</context>\n\nQuestion:\n{question}"),
    ]
    response = traced_invoke(
        "answer", describe_answer_backend(), get_answer_llm(), messages
    )
    return response.content or ""

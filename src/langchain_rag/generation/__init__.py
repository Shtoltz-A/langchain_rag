"""generation/ package: chat-model construction, invoke tracing, answer prompts.

Public API only (mirrors ``retrieval/``): application code imports from here;
the modules are implementation files.
"""

from .context import apply_budget, format_context
from .llm import (
    build_chat_model,
    describe_answer_backend,
    describe_model,
    get_answer_llm,
    get_rerank_chat_model,
    resolve_model,
    traced_invoke,
)
from .prompts import ANSWER_SYSTEM_PROMPT

__all__ = [
    "ANSWER_SYSTEM_PROMPT",
    "apply_budget",
    "build_chat_model",
    "describe_answer_backend",
    "describe_model",
    "format_context",
    "get_answer_llm",
    "get_rerank_chat_model",
    "resolve_model",
    "traced_invoke",
]

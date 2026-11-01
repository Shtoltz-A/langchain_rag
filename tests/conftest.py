"""Global test hygiene: clear every process-wide memoization around each test.

Settings and LLM clients are cached for the life of the process
(``get_settings``, ``generation.get_answer_llm``,
``generation.get_contextual_chat_model``, ``reranking._get_llm``).
Tests patch settings and clients, so a cache surviving one test would poison
the next — every test starts and ends with unbuilt caches.
"""

from __future__ import annotations

import pytest

from langchain_rag import config
from langchain_rag.generation import llm as generation_llm
from langchain_rag.retrieval import reranking


@pytest.fixture(autouse=True)
def _clear_caches() -> None:
    config.get_settings.cache_clear()
    generation_llm.get_answer_llm.cache_clear()
    generation_llm.get_contextual_chat_model.cache_clear()
    reranking._get_llm.cache_clear()
    yield
    config.get_settings.cache_clear()
    generation_llm.get_answer_llm.cache_clear()
    generation_llm.get_contextual_chat_model.cache_clear()
    reranking._get_llm.cache_clear()

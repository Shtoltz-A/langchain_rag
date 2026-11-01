"""Unit tests for ingestion.contextual (no Ollama, no network)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.documents import Document

from langchain_rag.config import Settings
from langchain_rag.ingestion import contextual
from langchain_rag.ingestion.contextual import (
    ContextualGenerationError,
    apply_contextual_summaries,
)
from langchain_rag.storage.summary_cache import (
    hash_text,
    load_summaries,
    save_summaries,
)


def _settings(**overrides: Any) -> Settings:
    """Real Settings with every contextual_* input the cache key hashes
    pinned explicitly, so a developer `.env` cannot shift prompt or key shape."""
    base: dict[str, Any] = {
        "contextual_retrieval_enabled": True,
        "contextual_provider": "ollama",
        "contextual_model": "sum-m",
        "contextual_source_max_chars": 3_000,
    }
    base.update(overrides)
    return Settings(**base)


class _FakeSummarizer:
    """Stands in for the contextual chat model: replays canned outputs.

    An ``Exception`` in ``outputs`` is raised when it is reached, which is how
    call failures are simulated. ``prompts`` records one entry per MISS —
    the traced_invoke seam passes the messages through untouched.
    """

    def __init__(self, outputs: list[Any]) -> None:
        self.outputs = list(outputs)
        self.prompts: list[Any] = []

    def invoke(self, messages: Any) -> Any:
        self.prompts.append(messages)
        if not self.outputs:
            raise AssertionError("fake summarizer ran out of canned outputs")
        output = self.outputs.pop(0)
        if isinstance(output, Exception):
            raise output
        return SimpleNamespace(content=output)


class _BuilderSpy:
    """Counts ``get_contextual_chat_model()`` calls; hands out the fake."""

    def __init__(self, llm: _FakeSummarizer | None) -> None:
        self.llm = llm
        self.calls = 0

    def __call__(self) -> Any:
        self.calls += 1
        if self.llm is None:
            raise AssertionError("contextual stage built an LLM in a no-LLM test")
        return self.llm


def _install(
    monkeypatch: pytest.MonkeyPatch,
    *,
    settings: Settings,
    llm: _FakeSummarizer | None = None,
) -> _BuilderSpy:
    """Point the contextual module's settings/client seams at the fakes.

    Patched where the names are *used* (the contextual module's own globals),
    mirroring tests/services/test_indexing.py. The backend description is
    stubbed so the
    traced lines never read the real ``.env`` either.
    """
    monkeypatch.setattr(contextual, "get_settings", lambda: settings)
    monkeypatch.setattr(
        contextual, "describe_contextual_backend", lambda: "ollama/sum-m"
    )
    builder = _BuilderSpy(llm)
    monkeypatch.setattr(contextual, "get_contextual_chat_model", builder)
    return builder


def _chunk(text: str, **metadata: Any) -> Document:
    return Document(page_content=text, metadata=dict(metadata))


def test_disabled_or_empty_input_touches_neither_llm_nor_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Switched off (or nothing to do) the stage is a pure passthrough."""
    cache = tmp_path / "chunk_summaries.json"
    fake = _FakeSummarizer(["unused"])
    spy = _install(
        monkeypatch,
        settings=_settings(contextual_retrieval_enabled=False),
        llm=fake,
    )
    chunks = [_chunk("alpha body"), _chunk("beta body")]

    out = apply_contextual_summaries(chunks, cache_path=cache)

    assert out is chunks  # same list, same objects, untouched
    assert [c.page_content for c in out] == ["alpha body", "beta body"]
    assert all("context_summary" not in c.metadata for c in out)
    assert spy.calls == 0 and fake.prompts == []
    assert not cache.exists()  # not even touched while disabled

    # Enabled but with nothing to summarize: still no client and no cache file.
    spy = _install(monkeypatch, settings=_settings())
    assert apply_contextual_summaries([], cache_path=cache) == []
    assert spy.calls == 0
    assert not cache.exists()


def test_summaries_are_generated_prepended_and_cached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Prompt shape, sanitized one-liners, metadata + [Context: …] line, cache."""
    settings = _settings(contextual_source_max_chars=200)
    cache = tmp_path / "chunk_summaries.json"
    long_body = "x" * 250 + "TAIL"
    fake = _FakeSummarizer(["  This  covers\n\tthe Sol \n 512 readings.", "y" * 400])
    spy = _install(monkeypatch, settings=settings, llm=fake)

    out = apply_contextual_summaries(
        [_chunk(long_body, relative_path="docs/a.txt"), _chunk("beta body")],
        cache_path=cache,
    )

    # One traced call per MISS: provenance line + body capped at the setting.
    (system_role, system), (human_role, human) = fake.prompts[0]
    assert (system_role, human_role) == ("system", "human")
    assert system  # anti-injection instruction present
    assert "<chunk>\n" + "x" * 200 + "\n</chunk>" in human
    assert "TAIL" not in human  # capped at contextual_source_max_chars
    assert "docs/a.txt" in human  # root-relative provenance line
    assert spy.calls == 1  # one client for the whole run, not per chunk
    assert len(fake.prompts) == 2

    # Whitespace collapses so the stored [Context: …] line stays single-line,
    # as doc_preview expects.
    assert out[0].metadata["context_summary"] == "This covers the Sol 512 readings."
    assert out[0].page_content == (
        f"[Context: This covers the Sol 512 readings.]\n{long_body}"
    )
    # The length cap is enforced, not requested.
    summary = out[1].metadata["context_summary"]
    assert len(summary) == contextual._SUMMARY_MAX_CHARS == 300
    assert summary.endswith("…")

    assert load_summaries(contextual._cache_key(settings), cache) == {
        hash_text(long_body): "This covers the Sol 512 readings.",
        hash_text("beta body"): summary,
    }

    # Idempotent: a second pass neither re-summarizes nor double-prepends.
    twice = apply_contextual_summaries(out, cache_path=cache)
    assert twice[0].page_content == out[0].page_content
    assert len(fake.prompts) == 2


def test_cached_summaries_are_reused_and_stale_keys_ignored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stored summary is replayed under its own key; other keys buy nothing."""
    settings = _settings()
    key = contextual._cache_key(settings)

    # Pure cache-hit run: no client is built (the spy explodes if one is).
    hit_cache = tmp_path / "hits.json"
    save_summaries({hash_text("alpha body"): "Cached summary."}, key, hit_cache)
    spy = _install(monkeypatch, settings=settings)

    out = apply_contextual_summaries([_chunk("alpha body")], cache_path=hit_cache)

    assert out[0].metadata["context_summary"] == "Cached summary."
    assert out[0].page_content == "[Context: Cached summary.]\nalpha body"
    assert spy.calls == 0

    # Entries stored under a stale key are invisible: the chunk is summarized
    # again and saved under the current key, so the next run is a hit.
    stale_cache = tmp_path / "stale.json"
    save_summaries(
        {hash_text("alpha body"): "Old model's summary."}, "stale-key", stale_cache
    )
    fake = _FakeSummarizer(["Fresh summary."])
    spy = _install(monkeypatch, settings=settings, llm=fake)

    out = apply_contextual_summaries([_chunk("alpha body")], cache_path=stale_cache)

    assert out[0].metadata["context_summary"] == "Fresh summary."
    assert spy.calls == 1
    assert len(fake.prompts) == 1
    assert load_summaries(key, stale_cache) == {
        hash_text("alpha body"): "Fresh summary."
    }


def test_blank_output_indexes_bare_while_a_failed_call_aborts_the_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The two failure classes stay apart: degrade one chunk vs retry the file."""
    settings = _settings()
    cache = tmp_path / "chunk_summaries.json"
    fake = _FakeSummarizer(["", "  \n\t ", "Summary of gamma."])
    _install(monkeypatch, settings=settings, llm=fake)
    chunks = [_chunk("alpha body"), _chunk("beta body"), _chunk("gamma body")]

    out = apply_contextual_summaries(chunks, cache_path=cache)  # no raise

    assert out[0].page_content == "alpha body"
    assert "context_summary" not in out[0].metadata
    assert out[1].page_content == "beta body"
    assert "context_summary" not in out[1].metadata
    # The other chunks are still summarized normally.
    assert out[2].page_content == "[Context: Summary of gamma.]\ngamma body"
    assert len(fake.prompts) == 3
    # Blanks are content, not summaries: only the real one gets cached.
    assert load_summaries(contextual._cache_key(settings), cache) == {
        hash_text("gamma body"): "Summary of gamma."
    }

    # A call failure aborts the file — but what succeeded before it is already
    # on disk, so the retry only pays for the chunks that never got a summary.
    failing_cache = tmp_path / "failing.json"
    fake = _FakeSummarizer(["Summary of alpha.", ConnectionError("ollama is down")])
    _install(monkeypatch, settings=settings, llm=fake)
    chunks = [_chunk("alpha body"), _chunk("beta body"), _chunk("gamma body")]

    with pytest.raises(ContextualGenerationError, match="1 generated / 0 cached"):
        apply_contextual_summaries(chunks, cache_path=failing_cache)

    assert len(fake.prompts) == 2  # stopped at the failing chunk
    assert load_summaries(contextual._cache_key(settings), failing_cache) == {
        hash_text("alpha body"): "Summary of alpha."
    }
    assert "context_summary" not in chunks[1].metadata
    assert "context_summary" not in chunks[2].metadata

    # Misconfiguration must abort loudly, NOT degrade into a
    # ContextualGenerationError that "retries" the file forever.
    broken_cache = tmp_path / "broken.json"

    def _boom() -> Any:
        raise ValueError(
            "contextual summaries uses the openai provider "
            "but OPENAI_BASE_URL is not set in .env"
        )

    monkeypatch.setattr(
        contextual, "describe_contextual_backend", lambda: "openai/<model-unset>"
    )
    monkeypatch.setattr(contextual, "get_contextual_chat_model", _boom)

    with pytest.raises(ValueError, match="OPENAI_BASE_URL") as excinfo:
        apply_contextual_summaries([_chunk("alpha body")], cache_path=broken_cache)

    assert not isinstance(excinfo.value, ContextualGenerationError)
    assert not broken_cache.exists()

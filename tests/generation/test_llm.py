"""Unit tests for provider dispatch and path factories (offline: no network)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from langchain_rag.errors import ConfigurationError
from langchain_rag.generation import llm
from langchain_rag.generation.llm import build_chat_model, describe_model, resolve_model


def _settings(**overrides: Any) -> SimpleNamespace:
    """Settings surface generation.llm actually touches."""
    base: dict[str, Any] = {
        "ollama_llm_model": "llama-test",
        "llm_provider": "ollama",
        "llm_model": None,
        "rerank_provider": "ollama",
        "rerank_model": None,
        "openai_model": "",
        "openai_base_url": "",
        "openai_api_key": None,
        "rerank_num_ctx": 16_384,
        "rerank_timeout": 120.0,
        "contextual_provider": "ollama",
        "contextual_model": None,
        "contextual_timeout": 60.0,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


_GATEWAY: dict[str, Any] = {
    "openai_model": "gw-model",
    "openai_base_url": "https://gateway.test/v1",
    "openai_api_key": SecretStr("k"),
}


def test_build_chat_model_forwards_provider_specific_kwargs() -> None:
    """Each provider receives exactly its own kwargs; unknown ones fail loudly."""
    model = build_chat_model("ollama", "m", temperature=0.0, num_ctx=8192, timeout=5)
    assert isinstance(model, ChatOllama)
    assert model.model == "m"
    assert model.temperature == 0.0
    assert model.num_ctx == 8192
    assert model.client_kwargs == {"timeout": 5}

    # unset kwargs stay untouched: the provider's own defaults win
    assert build_chat_model("ollama", "m").temperature is None

    model = build_chat_model(
        "openai",
        "m",
        temperature=0.0,
        timeout=7,
        max_retries=0,
        base_url="https://gateway.test/v1",
        api_key=SecretStr("k"),
    )
    assert isinstance(model, ChatOpenAI)
    assert model.model_name == "m"
    assert model.openai_api_base == "https://gateway.test/v1"
    assert model.max_retries == 0
    assert model.openai_api_key is not None
    assert model.openai_api_key.get_secret_value() == "k"

    with pytest.raises(ConfigurationError, match="provider"):
        build_chat_model("anthropic", "m")  # type: ignore[arg-type]


def test_resolve_model_and_describe_model() -> None:
    """Override wins; else the provider fallback; unset gateway model is loud."""
    settings = _settings(openai_model="gw-model")
    assert resolve_model("ollama", "picked", settings) == "picked"  # type: ignore[arg-type]
    assert resolve_model("openai", "picked", settings) == "picked"  # type: ignore[arg-type]
    assert resolve_model("ollama", None, _settings()) == "llama-test"  # type: ignore[arg-type]
    assert resolve_model("openai", None, settings) == "gw-model"  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match="OPENAI_MODEL"):
        resolve_model("openai", None, _settings())  # type: ignore[arg-type]

    # display form mirrors resolve_model but never raises
    assert describe_model("openai", "picked", settings) == "picked"  # type: ignore[arg-type]
    assert describe_model("openai", None, settings) == "gw-model"  # type: ignore[arg-type]
    assert describe_model("openai", None, _settings()) == "<model-unset>"  # type: ignore[arg-type]


def test_ollama_path_factories_pin_their_own_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """answer/rerank/contextual each build Ollama clients per their contract."""
    # answer: resolves the local model, keeps the provider's own temperature
    monkeypatch.setattr(llm, "get_settings", _settings)
    answer = llm.get_answer_llm()
    assert isinstance(answer, ChatOllama)
    assert answer.model == "llama-test"
    assert answer.temperature is None  # generation behavior stays untouched
    assert llm.get_answer_llm() is answer  # lazily built once

    # rerank: model override, zero temperature, pinned num_ctx and timeout
    monkeypatch.setattr(llm, "get_settings", lambda: _settings(rerank_model="rk"))
    rerank = llm.get_rerank_chat_model()
    assert isinstance(rerank, ChatOllama)
    assert rerank.model == "rk"
    assert rerank.temperature == 0.0
    assert rerank.num_ctx == 16_384
    assert rerank.client_kwargs == {"timeout": 120.0}

    # contextual: zero temperature (repeatable summaries), contextual_timeout
    # binds, and an explicit model override beats the local fallback
    monkeypatch.setattr(
        llm, "get_settings", lambda: _settings(contextual_model="sum-m")
    )
    contextual = llm.get_contextual_chat_model()
    assert isinstance(contextual, ChatOllama)
    assert contextual.model == "sum-m"
    assert contextual.temperature == 0.0
    assert contextual.client_kwargs == {"timeout": 60.0}
    assert llm.get_contextual_chat_model() is contextual  # lazily built once


def test_openai_path_factories_build_and_require_gateway_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """openai provider: gateway client when configured, else a ConfigurationError
    naming the missing .env key and the purpose (rerank's fail-open warns with
    it; the contextual probe aborts the whole index run on it)."""
    monkeypatch.setattr(
        llm, "get_settings", lambda: _settings(rerank_provider="openai", **_GATEWAY)
    )
    model = llm.get_rerank_chat_model()
    assert isinstance(model, ChatOpenAI)
    assert model.model_name == "gw-model"
    assert model.max_retries == 0  # rerank_timeout stays the real latency bound

    for provider_setting, getter, label in (
        ("rerank_provider", llm.get_rerank_chat_model, "reranking"),
        ("contextual_provider", llm.get_contextual_chat_model, "contextual summaries"),
    ):
        for missing, expected in (
            ("openai_base_url", "OPENAI_BASE_URL"),
            ("openai_api_key", "OPENAI_API_KEY"),
        ):
            overrides: dict[str, Any] = {provider_setting: "openai", **_GATEWAY}
            overrides[missing] = "" if missing == "openai_base_url" else None
            monkeypatch.setattr(llm, "get_settings", lambda o=overrides: _settings(**o))  # type: ignore[assignment]
            with pytest.raises(ConfigurationError, match=rf"{label}.*{expected}"):
                getter()


def test_traced_invoke_traces_measures_and_reraises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One trace line per call, sizes for both message shapes, reraise on error."""

    class _Msg:
        content = "abcde"

    log = _RecordingConsole()
    monkeypatch.setattr(llm, "c", log)

    # transparent passthrough: result untouched, messages passed as given;
    # ONE line for the whole call, and a bare-string result (no .content)
    # keeps the prompt-only size form
    runnable = _FakeRunnable(result=object())
    result = llm.traced_invoke("rerank", "openai/gw", runnable, [("human", "x" * 12)])
    assert result is runnable.result
    assert runnable.prompts == [[("human", "x" * 12)]]
    assert len(log.messages) == 1
    assert log.messages[0].startswith("LLM rerank · openai/gw · 12 chars in · ")

    # a result with .content shows prompt → response size
    llm.traced_invoke(
        "answer",
        "ollama/m",
        _FakeRunnable(result=SimpleNamespace(content="abcd")),
        [("human", "x" * 12)],
    )
    assert "12 → 4 chars" in log.messages[-1]

    # _chars tolerates BaseMessage-style objects, not just tuples
    llm.traced_invoke("t", "ollama/m", _FakeRunnable(), [_Msg()])
    assert "5 chars in" in log.messages[-1]

    # failure: error class and elapsed logged via debug, exception reraised
    with pytest.raises(ConnectionError, match="gateway down"):
        llm.traced_invoke(
            "answer",
            "ollama/m",
            _FakeRunnable(error=ConnectionError("gateway down")),
            [("human", "q")],
        )
    assert log.messages[-1].startswith("LLM[answer] !! ConnectionError after ")


class _RecordingConsole:
    """Collects debug/trace lines so the verbose output can be asserted."""

    def __init__(self) -> None:
        self.messages: list[str] = []

    def debug(self, message: str) -> None:
        self.messages.append(message)

    def trace(self, label: str, *parts: str, ok: bool = False) -> None:
        # Same join as Console.trace, minus the styling and timestamp.
        self.messages.append(" · ".join((label, *parts)))


class _FakeRunnable:
    def __init__(self, result: Any = "ok", error: Exception | None = None) -> None:
        self.result = result
        self.error = error
        self.prompts: list[Any] = []

    def invoke(self, messages: Any) -> Any:
        self.prompts.append(messages)
        if self.error is not None:
            raise self.error
        return self.result


def test_describe_backends_show_provider_and_resolved_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Display form of both paths, falling back without raising when unset."""
    monkeypatch.setattr(llm, "get_settings", _settings)
    assert llm.describe_answer_backend() == "ollama/llama-test"
    assert llm.describe_contextual_backend() == "ollama/llama-test"

    monkeypatch.setattr(llm, "get_settings", lambda: _settings(llm_provider="openai"))
    assert llm.describe_answer_backend() == "openai/<model-unset>"

    monkeypatch.setattr(
        llm, "get_settings", lambda: _settings(contextual_model="sum-m")
    )
    assert llm.describe_contextual_backend() == "ollama/sum-m"

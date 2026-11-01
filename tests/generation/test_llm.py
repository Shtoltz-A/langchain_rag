"""Unit tests for provider dispatch and path factories (offline: no network)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from langchain_rag.generation import llm
from langchain_rag.generation.llm import build_chat_model, describe_model, resolve_model


def _settings(**overrides: Any) -> SimpleNamespace:
    """Settings surface generation.llm actually touches."""
    base: dict[str, Any] = dict(
        OLLAMA_LLM_MODEL="llama-test",
        llm_provider="ollama",
        llm_model=None,
        rerank_provider="ollama",
        rerank_model=None,
        openai_model="",
        openai_base_url="",
        openai_api_key=None,
        rerank_num_ctx=16_384,
        rerank_timeout=120.0,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


class TestBuildChatModel:
    def test_ollama_gets_ollama_specific_kwargs(self) -> None:
        model = build_chat_model(
            "ollama", "m", temperature=0.0, num_ctx=8192, timeout=5
        )
        assert isinstance(model, ChatOllama)
        assert model.model == "m"
        assert model.temperature == 0.0
        assert model.num_ctx == 8192
        assert model.client_kwargs == {"timeout": 5}

    def test_ollama_keeps_provider_defaults_for_unset_kwargs(self) -> None:
        model = build_chat_model("ollama", "m")
        assert isinstance(model, ChatOllama)
        assert model.temperature is None  # provider default stays untouched

    def test_openai_gets_gateway_kwargs(self) -> None:
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

    def test_unknown_provider_raises(self) -> None:
        with pytest.raises(ValueError, match="provider"):
            build_chat_model("anthropic", "m")  # type: ignore[arg-type]


class TestResolveModel:
    def test_override_wins_on_both_providers(self) -> None:
        settings = _settings(openai_model="gw-model")
        assert resolve_model("ollama", "picked", settings) == "picked"  # type: ignore[arg-type]
        assert resolve_model("openai", "picked", settings) == "picked"  # type: ignore[arg-type]

    def test_ollama_falls_back_to_the_local_model(self) -> None:
        assert resolve_model("ollama", None, _settings()) == "llama-test"  # type: ignore[arg-type]

    def test_openai_falls_back_to_openai_model(self) -> None:
        settings = _settings(openai_model="gw-model")
        assert resolve_model("openai", None, settings) == "gw-model"  # type: ignore[arg-type]

    def test_openai_without_any_model_is_a_loud_misconfiguration(self) -> None:
        with pytest.raises(ValueError, match="OPENAI_MODEL"):
            resolve_model("openai", None, _settings())  # type: ignore[arg-type]

    def test_describe_model_never_raises(self) -> None:
        assert describe_model("openai", None, _settings()) == "<model-unset>"  # type: ignore[arg-type]


class TestPathFactories:
    def test_answer_path_defaults_to_ollama_without_temperature(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(llm, "get_settings", _settings)
        model = llm.get_answer_llm()
        assert isinstance(model, ChatOllama)
        assert model.model == "llama-test"
        assert model.temperature is None  # generation behavior stays untouched

    def test_answer_llm_is_built_once(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(llm, "get_settings", _settings)
        assert llm.get_answer_llm() is llm.get_answer_llm()

    def test_rerank_path_pins_zero_temperature_timeout_and_no_retries(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(llm, "get_settings", lambda: _settings(rerank_model="rk"))
        model = llm.get_rerank_chat_model()
        assert isinstance(model, ChatOllama)
        assert model.model == "rk"
        assert model.temperature == 0.0
        assert model.num_ctx == 16_384
        assert model.client_kwargs == {"timeout": 120.0}

    def test_openai_provider_builds_chatopenai(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            llm,
            "get_settings",
            lambda: _settings(
                rerank_provider="openai",
                openai_model="gw-model",
                openai_base_url="https://gateway.test/v1",
                openai_api_key=SecretStr("k"),
            ),
        )
        model = llm.get_rerank_chat_model()
        assert isinstance(model, ChatOpenAI)
        assert model.model_name == "gw-model"
        assert model.max_retries == 0  # rerank_timeout stays the real latency bound

    @pytest.mark.parametrize("missing", ["base_url", "api_key"])
    def test_openai_provider_without_gateway_config_names_the_setting(
        self, monkeypatch: pytest.MonkeyPatch, missing: str
    ) -> None:
        # Rerank's fail-open turns this ValueError into a warning that points
        # at the exact .env key to fix.
        overrides: dict[str, Any] = dict(
            rerank_provider="openai",
            openai_model="gw-model",
            openai_base_url="https://gateway.test/v1",
            openai_api_key=SecretStr("k"),
        )
        overrides["openai_base_url" if missing == "base_url" else "openai_api_key"] = (
            "" if missing == "base_url" else None
        )
        monkeypatch.setattr(llm, "get_settings", lambda: _settings(**overrides))
        expected = "OPENAI_BASE_URL" if missing == "base_url" else "OPENAI_API_KEY"
        with pytest.raises(ValueError, match=expected):
            llm.get_rerank_chat_model()


class _RecordingConsole:
    """Collects debug messages so trace lines can be asserted."""

    def __init__(self) -> None:
        self.messages: list[str] = []

    def debug(self, message: str) -> None:
        self.messages.append(message)


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


class TestTracedInvoke:
    def test_result_and_messages_pass_through_untouched(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        log = _RecordingConsole()
        monkeypatch.setattr(llm, "c", log)
        runnable = _FakeRunnable(result=object())

        result = llm.traced_invoke("answer", "ollama/m", runnable, [("human", "hi")])

        assert result is runnable.result
        assert runnable.prompts == [[("human", "hi")]]  # transparent passthrough

    def test_request_and_response_lines_carry_backend_and_prompt_size(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        log = _RecordingConsole()
        monkeypatch.setattr(llm, "c", log)

        llm.traced_invoke("rerank", "openai/gw", _FakeRunnable(), [("human", "x" * 12)])

        assert log.messages[0] == "LLM[rerank] -> openai/gw: prompt 12 chars"
        assert log.messages[1].startswith("LLM[rerank] <- response in ")

    def test_error_class_and_elapsed_are_logged_before_reraising(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        log = _RecordingConsole()
        monkeypatch.setattr(llm, "c", log)
        runnable = _FakeRunnable(error=ConnectionError("gateway down"))

        with pytest.raises(ConnectionError, match="gateway down"):
            llm.traced_invoke("answer", "ollama/m", runnable, [("human", "q")])

        assert log.messages[-1].startswith("LLM[answer] !! ConnectionError after ")

    def test_non_tuple_messages_are_measured_by_content(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # _chars must tolerate BaseMessage-style objects, not just tuples.
        log = _RecordingConsole()
        monkeypatch.setattr(llm, "c", log)

        class _Msg:
            content = "abcde"

        llm.traced_invoke("t", "ollama/m", _FakeRunnable(), [_Msg()])

        assert "prompt 5 chars" in log.messages[0]


class TestDescribeAnswerBackend:
    def test_ollama_shows_provider_and_resolved_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(llm, "get_settings", _settings)
        assert llm.describe_answer_backend() == "ollama/llama-test"

    def test_openai_falls_back_without_raising_when_model_unset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            llm, "get_settings", lambda: _settings(llm_provider="openai")
        )
        assert llm.describe_answer_backend() == "openai/<model-unset>"

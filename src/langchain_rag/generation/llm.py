"""Chat-model construction and invocation tracing.

Every chat client in the app is built here (conventions.md): the answer path
and the rerank path read their own ``*_provider``/model settings, so switching
either backend is a one-line ``.env`` change and no other module ever branches
on a provider. Ollama stays the code default so a fresh checkout never routes
to a corporate endpoint; `.env` opts into the gateway explicitly.

The invoke side is the same seam: ``traced_invoke()`` wraps every LLM call so
verbose mode shows the backend, prompt size, duration and failure class of
each one, without any call site owning its own timing/logging code.
"""

from __future__ import annotations

import time
from functools import lru_cache
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from ..config import Settings, fallback_chat_model, get_settings
from ..console import get_console

c = get_console()

Provider = Literal["ollama", "openai"]


def resolve_model(provider: Provider, override: str | None, settings: Settings) -> str:
    """Model name for a pipeline path: explicit override, else provider default.

    Provider defaults come from ``config.fallback_chat_model`` (shared with
    the index fingerprint so both hash/dispatch the *same* model). An unset
    gateway model is a misconfiguration: raise, don't guess — rerank turns
    this into its fail-open warning naming the missing setting.
    """
    if override:
        return override
    model = fallback_chat_model(provider, settings)
    if provider == "openai" and not model:
        raise ValueError("openai provider needs a model: set OPENAI_MODEL in .env")
    return model


def describe_model(provider: Provider, override: str | None, settings: Settings) -> str:
    """Display form of ``resolve_model`` for debug lines — never raises."""
    return override or fallback_chat_model(provider, settings) or "<model-unset>"


def build_chat_model(
    provider: Provider,
    model: str,
    *,
    temperature: float | None = None,
    num_ctx: int | None = None,
    timeout: float | None = None,
    max_retries: int | None = None,
    base_url: str | None = None,
    api_key: SecretStr | str | None = None,
) -> BaseChatModel:
    """Instantiate one chat model for ``provider`` (the only client factory).

    Optional kwargs are forwarded only when set, so each provider receives
    just the parameters it understands: ``num_ctx`` is Ollama-only (gateways
    manage context server-side) and ``base_url``/``api_key``/``max_retries``
    apply to the openai provider only.

    Raises:
        ValueError: unknown provider — a misconfiguration fails loudly, not
            silently reroutes (mirrors ``retrieval.pipeline.retrieve()``).
    """
    if provider == "ollama":
        kwargs: dict[str, Any] = {}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if num_ctx is not None:
            kwargs["num_ctx"] = num_ctx
        if timeout is not None:
            kwargs["client_kwargs"] = {"timeout": timeout}
        return ChatOllama(model=model, **kwargs)
    if provider == "openai":
        optional: dict[str, Any] = {}
        if temperature is not None:
            optional["temperature"] = temperature
        if timeout is not None:
            optional["timeout"] = timeout
        if max_retries is not None:
            optional["max_retries"] = max_retries
        if base_url:
            optional["base_url"] = base_url
        if api_key is not None:
            optional["api_key"] = api_key
        return ChatOpenAI(model=model, **optional)
    raise ValueError(f"Unknown llm provider {provider!r}: expected ollama or openai")


def _build_path_model(
    purpose: str,
    provider: Provider,
    model_override: str | None,
    *,
    temperature: float | None,
    num_ctx: int | None = None,
    timeout: float | None = None,
    max_retries: int | None = None,
) -> BaseChatModel:
    """Shared settings resolution for the two path factories (no drift)."""
    settings = get_settings()
    if provider == "openai":
        missing = [
            name
            for name, value in (
                ("OPENAI_BASE_URL", settings.openai_base_url),
                ("OPENAI_API_KEY", settings.openai_api_key),
            )
            if not value
        ]
        if missing:
            raise ValueError(
                f"{purpose} uses the openai provider but "
                f"{', '.join(missing)} is not set in .env"
            )
    model = resolve_model(provider, model_override, settings)
    if provider == "openai":
        built = build_chat_model(
            provider,
            model,
            temperature=temperature,
            timeout=timeout,
            max_retries=max_retries,
            base_url=settings.openai_base_url,
            api_key=settings.openai_api_key,
        )
    else:
        built = build_chat_model(
            provider,
            model,
            temperature=temperature,
            num_ctx=num_ctx,
            timeout=timeout,
        )
    c.debug(f"{purpose} client built: {provider}/{model}")
    return built


@lru_cache(maxsize=1)
def get_answer_llm() -> BaseChatModel:
    """Answer-path chat model, lazily built and cached (lazy-client rule).

    Deliberately sets no temperature: the model's own default is kept, so a
    provider switch cannot silently change generation behavior.
    """
    settings = get_settings()
    return _build_path_model(
        "answer generation",
        settings.llm_provider,
        settings.llm_model,
        temperature=None,
    )


def get_rerank_chat_model() -> BaseChatModel:
    """Rerank-path chat model for ``reranking._get_llm`` to wrap.

    Uncached on purpose: ``reranking._get_llm`` caches the structured-output
    wrapper built on top of this — one cache layer only. ``temperature=0.0``
    keeps scoring repeatable; ``timeout`` bounds the one big listwise call;
    ``max_retries=0`` keeps that timeout the real latency bound (the OpenAI
    SDK would otherwise retry twice, roughly tripling worst-case latency).
    """
    settings = get_settings()
    return _build_path_model(
        "reranking",
        settings.rerank_provider,
        settings.rerank_model,
        temperature=0.0,
        num_ctx=settings.rerank_num_ctx,
        timeout=settings.rerank_timeout,
        max_retries=0,
    )


@lru_cache(maxsize=1)
def get_contextual_chat_model() -> BaseChatModel:
    """Contextual-summary path chat model, lazily built and cached.

    No structured-output wrapper sits on top of this path, so — like
    ``get_answer_llm`` — the factory itself carries the one cache layer
    (``contextual`` imports and calls it directly; tests patch it there).
    ``temperature=0.0`` keeps summaries repeatable, ``max_retries=0`` keeps
    ``contextual_timeout`` the real latency bound — a summarize call that dies
    fails the *file* (manifest retry), and SDK retries would multiply the
    worst-case stall of every chunk in it.
    """
    settings = get_settings()
    return _build_path_model(
        "contextual summaries",
        settings.contextual_provider,
        settings.contextual_model,
        temperature=0.0,
        timeout=settings.contextual_timeout,
        max_retries=0,
    )


def describe_answer_backend() -> str:
    """Display form of the answer path's backend (``provider/model``).

    Reads the real settings itself so answer-side callers never touch provider
    settings directly — consumers stay provider-blind (conventions.md).
    """
    settings = get_settings()
    return (
        f"{settings.llm_provider}/"
        f"{describe_model(settings.llm_provider, settings.llm_model, settings)}"
    )


def describe_contextual_backend() -> str:
    """Display form of the contextual-summary path's backend (``provider/model``)."""
    settings = get_settings()
    return (
        f"{settings.contextual_provider}/"
        f"{describe_model(settings.contextual_provider, settings.contextual_model, settings)}"
    )


def traced_invoke(
    purpose: str, backend: str, runnable: Any, messages: list[Any]
) -> Any:
    """Invoke a chat (or structured-output) runnable with a verbose trace.

    The single logging seam around every LLM call the app makes: one styled
    completion line (backend, prompt→response size, duration) via
    ``Console.trace``, error class via ``Console.debug`` on failure — the
    trace appears exactly in verbose mode, and a successful call is one line.

    Deliberately a transparent passthrough: the result comes back untouched,
    which is what lets test fakes stand in for real runnables and makes this
    the one place call duration is measured.

    Args:
        purpose: label for the trace line (``answer``, ``rerank``).
        backend: display form of the backend, e.g. ``ollama/llama3.2``.
        runnable: anything with ``invoke(messages)``.
        messages: ``(role, content)`` tuples (LangChain's coerced form);
            objects with a ``.content`` attribute are measured too.
    """

    def _chars(message: Any) -> int:
        content = message[1] if isinstance(message, tuple) else message.content
        return len(str(content))

    prompt_chars = sum(_chars(message) for message in messages)
    start = time.perf_counter()
    try:
        result = runnable.invoke(messages)
    except Exception as exc:
        c.debug(
            f"LLM[{purpose}] !! {type(exc).__name__} after "
            f"{time.perf_counter() - start:.1f}s"
        )
        raise
    # Response size only where there is text to size: structured-output
    # results carry no message .content, and str()-ing them would print noise.
    content = getattr(result, "content", None)
    size = (
        f"{prompt_chars:,} → {len(str(content)):,} chars"
        if content is not None
        else f"{prompt_chars:,} chars in"
    )
    c.trace(f"LLM {purpose}", backend, size, f"{time.perf_counter() - start:.1f}s")
    return result

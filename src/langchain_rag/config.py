from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """All runtime settings, read once through :func:`get_settings`.

    Field names are lower ``snake_case``; ``.env`` keys use
    SCREAMING_SNAKE_CASE and mirror them (``LLM_PROVIDER`` →
    ``llm_provider``) — pydantic-settings matches case-insensitively. The
    `.env` file ships every everyday setting — providers, models, retrieval
    mode, feature toggles; tuning-only settings (chunk sizes, fusion
    constants, prompt caps, timeouts) keep the defaults below and join `.env`
    only on deliberate override. Block order and field order here mirror
    `.env`.
    """

    # --- Models & providers (see generation/) ---------------------------------
    # Three chat paths (answer, rerank, contextual summarizer) each pick a
    # backend; per-provider model fallback lives in ``fallback_chat_model``.
    llm_provider: Literal["ollama", "openai"] = Field(
        default="ollama",
        description="Answer-LLM backend: local Ollama or OpenAI-compatible gateway",
    )
    llm_model: str | None = Field(
        default=None,
        description="Answer model override; falls back to ollama_llm_model "
        "(ollama) or openai_model (openai)",
    )
    rerank_provider: Literal["ollama", "openai"] = Field(
        default="ollama",
        description="Reranker backend; .env can point it at the gateway "
        "(small local models return empty score lists on big listwise prompts)",
    )
    rerank_model: str | None = Field(
        default=None,
        description="Rerank model override; falls back to ollama_llm_model "
        "(ollama) or openai_model (openai)",
    )
    ollama_llm_model: str = Field(
        default="llama3.2",
        description="Ollama chat model for every ollama-provider path "
        "(must be pulled in Ollama)",
    )
    ollama_embedding_model: str = Field(
        default="embeddinggemma",
        description="Ollama embedding model (indexing and query embedding)",
    )
    openai_model: str = Field(
        default="",
        description="Model name on the OpenAI-compatible gateway, used by "
        "every path whose provider is openai",
    )
    openai_base_url: str = Field(
        default="",
        description="OpenAI-compatible API root URL (…/v1); required when a "
        "path selects the openai provider",
    )
    openai_api_key: SecretStr | None = Field(
        default=None,
        description="Gateway API key (SecretStr: never rendered by repr/logs)",
    )

    @field_validator("openai_api_key", mode="before")
    @classmethod
    def _empty_api_key_is_unset(cls, value: object) -> object:
        """An empty ``OPENAI_API_KEY=`` placeholder in .env means "unset"."""
        return None if value == "" else value

    # --- Retrieval (see retrieval/) --------------------------------------------
    retrieval_mode: Literal["vector", "keyword", "hybrid"] = Field(
        default="vector",
        description="Candidate source: dense vectors, BM25 keywords, or RRF-fused hybrid",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        description="Number of chunks retrieved per query (0 would silently "
        "retrieve nothing)",
    )
    rerank_enabled: bool = False
    max_context_chars: int = Field(
        default=12_000,
        ge=1,
        description="Hard cap on the assembled context string before "
        "prompting (0 would silently empty it)",
    )
    rrf_c: int = Field(
        default=60,
        ge=1,
        description="RRF damping constant: score += weight / (rrf_c + rank)",
    )
    hybrid_vector_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    hybrid_bm25_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    rerank_candidate_multiplier: int = Field(
        default=3,
        ge=1,
        description="Candidates fetched (top_k × this) before reranking",
    )
    rerank_doc_max_chars: int = Field(
        default=2_000,
        ge=100,
        description="Per-candidate character cap in the rerank prompt",
    )
    rerank_num_ctx: int = Field(
        default=16_384,
        ge=2_048,
        description="Context window for the listwise rerank prompt (Ollama "
        "only; gateways manage context server-side)",
    )
    rerank_timeout: float = Field(
        default=120.0,
        gt=0,
        description="Seconds before the rerank call is abandoned (fail-open)",
    )

    # --- Contextual retrieval (see ingestion/contextual.py) --------------------
    # Indexing side: a one-sentence LLM summary is prepended to each chunk
    # before it is embedded. Deliberately not ``chunk_``-prefixed (the
    # fingerprint collects those automatically); the manifest fingerprints
    # these explicitly, and only while enabled, so a query-side
    # ``contextual_*`` setting can never trigger a needless rebuild.
    contextual_retrieval_enabled: bool = Field(
        default=False,
        description="Generate a per-chunk LLM context summary during indexing "
        "(one LLM call per new/changed chunk, cached by chunk hash)",
    )
    contextual_provider: Literal["ollama", "openai"] = Field(
        default="ollama",
        description="Summarizer backend (used only while contextual retrieval "
        "is enabled)",
    )
    contextual_model: str | None = Field(
        default=None,
        description="Summary model override; falls back to ollama_llm_model "
        "(ollama) or openai_model (openai)",
    )
    contextual_source_max_chars: int = Field(
        default=3_000,
        ge=200,
        description="Per-chunk character cap in the summarize prompt",
    )
    contextual_timeout: float = Field(
        default=60.0,
        gt=0,
        description="Seconds before a summarize call is abandoned (the file "
        "is then retried on the next index run)",
    )

    # --- Ingestion (see ingestion/) --------------------------------------------
    cleaning_enabled: bool = True
    chunk_size_text_words: int = 150
    chunk_overlap_text_words: int = 40
    chunk_size_markdown_words: int = 300
    chunk_overlap_markdown_words: int = 50
    chunk_size_pdf_words: int = 150
    chunk_overlap_pdf_words: int = 40
    chunk_size_code_words: int = 400
    chunk_overlap_code_words: int = 100
    chunk_size_json_words: int = 400

    # --- Console output (see console.py) ----------------------------------------
    verbose: bool = Field(
        default=True,
        description="Show debug output (retrieval lines, context dumps); "
        "the CLI --verbose flag overrides this per run",
    )

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached settings object.

    Caching avoids re-reading `.env` on every CLI/UI call.
    """
    return Settings()


def fallback_chat_model(provider: str, settings: Settings) -> str:
    """Provider-default chat model name for a path with no explicit override.

    ``ollama`` → ``ollama_llm_model``; anything else → ``openai_model`` (may
    be empty — an unset gateway model is the *caller's* error to raise:
    ``generation.resolve_model`` fails loudly, the index fingerprint just
    hashes what it gets). Lives here (not in ``generation/``) because both
    need it and this module must stay importable without LLM client classes.
    """
    if provider == "ollama":
        return settings.ollama_llm_model
    return settings.openai_model

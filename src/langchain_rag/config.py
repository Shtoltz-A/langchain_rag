from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    OLLAMA_LLM_MODEL: str = Field(
        default="llama3.2", description="Model name for generation"
    )
    OLLAMA_EMBEDDING_MODEL: str = Field(
        default="embeddinggemma",
        description="Ollama embedding model",
    )

    # Output settings (see console.py)
    verbose: bool = Field(
        default=True,
        description="Show debug output (retrieval lines, context dumps); "
        "the CLI --verbose flag overrides this per run",
    )

    # Chunking settings (word-based; see ingestion/chunking.py)
    chunk_size_text_words: int = 150
    chunk_overlap_text_words: int = 40
    chunk_size_markdown_words: int = 300
    chunk_overlap_markdown_words: int = 50
    chunk_size_pdf_words: int = 150
    chunk_overlap_pdf_words: int = 40
    chunk_size_code_words: int = 400
    chunk_overlap_code_words: int = 100
    chunk_size_json_words: int = 400

    # Retrieval settings (see retrieval/)
    top_k: int = 5
    max_context_chars: int = 12_000
    retrieval_mode: Literal["vector", "keyword", "hybrid"] = Field(
        default="vector",
        description="Candidate source: dense vectors, BM25 keywords, or RRF-fused hybrid",
    )
    rrf_c: int = Field(
        default=60,
        ge=1,
        description="RRF damping constant: score += weight / (rrf_c + rank)",
    )
    hybrid_vector_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    hybrid_bm25_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    rerank_enabled: bool = False
    rerank_candidate_multiplier: int = Field(
        default=3,
        ge=1,
        description="Candidates fetched (top_k × this) before reranking",
    )
    rerank_model: str | None = Field(
        default=None,
        description="Rerank model override; falls back to OLLAMA_LLM_MODEL "
        "(ollama) or openai_model (openai)",
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

    # Generation settings (see generation/)
    llm_provider: Literal["ollama", "openai"] = Field(
        default="ollama",
        description="Answer-LLM backend: local Ollama or OpenAI-compatible gateway",
    )
    llm_model: str | None = Field(
        default=None,
        description="Answer model override; falls back to OLLAMA_LLM_MODEL "
        "(ollama) or openai_model (openai)",
    )
    rerank_provider: Literal["ollama", "openai"] = Field(
        default="ollama",
        description="Reranker backend; .env can point it at the gateway "
        "(small local models return empty score lists on big listwise prompts)",
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

    # Cleaning settings (see ingestion/cleaning.py)
    cleaning_enabled: bool = True

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

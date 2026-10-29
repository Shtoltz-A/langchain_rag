from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    OLLAMA_LLM_MODEL: str = Field(
        default="llama3.2", description="Model name for generation"
    )
    OLLAMA_EMBEDDING_MODEL: str = Field(
        default="embeddinggemma",
        description="Ollama embedding model",
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

    # Retrieval settings
    top_k: int = 5
    max_context_chars: int = 12_000

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

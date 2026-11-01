"""Settings-level tests: gateway API key handling."""

from __future__ import annotations

from pydantic import SecretStr

from langchain_rag.config import Settings


class TestOpenAiApiKeySetting:
    def test_empty_string_is_treated_as_unset(self) -> None:
        # The `OPENAI_API_KEY=` placeholder in .env must mean "not
        # configured", not an empty-but-set key.
        assert Settings(OPENAI_API_KEY="").openai_api_key is None

    def test_key_round_trips_as_a_secret(self) -> None:
        key = Settings(OPENAI_API_KEY="sk-test").openai_api_key
        assert isinstance(key, SecretStr)
        assert key.get_secret_value() == "sk-test"
        assert "sk-test" not in repr(key)

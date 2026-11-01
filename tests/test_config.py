"""Settings-level tests: gateway API key handling and numeric bounds."""

from __future__ import annotations

import pytest
from pydantic import SecretStr, ValidationError

from langchain_rag.config import Settings


class TestOpenAiApiKeySetting:
    def test_empty_string_is_treated_as_unset(self) -> None:
        # The `OPENAI_API_KEY=` placeholder in .env must mean "not
        # configured", not an empty-but-set key.
        assert Settings(openai_api_key="").openai_api_key is None

    def test_key_round_trips_as_a_secret(self) -> None:
        key = Settings(openai_api_key="sk-test").openai_api_key
        assert isinstance(key, SecretStr)
        assert key.get_secret_value() == "sk-test"
        assert "sk-test" not in repr(key)


class TestRetrievalBounds:
    """Zero/negative retrieval knobs must fail at parse, not at query time."""

    @pytest.mark.parametrize("value", [0, -3])
    def test_top_k_must_be_positive(self, value: int) -> None:
        # TOP_K=0 would silently retrieve an empty context instead of erroring.
        with pytest.raises(ValidationError):
            Settings(top_k=value)

    @pytest.mark.parametrize("value", [0, -1])
    def test_max_context_chars_must_be_positive(self, value: int) -> None:
        with pytest.raises(ValidationError):
            Settings(max_context_chars=value)

    def test_positive_values_are_accepted(self) -> None:
        settings = Settings(top_k=1, max_context_chars=1)
        assert settings.top_k == 1
        assert settings.max_context_chars == 1

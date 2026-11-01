"""Offline CLI tests: directory validation and clean error exits.

Everything runs without a daemon, a vector store or an LLM: the service
functions and the Ollama probe are monkeypatched at their import site in
``cli`` (the module looks these names up on itself at call time). What the
probe itself catches and raises is tested in ``tests/services/test_health.py``;
here only how the CLI turns a probe verdict or a service error into one
terminal line plus an exit code.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from langchain_rag import cli
from langchain_rag.errors import ConfigurationError, OllamaUnavailableError

runner = CliRunner()


@pytest.fixture
def ollama_ok(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Neutralize the daemon probe (no network); record that it ran.

    Patched on ``cli``, never on ``services.health``: cli name-imports the
    function, so patching the health module would silently let tests hit
    the real network.
    """
    calls: list[str] = []
    monkeypatch.setattr(cli, "ensure_ollama_running", lambda: calls.append("probe"))
    return calls


class TestIndexClearSkipsDirectoryValidation:
    def test_clear_succeeds_for_a_directory_that_no_longer_exists(
        self, monkeypatch: pytest.MonkeyPatch, ollama_ok: list[str], tmp_path: Path
    ) -> None:
        cleared: list[str] = []

        def fake_clear_index() -> SimpleNamespace:
            cleared.append("clear")
            return SimpleNamespace(chunks_removed=3, folders_pruned=1)

        def never_index(directory: Path) -> None:
            raise AssertionError("--clear must never index")

        monkeypatch.setattr(cli, "clear_index", fake_clear_index)
        monkeypatch.setattr(cli, "index_documents", never_index)

        result = runner.invoke(
            cli.app,
            ["index", "--clear", "--directory", str(tmp_path / "already-deleted")],
        )

        assert result.exit_code == 0
        assert cleared == ["clear"]
        assert "Index cleared: 3 chunks removed" in result.output


class TestIndexDirectoryValidation:
    def test_missing_directory_exits_cleanly_before_contacting_ollama(
        self, monkeypatch: pytest.MonkeyPatch, ollama_ok: list[str], tmp_path: Path
    ) -> None:
        def never_index(directory: Path) -> None:
            raise AssertionError("a missing directory must not reach indexing")

        monkeypatch.setattr(cli, "index_documents", never_index)

        result = runner.invoke(
            cli.app, ["index", "--directory", str(tmp_path / "nope")]
        )

        assert result.exit_code == 1
        assert isinstance(result.exception, SystemExit)  # exit, not a traceback
        assert "does not exist" in result.output
        assert ollama_ok == []  # validation happens before the probe


class TestOllamaProbeFailure:
    """A failed probe must end the run with one error line, not a traceback."""

    @staticmethod
    def _probe_fails(monkeypatch: pytest.MonkeyPatch) -> None:
        def down() -> None:
            raise OllamaUnavailableError(
                "Ollama is not reachable. Start it with `ollama serve`."
            )

        monkeypatch.setattr(cli, "ensure_ollama_running", down)

    def test_query_reports_an_error_line_not_a_traceback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._probe_fails(monkeypatch)

        result = runner.invoke(cli.app, ["query", "anything"])

        assert result.exit_code == 1
        assert isinstance(result.exception, SystemExit)  # exit, not a traceback
        assert "Ollama is not reachable" in result.output

    def test_index_reports_an_error_line_and_never_indexes(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # The never-called guard proves the probe gates the workflow: a failed
        # probe must stop the run, not degrade into an empty or partial index.
        def never_index(directory: Path) -> None:
            raise AssertionError("a failed probe must not reach indexing")

        self._probe_fails(monkeypatch)
        monkeypatch.setattr(cli, "index_documents", never_index)

        result = runner.invoke(cli.app, ["index", "--directory", str(tmp_path)])

        assert result.exit_code == 1
        assert isinstance(result.exception, SystemExit)  # exit, not a traceback
        assert "Ollama is not reachable" in result.output


class TestIndexOverlapError:
    def test_service_value_error_exits_with_its_message(
        self, monkeypatch: pytest.MonkeyPatch, ollama_ok: list[str], tmp_path: Path
    ) -> None:
        def overlap_error(directory: Path) -> None:
            raise ValueError(
                "chunk_overlap_text_words (40) must stay below "
                "chunk_size_text_words (30)"
            )

        monkeypatch.setattr(cli, "index_documents", overlap_error)

        result = runner.invoke(cli.app, ["index", "--directory", str(tmp_path)])

        assert result.exit_code == 1
        assert isinstance(result.exception, SystemExit)  # exit, not a traceback
        assert "must stay below" in result.output


class TestConfigurationErrorExit:
    """The (RagError, ValueError) catch must cover the new domain error too."""

    def test_index_misconfiguration_exits_with_its_message(
        self, monkeypatch: pytest.MonkeyPatch, ollama_ok: list[str], tmp_path: Path
    ) -> None:
        def misconfigured(directory: Path) -> None:
            raise ConfigurationError(
                "contextual summaries uses the openai provider "
                "but OPENAI_BASE_URL is not set in .env"
            )

        monkeypatch.setattr(cli, "index_documents", misconfigured)

        result = runner.invoke(cli.app, ["index", "--directory", str(tmp_path)])

        assert result.exit_code == 1
        assert isinstance(result.exception, SystemExit)  # exit, not a traceback
        assert "OPENAI_BASE_URL is not set" in result.output

    def test_query_misconfiguration_exits_with_its_message(
        self, monkeypatch: pytest.MonkeyPatch, ollama_ok: list[str]
    ) -> None:
        def misconfigured(question: str) -> None:
            raise ConfigurationError(
                "openai provider needs a model: set OPENAI_MODEL in .env"
            )

        monkeypatch.setattr(cli, "answer_question", misconfigured)

        result = runner.invoke(cli.app, ["query", "anything"])

        assert result.exit_code == 1
        assert isinstance(result.exception, SystemExit)  # exit, not a traceback
        assert "OPENAI_MODEL" in result.output

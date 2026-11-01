"""Offline CLI tests: directory validation, Ollama probe, clean error exits.

Everything runs without a daemon, a vector store or an LLM: the service
functions and the Ollama client are monkeypatched at their import site in
``cli`` (the module looks these names up on itself at call time).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import typer
from ollama import ResponseError
from typer.testing import CliRunner

from langchain_rag import cli

runner = CliRunner()


class _RaisingClient:
    """Stands in for ``cli.Client``; ``list()`` fails with *error*."""

    def __init__(self, error: Exception) -> None:
        self._error = error

    def list(self) -> None:
        raise self._error


@pytest.fixture
def ollama_ok(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Neutralize the daemon probe (no network); record that it ran."""
    calls: list[str] = []
    monkeypatch.setattr(cli, "_ensure_ollama_running", lambda: calls.append("probe"))
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


class TestEnsureOllamaRunning:
    @pytest.mark.parametrize(
        "error",
        [
            ConnectionError("Ollama is not running"),  # daemon down
            ResponseError("daemon up but erroring", 500),  # answers with an error
        ],
    )
    def test_client_errors_become_exit_1(
        self, monkeypatch: pytest.MonkeyPatch, error: Exception
    ) -> None:
        monkeypatch.setattr(cli, "Client", lambda: _RaisingClient(error))

        with pytest.raises(typer.Exit) as excinfo:
            cli._ensure_ollama_running()

        assert excinfo.value.exit_code == 1

    def test_query_reports_an_error_line_not_a_traceback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            cli, "Client", lambda: _RaisingClient(ResponseError("bad state", 500))
        )

        result = runner.invoke(cli.app, ["query", "anything"])

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

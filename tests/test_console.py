"""Unit tests for the shared console factory and the CLI verbosity override."""

from __future__ import annotations

import io
import sys
from types import SimpleNamespace

import pytest

from langchain_rag import cli
from langchain_rag import console as console_module
from langchain_rag.console import get_console


@pytest.fixture
def fresh_console(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """Start from an unbuilt shared console; settings patched per test.

    monkeypatch restores the real `_console` on teardown, so other test
    modules never see the instance these tests build.
    """
    monkeypatch.setattr(console_module, "_console", None)
    return monkeypatch


def _settings_verbose(monkeypatch: pytest.MonkeyPatch, verbose: bool) -> None:
    """Pin the shared console's build-time verbosity setting."""
    monkeypatch.setattr(
        console_module, "get_settings", lambda: SimpleNamespace(verbose=verbose)
    )


class TestGetConsole:
    def test_built_once_from_the_setting(
        self, fresh_console: pytest.MonkeyPatch
    ) -> None:
        _settings_verbose(fresh_console, True)
        shared = get_console()
        assert shared.verbose is True
        assert get_console() is shared  # one instance for the whole process

        # A fresh process state re-reads the setting (the default is quiet).
        fresh_console.setattr(console_module, "_console", None)
        _settings_verbose(fresh_console, False)
        assert get_console().verbose is False


class TestCliVerboseFlag:
    def test_flag_overrides_the_setting_every_way(
        self, fresh_console: pytest.MonkeyPatch
    ) -> None:
        # (setting, --verbose flag, resulting console verbosity): an absent
        # flag (None) leaves the setting in charge, present flag wins either way.
        for setting, flag, expected in [
            (False, True, True),
            (True, False, False),
            (True, None, True),
        ]:
            fresh_console.setattr(console_module, "_console", None)
            _settings_verbose(fresh_console, setting)

            cli.main(verbose=flag)

            assert get_console().verbose is expected


class TestTrace:
    """Stage summaries print only in verbose mode, as one styled line."""

    def test_narrow_stream_encoding_replaces_glyphs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Piping into a cp1252 stream (Windows default) must not crash on ✓:
        # the glyph comes back as a ? replacement instead of UnicodeEncodeError.
        buffer = io.BytesIO()
        stream = io.TextIOWrapper(buffer, encoding="cp1252")
        monkeypatch.setattr(sys, "stdout", stream)

        console_module.Console(verbose=True, color=False)._echo("done ✓")

        stream.flush()
        assert b"done ?" in buffer.getvalue()

    def test_line_shown_only_when_verbose_and_ok_swaps_the_icon(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        console_module.Console(verbose=False, color=False).trace(
            "vector", "9/9 candidates", ok=True
        )
        assert capsys.readouterr().out == ""

        console_module.Console(verbose=True, color=False).trace(
            "vector", "9/9 candidates", "0.4s"
        )
        out = capsys.readouterr().out
        assert "›" in out
        assert "vector · 9/9 candidates · 0.4s" in out

        console_module.Console(verbose=True, color=False).trace(
            "05_orbit_solver.py", "2 chunks", "2.5s", ok=True
        )
        out = capsys.readouterr().out
        assert "✓" in out and "›" not in out

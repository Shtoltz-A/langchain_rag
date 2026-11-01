"""Unit tests for the shared console factory and the CLI verbosity override."""

from __future__ import annotations

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


class TestGetConsole:
    def test_builds_once_from_settings_verbose(
        self, fresh_console: pytest.MonkeyPatch
    ) -> None:
        fresh_console.setattr(
            console_module, "get_settings", lambda: SimpleNamespace(verbose=True)
        )
        shared = get_console()
        assert shared.verbose is True
        assert get_console() is shared  # one instance for the whole process

    def test_default_is_quiet(self, fresh_console: pytest.MonkeyPatch) -> None:
        fresh_console.setattr(
            console_module, "get_settings", lambda: SimpleNamespace(verbose=False)
        )
        assert get_console().verbose is False


class TestCliVerboseFlag:
    def test_flag_forces_debug_on(self, fresh_console: pytest.MonkeyPatch) -> None:
        fresh_console.setattr(
            console_module, "get_settings", lambda: SimpleNamespace(verbose=False)
        )
        cli.main(verbose=True)
        assert get_console().verbose is True

    def test_flag_off_forces_debug_off(self, fresh_console: pytest.MonkeyPatch) -> None:
        fresh_console.setattr(
            console_module, "get_settings", lambda: SimpleNamespace(verbose=True)
        )
        cli.main(verbose=False)
        assert get_console().verbose is False

    def test_absent_flag_leaves_setting_in_charge(
        self, fresh_console: pytest.MonkeyPatch
    ) -> None:
        fresh_console.setattr(
            console_module, "get_settings", lambda: SimpleNamespace(verbose=True)
        )
        cli.main(verbose=None)
        assert get_console().verbose is True

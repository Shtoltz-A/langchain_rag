"""Facade over the service layer — the stable import surface for CLI and tests.

The workflows and their value objects live in ``services/`` (``indexing``,
``answering``, ``models``); this module only re-exports them, so callers keep
a single entry point. No logic may live here (and, per rule 6, no Typer
anywhere below the CLI).
"""

from .services import ClearResult, IndexResult, answer, clear_index, index

__all__ = ["ClearResult", "IndexResult", "answer", "clear_index", "index"]

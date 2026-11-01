"""The package's public surface: workflows, their results, their errors.

Re-export only (rule 6) — the implementations live in ``services/`` and
``errors.py``. ``cli.py``, the tests and the future UI all import from here
(or from ``services/`` directly); there is no separate app-level facade.
"""

from .errors import ConfigurationError, OllamaUnavailableError, RagError
from .services import (
    ClearResult,
    IndexResult,
    answer,
    clear_index,
    ensure_ollama_running,
    index,
)

__all__ = [
    "ClearResult",
    "ConfigurationError",
    "IndexResult",
    "OllamaUnavailableError",
    "RagError",
    "answer",
    "clear_index",
    "ensure_ollama_running",
    "index",
]

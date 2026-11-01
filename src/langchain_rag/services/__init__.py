"""services/ package: the workflow coordinators behind the public API.

Public API only (mirrors ``retrieval/``): application code imports from here
or from the package root ``__init__.py``; the modules are implementation files.
"""

from .answering import answer
from .health import ensure_ollama_running
from .indexing import clear_index, index
from .models import ClearResult, IndexResult

__all__ = [
    "ClearResult",
    "IndexResult",
    "answer",
    "clear_index",
    "ensure_ollama_running",
    "index",
]

"""services/ package: the workflow coordinators behind the ``app`` facade.

Public API only (mirrors ``retrieval/``): application code imports from here
(or through the facade); the modules are implementation files.
"""

from .answering import answer
from .indexing import clear_index, index
from .models import ClearResult, IndexResult

__all__ = ["ClearResult", "IndexResult", "answer", "clear_index", "index"]

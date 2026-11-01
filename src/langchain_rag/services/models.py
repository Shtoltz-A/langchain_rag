"""Value objects shared by the ``services/`` workflows and their callers.

Pydantic, consistent with ``config.Settings`` and the LLM-stage schemas: the
counts are plain internal facts, so validation is only a safety net here —
what the models buy is immutability and ``model_dump()`` JSON readiness for
the future API surface (roadmap backlog).
"""

from pydantic import BaseModel, ConfigDict


class IndexResult(BaseModel):
    """Summary of one index run, for CLI reporting."""

    model_config = ConfigDict(frozen=True)

    chunks_stored: int
    files_indexed: int = 0
    files_unchanged: int = 0
    files_deleted: int = 0


class ClearResult(BaseModel):
    """Summary of one deliberate teardown, for CLI reporting."""

    model_config = ConfigDict(frozen=True)

    chunks_removed: int
    folders_pruned: int = 0

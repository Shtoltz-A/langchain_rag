"""Ingestion pipeline: turning source files into documents, chunks, and clean text.

Sub-modules (filled progressively per roadmap Step 2):
    loaders    — file-type loaders (.txt, .md, .pdf, .py, .json) producing Documents.
    cleaning   — normalization of extracted content.
    chunking   — per-content-type chunking strategies.
    contextual — per-chunk LLM context summaries (Step 5); the only LLM-touching
        module here, applied by ``services/indexing`` *after*
        ``process_file()`` so that
        function stays LLM-free and batch-parity holds.

This package also exposes the pipeline compositions: load_documents() for a
whole directory and process_file() for a single file (used by incremental
indexing), plus iter_supported_files() for per-file enumeration.
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document

from ..console import get_console
from .chunking import split_documents
from .cleaning import apply_context_prefixes, normalize_documents
from .contextual import ContextualGenerationError, apply_contextual_summaries
from .loaders import get_loader, iter_supported_files, load_documents

c = get_console()


def process_file(path: Path, root: Path) -> list[Document]:
    """Full single-file pipeline: load → normalize → split → context-prefix.

    Args:
        path: Single file to process; must have a supported suffix.
        root: Index root directory anchoring the ``relative_path`` metadata
            (same role as the directory argument of load_documents()).

    Use cases:
        - Incremental indexing: re-process only the files the manifest
          reports as new or changed, with results identical to the batch
          chain ``apply_context_prefixes(split_documents(normalize_documents(
          load_documents(root))))`` filtered per source.

    Never raises: an unsupported suffix produces a warning, any failure
    during load/normalize/split/prefix (bad encoding, corrupt PDF, vanished
    file) an error — both return ``[]`` so the caller simply leaves the
    file out of the manifest.
    """
    loader = get_loader(path)
    if loader is None:
        c.warning(f"Unsupported file type, skipping: {path}")
        return []
    try:
        # Consume inside the guard: lazy_load is a generator, so file
        # errors (encoding, corrupt PDF) surface at consumption time.
        # No per-stage lines: the caller reports the file's only summary
        # (chunks + time), so load/split counts would just repeat it.
        documents = loader(path, root=root).load()
        return apply_context_prefixes(split_documents(normalize_documents(documents)))
    except Exception as exc:
        c.error(f"Failed to process {path}: {exc}")
        return []


__all__ = [
    "ContextualGenerationError",
    "apply_context_prefixes",
    "apply_contextual_summaries",
    "iter_supported_files",
    "load_documents",
    "normalize_documents",
    "process_file",
    "split_documents",
]

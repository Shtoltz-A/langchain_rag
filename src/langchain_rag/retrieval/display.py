"""Compact one-line rendering of retrieved chunks for the verbose pipeline logs.

Pure display helpers shared by the candidate and rerank debug tables, so every
stage prints chunks the same way (label + preview) and the pipeline modules
stay about pipeline logic instead of string formatting.
"""

from __future__ import annotations

import re
from pathlib import Path

from langchain_core.documents import Document

#: Characters of chunk text shown per debug-table row.
PREVIEW_WIDTH = 70

# Cleaning prepends a "[File: ...]" / "[Doc: ... p.N]" context prefix line
# and the contextual stage prepends a "[Context: ...]" summary line
# (ingestion/cleaning.py, ingestion/contextual.py); ``doc_label`` already
# says the file's identity, so previews skip *every* leading bracket line
# instead of repeating it. Implemented locally (not by importing ingestion)
# to keep retrieval independent of the ingest package.
_PREFIX_LINE = re.compile(r"^(?:\[.*\]\n)+")


def doc_label(doc: Document) -> str:
    """Human identity of a chunk: root-relative path, plus page when known."""
    meta = doc.metadata
    relative = str(meta.get("relative_path") or "")
    if not relative:
        source = str(meta.get("source") or "")
        relative = Path(source).name if source else "unknown"
    page = meta.get("page")
    if page is not None:
        # Same 1-based display as the cleaning context prefix ("p.1" for the
        # 0-based metadata page) so the two never show conflicting numbers.
        return f"{relative} p.{int(page) + 1}"
    return relative


def doc_preview(doc: Document, width: int = PREVIEW_WIDTH) -> str:
    """Single-line snippet of chunk text: prefix line skipped, whitespace
    collapsed, cut to ``width`` with an ellipsis."""
    text = doc.page_content
    if text.startswith("["):
        text = _PREFIX_LINE.sub("", text, count=1)
    collapsed = " ".join(text.split())
    if len(collapsed) > width:
        return collapsed[: width - 1] + "…"
    return collapsed or "<empty>"

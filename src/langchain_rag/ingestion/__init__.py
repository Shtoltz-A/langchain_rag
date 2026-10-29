"""Ingestion pipeline: turning source files into documents, chunks, and clean text.

Sub-modules (filled progressively per roadmap Step 2):
    loaders    — file-type loaders (.txt, .md, .pdf, .py, .json) producing Documents.
    cleaning   — normalization of extracted content (Step 2.4).
    chunking   — per-content-type chunking strategies (Step 2.3).
"""

from .chunking import split_documents
from .cleaning import apply_context_prefixes, normalize_documents
from .loaders import load_documents

__all__ = [
    "load_documents",
    "normalize_documents",
    "split_documents",
    "apply_context_prefixes",
]

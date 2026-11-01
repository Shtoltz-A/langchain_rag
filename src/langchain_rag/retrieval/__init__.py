"""Query-side retrieval pipeline: candidates → RRF fusion → rerank.

Single entry point: ``retrieve(question)`` returns the chunks
``services/answering.answer()``
feeds to the LLM (implementation in ``pipeline.py``).
"""

from .pipeline import retrieve
from .reranking import rerank
from .retrievers import rrf_fuse

__all__ = ["rerank", "retrieve", "rrf_fuse"]

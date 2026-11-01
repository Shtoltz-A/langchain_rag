"""Storage layer: vector store persistence, the index manifest, and the
chunk-summary cache.

Sub-modules:
    vectorstore    — VectorStore singleton class (Chroma + embeddings, store/delete/search/count; clear sweeps leaked segment dirs).
    manifest       — tracked indexed sources and content hashes for incremental indexing.
    summary_cache  — content-keyed per-chunk LLM summaries for contextual retrieval.
"""

from .manifest import (
    Manifest,
    SourceRecord,
    compute_fingerprint,
    delete_manifest,
    diff,
    hash_file,
    load_manifest,
    matches,
    overlapping_root,
    root_key,
    save_manifest,
)
from .summary_cache import (
    SUMMARY_CACHE_PATH,
    hash_text,
    load_summaries,
    save_summaries,
)
from .vectorstore import VectorStore

__all__ = [
    "SUMMARY_CACHE_PATH",
    "Manifest",
    "SourceRecord",
    "VectorStore",
    "compute_fingerprint",
    "delete_manifest",
    "diff",
    "hash_file",
    "hash_text",
    "load_manifest",
    "load_summaries",
    "matches",
    "overlapping_root",
    "root_key",
    "save_manifest",
    "save_summaries",
]

"""Storage layer: vector store persistence and the index manifest (Step 3).

Sub-modules:
    vectorstore — VectorStore singleton class (Chroma + embeddings, store/delete/search/count).
    manifest    — tracked indexed sources and content hashes for incremental indexing.
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
from .vectorstore import VectorStore

__all__ = [
    "Manifest",
    "SourceRecord",
    "VectorStore",
    "compute_fingerprint",
    "diff",
    "hash_file",
    "load_manifest",
    "matches",
    "overlapping_root",
    "root_key",
    "save_manifest",
]

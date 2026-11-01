"""Chunk-summary cache: reusable one-sentence LLM contexts.

Contextual retrieval spends one LLM call per chunk per *content*; this cache
makes that cost pay once across index runs. Entries are keyed by the SHA-256
of the chunk text as the summarize stage receives it (after context prefixes,
before any summary), so identical chunk content — across files, roots, or a
rebuild — is summarized exactly once, mirroring how the manifest makes
unchanged *files* free.

The file carries a ``key`` over everything the summaries depend on (prompt
version, summarizer provider/model, input cap). A key mismatch — or any
unreadable/wrongly shaped file — yields an empty cache (every chunk is simply
re-summarized): a stale cache is a cost, never a correctness problem, so
there is no migration code (same philosophy as manifest.py).

Deliberately survives ``index --clear``: the whole-store teardown wipes
*vectors and manifest*, but content-keyed summaries stay valid and keep a
rebuild cheap.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path

from ..config import PROJECT_ROOT

SUMMARY_CACHE_PATH = PROJECT_ROOT / "data" / "chunk_summaries.json"


def hash_text(text: str) -> str:
    """SHA-256 hex digest of a string (utf-8) — the cache key for chunk text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_summaries(key: str, path: Path = SUMMARY_CACHE_PATH) -> dict[str, str]:
    """Read the cache; return the ``chunk-hash -> summary`` map for ``key``.

    Args:
        key: current ``_cache_key()`` from the summarize stage — entries saved
            under a different key (model/prompt change) are ignored.
        path: cache file location.

    Returns:
        The matching entries, or ``{}`` when the file is missing, corrupt,
        wrongly shaped, or stored under a different key.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(raw, dict) or raw.get("key") != key:
        return {}
    raw_summaries = raw.get("summaries")
    if not isinstance(raw_summaries, dict):
        return {}
    if not all(
        isinstance(digest, str) and isinstance(summary, str) and summary
        for digest, summary in raw_summaries.items()
    ):
        return {}
    return dict(raw_summaries)


def save_summaries(
    summaries: Mapping[str, str], key: str, path: Path = SUMMARY_CACHE_PATH
) -> None:
    """Write the cache atomically (temp file in the same dir + replace).

    Args:
        summaries: full ``chunk-hash -> summary`` map to persist.
        key: the ``_cache_key()`` these entries were generated under.
        path: cache file location; parent directories are created.
    """
    payload = {
        "key": key,
        "summaries": dict(sorted(summaries.items())),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp_path, path)

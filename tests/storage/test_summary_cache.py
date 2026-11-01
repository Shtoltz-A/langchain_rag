"""Unit tests for storage.summary_cache (pure file logic: no Ollama, no Chroma)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from langchain_rag.storage.summary_cache import (
    hash_text,
    load_summaries,
    save_summaries,
)

_KEY = "current-cache-key"


def _write_raw(path: Path, payload: object) -> Path:
    """Write a raw JSON payload (bypassing save_summaries) for loader tests."""
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_hash_text() -> None:
    """The cache key: utf-8 SHA-256, stable for equal text, sensitive to a byte."""
    assert hash_text("abc") == hashlib.sha256(b"abc").hexdigest()
    assert hash_text("same text") == hash_text("same text")
    assert hash_text("hello world") != hash_text("hello worlD")


def test_save_load_round_trip(tmp_path: Path) -> None:
    """save persists exactly {key, summaries}; load returns the map; a missing
    file simply reads as empty."""
    summaries = {hash_text("a"): "Summary A.", hash_text("b"): "Summary B."}
    path = tmp_path / "chunk_summaries.json"

    save_summaries(summaries, _KEY, path)

    assert load_summaries(_KEY, path) == summaries
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert set(payload) == {"key", "summaries"}
    assert payload["key"] == _KEY
    assert payload["summaries"] == summaries
    assert load_summaries(_KEY, tmp_path / "nope.json") == {}


def test_atomic_save(tmp_path: Path) -> None:
    """save creates missing parent directories and leaves no temp file behind."""
    path = tmp_path / "data" / "nested" / "chunk_summaries.json"

    save_summaries({"hash-a": "Summary A."}, _KEY, path)

    assert path.exists()
    assert load_summaries(_KEY, path) == {"hash-a": "Summary A."}
    assert list(path.parent.glob("*.tmp")) == []


def test_load_summaries_unusable_reads_empty(tmp_path: Path) -> None:
    """A stale or damaged cache is a cost, never a correctness problem: every
    unusable shape reads as ``{}`` (re-summarize), never raising."""
    # corrupt json / valid json but not a dict
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{not valid json", encoding="utf-8")
    assert load_summaries(_KEY, corrupt) == {}
    assert load_summaries(_KEY, _write_raw(tmp_path / "list.json", ["not", "d"])) == {}
    # entries saved under a different key (model/prompt change) must not leak
    assert (
        load_summaries(
            _KEY,
            _write_raw(
                tmp_path / "stale.json",
                {"key": "some-other-key", "summaries": {"hash-a": "Stale."}},
            ),
        )
        == {}
    )
    # summaries must be a dict of non-empty strings; anything else distrusts all
    for label, summaries in (
        ("not-a-dict", []),
        ("non-string", {"hash-a": 3}),
        ("empty-string", {"h": ""}),  # an empty [Context: ...] would be a lie
    ):
        assert (
            load_summaries(
                _KEY,
                _write_raw(
                    tmp_path / f"{label}.json", {"key": _KEY, "summaries": summaries}
                ),
            )
            == {}
        ), label

"""Unit tests for storage.manifest (pure logic: no Ollama, no Chroma)."""

from __future__ import annotations

import copy
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from langchain_rag.config import Settings
from langchain_rag.storage import manifest as manifest_mod
from langchain_rag.storage.manifest import (
    Manifest,
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

_DEFAULT_ROOT = Path("C:/index")


def _manifest(
    pairs: dict[str, tuple[str, int]] | None = None,
    root: Path = _DEFAULT_ROOT,
) -> Manifest:
    """Manifest holding {source: (sha256, chunks)} records under one root."""
    manifest = Manifest(fingerprint="fp", roots={})
    for source, (sha256, chunks) in (pairs or {}).items():
        manifest.record(root, source, sha256, chunks)
    return manifest


def _patch_settings(monkeypatch: pytest.MonkeyPatch, **updates: Any) -> None:
    """Point the module's get_settings at a real Settings with overrides."""
    settings = Settings().model_copy(update=updates)
    monkeypatch.setattr(manifest_mod, "get_settings", lambda: settings)


def _fingerprint(monkeypatch: pytest.MonkeyPatch, **updates: Any) -> str:
    """compute_fingerprint() under a fresh Settings with the given overrides."""
    _patch_settings(monkeypatch, **updates)
    return compute_fingerprint()


def _write_raw(path: Path, payload: object) -> Path:
    """Write a raw JSON payload (bypassing save_manifest) for loader tests."""
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_hash_file(tmp_path: Path) -> None:
    """Equal bytes hash equal; any differing byte (or tail) hashes different."""
    a, b, c = tmp_path / "a.txt", tmp_path / "b.txt", tmp_path / "c.bin"
    a.write_bytes(b"hello world")
    b.write_bytes(b"hello world")
    c.write_bytes(b"hello worlD")
    assert hash_file(a) == hash_file(b)
    assert hash_file(a) != hash_file(c)
    # > 64 KiB forces several read blocks: a tail-only change proves the whole
    # file (not just the first block) feeds the digest.
    payload = b"x" * (200 * 1024)
    big, big_tail = tmp_path / "big.bin", tmp_path / "tail.bin"
    big.write_bytes(payload)
    big_tail.write_bytes(payload[:-1] + b"y")
    assert hash_file(big) != hash_file(big_tail)


def test_save_load_delete_round_trip(tmp_path: Path) -> None:
    """save writes only {fingerprint, roots} atomically; load rebuilds; delete."""
    outer, inner = Path(r"C:\Data\Reports Q3"), Path(r"D:\归档")
    kept = rf"{outer}\q3 report.txt"
    manifest = _manifest({kept: ("aaa", 3)}, root=outer)
    manifest.record(inner, rf"{inner}\旧文件.txt", "bbb", 1)
    manifest.fingerprint = compute_fingerprint()
    path = tmp_path / "sub" / "index_manifest.json"  # parent gets created

    save_manifest(manifest, path)

    loaded = load_manifest(path)
    assert loaded == manifest  # Windows/backslash/unicode keys survive as opaque
    assert list(loaded.roots) == [root_key(outer), root_key(inner)]
    assert loaded.sources_for(outer)[kept].chunks == 3
    # payload carries no legacy single-root keys and no version field
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert set(payload) == {"fingerprint", "roots"}
    assert payload["fingerprint"] == manifest.fingerprint
    assert payload["roots"][root_key(outer)][kept] == {
        "sha256": "aaa",
        "chunks": 3,
        "indexed_at": manifest.sources_for(outer)[kept].indexed_at,
    }
    assert list(path.parent.glob("*.tmp")) == []  # atomic replace leaves no tmp
    assert delete_manifest(path) is True
    assert not path.exists()
    assert delete_manifest(path) is False  # already-missing file is tolerated


def test_delete_manifest_swallows_unremovable_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every unlink failure is swallowed like an absent file (True only on a
    real delete). clear_index() calls this AFTER wiping the store, so raising
    here would leave vectors gone + manifest alive; a surviving manifest
    self-heals via the rebuild gates instead."""
    missing = tmp_path / "missing.json"
    assert delete_manifest(missing) is False  # FileNotFoundError stays silent

    present = tmp_path / "index_manifest.json"
    present.write_text("{}", encoding="utf-8")

    def _locked(*args: object, **kwargs: object) -> None:
        raise PermissionError(32, "another process is using the file")

    monkeypatch.setattr(Path, "unlink", _locked)
    assert delete_manifest(present) is False  # e.g. held open by an editor
    assert present.exists()  # swallowed, not raised: the caller moves on

    def _is_a_directory(*args: object, **kwargs: object) -> None:
        raise IsADirectoryError(21, "Is a directory")

    monkeypatch.setattr(Path, "unlink", _is_a_directory)
    assert delete_manifest(present) is False


def test_load_manifest_unusable_returns_none(tmp_path: Path) -> None:
    """Every structurally unusable file yields None (caller rebuilds)."""
    assert load_manifest(tmp_path / "nope.json") is None  # missing (OSError)
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{not valid json", encoding="utf-8")
    assert load_manifest(corrupt) is None  # corrupt json (ValueError)
    assert (
        load_manifest(_write_raw(tmp_path / "list.json", ["not", "a", "dict"])) is None
    )  # valid json but not a dict
    # legacy single-root shape: no "roots" key -> structural failure
    assert (
        load_manifest(
            _write_raw(
                tmp_path / "legacy.json",
                {"version": 1, "fingerprint": "fp", "root": "C:/docs", "sources": {}},
            )
        )
        is None
    )
    # a root whose record is malformed (chunks is not an int) poisons the file
    assert (
        load_manifest(
            _write_raw(
                tmp_path / "bad-record.json",
                {
                    "fingerprint": "fp",
                    "roots": {"c:/index": {"a.txt": {"sha256": "aaa", "chunks": "3"}}},
                },
            )
        )
        is None
    )


def test_load_manifest_ignores_unknown_extra_keys(tmp_path: Path) -> None:
    """No schema version exists: extra keys must not invalidate a valid file."""
    loaded = load_manifest(
        _write_raw(
            tmp_path / "index_manifest.json",
            {"version": 2, "fingerprint": "fp", "roots": {}, "whatever": 1},
        )
    )
    assert loaded is not None
    assert loaded.roots == {}


def test_record_and_total_chunks() -> None:
    """record stores/replaces (fresh seconds-ISO stamp, creates roots on use);
    sources_for returns the live map (empty for unknown roots, no mutation);
    total_chunks sums across every root."""
    root = Path("C:/fresh")
    manifest = Manifest(fingerprint="fp", roots={})

    manifest.record(root, "a", "hash-a", 2)

    assert list(manifest.roots) == [root_key(root)]  # root created on first use
    record = manifest.sources_for(root)["a"]
    assert record.sha256 == "hash-a" and record.chunks == 2
    parsed = datetime.fromisoformat(record.indexed_at)
    assert parsed.isoformat(timespec="seconds") == record.indexed_at
    assert manifest.sources_for(root) is manifest.roots[root_key(root)]  # live map

    manifest.record(root, "a", "hash-a2", 7)  # replace keeps one entry
    assert manifest.sources_for(root)["a"].sha256 == "hash-a2"
    assert manifest.total_chunks() == 7

    manifest.record(Path("C:/other"), "b", "hash-b", 3)  # sums across roots
    assert manifest.total_chunks() == 10

    roots_before = copy.deepcopy(manifest.roots)
    assert manifest.sources_for(Path("Z:/never-indexed")) == {}
    assert manifest.roots == roots_before  # lookup never mutates


def test_remove() -> None:
    """remove drops a source, is tolerant of missing keys, and prunes the root
    once it owns nothing."""
    other = Path("C:/other")
    manifest = _manifest({"a": ("hash-a", 2), "b": ("hash-b", 3)})
    manifest.record(other, "c", "hash-c", 4)

    manifest.remove(_DEFAULT_ROOT, "a")
    assert list(manifest.sources_for(_DEFAULT_ROOT)) == ["b"]

    manifest.remove(_DEFAULT_ROOT, "ghost")  # missing source: no-op
    assert list(manifest.sources_for(_DEFAULT_ROOT)) == ["b"]

    manifest.remove(Path("Z:/ghost"), "c")  # missing root: no-op
    assert manifest.total_chunks() == 7

    manifest.remove(_DEFAULT_ROOT, "b")  # last source gone -> root pruned
    assert list(manifest.roots) == [root_key(other)]
    assert manifest.total_chunks() == 4


def test_diff_buckets_sorted_and_windows_keys_opaque(tmp_path: Path) -> None:
    """diff classifies each path into a sorted bucket; an empty previous counts
    everything as added; Windows/unicode keys are opaque and survive a round trip."""
    previous = _manifest({"a": ("old-a", 1), "b": ("b", 1), "c": ("c", 1)})
    current = {"a": "new-a", "b": "b", "d": "d"}
    result = diff(current, previous.sources_for(_DEFAULT_ROOT))
    assert (result.added, result.changed, result.deleted, result.unchanged) == (
        ["d"],
        ["a"],
        ["c"],
        ["b"],
    )

    result = diff(
        {"m": "3", "n": "4", "z": "9", "y": "8"},
        _manifest({"z": ("1", 1), "y": ("2", 1)}).sources_for(_DEFAULT_ROOT),
    )
    assert result.added == ["m", "n"]  # every bucket is sorted
    assert result.changed == ["y", "z"]
    assert result.deleted == [] and result.unchanged == []

    result = diff({"b": "2", "a": "1", "c": "3"}, {})  # never-indexed root
    assert result.added == ["a", "b", "c"]
    assert not (result.changed or result.deleted or result.unchanged)

    kept = r"C:\data\reports\q3 report.txt"
    changed = r"C:\data\notes.md"
    gone = r"D:\归档\旧文件.txt"
    manifest = _manifest({kept: ("same", 2), changed: ("old", 1), gone: ("x", 1)})
    result = diff({kept: "same", changed: "new"}, manifest.sources_for(_DEFAULT_ROOT))
    assert result.unchanged == [kept]
    assert result.changed == [changed]
    assert result.deleted == [gone]
    path = tmp_path / "manifest with space.json"
    save_manifest(manifest, path)
    assert load_manifest(path) == manifest


def test_fingerprint_tracks_pipeline_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stable across calls; changes with embedding model / cleaning / chunking /
    pipeline version; unchanged by retrieval-only knobs (contextual off)."""
    # Contextual pinned off so ollama_llm_model (the resolved summary model while
    # enabled) cannot leak a .env override into the retrieval-only assertions.
    off = {"contextual_retrieval_enabled": False}
    _patch_settings(monkeypatch, **off)
    assert compute_fingerprint() == compute_fingerprint()  # stable

    base = _fingerprint(monkeypatch, **off)
    assert (
        _fingerprint(monkeypatch, **off, ollama_embedding_model="another-embedder")
        != base
    )
    assert _fingerprint(monkeypatch, **off, cleaning_enabled=False) != base
    assert _fingerprint(monkeypatch, **off, chunk_size_text_words=999) != base

    # retrieval-only knobs (and ollama_llm_model while contextual is off) stay out
    for field, value in (
        ("top_k", 42),
        ("max_context_chars", 999),
        ("ollama_llm_model", "another-llm"),
    ):
        assert _fingerprint(monkeypatch, **off, **{field: value}) == base

    before = _fingerprint(monkeypatch, **off)
    monkeypatch.setattr(manifest_mod, "_PIPELINE_VERSION", 2)
    assert compute_fingerprint() != before  # a code bump invalidates the store


def test_fingerprint_contextual_retrieval(monkeypatch: pytest.MonkeyPatch) -> None:
    """Contextual settings join the payload only while enabled; enabling is
    itself the invalidation; the resolved summary model is tracked."""
    disabled = _fingerprint(monkeypatch, contextual_retrieval_enabled=False)
    # With the feature off no contextual block joins the payload, whatever the
    # contextual_* values say -> existing manifests stay byte-for-byte valid.
    assert (
        _fingerprint(
            monkeypatch,
            contextual_retrieval_enabled=False,
            contextual_provider="openai",
            contextual_model="other-model",
            contextual_source_max_chars=900,
        )
        == disabled
    )

    base = _fingerprint(monkeypatch, contextual_retrieval_enabled=True)
    assert base != disabled  # enabling changes stored chunk text
    for field, value in (
        ("contextual_model", "other-model"),
        ("contextual_provider", "openai"),
        ("contextual_source_max_chars", 900),
    ):
        assert (
            _fingerprint(
                monkeypatch, contextual_retrieval_enabled=True, **{field: value}
            )
            != base
        )
    # timeout changes no stored text, so it must not force a rebuild
    assert (
        _fingerprint(
            monkeypatch, contextual_retrieval_enabled=True, contextual_timeout=5.0
        )
        == base
    )
    # No override: the payload hashes the *resolved* model (fallback_chat_model),
    # so changing ollama_llm_model invalidates instead of silently mixing
    # summary generations.
    resolved = _fingerprint(
        monkeypatch, contextual_retrieval_enabled=True, contextual_model=None
    )
    assert (
        _fingerprint(
            monkeypatch,
            contextual_retrieval_enabled=True,
            contextual_model=None,
            ollama_llm_model="another-llm",
        )
        != resolved
    )


def test_root_key_is_normcase(tmp_path: Path) -> None:
    """root_key agrees exactly with os.path.normcase for any root spelling."""
    assert root_key(tmp_path) == os.path.normcase(str(tmp_path))
    for raw in ("C:/MixedCase/Docs", "C:\\MixedCase\\Docs"):
        root = Path(raw)
        assert root_key(root) == os.path.normcase(str(root))
    # case-equivalent roots fold to one key exactly as normcase does (full
    # folding into a single key on Windows).
    upper, lower = Path("C:/MixedCase/Docs"), Path("C:/mixedcase/docs")
    assert root_key(upper) == os.path.normcase(str(upper))
    assert root_key(lower) == os.path.normcase(str(lower))


def test_overlapping_root(tmp_path: Path) -> None:
    """Overlap (equal / containing / contained) returns the tracked key; a clean
    root or an empty manifest returns None."""
    assert (
        overlapping_root(Manifest(fingerprint="fp", roots={}), Path("C:/docs")) is None
    )

    equal = tmp_path / "docs"
    assert overlapping_root(
        _manifest({"a": ("aaa", 1)}, root=equal), equal
    ) == root_key(equal)

    outer = tmp_path / "data"  # candidate inside a tracked root
    assert overlapping_root(
        _manifest({"a": ("aaa", 1)}, root=outer), outer / "inner"
    ) == root_key(outer)

    inner = tmp_path / "proj" / "docs"  # candidate containing a tracked root
    assert overlapping_root(
        _manifest({"a": ("aaa", 1)}, root=inner), tmp_path / "proj"
    ) == root_key(inner)

    # "ab" shares a string prefix with "a" but is not inside it -> no conflict
    assert (
        overlapping_root(
            _manifest({"a": ("aaa", 1)}, root=tmp_path / "a"), tmp_path / "ab"
        )
        is None
    )


def test_matches() -> None:
    """matches compares only the fingerprint, never the tracked roots."""
    manifest = Manifest(fingerprint="fp", roots={})
    assert matches(manifest, fingerprint="fp") is True
    assert matches(manifest, fingerprint="other") is False
    tracked = _manifest({"a": ("hash-a", 1)}, root=Path("C:/some/root"))
    assert matches(tracked, fingerprint="fp") is True

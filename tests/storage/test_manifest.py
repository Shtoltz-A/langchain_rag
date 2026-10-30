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


def _write_raw(path: Path, payload: object) -> Path:
    """Write a raw JSON payload (bypassing save_manifest) for loader tests."""
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class TestHashFile:
    def test_same_content_same_digest(self, tmp_path: Path) -> None:
        a = tmp_path / "a.txt"
        b = tmp_path / "b.txt"
        a.write_bytes(b"hello world")
        b.write_bytes(b"hello world")
        assert hash_file(a) == hash_file(b)

    def test_one_byte_change_changes_digest(self, tmp_path: Path) -> None:
        a = tmp_path / "a.txt"
        b = tmp_path / "b.txt"
        a.write_bytes(b"hello world")
        b.write_bytes(b"hello worlD")
        assert hash_file(a) != hash_file(b)

    def test_multi_block_file_hashed_completely(self, tmp_path: Path) -> None:
        # > 64 KiB forces several read blocks: a tail-only change proves the
        # whole file (not just the first block) feeds the digest.
        payload = b"x" * (200 * 1024)
        a = tmp_path / "a.bin"
        b = tmp_path / "b.bin"
        a.write_bytes(payload)
        b.write_bytes(payload[:-1] + b"y")
        assert hash_file(a) != hash_file(b)


class TestSaveLoadRoundTrip:
    def test_round_trip_preserves_data(self, tmp_path: Path) -> None:
        other_root = Path("/srv/docs")
        manifest = _manifest({r"C:\data\a.txt": ("aaa", 3), "/data/b.md": ("bbb", 1)})
        manifest.record(other_root, "/srv/docs/note.md", "ccc", 2)
        manifest.fingerprint = compute_fingerprint()
        path = tmp_path / "sub" / "index_manifest.json"  # parent gets created
        save_manifest(manifest, path)
        loaded = load_manifest(path)
        assert loaded == manifest

    def test_multi_root_round_trip_preserves_windows_backslash_keys(
        self, tmp_path: Path
    ) -> None:
        outer = Path(r"C:\Data\Reports Q3")
        inner = Path(r"D:\归档")
        manifest = _manifest({rf"{outer}\q3 report.txt": ("aaa", 3)}, root=outer)
        manifest.record(inner, rf"{inner}\旧文件.txt", "bbb", 1)
        path = tmp_path / "manifest with space.json"

        save_manifest(manifest, path)

        loaded = load_manifest(path)
        assert loaded == manifest
        assert list(loaded.roots) == [root_key(outer), root_key(inner)]
        assert loaded.sources_for(outer)[rf"{outer}\q3 report.txt"].chunks == 3

    def test_save_emits_only_fingerprint_and_roots(self, tmp_path: Path) -> None:
        manifest = _manifest({"a.txt": ("aaa", 3)})
        path = tmp_path / "index_manifest.json"

        save_manifest(manifest, path)

        payload = json.loads(path.read_text(encoding="utf-8"))
        assert set(payload) == {"fingerprint", "roots"}  # no version field
        assert payload["fingerprint"] == "fp"
        assert payload["roots"] == {
            root_key(_DEFAULT_ROOT): {
                "a.txt": {
                    "sha256": "aaa",
                    "chunks": 3,
                    "indexed_at": manifest.sources_for(_DEFAULT_ROOT)[
                        "a.txt"
                    ].indexed_at,
                }
            }
        }
        # Legacy single-root keys must not leak into the output.
        assert "root" not in payload
        assert "sources" not in payload

    def test_no_tmp_file_remains_after_save(self, tmp_path: Path) -> None:
        path = tmp_path / "index_manifest.json"
        save_manifest(_manifest({"a": ("aaa", 1)}), path)
        assert path.exists()
        assert list(tmp_path.glob("*.tmp")) == []

    def test_record_timestamp_is_seconds_isoformat(self) -> None:
        manifest = _manifest({"a": ("aaa", 1)})
        record = manifest.sources_for(_DEFAULT_ROOT)["a"]
        parsed = datetime.fromisoformat(record.indexed_at)
        assert parsed.isoformat(timespec="seconds") == record.indexed_at


class TestDeleteManifest:
    def test_deletes_existing_file(self, tmp_path: Path) -> None:
        path = tmp_path / "index_manifest.json"
        save_manifest(_manifest({"a": ("1", 1)}), path=path)

        assert delete_manifest(path) is True
        assert not path.exists()

    def test_missing_file_is_tolerated(self, tmp_path: Path) -> None:
        assert delete_manifest(tmp_path / "nope.json") is False


class TestLoadManifestInvalid:
    def test_missing_file_returns_none(self, tmp_path: Path) -> None:
        assert load_manifest(tmp_path / "nope.json") is None

    def test_corrupt_json_returns_none(self, tmp_path: Path) -> None:
        path = tmp_path / "index_manifest.json"
        path.write_text("{not valid json", encoding="utf-8")
        assert load_manifest(path) is None

    def test_unknown_extra_keys_are_ignored(self, tmp_path: Path) -> None:
        # No schema version exists: a leftover "version" key (or any other
        # extra) must not invalidate a file whose actual structure is valid.
        path = _write_raw(
            tmp_path / "index_manifest.json",
            {"version": 2, "fingerprint": "fp", "roots": {}, "whatever": 1},
        )
        loaded = load_manifest(path)
        assert loaded is not None
        assert loaded.roots == {}

    def test_legacy_single_root_file_returns_none(self, tmp_path: Path) -> None:
        # Old v1 shape (root + flat sources): no "roots" key -> structural
        # failure -> the caller rebuilds; the project keeps no migration code.
        path = _write_raw(
            tmp_path / "index_manifest.json",
            {
                "version": 1,
                "fingerprint": "fp",
                "root": str(tmp_path / "docs"),
                "sources": {},
            },
        )
        assert load_manifest(path) is None

    def test_non_dict_json_returns_none(self, tmp_path: Path) -> None:
        path = tmp_path / "index_manifest.json"
        path.write_text('["not", "a", "dict"]', encoding="utf-8")
        assert load_manifest(path) is None

    def test_roots_with_invalid_record_returns_none(self, tmp_path: Path) -> None:
        path = _write_raw(
            tmp_path / "index_manifest.json",
            {
                "fingerprint": "fp",
                "roots": {"c:/index": {"a.txt": {"sha256": "aaa", "chunks": "3"}}},
            },
        )
        assert load_manifest(path) is None


class TestManifestMethods:
    def test_record_and_total_chunks(self) -> None:
        manifest = _manifest({"a": ("hash-a", 2), "b": ("hash-b", 3)})
        sources = manifest.sources_for(_DEFAULT_ROOT)
        assert sources["a"].sha256 == "hash-a"
        assert sources["a"].chunks == 2
        assert manifest.total_chunks() == 5

    def test_total_chunks_sums_across_roots(self) -> None:
        manifest = _manifest({"a": ("hash-a", 2), "b": ("hash-b", 3)})
        manifest.record(Path("C:/other"), "c", "hash-c", 4)
        assert manifest.total_chunks() == 9

    def test_record_replaces_existing(self) -> None:
        manifest = _manifest({"a": ("hash-a", 2)})
        manifest.record(_DEFAULT_ROOT, "a", "hash-a2", 7)
        assert manifest.sources_for(_DEFAULT_ROOT)["a"].sha256 == "hash-a2"
        assert manifest.total_chunks() == 7

    def test_record_creates_root_entry_on_first_use(self) -> None:
        manifest = Manifest(fingerprint="fp", roots={})
        root = Path("C:/fresh")

        manifest.record(root, "a", "hash-a", 2)

        assert list(manifest.roots) == [root_key(root)]
        assert manifest.sources_for(root)["a"].chunks == 2

    def test_sources_for_unknown_root_is_empty_without_mutation(self) -> None:
        manifest = _manifest({"a": ("hash-a", 2)})
        roots_before = copy.deepcopy(manifest.roots)

        assert manifest.sources_for(Path("Z:/never-indexed")) == {}

        assert manifest.roots == roots_before

    def test_sources_for_returns_the_live_root_map(self) -> None:
        manifest = _manifest({"a": ("hash-a", 2)})
        assert (
            manifest.sources_for(_DEFAULT_ROOT)
            is manifest.roots[root_key(_DEFAULT_ROOT)]
        )

    def test_remove_existing(self) -> None:
        manifest = _manifest({"a": ("hash-a", 2), "b": ("hash-b", 3)})
        manifest.remove(_DEFAULT_ROOT, "a")
        assert list(manifest.sources_for(_DEFAULT_ROOT)) == ["b"]
        assert manifest.total_chunks() == 3

    def test_remove_prunes_emptied_root(self) -> None:
        other = Path("C:/other")
        manifest = _manifest({"a": ("hash-a", 2)})
        manifest.record(other, "b", "hash-b", 3)

        manifest.remove(_DEFAULT_ROOT, "a")

        assert list(manifest.roots) == [root_key(other)]
        assert manifest.total_chunks() == 3

    def test_remove_missing_source_is_noop(self) -> None:
        manifest = _manifest({"a": ("hash-a", 2)})
        manifest.remove(_DEFAULT_ROOT, "ghost")
        assert list(manifest.sources_for(_DEFAULT_ROOT)) == ["a"]

    def test_remove_missing_root_is_noop(self) -> None:
        manifest = _manifest({"a": ("hash-a", 2)})
        manifest.remove(Path("Z:/ghost"), "a")
        assert manifest.total_chunks() == 2


class TestDiff:
    def test_one_file_per_bucket(self) -> None:
        manifest = _manifest(
            {
                "a": ("hash-a-old", 1),  # changed
                "b": ("hash-b", 1),  # unchanged
                "c": ("hash-c", 1),  # deleted
            }
        )
        current = {"a": "hash-a-new", "b": "hash-b", "d": "hash-d"}
        result = diff(current, manifest.sources_for(_DEFAULT_ROOT))
        assert result.added == ["d"]
        assert result.changed == ["a"]
        assert result.deleted == ["c"]
        assert result.unchanged == ["b"]

    def test_buckets_are_sorted(self) -> None:
        manifest = _manifest({"z": ("1", 1), "y": ("2", 1)})
        current = {"m": "3", "n": "4", "z": "9", "y": "8"}
        result = diff(current, manifest.sources_for(_DEFAULT_ROOT))
        assert result.added == ["m", "n"]
        assert result.changed == ["y", "z"]
        assert result.deleted == []
        assert result.unchanged == []

    def test_all_added_when_previous_empty(self) -> None:
        result = diff({"b": "2", "a": "1", "c": "3"}, {})
        assert result.added == ["a", "b", "c"]
        assert result.changed == []
        assert result.deleted == []
        assert result.unchanged == []

    def test_other_roots_are_invisible(self) -> None:
        manifest = _manifest({"other.txt": ("hash-o", 1)}, root=Path("C:/other"))
        result = diff({"a": "1"}, manifest.sources_for(_DEFAULT_ROOT))
        assert result.added == ["a"]
        assert result.deleted == []

    def test_windows_backslash_keys_are_opaque(self, tmp_path: Path) -> None:
        kept = r"C:\data\reports\q3 report.txt"
        changed = r"C:\data\notes.md"
        gone = r"D:\归档\旧文件.txt"
        manifest = _manifest({kept: ("same", 2), changed: ("old", 1), gone: ("x", 1)})
        current = {kept: "same", changed: "new"}
        result = diff(current, manifest.sources_for(_DEFAULT_ROOT))
        assert result.unchanged == [kept]
        assert result.changed == [changed]
        assert result.deleted == [gone]

        path = tmp_path / "manifest with space.json"
        save_manifest(manifest, path)
        assert load_manifest(path) == manifest


class TestComputeFingerprint:
    def test_stable_across_calls(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_settings(monkeypatch)
        assert compute_fingerprint() == compute_fingerprint()

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("OLLAMA_EMBEDDING_MODEL", "another-embedder"),
            ("cleaning_enabled", False),
            ("chunk_size_text_words", 999),
        ],
    )
    def test_changes_with_pipeline_inputs(
        self, monkeypatch: pytest.MonkeyPatch, field: str, value: Any
    ) -> None:
        _patch_settings(monkeypatch)
        before = compute_fingerprint()
        _patch_settings(monkeypatch, **{field: value})
        assert compute_fingerprint() != before

    def test_changes_with_pipeline_version(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_settings(monkeypatch)
        before = compute_fingerprint()
        monkeypatch.setattr(manifest_mod, "_PIPELINE_VERSION", 2)
        assert compute_fingerprint() != before

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("OLLAMA_LLM_MODEL", "another-llm"),
            ("top_k", 42),
            ("max_context_chars", 999),
        ],
    )
    def test_ignores_retrieval_only_settings(
        self, monkeypatch: pytest.MonkeyPatch, field: str, value: Any
    ) -> None:
        _patch_settings(monkeypatch)
        before = compute_fingerprint()
        _patch_settings(monkeypatch, **{field: value})
        assert compute_fingerprint() == before


class TestRootKey:
    def test_is_normcase_of_the_path_string(self, tmp_path: Path) -> None:
        assert root_key(tmp_path) == os.path.normcase(str(tmp_path))

    def test_matches_normcase_for_case_mixed_roots(self) -> None:
        for raw in ("C:/MixedCase/Docs", "C:\\MixedCase\\Docs"):
            root = Path(raw)
            assert root_key(root) == os.path.normcase(str(root))

    def test_normcase_equivalent_roots_share_one_key(self) -> None:
        upper, lower = Path("C:/MixedCase/Docs"), Path("C:/mixedcase/docs")
        # Whichever folding normcase applies on this platform, root_key
        # agrees with it exactly (full folding into one key on Windows).
        expected = os.path.normcase(str(upper))
        assert root_key(upper) == expected
        assert root_key(lower) == os.path.normcase(str(lower))


class TestOverlappingRoot:
    def test_empty_manifest_has_no_conflicts(self) -> None:
        manifest = Manifest(fingerprint="fp", roots={})
        assert overlapping_root(manifest, Path("C:/docs")) is None

    def test_same_root_returns_its_key(self, tmp_path: Path) -> None:
        root = tmp_path / "docs"
        manifest = _manifest({"a": ("aaa", 1)}, root=root)
        assert overlapping_root(manifest, root) == root_key(root)

    def test_candidate_inside_tracked_root_returns_key(self, tmp_path: Path) -> None:
        outer = tmp_path / "data"
        manifest = _manifest({"a": ("aaa", 1)}, root=outer)
        assert overlapping_root(manifest, outer / "inner") == root_key(outer)

    def test_candidate_containing_tracked_root_returns_key(
        self, tmp_path: Path
    ) -> None:
        inner = tmp_path / "data" / "docs"
        manifest = _manifest({"a": ("aaa", 1)}, root=inner)
        assert overlapping_root(manifest, tmp_path / "data") == root_key(inner)

    def test_disjoint_sibling_roots_do_not_overlap(self, tmp_path: Path) -> None:
        # "ab" shares a string prefix with "a" but is not inside it.
        manifest = _manifest({"a": ("aaa", 1)}, root=tmp_path / "a")
        assert overlapping_root(manifest, tmp_path / "ab") is None


class TestMatches:
    def test_true_for_same_fingerprint(self) -> None:
        manifest = Manifest(fingerprint="fp", roots={})
        assert matches(manifest, fingerprint="fp") is True

    def test_false_when_fingerprint_differs(self) -> None:
        manifest = _manifest({"a": ("hash-a", 1)})
        assert matches(manifest, fingerprint="other") is False

    def test_root_is_irrelevant_to_the_verdict(self) -> None:
        manifest = _manifest({"a": ("hash-a", 1)}, root=Path("C:/some/root"))
        assert matches(manifest, fingerprint="fp") is True

"""Unit tests for app.index orchestration (pure fakes: no Ollama, no Chroma)."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from langchain_rag import app as app_mod
from langchain_rag.app import (
    IndexResult,
    _invalidation_reason,
    _update_root,
    index,
)
from langchain_rag.storage.manifest import Manifest, root_key


class TestInvalidationReason:
    """Pure gate: whole-store rebuild vs per-root update, no mocks needed."""

    @staticmethod
    def _manifest(fingerprint: str = "fp", **roots_chunks: int) -> Manifest:
        """Manifest with one tracked file per kwarg: {name: chunk_count}."""
        manifest = Manifest(fingerprint=fingerprint, roots={})
        for name, chunks in roots_chunks.items():
            manifest.record(Path(f"C:/{name}"), f"{name}.txt", f"sha-{name}", chunks)
        return manifest

    def test_no_usable_manifest(self) -> None:
        reason = _invalidation_reason(manifest=None, fingerprint="fp", store_count=0)
        assert reason == "no usable manifest"

    def test_fingerprint_changed(self) -> None:
        manifest = self._manifest(fingerprint="old", a=1)
        reason = _invalidation_reason(
            manifest=manifest, fingerprint="new", store_count=1
        )
        assert reason == "indexing settings or pipeline changed"

    def test_store_count_mismatch(self) -> None:
        manifest = self._manifest(a=5, b=2)
        reason = _invalidation_reason(
            manifest=manifest, fingerprint="fp", store_count=6
        )
        assert reason == "manifest and vector store disagree (store wiped or modified?)"

    def test_none_when_everything_agrees(self) -> None:
        manifest = self._manifest(a=5, b=2)
        reason = _invalidation_reason(
            manifest=manifest, fingerprint="fp", store_count=7
        )
        assert reason is None

    def test_none_when_both_empty(self) -> None:
        manifest = self._manifest()
        reason = _invalidation_reason(
            manifest=manifest, fingerprint="fp", store_count=0
        )
        assert reason is None


class _FakeVectorStore:
    """Stands in for VectorStore: records every call, serves a canned count."""

    def __init__(self, count: int = 0) -> None:
        self.calls: list[tuple] = []
        self._count = count

    def clear(self) -> None:
        self.calls.append(("clear",))
        self._count = 0

    def add_documents(self, documents: list[Any]) -> None:
        self.calls.append(("add_documents", len(documents)))
        self._count += len(documents)

    def delete_by_source(self, source: str) -> None:
        self.calls.append(("delete_by_source", source))

    def search(self, query: str, k: int) -> list[Any]:
        self.calls.append(("search", query, k))
        return []

    def count(self) -> int:
        return self._count


def _root(tmp_path: Path, name: str) -> Path:
    """An existing, resolved directory safe to use as an index root."""
    root = (tmp_path / name).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _manifest_with(
    root: Path, records: dict[str, tuple[str, int]], fingerprint: str = "fp"
) -> Manifest:
    """Manifest holding {source: (sha256, chunks)} records under one root."""
    manifest = Manifest(fingerprint=fingerprint, roots={})
    for source, (sha256, chunks) in records.items():
        manifest.record(root, source, sha256, chunks)
    return manifest


def _install(
    monkeypatch: pytest.MonkeyPatch,
    *,
    manifest: Manifest | None,
    files: dict[Path, list[Path]],
    hashes: dict[Path, str],
    store_count: int = 0,
    fingerprint: str = "fp",
    chunks_per_file: int = 1,
) -> tuple[_FakeVectorStore, list[Manifest]]:
    """Replace every collaborator ``app.index()`` reaches for with fakes.

    Args:
        manifest: what ``load_manifest()`` hands back (None = unusable file).
        files: resolved root -> files ``iter_supported_files()`` finds there.
        hashes: file -> ``hash_file()`` digest.
        store_count: chunk count ``VectorStore().count()`` starts at.
        fingerprint: what ``compute_fingerprint()`` returns.
        chunks_per_file: chunk count ``process_file()`` yields per file.

    Returns:
        The shared fake store plus a snapshot of every saved manifest.
    """
    store = _FakeVectorStore(count=store_count)
    saved: list[Manifest] = []

    def iter_files(root: Path) -> list[Path]:
        return files.get(root, [])

    def save(manifest: Manifest) -> None:
        # Snapshot: the live manifest keeps mutating after each save.
        saved.append(copy.deepcopy(manifest))

    def process(path: Path, root: Path) -> list[str]:
        return [f"chunk-of-{path.name}"] * chunks_per_file

    monkeypatch.setattr(app_mod, "VectorStore", lambda: store)
    monkeypatch.setattr(app_mod, "load_manifest", lambda: manifest)
    monkeypatch.setattr(app_mod, "save_manifest", save)
    monkeypatch.setattr(app_mod, "compute_fingerprint", lambda: fingerprint)
    monkeypatch.setattr(app_mod, "hash_file", lambda path: hashes[path])
    monkeypatch.setattr(app_mod, "iter_supported_files", iter_files)
    monkeypatch.setattr(app_mod, "process_file", process)
    return store, saved


class TestIndexNewRoot:
    def test_never_seen_root_is_added_not_wiped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root_a = _root(tmp_path, "a")
        root_b = _root(tmp_path, "b")
        file_a = root_a / "a.txt"
        manifest = _manifest_with(root_b, {str(root_b / "b.txt"): ("sha-b", 4)})
        store, saved = _install(
            monkeypatch,
            manifest=manifest,
            store_count=4,
            files={root_a: [file_a]},
            hashes={file_a: "sha-a"},
            chunks_per_file=3,
        )

        result = index(root_a)

        # Sibling root B is trusted, so nothing gets wiped or re-embedded.
        assert store.calls == [("add_documents", 3)]
        assert result == IndexResult(chunks_stored=3, files_indexed=1)
        final = saved[-1]
        assert final.sources_for(root_a)[str(file_a)].sha256 == "sha-a"
        assert final.sources_for(root_a)[str(file_a)].chunks == 3
        assert final.sources_for(root_b) == manifest.sources_for(root_b)


class TestClearIndex:
    def test_clear_index_wipes_store_and_deletes_manifest(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store = _FakeVectorStore(count=7)
        deletions: list[bool] = []

        def fake_delete_manifest() -> bool:
            deletions.append(True)
            return True

        monkeypatch.setattr(app_mod, "VectorStore", lambda: store)
        monkeypatch.setattr(app_mod, "delete_manifest", fake_delete_manifest)

        removed = app_mod.clear_index()

        assert removed == 7
        assert store.calls == [("clear",)]
        assert deletions == [True]

    def test_clear_index_on_empty_store_is_not_an_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store = _FakeVectorStore(count=0)
        deletions: list[bool] = []

        monkeypatch.setattr(app_mod, "VectorStore", lambda: store)
        monkeypatch.setattr(
            app_mod, "delete_manifest", lambda: deletions.append(False) or False
        )

        assert app_mod.clear_index() == 0
        assert store.calls == [("clear",)]
        assert deletions == [False]


class TestIndexWipeBranches:
    def test_fingerprint_mismatch_wipes_and_drops_other_roots(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root_a = _root(tmp_path, "a")
        root_b = _root(tmp_path, "b")
        file_a = root_a / "a.txt"
        manifest = _manifest_with(
            root_b, {str(root_b / "b.txt"): ("sha-b", 4)}, fingerprint="outdated"
        )
        store, saved = _install(
            monkeypatch,
            manifest=manifest,
            store_count=4,
            files={root_a: [file_a]},
            hashes={file_a: "sha-a"},
            chunks_per_file=2,
        )

        index(root_a)

        assert store.calls == [("clear",), ("add_documents", 2)]
        final = saved[-1]
        assert list(final.roots) == [root_key(root_a)]
        assert final.fingerprint == "fp"

    def test_store_count_mismatch_triggers_wipe(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root_a = _root(tmp_path, "a")
        root_b = _root(tmp_path, "b")
        file_a = root_a / "a.txt"
        manifest = _manifest_with(root_b, {str(root_b / "b.txt"): ("sha-b", 5)})
        store, saved = _install(
            monkeypatch,
            manifest=manifest,
            # Store only holds 2 of the 5 chunks the manifest claims.
            store_count=2,
            files={root_a: [file_a]},
            hashes={file_a: "sha-a"},
            chunks_per_file=1,
        )

        index(root_a)

        assert store.calls == [("clear",), ("add_documents", 1)]
        assert list(saved[-1].roots) == [root_key(root_a)]


class TestIndexOverlappingRoots:
    def test_candidate_inside_tracked_root_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        outer = _root(tmp_path, "data")
        inner = _root(tmp_path, "data/inner")
        file_inner = inner / "a.txt"
        manifest = _manifest_with(outer, {str(outer / "a.txt"): ("sha-o", 2)})
        store, saved = _install(
            monkeypatch,
            manifest=manifest,
            store_count=2,
            files={inner: [file_inner]},
            hashes={file_inner: "sha-i"},
        )

        with pytest.raises(ValueError, match="overlaps"):
            index(inner)

        assert store.calls == []
        assert saved == []

    def test_candidate_containing_tracked_root_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        inner = _root(tmp_path, "data/inner")
        outer = _root(tmp_path, "data")
        file_outer = outer / "a.txt"
        manifest = _manifest_with(inner, {str(inner / "a.txt"): ("sha-i", 2)})
        store, saved = _install(
            monkeypatch,
            manifest=manifest,
            store_count=2,
            files={outer: [file_outer]},
            hashes={file_outer: "sha-o"},
        )

        with pytest.raises(ValueError, match="overlaps"):
            index(outer)

        assert store.calls == []
        assert saved == []

    def test_disjoint_sibling_root_is_accepted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root_a = _root(tmp_path, "a")
        root_b = _root(tmp_path, "ab")  # prefix-sharing but disjoint
        file_b = root_b / "b.txt"
        manifest = _manifest_with(root_a, {str(root_a / "a.txt"): ("sha-a", 2)})
        store, saved = _install(
            monkeypatch,
            manifest=manifest,
            store_count=2,
            files={root_b: [file_b]},
            hashes={file_b: "sha-b"},
            chunks_per_file=1,
        )

        index(root_b)

        assert store.calls == [("add_documents", 1)]
        assert set(saved[-1].roots) == {root_key(root_a), root_key(root_b)}


class TestUpdateRootDeletionPath:
    def test_vanished_file_deletes_only_its_own_root(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root_a = _root(tmp_path, "a")
        root_b = _root(tmp_path, "b")
        kept = root_a / "keep.txt"
        gone = root_a / "gone.txt"
        b_source = str(root_b / "b.txt")
        manifest = Manifest(fingerprint="fp", roots={})
        manifest.record(root_a, str(kept), "sha-keep", 2)
        manifest.record(root_a, str(gone), "sha-gone", 3)
        manifest.record(root_b, b_source, "sha-b", 4)
        store, saved = _install(
            monkeypatch,
            manifest=manifest,
            store_count=9,
            # keep.txt is still there (same hash); gone.txt vanished.
            files={root_a: [kept]},
            hashes={kept: "sha-keep"},
        )

        result = _update_root(root_a, manifest)

        assert store.calls == [("delete_by_source", str(gone))]
        assert result == IndexResult(
            chunks_stored=0, files_unchanged=1, files_deleted=1
        )
        final = saved[-1]
        assert list(final.sources_for(root_a)) == [str(kept)]
        # Root B's records ride through untouched.
        assert final.sources_for(root_b) == manifest.sources_for(root_b)


class TestIndexTrackedRoot:
    def test_reindexing_tracked_root_updates_in_place(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root_a = _root(tmp_path, "a")
        root_b = _root(tmp_path, "b")
        kept = root_a / "keep.txt"
        gone = root_a / "gone.txt"
        changed = root_a / "changed.txt"
        new = root_a / "new.txt"
        b_source = str(root_b / "b.txt")
        manifest = Manifest(fingerprint="fp", roots={})
        manifest.record(root_a, str(kept), "sha-keep", 2)
        manifest.record(root_a, str(gone), "sha-gone", 3)
        manifest.record(root_a, str(changed), "sha-old", 1)
        manifest.record(root_b, b_source, "sha-b", 4)
        store, saved = _install(
            monkeypatch,
            manifest=manifest,
            store_count=10,
            files={root_a: [kept, changed, new]},
            hashes={kept: "sha-keep", changed: "sha-new", new: "sha-fresh"},
            chunks_per_file=1,
        )

        result = index(root_a)

        # Equality with a tracked root is the update path, not a conflict,
        # and a healthy manifest means nothing gets wiped.
        assert ("clear",) not in store.calls
        assert store.calls == [
            ("delete_by_source", str(gone)),
            ("delete_by_source", str(changed)),
            ("add_documents", 1),
            ("add_documents", 1),
        ]
        assert result == IndexResult(
            chunks_stored=2, files_indexed=2, files_unchanged=1, files_deleted=1
        )
        final = saved[-1]
        assert sorted(final.sources_for(root_a)) == sorted(
            [str(kept), str(changed), str(new)]
        )
        assert final.sources_for(root_a)[str(changed)].sha256 == "sha-new"
        # Root B survives the update untouched.
        assert final.sources_for(root_b) == manifest.sources_for(root_b)

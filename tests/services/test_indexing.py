"""Unit tests for the indexing workflow: index() and clear_index() (pure fakes: no Ollama, no Chroma)."""

from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from langchain_rag.ingestion import ContextualGenerationError
from langchain_rag.services import indexing as indexing_mod
from langchain_rag.services.indexing import (
    _invalidation_reason,
    _update_root,
    index,
)
from langchain_rag.services.models import ClearResult, IndexResult
from langchain_rag.storage.manifest import Manifest, root_key


class _FakeVectorStore:
    """Stands in for VectorStore: records every call, serves a canned count."""

    def __init__(self, count: int = 0, pruned: int = 0) -> None:
        self.calls: list[tuple] = []
        self._count = count
        self._pruned = pruned

    def clear(self) -> int:
        self.calls.append(("clear",))
        self._count = 0
        return self._pruned

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


def _gate_manifest(fingerprint: str = "fp", **roots_chunks: int) -> Manifest:
    """Manifest with one tracked file per kwarg: {name: chunk_count}."""
    manifest = Manifest(fingerprint=fingerprint, roots={})
    for name, chunks in roots_chunks.items():
        manifest.record(Path(f"C:/{name}"), f"{name}.txt", f"sha-{name}", chunks)
    return manifest


class _FakeContextual:
    """Stands in for apply_contextual_summaries: passthrough that records inputs."""

    def __init__(self) -> None:
        self.seen: list[list[Any]] = []

    def __call__(self, chunks: list[Any]) -> list[Any]:
        self.seen.append(list(chunks))
        return chunks


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
    """Replace every collaborator ``index()`` reaches for with fakes.

    Args:
        manifest: what ``load_manifest()`` hands back (None = unusable file).
        files: resolved root -> files ``iter_supported_files()`` finds there.
        hashes: file -> ``hash_file()`` digest.
        store_count: chunk count ``VectorStore().count()`` starts at.
        fingerprint: what ``compute_fingerprint()`` returns.
        chunks_per_file: chunk count ``process_file()`` yields per file.

    Returns:
        The shared fake store (Step 5 recorder on ``.contextual``) plus a
        snapshot of every saved manifest.
    """
    store = _FakeVectorStore(count=store_count)
    saved: list[Manifest] = []
    contextual = _FakeContextual()

    def iter_files(root: Path) -> list[Path]:
        return files.get(root, [])

    def save(manifest: Manifest) -> None:
        # Snapshot: the live manifest keeps mutating after each save.
        saved.append(copy.deepcopy(manifest))

    def process(path: Path, root: Path) -> list[str]:
        return [f"chunk-of-{path.name}"] * chunks_per_file

    monkeypatch.setattr(indexing_mod, "VectorStore", lambda: store)
    monkeypatch.setattr(indexing_mod, "load_manifest", lambda: manifest)
    monkeypatch.setattr(indexing_mod, "save_manifest", save)
    monkeypatch.setattr(indexing_mod, "compute_fingerprint", lambda: fingerprint)
    monkeypatch.setattr(indexing_mod, "hash_file", lambda path: hashes[path])
    monkeypatch.setattr(indexing_mod, "iter_supported_files", iter_files)
    monkeypatch.setattr(indexing_mod, "process_file", process)
    # Hermetic against a developer .env that flips contextual_retrieval_enabled:
    # the summarize stage stays a recorded passthrough, and the pre-wipe probe
    # never builds a real Ollama/gateway client either.
    monkeypatch.setattr(indexing_mod, "apply_contextual_summaries", contextual)
    monkeypatch.setattr(indexing_mod, "get_contextual_chat_model", lambda: object())
    store.contextual = contextual
    return store, saved


class TestClearIndex:
    def test_wipes_store_and_deletes_manifest(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Populated and empty stores both clear cleanly, reporting reality."""
        store = _FakeVectorStore(count=7, pruned=3)
        deletions: list[bool] = []

        monkeypatch.setattr(indexing_mod, "VectorStore", lambda: store)
        monkeypatch.setattr(
            indexing_mod, "delete_manifest", lambda: deletions.append(True) or True
        )

        result = indexing_mod.clear_index()

        assert result == ClearResult(chunks_removed=7, folders_pruned=3)
        assert store.calls == [("clear",)]

        # An empty store is not an error either: zero counts, no pruned folders.
        store = _FakeVectorStore(count=0)
        monkeypatch.setattr(indexing_mod, "VectorStore", lambda: store)
        monkeypatch.setattr(
            indexing_mod, "delete_manifest", lambda: deletions.append(False) or False
        )

        assert indexing_mod.clear_index() == ClearResult(
            chunks_removed=0, folders_pruned=0
        )
        assert store.calls == [("clear",)]
        assert deletions == [True, False]


class TestWholeStoreRebuild:
    def test_gate_reasons_and_resulting_wipes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Each rebuild trigger the gate names, then wipes and re-indexes."""
        stale = _gate_manifest(fingerprint="old", a=1)
        healthy = _gate_manifest(a=5, b=2)
        assert (
            _invalidation_reason(manifest=None, fingerprint="fp", store_count=0)
            == "no usable manifest"
        )
        assert (
            _invalidation_reason(manifest=stale, fingerprint="new", store_count=1)
            == "indexing settings or pipeline changed"
        )
        assert (
            _invalidation_reason(manifest=healthy, fingerprint="fp", store_count=6)
            == "manifest and vector store disagree (store wiped or modified?)"
        )
        assert (
            _invalidation_reason(manifest=healthy, fingerprint="fp", store_count=7)
            is None
        )
        assert (
            _invalidation_reason(
                manifest=_gate_manifest(), fingerprint="fp", store_count=0
            )
            is None
        )

        root_a = _root(tmp_path, "a")
        root_b = _root(tmp_path, "b")
        file_a = root_a / "a.txt"

        # Fingerprint drift: wipe, drop sibling root B's records, stamp fresh.
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

        # Count mismatch: store only holds 2 of the 5 chunks manifest claims.
        manifest = _manifest_with(root_b, {str(root_b / "b.txt"): ("sha-b", 5)})
        store, saved = _install(
            monkeypatch,
            manifest=manifest,
            store_count=2,
            files={root_a: [file_a]},
            hashes={file_a: "sha-a"},
            chunks_per_file=1,
        )

        index(root_a)

        assert store.calls == [("clear",), ("add_documents", 1)]
        assert list(saved[-1].roots) == [root_key(root_a)]


class TestIncrementalIndex:
    def test_new_and_tracked_roots_update_without_wipe(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Healthy manifests mean per-root updates: nothing wiped, siblings kept."""
        root_a = _root(tmp_path, "a")
        root_b = _root(tmp_path, "b")

        # A never-seen root is *added*: sibling root B is trusted, so nothing
        # gets wiped or re-embedded.
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

        assert store.calls == [("add_documents", 3)]
        assert result == IndexResult(chunks_stored=3, files_indexed=1)
        final = saved[-1]
        assert final.sources_for(root_a)[str(file_a)].sha256 == "sha-a"
        assert final.sources_for(root_a)[str(file_a)].chunks == 3
        assert final.sources_for(root_b) == manifest.sources_for(root_b)

        # A tracked root updates in place: vanished files are purged, changed
        # files deleted-then-re-added, new files added, unchanged untouched —
        # and root equality is the update path, not an overlap conflict.
        kept = root_a / "keep.txt"
        gone = root_a / "gone.txt"
        changed = root_a / "changed.txt"
        new = root_a / "new.txt"
        manifest = Manifest(fingerprint="fp", roots={})
        manifest.record(root_a, str(kept), "sha-keep", 2)
        manifest.record(root_a, str(gone), "sha-gone", 3)
        manifest.record(root_a, str(changed), "sha-old", 1)
        manifest.record(root_b, str(root_b / "b.txt"), "sha-b", 4)
        store, saved = _install(
            monkeypatch,
            manifest=manifest,
            store_count=10,
            files={root_a: [kept, changed, new]},
            hashes={kept: "sha-keep", changed: "sha-new", new: "sha-fresh"},
            chunks_per_file=1,
        )

        result = index(root_a)

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


class TestOverlappingRoots:
    def test_nesting_rejected_prefix_sharing_accepted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Roots that would claim the same files are refused before any store call."""
        outer = _root(tmp_path, "data")
        inner = _root(tmp_path, "data/inner")

        # Candidate inside a tracked root: rejected, nothing touched.
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

        # Mirror direction: candidate containing a tracked root.
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

        # Prefix-sharing but disjoint ("ab" vs "a"): accepted as its own root.
        root_a = _root(tmp_path, "a")
        root_ab = _root(tmp_path, "ab")
        file_b = root_ab / "b.txt"
        manifest = _manifest_with(root_a, {str(root_a / "a.txt"): ("sha-a", 2)})
        store, saved = _install(
            monkeypatch,
            manifest=manifest,
            store_count=2,
            files={root_ab: [file_b]},
            hashes={file_b: "sha-b"},
            chunks_per_file=1,
        )

        index(root_ab)

        assert store.calls == [("add_documents", 1)]
        assert set(saved[-1].roots) == {root_key(root_a), root_key(root_ab)}


class TestContextualStage:
    """Step 5 seam: the indexing service delegates chunks to the summarizer."""

    def test_wiring_per_file_failure_and_config_abort(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Every indexed file goes through the stage exactly once, fed exactly
        # the process_file() output.
        root = _root(tmp_path, "wire")
        first = root / "one.txt"
        second = root / "two.txt"
        store, _ = _install(
            monkeypatch,
            manifest=None,
            files={root: [first, second]},
            hashes={first: "sha-1", second: "sha-2"},
            chunks_per_file=2,
        )

        result = _update_root(root, Manifest(fingerprint="fp", roots={}))

        assert store.contextual.seen == [
            ["chunk-of-one.txt", "chunk-of-one.txt"],
            ["chunk-of-two.txt", "chunk-of-two.txt"],
        ]
        assert result == IndexResult(chunks_stored=4, files_indexed=2)

        # A per-file summarizer failure leaves that file out of the store AND
        # the manifest, so it stays "added" for the next run to retry whole;
        # sibling files proceed normally.
        root = _root(tmp_path, "fail")
        good = root / "good.txt"
        bad = root / "bad.txt"

        def failing_contextual(chunks: list[Any]) -> list[Any]:
            # _install's process_file fake names chunks after their file.
            if "bad" in str(chunks[0]):
                raise ContextualGenerationError("summarizer down")
            return chunks

        store, saved = _install(
            monkeypatch,
            manifest=None,
            files={root: [good, bad]},
            hashes={good: "sha-g", bad: "sha-b"},
            chunks_per_file=2,
        )
        monkeypatch.setattr(
            indexing_mod, "apply_contextual_summaries", failing_contextual
        )

        result = index(root)

        assert store.calls == [("clear",), ("add_documents", 2)]
        assert result == IndexResult(chunks_stored=2, files_indexed=1)
        final = saved[-1]
        assert str(good) in final.sources_for(root)
        assert str(bad) not in final.sources_for(root)

        # A broken .env aborts *before* any whole-store teardown, not leaving
        # the user with an empty store (rag-decisions.md).
        root = _root(tmp_path, "abort")
        file_a = root / "a.txt"
        store, _ = _install(
            monkeypatch,
            manifest=None,  # would trigger a wipe — must never be reached
            files={root: [file_a]},
            hashes={file_a: "sha-a"},
        )
        monkeypatch.setattr(
            indexing_mod,
            "get_settings",
            lambda: SimpleNamespace(contextual_retrieval_enabled=True),
        )

        def _boom() -> None:
            raise ValueError("OPENAI_BASE_URL is not set in .env")

        monkeypatch.setattr(indexing_mod, "get_contextual_chat_model", _boom)

        with pytest.raises(ValueError, match="OPENAI_BASE_URL"):
            index(root)

        assert store.calls == []  # the fake store never even saw ("clear",)

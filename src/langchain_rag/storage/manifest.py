"""Index manifest: track indexed sources for incremental indexing.

The manifest is a small JSON file grouping every indexed source file (its
absolute path, exactly as stored in Chroma metadata ``source``) by *index
root*, mapping it to the SHA-256 of its bytes plus the chunk count it
produced. Combined with a fingerprint of everything that influences chunk
*content* (pipeline code version, embedding model, cleaning, chunking
settings), it lets the indexer decide what is added / changed / deleted /
unchanged *per root*, while several roots coexist in one vector collection.

Use cases:
    - Incremental indexing (``app.index()``): ``diff()`` between a freshly
      scanned file→hash map of one root and that root's source records from
      ``Manifest.sources_for()`` decides which files to (re-)index or delete;
      other roots are never touched.
    - Whole-store rebuild triggers: ``matches()`` failing (or
      ``load_manifest()`` returning ``None``, or the manifest disagreeing
      with the stored chunk count) means the collection cannot be trusted
      and every root must be rebuilt.

Unusable means *structurally* unusable: the file carries no schema version
by design — anything that does not parse exactly as the current shape
yields ``None`` and triggers a whole-store rebuild. A future format change
is handled by wiping (``--clear``), never by migration code.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePath

from ..config import PROJECT_ROOT, get_settings

_PIPELINE_VERSION = (
    1  # manual bump: ingestion code that changes chunk content/boundaries
)
MANIFEST_PATH = PROJECT_ROOT / "data" / "index_manifest.json"

_BLOCK_SIZE = 64 * 1024


@dataclass(frozen=True)
class SourceRecord:
    """One tracked source file: content hash, chunk count, indexed-at stamp."""

    sha256: str
    chunks: int
    indexed_at: str  # datetime.now().isoformat(timespec="seconds")


def root_key(root: Path) -> str:
    """Canonical ``Manifest.roots`` key for a resolved index root.

    ``os.path.normcase`` makes root comparisons drive-letter- and path-
    case-insensitive on Windows, so ``C:\\Docs`` and ``C:\\docs`` cannot
    become two keys owning the same source files.

    Args:
        root: resolved index root (``Path.resolve()`` at the ``index()``
            boundary is still mandatory — this only normalizes case/sep).

    Returns:
        Normalized absolute path string.
    """
    return os.path.normcase(str(root))


@dataclass
class Manifest:
    """All tracked index roots and their sources, plus validity context.

    Attributes:
        fingerprint: ``compute_fingerprint()`` at write time (global: one
            embedding/chunking configuration covers the whole collection).
        roots: ``root_key()``-normalized root -> absolute source path string
            -> record (source key matches the Chroma ``source`` metadata
            value).
    """

    fingerprint: str
    roots: dict[str, dict[str, SourceRecord]]

    def total_chunks(self) -> int:
        """Sum of chunk counts across every root (whole-store invariant)."""
        return sum(
            record.chunks
            for sources in self.roots.values()
            for record in sources.values()
        )

    def sources_for(self, root: Path) -> dict[str, SourceRecord]:
        """Source records tracked for ``root``.

        Args:
            root: resolved index root.

        Returns:
            The root's source map, or ``{}`` for a never-indexed root
            (lookup never mutates the manifest).
        """
        return self.roots.get(root_key(root), {})

    def record(self, root: Path, source: str, sha256: str, chunks: int) -> None:
        """Insert or replace the record for ``source`` under ``root``.

        The root entry is created on first use; ``source`` gets a fresh
        timestamp.
        """
        self.roots.setdefault(root_key(root), {})[source] = SourceRecord(
            sha256=sha256,
            chunks=chunks,
            indexed_at=datetime.now().isoformat(timespec="seconds"),
        )

    def remove(self, root: Path, source: str) -> None:
        """Drop the record for ``source``; tolerant of missing keys.

        A root left without sources is pruned so the manifest only lists
        roots that actually own tracked files.
        """
        key = root_key(root)
        sources = self.roots.get(key)
        if sources is None:
            return
        sources.pop(source, None)
        if not sources:
            del self.roots[key]


@dataclass(frozen=True)
class ManifestDiff:
    """File paths classified against the manifest; every list is sorted."""

    added: list[str]  # present in current, absent in manifest
    changed: list[str]  # in both, hash differs
    deleted: list[str]  # in manifest, absent in current
    unchanged: list[str]  # in both, hash equal


def compute_fingerprint() -> str:
    """Hash everything that invalidates stored chunks when it changes.

    Covers the pipeline code version, the embedding model (chunk *ids* may
    stay but vectors must be re-embedded), cleaning, and all ``chunk_*``
    settings. Deliberately excludes retrieval-only knobs (``OLLAMA_LLM_MODEL``,
    ``top_k``, ``max_context_chars``): changing those invalidates nothing
    stored.

    Returns:
        Hex SHA-256 of the canonical (``sort_keys=True``) JSON payload.
    """
    settings = get_settings()
    payload = {
        "pipeline_version": _PIPELINE_VERSION,
        "embedding_model": settings.OLLAMA_EMBEDDING_MODEL,
        "cleaning_enabled": settings.cleaning_enabled,
        "chunking": {
            key: value
            for key, value in settings.model_dump().items()
            if key.startswith("chunk_")
        },
    }
    canonical = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def hash_file(path: Path) -> str:
    """SHA-256 hex digest of a file's bytes, read in 64 KiB blocks.

    Args:
        path: file to hash.

    Returns:
        Hex digest of the full file content (never held in memory whole).
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(_BLOCK_SIZE):
            digest.update(block)
    return digest.hexdigest()


def _parse_sources(raw_sources: object) -> dict[str, SourceRecord] | None:
    """Validate and parse one root's source→record mapping; ``None`` if invalid."""
    if not isinstance(raw_sources, dict):
        return None
    sources: dict[str, SourceRecord] = {}
    for source, entry in raw_sources.items():
        if not isinstance(source, str) or not isinstance(entry, dict):
            return None
        sha256 = entry.get("sha256")
        chunks = entry.get("chunks")
        indexed_at = entry.get("indexed_at")
        if (
            not isinstance(sha256, str)
            or not isinstance(chunks, int)
            or isinstance(chunks, bool)
            or not isinstance(indexed_at, str)
        ):
            return None
        sources[source] = SourceRecord(
            sha256=sha256, chunks=chunks, indexed_at=indexed_at
        )
    return sources


def load_manifest(path: Path = MANIFEST_PATH) -> Manifest | None:
    """Read and validate the manifest file.

    Never raises on bad data: any unreadable, corrupt, or wrongly shaped
    file yields ``None`` so the caller falls back to a whole-store rebuild.
    There is no schema version to check — the structure itself is the
    contract, and a file from an older or unknown format simply fails to
    parse (unknown extra keys are ignored).

    Args:
        path: manifest file location.

    Returns:
        The parsed ``Manifest``, or ``None`` when unusable.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(raw, dict):
        return None
    fingerprint = raw.get("fingerprint")
    raw_roots = raw.get("roots")
    if not isinstance(fingerprint, str) or not isinstance(raw_roots, dict):
        return None
    roots: dict[str, dict[str, SourceRecord]] = {}
    for root, raw_sources in raw_roots.items():
        if not isinstance(root, str):
            return None
        sources = _parse_sources(raw_sources)
        if sources is None:
            return None
        roots[root] = sources
    return Manifest(fingerprint=fingerprint, roots=roots)


def save_manifest(manifest: Manifest, path: Path = MANIFEST_PATH) -> None:
    """Write the manifest atomically (temp file in the same dir + replace).

    Args:
        manifest: state to persist.
        path: manifest file location; parent directories are created.
    """
    payload = {
        "fingerprint": manifest.fingerprint,
        "roots": {
            root: {
                source: {
                    "sha256": record.sha256,
                    "chunks": record.chunks,
                    "indexed_at": record.indexed_at,
                }
                for source, record in sources.items()
            }
            for root, sources in manifest.roots.items()
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp_path, path)


def delete_manifest(path: Path = MANIFEST_PATH) -> bool:
    """Delete the manifest file; tolerant of an already-missing file.

    Args:
        path: manifest file location.

    Returns:
        ``True`` when a file was actually deleted, ``False`` when absent.
    """
    try:
        path.unlink()
    except FileNotFoundError:
        return False
    return True


def diff(current: dict[str, str], previous: dict[str, SourceRecord]) -> ManifestDiff:
    """Classify the scanned file→hash map against one root's tracked sources.

    Args:
        current: source path -> ``hash_file()`` digest for every file found
            under the index root right now.
        previous: that root's records from ``Manifest.sources_for()`` — pass
            ``{}`` for a never-indexed root or after a whole-store wipe
            (everything then counts as added). Other roots are structurally
            invisible here, which is what keeps roots isolated.

    Returns:
        ``ManifestDiff`` with sorted path lists per bucket.
    """
    previous_hashes = {source: record.sha256 for source, record in previous.items()}
    added = sorted(set(current) - set(previous_hashes))
    deleted = sorted(set(previous_hashes) - set(current))
    common = set(current) & set(previous_hashes)
    changed = sorted(
        source for source in common if current[source] != previous_hashes[source]
    )
    unchanged = sorted(
        source for source in common if current[source] == previous_hashes[source]
    )
    return ManifestDiff(
        added=added, changed=changed, deleted=deleted, unchanged=unchanged
    )


def matches(manifest: Manifest, *, fingerprint: str) -> bool:
    """True when the manifest was built under the current pipeline fingerprint.

    Use cases:
        - Gate incremental updates: a ``False`` result means the collection
          mixes chunks embedded under different settings, so the whole store
          (every root) must be rebuilt — a fresh fingerprint can coexist with
          no stale one.
    """
    return manifest.fingerprint == fingerprint


def overlapping_root(manifest: Manifest, root: Path) -> str | None:
    """Key of a tracked root that overlaps ``root``, or ``None``.

    Overlap means equal, containing, or contained-by: two roots owning the
    same files would each record chunks that the other's re-index then
    deletes, corrupting both records.

    Args:
        manifest: current manifest.
        root: resolved candidate index root.

    Returns:
        The conflicting tracked root key, or ``None`` when ``root`` is clean.
    """
    key = root_key(root)
    for tracked in manifest.roots:
        if key == tracked:
            return tracked
        candidate, existing = PurePath(key), PurePath(tracked)
        if candidate.is_relative_to(existing) or existing.is_relative_to(candidate):
            return tracked
    return None

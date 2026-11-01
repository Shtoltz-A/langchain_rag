"""Indexing workflow: invalidation gate, per-root diff, per-file pipeline.

The coordinator between ``ingestion`` and ``storage`` for the index path: it
owns the wipe-vs-update decision, the per-root file diff, and the per-file
process → summarize → store → record cycle. All output goes through the
shared Console; no Typer, no process exits (rule 6).
"""

from pathlib import Path
from time import perf_counter

from ..config import get_settings
from ..console import get_console
from ..generation import get_contextual_chat_model
from ..ingestion import (
    ContextualGenerationError,
    apply_contextual_summaries,
    iter_supported_files,
    process_file,
)
from ..storage import (
    Manifest,
    VectorStore,
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
from .models import ClearResult, IndexResult

c = get_console()


def index(directory: Path) -> IndexResult:
    """Index the directory; incrementally while the manifest still applies.

    Several roots coexist in the one collection: a never-seen directory is
    *added* without touching other roots' vectors, and a tracked directory
    updates via a diff scoped to its own manifest records. The *whole* store
    is rebuilt instead only when ``_invalidation_reason()`` fires. To wipe
    everything deliberately, call ``clear_index()`` (CLI: ``index --clear``).

    Raises:
        ValueError: ``directory`` overlaps an already-tracked root
            (containing, or contained by it) — such roots would claim the
            same files and corrupt each other's records.
        ConfigurationError: contextual retrieval is enabled but the
            summarizer's provider is misconfigured — raised by the up-front
            probe below, before anything is wiped.
    """
    root = directory.resolve()
    if get_settings().contextual_retrieval_enabled:
        # Fail loudly on summarizer misconfiguration BEFORE any whole-store
        # wipe below: building the client is side-effect-free (lazy-client
        # rule), and aborting after ``clear()`` would leave the user with an
        # empty store and no explanation short of re-reading this code.
        get_contextual_chat_model()
    fingerprint = compute_fingerprint()
    manifest = load_manifest()
    store = VectorStore()

    reason = _invalidation_reason(
        manifest=manifest,
        fingerprint=fingerprint,
        store_count=store.count(),
    )
    if reason is not None:
        manifest = _rebuild_store(manifest, fingerprint, root, reason=reason)
    else:
        _check_overlap(manifest, root)
    return _update_root(root, manifest)


def clear_index() -> ClearResult:
    """Delete the whole index: every vector in the collection + manifest file.

    Order guards the crash window between the two steps: dying after
    ``clear()`` but before the unlink leaves a stale manifest that trips the
    count-mismatch gate on the next run (wipe again — self-heals); the
    reverse order self-heals through "no usable manifest". ``clear()`` also
    sweeps the segment directories Chroma leaked into ``data/chroma``.

    Returns:
        Chunks removed from the vector store + stale segment folders pruned.
    """
    store = VectorStore()
    removed = store.count()
    pruned = store.clear()
    delete_manifest()
    c.debug(
        f"Index cleared: {removed} chunks removed, manifest file deleted, "
        f"{pruned} stale segment folder(s) pruned"
    )
    return ClearResult(chunks_removed=removed, folders_pruned=pruned)


def _invalidation_reason(
    *,
    manifest: Manifest | None,
    fingerprint: str,
    store_count: int,
) -> str | None:
    """Why the *whole* store must be rebuilt, or ``None`` to update one root.

    Whole-store rebuild runs when: no usable manifest exists, the fingerprint
    (pipeline version, embedding model, chunking/cleaning settings) changed,
    or the manifest disagrees with the stored chunk count (e.g.
    ``data/chroma`` was wiped by hand or a crash landed between
    ``add_documents`` and the manifest save). Deliberate wipes are not a
    trigger here — ``clear_index()`` does those directly.

    Use cases:
        - Pure decision gate: keeps the wipe-vs-update choice unit-testable
          without a live Chroma store.
    """
    if manifest is None:
        return "no usable manifest"
    if not matches(manifest, fingerprint=fingerprint):
        return "indexing settings or pipeline changed"
    if manifest.total_chunks() != store_count:
        return "manifest and vector store disagree (store wiped or modified?)"
    return None


def _rebuild_store(
    manifest: Manifest | None, fingerprint: str, root: Path, *, reason: str
) -> Manifest:
    """Wipe the whole collection and return the fresh manifest to repopulate.

    Nothing is saved here: the first ``save_manifest()`` happens in
    ``_update_root``, so a crash between the wipe and that save still
    self-heals through the count-mismatch gate (same crash-window reasoning
    as ``clear_index()``).

    Args:
        manifest: the invalidated manifest (``None`` when unusable).
        fingerprint: fresh ``compute_fingerprint()`` stamped on the new one.
        root: the root about to be indexed; its records are recreated by the
            update, so its key is excluded from the dropped-roots warning.
        reason: trigger named by ``_invalidation_reason()``, for the debug.
    """
    c.debug(f"Whole-index rebuild: {reason}")
    dropped = (
        [tracked for tracked in manifest.roots if tracked != root_key(root)]
        if manifest
        else []
    )
    pruned = VectorStore().clear()
    if pruned:
        c.debug(f"Pruned {pruned} stale Chroma segment folder(s)")
    if dropped:
        c.warning(
            "Whole index wiped; other tracked roots were dropped: "
            f"{'; '.join(dropped)}. Re-index them to restore."
        )
    return Manifest(fingerprint=fingerprint, roots={})


def _check_overlap(manifest: Manifest, root: Path) -> None:
    """Refuse a root that would share files with an already-tracked root.

    Equal-to-tracked is the normal in-place update path; only a *different*
    root sharing files with this one is a conflict.

    Raises:
        ValueError: ``root`` contains or sits inside an indexed root.
    """
    conflict = overlapping_root(manifest, root)
    if conflict is not None and conflict != root_key(root):
        raise ValueError(
            f"Refusing to index {root}: it overlaps the already-indexed "
            f"root {conflict}. Index the outer folder instead of nesting."
        )


def _file_label(path: Path, root: Path) -> str:
    """Root-relative display label for per-file output (short, still unique)."""
    try:
        return str(path.relative_to(root))
    except ValueError:  # defensive: a source outside the root shows its name
        return path.name


def _update_root(root: Path, manifest: Manifest) -> IndexResult:
    """Re-embed one root's added/changed files; drop vectors of deleted ones.

    Vector deletions and manifest-record removals stay paired step for step,
    so the manifest never counts chunks the store has dropped — drift there
    would trip the count-mismatch gate into wiping every root.

    Other roots are never scanned or modified: the diff only sees this root's
    records, so a new root arrives as "all added" on an untouched collection
    and a tracked root updates in place.
    """
    current: dict[str, str] = {}
    for file in iter_supported_files(root):
        try:
            current[str(file)] = hash_file(file)
        except OSError as exc:
            # Per-file tolerance, like every load failure below: a locked or
            # vanished file (routine on Windows) must not abort the run. A
            # tracked file keeps its previous digest — a transient lock must
            # not read as a deletion and drop its vectors — while an
            # untracked one simply waits for a later run to read it.
            tracked = manifest.sources_for(root).get(str(file))
            if tracked is not None:
                current[str(file)] = tracked.sha256
                c.warning(
                    f"Could not re-read {file}: {exc}. Keeping its indexed "
                    "version; it counts as unchanged this run."
                )
            else:
                c.warning(f"Skipping unreadable file {file}: {exc}.")

    if not current and not manifest.sources_for(root):
        # Nothing on disk, nothing tracked: only a fresh wipe needs persisting
        # so the empty store and manifest stay consistent.
        c.warning(f"No loadable documents found under {root}.")
        save_manifest(manifest)
        return IndexResult(chunks_stored=0)
    if not current:
        c.warning(f"No loadable documents found under {root}.")

    file_diff = diff(current, manifest.sources_for(root))
    c.debug(
        f"Diff: {len(file_diff.added)} new, {len(file_diff.changed)} changed, "
        f"{len(file_diff.deleted)} deleted, {len(file_diff.unchanged)} unchanged"
    )

    store = VectorStore()
    if file_diff.deleted or file_diff.changed:
        for source in file_diff.deleted + file_diff.changed:
            # Delete before re-add (or after removal from disk): stale vectors
            # would otherwise linger beside the fresh ones. The manifest
            # record goes in the same breath — a changed file that fails to
            # re-index then simply re-appears as "added" next run, and the
            # manifest never counts chunks the store no longer holds.
            store.delete_by_source(source)
            manifest.remove(root, source)
        # Persist the removals before any re-add: the per-file saves below
        # only run for added/changed files, yet a delete-only run must not
        # leave stale records (or worse, nothing saved) behind.
        save_manifest(manifest)

    chunks_stored = 0
    files_indexed = 0
    for source in file_diff.added + file_diff.changed:
        stored = _index_file(source, root, store, manifest, current[source])
        if stored is None:
            continue  # summarizer failure: nothing was recorded, no save due
        if stored:
            chunks_stored += stored
            files_indexed += 1
        save_manifest(manifest)  # persist per file: a crash resumes here

    return IndexResult(
        chunks_stored=chunks_stored,
        files_indexed=files_indexed,
        files_unchanged=len(file_diff.unchanged),
        files_deleted=len(file_diff.deleted),
    )


def _index_file(
    source: str, root: Path, store: VectorStore, manifest: Manifest, digest: str
) -> int | None:
    """Run one file through the pipeline: process → summarize → store → record.

    Args:
        source: absolute path string of the added/changed file.
        root: index root (context prefixes and the trace label are relative).
        store: vector store receiving the chunks.
        manifest: records the file only on success — files that failed or
            yielded no chunks stay out and are retried next run (per-file
            error tolerance).
        digest: content hash of the file, recorded alongside the chunk count.

    Returns:
        Chunks stored (``0`` when the file yielded none), or ``None`` when
        the contextual summarizer failed — the caller then skips its
        ``save_manifest`` for this file, and since its delete step already
        dropped any old record, the file re-appears as "added" next run.
    """
    start = perf_counter()
    chunks = process_file(Path(source), root)
    if chunks:
        try:
            # Prepend the cached/generated context sentences.
            # ``process_file`` stays LLM-free — batch parity is preserved.
            chunks = apply_contextual_summaries(chunks)
        except ContextualGenerationError as exc:
            # Leave the file out of the manifest (like any per-file
            # failure): the next run retries it whole, and every summary
            # generated before the failure is already cached — the retry
            # only pays for the chunks that never got one.
            c.error(
                f"Contextual summaries failed for {source}: {exc}. "
                f"File NOT indexed; it will be retried on the next run."
            )
            return None
        store.add_documents(chunks)
        manifest.record(root, source, digest, len(chunks))
        # One line per file: the load/split counts add nothing between
        # here and the next file's line, and the root-relative path is
        # unique without the absolute prefix. Timing spans split,
        # summarizing AND embedding+store — the Ollama round-trips are
        # the slow part worth seeing.
        c.trace(
            _file_label(Path(source), root),
            f"{len(chunks)} chunks",
            f"{perf_counter() - start:.1f}s",
            ok=True,
        )
        return len(chunks)
    return 0

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from .config import get_settings
from .console import get_console
from .generation import (
    ANSWER_SYSTEM_PROMPT,
    apply_budget,
    describe_answer_backend,
    format_context,
    get_answer_llm,
    traced_invoke,
)
from .ingestion import (
    iter_supported_files,
    process_file,
)
from .retrieval import retrieve
from .storage import (
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]

c = get_console()


@dataclass
class IndexResult:
    """Summary of one index run, for CLI reporting."""

    chunks_stored: int
    files_indexed: int = 0
    files_unchanged: int = 0
    files_deleted: int = 0


def index(directory: Path) -> IndexResult:
    """Index the directory; incrementally while the manifest still applies.

    Several roots coexist in the one collection: a never-seen directory is
    *added* without touching other roots' vectors, and a tracked directory
    updates via a diff scoped to its own manifest records. The *whole* store
    is rebuilt instead only when ``_invalidation_reason()`` fires. To wipe
    everything deliberately, call ``clear_index()`` (CLI: ``index --clear``).

    Raises:
        ValueError: ``directory`` overlaps an already-tracked root (equal,
            containing, or contained by it) — such roots would claim the same
            files and corrupt each other's records.
    """
    root = directory.resolve()
    fingerprint = compute_fingerprint()
    manifest = load_manifest()
    store = VectorStore()

    reason = _invalidation_reason(
        manifest=manifest,
        fingerprint=fingerprint,
        store_count=store.count(),
    )
    if reason is not None:
        c.debug(f"Whole-index rebuild: {reason}")
        dropped = (
            [tracked for tracked in manifest.roots if tracked != root_key(root)]
            if manifest
            else []
        )
        store.clear()
        manifest = Manifest(fingerprint=fingerprint, roots={})
        if dropped:
            c.warning(
                "Whole index wiped; other tracked roots were dropped: "
                f"{'; '.join(dropped)}. Re-index them to restore."
            )
    else:
        conflict = overlapping_root(manifest, root)
        if conflict is not None and conflict != root_key(root):
            # Equal-to-tracked is the normal in-place update path; only a
            # *different* root sharing files with this one is a conflict.
            raise ValueError(
                f"Refusing to index {root}: it overlaps the already-indexed "
                f"root {conflict}. Index the outer folder instead of nesting."
            )
    return _update_root(root, manifest)


def clear_index() -> int:
    """Delete the whole index: every vector in the collection + manifest file.

    Order guards the crash window between the two steps: dying after
    ``clear()`` but before the unlink leaves a stale manifest that trips the
    count-mismatch gate on the next run (wipe again — self-heals); the
    reverse order self-heals through "no usable manifest".

    Returns:
        Number of chunks removed from the vector store.
    """
    store = VectorStore()
    removed = store.count()
    store.clear()
    delete_manifest()
    c.debug(f"Index cleared: {removed} chunks removed, manifest file deleted")
    return removed


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


def _update_root(root: Path, manifest: Manifest) -> IndexResult:
    """Re-embed one root's added/changed files; drop vectors of deleted ones.

    Other roots are never scanned or modified: the diff only sees this root's
    records, so a new root arrives as "all added" on an untouched collection
    and a tracked root updates in place.
    """
    current = {str(file): hash_file(file) for file in iter_supported_files(root)}
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
    for source in file_diff.deleted + file_diff.changed:
        # Delete before re-add (or after removal from disk): stale vectors
        # would otherwise linger beside the fresh ones.
        store.delete_by_source(source)

    chunks_stored = 0
    files_indexed = 0
    for source in file_diff.added + file_diff.changed:
        start = perf_counter()
        chunks = process_file(Path(source), root)
        if chunks:
            store.add_documents(chunks)
            chunks_stored += len(chunks)
            files_indexed += 1
            manifest.record(root, source, current[source], len(chunks))
            # Timing spans split AND embedding+store — the network round-trip
            # to Ollama is the slow part worth seeing.
            c.debug(
                f"Indexed {len(chunks)} chunks from {source} "
                f"in {perf_counter() - start:.1f}s"
            )
        # Files that failed or yielded no chunks stay out of the manifest
        # and are retried next run (per-file error tolerance).
        save_manifest(manifest)  # persist per file: a crash resumes here

    for source in file_diff.deleted:
        manifest.remove(root, source)
    save_manifest(manifest)

    return IndexResult(
        chunks_stored=chunks_stored,
        files_indexed=files_indexed,
        files_unchanged=len(file_diff.unchanged),
        files_deleted=len(file_diff.deleted),
    )


def answer(question: str) -> str:
    """Retrieve relevant chunks (Step 4 pipeline) and answer using their contents."""
    relevant = retrieve(question)

    if not relevant:
        # Empty store *or* e.g. BM25 finding no term overlap with the query.
        return (
            "No relevant chunks found. If the index is empty, run the "
            "index command first."
        )

    context = format_context(relevant)
    c.debug_block("context", context)
    budget = get_settings().max_context_chars
    if len(context) > budget:
        c.debug(f"Context: {len(context):,} chars cut to budget {budget:,}")
        context = apply_budget(context, budget)
    else:
        c.debug(f"Context: {len(context):,} chars (budget {budget:,} not reached)")

    messages = [
        ("system", ANSWER_SYSTEM_PROMPT),
        ("human", f"Context:\n{context}\n\nQuestion: {question}"),
    ]
    response = traced_invoke(
        "answer", describe_answer_backend(), get_answer_llm(), messages
    )
    answer_text = response.content or ""
    c.debug(f"Answer: {len(answer_text):,} chars")
    return answer_text

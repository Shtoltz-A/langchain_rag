"""Tests for per-file ingestion entry points: iter_supported_files, process_file.

Step 3.2b: incremental indexing needs per-file enumeration and a single-file
pipeline whose output matches the batch chain file-for-file.
"""

import json
from pathlib import Path

from langchain_rag.ingestion import (
    apply_context_prefixes,
    iter_supported_files,
    load_documents,
    normalize_documents,
    process_file,
    split_documents,
)


def _pairs(documents: list) -> list[tuple[str, dict]]:
    """Reduce documents to (page_content, metadata) for chunk-for-chunk diffs."""
    return [(d.page_content, d.metadata) for d in documents]


def test_iter_supported_files_enumerates_visible_loadable_files(
    tmp_path: Path,
) -> None:
    """Sorted recursive scan: files only, hidden paths and bad suffixes excluded."""
    (tmp_path / "sub" / "deep").mkdir(parents=True)
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "obj.txt").write_text("no", encoding="utf-8")
    (tmp_path / ".hidden.md").write_text("no", encoding="utf-8")
    (tmp_path / "looks_like.txt").mkdir()  # directory with a file-like name
    (tmp_path / "notes.docx").write_text("no loader for this", encoding="utf-8")
    (tmp_path / "image.png").write_bytes(b"\x89PNG")
    visible = {
        "b.txt": "b",
        "a.txt": "a",
        "sub/c.md": "# c",
        "sub/deep/d.py": "x = 1\n",
        "DATA.JSON": '{"a": 1}',  # uppercase suffix still counts
    }
    for rel, text in visible.items():
        tmp_path.joinpath(*rel.split("/")).write_text(text, encoding="utf-8")

    expected = sorted(tmp_path.joinpath(*rel.split("/")) for rel in visible)
    assert iter_supported_files(tmp_path) == expected
    assert iter_supported_files(tmp_path) == iter_supported_files(tmp_path)  # stable


def test_process_file_matches_the_batch_pipeline(tmp_path: Path) -> None:
    """process_file(path, root) equals the batch chain filtered per source."""
    sub = tmp_path / "sub"
    sub.mkdir()
    (tmp_path / "a.txt").write_text(
        "Plain text file. " + " ".join(f"word{i}" for i in range(60)),
        encoding="utf-8",
    )
    (sub / "b.md").write_text(
        "# Guide\n\nIntro prose "
        + "lorem ipsum " * 20
        + "\n\n## Deployment\n\nDeploy steps "
        + "dolor sit " * 20
        + "\n\n### Rollback\n\nRollback notes "
        + "consectetur " * 20,
        encoding="utf-8",
    )
    (tmp_path / "c.json").write_text(
        json.dumps({"alpha": {"x": 1}, "beta": [1, 2, 3], "gamma": "text"}),
        encoding="utf-8",
    )
    (tmp_path / "d.py").write_text(
        "def f():\n    return 1\n\n\ndef g():\n    return 2\n", encoding="utf-8"
    )

    batch = apply_context_prefixes(
        split_documents(normalize_documents(load_documents(tmp_path)))
    )
    assert batch, "batch pipeline produced no chunks"

    relative_paths = ["a.txt", "sub/b.md", "c.json", "d.py"]
    for rel in relative_paths:
        path = tmp_path.joinpath(*rel.split("/"))
        per_file = process_file(path, tmp_path)
        expected = [d for d in batch if d.metadata["source"] == str(path)]
        assert expected, f"batch produced no chunks for {rel}"
        assert _pairs(per_file) == _pairs(expected), f"parity broken for {rel}"
        # explicit metadata spot-checks (source/relative_path anchoring)
        for chunk in per_file:
            assert chunk.metadata["source"] == str(path)
            assert chunk.metadata["relative_path"] == rel


def test_process_file_returns_empty_list_for_unusable_files(tmp_path: Path) -> None:
    """Never raises: unsupported, malformed and vanished files yield []."""
    unsupported = tmp_path / "notes.docx"
    unsupported.write_text("nope", encoding="utf-8")
    assert process_file(unsupported, tmp_path) == []

    malformed = tmp_path / "bad.json"
    malformed.write_text("{ nope", encoding="utf-8")
    assert process_file(malformed, tmp_path) == []

    gone = tmp_path / "gone.txt"
    gone.write_text("was here", encoding="utf-8")
    gone.unlink()  # deleted after enumeration, before processing
    assert process_file(gone, tmp_path) == []

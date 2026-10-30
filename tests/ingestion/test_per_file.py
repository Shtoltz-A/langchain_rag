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


class TestIterSupportedFiles:
    def test_recurses_and_sorts_deterministically(self, tmp_path: Path):
        (tmp_path / "sub" / "deep").mkdir(parents=True)
        (tmp_path / "b.txt").write_text("b", encoding="utf-8")
        (tmp_path / "a.txt").write_text("a", encoding="utf-8")
        (tmp_path / "sub" / "c.md").write_text("# c", encoding="utf-8")
        (tmp_path / "sub" / "deep" / "d.py").write_text("x = 1\n", encoding="utf-8")
        expected = sorted(
            [
                tmp_path / "a.txt",
                tmp_path / "b.txt",
                tmp_path / "sub" / "c.md",
                tmp_path / "sub" / "deep" / "d.py",
            ]
        )
        assert iter_supported_files(tmp_path) == expected
        # stable across repeated calls
        assert iter_supported_files(tmp_path) == iter_supported_files(tmp_path)

    def test_hidden_files_and_folders_are_skipped(self, tmp_path: Path):
        (tmp_path / ".git").mkdir()
        (tmp_path / ".git" / "obj.txt").write_text("no", encoding="utf-8")
        (tmp_path / ".hidden.md").write_text("no", encoding="utf-8")
        visible = tmp_path / "real.txt"
        visible.write_text("yes", encoding="utf-8")
        assert iter_supported_files(tmp_path) == [visible]

    def test_unsupported_suffixes_are_excluded_silently(self, tmp_path: Path):
        (tmp_path / "notes.docx").write_text("no", encoding="utf-8")
        (tmp_path / "image.png").write_bytes(b"\x89PNG")
        supported = tmp_path / "readme.md"
        supported.write_text("# hi", encoding="utf-8")
        assert iter_supported_files(tmp_path) == [supported]

    def test_uppercase_suffix_is_included(self, tmp_path: Path):
        upper = tmp_path / "DATA.JSON"
        upper.write_text('{"a": 1}', encoding="utf-8")
        assert iter_supported_files(tmp_path) == [upper]

    def test_directories_are_excluded(self, tmp_path: Path):
        (tmp_path / "looks_like.txt").mkdir()  # directory with a file-like name
        file = tmp_path / "real.txt"
        file.write_text("real", encoding="utf-8")
        assert iter_supported_files(tmp_path) == [file]


class TestProcessFileParity:
    """process_file(path, root) must equal the batch chain filtered per source."""

    def test_chunk_for_chunk_parity_with_batch_pipeline(self, tmp_path: Path):
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


class TestProcessFileTolerance:
    def test_malformed_json_returns_empty_list(self, tmp_path: Path):
        file = tmp_path / "bad.json"
        file.write_text("{ nope", encoding="utf-8")
        assert process_file(file, tmp_path) == []

    def test_unsupported_suffix_returns_empty_list(self, tmp_path: Path):
        file = tmp_path / "notes.docx"
        file.write_text("nope", encoding="utf-8")
        assert process_file(file, tmp_path) == []

    def test_vanished_file_returns_empty_list(self, tmp_path: Path):
        file = tmp_path / "gone.txt"
        file.write_text("was here", encoding="utf-8")
        file.unlink()  # deleted after enumeration, before processing
        assert process_file(file, tmp_path) == []

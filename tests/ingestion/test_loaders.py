"""Unit tests for ingestion.loaders (loaders per format, registry, walking)."""

import json
from pathlib import Path

import pytest

from langchain_rag.ingestion.loaders import (
    JsonLoader,
    PythonLoader,
    TextFileLoader,
    get_loader,
    load_documents,
)


def test_each_loader_reads_its_file(tmp_path: Path) -> None:
    """Every loader yields its file's text plus source/relative_path metadata."""
    (tmp_path / "sub").mkdir()
    files = {
        "mod.py": "def f():\n    return 1\n",
        "sub/notes.txt": "plain body",
        "sub/readme.md": "# A\nmarkdown body",
        "sub/data.json": json.dumps({"alpha": 1, "beta": [1, 2]}),
    }
    for rel, text in files.items():
        path = tmp_path.joinpath(*rel.split("/"))
        path.write_text(text, encoding="utf-8")
        doc = get_loader(path)(path, root=tmp_path).load()[0]
        assert doc.page_content == text
        assert doc.metadata["source"] == str(path)
        assert doc.metadata["relative_path"] == rel  # POSIX, anchored on root

    # Dispatch is by suffix and case-insensitive; unknown suffixes get no loader.
    assert get_loader(Path("x/y.JSON")) is JsonLoader
    assert get_loader(Path("x/y.py")) is PythonLoader
    assert get_loader(Path("x/y.rst")) is None

    # root defaults to the file's own directory → relative_path is the basename.
    loose = tmp_path / "sub" / "loose.txt"
    loose.write_text("hi", encoding="utf-8")
    assert TextFileLoader(loose).load()[0].metadata["relative_path"] == "loose.txt"


def test_json_loader_rejects_malformed_json(tmp_path: Path) -> None:
    """Parsing happens at load time so a broken file fails early and loudly."""
    file = tmp_path / "bad.json"
    file.write_text("{ not valid json", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        JsonLoader(file).load()


def test_load_documents_walks_recursively_in_sorted_order(tmp_path: Path) -> None:
    """Subdirectories to any depth, sorted, hidden paths skipped silently."""
    (tmp_path / "md").mkdir()
    (tmp_path / "pdf" / "deep").mkdir(parents=True)
    (tmp_path / ".git" / "objects").mkdir(parents=True)
    (tmp_path / "root.txt").write_text("root file", encoding="utf-8")
    (tmp_path / "md" / "a.md").write_text("# A", encoding="utf-8")
    (tmp_path / "pdf" / "deep" / "b.txt").write_text("deep file", encoding="utf-8")
    (tmp_path / ".git" / "objects" / "x.md").write_text("# hidden", encoding="utf-8")
    (tmp_path / ".hidden.txt").write_text("dotfile", encoding="utf-8")

    docs = load_documents(tmp_path)

    # Hidden directories and dotfiles are skipped; the rest come sorted, with
    # POSIX relative_path from the root and the full source path.
    assert [d.metadata["relative_path"] for d in docs] == [
        "md/a.md",
        "pdf/deep/b.txt",
        "root.txt",
    ]
    assert docs[0].metadata["source"] == str(tmp_path / "md" / "a.md")


def test_load_documents_skips_broken_and_unsupported_files(tmp_path: Path) -> None:
    """One broken or unloadable file is reported and skipped, the run continues."""
    sub = tmp_path / "json"
    sub.mkdir()
    (sub / "bad.json").write_text("{ nope", encoding="utf-8")
    (sub / "good.json").write_text('{"a": 1}', encoding="utf-8")
    (tmp_path / "mod.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "notes.docx").write_text("no such loader", encoding="utf-8")

    docs = load_documents(tmp_path)

    assert [d.metadata["relative_path"] for d in docs] == ["json/good.json", "mod.py"]

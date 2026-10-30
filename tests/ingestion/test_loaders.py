"""Unit tests for ingestion.loaders (new .py/.json loaders and registry)."""

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


def test_python_loader_reads_source(tmp_path: Path):
    file = tmp_path / "mod.py"
    file.write_text("def f():\n    return 1\n", encoding="utf-8")
    docs = PythonLoader(file).load()
    assert len(docs) == 1
    assert "def f()" in docs[0].page_content
    assert docs[0].metadata["source"] == str(file)


def test_json_loader_accepts_valid_json(tmp_path: Path):
    file = tmp_path / "data.json"
    payload = {"alpha": 1, "beta": [1, 2]}
    file.write_text(json.dumps(payload), encoding="utf-8")
    docs = JsonLoader(file).load()
    assert json.loads(docs[0].page_content) == payload
    assert docs[0].metadata["source"] == str(file)


def test_json_loader_rejects_malformed_json(tmp_path: Path):
    file = tmp_path / "bad.json"
    file.write_text("{ not valid json", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        JsonLoader(file).load()


def test_registry_dispatch(tmp_path: Path):
    assert get_loader(Path("x/y.py")) is PythonLoader
    assert get_loader(Path("x/y.JSON")) is JsonLoader  # case-insensitive


def test_load_documents_skips_broken_json(tmp_path: Path):
    (tmp_path / "good.json").write_text('{"a": 1}', encoding="utf-8")
    (tmp_path / "bad.json").write_text("{ nope", encoding="utf-8")
    (tmp_path / "mod.py").write_text("x = 1\n", encoding="utf-8")
    docs = load_documents(tmp_path)
    sources = [Path(d.metadata["source"]).name for d in docs]
    assert sources == ["good.json", "mod.py"]  # bad.json skipped, run continues


class TestRecursiveLoading:
    def test_nested_files_are_found(self, tmp_path: Path):
        (tmp_path / "md").mkdir()
        (tmp_path / "pdf" / "deep").mkdir(parents=True)
        (tmp_path / "root.txt").write_text("root file", encoding="utf-8")
        (tmp_path / "md" / "a.md").write_text("# A", encoding="utf-8")
        (tmp_path / "pdf" / "deep" / "b.txt").write_text("deep file", encoding="utf-8")
        docs = load_documents(tmp_path)
        rels = [d.metadata["relative_path"] for d in docs]
        assert rels == ["md/a.md", "pdf/deep/b.txt", "root.txt"]

    def test_relative_path_is_posix_and_source_is_full(self, tmp_path: Path):
        sub = tmp_path / "json"
        sub.mkdir()
        file = sub / "data.json"
        file.write_text('{"a": 1}', encoding="utf-8")
        doc = load_documents(tmp_path)[0]
        assert doc.metadata["relative_path"] == "json/data.json"
        assert doc.metadata["source"] == str(file)

    def test_root_level_file_relative_path_is_basename(self, tmp_path: Path):
        (tmp_path / "loose.py").write_text("x = 1\n", encoding="utf-8")
        doc = load_documents(tmp_path)[0]
        assert doc.metadata["relative_path"] == "loose.py"

    def test_hidden_directories_and_files_are_skipped(self, tmp_path: Path):
        (tmp_path / ".git" / "objects").mkdir(parents=True)
        (tmp_path / ".git" / "objects" / "x.md").write_text(
            "# hidden", encoding="utf-8"
        )
        (tmp_path / ".hidden.txt").write_text("dotfile", encoding="utf-8")
        (tmp_path / "real.txt").write_text("real", encoding="utf-8")
        docs = load_documents(tmp_path)
        assert [d.metadata["relative_path"] for d in docs] == ["real.txt"]

    def test_broken_file_in_subfolder_does_not_abort(self, tmp_path: Path):
        sub = tmp_path / "json"
        sub.mkdir()
        (sub / "bad.json").write_text("{ nope", encoding="utf-8")
        (sub / "good.json").write_text('{"a": 1}', encoding="utf-8")
        (tmp_path / "mod.py").write_text("x = 1\n", encoding="utf-8")
        docs = load_documents(tmp_path)
        rels = [d.metadata["relative_path"] for d in docs]
        assert rels == ["json/good.json", "mod.py"]

    def test_loader_root_defaults_to_file_parent(self, tmp_path: Path):
        sub = tmp_path / "docs"
        sub.mkdir()
        file = sub / "n.txt"
        file.write_text("hi", encoding="utf-8")
        doc = TextFileLoader(file).load()[0]
        assert doc.metadata["relative_path"] == "n.txt"

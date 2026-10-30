"""Unit tests for ingestion.cleaning (pure logic: no Ollama, no Chroma)."""

from pathlib import Path

from langchain_core.documents import Document

from langchain_rag.ingestion.cleaning import apply_context_prefixes, normalize_documents


def _doc(text: str, suffix: str, **metadata) -> Document:
    return Document(
        page_content=text, metadata={"source": f"some/dir/file{suffix}", **metadata}
    )


class TestNormalize:
    def test_collapses_blank_runs_and_trims(self):
        doc = _doc("line1   \n\n\n\nline2\n\n\n", ".txt")
        out = normalize_documents([doc])[0]
        assert out.page_content == "line1\n\nline2"

    def test_normalizes_crlf(self):
        doc = _doc("a\r\nb\r\n\r\n\r\nc", ".txt")
        assert normalize_documents([doc])[0].page_content == "a\nb\n\nc"

    def test_hyphenation_joined_for_pdf(self):
        doc = _doc("this is a configu-\nration example", ".pdf", page=0)
        out = normalize_documents([doc])[0]
        assert "configuration" in out.page_content

    def test_hyphenation_not_joined_for_markdown(self):
        doc = _doc("a code-\nblock reference", ".md")
        out = normalize_documents([doc])[0]
        assert "code-\nblock" in out.page_content

    def test_empty_input(self):
        assert normalize_documents([]) == []

    def test_no_mutation_of_input(self):
        doc = _doc("a\n\n\n\nb", ".txt")
        normalize_documents([doc])
        assert doc.page_content == "a\n\n\n\nb"

    def test_repeated_pdf_header_footer_dropped(self):
        pages = [
            _doc(
                f"Confidential\n\npage {i} body text unique {i}\n\nPage 1 of 9",
                ".pdf",
                page=i,
            )
            for i in range(5)
        ]
        out = normalize_documents(pages)
        assert all("Confidential" not in d.page_content for d in out)
        assert all("Page 1 of 9" not in d.page_content for d in out)
        # Unique body text survives.
        assert any("page 2 body text unique 2" in d.page_content for d in out)

    def test_rare_line_not_dropped(self):
        # "Intro line" appears on only 2 of 5 pages (below both the count and
        # ratio thresholds), so it is treated as content and kept.
        pages = [
            _doc(
                f"Intro line\n\npage {i} body content different {i}"
                if i < 2
                else f"page {i} body content different {i}",
                ".pdf",
                page=i,
            )
            for i in range(5)
        ]
        out = normalize_documents(pages)
        assert any("Intro line" in d.page_content for d in out)

    def test_few_pdf_pages_keep_all_lines(self):
        # Below min_count pages, no boilerplate detection runs at all.
        pages = [_doc(f"Shared Header\n\nbody {i}", ".pdf", page=i) for i in range(2)]
        out = normalize_documents(pages)
        assert all("Shared Header" in d.page_content for d in out)

    def test_cleaning_disabled_returns_documents_unchanged(self, monkeypatch):
        monkeypatch.setattr(
            "langchain_rag.ingestion.cleaning._settings.cleaning_enabled", False
        )
        doc = _doc("a\n\n\n\nb   ", ".pdf", page=0)
        out = normalize_documents([doc])[0]
        assert out.page_content == "a\n\n\n\nb   "


class TestContextPrefixes:
    def test_txt_prefix(self):
        chunk = _doc("body", ".txt")
        assert apply_context_prefixes([chunk])[0].page_content.startswith(
            "[File: file.txt]\n"
        )

    def test_pdf_prefix_with_page(self):
        chunk = _doc("body", ".pdf", page=11)
        assert apply_context_prefixes([chunk])[0].page_content.startswith(
            "[Doc: file.pdf p.12]\n"
        )

    def test_markdown_prefix_with_heading_chain(self):
        chunk = _doc("body", ".md", h1="Runbook", h2="Deployment")
        assert (
            apply_context_prefixes([chunk])[0].page_content
            == "[File: file.md > Runbook > Deployment]\nbody"
        )

    def test_markdown_prefix_without_headings(self):
        chunk = _doc("body", ".md")
        assert apply_context_prefixes([chunk])[0].page_content.startswith(
            "[File: file.md]\n"
        )

    def test_json_prefix_with_key(self):
        chunk = _doc("body", ".json", json_key="database")
        assert apply_context_prefixes([chunk])[0].page_content.startswith(
            "[File: file.json · key: database]\n"
        )

    def test_code_prefix(self):
        chunk = _doc("body", ".py")
        assert apply_context_prefixes([chunk])[0].page_content.startswith(
            "[File: file.py]\n"
        )

    def test_prefix_uses_basename_not_full_path(self):
        doc = Document(
            page_content="body",
            metadata={"source": str(Path("/deep/nested/dir/notes.txt"))},
        )
        out = apply_context_prefixes([doc])[0].page_content
        assert out == "[File: notes.txt]\nbody"

    def test_idempotent_no_double_prefix(self):
        chunk = _doc("body", ".txt")
        once = apply_context_prefixes([chunk])[0]
        twice = apply_context_prefixes([once])[0]
        assert twice.page_content == once.page_content

    def test_metadata_preserved(self):
        chunk = _doc("body", ".pdf", page=3)
        out = apply_context_prefixes([chunk])[0]
        assert out.metadata["page"] == 3
        assert "source" in out.metadata

    def test_empty_input(self):
        assert apply_context_prefixes([]) == []

    def test_cleaning_disabled_leaves_chunks_unchanged(self, monkeypatch):
        monkeypatch.setattr(
            "langchain_rag.ingestion.cleaning._settings.cleaning_enabled", False
        )
        chunk = _doc("body", ".txt")
        assert apply_context_prefixes([chunk])[0].page_content == "body"


class TestFolderTrailInPrefixes:
    """Prefixes show the folder grouping from relative_path, when present."""

    def _rel(self, text: str, rel: str, **metadata) -> Document:
        return Document(
            page_content=text,
            metadata={"source": f"some/dir/{rel}", "relative_path": rel, **metadata},
        )

    def test_pdf_prefix_includes_folder(self):
        chunk = self._rel("body", "pdf/report.pdf", page=0)
        out = apply_context_prefixes([chunk])[0].page_content
        assert out == "[Doc: pdf/report.pdf p.1]\nbody"

    def test_markdown_prefix_folder_and_headings(self):
        chunk = self._rel("body", "md/runbook.md", h1="Runbook", h2="Deployment")
        out = apply_context_prefixes([chunk])[0].page_content
        assert out == "[File: md/runbook.md > Runbook > Deployment]\nbody"

    def test_json_prefix_folder_and_key(self):
        chunk = self._rel("body", "json/config.json", json_key="database")
        out = apply_context_prefixes([chunk])[0].page_content
        assert out.startswith("[File: json/config.json \u00b7 key: database]\n")

    def test_nested_folders_kept_in_trail(self):
        chunk = self._rel("body", "repo/docs/deep.txt")
        out = apply_context_prefixes([chunk])[0].page_content
        assert out == "[File: repo/docs/deep.txt]\nbody"

    def test_falls_back_to_basename_without_relative_path(self):
        chunk = _doc("body", ".txt")  # no relative_path metadata
        out = apply_context_prefixes([chunk])[0].page_content
        assert out == "[File: file.txt]\nbody"

    def test_idempotent_with_folder_trail(self):
        chunk = self._rel("body", "pdf/report.pdf", page=4)
        once = apply_context_prefixes([chunk])[0]
        twice = apply_context_prefixes([once])[0]
        assert twice.page_content == once.page_content

"""Tests for app/rag/ingestion/parsers/ — detect_parser, registry, and format-specific parsing."""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Helper: ensure sys.path so local imports work regardless of launch dir.
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rag.ingestion.parsers.base import BaseParser, detect_parser
from app.rag.ingestion.parsers.txt_parser import TextParser
from app.rag.ingestion.parsers.md_parser import MarkdownParser


# ===================================================================
# detect_parser
# ===================================================================

class TestDetectParser:
    @pytest.mark.parametrize(
        "filename,expected_name",
        [
            ("doc.txt", "txt"),
            ("doc.md", "md"),
            ("doc.markdown", "md"),
            ("doc.html", "html"),
            ("doc.htm", "html"),
            ("doc.docx", "docx"),
            ("doc.pdf", "pdf"),
        ],
    )
    def test_detect_parser_known_extensions(self, tmp_path, filename, expected_name):
        f = tmp_path / filename
        f.write_text("hello")
        parser = detect_parser(f)
        assert parser is not None, f"No parser detected for {filename}"
        assert parser.NAME == expected_name

    def test_detect_parser_unknown_extension(self, tmp_path):
        f = tmp_path / "doc.xyz"
        f.write_text("hello")
        assert detect_parser(f) is None


# ===================================================================
# Parser registry
# ===================================================================

class TestParserRegistry:
    def test_parser_registry_all_registered(self):
        """Every concrete parser should have an entry in the registry."""
        from app.rag.ingestion.parsers.base import _PARSER_REGISTRY, _EXT_INDEX

        expected_names = {"txt", "md", "html", "docx", "pdf"}
        assert expected_names.issubset(set(_PARSER_REGISTRY.keys()))
        # Extension index should cover all common extensions
        for ext in {".txt", ".md", ".markdown", ".html", ".htm", ".docx", ".pdf"}:
            assert ext in _EXT_INDEX, f"Extension {ext} not in extension index"


# ===================================================================
# TextParser
# ===================================================================

class TestTextParserParse:
    def test_text_parser_parse(self, tmp_path):
        content = "This is a plain text document.\nIt has two lines."
        f = tmp_path / "sample.txt"
        f.write_text(content)

        parser = TextParser()
        docs = parser.parse(f)

        assert len(docs) == 1
        doc = docs[0]
        assert doc.text == content
        assert doc.metadata["source_path"] == str(f)
        assert doc.metadata["file_type"] == "txt"


# ===================================================================
# MarkdownParser
# ===================================================================

class TestMarkdownParserParse:
    def test_markdown_parser_parse(self, tmp_path):
        content = (
            "# Title\n"
            "Some content.\n\n"
            "## Section 1\n"
            "More content.\n\n"
            "### Subsection\n"
            "Detail.\n"
        )
        f = tmp_path / "sample.md"
        f.write_text(content)

        parser = MarkdownParser()
        docs = parser.parse(f)

        assert len(docs) == 1
        doc = docs[0]
        assert doc.text == content
        assert doc.metadata["file_type"] == "md"

    def test_parse_returns_sections(self, tmp_path):
        content = (
            "# Introduction\n"
            "Body text here.\n\n"
            "## Methods\n"
            "Details.\n\n"
            "### Caveats\n"
            "More.\n"
        )
        f = tmp_path / "doc.md"
        f.write_text(content)

        parser = MarkdownParser()
        docs = parser.parse(f)
        sections = docs[0].metadata["sections"]

        assert len(sections) == 3
        assert sections[0]["heading"] == "Introduction"
        assert sections[0]["level"] == 1
        assert sections[1]["heading"] == "Methods"
        assert sections[1]["level"] == 2
        assert sections[2]["heading"] == "Caveats"
        assert sections[2]["level"] == 3


# ===================================================================
# Shared metadata tests
# ===================================================================

class TestParserMetadata:
    def test_parser_returns_metadata(self, tmp_path):
        """Both txt and md parsers should return source_path, file_type, title."""
        f = tmp_path / "my_doc.txt"
        f.write_text("hello")

        parser = TextParser()
        docs = parser.parse(f)

        meta = docs[0].metadata
        assert meta["source_path"] == str(f)
        assert meta["file_type"] == "txt"
        assert meta["title"] == "my_doc"


# ===================================================================
# BaseParser.can_handle
# ===================================================================

class TestBaseParserCanHandle:
    def test_can_handle_true(self, tmp_path):
        f = tmp_path / "test.txt"
        parser = TextParser()
        assert parser.can_handle(f) is True

    def test_can_handle_false(self, tmp_path):
        f = tmp_path / "test.pdf"
        parser = TextParser()
        assert parser.can_handle(f) is False

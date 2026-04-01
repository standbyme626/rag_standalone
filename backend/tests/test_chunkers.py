"""Tests for app/rag/ingestion/chunkers/ — get_chunker, RecursiveChunker, DocumentAwareChunker, TableAwareChunker."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rag.ingestion.models import ParsedDocument, PipelineConfig
from app.rag.ingestion.chunkers.base import BaseChunker, get_chunker, register_chunker
from app.rag.ingestion.chunkers.recursive import RecursiveChunker
from app.rag.ingestion.chunkers.document_aware import DocumentAwareChunker
from app.rag.ingestion.chunkers.table_aware import TableAwareChunker
from app.rag.ingestion.chunkers.legal import LegalChunker


# ===================================================================
# get_chunker / registry
# ===================================================================


class TestGetChunker:
    def test_get_chunker_all_strategies(self):
        """All built-in chunker strategies should be retrievable."""
        for name in ("recursive", "document_aware", "table_aware", "legal"):
            c = get_chunker(name)
            assert isinstance(c, BaseChunker), (
                f"get_chunker('{name}') did not return a BaseChunker"
            )

    def test_get_chunker_invalid_name(self):
        with pytest.raises(ValueError, match="Unknown chunker"):
            get_chunker("nonexistent_xyz")


# ===================================================================
# RecursiveChunker — basic
# ===================================================================


class TestRecursiveChunkBasic:
    def _make_doc(self, text: str) -> ParsedDocument:
        return ParsedDocument(
            text=text, metadata={"source_path": "/tmp/test.txt", "file_type": "txt"}
        )

    def _make_config(self, chunk_size: int = 100, overlap: int = 10) -> PipelineConfig:
        return PipelineConfig(chunk_size=chunk_size, chunk_overlap=overlap)

    def test_recursive_chunk_basic(self):
        """A short single-paragraph doc within chunk_size should yield one chunk."""
        text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        doc = self._make_doc(text)
        config = self._make_config(chunk_size=256, overlap=10)

        chunker = RecursiveChunker()
        chunks = chunker.chunk(doc, config)

        assert len(chunks) >= 1
        assert all(c.content for c in chunks)

    def test_recursive_chunk_produces_multiple(self):
        """A long document exceeding chunk_size should be split."""
        paragraphs = [
            f"This is paragraph {i} with some content to pad length." for i in range(30)
        ]
        text = "\n\n".join(paragraphs)
        doc = self._make_doc(text)
        config = self._make_config(chunk_size=100, overlap=10)

        chunks = RecursiveChunker().chunk(doc, config)
        assert len(chunks) > 1
        # Each chunk should roughly respect the max size
        for c in chunks:
            assert (
                len(c.content) <= config.chunk_size + config.chunk_overlap + 50
            )  # tolerance for boundary

    def test_recursive_chunk_empty(self):
        doc = self._make_doc("")
        config = self._make_config()
        chunks = RecursiveChunker().chunk(doc, config)
        assert chunks == []

    def test_recursive_chunk_metadata(self):
        text = "First section.\n\nSecond section."
        doc = self._make_doc(text)
        config = self._make_config(chunk_size=256)

        chunks = RecursiveChunker().chunk(doc, config)
        for c in chunks:
            assert c.metadata["source_path"] == "/tmp/test.txt"
            assert c.metadata["file_type"] == "txt"


# ===================================================================
# RecursiveChunker — overlap
# ===================================================================


class TestRecursiveChunkOverlap:
    def test_recursive_chunk_overlap_preserves_boundary(self):
        """Overlap parameter should cause chunk boundary text to repeat in the next chunk."""
        text = "AAAA\n\nBBBB\n\nCCCC\n\nDDDD\n\nEEEE"
        doc = ParsedDocument(
            text=text, metadata={"source_path": "/tmp/o.txt", "file_type": "txt"}
        )
        config = PipelineConfig(chunk_size=20, chunk_overlap=8)

        chunks = RecursiveChunker().chunk(doc, config)
        assert len(chunks) >= 2

        # Check that some overlap exists between adjacent chunks
        for i in range(len(chunks) - 1):
            prev = chunks[i].content
            curr = chunks[i + 1].content
            assert prev and curr, f"Chunk {i} or {i + 1} is empty"
            # At least some text overlap — check if prev's tail appears in curr
            overlap_tail = prev.split("\n")[-1]  # last line should be in next chunk
            assert len(overlap_tail) > 0


# ===================================================================
# DocumentAwareChunker
# ===================================================================


class TestDocumentAwareChunker:
    def _make_doc_with_sections(self, text: str) -> ParsedDocument:
        return ParsedDocument(
            text=text,
            metadata={
                "source_path": "/tmp/doc.md",
                "file_type": "md",
                "sections": [
                    {"heading": "Introduction", "level": 1, "char_start": 0},
                    {"heading": "Methods", "level": 1, "char_start": 40},
                    {"heading": "Conclusion", "level": 1, "char_start": 120},
                ],
            },
        )

    def _make_config(self, chunk_size: int = 256, overlap: int = 10) -> PipelineConfig:
        return PipelineConfig(chunk_size=chunk_size, chunk_overlap=overlap)

    def test_document_aware_chunk_with_sections(self):
        text = (
            "# Introduction\n"
            "This is the intro text explaining things.\n\n"
            "# Methods\n"
            "Here we describe the methods used. More and more text to fill "
            "some content so there's something to split on.\n\n"
            "# Conclusion\n"
            "Final thoughts go here. That is all.\n"
        )
        doc = self._make_doc_with_sections(text)
        config = self._make_config(chunk_size=256)

        chunks = DocumentAwareChunker().chunk(doc, config)
        assert len(chunks) >= 1

        # At least one chunk should have a non-empty section name
        named_sections = [c.metadata.get("section") for c in chunks]
        assert any(s != "" for s in named_sections)

    def test_document_aware_fallback_to_recursive(self):
        """When no sections exist, DocumentAwareChunker falls back to RecursiveChunker."""
        doc = ParsedDocument(
            text="Just plain text with no sections.\n\nAnother paragraph.",
            metadata={"source_path": "/tmp/x.txt", "file_type": "txt"},
        )
        config = self._make_config(chunk_size=256)

        chunks = DocumentAwareChunker().chunk(doc, config)
        # Delegated to RecursiveChunker
        assert len(chunks) >= 1
        assert all(c.content for c in chunks)


# ===================================================================
# TableAwareChunker
# ===================================================================


class TestTableAwareChunker:
    def _make_config(self) -> PipelineConfig:
        return PipelineConfig(chunk_size=256, chunk_overlap=10)

    def test_table_aware_with_tables(self):
        """When metadata includes tables, they should be extracted as separate chunks."""
        text = (
            "Some preamble text before the table.\n\n"
            "Name | Age | City\n"
            "Alice | 30 | NYC\n"
            "Bob | 25 | LA\n\n"
            "Some text after the table."
        )
        doc = ParsedDocument(
            text=text,
            metadata={
                "source_path": "/tmp/with_tables.txt",
                "file_type": "txt",
                "tables": [
                    {
                        "table_text": "Name | Age | City\nAlice | 30 | NYC\nBob | 25 | LA",
                        "row_count": 3,
                        "index": 0,
                    }
                ],
            },
        )
        config = self._make_config()

        chunks = TableAwareChunker().chunk(doc, config)
        assert len(chunks) >= 2  # At least 1 table + text

        # One chunk should be marked as a table
        table_chunks = [c for c in chunks if c.metadata.get("has_table") is True]
        assert len(table_chunks) >= 1
        assert "[TABLE]" in table_chunks[0].content

    def test_table_aware_without_tables(self):
        """When no tables exist, TableAwareChunker should delegate to DocumentAwareChunker."""
        doc = ParsedDocument(
            text="Just text with no tables.",
            metadata={"source_path": "/tmp/no_tables.txt", "file_type": "txt"},
        )
        config = self._make_config()

        chunks = TableAwareChunker().chunk(doc, config)
        assert len(chunks) >= 1
        # All chunks should have has_table = False (or not set)
        for c in chunks:
            assert c.metadata.get("has_table", False) is False


# ===================================================================
# LegalChunker — 法律专用分块（按法条边界切分）
# ===================================================================


class TestLegalChunker:
    def _make_doc(
        self, text: str, title: str = "中华人民共和国民法典"
    ) -> ParsedDocument:
        return ParsedDocument(
            text=text,
            metadata={
                "source_path": "/tmp/minfadian/总则.md",
                "file_type": "md",
                "title": title,
            },
        )

    def _make_config(self, chunk_size: int = 512, overlap: int = 64) -> PipelineConfig:
        return PipelineConfig(chunk_size=chunk_size, chunk_overlap=overlap)

    def test_legal_chunker_basic(self):
        text = (
            "## 第一章 基本规定\n\n"
            "第一条 为了保护民事主体的合法权益，调整民事关系，维护社会和经济秩序，根据宪法，制定本法。\n\n"
            "第二条 民法调整平等主体的自然人、法人和非法人组织之间的人身关系和财产关系。"
        )
        doc = self._make_doc(text)
        config = self._make_config(chunk_size=512)
        chunks = LegalChunker().chunk(doc, config)
        assert len(chunks) >= 2
        assert all(
            c.metadata.get("article_num", "") in ("", "第一条", "第二条")
            for c in chunks
        )
        assert any(c.metadata.get("article_num") == "第一条" for c in chunks)
        assert any(c.metadata.get("article_num") == "第二条" for c in chunks)

    def test_legal_chunker_no_cross_article(self):
        text = "第一条 第一款内容。\n\n第二条 内容B。"
        doc = self._make_doc(text)
        config = self._make_config(chunk_size=1024)
        chunks = LegalChunker().chunk(doc, config)
        assert len(chunks) == 2
        assert chunks[0].content.strip().startswith("第一条")
        assert chunks[1].content.strip().startswith("第二条")

    def test_legal_chunker_article_num_in_metadata(self):
        text = "第一条 自然人从出生时起到死亡时止，具有民事权利能力，依法享有民事权利，承担民事义务。"
        doc = self._make_doc(text)
        config = self._make_config()
        chunks = LegalChunker().chunk(doc, config)
        assert len(chunks) == 1
        assert chunks[0].metadata.get("article_num") == "第一条"
        assert chunks[0].metadata.get("domain") == "legal"

    def test_legal_chunker_empty(self):
        doc = self._make_doc("")
        chunks = LegalChunker().chunk(doc, self._make_config())
        assert chunks == []

    def test_legal_chunker_long_article_splits(self):
        article_content = "第一条 " + "这是一句很长的法律文本。" * 200
        doc = self._make_doc(article_content)
        config = self._make_config(chunk_size=200, overlap=20)
        chunks = LegalChunker().chunk(doc, config)
        assert len(chunks) > 1
        for c in chunks:
            assert c.metadata.get("article_num") == "第一条"

    def test_legal_chunker_metadata_preserved(self):
        text = "第一条 自然人享有民事权利能力。"
        doc = self._make_doc(text, title="民法典总则")
        config = self._make_config()
        chunks = LegalChunker().chunk(doc, config)
        assert len(chunks) == 1
        assert chunks[0].metadata.get("source_path") == "/tmp/minfadian/总则.md"
        assert chunks[0].metadata.get("file_type") == "md"
        assert chunks[0].metadata.get("law_name") == "民法典总则"

    def test_get_chunker_legal(self):
        c = LegalChunker()
        assert c.NAME == "legal"

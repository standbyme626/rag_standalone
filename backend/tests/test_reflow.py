"""Tests for ReflowCleaner (paragraph reflow from UltraRAG)."""

from app.rag.ingestion.cleaners import get_cleaner
from app.rag.ingestion.models import DocumentChunk


class TestReflowCleaner:
    def test_merge_lines_without_sentence_end(self):
        cleaner = get_cleaner("reflow")
        chunks = [
            DocumentChunk(
                content="This is a sentence that continues\nonto the next line without punctuation.",
            )
        ]
        result = cleaner.clean(chunks)
        assert len(result) == 1
        assert "continues onto" in result[0].content
        assert "\n" not in result[0].content

    def test_keep_lines_with_sentence_end(self):
        cleaner = get_cleaner("reflow")
        chunks = [
            DocumentChunk(
                content="First sentence.\nSecond sentence.",
            )
        ]
        result = cleaner.clean(chunks)
        assert len(result) == 1
        # "First sentence." ends with period, should remain as separate
        assert "First sentence." in result[0].content

    def test_handle_hyphenated_line_break(self):
        cleaner = get_cleaner("reflow")
        chunks = [
            DocumentChunk(
                content="This is a long hyphenated-\nword that was split.",
            )
        ]
        result = cleaner.clean(chunks)
        # Hyphenated line break: trailing - removed and word joined directly
        assert "hyphenatedword" in result[0].content

    def test_chinese_sentence_breaks(self):
        cleaner = get_cleaner("reflow")
        chunks = [
            DocumentChunk(
                content="这是一句话。\n这是另一句话。",
            )
        ]
        result = cleaner.clean(chunks)
        assert len(result) == 1
        # Chinese sentences should remain separate (split by blank lines)
        assert "一句话" in result[0].content

    def test_empty_text(self):
        cleaner = get_cleaner("reflow")
        chunks = [DocumentChunk(content="")]
        result = cleaner.clean(chunks)
        assert len(result) == 0

    def test_whitespace_only(self):
        cleaner = get_cleaner("reflow")
        chunks = [DocumentChunk(content="   \n\n  \n  ")]
        result = cleaner.clean(chunks)
        assert len(result) == 0

    def test_reflow_via_get_cleaner(self):
        from app.rag.ingestion.cleaners.reflow import ReflowCleaner
        cleaner = get_cleaner("reflow")
        assert isinstance(cleaner, ReflowCleaner)

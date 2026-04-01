"""Tests for TOC extraction module (Phase 2.3)."""

import asyncio
import pytest
from unittest.mock import AsyncMock

from app.rag.toc.extractor import (
    TocExtractor,
    TocEntry,
    relevant_chunks_with_toc,
    _tokenize,
)


class TestTocEntry:
    def test_entry_to_dict(self):
        entry = TocEntry("第一章 总则", "c1", ids=["c1"])
        d = entry.to_dict()
        assert d["title"] == "第一章 总则"
        assert d["chunk_id"] == "c1"
        assert d["ids"] == ["c1"]
        assert d["level"] == 0
        assert d["score"] == 0.0


class TestTocExtractor:
    def test_detect_level_chinese(self):
        assert TocExtractor._detect_level("第一章 总则") == 1
        assert TocExtractor._detect_level("第二章 定义") == 1

    def test_detect_level_numbered(self):
        assert TocExtractor._detect_level("1. Introduction") == 2
        assert TocExtractor._detect_level("1.2.3 Details") == 3

    def test_detect_level_roman(self):
        assert TocExtractor._detect_level("I. Overview") == 1

    @pytest.mark.asyncio
    async def test_no_llm_returns_empty(self):
        extractor = TocExtractor()
        result = await extractor.extract([
            {"id": "c1", "text": "第一章 总则"},
        ])
        assert result == []

    @pytest.mark.asyncio
    async def test_with_mock_llm(self):
        mock_llm = AsyncMock(return_value='[{"title": "第一章 总则", "chunk_id": "c1"}, {"title": "第二章 定义", "chunk_id": "c2"}]')
        extractor = TocExtractor(llm_call=mock_llm)
        result = await extractor.extract([
            {"id": "c1", "text": "第一章 总则的内容"},
            {"id": "c2", "text": "第二章 定义的内容"},
        ])
        assert len(result) == 2
        assert result[0].title == "第一章 总则"
        assert result[1].title == "第二章 定义"


class TestTokenize:
    def test_chinese(self):
        tokens = _tokenize("糖尿病饮食")
        assert len(tokens) == 5  # each Chinese char

    def test_english(self):
        tokens = _tokenize("diabetes treatment")
        assert "diabetes" in tokens
        assert "treatment" in tokens

    def test_mixed(self):
        tokens = _tokenize("糖尿病diabetes")
        assert len(tokens) > 0


class TestRelevantChunksWithToc:
    def test_no_toc_returns_original(self):
        chunks = [
            {"id": "c1", "text": "Content about topic A"},
            {"id": "c2", "text": "Content about topic B"},
        ]
        result = relevant_chunks_with_toc([], "query", chunks, top_k=1)
        assert len(result) == 1

    def test_toc_match_returns_matched(self):
        entries = [
            TocEntry("糖尿病", "c1", ids=["c1"]),
            TocEntry("高血压", "c2", ids=["c2"]),
        ]
        chunks = [
            {"id": "c1", "text": "Diabetes content"},
            {"id": "c2", "text": "Hypertension content"},
        ]
        result = relevant_chunks_with_toc(entries, "糖尿病", chunks)
        # "糖尿" and "病病" are the char tokens for "糖尿病"
        # The matching should find overlap for c1
        assert len(result) > 0

    def test_no_toc_match_returns_original(self):
        entries = [
            TocEntry("糖尿病", "c1", ids=["c1"]),
        ]
        chunks = [
            {"id": "c1", "text": "Diabetes content"},
            {"id": "c2", "text": "Hypertension content"},
        ]
        result = relevant_chunks_with_toc(entries, "骨折", chunks)
        assert len(result) > 0

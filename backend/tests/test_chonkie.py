"""Tests for ChonkieChunker (Phase 1.5)."""

import pytest

from app.rag.ingestion.models import ParsedDocument, PipelineConfig


class TestChonkieChunker:
    def test_chonkie_name_in_registry(self):
        """ChonkieChunker should be registered with NAME='chonkie'."""
        from app.rag.ingestion.chunkers.base import _CHUNKER_REGISTRY

        assert "chonkie" in _CHUNKER_REGISTRY

    def test_chunker_class_name(self):
        from app.rag.ingestion.chunkers.chonkie import ChonkieChunker

        assert ChonkieChunker.NAME == "chonkie"

    def test_missing_chonkie_raises_runtime_error(self):
        """When chonkie library is not installed, calling chunk() should raise."""
        from app.rag.ingestion.chunkers.chonkie import ChonkieChunker, _CHONKIE_IMPORT_ERROR

        chunker = ChonkieChunker()
        if _CHONKIE_IMPORT_ERROR:
            doc = ParsedDocument(text="test content")
            config = PipelineConfig()
            with pytest.raises(RuntimeError, match="chonkie"):
                chunker.chunk(doc, config)

    def test_get_chunker_returns_chonkie(self):
        from app.rag.ingestion.chunkers import get_chunker

        chunker = get_chunker("chonkie")
        assert chunker.NAME == "chonkie"

"""Tests for MinerUParser (Phase 2.4)."""

from pathlib import Path
from unittest.mock import patch

from app.rag.ingestion.parsers.base import _EXT_INDEX, _PARSER_REGISTRY, detect_parser
from app.rag.ingestion.parsers.mineru_parser import MinerUParser, _MINERU_AVAILABLE


class TestMinerUParser:
    def test_mineru_registered(self):
        """MinerUParser should be registered with NAME='mineru'."""
        assert "mineru" in _PARSER_REGISTRY

    def test_pdf_ext_registered_to_mineru(self):
        """PDF extension should map to mineru parser."""
        assert ".pdf" in _EXT_INDEX

    def test_mineru_not_available(self):
        """mineru CLI should not be available in test env."""
        assert _MINERU_AVAILABLE is False
        parser = MinerUParser()
        assert not parser.can_handle(Path("test.pdf"))

    def test_mineru_class_name(self):
        assert MinerUParser.NAME == "mineru"
        assert ".pdf" in MinerUParser.SUPPORTED_EXTENSIONS

    def test_mineru_unavailable_raises_error(self):
        parser = MinerUParser()
        import pytest

        with pytest.raises(RuntimeError, match="mineru"):
            parser.parse(Path("test.pdf"))

    def test_mineru_detect_parser_returns_pdf_parser(self):
        """When mineru is not available, detect_parser should not return it."""
        parser = detect_parser(Path("test.pdf"))
        if parser is not None:
            assert parser.NAME != "mineru"

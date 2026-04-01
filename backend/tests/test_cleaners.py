"""Tests for app/rag/ingestion/cleaners/ — NoiseFilter, Dedup, PII Redactor."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rag.ingestion.models import DocumentChunk
from app.rag.ingestion.cleaners.noise_filter import NoiseFilterCleaner
from app.rag.ingestion.cleaners.dedup import DedupCleaner
from app.rag.ingestion.cleaners.pii_redactor import PIIRedactorCleaner
from app.core.security.pii import PIIMasker


# ===================================================================
# NoiseFilterCleaner
# ===================================================================

class TestNoiseFilterCleaner:
    def _make_chunk(self, content: str, **kw) -> DocumentChunk:
        return DocumentChunk(content=content, metadata=kw)

    def test_noise_filter_removes_short_chunks(self):
        chunks = [
            self._make_chunk("hi"),              # 2 chars — below MIN_CONTENT_LENGTH (20)
            self._make_chunk("x"),              # 1 char
            self._make_chunk("This chunk has enough content to pass."),
        ]
        result = NoiseFilterCleaner().clean(chunks)
        # Only the long chunk survives
        assert len(result) == 1
        assert "enough content" in result[0].content

    def test_noise_filter_normalizes_newlines(self):
        chunk = self._make_chunk("line1\n\n\n\n\nline2\n\n\nline3")
        [cleaned] = NoiseFilterCleaner().clean([chunk])
        # Should collapse to at most \n\n
        assert "\n\n\n" not in cleaned.content

    def test_noise_filter_collapses_page_breaks(self):
        chunk = self._make_chunk(
            "Before\n\n---PAGE_BREAK---\n\n---PAGE_BREAK---\n\nAfter"
        )
        [cleaned] = NoiseFilterCleaner().clean([chunk])
        # Adjacent PAGE_BREAK markers should be collapsed into one
        assert "---PAGE_BREAK---\n\n---PAGE_BREAK---" not in cleaned.content


# ===================================================================
# DedupCleaner
# ===================================================================

class TestDedupCleaner:
    def _make_chunk(self, content: str, **kw) -> DocumentChunk:
        return DocumentChunk(content=content, metadata=kw)

    def test_dedup_exact_duplicates(self):
        text_a = "This is some unique content about topic A."
        text_b = "This is about a different topic B entirely."
        chunks = [
            self._make_chunk(text_a),
            self._make_chunk(text_b),
            self._make_chunk(text_a),  # exact duplicate
        ]
        result = DedupCleaner().clean(chunks)
        assert len(result) == 2
        contents = {c.content for c in result}
        assert text_a in contents
        assert text_b in contents

    def test_dedup_all_unique(self):
        chunks = [self._make_chunk(f"Content number {i}. Some padding text to vary.") for i in range(10)]
        result = DedupCleaner().clean(chunks)
        # Below SEMANTIC_MIN_CHUNKS threshold, so only exact dedup runs — all are unique
        assert len(result) == 10


# ===================================================================
# PIIRedactorCleaner (delegates to PIIMasker — no settings dependency)
# ===================================================================

class TestPIIRedactorCleaner:
    def _make_chunk(self, content: str, **kw) -> DocumentChunk:
        return DocumentChunk(content=content, metadata=kw)

    def test_pii_redactor_mocks_phone_numbers(self):
        chunk = self._make_chunk("Call me at 13800138000 or 15912345678 for help.")
        [cleaned] = PIIRedactorCleaner().clean([chunk])
        assert "<PHONE>" in cleaned.content
        assert "13800138000" not in cleaned.content
        assert "15912345678" not in cleaned.content

    def test_pii_redactor_masks_emails(self):
        chunk = self._make_chunk("Contact admin@example.com or support@test.org.")
        [cleaned] = PIIRedactorCleaner().clean([chunk])
        assert "<EMAIL>" in cleaned.content
        assert "admin@example.com" not in cleaned.content
        assert "support@test.org" not in cleaned.content

    def test_pii_redactor_preserves_safe_text(self):
        chunk = self._make_chunk("The weather today is sunny and warm.")
        [cleaned] = PIIRedactorCleaner().clean([chunk])
        assert cleaned.content == chunk.content

    def test_pii_redactor_empty_content(self):
        chunk = self._make_chunk("")
        [cleaned] = PIIRedactorCleaner().clean([chunk])
        assert cleaned.content == ""


# ===================================================================
# PIIMasker unit tests (direct)
# ===================================================================

class TestPIIMasker:
    def test_mask_phone(self):
        result = PIIMasker.mask("Phone: 13800138000")
        assert "<PHONE>" in result

    def test_mask_email(self):
        result = PIIMasker.mask("Email: user@test.com")
        assert "<EMAIL>" in result

    def test_mask_id_card(self):
        result = PIIMasker.mask("ID: 110101199001011234")
        assert "<ID_CARD>" in result

    def test_mask_no_pii(self):
        text = "The cat sat on the mat."
        assert PIIMasker.mask(text) == text

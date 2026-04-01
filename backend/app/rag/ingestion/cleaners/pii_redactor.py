"""PII 脱敏清洗器 — 复用 app.core.security.pii.PIIMasker。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.core.security.pii import PIIMasker

if TYPE_CHECKING:
    from ..models import DocumentChunk


class PIIRedactorCleaner:
    """对 chunk 内容执行 PII 脱敏。委托给现有 PIIMasker。"""

    NAME = "pii_redactor"

    def clean(self, chunks: list[DocumentChunk]) -> list[DocumentChunk]:
        for chunk in chunks:
            if chunk.content:
                chunk.content = PIIMasker.mask(chunk.content)
        return chunks

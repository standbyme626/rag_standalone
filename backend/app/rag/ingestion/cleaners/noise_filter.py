"""噪声过滤清洗器。"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..models import DocumentChunk


class NoiseFilterCleaner:
    """过滤过短 chunk、压缩多余空行、去除反复出现的页分割标记。"""

    NAME = "noise_filter"
    MIN_CONTENT_LENGTH = 20

    def clean(self, chunks: list[DocumentChunk]) -> list[DocumentChunk]:
        result: list[DocumentChunk] = []
        for chunk in chunks:
            c = chunk.content.strip()
            # 丢弃过短内容
            if len(c) < self.MIN_CONTENT_LENGTH:
                continue
            # 压缩反复出现的 PAGE_BREAK
            c = re.sub(r"---PAGE_BREAK---\s*---PAGE_BREAK---", "---PAGE_BREAK---", c)
            # 压缩连续空行
            c = re.sub(r"\n{3,}", "\n\n", c).strip()
            # 丢弃清理后为空的
            if not c:
                continue
            chunk.content = c
            result.append(chunk)
        return result

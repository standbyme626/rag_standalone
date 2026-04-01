"""去重清洗器 — 精确去重 + 语义去重（大批次）。"""

from __future__ import annotations

import hashlib
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..models import DocumentChunk


class DedupCleaner:
    """精确去重（MD5） + 语义去重（embedding 余弦相似度 > 0.95）。"""

    NAME = "dedup"
    SIMILARITY_THRESHOLD: float = 0.95
    SEMANTIC_MIN_CHUNKS = 20  # 最少 chunk 数才启用语义去重

    def clean(self, chunks: list[DocumentChunk]) -> list[DocumentChunk]:
        # Phase 1: 精确去重
        seen_hashes: set[str] = set()
        unique: list[DocumentChunk] = []
        for chunk in chunks:
            h = hashlib.md5(chunk.content.encode("utf-8")).hexdigest()
            if h not in seen_hashes:
                seen_hashes.add(h)
                unique.append(chunk)

        # Phase 2: 语义去重（仅在批次足够大时启用，避免小批次过度开销）
        if len(unique) > self.SEMANTIC_MIN_CHUNKS:
            unique = self._semantic_dedup(unique)

        return unique

    def _semantic_dedup(
        self, chunks: list[DocumentChunk], window: int = 10
    ) -> list[DocumentChunk]:
        """局部窗口内语义去重。仅比较相邻 chunk，避免 O(n^2) 全配对。"""
        texts = [c.content for c in chunks]
        from app.services.embedding import EmbeddingService

        embedding_svc = EmbeddingService()
        embeddings = embedding_svc.batch_get_embeddings(texts, batch_size=32)

        if len(embeddings) != len(chunks):
            return chunks  # embedding 不完整，返回原数据

        keep_indices: set[int] = set(range(len(chunks)))
        for i in range(len(chunks) - 1):
            if i not in keep_indices:
                continue
            for j in range(i + 1, min(i + 1 + window, len(chunks))):
                if j not in keep_indices:
                    continue
                sim = self._cosine_sim(embeddings[i], embeddings[j])
                if sim > self.SIMILARITY_THRESHOLD:
                    # 保留内容更长的那个
                    if len(chunks[i].content) >= len(chunks[j].content):
                        keep_indices.discard(j)
                    else:
                        keep_indices.discard(i)
                        break  # i 被丢弃，跳出内层窗口

        return [chunks[i] for i in sorted(keep_indices)]

    @staticmethod
    def _cosine_sim(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

"""语义分块 — 基于 embedding 相似度边界检测。"""

from __future__ import annotations

import math
import re

from ..models import DocumentChunk, ParsedDocument, PipelineConfig
from .base import BaseChunker, register_chunker
from .recursive import RecursiveChunker


@register_chunker
class SemanticChunker(BaseChunker):
    NAME = "semantic"
    SIMILARITY_THRESHOLD: float = 0.5

    def chunk(self, document: ParsedDocument, config: PipelineConfig) -> list[DocumentChunk]:
        text = document.text

        # 中文+英文句子拆分（启发式）
        sentences = re.split(r"[。！？；\.\!\?\n]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return []

        # 句子太少时回退到递归分块
        if len(sentences) <= 3:
            return RecursiveChunker().chunk(document, config)

        # 批量嵌入句子
        from app.services.embedding import EmbeddingService

        embedding_svc = EmbeddingService()
        batch_size = min(config.embedding_batch_size, 32)
        embeddings = embedding_svc.batch_get_embeddings(sentences, batch_size=batch_size)

        if len(embeddings) != len(sentences):
            # embedding 不完整，回退
            return RecursiveChunker().chunk(document, config)

        # 检测边界：相邻句子 embedding 余弦相似度突降
        boundaries = [0]
        for i in range(1, len(embeddings)):
            sim = self._cosine_sim(embeddings[i - 1], embeddings[i])
            if sim < self.SIMILARITY_THRESHOLD:
                boundaries.append(i)
        boundaries.append(len(sentences))

        # 合并句子为 chunks，遵守 chunk_size
        chunks: list[DocumentChunk] = []
        chunk_index = 0
        for b_idx in range(len(boundaries) - 1):
            group = sentences[boundaries[b_idx] : boundaries[b_idx + 1]]
            group_text = "".join(group)

            if len(group_text) <= config.chunk_size:
                chunks.append(self._make_chunk_from_text(
                    group_text, document, chunk_index,
                ))
                chunk_index += 1
            else:
                # 大组再切分
                section_doc = ParsedDocument(text=group_text, metadata=document.metadata)
                sub_chunks = RecursiveChunker().chunk(section_doc, config)
                chunks.extend(sub_chunks)
                for sc in sub_chunks:
                    sc.metadata["chunk_index"] = chunk_index
                    sc.metadata["section"] = document.metadata.get("domain", "")
                    chunk_index += 1

        return chunks

    @staticmethod
    def _make_chunk_from_text(
        content: str,
        document: ParsedDocument,
        chunk_index: int,
    ) -> DocumentChunk:
        return DocumentChunk(
            content=content.strip(),
            metadata={
                "chunk_index": chunk_index,
                "source_path": document.metadata.get("source_path", ""),
                "domain": document.metadata.get("domain", ""),
                "section": "",
                "file_type": document.metadata.get("file_type", ""),
                "has_table": False,
            },
        )

    @staticmethod
    def _cosine_sim(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

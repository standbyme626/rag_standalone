"""递归字符分块 — 多级分隔符 + 滑动窗口。"""

from __future__ import annotations

from ..models import DocumentChunk, ParsedDocument, PipelineConfig
from .base import BaseChunker, register_chunker


@register_chunker
class RecursiveChunker(BaseChunker):
    NAME = "recursive"
    SEPARATORS = ["\n\n", "\n"]

    def chunk(self, document: ParsedDocument, config: PipelineConfig) -> list[DocumentChunk]:
        text = document.text
        chunk_size = config.chunk_size
        overlap = config.chunk_overlap

        # 用主分隔符切分原文档
        segments = self._split_by_separators(text, self.SEPARATORS)

        chunks: list[DocumentChunk] = []
        current_parts: list[str] = []
        current_len = 0
        chunk_index = 0
        pos = 0

        for seg in segments:
            seg_len = len(seg)
            if current_len + seg_len <= chunk_size:
                current_parts.append(seg)
                current_len += seg_len
            else:
                if current_parts:
                    content = "".join(current_parts)
                    if content.strip():
                        chunks.append(self._make_chunk(
                            content, document, chunk_index, pos - current_len, pos,
                        ))
                        chunk_index += 1

                # 保留末尾 overlap 作为下一个 chunk 的起点
                overlap_text = ""
                if current_parts:
                    joined = "".join(current_parts)
                    if len(joined) > overlap:
                        overlap_text = joined[-overlap:]
                    else:
                        overlap_text = joined

                current_parts = [overlap_text, seg]
                current_len = len(overlap_text) + seg_len

            pos += seg_len

        # 尾部 chunk
        if current_parts:
            content = "".join(current_parts)
            if content.strip():
                chunks.append(self._make_chunk(
                    content, document, chunk_index, pos - current_len, pos,
                ))

        return chunks

    @staticmethod
    def _split_by_separators(text: str, separators: list[str]) -> list[str]:
        """按优先级分隔符递归切分文本，保留分隔符到每个片段末尾以便重叠。"""
        if not text:
            return []
        if len(separators) == 0:
            return list(text)
        sep = separators[0]
        result = []
        parts = text.split(sep)
        for i, part in enumerate(parts):
            if not part:
                continue
            # 附加分隔符，保留段落边界标记
            result.append(part + sep if i < len(parts) - 1 else part)
        return result

    @staticmethod
    def _make_chunk(
        content: str,
        document: ParsedDocument,
        chunk_index: int,
        char_start: int,
        char_end: int,
    ) -> DocumentChunk:
        section = ""
        for sec in reversed(document.metadata.get("sections", [])):
            if sec.get("char_start", 0) <= char_start:
                section = sec.get("heading", "")
                break

        return DocumentChunk(
            content=content.strip(),
            metadata={
                "chunk_index": chunk_index,
                "source_path": document.metadata.get("source_path", ""),
                "domain": document.metadata.get("domain", ""),
                "section": section,
                "file_type": document.metadata.get("file_type", ""),
                "char_start": char_start,
                "char_end": char_end,
                "has_table": False,
            },
        )

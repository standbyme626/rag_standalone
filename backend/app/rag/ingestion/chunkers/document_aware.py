"""文档结构感知分块 — 按标题边界切分。"""

from __future__ import annotations

from ..models import DocumentChunk, ParsedDocument, PipelineConfig
from .base import BaseChunker, register_chunker
from .recursive import RecursiveChunker


@register_chunker
class DocumentAwareChunker(BaseChunker):
    NAME = "document_aware"

    def chunk(self, document: ParsedDocument, config: PipelineConfig) -> list[DocumentChunk]:
        text = document.text
        sections = document.metadata.get("sections", [])

        if not sections:
            # 无结构信息，回退到递归分块
            return RecursiveChunker().chunk(document, config)

        # 按 section 边界切分
        chunks: list[DocumentChunk] = []
        chunk_index = 0
        max_section_size = config.chunk_size * 3

        for i, sec in enumerate(sections):
            start = sec.get("char_start", 0)
            end = sec.get("char_end", len(text) if i < len(sections) - 1 else len(text))
            # 使用下一个 section 的 char_start 作为当前 end
            if i + 1 < len(sections):
                end = sections[i + 1]["char_start"]
            else:
                end = len(text)

            sec_text = text[start:end]
            heading = sec.get("heading", "")

            if len(sec_text) > max_section_size:
                # 超长 section：递归分块
                temp_doc = ParsedDocument(text=sec_text, metadata=document.metadata)
                sub_chunks = RecursiveChunker().chunk(temp_doc, config)
                for sc in sub_chunks:
                    sc.metadata["section"] = heading
                    sc.metadata["chunk_index"] = chunk_index
                    chunks.append(sc)
                    chunk_index += 1
            elif sec_text.strip():
                chunks.append(DocumentChunk(
                    content=sec_text.strip(),
                    metadata={
                        "chunk_index": chunk_index,
                        "source_path": document.metadata.get("source_path", ""),
                        "domain": document.metadata.get("domain", ""),
                        "section": heading,
                        "file_type": document.metadata.get("file_type", ""),
                        "char_start": start,
                        "char_end": end,
                        "has_table": False,
                    },
                ))
                chunk_index += 1

        return chunks

"""表格感知分块 —— 表格单独提取为 chunk，其余文本正常分块。"""

from __future__ import annotations

from ..models import DocumentChunk, ParsedDocument, PipelineConfig
from .base import BaseChunker, register_chunker
from .document_aware import DocumentAwareChunker
from .recursive import RecursiveChunker


@register_chunker
class TableAwareChunker(BaseChunker):
    NAME = "table_aware"

    def chunk(self, document: ParsedDocument, config: PipelineConfig) -> list[DocumentChunk]:
        tables = document.metadata.get("tables", [])
        if not tables:
            # 无表格，回退到结构感知分块
            return DocumentAwareChunker().chunk(document, config)

        chunks: list[DocumentChunk] = []
        chunk_index = 0

        # 提取表格为独立 chunk
        for table in tables:
            table_text = f"[TABLE]\n{table['table_text']}\n[/TABLE]"
            chunks.append(DocumentChunk(
                content=table_text,
                metadata={
                    "chunk_index": chunk_index,
                    "source_path": document.metadata.get("source_path", ""),
                    "domain": document.metadata.get("domain", ""),
                    "section": "tables",
                    "file_type": document.metadata.get("file_type", ""),
                    "has_table": True,
                },
            ))
            chunk_index += 1

        # 去掉表格后的文本用递归分块
        table_texts = [t.get("table_text", "") for t in tables]
        text_without_tables = document.text
        for tt in table_texts:
            text_without_tables = text_without_tables.replace(tt, "", 1)

        clean_doc = ParsedDocument(
            text=text_without_tables,
            metadata={k: v for k, v in document.metadata.items() if k != "tables"},
        )

        text_chunks = RecursiveChunker().chunk(clean_doc, config)
        for tc in text_chunks:
            tc.metadata["chunk_index"] = chunk_index
            chunks.append(tc)
            chunk_index += 1

        return chunks

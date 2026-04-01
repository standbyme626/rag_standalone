"""数据管线统一数据模型。"""

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class ParsedDocument(BaseModel):
    """解析后的文档。parse() 的返回单元。"""

    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    # 约定字段:
    #   source_path: str    — 原始文件路径
    #   file_type: str      — pdf|docx|html|md|txt
    #   domain: str         — medical|legal
    #   title: str          — 文档标题或文件名
    #   sections: list[dict] — [{heading, level, char_start, char_end}]
    #   tables: list[dict]   — [{table_text, row_count, index}]
    #   num_pages: int      — 页数（非分页文档为 0）
    parsed_at: datetime = Field(default_factory=datetime.utcnow)


class DocumentChunk(BaseModel):
    """向量化后入库的最小单元。"""

    id: str = ""
    content: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    # 约定字段:
    #   chunk_index: int      — 文档内序号
    #   source_path: str      — 继承自 ParsedDocument
    #   domain: str           — 继承
    #   section: str          — 所属标题
    #   file_type: str        — 来源格式
    #   has_table: bool       — 是否包含表格
    #   char_start / char_end — 原文偏移
    embedding: list[float] | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

    def assign_id(self) -> str:
        """自动生成 UUID，若 ID 为空。"""
        if not self.id:
            self.id = str(uuid.uuid4())
        return self.id


class PipelineConfig(BaseModel):
    """管线运行配置。"""

    domain: str = "medical"
    chunk_size: int = 512
    chunk_overlap: int = 64
    chunk_strategy: str = "recursive"
    parser_overrides: dict[str, str] = {}
    cleaners: list[str] = ["dedup", "noise_filter", "pii_redactor"]
    embedding_batch_size: int = 32
    collection_name: str = "huatuo_knowledge"
    milvus_filter_expr: str = ""
    input_paths: list[str] = []
    dry_run: bool = False

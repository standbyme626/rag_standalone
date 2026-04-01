"""数据管线 — 文档 ingest pipeline。

入口:
    from app.rag.ingestion import IngestionPipeline, PipelineConfig
    pipeline = IngestionPipeline(PipelineConfig(domain="medical", ...))
    chunks = pipeline.ingest_paths([Path("data/docs")])
"""

from .models import DocumentChunk, PipelineConfig
from .pipeline import IngestionPipeline

__all__ = ["IngestionPipeline", "PipelineConfig", "DocumentChunk"]

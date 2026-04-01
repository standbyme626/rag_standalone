"""数据管线 CLI 入口 — typer。"""

from __future__ import annotations

from pathlib import Path

import typer

from app.core.config import settings

from .models import PipelineConfig
from .pipeline import IngestionPipeline

app = typer.Typer(help="RAG 数据管线 CLI - 文档 ingest pipeline")


@app.command()
def ingest(
    domain: str = typer.Option(
        "medical", "--domain", "-d", help="领域: medical | legal"
    ),
    input_path: str = typer.Option(
        ..., "--input", "-i", help="输入文件或目录路径"
    ),
    collection: str = typer.Option(
        None, "--collection", "-c", help="Milvus collection name（默认从配置读取）"
    ),
    chunk_size: int = typer.Option(
        None, "--chunk-size", help="目标 chunk 大小（字符数）"
    ),
    chunk_overlap: int = typer.Option(
        None, "--chunk-overlap", help="chunk 重叠大小（字符数）"
    ),
    strategy: str = typer.Option(
        "recursive",
        "--strategy",
        "-s",
        help="分块策略: recursive|semantic|document_aware|table_aware",
    ),
    no_pii: bool = typer.Option(
        False, "--no-pii", help="关闭 PII 脱敏"
    ),
    no_dedup: bool = typer.Option(
        False, "--no-dedup", help="关闭去重"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="仅解析+分块，跳过 embedding 和 Milvus 入库"
    ),
):
    """将文档导入到 RAG 知识库。"""
    config = PipelineConfig(
        domain=domain,
        input_paths=[input_path],
        collection_name=collection or settings.INGESTION_COLLECTION_NAME,
        chunk_size=chunk_size or settings.INGESTION_CHUNK_SIZE,
        chunk_overlap=chunk_overlap or settings.INGESTION_CHUNK_OVERLAP,
        chunk_strategy=strategy,
        dry_run=dry_run,
    )

    cleaners = []
    if not no_dedup:
        cleaners.append("dedup")
    cleaners.append("noise_filter")
    if not no_pii:
        cleaners.append("pii_redactor")
    config.cleaners = cleaners

    pipeline = IngestionPipeline(config)
    paths = [Path(input_path)]
    chunks = pipeline.ingest_paths(paths)

    typer.echo(f"\nIngested: {len(chunks)} chunks from {len(paths)} path(s).")


if __name__ == "__main__":
    app()

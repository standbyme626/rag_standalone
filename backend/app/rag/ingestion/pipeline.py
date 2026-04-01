"""管线编排器 — parse → chunk → clean → embed → store。"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from .chunkers import get_chunker
from .cleaners import get_cleaner
from .models import DocumentChunk, ParsedDocument, PipelineConfig
from .parsers import detect_parser

logger = structlog.get_logger(__name__)


class IngestionPipeline:
    """数据管线：将原始文档解析、分块、清洗、向量化后存入 Milvus。"""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._logger = logger.bind(domain=config.domain)

    # ── 单文件入口 ───────────────────────────────────

    def ingest_file(self, file_path: Path) -> list[DocumentChunk]:
        """对单个文件执行完整管线。"""
        # Step 1: 解析
        parser = self._resolve_parser(file_path)
        if parser is None:
            self._logger.warning("no_parser_for_file", path=str(file_path))
            return []

        docs = parser.parse(file_path)
        self._logger.info("parsed", path=str(file_path), num_docs=len(docs))

        # Step 2: 分块
        all_chunks: list[DocumentChunk] = []
        chunker = get_chunker(self.config.chunk_strategy)
        for doc in docs:
            doc.metadata["domain"] = self.config.domain
            doc.metadata.setdefault("source_path", str(file_path))
            doc.metadata.setdefault("file_type", file_path.suffix.lstrip(".").lower())
            chunks = chunker.chunk(doc, self.config)
            all_chunks.extend(chunks)
        self._logger.info("chunked", num_chunks=len(all_chunks))

        # Step 3: 清洗
        for cleaner_name in self.config.cleaners:
            cleaner = get_cleaner(cleaner_name)
            all_chunks = cleaner.clean(all_chunks)
        self._logger.info("cleaned", num_chunks=len(all_chunks))

        # Step 4: 向量化 + 入库（非 dry_run）
        if self.config.dry_run:
            self._logger.info("dry_run_complete", num_chunks=len(all_chunks))
            return all_chunks

        all_chunks = self._embed_chunks(all_chunks)
        self._logger.info("embedded", num_chunks=len(all_chunks))

        self._store_to_milvus(all_chunks)
        return all_chunks

    # ── 多文件入口 ───────────────────────────────────

    def ingest_paths(self, paths: list[Path]) -> list[DocumentChunk]:
        """对多个文件/目录执行管线。"""
        files = self._expand_paths(paths)
        if not files:
            self._logger.warning("no_files_found", paths=[str(p) for p in paths])
            return []

        self._logger.info("ingest_start", num_files=len(files))
        t0 = time.time()
        all_chunks: list[DocumentChunk] = []
        for fp in files:
            chunks = self.ingest_file(fp)
            all_chunks.extend(chunks)
        self._logger.info(
            "ingest_complete",
            total_chunks=len(all_chunks),
            elapsed_s=round(time.time() - t0, 2),
        )
        return all_chunks

    # ── 内部方法 ─────────────────────────────────────

    def _resolve_parser(self, file_path: Path):
        # 允许通过 config 覆盖文件扩展名映射
        suffix = file_path.suffix.lower()
        if suffix in self.config.parser_overrides:
            override_name = self.config.parser_overrides[suffix]
            from .parsers.base import _PARSER_REGISTRY
            cls = _PARSER_REGISTRY.get(override_name)
            if cls:
                return cls()

        return detect_parser(file_path)

    def _expand_paths(self, paths: list[Path]) -> list[Path]:
        """将路径展开为文件列表，支持递归扫描目录。"""
        files: list[Path] = []
        for p in paths:
            p = p.resolve()
            if p.is_file():
                files.append(p)
            elif p.is_dir():
                for ext in [".pdf", ".docx", ".html", ".htm", ".md", ".markdown", ".txt"]:
                    files.extend(p.rglob(f"*{ext}"))
        return sorted(set(files))

    def _embed_chunks(self, chunks: list[DocumentChunk]) -> list[DocumentChunk]:
        """分批调用 EmbeddingService 生成向量。"""
        from app.services.embedding import EmbeddingService

        svc = EmbeddingService()
        texts = [c.content for c in chunks]
        batch_size = self.config.embedding_batch_size

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            embeddings = svc.batch_get_embeddings(batch, batch_size=batch_size)
            for j, emb in enumerate(embeddings):
                chunks[i + j].embedding = emb

        return chunks

    def _store_to_milvus(self, chunks: list[DocumentChunk]) -> None:
        """批量插入 Milvus，兼容现有 schema。

        现有 schema: [content, vector/embedding, department, disease, source]
        """
        from pymilvus import connections

        from app.core.config import settings

        if not connections.has_connection("default"):
            milvus_host = getattr(settings, "MILVUS_HOST", "localhost")
            milvus_port = getattr(settings, "MILVUS_PORT", 19530)
            connections.connect(
                alias="default",
                host=milvus_host,
                port=milvus_port,
                timeout=5.0,
            )

        from pymilvus import Collection, utility

        if not utility.has_collection(self.config.collection_name):
            self._logger.warning(
                "collection_not_found",
                collection=self.config.collection_name,
            )
            return

        collection = Collection(self.config.collection_name)
        available_fields = {f.name for f in collection.schema.fields}

        # 确定向量字段名
        vector_field = (
            "embedding" if "embedding" in available_fields else "vector"
        )

        # 构建实体
        max_batch = 1000
        for i in range(0, len(chunks), max_batch):
            batch = chunks[i : i + max_batch]

            ids = [c.assign_id() for c in batch]
            contents = [c.content for c in batch]
            vectors = [c.embedding for c in batch]
            departments = [c.metadata.get("section", "") for c in batch]
            diseases = [c.metadata.get("disease", "") for c in batch]
            sources = [c.metadata.get("domain", self.config.domain) for c in batch]

            # 根据字段名构造插入顺序
            data_parts = []
            for field in ["content", vector_field, "department", "disease", "source"]:
                if field == "content":
                    data_parts.append(contents)
                elif field == vector_field:
                    data_parts.append(vectors)
                elif field == "department":
                    data_parts.append([d[:1000] for d in departments])
                elif field == "disease":
                    data_parts.append([d[:500] for d in diseases])
                elif field == "source":
                    data_parts.append(sources)

            collection.insert(data_parts)

        collection.flush()
        self._logger.info(
            "stored_to_milvus",
            num_chunks=len(chunks),
            collection=self.config.collection_name,
        )

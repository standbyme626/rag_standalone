"""Chonkie 分块器 — 来源：UltraRAG servers/corpus/src/corpus.py chunk_documents

支持 token / sentence / recursive 分块模式。
依赖：pip install chonkie
"""

from __future__ import annotations

from typing import Optional

from app.rag.ingestion.models import DocumentChunk, ParsedDocument, PipelineConfig
from .base import BaseChunker, register_chunker


# Lazy register only when chonkie is available — avoids import error on CI
try:
    import chonkie  # noqa: F401
    _CHONKIE_IMPORT_ERROR: Optional[str] = None
except ImportError:
    _CHONKIE_IMPORT_ERROR = (
        "chonkie is not installed. Please install with: pip install chonkie"
    )


@register_chunker
class ChonkieChunker(BaseChunker):
    """基于 Chonkie 库的分块器

    Modes:
      - token: GPT-2 等 tokenizer 按 token 切分
      - sentence: 基于句子边界的分块
      - recursive: 递归分块（尝试 sentence → word → character）
    """

    NAME: str = "chonkie"

    def __init__(self) -> None:
        self._import_error = _CHONKIE_IMPORT_ERROR

    def _get_backend(self, mode: str, chunk_size: int, chunk_overlap: int):
        if self._import_error:
            raise RuntimeError(self._import_error)

        tokenizer = self._get_tokenizer()

        if mode == "token":
            return self._chonkie.TokenChunker(
                tokenizer=tokenizer,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        elif mode == "sentence":
            return self._chonkie.SentenceChunker(
                tokenizer=tokenizer,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        else:
            return self._chonkie.RecursiveChunker(
                tokenizer=tokenizer,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

    def _get_tokenizer(self):
        """默认使用 gpt2 tokenizer（Chonkie 内置支持）"""
        return "gpt2"

    def chunk(
        self, document: ParsedDocument, config: PipelineConfig
    ) -> list[DocumentChunk]:
        mode = getattr(config, "chonkie_mode", "token")
        chunk_size = getattr(config, "chunk_size", 512)
        chunk_overlap = getattr(config, "chunk_overlap", 50)

        if self._import_error:
            raise RuntimeError(self._import_error)

        backend = self._get_backend(mode, chunk_size, chunk_overlap)

        chunks = backend.chunk_text(document.text)

        result: list[DocumentChunk] = []
        for i, c in enumerate(chunks):
            result.append(
                DocumentChunk(
                    content=c.text if hasattr(c, "text") else str(c),
                    metadata={
                        **document.metadata,
                        "chunk_index": i,
                        "chunker_mode": mode,
                    },
                )
            )
        return result

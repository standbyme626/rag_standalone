"""分块器抽象基类与注册中心。Chunker ABC and registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from ..models import DocumentChunk, ParsedDocument, PipelineConfig

_CHUNKER_REGISTRY: dict[str, type[BaseChunker]] = {}


class BaseChunker(ABC):
    NAME: ClassVar[str] = "base"

    @abstractmethod
    def chunk(self, document: ParsedDocument, config: PipelineConfig) -> list[DocumentChunk]:
        """将解析后的文档切分为块。"""
        ...


def register_chunker(cls: type[BaseChunker]) -> type[BaseChunker]:
    """分块器注册装饰器。"""
    _CHUNKER_REGISTRY[cls.NAME] = cls
    return cls


def get_chunker(name: str) -> BaseChunker:
    cls = _CHUNKER_REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown chunker: {name}. Available: {list(_CHUNKER_REGISTRY.keys())}")
    return cls()

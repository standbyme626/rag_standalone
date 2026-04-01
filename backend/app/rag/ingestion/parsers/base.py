"""解析器抽象基类与注册中心。Parser ABC and registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar

from ..models import ParsedDocument

_PARSER_REGISTRY: dict[str, type[BaseParser]] = {}
_EXT_INDEX: dict[str, str] = {}  # ext -> parser name


class BaseParser(ABC):
    """文档解析器基类。解析一个文件为列表，因为一个文件可能包含多个逻辑文档。"""

    SUPPORTED_EXTENSIONS: ClassVar[set[str]] = set()
    NAME: ClassVar[str] = "base"

    @abstractmethod
    def parse(self, file_path: Path) -> list[ParsedDocument]:
        """解析文件为结构化文档列表。"""
        ...

    def can_handle(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS


def register_parser(cls: type[BaseParser]) -> type[BaseParser]:
    """解析器注册装饰器。"""
    _PARSER_REGISTRY[cls.NAME] = cls
    for ext in cls.SUPPORTED_EXTENSIONS:
        # 不覆盖已有解析器 — 第一个注册的优先
        _EXT_INDEX.setdefault(ext, cls.NAME)
    return cls


def detect_parser(file_path: Path) -> BaseParser | None:
    """根据文件扩展名自动检测解析器。"""
    ext = file_path.suffix.lower()
    name = _EXT_INDEX.get(ext)
    if name and name in _PARSER_REGISTRY:
        return _PARSER_REGISTRY[name]()
    return None

"""纯文本解析器 — 直接读取文件内容。"""

from __future__ import annotations

from pathlib import Path

from ..models import ParsedDocument
from .base import BaseParser, register_parser


@register_parser
class TextParser(BaseParser):
    SUPPORTED_EXTENSIONS = {".txt"}
    NAME = "txt"

    def parse(self, file_path: Path) -> list[ParsedDocument]:
        text = file_path.read_text(encoding="utf-8")
        return [ParsedDocument(
            text=text,
            metadata={
                "source_path": str(file_path),
                "file_type": "txt",
                "title": file_path.stem,
            },
        )]

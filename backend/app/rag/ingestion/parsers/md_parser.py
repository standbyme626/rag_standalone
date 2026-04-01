"""Markdown 解析器 — 正则表达式提取标题。"""

from __future__ import annotations

import re
from pathlib import Path

from ..models import ParsedDocument
from .base import BaseParser, register_parser


@register_parser
class MarkdownParser(BaseParser):
    SUPPORTED_EXTENSIONS = {".md", ".markdown"}
    NAME = "md"

    def parse(self, file_path: Path) -> list[ParsedDocument]:
        text = file_path.read_text(encoding="utf-8")

        sections: list[dict] = []
        for match in re.finditer(r"^(#{1,6})\s+(.+)$", text, re.MULTILINE):
            sections.append({
                "level": len(match.group(1)),
                "heading": match.group(2).strip(),
                "char_start": match.start(),
            })

        return [ParsedDocument(
            text=text,
            metadata={
                "source_path": str(file_path),
                "file_type": "md",
                "sections": sections,
                "title": file_path.stem,
            },
        )]

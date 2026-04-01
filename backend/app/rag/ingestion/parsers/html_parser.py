"""HTML 解析器 — 基于 BeautifulSoup。"""

from __future__ import annotations

import re
from pathlib import Path

from ..models import ParsedDocument
from .base import BaseParser, register_parser


@register_parser
class HTMLParser(BaseParser):
    SUPPORTED_EXTENSIONS = {".html", ".htm"}
    NAME = "html"

    def parse(self, file_path: Path) -> list[ParsedDocument]:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(file_path.read_text(encoding="utf-8"), "html.parser")

        # 移除噪声标签
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        # 提取标题层次
        headings: list[dict] = []
        for h in soup.find_all(re.compile(r"^h[1-6]$")):
            heading_text = h.get_text(strip=True)
            if heading_text:
                level = int(h.name[1])
                headings.append({"heading": heading_text, "level": level})

        text = soup.get_text(separator="\n", strip=True)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()

        title_el = soup.title.string if soup.title and soup.title.string else None

        return [ParsedDocument(
            text=text,
            metadata={
                "source_path": str(file_path),
                "file_type": "html",
                "sections": headings,
                "title": title_el or file_path.stem,
            },
        )]

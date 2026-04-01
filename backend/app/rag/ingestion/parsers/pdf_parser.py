"""PDF 解析器 — 基于 PyMuPDF（fitz）。"""

from __future__ import annotations

import re
from pathlib import Path

from ..models import ParsedDocument
from .base import BaseParser, register_parser


@register_parser
class PDFParser(BaseParser):
    SUPPORTED_EXTENSIONS = {".pdf"}
    NAME = "pdf"

    def parse(self, file_path: Path) -> list[ParsedDocument]:
        import fitz  # PyMuPDF

        doc = fitz.open(str(file_path))
        pages_text: list[str] = []
        sections: list[dict] = []

        for page_idx in range(len(doc)):
            page = doc[page_idx]
            text = page.get_text("text")
            pages_text.append(text)

            # 基于字体大小的启发式标题检测 (>14pt = heading)
            try:
                blocks = page.get_text("dict")["blocks"]
                for block in blocks:
                    if block.get("type") != 0:
                        continue
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            if span.get("size", 0) > 14:
                                heading = span["text"].strip()
                                if heading:
                                    sections.append({
                                        "heading": heading,
                                        "page": page_idx,
                                    })
            except Exception:
                pass  # 某些 PDF 可能没有 dict 格式文本，静默跳过

        full_text = "\n\n---PAGE_BREAK---\n\n".join(pages_text)
        # 压缩连续空行
        full_text = re.sub(r"\n{3,}", "\n\n", full_text).strip()

        return [ParsedDocument(
            text=full_text,
            metadata={
                "source_path": str(file_path),
                "file_type": "pdf",
                "num_pages": len(doc),
                "sections": sections,
                "title": file_path.stem,
            },
        )]

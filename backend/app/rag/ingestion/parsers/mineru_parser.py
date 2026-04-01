"""MinerU PDF 解析器 — 来源：UltraRAG servers/corpus/src/corpus.py mineru_parse()

CLI 调用 MinerU，自动提取文本+图片/表格 corpus。
要求：mineru CLI 已安装且在 PATH 中
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import structlog

from ..models import ParsedDocument
from .base import BaseParser, register_parser

logger = structlog.get_logger(__name__)

# 检查 mineru 是否可用
_MINERU_AVAILABLE = shutil.which("mineru") is not None


@register_parser
class MinerUParser(BaseParser):
    """MinerU 文档解析器

    MinerU 是一款开源 PDF 解析工具，能提取文本、图片、表格、公式等。
    输出目录结构为 mid_json + images + tables。
    """

    SUPPORTED_EXTENSIONS = {".pdf"}
    NAME = "mineru"

    def __init__(self, extra_params: dict | None = None):
        self.extra_params = extra_params or {}

    def parse(self, file_path: Path) -> list[ParsedDocument]:
        """使用 MinerU CLI 解析 PDF 文件

        MinerU 输出为目录结构：
        - {stem}_middle.json: 中间结果（文本+位置）
        - images/: 提取的图片
        - tables/: 提取的表格
        """
        if not _MINERU_AVAILABLE:
            raise RuntimeError(
                "mineru CLI is not installed or not in PATH. "
                "Install it or use the plain PDF parser instead."
            )

        file_path = file_path.resolve()
        with tempfile.TemporaryDirectory(prefix="archrag_mineru_") as tmpdir:
            cmd = [
                "mineru",
                "-p",
                str(file_path),
                "-o",
                tmpdir,
            ]
            # 添加额外参数
            for key, value in self.extra_params.items():
                cmd.append(f"--{key}")
                if value is not None:
                    cmd.append(str(value))

            logger.info("mineru_start", file=str(file_path), cmd=" ".join(cmd))
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10 分钟超时
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"MinerU failed with code {result.returncode}: {result.stderr.strip()}"
                )
            logger.info("mineru_done", file=str(file_path))

            # 读取解析结果
            return self._extract_results(tmpdir, file_path)

    def _extract_results(
        self, tmpdir: str, file_path: Path
    ) -> list[ParsedDocument]:
        """提取 MinerU 输出为 ParsedDocument(s)"""
        import os

        text_parts: list[str] = []
        metadata: dict = {"source_path": str(file_path), "file_type": "pdf"}

        # 读取 JSON 中间结果
        for f in Path(tmpdir).rglob("*_middle.json"):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                # MinerU mid_json: pages 列表
                pages = data.get("pages", [])
                metadata["num_pages"] = len(pages)
                for page in pages:
                    page_num = page.get("page_no", 0)
                    blocks = page.get("blocks", [])
                    for block in blocks:
                        lines = block.get("lines", [])
                        for line in lines:
                            text = line.get("text", "")
                            if text.strip():
                                text_parts.append(text.strip())
            except Exception as e:
                logger.warning("mineru_json_parse_failed", error=str(e))

        # 读取图片目录信息
        img_dir = Path(tmpdir) / "images"
        if img_dir.exists():
            metadata["images"] = [
                str(p.name)
                for p in img_dir.iterdir()
                if p.is_file()
            ]

        # 读取表格目录信息
        tbl_dir = Path(tmpdir) / "tables"
        if tbl_dir.exists():
            metadata["tables"] = [
                str(p.name)
                for p in tbl_dir.iterdir()
                if p.is_file()
            ]

        # 读取 tables 的 markdown 内容（如果有）
        for md_file in Path(tmpdir).rglob("*_content_list.md"):
            try:
                content = md_file.read_text(encoding="utf-8")
                if content.strip():
                    text_parts.append(f"\n=== 表格 ===\n{content.strip()}")
            except Exception as e:
                logger.warning("mineru_table_read_failed", error=str(e))

        if not text_parts:
            return []

        return [
            ParsedDocument(
                text="\n\n".join(text_parts),
                metadata=metadata,
            )
        ]

    def can_handle(self, file_path: Path) -> bool:
        return (
            super().can_handle(file_path)
            and file_path.suffix.lower() == ".pdf"
            and _MINERU_AVAILABLE
        )

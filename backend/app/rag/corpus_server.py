"""语料处理 Server

Phase 4.3: 语料处理 Server

功能：
1. 多格式文件内容提取 (PDF/DOCX/HTML/MD/TXT)
2. 语料处理管线集成 (MinerU/Chonkie/Reflow)
3. 路径穿越防护 _validate_path()

用法:
    corpus = CorpusServer()
    text = corpus.extract("/path/to/file.pdf")
    chunks = corpus.chunk(text, method="chonkie", chunk_size=512)
"""

from __future__ import annotations

import os
import re
from pathlib import Path, PurePath
from typing import Any, Dict, List, Optional

import structlog

logger = structlog.get_logger(__name__)


# --------------- 路径安全 ---------------

def validate_path(file_path: str | Path, *, base_dir: Optional[Path] = None) -> Path:
    """路径校验，防止目录穿越攻击

    - 解析 `..` 和符号链接
    - 确保文件在 `base_dir` 下（默认禁止向上穿越）
    - 禁止绝对路径指向系统敏感目录

    来源：UltraRAG `_validate_path()`

    Args:
        file_path: 待验证的路径
        base_dir: 基准目录（默认禁止向上穿越到父目录外）

    Returns:
        解析后的绝对路径

    Raises:
        ValueError: 路径非法或存在穿越风险
    """
    path = Path(file_path).resolve()

    # 禁止系统敏感目录
    SENSITIVE_PATTERNS = [
        r"^/etc/", r"^/proc/", r"^/sys/", r"^/dev/",
        r"^/root/", r"^/var/",
    ]
    for pattern in SENSITIVE_PATTERNS:
        if re.match(pattern, str(path)):
            raise ValueError(f"Path traversal blocked: '{path}' matches sensitive pattern")

    # 如果指定了 base_dir，确保路径在其下
    if base_dir is not None:
        base = Path(base_dir).resolve()
        if not str(path).startswith(str(base)):
            raise ValueError(
                f"Path traversal blocked: '{path}' is outside base directory '{base}'"
            )

    return path


# --------------- 文件类型检测 ---------------

def detect_format(file_path: str | Path) -> str:
    """检测文件格式

    Returns:
        文件扩展名（如 pdf/docx/html/md/txt），未知则返回 "unknown"
    """
    ext = Path(file_path).suffix.lower().lstrip(".")
    if ext == "md":
        return "markdown"
    if ext in ("pdf", "docx", "html", "htm", "txt", "json", "csv"):
        return ext
    return ext or "unknown"


# --------------- 文本提取器 ---------------

class TextExtractor:
    """多格式文件文本提取"""

    # 支持的后缀 -> 提取方法映射
    SUPPORTED = {"pdf", "docx", "html", "htm", "md", "txt", "json", "csv"}

    @staticmethod
    def extract_txt(file_path: str | Path) -> str:
        """提取纯文本"""
        path = validate_path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        return path.read_text(encoding="utf-8", errors="replace")

    @classmethod
    def extract_pdf(cls, file_path: str | Path) -> str:
        """提取 PDF 文本"""
        path = validate_path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        try:
            from pypdf import PdfReader
            reader = PdfReader(str(path))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        except ImportError:
            logger.warning("pypdf_not_installed")
            return ""

    @classmethod
    def extract_docx(cls, file_path: str | Path) -> str:
        """提取 DOCX 富文本"""
        path = validate_path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        try:
            from docx import Document
            doc = Document(str(path))
            return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        except ImportError:
            logger.warning("docx_not_installed")
            return ""

    @classmethod
    def extract_html(cls, file_path: str | Path) -> str:
        """提取 HTML 纯文本内容（去除标签）"""
        path = validate_path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        text = path.read_text(encoding="utf-8", errors="replace")
        return re.sub(r"<[^>]+>", " ", text)

    @classmethod
    def extract_markdown(cls, file_path: str | Path) -> str:
        """提取 Markdown 内容"""
        path = validate_path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        return path.read_text(encoding="utf-8", errors="replace")

    @classmethod
    def extract(cls, file_path: str | Path) -> Dict[str, Any]:
        """根据格式自动选择提取方式

        Returns:
            {"format": str, "text": str, "path": str}
        """
        path = Path(file_path)
        fmt = detect_format(path)

        extractors = {
            "pdf": cls.extract_pdf,
            "docx": cls.extract_docx,
            "html": cls.extract_html,
            "htm": cls.extract_html,
            "md": cls.extract_markdown,
            "markdown": cls.extract_markdown,
            "txt": cls.extract_txt,
            "json": cls.extract_txt,
            "csv": cls.extract_txt,
        }

        extractor = extractors.get(fmt, cls.extract_txt)
        try:
            text = extractor(path)
        except Exception as e:
            logger.error("extract_failed", path=str(path), fmt=fmt, error=str(e))
            text = ""

        return {
            "format": fmt,
            "text": text,
            "path": str(path),
            "length": len(text),
            "word_count": len(text.split()),
        }


# --------------- 语料分块 ---------------

def chunk_corpus(
    text: str,
    *,
    method: str = "default",
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    **kwargs,
) -> List[Dict[str, Any]]:
    """语料分块

    集成现有分块器（chonkie/recursive）。

    Args:
        text: 原始文本
        method: 分块方法 ("default" / "chonkie" / "recursive")
        chunk_size: 每块最大字符数
        chunk_overlap: 块间重叠字符数
        kwargs: 方法额外参数

    Returns:
        [{"chunk": str, "start": int, "end": int}, ...]
    """
    if method == "recursive":
        return _recursive_chunk(text, chunk_size, chunk_overlap)

    if method == "chonkie":
        try:
            from app.rag.ingestion.chunkers.chonkie import ChonkieChunker
            chunker = ChonkieChunker(chunk_size=chunk_size)
            pieces = chunker.chunk(text)
            return [
                {"chunk": p.get("text", p), "start": 0, "end": 0}
                for p in pieces
            ]
        except Exception:
            pass

    return _default_chunk(text, chunk_size, chunk_overlap)


def _default_chunk(text: str, chunk_size: int = 512, overlap: int = 50) -> List[Dict[str, Any]]:
    """默认分段：按 chunk_size 切分，保留 overlap"""
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        # Try to break on sentence/paragraph boundary
        if end < len(text):
            best_end = end
            for boundary in ("\n\n", "\n", ". ", "? ", "! "):
                idx = text.rfind(boundary, start, end + 20)
                if idx > start + chunk_size // 2:
                    best_end = idx + len(boundary)
                    break
            end = best_end
        chunks.append({
            "chunk": text[start:end],
            "start": start,
            "end": end,
        })
        if end >= len(text):
            break
        start = end - overlap
        if start <= chunks[-1]["start"]:
            start = end
    return chunks


def _recursive_chunk(text: str, chunk_size: int = 512, overlap: int = 50) -> List[Dict[str, Any]]:
    """递归分块：段落 → 句子 → 词"""
    if not text:
        return []

    # 1. 按段落切
    paragraphs = text.split("\n\n")
    if all(len(p) <= chunk_size for p in paragraphs):
        chunks = []
        start = 0
        for p in paragraphs:
            if p.strip():
                chunks.append({"chunk": p.strip(), "start": start, "end": start + len(p)})
            start += len(p) + 2
        return chunks

    # 2. 段落过长，按句子切
    sentences = re.split(r"(?<=[.!?。！？\n])\s*", text)
    return _group_sentences(sentences, chunk_size, overlap)


def _group_sentences(sentences: List[str], max_size: int, overlap: int) -> List[Dict[str, Any]]:
    """将句子组块成最大 max_size 字符的 chunks"""
    chunks = []
    current: List[str] = []
    current_len = 0
    chunk_start = 0

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        sent_len = len(sent)

        if current_len + sent_len > max_size and current:
            chunks.append({
                "chunk": " ".join(current),
                "start": chunk_start,
                "end": chunk_start + current_len,
            })
            # Overlap: keep last few sentences
            overlap_sents = current[-3:] if len(current) > 3 else current[:]
            overlap_text = " ".join(overlap_sents)
            current = overlap_sents + [sent]
            chunk_start = chunks[-1]["end"] - len(overlap_text)
            current_len = len(overlap_text) + sent_len
        else:
            current.append(sent)
            current_len += sent_len + 1

    if current:
        chunks.append({
            "chunk": " ".join(current),
            "start": chunk_start,
            "end": chunk_start + current_len,
        })

    return chunks


# --------------- CorpusServer ---------------

class CorpusServer:
    """语料处理 Server

    集成文件提取 + 分块 + 清理管线。

    用法:
        server = CorpusServer()
        result = server.extract("/path/to/file.pdf")
        chunks = server.chunk(result["text"], method="recursive")
    """

    def __init__(self, *, base_dir: Optional[Path] = None):
        """
        Args:
            base_dir: 基准目录，限制文件访问范围
        """
        self.base_dir = Path(base_dir).resolve() if base_dir else None

    def extract(self, file_path: str | Path) -> Dict[str, Any]:
        """提取文件内容

        Args:
            file_path: 文件路径

        Returns:
            {"format": str, "text": str, "path": str, "length": int, "word_count": int}
        """
        safe_path = validate_path(file_path, base_dir=self.base_dir)
        return TextExtractor.extract(safe_path)

    def chunk(
        self,
        text: str,
        *,
        method: str = "recursive",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ) -> List[Dict[str, Any]]:
        """分块处理

        Args:
            text: 文本
            method: "default" | "recursive" | "chonkie"
            chunk_size: 每块字符数
            chunk_overlap: 重叠字符数

        Returns:
            [{"chunk": str, "start": int, "end": int}, ...]
        """
        return chunk_corpus(
            text,
            method=method,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def clean(
        self,
        text: str,
        *,
        steps: Optional[List[str]] = None,
    ) -> str:
        """清理文本

        Args:
            text: 原始文本
            steps: 清理步骤 ["strip", "deduplicate_newlines", "remove_extra_spaces", "reflow"]

        Returns:
            清理后文本
        """
        if steps is None:
            steps = ["strip", "deduplicate_newlines", "remove_extra_spaces"]

        result = text
        for step in steps:
            if step == "strip":
                result = result.strip()
            elif step == "deduplicate_newlines":
                result = re.sub(r"\n{3,}", "\n\n", result)
            elif step == "remove_extra_spaces":
                result = re.sub(r" {2,}", " ", result)
            elif step == "reflow":
                try:
                    from app.rag.ingestion.cleaners.reflow import ReflowCleaner
                    result = ReflowCleaner().process(result)
                except Exception:
                    result = re.sub(r"(?<!\n)\n(?!\n)", " ", result)

        return result

    def process(self, file_path: str | Path, **chunk_kwargs) -> Dict[str, Any]:
        """完整处理管线：提取 → 清理 → 分块

        Args:
            file_path: 文件路径
            chunk_kwargs: 传给 chunk() 的关键字参数

        Returns:
            {"format": str, "path": str, "text_length": int, "chunks": [...], "num_chunks": int}
        """
        extracted = self.extract(file_path)
        text = self.clean(extracted["text"])
        chunks = self.chunk(text, **chunk_kwargs)
        return {
            "format": extracted["format"],
            "path": extracted["path"],
            "text_length": extracted["length"],
            "chunks": chunks,
            "num_chunks": len(chunks),
        }

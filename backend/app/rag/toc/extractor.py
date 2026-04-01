"""TOC 语义提取管线

来源：ragflow rag/flow/extractor/extractor.py + rag/prompts/toc_*.md

从文档 chunk 中提取目录（TOC），用于后续检索路由。
"""

from __future__ import annotations

import json
import re
from typing import Any, Awaitable, Callable, Dict, List, Optional

import structlog

from app.rag.toc.prompts import TOC_DETECTION_SYSTEM, TOC_DETECTION_USER

logger = structlog.get_logger(__name__)

# --------------- 数据结构 ---------------

class TocEntry:
    """目录条目"""

    __slots__ = ("title", "chunk_id", "ids", "level", "score")

    def __init__(
        self,
        title: str,
        chunk_id: str,
        ids: Optional[List[str]] = None,
        level: int = 0,
    ):
        self.title = title
        self.chunk_id = chunk_id
        self.ids = ids or []
        self.level = level
        self.score: float = 0.0  # set during relevance scoring

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "chunk_id": self.chunk_id,
            "ids": self.ids,
            "level": self.level,
            "score": self.score,
        }


# --------------- 提取 ---------------

class TocExtractor:
    """从文档 chunk 中提取结构化目录。

    用法:
        extractor = TocExtractor(llm_call=async_chat_fn)
        toc_entries = await extractor.extract([
            {"id": "c1", "text": "第一章 总则 ..."},
            {"id": "c2", "text": "第二章 定义 ..."},
        ])
    """

    def __init__(self, llm_call: Optional[Callable[..., Awaitable[str]]] = None):
        self.llm_call = llm_call

    async def extract(
        self,
        chunks: List[Dict[str, Any]],
        **llm_kwargs,
    ) -> List[TocEntry]:
        """从 chunk 列表生成 TOC.

        Args:
            chunks: [{"id": str, "text": str}, ...]

        Returns:
            List of TocEntry 条目
        """
        if not self.llm_call:
            logger.warning("toc_llm_not_configured")
            return []

        # 1. 调用 LLM 提取原始 heading
        raw = await self._detect_toc(chunks)
        if not raw:
            return []

        # 2. 过滤 "-1" 条目
        raw = [e for e in raw if e.get("title", "-1") != "-1"]
        if not raw:
            return []

        # 3. 级别推测（基于编号模式）+ chunk 映射
        entries = self._build_entries(raw, chunks)
        return entries

    async def _detect_toc(
        self, chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        """调用 LLM 生成 TOC JSON"""
        # 构造 chunks dict: {chunk_id: text}
        chunks_dict = {c["id"]: c.get("text", "")[:2000] for c in chunks}

        prompt = TOC_DETECTION_SYSTEM + "\n\n" + TOC_DETECTION_USER.format(
            chunks_json=json.dumps(chunks_dict, ensure_ascii=False, indent=2)
        )

        try:
            response = await self.llm_call(
                system_prompt="",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4096,
            )
            # 提取 JSON
            response = response or ""
            json_str = self._extract_json(response)
            if not json_str:
                logger.error("toc_llm_no_json")
                return []
            items = json.loads(json_str)
            if not isinstance(items, list):
                logger.error("toc_llm_not_array")
                return []
            return items
        except Exception as e:
            logger.error("toc_detection_llm_failed", error=str(e))
            return []

    @staticmethod
    def _extract_json(text: str) -> Optional[str]:
        """从 LLM 输出中提取第一个 JSON 数组"""
        # 找第一个 '[' 和匹配的 ']'
        start = text.find("[")
        if start < 0:
            return None
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "[":
                depth += 1
            elif text[i] == "]":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        return None

    @staticmethod
    def _detect_level(title: str) -> int:
        """根据编号模式推测层级"""
        if re.match(r"^(第[一二三四五六七八九十百]+[篇章节])", title):
            return 1  # 章/节 级
        if re.match(r"^\d+(\.\d+)*[\.\s]", title):
            parts = title.split(".", 2)
            return len(parts) if len(parts) <= 3 else 3  # 1.x.y
        if re.match(r"^[IVXLCDM]+[).]", title):
            return 1  # 罗马数字，顶层
        if re.match(r"^[（(][一二三四五六七八九十]+[)）]", title):
            return 2  # 二级节
        return 1  # 默认第一层

    def _build_entries(
        self,
        raw: List[Dict[str, str]],
        chunks: List[Dict[str, Any]],
    ) -> List[TocEntry]:
        """将 LLM 原始输出映射为 TocEntry 列表"""
        chunk_id_to_idx = {c["id"]: i for i, c in enumerate(chunks)}
        chunk_id_to_chunk = {c["id"]: c for c in chunks}

        entries: List[TocEntry] = []
        for item in raw:
            cid = item.get("chunk_id", "")
            title = item.get("title", "")
            if not cid or not title or title == "-1":
                continue

            level = self._detect_level(title)
            entry = TocEntry(title=title, chunk_id=cid, level=level)

            # 映射 chunk ID → 实际 chunk
            chunk = chunk_id_to_chunk.get(cid)
            if chunk:
                entry.ids = [chunk.get("id", cid)]

            entries.append(entry)

        return entries


# --------------- 检索时 TOC 路由 ---------------

def relevant_chunks_with_toc(
    toc_entries: List[TocEntry],
    query: str,
    chunks: List[Dict[str, Any]],
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """使用 TOC 语义层级路由检索结果。

    简单版本：如果 TOC 与查询关键词匹配，则返回对应 TOC 条目映射的 chunk。
    否则返回原始 chunk（按顺序截断至 top_k）。

    Args:
        toc_entries: 已提取的目录条目
        query: 用户查询
        chunks: 原始文档 chunk
        top_k: 返回数量

    Returns:
        按 TOC 相关性排序的 chunk 列表
    """
    if not toc_entries:
        # 没有 TOC 信息，使用原始 chunk
        return chunks[:top_k]

    # 简单的关键词匹配 TOC
    query_terms = set(_tokenize(query))
    scored: List[tuple] = []

    for entry in toc_entries:
        entry_terms = set(_tokenize(entry.title))
        overlap = len(query_terms & entry_terms)
        if overlap > 0:
            entry.score = overlap / max(len(query_terms), 1)
            for cid in entry.ids:
                chunk = next((c for c in chunks if c.get("id") == cid), None)
                if chunk:
                    scored.append((entry.score, chunk))

    if scored:
        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored[:top_k]]

    # 无匹配时返回原始 chunk
    return chunks[:top_k]


def _tokenize(text: str) -> List[str]:
    """简单中文分词（字符级 + 连续词元）"""
    import re
    return re.findall(r"[\u4e00-\u9fff]|[a-zA-Z]+", text.lower())

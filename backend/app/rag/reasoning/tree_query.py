"""Tree-Structured Query Decomposition

来源：ragflow rag/advanced_rag/tree_structured_query_decomposition_retrieval.py

递归式深度检索：多路检索 -> 充分性判断 -> 子查询生成 -> 递归研究

用法:
    tree = TreeQueryOrchestrator(
        llm_call=async_chat_fn,
        retrieve_fn=async_retrieve_fn,
    )
    result = await tree.research("复杂问题", max_depth=3)
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Awaitable, Callable, Dict, List, Optional, Protocol

import structlog

from app.rag.reasoning._prompts import TREE_SUFFICIENCY_CHECK, TREE_MULTI_QUERY_GEN

logger = structlog.get_logger(__name__)


# --------------- 类型定义 ---------------

class RetrievalFn(Protocol):
    """检索函数签名"""
    async def __call__(self, query: str) -> Dict[str, Any]: ...


class ResearchResult:
    """研究结果"""
    __slots__ = ("chunks", "doc_aggs", "is_sufficient", "reasoning", "depth")

    def __init__(
        self,
        chunks: Optional[List[Dict]] = None,
        doc_aggs: Optional[List[Dict]] = None,
        is_sufficient: bool = False,
        reasoning: str = "",
        depth: int = 0,
    ):
        self.chunks = chunks or []
        self.doc_aggs = doc_aggs or []
        self.is_sufficient = is_sufficient
        self.reasoning = reasoning
        self.depth = depth

    def merge(self, other: "ResearchResult") -> None:
        """合并另一份结果，避免重复"""
        cids = {c.get("chunk_id", c.get("id")) for c in self.chunks}
        for c in other.chunks:
            cid = c.get("chunk_id", c.get("id"))
            if cid and cid not in cids:
                self.chunks.append(c)
                cids.add(cid)
        dids = {d.get("doc_id", d.get("id")) for d in self.doc_aggs}
        for d in other.doc_aggs:
            did = d.get("doc_id", d.get("id"))
            if did and did not in dids:
                self.doc_aggs.append(d)
                dids.add(did)


class TreeQueryOrchestrator:
    """树状查询分解编排器。

    流程:
    1. 多路检索 (可接知识库/Web/图谱)
    2. LLM 充分性判断
    3. 充分 -> 返回结果
    4. 不充分 -> 生成子查询，每个子查询递归研究(深度-1)
    5. 合并所有子查询结果

    Args:
        llm_call: async chat function — takes (system_prompt, messages, **kwargs), returns str
        retrieve_fn: async retrieval function — takes query str, returns {"chunks": [...], "doc_aggs": [...]}
        max_depth: 递归深度，默认 3
        max_iterations: 单次子查询数上限，默认 3
    """

    def __init__(
        self,
        llm_call: Optional[Callable[..., Awaitable[str]]] = None,
        retrieve_fn: Optional[RetrievalFn] = None,
        *,
        max_depth: int = 3,
        max_iterations: int = 3,
    ):
        self.llm_call = llm_call
        self.retrieve_fn = retrieve_fn
        self.max_depth = max_depth
        self.max_iterations = max_iterations
        self._lock = asyncio.Lock()
        self._all_chunks: List[Dict] = []
        self._all_docs: List[Dict] = []
        self._search_log: List[str] = []

    async def research(
        self,
        question: str,
        *,
        query: Optional[str] = None,
        max_depth: Optional[int] = None,
        callback: Optional[Callable[[str], Awaitable[None]]] = None,
    ) -> ResearchResult:
        """启动树状研究。

        Args:
            question: 最终要回答的问题
            query: 当前轮次的查询语句 (默认同 question)
            max_depth: 覆盖构造函数默认值
            callback: 可选的异步回调，用于实时输出进度
        """
        depth = max_depth if max_depth is not None else self.max_depth
        self._all_chunks = []
        self._all_docs = []
        self._search_log = []
        result = await self._research(question, query or question, depth, callback)
        result.depth = self.max_depth - depth
        return result

    # --------------- 内部递归 ---------------

    async def _research(
        self,
        question: str,
        query: str,
        depth: int,
        callback: Optional[Callable[[str], Awaitable[None]]] = None,
    ) -> ResearchResult:
        if depth <= 0:
            msg = f"[depth=0] 达到最大检索深度"
            self._search_log.append(msg)
            if callback:
                await callback(msg)
            return ResearchResult(
                chunks=list(self._all_chunks),
                doc_aggs=list(self._all_docs),
                reasoning="达到最大深度",
            )

        # 1. 检索
        if callback:
            await callback(f"[depth={depth}] 检索: {query}")
        self._search_log.append(f"[depth={depth}] 检索: {query}")

        kbinfos: Dict[str, Any] = {"chunks": [], "doc_aggs": []}
        try:
            if self.retrieve_fn:
                kbinfos = await self.retrieve_fn(query)
                self._search_log.append(
                    f"[depth={depth}] 检索 {len(kbinfos.get('chunks', []))} 条结果"
                )
            else:
                self._search_log.append(f"[depth={depth}] 检索函数未配置")
        except Exception as e:
            logger.error("retrieve_failed", error=str(e), depth=depth, query=query)
            self._search_log.append(f"[depth={depth}] 检索失败: {e}")

        await self._update_chunk_info(kbinfos)

        if callback:
            await callback(f"[depth={depth}] 检索到 {len(kbinfos.get('chunks', []))} 条结果")

        # 2. 充分性判断
        if not self.llm_call:
            return ResearchResult(
                chunks=list(self._all_chunks),
                doc_aggs=list(self._all_docs),
                reasoning="LLM未配置，跳过充分性检查",
            )

        if callback:
            await callback(f"[depth={depth}] 检查充分性...")

        suff = await self._check_sufficiency(question, kbinfos)

        if suff.get("is_sufficient"):
            if callback:
                await callback(f"[depth={depth}] 信息充分")
            self._search_log.append(f"[depth={depth}] 信息充分")
            return ResearchResult(
                chunks=list(self._all_chunks),
                doc_aggs=list(self._all_docs),
                is_sufficient=True,
                reasoning=suff.get("reasoning", ""),
            )

        # 3. 生成子查询
        missing = suff.get("missing_info", [])
        if callback:
            await callback(f"[depth={depth}] 信息不足，缺失: {', '.join(missing) if missing else '未知'}")
        self._search_log.append(f"[depth={depth}] 信息不足")

        sub_queries = await self._gen_multi_queries(
            question, query, missing, kbinfos
        )
        if not sub_queries:
            return ResearchResult(
                chunks=list(self._all_chunks),
                doc_aggs=list(self._all_docs),
                reasoning=suff.get("reasoning", ""),
            )

        if callback:
            await callback(f"[depth={depth}] 生成 {len(sub_queries)} 个子查询")
        self._search_log.append(f"[depth={depth}] 生成 {len(sub_queries)} 个子查询")

        # 4. 并发递归研究子查询
        tasks = []
        for sub_q in sub_queries:
            sub_query = sub_q.get("query", sub_q.get("question", ""))
            sub_question = sub_q.get("question", sub_query)
            task = asyncio.create_task(
                self._research(sub_question, sub_query, depth - 1, callback)
            )
            tasks.append(task)

        if callback:
            await callback(f"[depth={depth}] 并发研究 {len(tasks)} 个子问题...")

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for r in results:
            if isinstance(r, Exception):
                logger.error("sub_research_failed", error=str(r))

        return ResearchResult(
            chunks=list(self._all_chunks),
            doc_aggs=list(self._all_docs),
            reasoning=suff.get("reasoning", ""),
        )

    async def _update_chunk_info(self, kbinfos: Dict[str, Any]) -> None:
        """合并新检索结果到全局缓存 (asyncio.Lock thread-safe)

        来源：ragflow `_async_update_chunk_info()`
        """
        async with self._lock:
            existing_cids = {
                c.get("chunk_id", c.get("id"))
                for c in self._all_chunks
            }
            for c in kbinfos.get("chunks", []):
                cid = c.get("chunk_id", c.get("id"))
                if cid and cid not in existing_cids:
                    self._all_chunks.append(c)
                    existing_cids.add(cid)

            existing_dids = {
                d.get("doc_id", d.get("id"))
                for d in self._all_docs
            }
            for d in kbinfos.get("doc_aggs", []):
                did = d.get("doc_id", d.get("id"))
                if did and did not in existing_dids:
                    self._all_docs.append(d)
                    existing_dids.add(did)

    async def _check_sufficiency(
        self, question: str, kbinfos: Dict[str, Any]
    ) -> Dict[str, Any]:
        """LLM 判断检索到的内容是否足以回答问题。

        Returns:
            {"is_sufficient": bool, "reasoning": str, "missing_info": [...]}
        """
        prompt = TREE_SUFFICIENCY_CHECK.format(
            question=question,
            retrieved_docs=_format_docs(kbinfos.get("chunks", [])),
        )

        try:
            response = await self.llm_call(
                system_prompt="",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
            )
            response = response or ""
            json_str = _extract_json(response)
            if json_str:
                data = json.loads(json_str)
                return {
                    "is_sufficient": data.get("is_sufficient", False),
                    "reasoning": data.get("reasoning", ""),
                    "missing_info": data.get("missing_info", []),
                }
        except Exception as e:
            logger.error("sufficiency_check_failed", error=str(e))

        return {"is_sufficient": False, "reasoning": "解析失败", "missing_info": []}

    async def _gen_multi_queries(
        self,
        original_question: str,
        query: str,
        missing_info: List[str],
        kbinfos: Dict[str, Any],
    ) -> List[Dict[str, str]]:
        """为缺失信息生成多条互补查询。

        Returns:
            [{"question": str, "query": str}, ...]
        """
        prompt = TREE_MULTI_QUERY_GEN.format(
            original_query=query,
            original_question=original_question,
            retrieved_docs=_format_docs(kbinfos.get("chunks", [])),
            missing_info="\n".join(f"- {m}" for m in missing_info) if missing_info else "未指定",
        )

        try:
            response = await self.llm_call(
                system_prompt="",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
            )
            response = response or ""
            json_str = _extract_json(response)
            if json_str:
                data = json.loads(json_str)
                questions = data.get("questions", [])
                if isinstance(questions, list):
                    return questions[:self.max_iterations]
        except Exception as e:
            logger.error("multi_query_gen_failed", error=str(e))

        return []

    def get_search_log(self) -> List[str]:
        """返回研究过程日志"""
        return list(self._search_log)


# --------------- 工具函数 ---------------

def _format_docs(chunks: List[Dict], max_chars: int = 2000) -> str:
    """格式化 chunk 为可读文本"""
    parts: List[str] = []
    for i, c in enumerate(chunks):
        content = c.get("content", c.get("text", ""))
        parts.append(f"[Chunk {i}]{content[:max_chars]}")
    return "\n---\n".join(parts) if parts else "<no content>"


def _extract_json(text: str) -> Optional[str]:
    """从 LLM 输出中提取第一个 JSON 对象或数组"""
    start = text.find("{")
    if start >= 0:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i+1]
    start = text.find("[")
    if start >= 0:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "[":
                depth += 1
            elif text[i] == "]":
                depth -= 1
                if depth == 0:
                    return text[start:i+1]
    return None

"""
Search-o1 — 深度推理搜索

来源：UltraRAG servers/custom/src/custom.py search_o1_* 工具

流程：LLM 生成查询标签 → 提取搜索 → 检索 → 累积 → 最终答案提取
可作为 RAG pipeline 的可选推理模式。
"""

from __future__ import annotations

import re
from typing import Any, Dict, List


def search_o1_init(queries: List[str]) -> Dict[str, List[Any]]:
    """初始化 Search-o1 的列表状态"""
    n = len(queries)
    return {
        "total_subq_list": [["<PAD>"] for _ in range(n)],
        "total_reason_list": [["<PAD>"] for _ in range(n)],
        "total_final_info_list": [["<PAD>"] for _ in range(n)],
    }


def search_o1_combine(
    total_subq_list: List[List[Any]],
    extract_query_list: List[str],
    total_reason_list: List[List[Any]],
    extract_reason_list: List[str],
) -> Dict[str, List[Any]]:
    """将新提取的查询/推理累积到总列表中"""
    PAD = "<PAD>"

    for q, bucket in zip(extract_query_list, total_subq_list):
        if len(bucket) == 1 and bucket[0] == PAD:
            bucket[0] = q
        else:
            bucket.append(q)

    for c, bucket in zip(extract_reason_list, total_reason_list):
        if len(bucket) == 1 and bucket[0] == PAD:
            bucket[0] = c
        else:
            bucket.append(c)

    return {
        "total_subq_list": total_subq_list,
        "total_reason_list": total_reason_list,
    }


def search_o1_extract_query(ans_ls: List[str]) -> Dict[str, List[str]]:
    """从 LLM 响应中提取查询：
    <|begin_search_query|>...</|end_search_query|>
    """
    BEGIN = "<|begin_search_query|>"
    END = "<|end_search_query|>"
    pattern = re.escape(BEGIN) + r"(.*?)" + re.escape(END)

    def get_query(text: str) -> str:
        matches = re.findall(pattern, text, flags=re.DOTALL)
        if not matches:
            return ""
        q = matches[-1].strip()
        q = re.sub(r"\s+", " ", q).strip(" \"'")
        return q

    return {"extract_query_list": [get_query(ans) for ans in ans_ls]}


def search_o1_extract_reasoning(ans_ls: List[str]) -> Dict[str, List[str]]:
    """提取 search_o1 中的推理内容（在 search query 标签之前的部分）"""
    BEGIN = "<|begin_search_query|>"

    def get_content_before(text: str) -> str:
        if BEGIN not in text:
            return text.strip()
        return text.split(BEGIN, 1)[0].strip()

    return {"extract_reason_list": [get_content_before(ans) for ans in ans_ls]}


def search_o1_extract_final_info(ans_ls: List[str]) -> Dict[str, List[str]]:
    """提取 **Final Information** 之后的内容"""
    BEGIN = "**Final Information**"

    def get_content_after(text: str) -> str:
        if BEGIN not in text:
            return text.strip()
        return text.split(BEGIN, 1)[1].strip()

    return {"extract_final_infor_list": [get_content_after(ans) for ans in ans_ls]}


def search_o1_combine_final_info(
    total_final_info_list: List[List[Any]],
    extract_final_info_list: List[str],
) -> Dict[str, List[str]]:
    """将最终信息累积到总列表"""
    PAD = "<PAD>"

    for info, bucket in zip(extract_final_info_list, total_final_info_list):
        if len(bucket) == 1 and bucket[0] == PAD:
            bucket[0] = info
        else:
            bucket.append(info)

    return {"total_final_info_list": total_final_info_list}


class SearchO1Orchestrator:
    """Search-o1 推理管线编排器"""

    def __init__(self, llm_factory=None, retriever=None, max_depth: int = 3):
        self.llm = llm_factory
        self.retriever = retriever
        self.max_depth = max_depth

    async def run(self, query: str, context: str = "") -> Dict:
        """执行 Search-o1 深度推理

        Args:
            query: 用户查询
            context: 已有上下文

        Returns:
            {"subqueries": list, "reasoning": list, "final_info": list}
        """
        state = search_o1_init([query])

        for depth in range(self.max_depth):
            # LLM 生成带查询/推理的响应
            prompt = await self._build_prompt(query, context)
            response = await prompt

            # 提取查询和推理
            queries = search_o1_extract_query([response])
            reasons = search_o1_extract_reasoning([response])

            # 累积
            search_o1_combine(
                state["total_subq_list"],
                queries["extract_query_list"],
                state["total_reason_list"],
                reasons["extract_reason_list"],
            )

            # 实际检索
            for q in queries["extract_query_list"]:
                if q and q != "<PAD>" and self.retriever:
                    docs = await self.retriever.search(q, top_k=3)
                    context += "\n\n".join(
                        d.get("content", "") for d in docs
                    )

            # 提取最终信息
            final_info = search_o1_extract_final_info([response])
            search_o1_combine_final_info(
                state["total_final_info_list"],
                final_info["extract_final_infor_list"],
            )

            next_query = queries["extract_query_list"][-1] if queries["extract_query_list"] else ""
            if not next_query:
                break
            query = next_query

        return {
            "subqueries": state["total_subq_list"],
            "reasoning": state["total_reason_list"],
            "final_info": state["total_final_info_list"],
        }

    async def _build_prompt(self, query: str, context: str) -> str:
        """构建 Search-o1 prompt"""
        from app.rag.reasoning._prompts import SEARCH_O1_PROMPT

        return SEARCH_O1_PROMPT.format(
            query=query, context=context or "N/A"
        )

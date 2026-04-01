"""
IRCoT — Iterative Retrieval with Chain-of-Thought

来源：UltraRAG servers/custom/src/custom.py ircot_* 工具

流程：检索 → 推理 → 再检索 → 推理 循环
可作为 RAG pipeline 的可选推理模式。
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional


def ircot_get_first_sentence(ans_ls: List[str]) -> Dict[str, List[str]]:
    """从回答中提取第一句话，作为下一轮 IRCoT 的子查询种子"""
    result = []
    for ans in ans_ls:
        match = re.search(r"(.+?[。！？.!?])", ans)
        if match:
            result.append(match.group(1))
        else:
            result.append(ans.strip())
    return {"q_ls": result}


def ircot_extract_answer(ans_ls: List[str]) -> Dict[str, List[str]]:
    """使用 'so the answer is' 模式从 IRCoT 响应中提取最终答案"""
    pattern = re.compile(r"so the answer is[\s:]*([^\n]*)", re.IGNORECASE)
    extracted = []
    for ans in ans_ls:
        match = pattern.search(ans)
        if match:
            extracted.append(match.group(1).strip())
        else:
            extracted.append(ans.strip())
    return {"pred_ls": extracted}


class IRCoTOrchestrator:
    """IRCoT 推理管线编排器

    用法（伪代码）：
        orchestrator = IRCoTOrchestrator(llm=..., retriever=...)
        result = orchestrator.run(query, max_rounds=3)
    """

    def __init__(
        self,
        llm_factory=None,
        retriever=None,
        max_rounds: int = 3,
    ):
        self.llm = llm_factory
        self.retriever = retriever
        self.max_rounds = max_rounds

    async def run(
        self,
        query: str,
        context: Optional[str] = None,
        max_rounds: Optional[int] = None,
    ) -> Dict:
        """执行 IRCoT 迭代推理检索

        Args:
            query: 用户查询
            context: 已有上下文（可选）
            max_rounds: 最大迭代轮数

        Returns:
            {"answer": str, "reasoning": list[dict], "rounds": int}
        """
        rounds = max_rounds or self.max_rounds
        history: List[Dict] = []
        current_query = query
        accumulated_context = context or ""

        for i in range(rounds):
            # Step 1: LLM 推理（生成思考 + 子查询）
            llm_response = await self._llm_reason_with_context(
                current_query, accumulated_context
            )
            history.append({"round": i, "reasoning": llm_response})

            # Step 2: 提取第一句话作为下一轮查询种子
            first_sent_result = ircot_get_first_sentence([llm_response])
            next_query = first_sent_result.get("q_ls", [""])[0]

            if not next_query or i == rounds - 1:
                break

            # Step 3: 再检索
            new_context = ""
            if self.retriever:
                docs = await self.retriever.search(next_query, top_k=3)
                new_context = "\n\n".join(
                    d.get("content", "") for d in docs
                )
                accumulated_context += "\n\n" + new_context if accumulated_context else new_context

            current_query = next_query

        # 提取最终答案
        answers = ircot_extract_answer([h["reasoning"] for h in history])
        final_answer = answers["pred_ls"][-1] if answers["pred_ls"] else ""

        return {
            "answer": final_answer,
            "reasoning": history,
            "rounds": len(history),
        }

    async def _llm_reason_with_context(
        self, query: str, context: str
    ) -> str:
        """调用 LLM 生成带上下文的推理响应"""
        from app.rag.reasoning._prompts import IRCOT_PROM

        prompt = IRCOT_PROM.format(query=query, context=context)
        if self.llm:
            return await self.llm(prompt)
        return "No LLM available for reasoning"

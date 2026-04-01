"""医疗意图分类 — 原 app/rag/router.py 的 MedicalRouter。"""

from __future__ import annotations

import structlog

from app.core.config import settings
from app.core.llm.llm_factory import SmartRotatingLLM
from app.core.models.local_slm import local_slm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langsmith import traceable

logger = structlog.get_logger(__name__)


class MedicalRouter:
    """医疗意图分类器。retrieval / direct / crisis。"""

    def __init__(self):
        self.llm = SmartRotatingLLM(
            model_name=settings.LLM_MODEL,
            prefer_local=False,
        )
        self.prompt = ChatPromptTemplate.from_template(
            """你是一个医疗意图分类器。请判断用户输入的意图。

            可选类别:
            - retrieval: 用户询问医学知识、疾病症状、药品信息、治疗方案等 (例如: "感冒吃什么药", "糖尿病的症状").
            - direct: 用户进行打招呼、感谢、或者询问你是谁 (例如: "你好", "谢谢", "你是谁").
            - crisis: 用户表达自残、自杀倾向 (例如: "我想死").

            用户输入: {query}

            仅输出类别名称，不要输出其他内容。"""
        )
        self.chain = self.prompt | self.llm | StrOutputParser()

    def route(self, query: str) -> str:
        return self._route(query)

    @traceable(run_type="chain", name="MedicalRouter")
    def _route(self, query: str) -> str:
        try:
            category = self.chain.invoke({"query": query}).strip().lower()
            if category not in ("retrieval", "direct", "crisis"):
                return "retrieval"
            return category
        except Exception:
            return "retrieval"


class SemanticRouter:
    """语义路由 — Local SLM 意图分类。"""

    @traceable(run_type="chain", name="SemanticRouter")
    def route(self, query: str) -> dict:
        categories = ["GREETING", "CRISIS", "VAGUE_SYMPTOM", "COMPLEX_SYMPTOM"]
        try:
            import asyncio
            category = asyncio.run(local_slm.constrained_classify(
                query, categories, reasoning=False,
            ))
            intent_map = {
                "GREETING": "fast",
                "CRISIS": "expert",
                "VAGUE_SYMPTOM": "vague",
                "COMPLEX_SYMPTOM": "standard",
            }
            path = intent_map.get(category, "standard")
            return {
                "path": path,
                "complexity_score": 8 if path == "expert" else 3,
                "intent_raw": category,
                "reason": "Fast Track Classification (No-Think Mode)",
            }
        except Exception as e:
            logger.error(f"SemanticRouter Error: {e}")
            return {
                "path": "standard", "complexity_score": 5,
                "error": str(e), "intent_raw": "ERROR",
            }

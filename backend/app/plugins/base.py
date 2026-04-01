"""插件接口 — DomainPlugin ABC 及共享模型。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from pydantic import BaseModel, Field


class QueryContext(BaseModel):
    query: str
    domain: str = "medical"
    intent: str = "retrieval"
    top_k: int = 5


class SafetyResult(BaseModel):
    safe: bool = True
    warnings: list[str] = []
    crisis_detected: bool = False


class PluginResponse(BaseModel):
    answer: str
    context_docs: list[dict] = []
    safety: SafetyResult = Field(default_factory=SafetyResult)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DomainPlugin(ABC):
    """领域插件接口。每个领域（医疗/法律/…）实现此接口。"""

    NAME: ClassVar[str] = "base"

    @abstractmethod
    async def initialize(self) -> None:
        """加载插件规则、映射、模型等。"""
        ...

    @abstractmethod
    def classify_intent(self, query: str) -> str:
        """对用户查询进行意图分类。"""
        ...

    @abstractmethod
    async def check_safety(self, query: str) -> SafetyResult:
        """检查查询是否安全（危机词、危险内容等）。"""
        ...

    @abstractmethod
    def post_process(self, chunks: list[dict], ctx: QueryContext) -> list[dict]:
        """检索结果后处理（重排、部门一致性等）。"""
        ...

    @abstractmethod
    def format_response(self, answer: str, ctx: QueryContext) -> str:
        """格式化回答（添加免责声明、引用等）。"""
        ...

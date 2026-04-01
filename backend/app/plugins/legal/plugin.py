"""法律领域插件 — LegalDomainPlugin。"""

from __future__ import annotations

from ...plugins.base import DomainPlugin, QueryContext, SafetyResult

from .article_fetcher import ArticleFetcher
from .citation_formatter import CitationFormatter
from .validity_checker import ValidityChecker


class LegalDomainPlugin(DomainPlugin):
    NAME = "legal"

    def __init__(self):
        self.fetcher = ArticleFetcher()
        self.formatter = CitationFormatter()
        self.validity = ValidityChecker()

    async def initialize(self) -> None:
        pass  # 无需额外初始化

    def classify_intent(self, query: str) -> str:
        """法律意图分类：consult / lookup / greeting / analysis。"""
        if any(w in query for w in ("你好", "谢谢", "是谁")):
            return "greeting"
        if any(w in query for w in ("第", "条", "法条")):
            return "lookup"
        return "consult"

    async def check_safety(self, query: str) -> SafetyResult:
        warnings: list[str] = []
        # 检查是否提及已废止的旧法
        for law_name in ValidityChecker._superseded_laws:
            if law_name in query:
                warn = self.validity.warn_superseded(law_name)
                if warn:
                    warnings.append(warn)
        return SafetyResult(
            safe=True,
            warnings=warnings,
            crisis_detected=False,
        )

    def post_process(self, chunks: list[dict], ctx: QueryContext) -> list[dict]:
        for chunk in chunks:
            chunk["citation"] = self.formatter.format_context(chunk)
            article = chunk.get("metadata", {}).get("article_number", "")
            if article and not self.validity.is_article_valid(article):
                chunk["validity_warning"] = "此条文可能已失效"
        return sorted(chunks, key=lambda x: x.get("score", 0), reverse=True)

    def format_response(self, answer: str, ctx: QueryContext) -> str:
        disclaimer = (
            "\n\n以上信息仅供参考，不构成正式法律意见。"
            "具体法律问题请咨询专业律师。"
        )
        return f"{answer}{disclaimer}"


# Module-level export
legal_plugin = LegalDomainPlugin()

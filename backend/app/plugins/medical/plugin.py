"""医疗领域插件 — MedicalDomainPlugin。"""

from __future__ import annotations

import json
from pathlib import Path

from ...plugins.base import DomainPlugin, QueryContext, SafetyResult
from .ddinter import DDInterChecker
from .rules import MedicalRuleService, medical_rule_service


# 内置危机词（DB guardrail 的补充）
_CRISIS_TERMS = [
    "自杀", "自残", "想死", "活不下去", "不想活了", "跳楼", "割腕",
    "上吊", "服毒", "自焚", "轻生", "结束生命",
]


class MedicalDomainPlugin(DomainPlugin):
    NAME = "medical"

    def __init__(self):
        self.router = None
        self._semantic_router = None
        self.ddinter = None
        self.rule_service = medical_rule_service

        # 从 mappings 加载科室别名和症状映射
        self._dept_aliases: dict[str, str] = {}
        self._symptom_map: dict[str, str] = {}
        self._load_mappings()

    def _load_mappings(self):
        mappings_dir = Path(__file__).parent / "mappings"
        try:
            dept_path = mappings_dir / "departments.json"
            dept_data = json.loads(dept_path.read_text(encoding="utf-8"))
            self._dept_aliases = dept_data.get("aliases", {})
        except Exception:
            pass
        try:
            symp_path = mappings_dir / "symptoms.json"
            self._symptom_map = json.loads(symp_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    async def initialize(self) -> None:
        from .triage import MedicalRouter
        # 从 DB 加载医疗规则
        await self.rule_service.initialize()
        self.router = MedicalRouter()
        self.ddinter = DDInterChecker(rule_service=self.rule_service)

    def classify_intent(self, query: str) -> str:
        if self.router:
            return self.router.route(query)
        return "retrieval"

    async def check_safety(self, query: str) -> SafetyResult:
        warnings: list[str] = []
        crisis = False
        for term in _CRISIS_TERMS:
            if term in query:
                crisis = True
                warnings.append(f"Crisis keyword detected: {term}")
                break
        # DDI scan
        if self.ddinter:
            ddi_warnings = self.ddinter.scan_query_for_warnings(query)
            warnings.extend(ddi_warnings)
        return SafetyResult(
            safe=not crisis and not any("⚠️" in w for w in warnings),
            warnings=warnings,
            crisis_detected=crisis,
        )

    def post_process(self, chunks: list[dict], ctx: QueryContext) -> list[dict]:
        # 科室别名归一化
        normalized: list[str] = []
        for chunk in chunks:
            dept = chunk.get("department", "")
            if dept in self._dept_aliases:
                chunk["department"] = self._dept_aliases[dept]
            normalized.append(chunk)
        return sorted(normalized, key=lambda x: x.get("score", 0), reverse=True)

    def format_response(self, answer: str, ctx: QueryContext) -> str:
        disclaimer = "以上信息仅供参考，具体诊疗请咨询线下医生。"
        return f"{answer}\n\n{disclaimer}"


# Module-level export
medical_plugin = MedicalDomainPlugin()

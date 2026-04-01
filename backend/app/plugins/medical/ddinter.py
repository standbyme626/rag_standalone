"""药物相互作用检查 — 原 app/rag/ddinter_checker.py。"""

from __future__ import annotations

import asyncio
from typing import List, Dict, Optional

import structlog

logger = structlog.get_logger(__name__)


class DDInterChecker:
    """药物相互作用检查器。"""

    _db_available = True

    def __init__(self, rule_service=None):
        self.rule_service = rule_service

    def _extract_drugs(self, query: str) -> List[str]:
        if self.rule_service is None:
            return []
        found_drugs = []
        translation_map = self.rule_service.translation_map
        for cn_name in translation_map.keys():
            if len(cn_name) > 1 and cn_name in query:
                found_drugs.append(cn_name)
        common_drugs = self.rule_service.get_common_drugs()
        for d in common_drugs:
            if d not in found_drugs and d in query:
                found_drugs.append(d)
        return list(set(found_drugs))

    def _translate_to_en(self, drug_name: str) -> str:
        if self.rule_service:
            return self.rule_service.get_translation(drug_name)
        return drug_name

    async def check_interaction_in_db_async(
        self, drug_a_en: str, drug_b_en: str
    ) -> Optional[str]:
        if self.rule_service:
            risk_msg = self.rule_service.check_interaction(drug_a_en, drug_b_en)
            if risk_msg:
                return risk_msg
        if not self._db_available:
            return None
        try:
            from app.db.session import AsyncSessionLocal
            async with AsyncSessionLocal() as session:
                stmt = select(DDInterInteraction).where(
                    or_(
                        and_(
                            DDInterInteraction.drug_a == drug_a_en,
                            DDInterInteraction.drug_b == drug_b_en,
                        ),
                        and_(
                            DDInterInteraction.drug_a == drug_b_en,
                            DDInterInteraction.drug_b == drug_a_en,
                        ),
                    )
                ).limit(1)
                result = await session.execute(stmt)
                interaction = result.scalar_one_or_none()
                if interaction:
                    return (
                        f"Recorded Interaction: {interaction.description} "
                        f"(Severity: {interaction.severity})"
                    )
        except Exception as e:
            logger.warning("ddinter_db_async_check_failed", error=str(e))
        return None

    def check(self, drugs: list[str]) -> list[str]:
        warnings = []
        n = len(drugs)
        if n < 2:
            return warnings
        for i in range(n):
            for j in range(i + 1, n):
                d1_cn, d2_cn = drugs[i], drugs[j]
                d1_en = self._translate_to_en(d1_cn)
                d2_en = self._translate_to_en(d2_cn)
                risk_msg = self.rule_service.check_interaction(d1_en, d2_en) if self.rule_service else None
                if risk_msg:
                    warnings.append(
                        f"⚠️ 发现高危药物相互作用: [{d1_cn} + {d2_cn}] -> {risk_msg}"
                    )
        return list(set(warnings))

    async def check_async(self, drugs: list[str]) -> list[str]:
        warnings = []
        n = len(drugs)
        if n < 2:
            return warnings
        tasks = []
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                d1_cn, d2_cn = drugs[i], drugs[j]
                d1_en = self._translate_to_en(d1_cn)
                d2_en = self._translate_to_en(d2_cn)
                pairs.append((d1_cn, d2_cn))
                tasks.append(self.check_interaction_in_db_async(d1_en, d2_en))
        if not tasks:
            return []
        results = await asyncio.gather(*tasks)
        for idx, risk_msg in enumerate(results):
            if risk_msg:
                d1, d2 = pairs[idx]
                warnings.append(
                    f"⚠️ 发现高危药物相互作用: [{d1} + {d2}] -> {risk_msg}"
                )
        return list(set(warnings))

    def check_query_safety(self, query: str) -> bool:
        if self.rule_service:
            return self.rule_service.check_query_safety(query)
        return True

    def scan_query_for_warnings(self, query: str) -> List[str]:
        drugs = self._extract_drugs(query)
        if len(drugs) >= 2:
            return self.check(drugs)
        return []

    async def scan_query_for_warnings_async(self, query: str) -> List[str]:
        drugs = self._extract_drugs(query)
        if len(drugs) >= 2:
            return await self.check_async(drugs)
        return []

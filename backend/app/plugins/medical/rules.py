"""医疗规则服务 — 原 app/services/medical_rule_service.py。"""

from __future__ import annotations

from typing import List, Dict, Optional, Set

import structlog
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = structlog.get_logger(__name__)


class MedicalRuleService:
    """医疗规则服务。从 DB 加载规则到内存缓存。"""

    _instance = None

    _translation_cache: Dict[str, str] = {}
    _interaction_cache: List[Dict] = []
    _guardrail_cache: Set[str] = set()
    _common_drugs_cache: List[str] = []
    _cache_initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MedicalRuleService, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        pass

    async def initialize(self):
        if not self._cache_initialized:
            await self.refresh_rules()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry_error_callback=lambda retry_state: logger.error(
            "Retry exhausted", last_exception=retry_state.outcome.exception()
        ),
    )
    async def refresh_rules(self):
        logger.info("Refreshing medical rules from database...")
        try:
            from app.db.session import AsyncSessionLocal
            from app.db.models.medical_rules import DrugTranslation, DrugInteraction, SafetyGuardrail
            from sqlalchemy import select
            async with AsyncSessionLocal() as session:
                stmt_trans = select(DrugTranslation)
                result_trans = await session.execute(stmt_trans)
                translations = result_trans.scalars().all()
                new_trans_cache = {}
                common_drugs = []
                for t in translations:
                    new_trans_cache[t.cn_name] = t.en_name
                    common_drugs.append(t.cn_name)
                self._translation_cache = new_trans_cache
                self._common_drugs_cache = common_drugs

                stmt_inter = select(DrugInteraction)
                result_inter = await session.execute(stmt_inter)
                interactions = result_trans.scalars().all()
                new_inter_cache = []
                for i in interactions:
                    new_inter_cache.append({
                        "drugs": {i.drug_a, i.drug_b},
                        "description": i.description,
                        "severity": i.severity,
                    })
                self._interaction_cache = new_inter_cache

                stmt_guard = select(SafetyGuardrail)
                result_guard = await session.execute(stmt_guard)
                guardrails = result_guard.scalars().all()
                self._guardrail_cache = {
                    g.keyword for g in guardrails if g.action_type == "block"
                }
                self._cache_initialized = True
                logger.info(
                    "Medical rules refreshed",
                    translations=len(self._translation_cache),
                    interactions=len(self._interaction_cache),
                    guardrails=len(self._guardrail_cache),
                )
        except Exception as e:
            logger.error("Failed to refresh medical rules", error=str(e))

    def get_translation(self, cn_name: str) -> str:
        if cn_name in self._translation_cache:
            return self._translation_cache[cn_name]
        for k, v in self._translation_cache.items():
            if k in cn_name:
                return v
        return cn_name

    def check_interaction(self, drug_a_en: str, drug_b_en: str) -> Optional[str]:
        pair = {drug_a_en, drug_b_en}
        for combo in self._interaction_cache:
            if pair == combo["drugs"]:
                return f"{combo['description']} (Severity: {combo['severity']})"
        return None

    def check_query_safety(self, query: str) -> bool:
        for kw in self._guardrail_cache:
            if kw in query:
                return False
        return True

    def get_common_drugs(self) -> List[str]:
        return self._common_drugs_cache

    @property
    def translation_map(self) -> Dict[str, str]:
        return self._translation_cache.copy()


# Global instance (向后兼容)
medical_rule_service = MedicalRuleService()

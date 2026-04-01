# ⚠️ DEPRECATED — 已迁移到 app.plugins.medical.rules。
# 此文件仅保留向后兼容 re-export，将在后续版本移除。

import warnings
warnings.warn(
    "app.services.medical_rule_service 已弃用，请使用 app.plugins.medical.rules",
    DeprecationWarning,
    stacklevel=2,
)

from app.plugins.medical.rules import MedicalRuleService, medical_rule_service

__all__ = ["MedicalRuleService", "medical_rule_service"]

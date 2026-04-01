"""⚠️ DEPRECATED — 已迁移到 app.plugins.medical.triage。

此文件仅保留向后兼容 re-export，将在后续版本移除。
"""

import warnings
warnings.warn(
    "app.rag.router 已弃用，请使用 app.plugins.medical.triage",
    DeprecationWarning,
    stacklevel=2,
)

from app.plugins.medical.triage import MedicalRouter, SemanticRouter

__all__ = ["MedicalRouter", "SemanticRouter"]

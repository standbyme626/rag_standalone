# ⚠️ DEPRECATED — 已迁移到 app.plugins.medical.ddinter。
# 此文件仅保留向后兼容 re-export，将在后续版本移除。

import warnings
warnings.warn(
    "app.rag.ddinter_checker 已弃用，请使用 app.plugins.medical.ddinter",
    DeprecationWarning,
    stacklevel=2,
)

from app.plugins.medical.ddinter import DDInterChecker

__all__ = ["DDInterChecker"]

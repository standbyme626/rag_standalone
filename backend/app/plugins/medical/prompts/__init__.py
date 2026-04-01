"""
医疗领域插件 - Prompt 加载器
从 prompts/ 目录动态加载 prompt 模板，支持 f-string 格式化。
"""

from pathlib import Path
from typing import Optional
import re

_PROMPTS_DIR = Path(__file__).parent
_CACHE: dict[str, str] = {}


def _load(name: str) -> str:
    if name not in _CACHE:
        path = _PROMPTS_DIR / f"{name}.md"
        _CACHE[name] = path.read_text(encoding="utf-8")
    return _CACHE[name]


def intent_classification(query: str) -> str:
    return _load("intent_classification").format(query=query)


def system_intent() -> str:
    return _load("system_intent").strip()


def query_rewrite(query: str) -> str:
    return _load("query_rewrite").format(query=query)


def system_rewrite() -> str:
    return _load("system_rewrite").strip()


def hyde(query: str) -> str:
    return _load("hyde").format(query=query)


def system_hyde() -> str:
    return _load("system_hyde").strip()


def summarization(query: str, document: str, safety_instruction: str = "") -> str:
    return _load("summarization").format(
        query=query,
        document=document[:4000],
        safety_instruction=safety_instruction,
    )


def system_summarize() -> str:
    return _load("system_summarize").strip()


def safety_disclaimer() -> str:
    return _load("safety_disclaimer").strip()

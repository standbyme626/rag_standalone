"""查询增强辅助模块

Phase 3.4: 辅助能力
- 跨语言查询扩展 (3.4.1)
- 元数据过滤 LLM 提取 (3.4.2)
- Model Family Policy Engine (3.4.3)

用法:
    # 跨语言扩展
    enhancer = QueryEnhancer(llm_call=async_chat_fn)
    translations = await enhancer.cross_language_expand("diabetes", ["zh", "fr"])

    # 元数据过滤
    filters = await enhancer.extract_meta_filter(
        "2024年七月份上市的新品，不要蓝色的",
        metadata_keys=["color", "listing_date", "category"],
    )

    # 模型策略
    params = ModelFamilyPolicy.apply("Qwen3-72B", temperature=0.7)
"""

from __future__ import annotations

import json
import re
from datetime import date
from typing import Any, Awaitable, Callable, Dict, List, Optional

import structlog

logger = structlog.get_logger(__name__)


# --------------- Prompts ---------------

CROSS_LANGUAGE_SYS = """You are a streamlined multilingual translator.

Accept batch translation requests:
**Input:** `[text]`
===
`[comma-separated languages]`

Output translations separated by '###':
[Translation in language1]
###
[Translation in language2]

Maintain:
- Original formatting
- Technical terminology accuracy"""

CROSS_LANGUAGE_USER = """**Input:**
{query}
===
{languages}

**Output:**
"""

META_FILTER_SYS = """You are a metadata filtering condition generator. Analyze the user's question and output a JSON filter object.

Rules:
1. Output a dict with 'conditions' (array) and 'logic' ('and' or 'or').
2. Each condition: {{"key": str, "value": str, "op": str}}
3. Allowed ops: ["contains","not contains","in","not in","start with","end with","empty","not empty","=","≠",">","<","≥","≤"]
4. Dates: format as YYYY-MM-DD. Ranges: [>= start, < end]
5. Negations: use "≠" for exclusion terms.
6. Skip conditions if the attribute doesn't exist in metadata_keys.

Output ONLY the JSON dict. No additional text.

Today's date: {current_date}
Available metadata keys: {metadata_keys}
"""

META_FILTER_USER = """User query: "{user_question}"
Operator constraints: {constraints}"""


# --------------- Query Enhancer ---------------

class QueryEnhancer:
    """跨语言查询扩展 + 元数据过滤提取"""

    def __init__(self, llm_call: Optional[Callable[..., Awaitable[str]]] = None):
        self.llm_call = llm_call

    async def cross_language_expand(
        self, query: str, target_languages: List[str]
    ) -> Dict[str, str]:
        """跨语言查询扩展

        翻译查询 → 多语言同时检索 → 返回翻译映射

        来源：ragflow rag/prompts/cross_languages_*.md

        Args:
            query: 原始查询
            target_languages: 目标语言列表，如 ["zh", "en", "fr", "ja"]

        Returns:
            {language: translation, ...}
        """
        if not self.llm_call:
            return {}

        lang_map = {
            "zh": "Chinese", "en": "English", "fr": "French",
            "ja": "Japanese", "de": "German", "es": "Spanish",
            "ko": "Korean", "pt": "Portuguese", "ru": "Russian",
            "ar": "Arabic",
        }
        lang_names = [lang_map.get(l, l) for l in target_languages]

        sys_prompt = CROSS_LANGUAGE_SYS
        user_prompt = CROSS_LANGUAGE_USER.format(
            query=query,
            languages=", ".join(lang_names),
        )

        try:
            response = await self.llm_call(
                system_prompt=sys_prompt,
                messages=[{"role": "user", "content": user_prompt}],
                max_tokens=1024,
            )
            response = response or ""
            translations = [t.strip() for t in response.split("###")]
            result = {}
            for i, lang_code in enumerate(target_languages):
                if i < len(translations):
                    result[lang_code] = translations[i]
            return result
        except Exception as e:
            logger.error("cross_language_expand_failed", error=str(e))
            return {}

    async def extract_meta_filter(
        self,
        user_question: str,
        *,
        metadata_keys: Optional[List[str]] = None,
        constraints: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """元数据过滤 LLM 提取

        LLM 从查询提取 date/author/type → 检索层应用 filter

        来源：ragflow rag/prompts/meta_filter.md

        Args:
            user_question: 用户查询
            metadata_keys: 可用的元数据键
            constraints: 操作符约束

        Returns:
            {"logic": "and"|"or", "conditions": [{"key": str, "value": str, "op": str}]}
        """
        if not self.llm_call:
            return {"logic": "and", "conditions": []}

        sys_prompt = META_FILTER_SYS.format(
            current_date=date.today().isoformat(),
            metadata_keys=", ".join(metadata_keys) if metadata_keys else "无",
        )
        user_prompt = META_FILTER_USER.format(
            user_question=user_question,
            constraints=constraints,
        )

        try:
            response = await self.llm_call(
                system_prompt="",
                messages=[{"role": "user", "content": user_prompt}],
                max_tokens=512,
            )
            response = response or ""
            json_str = _extract_json(response)
            if json_str:
                data = json.loads(json_str)
                return {
                    "logic": data.get("logic", "and"),
                    "conditions": data.get("conditions", []),
                }
        except Exception as e:
            logger.error("meta_filter_extract_failed", error=str(e))

        return {"logic": "and", "conditions": []}


# --------------- Model Family Policy Engine ---------------

class ModelFamilyPolicy:
    """Model Family Policy Engine

    根据不同模型名自动调整参数。
    Qwen3/GPT-5/Kimi/Claude 等模型有不同的默认参数偏好。

    来源：ragflow rag/llm/chat_model.py _apply_model_family_policies
    """

    # 模型家族参数预设
    FAMILY_PARAMS = {
        "qwen": {
            "temperature": 0.7,
            "top_p": 0.8,
            "max_tokens": 4096,
            "enable_thinking": False,
        },
        "claude": {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 4096,
            "thinking": {"type": "disabled"},
        },
        "gpt": {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 4096,
        },
        "kimi": {
            "temperature": 0.7,
            "top_p": 0.8,
            "max_tokens": 4096,
        },
        "deepseek": {
            "temperature": 0.6,
            "top_p": 0.85,
            "max_tokens": 8192,
        },
        "llama": {
            "temperature": 0.6,
            "top_p": 0.9,
            "max_tokens": 4096,
        },
    }

    @classmethod
    def _detect_family(cls, model_name: str) -> str:
        """从模型名推断家族"""
        name = model_name.lower()
        if "qwen" in name:
            return "qwen"
        if "claude" in name:
            return "claude"
        if "gpt" in name or "chatgpt" in name or "openai" in name:
            return "gpt"
        if "kimi" in name or "moonshot" in name:
            return "kimi"
        if "deepseek" in name:
            return "deepseek"
        if "llama" in name:
            return "llama"
        return "generic"

    @classmethod
    def apply(
        cls,
        model_name: str,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 4096,
    ) -> Dict[str, Any]:
        """应用模型策略

        Args:
            model_name: 模型名称，如 "qwen3-72b" / "gpt-4o" / "claude-sonnet-4-6"
            temperature: 用户指定的温度 (会被策略覆盖)
            top_p: 用户 top_p
            max_tokens: 用户 max_tokens

        Returns:
            优化后的参数
        """
        family = cls._detect_family(model_name)
        params = cls.FAMILY_PARAMS.get(family, {
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        })

        # 用户参数覆盖（仅当策略中没有明确需要强制的项时）
        result = dict(params)
        if temperature != 0.7:
            result["temperature"] = temperature
        if top_p != 0.9:
            result["top_p"] = top_p
        if max_tokens != 4096:
            result["max_tokens"] = max_tokens

        return result

    @classmethod
    def recommend_model(
        cls,
        task: str,
        *,
        family: Optional[str] = None,
    ) -> Dict[str, str]:
        """根据任务类型推荐模型

        Args:
            task: "reasoning" / "generation" / "summarization" / "extraction"
            family: 可选 "qwen" / "claude" / "gpt" / None

        Returns:
            {"model": str, "reason": str}
        """
        if family == "qwen":
            recs = {
                "reasoning": "qwen3-72b",
                "generation": "qwen3-235b",
                "summarization": "qwen3-32b",
                "extraction": "qwen3-72b",
            }
        elif family == "claude":
            recs = {
                "reasoning": "claude-opus-4-6",
                "generation": "claude-sonnet-4-6",
                "summarization": "claude-sonnet-4-6",
                "extraction": "claude-sonnet-4-6",
            }
        elif family == "gpt":
            recs = {
                "reasoning": "o3",
                "generation": "gpt-4.1",
                "summarization": "gpt-4.1-mini",
                "extraction": "gpt-4.1-mini",
            }
        else:
            recs = {
                "reasoning": "qwen3-72b",
                "generation": "qwen3-235b",
                "summarization": "qwen3-32b",
                "extraction": "qwen3-72b",
            }

        return {"model": recs.get(task, "qwen3-32b"), "family": family or "qwen"}


# --------------- 工具函数 ---------------

def _extract_json(text: str) -> Optional[str]:
    """从文本中提取第一个 JSON 对象"""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None

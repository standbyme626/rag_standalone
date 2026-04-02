"""Tests for query enhancements (Phase 3.4)."""

import pytest
from unittest.mock import AsyncMock

from app.rag.query_enhance import (
    QueryEnhancer,
    ModelFamilyPolicy,
    _extract_json,
)


class TestExtractJson:
    def test_extract_object(self):
        text = 'Here: {"key":"value"}'
        assert _extract_json(text) == '{"key":"value"}'

    def test_no_json(self):
        assert _extract_json("no json") is None


class TestQueryEnhancer:
    @pytest.mark.asyncio
    async def test_cross_language_no_llm(self):
        enhancer = QueryEnhancer(llm_call=None)
        result = await enhancer.cross_language_expand("hello", ["zh"])
        assert result == {}

    @pytest.mark.asyncio
    async def test_cross_language_with_mock(self):
        mock_llm = AsyncMock()
        mock_llm.return_value = """你好世界
###
Bonjour le monde"""
        enhancer = QueryEnhancer(llm_call=mock_llm)
        result = await enhancer.cross_language_expand("Hello World", ["zh", "fr"])
        assert result["zh"] == "你好世界"
        assert result["fr"] == "Bonjour le monde"

    @pytest.mark.asyncio
    async def test_cross_language_error(self):
        mock_llm = AsyncMock(side_effect=RuntimeError("fail"))
        enhancer = QueryEnhancer(llm_call=mock_llm)
        result = await enhancer.cross_language_expand("test", ["zh"])
        assert result == {}

    @pytest.mark.asyncio
    async def test_meta_filter_no_llm(self):
        enhancer = QueryEnhancer(llm_call=None)
        result = await enhancer.extract_meta_filter("test")
        assert result == {"logic": "and", "conditions": []}

    @pytest.mark.asyncio
    async def test_meta_filter_with_mock(self):
        mock_llm = AsyncMock()
        mock_llm.return_value = """{"logic":"and","conditions":[{"key":"color","value":"blue","op":"≠"}]}"""
        enhancer = QueryEnhancer(llm_call=mock_llm)
        result = await enhancer.extract_meta_filter(
            "不要蓝色的",
            metadata_keys=["color", "category"],
        )
        assert result["logic"] == "and"
        assert len(result["conditions"]) == 1
        assert result["conditions"][0]["key"] == "color"

    @pytest.mark.asyncio
    async def test_meta_filter_error(self):
        mock_llm = AsyncMock(side_effect=RuntimeError("fail"))
        enhancer = QueryEnhancer(llm_call=mock_llm)
        result = await enhancer.extract_meta_filter("test")
        assert result == {"logic": "and", "conditions": []}


class TestModelFamilyPolicy:
    def test_detect_qwen(self):
        assert ModelFamilyPolicy._detect_family("qwen3-72b") == "qwen"

    def test_detect_claude(self):
        assert ModelFamilyPolicy._detect_family("claude-sonnet-4-6") == "claude"

    def test_detect_gpt(self):
        assert ModelFamilyPolicy._detect_family("gpt-4o") == "gpt"

    def test_detect_kimi(self):
        assert ModelFamilyPolicy._detect_family("kimi-k2") == "kimi"

    def test_detect_deepseek(self):
        assert ModelFamilyPolicy._detect_family("deepseek-v3") == "deepseek"

    def test_detect_llama(self):
        assert ModelFamilyPolicy._detect_family("llama-4") == "llama"

    def test_detect_generic(self):
        assert ModelFamilyPolicy._detect_family("unknown-model") == "generic"

    def test_apply_qwen(self):
        params = ModelFamilyPolicy.apply("qwen3-72b")
        assert "temperature" in params
        assert "enable_thinking" in params

    def test_apply_claude(self):
        params = ModelFamilyPolicy.apply("claude-sonnet-4-6")
        assert params["temperature"] == 0.7
        assert "thinking" in params

    def test_apply_user_override(self):
        params = ModelFamilyPolicy.apply("gpt-4o", temperature=0.3)
        assert params["temperature"] == 0.3  # user override

    def test_apply_generic_uses_defaults(self):
        params = ModelFamilyPolicy.apply("unknown-model")
        assert params["temperature"] == 0.7
        assert params["top_p"] == 0.9

    def test_recommend_qwen(self):
        rec = ModelFamilyPolicy.recommend_model("reasoning", family="qwen")
        assert "qwen3" in rec["model"]

    def test_recommend_claude(self):
        rec = ModelFamilyPolicy.recommend_model("reasoning", family="claude")
        assert "claude" in rec["model"]

    def test_recommend_default(self):
        rec = ModelFamilyPolicy.recommend_model("generation")
        assert rec["family"] == "qwen"  # default family

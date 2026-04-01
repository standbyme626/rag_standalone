"""Tests for app/plugins/ — ABC methods, models, registry, Medical & Legal plugins."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ===================================================================
# Plugin base — ABC methods
# ===================================================================

class TestDomainPluginAbc:
    def test_domain_plugin_abc_methods(self):
        """DomainPlugin should define the expected abstract methods."""
        from app.plugins.base import DomainPlugin
        import inspect

        required = {"initialize", "classify_intent", "check_safety", "post_process", "format_response"}
        for name in required:
            method = getattr(DomainPlugin, name, None)
            assert method is not None, f"DomainPlugin missing method '{name}'"
            assert inspect.isfunction(method) or inspect.ismethod(method), \
                f"'{name}' is not a function/method"

    def test_domain_plugin_cannot_instantiate(self):
        from app.plugins.base import DomainPlugin
        with pytest.raises(TypeError):
            DomainPlugin()


# ===================================================================
# Plugin models (Pydantic)
# ===================================================================

class TestQueryContext:
    def test_query_context_creation(self):
        from app.plugins.base import QueryContext
        ctx = QueryContext(query="hello", domain="medical", intent="retrieval", top_k=3)
        assert ctx.query == "hello"
        assert ctx.domain == "medical"
        assert ctx.intent == "retrieval"
        assert ctx.top_k == 3

    def test_query_context_defaults(self):
        from app.plugins.base import QueryContext
        ctx = QueryContext(query="test")
        assert ctx.domain == "medical"
        assert ctx.intent == "retrieval"
        assert ctx.top_k == 5


class TestSafetyResult:
    def test_safety_result_creation(self):
        from app.plugins.base import SafetyResult
        sr = SafetyResult(safe=False, warnings=["bad"], crisis_detected=True)
        assert sr.safe is False
        assert sr.warnings == ["bad"]
        assert sr.crisis_detected is True

    def test_safety_result_defaults(self):
        from app.plugins.base import SafetyResult
        sr = SafetyResult()
        assert sr.safe is True
        assert sr.warnings == []
        assert sr.crisis_detected is False


class TestPluginResponse:
    def test_plugin_response_creation(self):
        from app.plugins.base import PluginResponse, SafetyResult
        resp = PluginResponse(
            answer="yes",
            context_docs=[{"url": "/doc1"}],
            safety=SafetyResult(safe=True),
            metadata={"extra": 1},
        )
        assert resp.answer == "yes"
        assert len(resp.context_docs) == 1
        assert resp.safety.safe is True
        assert resp.metadata["extra"] == 1

    def test_plugin_response_defaults(self):
        from app.plugins.base import PluginResponse
        resp = PluginResponse(answer="ok")
        assert resp.context_docs == []
        assert resp.safety.safe is True
        assert resp.metadata == {}


# ===================================================================
# Plugin registry
# ===================================================================

class TestPluginRegistry:
    def test_registry_register_and_get(self):
        from app.plugins.base import DomainPlugin
        from app.plugins.registry import register_plugin, get_plugin, _PLUGIN_REGISTRY

        # Save state
        old_keys = set(_PLUGIN_REGISTRY.keys())

        # Create a minimal concrete plugin
        class DummyPlugin(DomainPlugin):
            NAME = "dummy_test"
            async def initialize(self): pass
            def classify_intent(self, query): return "retrieval"
            async def check_safety(self, query):
                from app.plugins.base import SafetyResult
                return SafetyResult()
            def post_process(self, chunks, ctx): return chunks
            def format_response(self, answer, ctx): return answer

        plugin = DummyPlugin()
        register_plugin("dummy_test", plugin)
        assert get_plugin("dummy_test") is plugin

        # Cleanup
        for k in list(_PLUGIN_REGISTRY.keys()):
            if k not in old_keys:
                del _PLUGIN_REGISTRY[k]

    def test_list_plugins(self):
        from app.plugins.registry import list_plugins, register_plugin, _PLUGIN_REGISTRY
        from app.plugins.base import DomainPlugin, SafetyResult, QueryContext

        old_keys = set(_PLUGIN_REGISTRY.keys())

        class TempPlugin(DomainPlugin):
            NAME = "temp_list_test"
            async def initialize(self): pass
            def classify_intent(self, q): return "consult"
            async def check_safety(self, q): return SafetyResult()
            def post_process(self, c, ctx): return c
            def format_response(self, a, ctx): return a

        register_plugin("temp_list_test", TempPlugin())
        names = list_plugins()
        assert "temp_list_test" in names

        # Cleanup
        for k in list(_PLUGIN_REGISTRY.keys()):
            if k not in old_keys:
                del _PLUGIN_REGISTRY[k]

    def test_get_plugin_missing_raises(self):
        from app.plugins.registry import get_plugin
        with pytest.raises(ValueError, match="not found"):
            get_plugin("definitely_not_registered_xyz")


# ===================================================================
# MedicalDomainPlugin
# ===================================================================

class TestMedicalPluginSafety:
    def test_medical_plugin_safety_crisis_keyword(self):
        from app.plugins.medical import MedicalDomainPlugin
        plugin = MedicalDomainPlugin()
        import asyncio
        result = asyncio.run(plugin.check_safety("I feel like I want to die 我想死"))
        assert result.crisis_detected is True
        assert result.safe is False
        assert any("Crisis keyword" in w for w in result.warnings)

    def test_medical_plugin_safety_non_crisis(self):
        from app.plugins.medical import MedicalDomainPlugin
        plugin = MedicalDomainPlugin()
        import asyncio
        result = asyncio.run(plugin.check_safety("What causes a headache?"))
        assert result.crisis_detected is False
        assert result.safe is True

    def test_medical_post_process_dept_alias(self):
        from app.plugins.medical import MedicalDomainPlugin
        from app.plugins.base import QueryContext
        plugin = MedicalDomainPlugin()
        ctx = QueryContext(query="test", domain="medical")

        chunks = [
            {"department": "cardiology", "score": 0.9},
            {"department": "骨科", "score": 0.8},
            {"department": "消化科", "score": 0.5},
        ]
        result = plugin.post_process(chunks, ctx)

        # "cardiology" alias should be mapped to "心内科"
        cardiology_chunk = [c for c in result if c["department"] == "心内科"][0]
        assert cardiology_chunk["score"] == 0.9
        # "骨科" stays as "骨科" (top-level key, not an alias)
        assert any(c["department"] == "骨科" for c in result)

    def test_medical_post_process_sort_score_desc(self):
        from app.plugins.medical import MedicalDomainPlugin
        from app.plugins.base import QueryContext
        plugin = MedicalDomainPlugin()
        ctx = QueryContext(query="test")
        chunks = [
            {"score": 0.1, "department": ""},
            {"score": 0.9, "department": ""},
            {"score": 0.5, "department": ""},
        ]
        result = plugin.post_process(chunks, ctx)
        scores = [c["score"] for c in result]
        assert scores == sorted(scores, reverse=True)


class TestMedicalFormatResponse:
    def test_medical_format_disclaimer(self):
        from app.plugins.medical import MedicalDomainPlugin
        from app.plugins.base import QueryContext
        plugin = MedicalDomainPlugin()
        resp = plugin.format_response("Take two aspirin", QueryContext(query="cough"))
        assert "仅供参考" in resp
        assert "咨询线下医生" in resp


class TestMedicalClassifyIntent:
    """Non-async parts of MedicalDomainPlugin — without router (router requires settings)."""
    def test_medical_classify_intent_fallback(self):
        from app.plugins.medical import MedicalDomainPlugin
        plugin = MedicalDomainPlugin()
        # router is None when not initialized, so classify_intent returns "retrieval"
        result = plugin.classify_intent("hello")
        assert result == "retrieval"


# ===================================================================
# LegalDomainPlugin
# ===================================================================

class TestLegalClassifyIntent:
    def test_legal_classify_intent_greeting(self):
        from app.plugins.legal import LegalDomainPlugin
        plugin = LegalDomainPlugin()
        assert plugin.classify_intent("你好") == "greeting"
        assert plugin.classify_intent("谢谢") == "greeting"

    def test_legal_classify_intent_lookup(self):
        from app.plugins.legal import LegalDomainPlugin
        plugin = LegalDomainPlugin()
        assert plugin.classify_intent("第二条是什么") == "lookup"
        assert plugin.classify_intent("查一下刑法第一条 法条") == "lookup"

    def test_legal_classify_intent_consult(self):
        from app.plugins.legal import LegalDomainPlugin
        plugin = LegalDomainPlugin()
        assert plugin.classify_intent("How do I file a lawsuit?") == "consult"


class TestLegalFormatResponse:
    def test_legal_format_disclaimer(self):
        from app.plugins.legal import LegalDomainPlugin
        from app.plugins.base import QueryContext
        plugin = LegalDomainPlugin()
        resp = plugin.format_response("You can sue here", QueryContext(query="lawsuit"))
        assert "不构成正式法律意见" in resp
        assert "咨询专业律师" in resp


class TestLegalSafety:
    def test_legal_safety_warns_superseded(self):
        from app.plugins.legal import LegalDomainPlugin
        import asyncio
        plugin = LegalDomainPlugin()
        result = asyncio.run(plugin.check_safety("According to 合同法 section 3"))
        assert result.safe is True  # safety itself is True
        assert len(result.warnings) >= 1
        assert any("民法典" in w for w in result.warnings)

    def test_legal_safety_no_warning_current_law(self):
        from app.plugins.legal import LegalDomainPlugin
        import asyncio
        plugin = LegalDomainPlugin()
        result = asyncio.run(plugin.check_safety("What does the Civil Code say?"))
        assert result.safe is True
        assert result.crisis_detected is False
        assert result.warnings == []


# ===================================================================
# CitationFormatter
# ===================================================================

class TestCitationFormatter:
    def test_citation_formatter_format_article(self):
        from app.plugins.legal.citation_formatter import CitationFormatter
        result = CitationFormatter.format_article(128)
        assert "中华人民共和国民法典" in result
        assert "第128条" in result

    def test_citation_formatter_format_article_with_title(self):
        from app.plugins.legal.citation_formatter import CitationFormatter
        result = CitationFormatter.format_article(42, "违约责任")
        assert "违约责任" in result
        assert "《中华人民共和国民法典》第42条" in result

    def test_citation_formatter_format_section(self):
        from app.plugins.legal.citation_formatter import CitationFormatter
        result = CitationFormatter.format_section("总则编")
        assert "总则编" in result

    def test_citation_formatter_format_context(self):
        from app.plugins.legal.citation_formatter import CitationFormatter
        doc = {"metadata": {"article_number": 15, "section": "总则"}}
        result = CitationFormatter.format_context(doc)
        assert "第15条" in result
        assert "总则" in result


# ===================================================================
# ValidityChecker
# ===================================================================

class TestValidityChecker:
    def test_validity_checker_article_bounds_valid(self):
        from app.plugins.legal.validity_checker import ValidityChecker
        vc = ValidityChecker()
        assert vc.is_article_valid(1) is True
        assert vc.is_article_valid(1260) is True
        assert vc.is_article_valid("500") is True

    def test_validity_checker_article_bounds_invalid(self):
        from app.plugins.legal.validity_checker import ValidityChecker
        vc = ValidityChecker()
        assert vc.is_article_valid(0) is False
        assert vc.is_article_valid(1261) is False
        assert vc.is_article_valid(-1) is False

    def test_validity_checker_superseded_law(self):
        from app.plugins.legal.validity_checker import ValidityChecker
        vc = ValidityChecker()
        assert vc.check_superseded("婚姻法") is True
        assert vc.check_superseded("合同法") is True
        assert vc.check_superseded("担保法") is True
        assert vc.check_superseded("物权法") is True
        assert vc.check_superseded("民法典") is False  # current law
        assert vc.check_superseded("random law xyz") is False

    def test_validity_checker_warn_superseded(self):
        from app.plugins.legal.validity_checker import ValidityChecker
        vc = ValidityChecker()
        warn = vc.warn_superseded("婚姻法")
        assert warn is not None
        assert "民法典" in warn
        assert "2021-01-01" in warn

    def test_validity_checker_warn_superseded_none(self):
        from app.plugins.legal.validity_checker import ValidityChecker
        vc = ValidityChecker()
        assert vc.warn_superseded("民法典") is None

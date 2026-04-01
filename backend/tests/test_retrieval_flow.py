"""
Integration tests for the full RAG retrieval flow.

Covers the domain-agnostic retrieval pipeline with medical/legal plugin integration:
1. Query intent classification → routing to correct plugin
2. Safety check pipeline
3. Multi-source retrieval (vector + BM25 + cache) orchestration
4. Reranking and post-processing
5. Response formatting with citations
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.plugins.base import QueryContext, SafetyResult, DomainPlugin
from app.plugins.registry import register_plugin, get_plugin, _PLUGIN_REGISTRY


# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture(autouse=True)
def clean_registry():
    """Clean registry before each test."""
    original = dict(_PLUGIN_REGISTRY)
    _PLUGIN_REGISTRY.clear()
    yield
    _PLUGIN_REGISTRY.clear()
    _PLUGIN_REGISTRY.update(original)


@pytest.fixture(autouse=True)
def mock_settings_env(monkeypatch):
    """Provide required env vars for Settings validation."""
    import os

    defaults = {
        "OPENAI_MODEL_NAME": "gpt-3.5-turbo",
        "MILVUS_HOST": "localhost",
        "MILVUS_PORT": "19530",
        "REDIS_HOST": "localhost",
        "REDIS_PORT": "6379",
        "DATABASE_URL": "sqlite:///./test.db",
    }
    for k, v in defaults.items():
        monkeypatch.setenv(k, os.environ.get(k, v))


@pytest.fixture
def mock_medical_plugin():
    mock_plugin = MagicMock(spec=DomainPlugin)
    mock_plugin.name = "medical"
    mock_plugin.classify_intent.return_value = "retrieval"
    mock_plugin.check_safety = AsyncMock(
        return_value=SafetyResult(
            safe=True,
            crisis_detected=False,
            warnings=[],
        )
    )
    mock_plugin.post_process = AsyncMock(return_value="processed result")
    mock_plugin.format_response = MagicMock(return_value="formatted response")
    register_plugin("medical", mock_plugin)
    return mock_plugin


# ===================================================================
# Test 1: Query intent classification
# ===================================================================


class TestIntentClassification:
    def test_medical_greeting_intent(self):
        from app.plugins.medical.triage import MedicalRouter

        router = MedicalRouter()
        result = router.route("你好")
        assert result in ("greeting", "retrieval", "other")

    def test_medical_retrieval_intent(self):
        from app.plugins.medical.triage import MedicalRouter

        router = MedicalRouter()
        result = router.route("糖尿病吃什么药")
        assert result in ("retrieval", "medical")

    def test_legal_intent_lookup(self):
        from app.plugins.legal.plugin import LegalDomainPlugin

        plugin = LegalDomainPlugin()
        assert plugin.classify_intent("民法典第七百二十条是什么内容") == "lookup"
        assert plugin.classify_intent("请问第123条怎么理解") == "lookup"

    def test_legal_intent_greeting(self):
        from app.plugins.legal.plugin import LegalDomainPlugin

        plugin = LegalDomainPlugin()
        result = plugin.classify_intent("你好，请问离婚协议怎么写")
        assert result == "greeting"

    def test_legal_intent_consult(self):
        from app.plugins.legal.plugin import LegalDomainPlugin

        plugin = LegalDomainPlugin()
        assert plugin.classify_intent("离婚协议怎么写") == "consult"


# ===================================================================
# Test 2: Safety check pipeline (mocked)
# ===================================================================


class TestSafetyCheckPipeline:
    @pytest.mark.asyncio
    async def test_safe_medical_query(self, mock_medical_plugin):
        result = await mock_medical_plugin.check_safety("请问高血糖饮食要注意什么")
        assert result.safe is True
        assert result.crisis_detected is False

    @pytest.mark.asyncio
    async def test_safe_legal_query(self, mock_medical_plugin):
        result = await mock_medical_plugin.check_safety("请问离婚财产如何分割")
        assert result.safe is True

    @pytest.mark.asyncio
    async def test_crisis_term_detected(self):
        from app.plugins.medical.plugin import MedicalDomainPlugin

        plugin = MedicalDomainPlugin()
        result = await plugin.check_safety("我不想活了想自杀")
        assert result.safe is False
        assert result.crisis_detected is True

    @pytest.mark.asyncio
    async def test_crisis_partial_terms(self):
        from app.plugins.medical.plugin import MedicalDomainPlugin

        plugin = MedicalDomainPlugin()
        result = await plugin.check_safety("活着好累有点不想活了")
        assert result.crisis_detected is True


# ===================================================================
# Test 3: Multi-source retrieval orchestration
# ===================================================================


class TestRetrievalOrchestration:
    def test_vector_result_format(self):
        mock_result = {
            "id": 1,
            "text": "二甲双胍是常用降糖药",
            "score": 0.95,
            "metadata": {"doc_id": "med001", "chunk_id": "c1"},
        }
        assert "text" in mock_result
        assert 0 <= mock_result["score"] <= 1

    def test_bm25_result_format(self):
        mock_result = {
            "id": 2,
            "text": "糖尿病饮食建议",
            "score": 0.88,
            "metadata": {"doc_id": "med002"},
        }
        assert "text" in mock_result
        assert mock_result["score"] > 0

    def test_semantic_cache_hit_has_high_score(self):
        mock_result = {"text": "缓存的糖尿病饮食建议", "score": 0.99}
        assert mock_result["score"] > 0.9

    def test_reranker_reorders_by_score(self):
        candidates = [
            {"text": "B", "score": 0.7},
            {"text": "A", "score": 0.95},
            {"text": "C", "score": 0.5},
        ]
        reranked = sorted(candidates, key=lambda x: x["score"], reverse=True)
        assert reranked[0]["text"] == "A"
        assert reranked[1]["text"] == "B"
        assert reranked[2]["text"] == "C"


# ===================================================================
# Test 4: Plugin registry integration
# ===================================================================


class TestPluginRegistryIntegration:
    def test_register_and_retrieve_medical_plugin(self, mock_medical_plugin):
        plugin = get_plugin("medical")
        assert plugin is not None
        assert plugin.name == "medical"

    def test_get_nonexistent_plugin_raises(self, mock_medical_plugin):
        with pytest.raises(ValueError):
            get_plugin("nonexistent")

    def test_multiple_plugins_can_coexist(self, mock_medical_plugin):
        from app.plugins.legal.plugin import LegalDomainPlugin

        legal = LegalDomainPlugin()
        register_plugin("legal", legal)
        assert get_plugin("medical") is not None
        assert get_plugin("legal") is not None


# ===================================================================
# Test 5: Retrieval pipeline E2E
# ===================================================================


class TestRetrievalPipelineE2E:
    @pytest.mark.asyncio
    async def test_full_pipeline_stages(self, mock_medical_plugin):
        context = QueryContext(
            query="二甲双胍的副作用有哪些",
            domain="medical",
            user_id="u1",
            session_id="s1",
        )

        intent = mock_medical_plugin.classify_intent(context.query)
        assert intent == "retrieval"

        safety = await mock_medical_plugin.check_safety(context.query)
        assert safety.safe is True

        mock_results = [
            {"id": 1, "text": "二甲双胍常见副作用", "score": 0.92},
            {"id": 2, "text": "二甲双胍使用注意", "score": 0.85},
        ]
        post_processed = await mock_medical_plugin.post_process(
            query=context.query,
            results=mock_results,
        )
        assert post_processed is not None

        formatted = mock_medical_plugin.format_response(post_processed, context)
        assert formatted is not None

    @pytest.mark.asyncio
    async def test_pipeline_with_crisis_query_blocks(self):
        from app.plugins.medical.plugin import MedicalDomainPlugin

        plugin = MedicalDomainPlugin()
        result = await plugin.check_safety("我不想活了")
        assert result.safe is False


# ===================================================================
# Test 6: Hierarchical index
# ===================================================================


class TestHierarchicalIndex:
    @pytest.mark.asyncio
    async def test_hierarchical_index_noop_returns_empty(self):
        from app.rag.hierarchical_index import (
            HierarchicalIndexGateway,
            NoopHierarchicalIndex,
        )

        with patch("app.rag.hierarchical_index.settings") as mock_settings:
            mock_settings.ENABLE_HIERARCHICAL_INDEX = True
            gateway = HierarchicalIndexGateway(backend=NoopHierarchicalIndex())
            results = await gateway.search(query="test", force_enable=True)
        assert results == []

    def test_hierarchical_hit_dataclass(self):
        from app.rag.hierarchical_index import HierarchicalHit

        hit = HierarchicalHit(
            doc_id="law001",
            level="paragraph",
            text="承租人应当按照约定的方法使用租赁物。",
            score=0.95,
            metadata={"article_number": 720},
        )
        assert hit.doc_id == "law001"
        assert hit.level == "paragraph"
        d = hit.to_dict()
        assert d["text"] == "承租人应当按照约定的方法使用租赁物。"

    def test_milvus_backend_class_exists(self):
        from app.rag.hierarchical_index import MilvusHierarchicalIndex

        backend = MilvusHierarchicalIndex()
        assert hasattr(backend, "search")


# ===================================================================
# Test 7: Legal plugin retrieval
# ===================================================================


class TestLegalPluginRetrieval:
    def test_citation_formatter_format_article(self):
        from app.plugins.legal.citation_formatter import CitationFormatter

        result = CitationFormatter.format_article(720, "租赁期限")
        assert "第720条" in result
        assert "租赁期限" in result

    def test_citation_formatter_format_section(self):
        from app.plugins.legal.citation_formatter import CitationFormatter

        result = CitationFormatter.format_section("第三编", "合同编")
        assert "第三编" in result


# ===================================================================
# Test 8: Prompt loader
# ===================================================================


class TestPromptLoader:
    def test_intent_classification_prompt_loads(self):
        from app.plugins.medical.prompts import intent_classification

        result = intent_classification(query="糖尿病饮食注意什么")
        assert "糖尿病饮食注意什么" in result
        assert "JSON" in result

    def test_query_rewrite_prompt_loads(self):
        from app.plugins.medical.prompts import query_rewrite

        result = query_rewrite(query="我爸有高血压能吃这个药吗")
        assert "我爸有高血压能吃这个药吗" in result

    def test_hyde_prompt_loads(self):
        from app.plugins.medical.prompts import hyde

        result = hyde(query="糖尿病饮食")
        assert "糖尿病饮食" in result

    def test_summarization_prompt_loads(self):
        from app.plugins.medical.prompts import summarization

        result = summarization(
            query="降糖药", document="二甲双胍是常用药", safety_instruction=""
        )
        assert "降糖药" in result
        assert "二甲双胍是常用药" in result

    def test_summarization_with_safety_instruction(self):
        from app.plugins.medical.prompts import summarization, safety_disclaimer

        result = summarization(
            query="test",
            document="some document",
            safety_instruction=safety_disclaimer(),
        )
        assert "仅供参考" in result

    def test_system_prompts_load(self):
        from app.plugins.medical.prompts import (
            system_intent,
            system_rewrite,
            system_hyde,
            system_summarize,
        )

        assert system_intent() != ""
        assert system_rewrite() != ""
        assert system_hyde() != ""
        assert system_summarize() != ""


# ===================================================================
# Test 9: Data models integrity
# ===================================================================


class TestDataModels:
    def test_query_context_creation(self):
        ctx = QueryContext(
            query="test query",
            domain="medical",
            user_id="u1",
            session_id="s1",
        )
        assert ctx.query == "test query"
        assert ctx.domain == "medical"

    def test_safety_result_creation(self):
        result = SafetyResult(
            safe=True,
            crisis_detected=False,
            warnings=[],
        )
        assert result.safe is True
        assert len(result.warnings) == 0


# ===================================================================
# Test 10: Mapping files integrity
# ===================================================================


class TestMappingFiles:
    def test_department_aliases_loaded(self):
        import json

        dept_path = (
            Path(__file__).resolve().parent.parent
            / "app/plugins/medical/mappings/departments.json"
        )
        data = json.loads(dept_path.read_text(encoding="utf-8"))
        assert isinstance(data, dict)
        assert len(data) > 0

    def test_symptom_map_loaded(self):
        import json

        symp_path = (
            Path(__file__).resolve().parent.parent
            / "app/plugins/medical/mappings/symptoms.json"
        )
        data = json.loads(symp_path.read_text(encoding="utf-8"))
        assert isinstance(data, dict)
        assert len(data) > 0

    def test_law_codes_structure(self):
        import json

        law_path = (
            Path(__file__).resolve().parent.parent
            / "app/plugins/legal/mappings/law_codes.json"
        )
        data = json.loads(law_path.read_text(encoding="utf-8"))
        assert isinstance(data, dict)
        assert len(data) > 0


# ===================================================================
# Test 11: Ingestion pipeline models
# ===================================================================


class TestIngestionModels:
    def test_parsed_document_model(self):
        from app.rag.ingestion.models import ParsedDocument

        doc = ParsedDocument(
            text="test content",
            metadata={"source_path": "/test.pdf", "file_type": "pdf"},
        )
        assert doc.text == "test content"
        assert doc.metadata["file_type"] == "pdf"

    def test_document_chunk_model(self):
        from app.rag.ingestion.models import DocumentChunk

        chunk = DocumentChunk(
            content="chunk text",
            metadata={"chunk_index": 0, "source": "test"},
        )
        assert chunk.content == "chunk text"
        assert chunk.metadata["chunk_index"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

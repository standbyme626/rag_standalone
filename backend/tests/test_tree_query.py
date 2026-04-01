"""Tests for Tree-Structured Query Decomposition (Phase 3.1)."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

from app.rag.reasoning.tree_query import (
    TreeQueryOrchestrator,
    ResearchResult,
    _format_docs,
    _extract_json,
)


class TestResearchResult:
    def test_default_values(self):
        r = ResearchResult()
        assert r.chunks == []
        assert r.doc_aggs == []
        assert r.is_sufficient is False
        assert r.depth == 0

    def test_merge_no_duplicates(self):
        a = ResearchResult(
            chunks=[{"chunk_id": "c1"}, {"chunk_id": "c2"}],
            doc_aggs=[{"doc_id": "d1"}],
        )
        b = ResearchResult(
            chunks=[{"chunk_id": "c2"}, {"chunk_id": "c3"}],
            doc_aggs=[{"doc_id": "d1"}, {"doc_id": "d2"}],
        )
        a.merge(b)
        assert len(a.chunks) == 3
        assert len(a.doc_aggs) == 2


class TestFormatDocs:
    def test_empty(self):
        assert _format_docs([]) == "<no content>"

    def test_single_chunk(self):
        result = _format_docs([{"content": "hello world"}])
        assert "[Chunk 0]" in result
        assert "hello world" in result

    def test_truncates_long_content(self):
        long = "x" * 5000
        result = _format_docs([{"content": long}], max_chars=100)
        assert len(result) < 200


class TestExtractJson:
    def test_extract_object(self):
        text = 'Here is the result:\n```json\n{"key":"value"}\n```'
        assert _extract_json(text) == '{"key":"value"}'

    def test_extract_array(self):
        text = '```\n[1,2,3]\n```'
        assert _extract_json(text) == "[1,2,3]"

    def test_no_json(self):
        assert _extract_json("no json here") is None

    def test_nested_object(self):
        text = '{"outer":{"inner":"value"}}'
        assert _extract_json(text) == text


class TestTreeQueryOrchestratorUnit:
    """测试单步逻辑（不递归）"""

    @pytest.mark.asyncio
    async def test_no_retrieve_fn_logs_failure(self):
        orch = TreeQueryOrchestrator(
            llm_call=None,  # no LLM
            retrieve_fn=None,  # no retrieval
            max_depth=1,
        )
        result = await orch.research("test question")
        assert result.chunks == []
        assert "检索函数未配置" in orch.get_search_log()[-1]

    @pytest.mark.asyncio
    async def test_no_llm_skips_sufficiency(self):
        """没有 LLM 时直接返回检索结果，不检查充分性"""
        mock_retrieve = AsyncMock(return_value={
            "chunks": [{"chunk_id": "c1", "content": "some content"}],
            "doc_aggs": [{"doc_id": "d1"}],
        })
        orch = TreeQueryOrchestrator(
            llm_call=None,
            retrieve_fn=mock_retrieve,
            max_depth=1,
        )
        result = await orch.research("test")
        assert len(result.chunks) == 1
        assert mock_retrieve.await_count == 1

    @pytest.mark.asyncio
    async def test_sufficient_early_return(self):
        """充分性判断返回 sufficient 时不再递归"""
        mock_retrieve = AsyncMock(return_value={
            "chunks": [{"chunk_id": "c1", "content": "complete answer"}],
            "doc_aggs": [],
        })
        mock_llm = AsyncMock(return_value='```\n{"is_sufficient":true,"reasoning":"enough info","missing_information":[]}\n```')
        orch = TreeQueryOrchestrator(
            llm_call=mock_llm,
            retrieve_fn=mock_retrieve,
            max_depth=3,
        )
        result = await orch.research("test")
        assert result.is_sufficient is True
        assert mock_llm.await_count == 1
        assert mock_retrieve.await_count == 1

    @pytest.mark.asyncio
    async def test_insufficient_generates_subqueries(self):
        """不充分时生成子 queries, 递归到深度 0"""
        mock_retrieve = AsyncMock(return_value={
            "chunks": [{"chunk_id": "c1", "content": "partial"}],
            "doc_aggs": [],
        })
        mock_llm = AsyncMock(side_effect=[
            '{"is_sufficient":false,"reasoning":"not enough","missing_information":["details"]}',
            '{"reasoning":"ok","questions":[{"question":"sub q?","query":"sub"}]}',
        ])
        orch = TreeQueryOrchestrator(
            llm_call=mock_llm,
            retrieve_fn=mock_retrieve,
            max_depth=1,
        )
        result = await orch.research("main question")
        # depth=1: main retrieve -> insufficient -> gen sub-queries -> depth=0 stop
        assert mock_retrieve.call_count >= 1

    @pytest.mark.asyncio
    async def test_update_chunk_info_dedup(self):
        """合并 chunk 时去重"""
        orch = TreeQueryOrchestrator(max_depth=1)
        await orch._update_chunk_info({"chunks": [{"chunk_id": "c1"}, {"chunk_id": "c2"}], "doc_aggs": []})
        await orch._update_chunk_info({"chunks": [{"chunk_id": "c2"}, {"chunk_id": "c3"}], "doc_aggs": []})
        assert len(orch._all_chunks) == 3

    @pytest.mark.asyncio
    async def test_callback_receives_progress(self):
        callback_msgs = []
        async def callback(msg):
            callback_msgs.append(msg)

        mock_retrieve = AsyncMock(return_value={
            "chunks": [{"chunk_id": "c1", "content": "answer"}],
            "doc_aggs": [],
        })
        mock_llm = AsyncMock(return_value='{"is_sufficient":true,"reasoning":"done","missing_information":[]}')

        orch = TreeQueryOrchestrator(
            llm_call=mock_llm,
            retrieve_fn=mock_retrieve,
            max_depth=1,
        )
        await orch.research("test", callback=callback)
        assert len(callback_msgs) > 0
        assert any("检索" in m for m in callback_msgs)

    @pytest.mark.asyncio
    async def test_retrieve_error_doesnt_crash(self):
        mock_retrieve = AsyncMock(side_effect=RuntimeError("connection refused"))
        orch = TreeQueryOrchestrator(
            llm_call=None,
            retrieve_fn=mock_retrieve,
            max_depth=1,
        )
        result = await orch.research("test")
        assert result.chunks == []

    @pytest.mark.asyncio
    async def test_sufficiency_parse_failure(self):
        mock_llm = AsyncMock(return_value="not json at all")
        mock_retrieve = AsyncMock(return_value={
            "chunks": [], "doc_aggs": []
        })
        orch = TreeQueryOrchestrator(
            llm_call=mock_llm,
            retrieve_fn=mock_retrieve,
            max_depth=1,
        )
        # Parse失败 -> 认为不充分 -> 生成子查询也失败 -> 返回
        result = await orch.research("test")
        assert result.is_sufficient is False

    @pytest.mark.asyncio
    async def test_max_iterations_limit(self):
        """子查询数量不超过 max_iterations"""
        mock_llm = AsyncMock(return_value='{"reasoning":"","questions":[{"question":"q1","query":"q1"},{"question":"q2","query":"q2"},{"question":"q3","query":"q3"},{"question":"q4","query":"q4"},{"question":"q5","query":"q5"}]}')
        orch = TreeQueryOrchestrator(
            llm_call=mock_llm,
            retrieve_fn=None,
            max_depth=1,
            max_iterations=3,
        )
        # _gen_multi_queries should cap at max_iterations
        queries = await orch._gen_multi_queries("q", "q", ["x"], {"chunks": []})
        assert len(queries) == 3

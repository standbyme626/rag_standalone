"""可观测性测试 — Trace ID middleware + Query analyzer + Langfuse。"""

import pytest
from unittest.mock import MagicMock, patch


# ===================================================================
# TraceIDMiddleware
# ===================================================================

class TestTraceIDMiddleware:
    def test_generates_trace_id(self):
        from app.core.middleware.traceid import TraceIDMiddleware
        from starlette.testclient import TestClient
        from starlette.applications import Starlette
        from starlette.responses import PlainTextResponse
        from starlette.routing import Route

        async def endpoint(request):
            return PlainTextResponse("ok")

        app = Starlette(routes=[Route("/", endpoint)])
        app.add_middleware(TraceIDMiddleware)

        client = TestClient(app)
        resp = client.get("/")
        assert "x-trace-id" in resp.headers
        trace_id = resp.headers["x-trace-id"]
        assert trace_id  # UUID format

    def test_reuses_client_trace_id(self):
        from app.core.middleware.traceid import TraceIDMiddleware
        from starlette.testclient import TestClient
        from starlette.applications import Starlette
        from starlette.responses import PlainTextResponse
        from starlette.routing import Route

        async def endpoint(request):
            return PlainTextResponse("ok")

        app = Starlette(routes=[Route("/", endpoint)])
        app.add_middleware(TraceIDMiddleware)

        client = TestClient(app)
        resp = client.get("/", headers={"x-trace-id": "my-trace-123"})
        assert resp.headers["x-trace-id"] == "my-trace-123"


# ===================================================================
# QueryAnalyzer
# ===================================================================

class TestQueryAnalyzerNoRedis:
    """QueryAnalyzer without Redis — should still record Prometheus metrics."""

    def test_record_without_redis(self):
        from app.core.monitoring.query_analyzer import QueryAnalyzer
        analyzer = QueryAnalyzer(redis_client=None)
        analyzer.record(
            query="感冒吃什么药",
            domain="medical",
            intent="retrieval",
            latency_s=0.5,
            num_results=3,
            trace_id="test-trace",
        )
        # No exception should indicate success (Prometheus metrics are thread-safe)

    def test_zero_result_increments_counter(self):
        from app.core.monitoring.query_analyzer import QueryAnalyzer, RAG_ZERO_RESULT
        analyzer = QueryAnalyzer(redis_client=None)
        before = RAG_ZERO_RESULT.labels(domain="test")._value.get()
        analyzer.record(
            query="rare disease",
            domain="test",
            intent="retrieval",
            latency_s=0.1,
            num_results=0,
        )
        after = RAG_ZERO_RESULT.labels(domain="test")._value.get()
        assert after == before + 1

    def test_get_methods_return_empty_without_redis(self):
        from app.core.monitoring.query_analyzer import QueryAnalyzer
        analyzer = QueryAnalyzer(redis_client=None)
        assert analyzer.get_top_queries() == []
        assert analyzer.get_zero_result_queries() == []
        assert analyzer.get_recent_queries() == []


class TestQueryAnalyzerWithRedis:
    def test_record_with_redis(self):
        from app.core.monitoring.query_analyzer import QueryAnalyzer
        fake_redis = MagicMock()
        analyzer = QueryAnalyzer(redis_client=fake_redis)
        analyzer.record(
            query="test query",
            domain="medical",
            intent="retrieval",
            latency_s=0.3,
            num_results=2,
            trace_id="trace-1",
        )
        # Redis pipeline should have been called
        assert fake_redis.pipeline.called


# ===================================================================
# Langfuse setup
# ===================================================================

class TestLangfuseSetup:
    def test_flush_without_init(self):
        """flush_langfuse should not error even with no SDK."""
        from app.core.monitoring.langfuse_setup import flush_langfuse
        flush_langfuse()  # should not raise

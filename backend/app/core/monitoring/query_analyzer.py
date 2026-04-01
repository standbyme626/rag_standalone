"""查询分析器 — 记录查询指标、维护 Redis 统计。"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING

import structlog

from prometheus_client import Counter, Histogram

if TYPE_CHECKING:
    pass

logger = structlog.get_logger(__name__)

# ── Prometheus RAG query metrics ──────────────────────────────

RAG_QUERY_TOTAL = Counter(
    "rag_query_total",
    "Total RAG queries",
    ["domain", "intent", "has_results"],
)

RAG_QUERY_LATENCY = Histogram(
    "rag_query_latency_seconds",
    "Latency of end-to-end RAG query",
    ["domain", "intent"],
)

RAG_ZERO_RESULT = Counter(
    "rag_zero_result_total",
    "RAG queries that returned zero results",
    ["domain"],
)

# ── Query analyzer ─────────────────────────────────────────────


class QueryAnalyzer:
    """记录每次 RAG 查询的统计信息。

    无 Redis 时仅记录 Prometheus 指标 + structlog。
    """

    def __init__(self, redis_client=None):
        self.redis_client = redis_client

    # ── 记录一次查询 ─────────

    def record(
        self,
        query: str,
        domain: str = "medical",
        intent: str = "retrieval",
        latency_s: float = 0.0,
        num_results: int = 0,
        trace_id: str = "",
    ) -> None:
        # 1. Prometheus 指标
        has_results = "yes" if num_results > 0 else "no"
        RAG_QUERY_TOTAL.labels(
            domain=domain, intent=intent, has_results=has_results
        ).inc()
        RAG_QUERY_LATENCY.labels(domain=domain, intent=intent).observe(latency_s)
        if num_results == 0:
            RAG_ZERO_RESULT.labels(domain=domain).inc()

        # 2. 结构化日志
        logger.info(
            "rag_query_recorded",
            query=query[:200],
            domain=domain,
            intent=intent,
            latency_s=round(latency_s, 3),
            num_results=num_results,
            trace_id=trace_id,
        )

        # 3. Redis-backed stats (可选)
        if self.redis_client is None:
            return

        try:
            # Recent queries — 滑动窗口 1000 条
            recent_key = f"rag:recent_queries:{domain}"
            entry = json.dumps({
                "query": query,
                "intent": intent,
                "results": num_results,
                "ts": time.time(),
                "trace_id": trace_id,
            })
            pipe = self.redis_client.pipeline()
            pipe.lpush(recent_key, entry)
            pipe.ltrim(recent_key, 0, 999)
            pipe.expire(recent_key, 86400 * 7)  # 7 day TTL

            # Top queries — 精确匹配计数
            if num_results > 0:
                top_key = f"rag:top_queries:{domain}"
                pipe.incrby(f"{top_key}:{query[:100]}", 1)
                pipe.expire(f"{top_key}:{query[:100]}", 86400 * 7)

            # Zero-result queries for gap analysis
            if num_results == 0:
                zero_key = f"rag:zero_results:{domain}"
                pipe.lpush(zero_key, entry)
                pipe.ltrim(zero_key, 0, 499)
                pipe.expire(zero_key, 86400 * 7)

            pipe.execute()
        except Exception as e:
            logger.warning("query_analyzer_redis_failed", error=str(e))

    # ── 读方法（供 API 端点使用）─────────

    def get_top_queries(self, domain: str = "medical", limit: int = 10) -> list[dict]:
        """获取热门查询。需 Redis 支持。"""
        if self.redis_client is None:
            return []
        try:
            top_key = f"rag:top_queries:{domain}"
            keys = self.redis_client.keys(f"{top_key}:*")
            if not keys:
                return []
            values = {}
            for k in keys:
                count = int(self.redis_client.get(k) or 0)
                query = k.decode().replace(f"{top_key}:", "", 1)
                values[query] = count
            sorted_items = sorted(values.items(), key=lambda x: x[1], reverse=True)
            return [{"query": q, "count": c} for q, c in sorted_items[:limit]]
        except Exception as e:
            logger.warning("get_top_queries_failed", error=str(e))
            return []

    def get_zero_result_queries(
        self, domain: str = "medical", limit: int = 20
    ) -> list[dict]:
        """获取无结果查询列表。需 Redis 支持。"""
        if self.redis_client is None:
            return []
        try:
            zero_key = f"rag:zero_results:{domain}"
            entries = self.redis_client.lrange(zero_key, 0, limit - 1)
            return [json.loads(e) for e in entries]
        except Exception as e:
            logger.warning("get_zero_result_queries_failed", error=str(e))
            return []

    def get_recent_queries(
        self, domain: str = "medical", limit: int = 20
    ) -> list[dict]:
        """获取最近查询。需 Redis 支持。"""
        if self.redis_client is None:
            return []
        try:
            recent_key = f"rag:recent_queries:{domain}"
            entries = self.redis_client.lrange(recent_key, 0, limit - 1)
            return [json.loads(e) for e in entries]
        except Exception as e:
            logger.warning("get_recent_queries_failed", error=str(e))
            return []


# 全局单例 (懒初始化：Redis 在首次使用时尝试获取)
query_analyzer = QueryAnalyzer()

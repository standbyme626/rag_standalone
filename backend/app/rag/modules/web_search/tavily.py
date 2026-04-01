"""Tavily Web Search 后端 — 来源：UltraRAG servers/retriever/src/websearch_backends/tavily_backend.py"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import structlog

from app.rag.modules.web_search.base import BaseWebSearchBackend

logger = structlog.get_logger(__name__)


class TavilyWebSearchBackend(BaseWebSearchBackend):
    """基于 Tavily API 的网络搜索引擎"""

    backend_name: str = "tavily"

    def __init__(
        self,
        api_key: Optional[str] = None,
        max_results: int = 5,
        search_depth: str = "basic",
    ):
        try:
            from tavily import TavilyClient
        except ImportError:
            raise ImportError(
                "tavily-python is not installed. Please install with: pip install tavily-python"
            )

        import os

        key = api_key or os.environ.get("TAVILY_API_KEY", "")
        if not key:
            raise ValueError(
                "TAVILY_API_KEY is not set. "
                "Please set TAVILY_API_KEY env var or pass api_key=..."
            )

        self._client = TavilyClient(api_key=key)
        self.max_results = max_results
        self.search_depth = search_depth

    async def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """使用 Tavily 执行网络搜索"""
        k = min(top_k, self.max_results) if self.max_results else top_k

        try:
            response = await self._client.search(
                query=query,
                max_results=k,
                search_depth=self.search_depth,
            )
            results: List[Dict[str, Any]] = response.get("results", []) or []
            return [
                {
                    "title": r.get("title", ""),
                    "content": r.get("content", ""),
                    "url": r.get("url", ""),
                    "score": r.get("score", 0.0),
                    "source": "tavily",
                }
                for r in results[:k]
            ]
        except Exception as e:
            logger.error("tavily_search_failed", error=str(e))
            return []

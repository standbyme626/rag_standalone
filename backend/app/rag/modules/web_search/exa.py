"""Exa Web Search 后端 — 来源：UltraRAG servers/retriever/src/websearch_backends/exa_backend.py"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import structlog

from app.rag.modules.web_search.base import BaseWebSearchBackend

logger = structlog.get_logger(__name__)


class ExaWebSearchBackend(BaseWebSearchBackend):
    """基于 Exa API 的网络搜索引擎"""

    backend_name: str = "exa"

    def __init__(
        self,
        api_key: Optional[str] = None,
        num_results: int = 5,
        use_highlights: bool = False,
    ):
        try:
            from exa_py import AsyncExa
        except ImportError:
            raise ImportError(
                "exa_py is not installed. Please install with: pip install exa_py"
            )

        import os

        key = api_key or os.environ.get("EXA_API_KEY", "")
        if not key:
            raise ValueError(
                "EXA_API_KEY is not set. "
                "Please set EXA_API_KEY env var or pass api_key=..."
            )

        self._client = AsyncExa(api_key=key)
        self.num_results = num_results
        self.use_highlights = use_highlights

    async def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """使用 Exa 执行网络搜索"""
        k = min(top_k, self.num_results) if self.num_results else top_k

        try:
            params: Dict[str, Any] = {
                "text": True,
                "num_results": k,
            }
            if self.use_highlights:
                params["highlights"] = True

            response = await self._client.search_and_contents(query, **params)
            results = getattr(response, "results", []) or []
            return [
                {
                    "title": getattr(r, "title", ""),
                    "content": getattr(r, "text", ""),
                    "url": getattr(r, "url", ""),
                    "score": getattr(r, "score", 0.0),
                    "source": "exa",
                }
                for r in results[:k]
            ]
        except Exception as e:
            logger.error("exa_search_failed", error=str(e))
            if "401" in str(e):
                logger.error("exa_api_key_invalid")
            return []

"""Web Search 统一多后端接口 — 来源：UltraRAG websearch_backends/

支持的 backend:
  - tavily: Tavily 搜索（默认）
  - exa: Exa 搜索

用法:
    from app.rag.modules.web_search import create_web_search_backend
    backend = create_web_search_backend(backend="tavily")
    results = await backend.search("query", top_k=5)
"""

from __future__ import annotations

from typing import Optional

import structlog

from app.rag.modules.web_search.base import BaseWebSearchBackend

logger = structlog.get_logger(__name__)

__all__ = ["create_web_search_backend", "BaseWebSearchBackend"]


def create_web_search_backend(
    backend: Optional[str] = None,
    tavily_api_key: Optional[str] = None,
    exa_api_key: Optional[str] = None,
    top_k: int = 5,
    **kwargs,
) -> Optional[BaseWebSearchBackend]:
    """根据配置创建 Web Search 后端实例

    Args:
        backend: 后端类型（None 时取 "tavily"）
        tavily_api_key: Tavily API key
        exa_api_key: Exa API key
        top_k: 默认返回结果数量

    Returns:
        BaseWebSearchBackend 实例或 None（当未配置或不可用时）
    """
    backend_name = (backend or "tavily").lower().strip()

    if backend_name == "tavily":
        try:
            from app.rag.modules.web_search.tavily import TavilyWebSearchBackend

            return TavilyWebSearchBackend(
                api_key=tavily_api_key,
                max_results=top_k,
            )
        except (ImportError, ValueError) as e:
            logger.error("tavily_backend_unavailable", error=str(e))
            return None

    if backend_name == "exa":
        try:
            from app.rag.modules.web_search.exa import ExaWebSearchBackend

            return ExaWebSearchBackend(
                api_key=exa_api_key,
                num_results=top_k,
            )
        except (ImportError, ValueError) as e:
            logger.error("exa_backend_unavailable", error=str(e))
            return None

    logger.error("unknown_web_search_backend", backend=backend_name)
    return None

"""Langfuse SDK 集成。"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    pass

logger = structlog.get_logger(__name__)

_langfuse = None


def init_langfuse(
    public_key: str,
    secret_key: str,
    host: str = "https://cloud.langfuse.com",
    enabled: bool = True,
):
    """初始化 Langfuse 客户端。若 SDK 未安装则静默降级。"""
    global _langfuse
    if not enabled:
        logger.info("langfuse_disabled_by_config")
        _langfuse = None
        return

    try:
        from langfuse import Langfuse

        _langfuse = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
        )
        logger.info("langfuse_initialized")
    except ImportError:
        logger.warning("langfuse_sdk_not_installed")
        _langfuse = None
    except Exception as e:
        logger.warning("langfuse_init_failed", error=str(e))
        _langfuse = None


def get_langfuse():
    """获取 Langfuse 客户端实例。"""
    return _langfuse


@contextmanager
def trace_rag_query(
    query: str,
    domain: str = "medical",
    intent: str = "retrieval",
    metadata: dict[str, Any] | None = None,
):
    """Langfuse trace context manager，用于追踪一次 RAG 查询。

    with trace_rag_query(query, domain="medical") as trace:
        # ...retrieval steps...
        trace.update(output=final_answer)
    """
    if _langfuse is None:
        yield None
        return

    trace_name = f"rag_query_{domain}"
    session_id = metadata.get("session_id") if metadata else None
    user_id = metadata.get("user_id") if metadata else None

    trace = _langfuse.trace(
        name=trace_name,
        input={"query": query, "domain": domain, "intent": intent},
        session_id=session_id,
        user_id=user_id,
        metadata=metadata or {},
    )

    try:
        span = trace.span(name="retrieval_pipeline")
        yield trace
        span.end()
        trace.update(output={"status": "completed"})
    except Exception as e:
        trace.update(output={"status": "error", "error": str(e)})
        raise
    finally:
        _langfuse.flush()


def flush_langfuse():
    """强制刷新 Langfuse 事件队列。"""
    if _langfuse is not None:
        try:
            _langfuse.flush()
        except Exception as e:
            logger.warning("langfuse_flush_failed", error=str(e))

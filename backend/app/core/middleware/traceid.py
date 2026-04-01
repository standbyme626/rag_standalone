"""Trace ID 中间件 — 请求级 trace ID 贯穿全链路。"""

from __future__ import annotations

import uuid

import structlog
import structlog.contextvars
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.types import ASGIApp


def get_trace_id() -> str | None:
    """获取当前请求的 trace ID，若在请求上下文中则返回。"""
    return structlog.contextvars.get_contextvars().get("trace_id")


class TraceIDMiddleware(BaseHTTPMiddleware):
    """为每个请求生成/传递 x-trace-id，注入 structlog 上下文。"""

    def __init__(self, app: ASGIApp, header_name: str = "x-trace-id"):
        super().__init__(app)
        self.header_name = header_name

    async def dispatch(self, request: Request, call_next):
        # 优先使用客户端传入的 trace_id，否则生成 UUIDv4
        trace_id = request.headers.get(self.header_name) or str(uuid.uuid4())

        with structlog.contextvars.bound_contextvars(trace_id=trace_id):
            response = await call_next(request)
            response.headers[self.header_name] = trace_id
            return response

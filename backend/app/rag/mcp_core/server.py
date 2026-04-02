"""
MCPServer - abstract MCP server with tool registration and invocation.

Usage:
    server = MCPServer(name="retriever")

    @server.tool("bm25_search", "BM25 full-text search")
    def bm25_search(query: str, k: int = 10) -> dict:
        return {"results": []}

    server.call_tool("bm25_search", query="headache", k=5)
    server.list_tools()
    server.shutdown()
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import structlog

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class MCPToolError(Exception):
    """Raised when a tool invocation fails."""

    pass


# ---------------------------------------------------------------------------
# Tool metadata
# ---------------------------------------------------------------------------


@dataclass
class MCPToolSpec:
    """Specification for a registered MCP tool."""

    name: str
    fn: Callable
    description: str = ""
    input_schema: Dict[str, Any] = field(default_factory=dict)
    server: str = ""


# ---------------------------------------------------------------------------
# MCPServer
# ---------------------------------------------------------------------------


class MCPServer:
    """Abstract MCP server that registers tools and exposes a call interface.

    Each server has a name.  Tools registered on a server are namespaced under
    that name when exposed via :class:`ToolCall`.

    Attributes:
        name: Server identifier (e.g. "retriever", "reranker").
        tools: Dict of tool name -> MCPToolSpec.
    """

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self._tools: Dict[str, MCPToolSpec] = {}
        self._shutdown_hooks: List[Callable[[], None]] = []

    # -- registration --

    def tool(
        self,
        name: str,
        description: str = "",
        input_schema: Optional[Dict[str, Any]] = None,
    ) -> Callable:
        """Decorator to register a function as a tool on this server.

        Args:
            name: Tool identifier, unique within this server.
            description: Human-readable description.
            input_schema: Optional dict describing expected input keys and types.
                         If omitted, inferred from the function signature.

        Returns:
            Decorator function that registers the decorated callable.
        """

        def decorator(fn: Callable) -> Callable:
            schema = input_schema or self._infer_input_schema(fn)
            spec = MCPToolSpec(
                name=name,
                fn=fn,
                description=description,
                input_schema=schema,
                server=self.name,
            )
            self._tools[name] = spec
            logger.debug(
                "mcp_tool_registered",
                server=self.name,
                tool=name,
                has_schema=bool(schema),
            )
            return fn

        return decorator

    def register(
        self,
        name: str,
        fn: Callable,
        description: str = "",
        input_schema: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register a callable imperatively (non-decorator form)."""
        schema = input_schema or self._infer_input_schema(fn)
        spec = MCPToolSpec(
            name=name,
            fn=fn,
            description=description,
            input_schema=schema,
            server=self.name,
        )
        self._tools[name] = spec

    def deregister(self, name: str) -> None:
        """Remove a tool by name. No-op if not found."""
        self._tools.pop(name, None)

    # -- invocation --

    def call_tool(self, name: str, **kwargs: Any) -> Any:
        """Invoke a registered tool by name with keyword arguments.

        Args:
            name: Tool name (without server prefix).
            **kwargs: Arguments forwarded to the underlying callable.

        Returns:
            Whatever the underlying callable returns.

        Raises:
            MCPToolError: If the tool is not found or execution fails.
        """
        if name not in self._tools:
            available = ", ".join(sorted(self._tools)) or "(none)"
            raise MCPToolError(
                f"Tool '{name}' not found on server '{self.name}'. "
                f"Available: {available}"
            )
        spec = self._tools[name]
        try:
            logger.debug(
                "mcp_tool_call",
                server=self.name,
                tool=name,
                keys=list(kwargs.keys()),
            )
            return spec.fn(**kwargs)
        except TypeError as exc:
            raise MCPToolError(
                f"Tool '{self.name}.{name}' argument error: {exc}"
            ) from exc
        except Exception as exc:
            raise MCPToolError(
                f"Tool '{self.name}.{name}' execution failed: {exc}"
            ) from exc

    def get_tool_fn(self, name: str) -> Callable:
        """Return the raw callable for a tool (bypasses error handling)."""
        if name not in self._tools:
            raise KeyError(
                f"Tool '{name}' not found on server '{self.name}'"
            )
        return self._tools[name].fn

    # -- introspection --

    def list_tools(self) -> List[Dict[str, Any]]:
        """Return tool metadata as a list of dicts.

        Each dict contains: name, description, input_schema, server.
        """
        return [
            {
                "name": spec.name,
                "description": spec.description,
                "input_schema": spec.input_schema,
                "server": spec.server,
            }
            for spec in self._tools.values()
        ]

    def has_tool(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tools

    @property
    def tool_names(self) -> List[str]:
        """List all registered tool names."""
        return sorted(self._tools.keys())

    # -- lifecycle --

    def on_shutdown(self, fn: Callable[[], None]) -> None:
        """Register a callback to be called on shutdown."""
        self._shutdown_hooks.append(fn)

    def shutdown(self) -> None:
        """Run all shutdown hooks and clear tools."""
        for hook in self._shutdown_hooks:
            try:
                hook()
            except Exception:
                logger.exception("mcp_shutdown_hook_failed", server=self.name)
        self._shutdown_hooks.clear()
        self._tools.clear()
        logger.debug("mcp_server_shutdown", server=self.name)

    def __repr__(self) -> str:
        return f"MCPServer(name='{self.name}', tools={self.tool_names})"

    # -- internals --

    @staticmethod
    def _infer_input_schema(fn: Callable) -> Dict[str, Any]:
        """Infer a simple input schema from function signature."""
        sig = inspect.signature(fn)
        schema: Dict[str, Any] = {}
        for name, param in sig.parameters.items():
            if param.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue
            type_str = ""
            if param.annotation is not param.empty:
                type_str = getattr(param.annotation, "__name__", str(param.annotation))
            schema[name] = {
                "source": name,
                "default": (
                    param.default if param.default is not param.empty else None
                ),
                "type": type_str,
            }
        return schema

"""
ToolCall - unified router that maps "server.tool_name" strings to callables.

Usage:
    router = ToolCall()
    router.register_server(retriever_server)
    router.register_server(reranker_server)

    # Call style: "retriever.bm25_search" with args
    result = router.call("retriever.bm25_search", query="headache", k=5)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import structlog

from app.rag.mcp_core.server import MCPServer, MCPToolError

logger = structlog.get_logger(__name__)


class ToolCallError(Exception):
    """Raised when a tool call routing or execution error occurs."""

    pass


class ToolCall:
    """Unified router for dispatching ``server.tool_name(arg)``, style calls.

    Holds references to multiple :class:`MCPServer` instances and resolves
    ``"server_name.tool_name"`` strings to the right callable.
    """

    SEPARATOR = "."

    def __init__(self, default_server: Optional[MCPServer] = None):
        self._servers: Dict[str, MCPServer] = {}
        self._default: Optional[MCPServer] = default_server
        if default_server:
            self._servers[default_server.name] = default_server

    # -- server management --

    def register_server(self, server: MCPServer) -> None:
        """Register an MCPServer instance.

        After registration, all tools on this server become callable via
        ``"<server.name>.<tool_name>"``.
        """
        self._servers[server.name] = server
        logger.debug(
            "toolcall_server_registered",
            server=server.name,
            tool_count=len(server.tool_names),
        )

    def remove_server(self, server_name: str) -> None:
        """Remove a registered server. No-op if not found."""
        self._servers.pop(server_name, None)
        logger.debug("toolcall_server_removed", server=server_name)

    def list_servers(self) -> List[str]:
        """Return list of registered server names."""
        return sorted(self._servers.keys())

    def get_server(self, server_name: str) -> Optional[MCPServer]:
        """Get an MCPServer instance by name."""
        return self._servers.get(server_name)

    # -- all tools listing across servers --

    def list_all_tools(self) -> List[Dict[str, Any]]:
        """Return tool metadata from all registered servers."""
        result: List[Dict[str, Any]] = []
        for server in self._servers.values():
            result.extend(server.list_tools())
        return result

    # -- calling --

    def call(self, tool_ref: str, **kwargs: Any) -> Any:
        """Call a tool identified by ``"server.tool_name"`` string.

        Args:
            tool_ref: Dot-separated string in the form ``"server_name.tool_name"``.
                     If no dot is present and a default server is set, the tool
                     name is resolved on the default server.
            **kwargs: Arguments forwarded to the tool function.

        Returns:
            Tool execution result.

        Raises:
            ToolCallError: If the tool reference is malformed, the server or
                          tool is not found, or execution fails.
        """
        server_name, tool_name = self._parse_ref(tool_ref)
        server = self._servers.get(server_name)
        if server is None:
            available = ", ".join(sorted(self._servers)) or "(none)"
            raise ToolCallError(
                f"Server '{server_name}' not found. Available servers: {available}"
            )
        try:
            return server.call_tool(tool_name, **kwargs)
        except MCPToolError as exc:
            raise ToolCallError(str(exc)) from exc

    def call_direct(self, server_name: str, tool_name: str, **kwargs: Any) -> Any:
        """Call a tool with explicit server/tool separation (no string parsing)."""
        server = self._servers.get(server_name)
        if server is None:
            raise ToolCallError(f"Server '{server_name}' not found")
        return server.call_tool(tool_name, **kwargs)

    def has_tool(self, tool_ref: str) -> bool:
        """Check if a tool reference is resolvable."""
        try:
            server_name, tool_name = self._parse_ref(tool_ref)
        except ToolCallError:
            return False
        server = self._servers.get(server_name)
        if server is None:
            return False
        return server.has_tool(tool_name)

    # -- parsing --

    def _parse_ref(self, tool_ref: str) -> tuple:
        """Parse a ``"server.tool"`` reference into (server, tool)."""
        if self.SEPARATOR in tool_ref:
            parts = tool_ref.split(self.SEPARATOR, 1)
            return parts[0], parts[1]
        # Fallback to default server
        if self._default is not None:
            return self._default.name, tool_ref
        raise ToolCallError(
            f"Malformed tool reference '{tool_ref}'. "
            f"Expected '<server>.<tool>' format."
        )

    def __repr__(self) -> str:
        return (
            f"ToolCall(servers={self.list_servers()}, "
            f"default='{self._default.name if self._default else None}')"
        )

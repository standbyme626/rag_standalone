"""
MCP Core - Model Context Protocol server ecology for ArchRAG.

Provides:
- MCPServer: abstract server with tool registration and invocation.
- ToolCall: unified router that maps "server.tool_name(args)" strings to callables.
- PromptSandbox: safe Jinja2 template rendering with XSS protection.
- ServerConfig: YAML-based server configuration with hot-reload support.
"""

from app.rag.mcp_core.server import MCPServer, MCPToolError, MCPToolSpec
from app.rag.mcp_core.tool_call import ToolCall, ToolCallError
from app.rag.mcp_core.prompt_sandbox import PromptSandbox, SandboxError
from app.rag.mcp_core.config import ServerConfig, ConfigReloadResult

__all__ = [
    "MCPServer",
    "MCPToolError",
    "MCPToolSpec",
    "ToolCall",
    "ToolCallError",
    "PromptSandbox",
    "SandboxError",
    "ServerConfig",
    "ConfigReloadResult",
]

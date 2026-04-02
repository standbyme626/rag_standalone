"""
Tests for mcp_core module: MCPServer, ToolCall, PromptSandbox, ServerConfig.

Tests cover:
  1. MCPServer tool registration, invocation, introspection, shutdown.
  2. ToolCall parsing, routing, error handling, multi-server dispatch.
  3. PromptSandbox XSS protection, unsafe pattern blocking, rendering.
  4. ServerConfig load/save, dotted-path access, hot-reload detection.
"""

from __future__ import annotations

import sys
from pathlib import Path
from textwrap import dedent
import time

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def mock_settings_env(monkeypatch):
    defaults = {
        "OPENAI_MODEL_NAME": "gpt-3.5-turbo",
        "MILVUS_HOST": "localhost",
        "MILVUS_PORT": "19530",
        "REDIS_HOST": "localhost",
        "REDIS_PORT": "6379",
        "DATABASE_URL": "sqlite:///./test.db",
    }
    for k, v in defaults.items():
        monkeypatch.setenv(k, v)


from app.rag.mcp_core.server import MCPServer, MCPToolError, MCPToolSpec
from app.rag.mcp_core.tool_call import ToolCall, ToolCallError
from app.rag.mcp_core.prompt_sandbox import PromptSandbox, SandboxError
from app.rag.mcp_core.config import ServerConfig, ConfigReloadResult


# ===================================================================
# 1. MCPServer tests
# ===================================================================

class TestMCPServer:

    def test_register_decorator(self):
        server = MCPServer(name="test")

        @server.tool("hello")
        def hello(name: str) -> str:
            return f"hello {name}"

        assert server.has_tool("hello")
        assert "hello" in server.tool_names

    def test_register_decorator_with_description(self):
        server = MCPServer(name="test")

        @server.tool("search", "BM25 search")
        def search(query: str) -> list:
            return []

        tools = server.list_tools()
        assert any(t["name"] == "search" and t["description"] == "BM25 search" for t in tools)

    def test_register_imperative(self):
        server = MCPServer(name="test")
        def add(a: int, b: int) -> int:
            return a + b
        server.register("add", add, description="Add two numbers")
        assert server.has_tool("add")

    def test_call_tool_success(self):
        server = MCPServer(name="calc")

        @server.tool("add")
        def add(a: int, b: int) -> int:
            return a + b

        result = server.call_tool("add", a=3, b=4)
        assert result == 7

    def test_call_tool_missing(self):
        server = MCPServer(name="test")
        with pytest.raises(MCPToolError, match="not found"):
            server.call_tool("nonexistent")

    def test_call_tool_argument_error(self):
        server = MCPServer(name="test")

        @server.tool("greet")
        def greet(name: str) -> str:
            return f"hi {name}"

        with pytest.raises(MCPToolError, match="argument error"):
            server.call_tool("greet", wrong_arg=1)

    def test_call_tool_execution_error(self):
        server = MCPServer(name="test")

        @server.tool("boom")
        def boom():
            raise ValueError("kaboom")

        with pytest.raises(MCPToolError, match="kaboom"):
            server.call_tool("boom")

    def test_list_tools_metadata(self):
        server = MCPServer(name="retriever")

        @server.tool("bm25", "BM25 search")
        def bm25(query: str, k: int = 10):
            return []

        tools = server.list_tools()
        assert len(tools) == 1
        assert tools[0]["name"] == "bm25"
        assert tools[0]["server"] == "retriever"
        assert "input_schema" in tools[0]

    def test_deregister(self):
        server = MCPServer(name="test")

        @server.tool("temp")
        def temp():
            pass

        server.deregister("temp")
        assert not server.has_tool("temp")
        server.deregister("temp")  # idempotent

    def test_shutdown_clears_tools(self):
        server = MCPServer(name="test")

        @server.tool("x")
        def x():
            pass

        server.shutdown()
        assert not server.has_tool("x")

    def test_shutdown_hooks(self):
        calls = []
        server = MCPServer(name="test")
        server.on_shutdown(lambda: calls.append("a"))
        server.on_shutdown(lambda: calls.append("b"))
        server.shutdown()
        assert calls == ["a", "b"]

    def test_get_tool_fn(self):
        server = MCPServer(name="test")

        @server.tool("fn")
        def fn(x: int) -> int:
            return x * 2

        f = server.get_tool_fn("fn")
        assert f(5) == 10

    def test_get_tool_fn_missing(self):
        server = MCPServer(name="test")
        with pytest.raises(KeyError):
            server.get_tool_fn("nope")

    def test_repr(self):
        server = MCPServer(name="test")
        assert "test" in repr(server)

    def test_input_schema_inference(self):
        server = MCPServer(name="test")

        @server.tool("with_types")
        def with_types(query: str, k: int = 5) -> list:
            return []

        tools = server.list_tools()
        schema = tools[0]["input_schema"]
        assert "query" in schema
        assert "k" in schema
        assert schema["k"]["default"] == 5

    def test_server_name_on_tool_spec(self):
        server = MCPServer(name="my_server")

        @server.tool("t")
        def t():
            pass

        spec = server._tools["t"]
        assert spec.server == "my_server"


# ===================================================================
# 2. ToolCall tests
# ===================================================================

class TestToolCall:

    def test_register_server(self):
        server = MCPServer(name="retriever")

        @server.tool("search")
        def search(query: str) -> list:
            return [query]

        tc = ToolCall()
        tc.register_server(server)
        assert "retriever" in tc.list_servers()

    def test_call_with_dot_notation(self):
        server = MCPServer(name="retriever")

        @server.tool("search")
        def search(query: str) -> dict:
            return {"q": query}

        tc = ToolCall(default_server=server)
        result = tc.call("retriever.search", query="test")
        assert result == {"q": "test"}

    def test_call_default_server(self):
        server = MCPServer(name="default")

        @server.tool("ping")
        def ping() -> str:
            return "pong"

        tc = ToolCall(default_server=server)
        # No dot => resolves on default server
        result = tc.call("ping")
        assert result == "pong"

    def test_call_missing_server(self):
        tc = ToolCall()
        with pytest.raises(ToolCallError, match="not found"):
            tc.call("retriever.search")

    def test_call_missing_tool(self):
        server = MCPServer(name="retriever")
        tc = ToolCall(default_server=server)
        with pytest.raises(ToolCallError, match="not found"):
            tc.call("retriever.no_such_tool")

    def test_call_direct(self):
        server = MCPServer(name="reranker")

        @server.tool("rank")
        def rank(query: str, docs: list) -> list:
            return docs

        tc = ToolCall()
        tc.register_server(server)
        result = tc.call_direct("reranker", "rank", query="hi", docs=["a"])
        assert result == ["a"]

    def test_remove_server(self):
        server = MCPServer(name="x")
        tc = ToolCall()
        tc.register_server(server)
        tc.remove_server("x")
        assert "x" not in tc.list_servers()

    def test_list_all_tools(self):
        s1 = MCPServer(name="s1")
        s1.register("a", lambda: 1)
        s2 = MCPServer(name="s2")
        s2.register("b", lambda: 2)
        tc = ToolCall()
        tc.register_server(s1)
        tc.register_server(s2)
        tools = tc.list_all_tools()
        names = {t["name"] for t in tools}
        assert "a" in names
        assert "b" in names

    def test_has_tool(self):
        server = MCPServer(name="s")
        server.register("t", lambda: 1)
        tc = ToolCall(default_server=server)
        assert tc.has_tool("s.t")
        assert not tc.has_tool("s.missing")

    def test_ref_without_dot_no_default(self):
        tc = ToolCall()
        with pytest.raises(ToolCallError, match="Malformed"):
            tc.call("justname")

    def test_repr(self):
        tc = ToolCall()
        assert "ToolCall" in repr(tc)


# ===================================================================
# 3. PromptSandbox tests
# ===================================================================

class TestPromptSandbox:

    def test_basic_render(self):
        ps = PromptSandbox()
        result = ps.render("Hello {{ name }}", {"name": "World"})
        assert result == "Hello World"

    def test_xss_escaping(self):
        ps = PromptSandbox()
        result = ps.render("Hi {{ user }}", {"user": "<script>alert(1)</script>"})
        assert "<script>" not in result
        assert "&lt;script&gt;" in result

    def test_block_import(self):
        ps = PromptSandbox()
        with pytest.raises(SandboxError, match="Dangerous pattern"):
            ps.render("{% import 'os' as os %}")

    def test_block_eval(self):
        ps = PromptSandbox()
        with pytest.raises(SandboxError, match="Dangerous pattern"):
            ps.render("{{ eval('1+1') }}")

    def test_block_exec(self):
        ps = PromptSandbox()
        with pytest.raises(SandboxError, match="Dangerous pattern"):
            ps.render("{{ exec('print(1)') }}")

    def test_block_dunder_access(self):
        ps = PromptSandbox()
        with pytest.raises(SandboxError, match="Dangerous pattern"):
            ps.render("{{ __class__ }}")

    def test_undefined_variable_raises(self):
        ps = PromptSandbox()
        with pytest.raises(SandboxError, match="Undefined"):
            ps.render("Hello {{ missing }}")

    def test_safe_render_alias(self):
        ps = PromptSandbox()
        result = ps._safe_render("{{ x }}", {"x": 42})
        assert result == "42"

    def test_validate_template_ok(self):
        ps = PromptSandbox()
        assert ps.validate_template("Hello {{ name }}") is True

    def test_validate_template_syntax_error(self):
        ps = PromptSandbox()
        with pytest.raises(SandboxError, match="syntax"):
            ps.validate_template("Hello {{ name }")

    def test_dict_context_escaped(self):
        ps = PromptSandbox()
        ctx = {"data": {"html": "<b>bold</b>"}}
        # Nested dicts are escaped too
        assert ps.render("ok", ctx) == "ok"

    def test_nested_list_escaping(self):
        ps = PromptSandbox()
        ctx = {"items": ["<a>", "<b>"]}
        result = ps.render("{{ items }}", ctx)
        assert "<a>" not in result
        assert "&lt;a&gt;" in result

    def test_conditional_render(self):
        ps = PromptSandbox()
        tmpl = "{% if show %}YES{% else %}NO{% endif %}"
        assert ps.render(tmpl, {"show": True}) == "YES"
        assert ps.render(tmpl, {"show": False}) == "NO"

    def test_loop_render(self):
        ps = PromptSandbox()
        tmpl = "{% for i in items %}{{ i }}{% endfor %}"
        result = ps.render(tmpl, {"items": ["a", "b", "c"]})
        assert result == "abc"


# ===================================================================
# 4. ServerConfig tests
# ===================================================================

class TestServerConfig:

    def test_from_dict(self):
        config = ServerConfig.from_dict({"top_k": 10, "backend": "qwen"})
        assert config.get("top_k") == 10
        assert config.get("backend") == "qwen"

    def test_dotted_path_get(self):
        config = ServerConfig.from_dict({
            "retriever": {"top_k": 20, "backend": "bm25"},
        })
        assert config.get("retriever.top_k") == 20
        assert config.get("retriever.backend") == "bm25"

    def test_dotted_path_default(self):
        config = ServerConfig.from_dict({})
        assert config.get("missing", 42) == 42

    def test_set_creates_nested(self):
        config = ServerConfig.from_dict({})
        config.set("a.b.c", 99)
        assert config.get("a.b.c") == 99

    def test_set_overwrites(self):
        config = ServerConfig.from_dict({"a": {"b": 1}})
        config.set("a.b", 2)
        assert config.get("a.b") == 2

    def test_delete(self):
        config = ServerConfig.from_dict({"x": {"y": 1}})
        assert config.delete("x.y") is True
        assert config.get("x.y") is None

    def test_delete_missing(self):
        config = ServerConfig.from_dict({})
        assert config.delete("a.b") is False

    def test_keys(self):
        config = ServerConfig.from_dict({"a": 1, "b": 2})
        assert sorted(config.keys()) == ["a", "b"]

    def test_to_dict(self):
        data = {"retriever": {"top_k": 5}}
        config = ServerConfig.from_dict(data)
        assert config.to_dict()["retriever"]["top_k"] == 5

    def test_save_and_load(self, tmp_path):
        path = str(tmp_path / "config.yaml")
        config = ServerConfig.from_dict({
            "server": {"name": "retriever", "top_k": 15}
        })
        config.save(path)
        config2 = ServerConfig.from_file(path)
        assert config2.get("server.top_k") == 15

    def test_reload_detects_changes(self, tmp_path):
        path = tmp_path / "hot.yaml"
        path.write_text("top_k: 10\n")
        config = ServerConfig.from_file(str(path))
        assert config.get("top_k") == 10

        # Modify file
        time.sleep(0.05)
        path.write_text("top_k: 20\n")
        result = config.reload()
        assert result.success is True
        assert "top_k" in result.changed_keys
        assert config.get("top_k") == 20

    def test_reload_error_bad_file(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text("valid: true\n")
        config = ServerConfig.from_file(str(path))
        # Write invalid YAML
        time.sleep(0.05)
        path.write_text(": : : broken\n")
        result = config.reload()
        assert result.success is False
        assert result.error is not None

    def test_has_changed(self, tmp_path):
        path = tmp_path / "watch.yaml"
        path.write_text("a: 1\n")
        config = ServerConfig.from_file(str(path))
        assert not config.has_changed()
        time.sleep(0.05)
        path.write_text("a: 2\n")
        assert config.has_changed()

    def test_poll_auto_reload(self, tmp_path):
        path = tmp_path / "poll.yaml"
        path.write_text("b: 1\n")
        config = ServerConfig.from_file(str(path))
        time.sleep(0.05)
        path.write_text("b: 2\n")
        assert config.poll() is True
        assert config.get("b") == 2

    def test_poll_no_change(self, tmp_path):
        path = tmp_path / "still.yaml"
        path.write_text("c: 1\n")
        config = ServerConfig.from_file(str(path))
        assert config.poll() is False
        # No second write, no change

    def test_on_change_callback(self, tmp_path):
        path = tmp_path / "cb.yaml"
        path.write_text("v: 1\n")
        config = ServerConfig.from_file(str(path))
        results = []
        config.on_change(lambda r: results.append(r))
        time.sleep(0.05)
        path.write_text("v: 2\n")
        config.reload()
        assert len(results) == 1
        assert results[0].success is True

    def test_no_file_path_reload(self):
        config = ServerConfig.from_dict({"a": 1})
        result = config.reload()
        assert result.success is False
        assert "No file path" in result.error

    def test_from_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            ServerConfig.from_file("/nonexistent/path/config.yaml")

    def test_repr(self):
        config = ServerConfig.from_dict({"x": 1, "y": 2})
        assert "ServerConfig" in repr(config)

    def test_diff_nested(self):
        old = {"a": {"b": 1, "c": 2}}
        new = {"a": {"b": 1, "c": 3}}
        changed = ServerConfig._diff(old, new)
        assert "a.c" in changed

    def test_key_added_then_reloaded(self, tmp_path):
        path = tmp_path / "add_key.yaml"
        path.write_text("existing: 1\n")
        config = ServerConfig.from_file(str(path))
        time.sleep(0.05)
        path.write_text("existing: 1\nnew_key: 2\n")
        result = config.reload()
        assert "new_key" in result.changed_keys

    def test_key_removed_then_reloaded(self, tmp_path):
        path = tmp_path / "remove_key.yaml"
        path.write_text("a: 1\nb: 2\n")
        config = ServerConfig.from_file(str(path))
        time.sleep(0.05)
        path.write_text("a: 1\n")
        result = config.reload()
        assert "b" in result.changed_keys

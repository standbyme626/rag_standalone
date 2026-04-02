"""
Tests for RAG Pipeline Executor and ToolRegistry.

Tests cover:
1. YAML parsing (single-key, string shorthand, edge cases)
2. Variable resolution (${} references, dotted paths, string interpolation)
3. Pipeline execution (sequential steps, data flow, error handling)
4. ToolRegistry (registration, lookup, input mapping)
5. End-to-end integration (retrieve -> rerank -> format)
"""

from __future__ import annotations

import sys
from pathlib import Path
from textwrap import dedent
from typing import Any, Dict, List

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def mock_settings_env(monkeypatch):
    """Provide required env vars for Settings validation."""
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


# Import after env is set up (Settings is evaluated at import time)
from app.rag.pipeline.pipeline import (
    PipelineConfig,
    PipelineExecutor,
    PipelineStep,
    PipelineStepError,
    PipelineStepTimeout,
    _lookup,
    _parse_step,
    _resolve_value,
    _try_cast,
    load_pipeline_yaml,
    parse_pipeline_yaml,
)
from app.rag.pipeline.retriever_tools import (
    format_results,
    identity,
    query_expand,
    rerank,
    retrieve,
)
from app.rag.pipeline.tool_registry import ToolRegistry


# ===================================================================
# Test: Variable resolution (_resolve_value)
# ===================================================================

class TestResolveValue:
    """Test the ${...} variable resolution system."""

    def test_plain_string_no_refs(self):
        """Strings without ${{}} patterns are returned as-is."""
        assert _resolve_value("hello world", {"foo": "bar"}) == "hello world"

    def test_single_ref_resolves_to_value(self):
        """A standalone ${{var}} reference returns the underlying value."""
        state = {"x": 42, "y": [1, 2, 3]}
        assert _resolve_value("${x}", state) == 42
        assert _resolve_value("${y}", state) == [1, 2, 3]

    def test_dotted_path_resolution(self):
        """Key paths like ${{a.b}} resolve nested dicts."""
        state = {"outer": {"inner": "deep_value"}}
        assert _resolve_value("${outer.inner}", state) == "deep_value"

    def test_multiple_refs_string_interpolation(self):
        """Multiple ${{vars}} in one string produce string interpolation."""
        state = {"name": "Alice", "age": 30}
        result = _resolve_value("${name} is ${age} years old", state)
        assert result == "Alice is 30 years old"

    def test_non_string_passthrough(self):
        """Non-string values pass through unchanged."""
        assert _resolve_value(123, {}) == 123
        assert _resolve_value(3.14, {}) == 3.14
        assert _resolve_value(None, {}) is None
        assert _resolve_value(True, {}) is True

    def test_undefined_variable_raises(self):
        """Referencing a key not in state raises PipelineStepError."""
        with pytest.raises(PipelineStepError, match="Cannot resolve"):
            _resolve_value("${undefined}", {})


# ===================================================================
# Test: _try_cast
# ===================================================================

class TestTryCast:
    """Test string-to-type coercion for inline shorthand params."""

    def test_int_cast(self):
        assert _try_cast("42") == 42
        assert _try_cast("-1") == -1
        assert _try_cast("0") == 0

    def test_float_cast(self):
        assert _try_cast("3.14") == 3.14
        assert _try_cast("-0.5") == -0.5

    def test_bool_cast(self):
        assert _try_cast("true") is True
        assert _try_cast("True") is True
        assert _try_cast("yes") is True
        assert _try_cast("false") is False
        assert _try_cast("False") is False
        assert _try_cast("no") is False

    def test_string_passthrough(self):
        assert _try_cast("hello") == "hello"
        assert _try_cast("3.14abc") == "3.14abc"


# ===================================================================
# Test: _lookup
# ===================================================================

class TestLookup:
    """Test dotted path resolution into state dict."""

    def test_top_level_key(self):
        assert _lookup({"a": 1, "b": 2}, "a") == 1

    def test_nested_key(self):
        state = {"outer": {"middle": {"leaf": "value"}}}
        assert _lookup(state, "outer.middle.leaf") == "value"

    def test_missing_key_raises(self):
        with pytest.raises(PipelineStepError, match="Cannot resolve"):
            _lookup({"a": 1}, "b")

    def test_nested_missing_key_raises(self):
        state = {"outer": {"leaf": "x"}}
        with pytest.raises(PipelineStepError, match="Cannot resolve"):
            _lookup(state, "outer.missing.leaf")


# ===================================================================
# Test: YAML parsing (_parse_step)
# ===================================================================

class TestParseStep:
    """Test individual pipeline step parsing."""

    def test_dict_step_full_config(self):
        """Parse a dict step with input, output, and timeout."""
        yaml_str = dedent("""\
            servers:
              t: app.rag.pipeline.retriever_tools
            pipeline:
              - t.retrieve:
                  input:
                    query: "hello"
                    top_k: 5
                  output: [chunks, query, count]
                  timeout: 10.0
            global_vars: {}
        """)
        config = parse_pipeline_yaml(yaml_str)
        step = config.pipeline[0]
        assert step.server == "t"
        assert step.tool == "retrieve"
        assert step.step_input == {"query": "hello", "top_k": 5}
        assert step.step_output == ["chunks", "query", "count"]
        assert step.timeout == 10.0

    def test_dict_step_minimal(self):
        """Parse a dict step with only input."""
        yaml_str = dedent("""\
            servers: {}
            pipeline:
              - tools.retrieve:
                  input:
                    query: "x"
            global_vars: {}
        """)
        config = parse_pipeline_yaml(yaml_str)
        step = config.pipeline[0]
        assert step.server == "tools"
        assert step.tool == "retrieve"
        assert step.step_input == {"query": "x"}
        assert step.step_output == ["result"]
        assert step.timeout == 30.0

    def test_string_step_simple(self):
        """Parse a simple string step like 'tools.retrieve'."""
        step = _parse_step("tools.retrieve")
        assert step.server == "tools"
        assert step.tool == "retrieve"
        assert step.step_input == {}
        assert step.step_output == ["result"]

    def test_string_step_with_overrides(self):
        """Parse a string step with inline param overrides."""
        step = _parse_step("tools.retrieve:top_k=3:intent=medical")
        assert step.server == "tools"
        assert step.tool == "retrieve"
        assert step.step_input == {"top_k": 3, "intent": "medical"}

    def test_string_no_dot_defaults_to_default_server(self):
        """A string like 'retrieve' defaults server to 'default'."""
        step = _parse_step("retrieve:top_k=5")
        assert step.server == "default"
        assert step.tool == "retrieve"

    def test_parse_multi_key_dict_raises(self):
        """Each pipeline entry must have exactly one key."""
        with pytest.raises(PipelineStepError, match="exactly one key"):
            _parse_step({"a": {}, "b": {}})


# ===================================================================
# Test: parse_pipeline_yaml
# ===================================================================

class TestParsePipelineYaml:
    """Test full YAML parsing."""

    def test_full_yaml_with_multi_steps(self):
        yaml_str = dedent("""\
            servers:
              tools: app.rag.pipeline.retriever_tools
            global_vars:
              top_k: 5
              mode: fast
            pipeline:
              - tools.retrieve:
                  input:
                    query: "${query}"
                    top_k: "${top_k}"
                  output: [chunks, query, count]
              - tools.rerank:
                  input:
                    docs: "${chunks}"
                  output: [top_docs]
        """)
        config = parse_pipeline_yaml(yaml_str)
        assert config.servers == {"tools": "app.rag.pipeline.retriever_tools"}
        assert config.global_vars == {"top_k": 5, "mode": "fast"}
        assert len(config.pipeline) == 2
        assert config.pipeline[0].tool == "retrieve"
        assert config.pipeline[1].tool == "rerank"

    def test_empty_pipeline(self):
        """A pipeline with no steps is valid."""
        yaml_str = "servers: {}\nglobal_vars: {}\npipeline: []"
        config = parse_pipeline_yaml(yaml_str)
        assert config.pipeline == []
        assert config.global_vars == {}

    def test_non_dict_root_raises(self):
        with pytest.raises(PipelineStepError, match="must be a mapping"):
            parse_pipeline_yaml("- step1\n- step2")

    def test_non_list_pipeline_raises(self):
        with pytest.raises(PipelineStepError, match="must be a list"):
            parse_pipeline_yaml("pipeline: single_step")


# ===================================================================
# Test: PipelineExecutor - execution
# ===================================================================

class TestPipelineExecutor:
    """Test PipelineExecutor execution flow."""

    YAML_BASE = dedent("""\
        servers:
          tools: app.rag.pipeline.retriever_tools
        global_vars: {}
        pipeline:
          - tools.retrieve:
              input:
                query: "headache remedies"
                top_k: 5
              output: [chunks, query, count]
    """)

    def test_single_step_execution(self):
        """Execute one tool step, verify outputs land in state."""
        executor = PipelineExecutor.from_string(self.YAML_BASE)
        result = executor.execute()
        assert "chunks" in result
        assert "query" in result
        assert result["query"] == "headache remedies"
        assert len(result["chunks"]) == 5

    def test_from_string_classmethod(self):
        """from_string creates an executor from a YAML string."""
        executor = PipelineExecutor.from_string(self.YAML_BASE)
        assert executor is not None
        assert isinstance(executor.config, PipelineConfig)

    def test_from_classmethod_accepts_config(self):
        """Constructor accepts a PipelineConfig directly."""
        config = parse_pipeline_yaml(self.YAML_BASE)
        executor = PipelineExecutor(config)
        assert len(executor.config.pipeline) == 1

    def test_initial_vars_override_global_vars(self):
        """initial_vars to execute() should override YAML global_vars."""
        yaml_str = dedent("""\
            servers:
              tools: app.rag.pipeline.retriever_tools
            global_vars:
              query: "default"
              top_k: 1
            pipeline:
              - tools.retrieve:
                  input:
                    query: "${query}"
                    top_k: "${top_k}"
                  output: [chunks, query, count]
        """)
        executor = PipelineExecutor.from_string(yaml_str)
        result = executor.execute(initial_vars={"query": "headache", "top_k": 3})
        assert result["query"] == "headache"
        assert len(result["chunks"]) == 3

    def test_multi_step_data_flow(self):
        """Output of step 1 feeds into step 2 input via ${} reference."""
        yaml_str = dedent("""\
            servers:
              tools: app.rag.pipeline.retriever_tools
            global_vars: {}
            pipeline:
              - tools.retrieve:
                  input:
                    query: "headache"
                    top_k: 3
                  output: [chunks, query, count]
              - tools.rerank:
                  input:
                    docs: "${chunks}"
                    query: "${query}"
                    top_k: 2
                  output: [top_docs]
        """)
        executor = PipelineExecutor.from_string(yaml_str)
        result = executor.execute()
        assert "chunks" in result
        assert "top_docs" in result
        assert len(result["top_docs"]) == 2

    def test_three_step_pipeline(self):
        """retrieve -> rerank -> format_results chain."""
        yaml_str = dedent("""\
            servers:
              tools: app.rag.pipeline.retriever_tools
            global_vars: {}
            pipeline:
              - tools.retrieve:
                  input:
                    query: "test"
                    top_k: 3
                  output: [chunks, query]
              - tools.rerank:
                  input:
                    query: "${query}"
                    docs: "${chunks}"
                    top_k: 2
                  output: [top_docs]
              - tools.format_results:
                  input:
                    top_docs: "${top_docs}"
                  output: [formatted_answer]
        """)
        executor = PipelineExecutor.from_string(yaml_str)
        result = executor.execute()
        assert "chunks" in result
        assert "top_docs" in result
        assert "formatted_answer" in result
        assert "Result for" in result["formatted_answer"]

    def test_step_results_accumulation(self):
        """step_results records raw tool output per step."""
        executor = PipelineExecutor.from_string(self.YAML_BASE)
        executor.execute()
        assert "tools.retrieve" in executor.step_results
        raw = executor.step_results["tools.retrieve"]
        assert "chunks" in raw

    def test_get_output(self):
        """get_output convenience method."""
        executor = PipelineExecutor.from_string(self.YAML_BASE)
        executor.execute()
        assert executor.get_output("query") == "headache remedies"
        assert executor.get_output("missing", "fallback") == "fallback"

    def test_final_result_property(self):
        """final_result returns a copy of the state."""
        executor = PipelineExecutor.from_string(self.YAML_BASE)
        executor.execute()
        final = executor.final_result
        assert final is not executor.state
        assert final == executor.state

    def test_output_key_not_found_raises(self):
        """Requesting an output key not in the result dict raises."""
        yaml_str = dedent("""\
            servers:
              tools: app.rag.pipeline.retriever_tools
            global_vars: {}
            pipeline:
              - tools.retrieve:
                  input:
                    query: "test"
                    top_k: 2
                  output: [chunks, this_key_does_not_exist]
        """)
        executor = PipelineExecutor.from_string(yaml_str)
        with pytest.raises(PipelineStepError, match="this_key_does_not_exist"):
            executor.execute()

    def test_resolve_unknown_server_uses_defaults(self):
        """If a server is not in servers map, fall back to defaults[module]."""
        yaml_str = dedent("""\
            servers: {}
            defaults:
              module: app.rag.pipeline.retriever_tools
            global_vars: {}
            pipeline:
              - retrieve:
                  input:
                    query: "test"
                    top_k: 2
                  output: [chunks, query, count]
        """)
        executor = PipelineExecutor.from_string(yaml_str)
        result = executor.execute()
        assert "chunks" in result

    def test_resolve_unknown_tool_raises(self):
        """Non-existent tool under a valid server raises PipelineStepError."""
        yaml_str = dedent("""\
            servers:
              tools: app.rag.pipeline.retriever_tools
            global_vars: {}
            pipeline:
              - tools.this_tool_does_not_exist:
                  input: {}
        """)
        executor = PipelineExecutor.from_string(yaml_str)
        with pytest.raises(PipelineStepError, match="not found"):
            executor.execute()

    def test_resolve_unknown_server_and_no_defaults_raises(self):
        """Unknown server with no defaults raises."""
        yaml_str = dedent("""\
            servers: {}
            global_vars: {}
            pipeline:
              - unknown.retrieve:
                  input:
                    query: "test"
                  output: [chunks]
        """)
        executor = PipelineExecutor.from_string(yaml_str)
        with pytest.raises((PipelineStepError, ModuleNotFoundError)):
            executor.execute()


# ===================================================================
# Test: ToolRegistry
# ===================================================================

class TestToolRegistry:
    """Test ToolRegistry registration and lookup."""

    def test_register_and_get(self):
        """Basic tool registration and retrieval."""
        registry = ToolRegistry()

        def my_fn(a, b):
            return a + b

        registry.register("add", my_fn)
        assert registry.get("add") == my_fn

    def test_decorator_registration(self):
        """@registry.tool should register the decorated function."""
        registry = ToolRegistry()

        @registry.tool("greet", "Say hello")
        def greet(name: str) -> str:
            return f"Hello, {name}!"

        assert "greet" in registry
        assert registry.get("greet")("World") == "Hello, World!"

    def test_list_tools(self):
        """List all registered tool names."""
        registry = ToolRegistry()
        registry.register("a", lambda: 1)
        registry.register("b", lambda: 2)
        names = registry.list_tools()
        assert set(names) == {"a", "b"}

    def test_deregister(self):
        """deregister removes a tool."""
        registry = ToolRegistry()
        registry.register("tmp", lambda: 1)
        assert "tmp" in registry
        registry.deregister("tmp")
        assert "tmp" not in registry

    def test_deregister_nonexistent_is_noop(self):
        registry = ToolRegistry()
        registry.deregister("does-not-exist")
        assert len(registry) == 0

    def test_wrap_missing_raises(self):
        """wrap() raises KeyError for unknown tools."""
        registry = ToolRegistry()
        with pytest.raises(KeyError, match="not found"):
            registry.wrap("unknown")

    def test_wrap_returns_callable(self):
        registry = ToolRegistry()
        registry.register("hello", lambda name: f"Hi {name}")
        fn = registry.wrap("hello")
        assert fn("Alice") == "Hi Alice"

    def test_contains_and_len(self):
        registry = ToolRegistry()
        assert len(registry) == 0
        registry.register("x", lambda: 1)
        assert "x" in registry
        assert len(registry) == 1

    def test_input_schema_inferred_from_signature(self):
        """input_schema should be auto-inferred from fn signature."""
        registry = ToolRegistry()

        def fn(a: int, b: str, c: float = 1.0):
            return {}

        registry.register("auto", fn)
        spec = registry.get_spec("auto")
        assert "a" in spec.input_schema
        assert "b" in spec.input_schema
        assert "c" in spec.input_schema
        assert spec.input_schema["c"]["default"] == 1.0

    def test_map_inputs_from_state(self):
        """map_inputs builds kwargs from state using the input_schema."""
        registry = ToolRegistry()
        registry.register(
            "add",
            lambda x, y: x + y,
            input_schema={"x": "val1", "y": "val2"},
        )
        state = {"val1": 10, "val2": 20}
        kwargs = registry.map_inputs("add", state)
        assert kwargs == {"x": 10, "y": 20}

    def test_map_inputs_with_default_fallback(self):
        """input_spec with source/default falls back to default."""
        registry = ToolRegistry()
        registry.register(
            "greet",
            lambda name: f"Hi {name}",
            input_schema={"name": {"source": "missing", "default": "World"}},
        )
        state = {}
        kwargs = registry.map_inputs("greet", state)
        assert kwargs == {"name": "World"}

    def test_map_inputs_unknown_tool_returns_empty(self):
        registry = ToolRegistry()
        assert registry.map_inputs("nope", {}) == {}

    def test_repr(self):
        registry = ToolRegistry()
        registry.register("a", lambda: 1)
        assert "a" in repr(registry)
        assert "ToolRegistry" in repr(registry)

    def test_get_spec_returns_none_for_unknown(self):
        registry = ToolRegistry()
        assert registry.get_spec("unknown") is None


# ===================================================================
# Test: load_pipeline_yaml (file-based)
# ===================================================================

class TestLoadPipelineYaml:
    """Test loading pipeline YAML from a file."""

    def test_load_from_file(self, tmp_path: Path):
        """load_pipeline_yaml reads and parses a YAML file from disk."""
        yaml_path = tmp_path / "pipeline.yaml"
        yaml_path.write_text(dedent("""\
            servers:
              tools: app.rag.pipeline.retriever_tools
            global_vars: {}
            pipeline:
              - tools.retrieve:
                  input:
                    query: "file test"
                    top_k: 1
                  output: [chunks]
        """))
        executor = PipelineExecutor.from_yaml(str(yaml_path))
        result = executor.execute()
        assert "chunks" in result
        assert result["chunks"][0]["text"]
        assert len(result["chunks"]) == 1


# ===================================================================
# Test: Pipeline dataclass
# ===================================================================

class TestPipelineStepDataclass:
    """Test the PipelineStep dataclass."""

    def test_defaults(self):
        step = PipelineStep(server="x", tool="y")
        assert step.step_input == {}
        assert step.step_output == []
        assert step.timeout == 30.0

    def test_full_spec(self):
        step = PipelineStep(
            server="tools",
            tool="retrieve",
            step_input={"query": "q"},
            step_output=["chunks"],
            timeout=5.0,
        )
        assert step.step_input == {"query": "q"}
        assert step.step_output == ["chunks"]
        assert step.timeout == 5.0


class TestPipelineStepTimeout:
    """Test timeout exception."""

    def test_exception_attributes(self):
        exc = PipelineStepTimeout("slow_tool", 2.0)
        assert exc.step_name == "slow_tool"
        assert exc.timeout == 2.0
        assert "timed out after 2.0" in str(exc)

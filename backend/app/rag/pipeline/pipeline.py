"""
PipelineExecutor - loads YAML pipeline definitions and executes steps sequentially.

YAML format:

```yaml
servers:
  retriever: app.rag.pipeline.retriever_tools
  enhance: app.rag.pipeline.enhance_tools

pipeline:
  - retrieve:
      input:
        query: "${query}"
      output: [chunks]

  - rerank:
      input:
        query: "${query}"
        docs: "${chunks}"
      output: [top_chunks]

global_vars:
  top_k: 5
```

Step resolution:
1. Resolve server.tool_name from the `servers` map, import module, get callable.
2. Build kwargs from step `input`, resolving `${var}` references from global_vars
   or prior step outputs.
3. Call the tool, save outputs to global_vars by key names declared in `output`.
"""

from __future__ import annotations

import importlib
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import structlog
import yaml

logger = structlog.get_logger(__name__)

# Pattern for variable references: ${var_name} or ${var_name.sub.key}
_VAR_PATTERN = re.compile(r"\$\{([a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)*)\}")


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class PipelineStepError(Exception):
    """Raised when a pipeline step fails."""

    def __init__(self, step_name: str, reason: str):
        self.step_name = step_name
        self.reason = reason
        super().__init__(f"Step '{step_name}' failed: {reason}")


class PipelineStepTimeout(Exception):
    """Raised when a pipeline step exceeds its time budget."""

    def __init__(self, step_name: str, timeout: float):
        self.step_name = step_name
        self.timeout = timeout
        super().__init__(f"Step '{step_name}' timed out after {timeout}s")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PipelineStep:
    """A single tool invocation in the pipeline.

    Represents `resolve.tool_name` with optional input/output overrides.
    """
    server: str
    tool: str
    step_input: Dict[str, Any] = field(default_factory=dict)
    step_output: List[str] = field(default_factory=list)
    timeout: float = 30.0  # seconds


@dataclass
class PipelineConfig:
    """Parsed pipeline definition from YAML."""
    servers: Dict[str, str] = field(default_factory=dict)   # {name: module_path}
    pipeline: List[PipelineStep] = field(default_factory=list)
    global_vars: Dict[str, Any] = field(default_factory=dict)

    # Optional module aliases that tools can use at import resolution
    # e.g. {resolve: app.rag.pipeline.tools} means any tool not under a named
    # server falls through to this default module.
    defaults: Dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Variable resolution
# ---------------------------------------------------------------------------

def _resolve_value(value: Any, state: Dict[str, Any]) -> Any:
    """Resolve a value, substituting '${...}' references from state."""
    if isinstance(value, str):
        matches = list(_VAR_PATTERN.finditer(value))
        if not matches:
            return value
        # Single reference - return the resolved object directly (preserving type)
        if len(matches) == 1 and matches[0].group(0) == value:
            return _lookup(state, matches[0].group(1))
        # Multiple references or embedded in text - do string substitution
        def _replace(m: re.Match) -> str:
            val = _lookup(state, m.group(1))
            return str(val)
        return _VAR_PATTERN.sub(_replace, value)
    return value


def _lookup(state: Dict[str, Any], path: str) -> Any:
    """Look up a dotted path in the state dict."""
    keys = path.split(".")
    current: Any = state
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            raise PipelineStepError(
                "variable_resolution",
                f"Cannot resolve '${{{path}}}': key '{key}' not found.",
            )
    return current


# ---------------------------------------------------------------------------
# Tool discovery
# ---------------------------------------------------------------------------

def _get_tool_callable(module_path: str, tool_name: str,
                       servers: Dict[str, str]) -> Callable:
    """Import a module and return the callable named `tool_name`.

    `module_path` can be either:
      1. A plain name that exists in `servers` (e.g. "retriever")
      2. A full dotted module path (e.g. "app.rag.pipeline.tools.retrieve")

    In case (1), the tool_name is appended: f"{servers[name]}.{tool_name}".
    In case (2) the module_path is used directly and tool_name appended.
    """
    if module_path in servers:
        module_path = servers[module_path]

    full_path = f"{module_path}.{tool_name}"
    mod_name, func_name = full_path.rsplit(".", 1)
    mod = importlib.import_module(mod_name)
    func = getattr(mod, func_name, None)
    if func is None:
        raise PipelineStepError(
            f"{full_path}", f"Attribute '{func_name}' not found in module '{mod_name}'."
        )
    if not callable(func):
        raise PipelineStepError(
            f"{full_path}", f"'{func_name}' in '{mod_name}' is not callable."
        )
    return func


# ---------------------------------------------------------------------------
# YAML parsing
# ---------------------------------------------------------------------------

def _parse_step(raw: Union[str, Dict[str, Any]]) -> PipelineStep:
    """Parse a single pipeline entry into a PipelineStep."""
    if isinstance(raw, str):
        # e.g. "retrieve:top_k=10" or just "retrieve"
        parts = raw.split(":")
        server_tool = parts[0]
        overrides: Dict[str, Any] = {}
        for part in parts[1:]:
            if "=" in part:
                k, v = part.split("=", 1)
                overrides[k.strip()] = _try_cast(v.strip())
        if "." in server_tool:
            server, tool = server_tool.split(".", 1)
        else:
            server = "default"
            tool = server_tool
        return PipelineStep(server=server, tool=tool, step_input=overrides,
                            step_output=["result"])

    if isinstance(raw, dict):
        if len(raw) != 1:
            raise PipelineStepError(
                "parse", f"Each pipeline entry must have exactly one key, got {list(raw.keys())}"
            )
        key = next(iter(raw))
        config = raw[key] or {}

        if "." in key:
            server, tool = key.split(".", 1)
        else:
            server = "default"
            tool = key

        step_input: Dict[str, Any] = {}
        step_output: List[str] = ["result"]
        timeout: float = 30.0

        if isinstance(config, dict):
            step_input = dict(config.get("input", {}))
            out = config.get("output")
            if out:
                step_output = out if isinstance(out, list) else [out]
            timeout = float(config.get("timeout", 30.0))

        return PipelineStep(server=server, tool=tool, step_input=step_input,
                            step_output=step_output, timeout=timeout)

    raise PipelineStepError("parse", f"Unknown pipeline step format: {type(raw)}")


def _try_cast(value: str) -> Any:
    """Try to cast a string value to int/float/bool, else return string."""
    if value.lower() in ("true", "yes"):
        return True
    if value.lower() in ("false", "no"):
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def parse_pipeline_yaml(yaml_str: str) -> PipelineConfig:
    """Parse a YAML string into a PipelineConfig."""
    data = yaml.safe_load(yaml_str)
    if not isinstance(data, dict):
        raise PipelineStepError("parse", "YAML root must be a mapping.")

    servers: Dict[str, str] = dict(data.get("servers", {}))
    global_vars: Dict[str, Any] = dict(data.get("global_vars", {}))
    defaults: Dict[str, str] = dict(data.get("defaults", {}))

    raw_steps = data.get("pipeline", [])
    if not isinstance(raw_steps, list):
        raise PipelineStepError("parse", "'pipeline' must be a list.")

    pipeline = [_parse_step(raw) for raw in raw_steps]

    config = PipelineConfig(servers=servers, pipeline=pipeline,
                            global_vars=global_vars, defaults=defaults)
    return config


def load_pipeline_yaml(path: Union[str, Path]) -> PipelineConfig:
    """Load and parse a pipeline YAML file."""
    p = Path(path)
    with open(p, "r", encoding="utf-8") as f:
        return parse_pipeline_yaml(f.read())


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

class PipelineExecutor:
    """Execute a sequential pipeline of tool steps."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.state: Dict[str, Any] = dict(config.global_vars)
        self.step_results: Dict[str, Any] = {}
        self._tool_cache: Dict[str, Callable] = {}

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "PipelineExecutor":
        """Create an executor from a YAML file path."""
        config = load_pipeline_yaml(path)
        return cls(config)

    @classmethod
    def from_string(cls, yaml_str: str) -> "PipelineExecutor":
        """Create an executor from a YAML string."""
        config = parse_pipeline_yaml(yaml_str)
        return cls(config)

    # -- public API --

    def execute(self, initial_vars: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run all pipeline steps sequentially, returning the final state.

        Args:
            initial_vars: Optional variables to merge into global_vars
                          before execution. Takes precedence over YAML global_vars.

        Returns:
            The final global_vars dict after all steps have completed.
        """
        if initial_vars:
            self.state.update(initial_vars)

        for step in self.config.pipeline:
            self._run_step(step)

        return self.state

    def get_output(self, key: str, default: Any = None) -> Any:
        """Get a value from the final state."""
        return self.state.get(key, default)

    @property
    def final_result(self) -> Dict[str, Any]:
        """Convenience property returning the final state."""
        return dict(self.state)

    # -- internal --

    def _run_step(self, step: PipelineStep) -> None:
        """Resolve and execute a single pipeline step."""
        logger.debug("pipeline_step_start", step=f"{step.server}.{step.tool}")

        try:
            func = self._resolve_tool(step)
        except PipelineStepError:
            raise
        except Exception as exc:
            raise PipelineStepError(str(step.tool), f"Tool resolution error: {exc}")

        # Build kwargs from step_input, resolving references
        kwargs = self._build_kwargs(step.step_input)

        start = time.monotonic()
        try:
            result = func(**kwargs)
        except PipelineStepError:
            raise
        except Exception as exc:
            raise PipelineStepError(str(step.tool), f"Tool execution error: {exc}")
        finally:
            elapsed = time.monotonic() - start
            if elapsed > step.timeout:
                raise PipelineStepTimeout(str(step.tool), step.timeout)

        logger.debug(
            "pipeline_step_done",
            step=f"{step.server}.{step.tool}",
            elapsed_ms=round(elapsed * 1000, 1),
        )

        # Save outputs
        self._save_outputs(step, result)

    def _resolve_tool(self, step: PipelineStep) -> Callable:
        """Get the callable for a step, with caching."""
        cache_key = f"{step.server}.{step.tool}"
        if cache_key not in self._tool_cache:
            # Determine module path: check servers first, then defaults
            module_path = step.server
            if module_path not in self.config.servers:
                # Check defaults for a fallback module
                fallback = self.config.defaults.get("module")
                if fallback:
                    module_path = fallback
            self._tool_cache[cache_key] = _get_tool_callable(
                module_path, step.tool, self.config.servers
            )
        return self._tool_cache[cache_key]

    def _build_kwargs(self, step_input: Dict[str, Any]) -> Dict[str, Any]:
        """Build keyword arguments from step input, resolving variables."""
        return {k: _resolve_value(v, self.state) for k, v in step_input.items()}

    def _save_outputs(self, step: PipelineStep, result: Any) -> None:
        """Save step result(s) to state according to the output declaration."""
        # Record result under step's qualified name
        step_key = f"{step.server}.{step.tool}"
        self.step_results[step_key] = result

        if not step.step_output:
            return

        if isinstance(result, dict):
            # Map requested output keys to actual dict keys
            for out_key in step.step_output:
                if out_key in result:
                    self.state[out_key] = result[out_key]
                else:
                    raise PipelineStepError(
                        step_key,
                        f"Output key '{out_key}' not found in step result "
                        f"(available: {list(result.keys())}).",
                    )
        else:
            # Non-dict result: bind to first output key (or 'result')
            primary_key = step.step_output[0] if step.step_output else "result"
            self.state[primary_key] = result

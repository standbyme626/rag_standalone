"""
ToolRegistry - register backend functions as pipeline tools.

Provides input/output mapping and a callable interface so tools
can be discovered by name without import-path coupling.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class ToolSpec:
    """Specification for a registered tool."""
    name: str
    fn: Callable
    description: str = ""
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_keys: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# ToolRegistry
# ---------------------------------------------------------------------------

class ToolRegistry:
    """Register and look up callable tools by name.

    Usage:
        registry = ToolRegistry()

        @registry.tool("enhance_query", "Expand query with synonyms")
        def enhance_query(query: str) -> dict:
            return {"expanded_query": query}

        fn = registry.get("enhance_query")
        fn(query="headache")
    """

    def __init__(self):
        self._tools: Dict[str, ToolSpec] = {}

    # -- registration --

    def tool(self, name: str, description: str = "",
             input_schema: Optional[Dict[str, Any]] = None,
             output_keys: Optional[List[str]] = None):
        """Decorator to register a function as a named tool.

        Args:
            name: Tool identifier (e.g. "rerank", "expand_query")
            description: Human-readable description
            input_schema: Dict describing expected input keys
            output_keys: List of output key names

        Returns:
            Decorator function
        """
        def decorator(fn: Callable) -> Callable:
            schema = input_schema or self._infer_input_schema(fn)
            out_keys = output_keys or []
            spec = ToolSpec(
                name=name,
                fn=fn,
                description=description,
                input_schema=schema,
                output_keys=out_keys,
            )
            self._tools[name] = spec
            logger.debug("tool_registered", tool=name)
            return fn
        return decorator

    def register(self, name: str, fn: Callable,
                 description: str = "",
                 input_schema: Optional[Dict[str, Any]] = None,
                 output_keys: Optional[List[str]] = None) -> None:
        """Register a function imperatively (non-decorator form)."""
        schema = input_schema or self._infer_input_schema(fn)
        out_keys = output_keys or []
        spec = ToolSpec(
            name=name,
            fn=fn,
            description=description,
            input_schema=schema,
            output_keys=out_keys,
        )
        self._tools[name] = spec

    def deregister(self, name: str) -> None:
        """Remove a tool by name. No-op if not found."""
        self._tools.pop(name, None)

    # -- lookup --

    def get(self, name: str) -> Optional[Callable]:
        """Get the callable for a tool by name."""
        spec = self._tools.get(name)
        return spec.fn if spec else None

    def get_spec(self, name: str) -> Optional[ToolSpec]:
        """Get the full ToolSpec for a tool."""
        return self._tools.get(name)

    def list_tools(self) -> List[str]:
        """List all registered tool names."""
        return list(self._tools.keys())

    def wrap(self, name: str) -> Callable:
        """Get a tool wrapper that raises if the tool is not found."""
        if name not in self._tools:
            available = ", ".join(self._tools) or "(none)"
            raise KeyError(
                f"Tool '{name}' not found. Available: {available}"
            )
        return self._tools[name].fn

    # -- batch / pipeline helpers --

    def map_inputs(self, name: str, state: Dict[str, Any]) -> Dict[str, Any]:
        """Build kwargs for a tool from state using its input_schema.

        For each key in the tool's input_schema, look up the value in state.
        If the schema value is a string with a dot path, it traverses dicts.
        If the schema value is a dict with 'source' and 'default', it uses
        the default as fallback.
        """
        spec = self._tools.get(name)
        if not spec:
            return {}
        kwargs: Dict[str, Any] = {}
        for param_name, spec_val in spec.input_schema.items():
            kwargs[param_name] = self._resolve_from_state(spec_val, state)
        return kwargs

    def _resolve_from_state(self, spec_val: Any, state: Dict[str, Any]) -> Any:
        """Resolve a single input_spec value against state."""
        if isinstance(spec_val, str):
            # Direct key or dotted path
            return _dot_get(state, spec_val)
        if isinstance(spec_val, dict):
            source = spec_val.get("source", "")
            default = spec_val.get("default")
            if source:
                try:
                    return _dot_get(state, source)
                except (KeyError, TypeError):
                    return default
            return default
        return spec_val

    # -- introspection --

    def _infer_input_schema(self, fn: Callable) -> Dict[str, Any]:
        """Infer a simple input schema from function signature."""
        sig = inspect.signature(fn)
        schema: Dict[str, Any] = {}
        for name, param in sig.parameters.items():
            if param.kind in (inspect.Parameter.VAR_POSITIONAL,
                              inspect.Parameter.VAR_KEYWORD):
                continue
            schema[name] = {
                "source": name,
                "default": (param.default if param.default is not param.empty
                            else None),
            }
        return schema

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def __repr__(self) -> str:
        return f"ToolRegistry(tools={self.list_tools()})"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dot_get(obj: Any, path: str) -> Any:
    """Traverse a dotted path on a nested dict."""
    keys = path.split(".")
    current = obj
    for key in keys:
        if isinstance(current, dict):
            if key in current:
                current = current[key]
            else:
                raise KeyError(
                    f"Key '{key}' not found. Available: {list(current.keys())}"
                )
        else:
            raise TypeError(
                f"Cannot traverse '{key}' on {type(current).__name__}"
            )
    return current

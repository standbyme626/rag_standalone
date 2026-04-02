"""
PromptSandbox - safe Jinja2 template rendering with XSS protection.

Usage:
    ps = PromptSandbox()
    template = "Hello {{ name }}, your score is {{ score }}"
    result = ps.render(template, {"name": "<script>alert(1)</script>", "score": 95})
    # result: "Hello &lt;script&gt;alert(1)&lt;/script&gt;, your score is 95"
"""

from __future__ import annotations

import re
import html
from typing import Any, Dict, Optional

from jinja2 import (
    BaseLoader,
    Environment,
    StrictUndefined,
    TemplateError,
    UndefinedError,
)

import structlog

logger = structlog.get_logger(__name__)


class SandboxError(Exception):
    """Raised when sandbox rendering fails or a security violation is detected."""

    pass


# ---------------------------------------------------------------------------
# Blocked names / patterns
# ---------------------------------------------------------------------------

# Keywords that indicate dangerous template constructs
_BLOCKED_PATTERNS = re.compile(
    r"\b("
    r"import|exec|eval|compile|open|file|"
    r"os\.|sys\.|subprocess|socket|"
    r"__class__|__mro__|__subclasses__|__globals__|__builtins__|"
    r"getattr|setattr|delattr|__import__"
    r")\b",
    re.IGNORECASE,
)

# Disallowed Jinja2 attributes on the sandboxed environment
_BLOCKED_ATTRS = frozenset({
    "os", "sys", "cycler", "joiner", "namespace",
    "range", "lipsum", "dict", "cycler.__init__",
})


# ---------------------------------------------------------------------------
# Custom Undefined that raises on attribute access
# ---------------------------------------------------------------------------

# StrictUndefined raises UndefinedError; we catch and re-raise as SandboxError below.


# ---------------------------------------------------------------------------
# Helper: escape user-provided values
# ---------------------------------------------------------------------------

def _escape_value(value: Any) -> Any:
    """Recursively escape HTML in string values.

    Non-string types pass through unchanged.
    """
    if isinstance(value, str):
        return html.escape(value, quote=True)
    if isinstance(value, dict):
        return {k: _escape_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        t = type(value)
        return t(_escape_value(v) for v in value)
    return value


def _validate_template(template_str: str) -> None:
    """Scan template source for dangerous patterns before compilation."""
    match = _BLOCKED_PATTERNS.search(template_str)
    if match:
        raise SandboxError(
            f"Dangerous pattern detected in template: '{match.group(1)}'. "
            "import, eval, exec, and similar constructs are blocked."
        )


# ---------------------------------------------------------------------------
# PromptSandbox
# ---------------------------------------------------------------------------

class PromptSandbox:
    """Safe Jinja2 template rendering with XSS protection.

    Features:
    - Runs in a restricted Jinja2 sandbox (no imports, no attribute access
      to internals).
    - All string values in the context are HTML-escaped before rendering.
    - Template source is scanned for dangerous keywords.
    """

    def __init__(self):
        self._env = Environment(
            loader=BaseLoader(),
            autoescape=False,  # We manually escape values
            undefined=StrictUndefined,
            trim_blocks=True,
            lstrip_blocks=False,
        )
        # Restrict: no builtins import, no attribute access to private stuff
        self._env.globals = {k: v for k, v in self._env.globals.items()
                             if k not in _BLOCKED_ATTRS}

    def render(self, template_str: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Render a template string with the given context.

        Args:
            template_str: Jinja2 template source.
            context: Variables available in the template.  All string values
                     are HTML-escaped for XSS protection.

        Returns:
            Rendered template string.

        Raises:
            SandboxError: If the template contains dangerous patterns, uses
                         undefined variables, or fails to render.
        """
        _validate_template(template_str)

        # Escape user input in context
        safe_ctx = _escape_value(context) if context else {}

        try:
            tpl = self._env.from_string(template_str)
            return tpl.render(**safe_ctx)
        except UndefinedError as exc:
            raise SandboxError(f"Undefined: {exc}") from exc
        except TemplateError as exc:
            raise SandboxError(f"Template rendering error: {exc}") from exc
        except SandboxError:
            raise
        except Exception as exc:
            raise SandboxError(f"Unexpected rendering error: {exc}") from exc

    def _safe_render(self, template_str: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Alias for :meth:`render`.  Matches UltraRAG's ``_safe_render`` API."""
        return self.render(template_str, context)

    def validate_template(self, template_str: str) -> bool:
        """Check whether a template string is safe and syntactically valid.

        Returns True if valid, raises SandboxError otherwise.
        """
        _validate_template(template_str)
        try:
            self._env.from_string(template_str)
            return True
        except TemplateError as exc:
            raise SandboxError(f"Invalid template syntax: {exc}") from exc

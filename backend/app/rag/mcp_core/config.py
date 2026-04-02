"""
ServerConfig - YAML-based server configuration with hot-reload support.

Usage:
    config = ServerConfig(path="config/server.yaml")
    config.load()
    value = config.get("retriever.top_k", default=10)

    # Hot-reload (detects file changes)
    config.watch()

    # Manual reload
    config.reload()
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import structlog
import yaml

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Reload result
# ---------------------------------------------------------------------------


@dataclass
class ConfigReloadResult:
    """Result of a config reload attempt."""

    success: bool
    changed_keys: List[str] = field(default_factory=list)
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# ServerConfig
# ---------------------------------------------------------------------------


class ServerConfig:
    """Load / manage server configs from YAML with hot-reload support.

    The config is a flat nested dict.  Keys can be accessed via dotted paths:
    ``"retriever.top_k"``, ``"reranker.backend"`` etc.

    Attributes:
        path: Path to the YAML config file.
        data: Current config data as a dict.
    """

    # Config directory layout convention
    CONFIG_DIR_NAME = "mcp_config"

    def __init__(self, path: Optional[str] = None, data: Optional[Dict[str, Any]] = None):
        self._path: Optional[str] = path
        self._data: Dict[str, Any] = data or {}
        self._watchers: List[Callable[[ConfigReloadResult], None]] = []
        self._file_hash: Optional[float] = None
        self._last_loaded_at: Optional[float] = None

    # -- loading --

    @classmethod
    def from_file(cls, path: str) -> "ServerConfig":
        """Create a ServerConfig from a YAML file path."""
        config = cls(path=path)
        config.load()
        return config

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ServerConfig":
        """Create a ServerConfig from a dict (no file)."""
        return cls(data=data)

    def load(self) -> Dict[str, Any]:
        """Load config from the YAML file.

        Returns:
            The parsed config dict.

        Raises:
            FileNotFoundError: If the path does not exist.
            yaml.YAMLError: If the file is not valid YAML.
        """
        if self._path is None:
            raise ValueError("No file path set. Use ServerConfig.from_dict() or set path.")

        p = Path(self._path)
        if not p.exists():
            raise FileNotFoundError(f"Config file not found: {self._path}")

        with open(p, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        self._data = raw if isinstance(raw, dict) else {}
        self._file_hash = p.stat().st_mtime
        self._last_loaded_at = time.time()
        logger.info(
            "config_loaded",
            path=self._path,
            keys=len(self._data),
        )
        return self._data

    def save(self, path: Optional[str] = None) -> None:
        """Write current config data back to a YAML file.

        Args:
            path: Override path.  Defaults to the configured path.
        """
        target = path or self._path
        if target is None:
            raise ValueError("No file path set.")

        p = Path(target)
        p.parent.mkdir(parents=True, exist_ok=True)

        with open(p, "w", encoding="utf-8") as f:
            yaml.dump(self._data, f, default_flow_style=False, sort_keys=False)

        logger.info("config_saved", path=target)

    # -- access --

    def get(self, key: str, default: Any = None) -> Any:
        """Get a config value by dotted path.

        Example:
            config.get("retriever.top_k", default=10)
            config.get("reranker.backend", default="qwen")
        """
        keys = key.split(".")
        current: Any = self._data
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return default
        return current

    def set(self, key: str, value: Any) -> None:
        """Set a config value by dotted path.

        Creates intermediate dicts if they don't exist.

        Example:
            config.set("retriever.top_k", 20)
        """
        keys = key.split(".")
        current = self._data
        for k in keys[:-1]:
            if k not in current or not isinstance(current[k], dict):
                current[k] = {}
            current = current[k]
        current[keys[-1]] = value

    def delete(self, key: str) -> bool:
        """Delete a config value by dotted path.

        Returns True if the key existed and was deleted.
        """
        keys = key.split(".")
        current: Any = self._data
        for k in keys[:-1]:
            if not isinstance(current, dict) or k not in current:
                return False
            current = current[k]
        if not isinstance(current, dict) or keys[-1] not in current:
            return False
        del current[keys[-1]]
        return True

    def keys(self) -> List[str]:
        """Return top-level keys."""
        return list(self._data.keys())

    def to_dict(self) -> Dict[str, Any]:
        """Return the full config as a dict."""
        return dict(self._data)

    @property
    def path(self) -> Optional[str]:
        return self._path

    @property
    def last_loaded_at(self) -> Optional[float]:
        return self._last_loaded_at

    # -- hot-reload --

    def reload(self) -> ConfigReloadResult:
        """Reload config from file, detecting changed keys.

        Returns:
            ConfigReloadResult with success status, changed keys, and any error.
        """
        if self._path is None:
            return ConfigReloadResult(
                success=False, error="No file path configured"
            )

        try:
            old_data = dict(self._data)
            self.load()
            changed = self._diff(old_data, self._data)
            result = ConfigReloadResult(success=True, changed_keys=changed)
            if changed:
                logger.info(
                    "config_reloaded",
                    path=self._path,
                    changed_keys=changed,
                )
            self._notify(result)
            return result
        except Exception as exc:
            logger.error("config_reload_failed", path=self._path, error=str(exc))
            result = ConfigReloadResult(success=False, error=str(exc))
            self._notify(result)
            return result

    def has_changed(self) -> bool:
        """Check if the backing file has been modified since last load."""
        if self._path is None or self._file_hash is None:
            return False
        try:
            return Path(self._path).stat().st_mtime != self._file_hash
        except OSError:
            return False

    def watch(self, callback: Optional[Callable[[ConfigReloadResult], None]] = None) -> None:
        """Register a callback for reload detection.

        The callback receives a ConfigReloadResult.  Call :meth:`poll` to
        trigger the check.  This is a lightweight file mtime-based approach;
        for production, consider ``watchdog`` library.
        """
        if callback:
            self._watchers.append(callback)

    def poll(self) -> bool:
        """Check for file changes and auto-reload if detected.

        Returns:
            True if a reload was triggered, False otherwise.
        """
        if self.has_changed():
            self.reload()
            return True
        return False

    def on_change(self, callback: Callable[[ConfigReloadResult], None]) -> None:
        """Register a callback invoked after each reload (manual or polled)."""
        self._watchers.append(callback)

    # -- internals --

    def _notify(self, result: ConfigReloadResult) -> None:
        """Fire all registered watchers."""
        for fn in self._watchers:
            try:
                fn(result)
            except Exception:
                logger.exception("config_watcher_error")

    @staticmethod
    def _diff(old: Dict[str, Any], new: Dict[str, Any], prefix: str = "") -> List[str]:
        """Compute changed keys between two flat/nested dicts."""
        changed: List[str] = []
        all_keys = set(list(old.keys()) + list(new.keys()))
        for key in all_keys:
            dotted = f"{prefix}{key}" if not prefix else f"{prefix}.{key}"
            if key not in old:
                changed.append(dotted)
            elif key not in new:
                changed.append(dotted)
            elif isinstance(old[key], dict) and isinstance(new[key], dict):
                changed.extend(ServerConfig._diff(old[key], new[key], dotted))
            elif old[key] != new[key]:
                changed.append(dotted)
        return changed

    def __repr__(self) -> str:
        return f"ServerConfig(path={self._path!r}, keys={list(self._data.keys())})"

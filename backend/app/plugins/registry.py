"""插件注册中心。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import DomainPlugin

if TYPE_CHECKING:
    pass

_PLUGIN_REGISTRY: dict[str, DomainPlugin] = {}


def register_plugin(name: str, plugin: DomainPlugin) -> None:
    """注册插件实例。"""
    _PLUGIN_REGISTRY[name] = plugin


def get_plugin(name: str) -> DomainPlugin:
    """获取已注册的插件实例。"""
    plugin = _PLUGIN_REGISTRY.get(name)
    if plugin is None:
        available = list(_PLUGIN_REGISTRY.keys())
        raise ValueError(
            f"Plugin '{name}' not found. Available: {available}"
        )
    return plugin


def list_plugins() -> list[str]:
    return list(_PLUGIN_REGISTRY.keys())


async def initialize_plugins(plugins: list[DomainPlugin]) -> None:
    """批量初始化插件。"""
    for plugin in plugins:
        await plugin.initialize()
        register_plugin(plugin.NAME, plugin)

"""Web Search 抽象基类 — 统一多后端接口"""

from __future__ import annotations

import abc
from typing import Any, Dict, List, Optional


class BaseWebSearchBackend(abc.ABC):
    """Web 搜索后端抽象基类"""

    backend_name: str = "base"

    @abc.abstractmethod
    async def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """搜索网络内容

        Args:
            query: 搜索查询
            top_k: 返回结果数量

        Returns:
            [{"title": str, "content": str, "url": str, "score": float, "source": str}]
        """
        ...

    def close(self) -> None:
        """释放资源（可选）"""
        return None

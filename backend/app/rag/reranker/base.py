from abc import ABC, abstractmethod
from typing import Dict, List


class BaseReranker(ABC):
    """重排器基类 — 所有 reranker 后端必须实现此接口"""

    backend_name: str = "base"

    @abstractmethod
    def rerank(
        self, query: str, docs: List[Dict], top_k: int = 3
    ) -> List[Dict]:
        """
        对文档按查询相关性进行重排

        Args:
            query: 查询文本
            docs: 文档列表（每个文档至少含 "content" 字段）
            top_k: 返回结果数量

        Returns:
            按相关性排序的文档列表，每个文档附加 "score" 和 "source" 字段
        """
        pass

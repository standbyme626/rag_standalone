"""
CrossEncoder 重排后端 — 使用 SentenceTransformers CrossEncoder 模型

参考: UltraRAG servers/reranker/src/reranker.py (sentence_transformers backend)
支持模型: BAAI/bge-reranker-base, BAAI/bge-reranker-large, jina-reranker 等
"""

import time
from typing import Dict, List, Optional

from app.rag.reranker.base import BaseReranker

# 懒加载避免 import 时 GPU 初始化
_sentence_transformers = None


def _import_sentence_transformers():
    global _sentence_transformers
    if _sentence_transformers is None:
        from sentence_transformers import CrossEncoder
        _sentence_transformers = CrossEncoder
    return _sentence_transformers


class CrossEncoderReranker(BaseReranker):
    """基于 SentenceTransformers CrossEncoder 的重排器"""

    backend_name: str = "cross_encoder"

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        device: Optional[str] = None,
        batch_size: int = 32,
        max_length: int = 512,
        trust_remote_code: bool = False,
    ):
        """
        Args:
            model_name: 模型名称或路径（HuggingFace 格式）
            device: 设备（默认自动检测）
            batch_size: 批处理大小
            max_length: 最大序列长度
            trust_remote_code: 是否信任远程代码
        """
        import torch

        CrossEncoder = _import_sentence_transformers()

        # 设备自动检测
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length

        self.model = CrossEncoder(
            model_name,
            max_length=max_length,
            device=device,
            trust_remote_code=trust_remote_code,
        )

    def rerank(
        self,
        query: str,
        docs: List[Dict],
        top_k: int = 3,
    ) -> List[Dict]:
        """
        使用 CrossEncoder 对文档进行重排

        Args:
            query: 查询文本
            docs: 文档列表（每个文档至少含 "content" 字段）
            top_k: 返回结果数量

        Returns:
            排序后的文档列表
        """
        if not docs:
            return []

        # 构造 (query, doc) 配对
        pairs = [(query, doc.get("content", "")) for doc in docs]

        # 批量打分
        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )

        # 确保 scores 是 list
        if hasattr(scores, "tolist"):
            scores = scores.tolist()

        # 回填分数
        for i, doc in enumerate(docs):
            doc["score"] = float(scores[i])
            doc["source"] = "reranked"

        # 排序
        docs.sort(key=lambda x: x["score"], reverse=True)
        return docs[:top_k]

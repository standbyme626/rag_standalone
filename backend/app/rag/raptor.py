"""RAPTOR — Recursive Abstractive Processing for Tree-Organized Retrieval

来源: ragflow rag/raptor.py

UMAP 降维 + GMM 聚类（BIC 自动定簇数）+ 每层 LLM 递归摘要。
"""

from __future__ import annotations

import asyncio
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import structlog
from sklearn.mixture import GaussianMixture

logger = structlog.get_logger(__name__)


# --------------- 可配置 Prompt ---------------

DEFAULT_SUMMARIZATION_PROMPT = """你是一个内容总结助手。请对以下内容进行精炼总结，保留所有关键信息：

{cluster_content}

请用中文简明扼要地总结上述内容的核心要点。"""


# --------------- 数据类型 ---------------

class RaptorChunk:
    """RAPTOR 处理的 chunk 单元。

    Attributes:
        text: 文本内容
        embedding: 向量表示
        meta: 任意元数据
    """

    __slots__ = ("text", "embedding", "meta")

    def __init__(
        self,
        text: str,
        embedding: Optional[List[float]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ):
        self.text = text
        self.embedding = embedding
        self.meta = meta or {}

    def is_valid(self) -> bool:
        return bool(self.text) and self.embedding is not None and len(self.embedding) > 0

    def to_dict(self) -> Dict[str, Any]:
        return {"text": self.text, "embedding": self.embedding, "meta": self.meta}


# --------------- 核心编排器 ---------------

class Raptor:
    """RAPTOR 递归抽象聚类器

    用法:
        model = Raptor(
            llm_call=async_chat_fn,
            embed_call=async_embed_fn,
            summarization_prompt=custom_prompt,
        )
        enriched_chunks = await model(chunks)
    """

    def __init__(
        self,
        llm_call: Optional[Callable] = None,
        embed_call: Optional[Callable] = None,
        max_cluster: int = 10,
        max_token: int = 512,
        threshold: float = 0.1,
        max_errors: int = 3,
        summarization_prompt: Optional[str] = None,
        random_state: int = 42,
    ):
        self.llm_call = llm_call
        self.embed_call = embed_call
        self.max_cluster = max_cluster
        self.max_token = max_token
        self.threshold = threshold
        self.max_errors = max(1, max_errors)
        self._error_count = 0
        self.summarization_prompt = summarization_prompt or DEFAULT_SUMMARIZATION_PROMPT
        self.random_state = random_state

    # ---- public ----

    async def __call__(
        self, chunks: List[RaptorChunk]
    ) -> List[RaptorChunk]:
        """执行 RAPTOR 聚类摘要。

        将原始 chunks 按语义聚类，每类生成交叉摘要，循环至无法进一步聚类。

        Args:
            chunks: 输入 chunk 列表（text + embedding）

        Returns:
            包含所有原始 chunk 以及生成的层次摘要 chunk 的列表。
        """
        if not chunks or len(chunks) <= 1:
            return chunks

        # 转为内部 (text, embedding) 元组，便于就地扩展
        pool: List[Tuple[str, np.ndarray]] = []
        for c in chunks:
            if c.is_valid():
                pool.append((c.text, np.array(c.embedding, dtype=np.float32)))

        if len(pool) <= 1:
            return [RaptorChunk(t, e.tolist()) for t, e in pool]

        start, end = 0, len(pool)

        while end - start > 1:
            batch = pool[start:end]
            embeddings = np.array([e for _, e in batch])

            # 只有 2 个 chunk：直接合并
            if len(batch) == 2:
                pool = await self._summarize(pool, [start, start + 1])
                start = end
                end = len(pool)
                continue

            # >2 个 chunk：UMAP + GMM 聚类
            reduced = await self._reduce_dim(embeddings)
            n_clusters = self._best_n_clusters(reduced)

            labels = self._assign_labels(reduced, n_clusters)
            tasks = []
            for c in range(n_clusters):
                ck_idx = [i + start for i, l in enumerate(labels) if l == c]
                if ck_idx:
                    tasks.append(self._summarize(pool, ck_idx))

            results = await asyncio.gather(*tasks, return_exceptions=True)
            for r in results:
                if isinstance(r, Exception):
                    logger.error("raptor_summarize_error", error=str(r))

            start = end
            end = len(pool)

        # 转回 RaptorChunk 列表
        return [RaptorChunk(text=t, embedding=e.tolist()) for t, e in pool]

    # ---- 内部方法 ----

    async def _reduce_dim(self, embeddings: np.ndarray) -> np.ndarray:
        """UMAP 降维到 2-12 维。"""
        import umap

        n_neighbors = int((len(embeddings) - 1) ** 0.8)
        n_components = max(2, min(12, len(embeddings) - 2))
        n_components = min(n_components, len(embeddings))

        return umap.UMAP(
            n_neighbors=max(2, n_neighbors),
            n_components=n_components,
            metric="cosine",
            random_state=self.random_state,
        ).fit_transform(embeddings)

    def _best_n_clusters(self, embeddings: np.ndarray) -> int:
        """BIC 选择最优簇数。"""
        max_clusters = min(self.max_cluster, len(embeddings))
        if max_clusters <= 1:
            return 1

        n_range = np.arange(1, max_clusters)
        bics = []
        for n in n_range:
            gm = GaussianMixture(n_components=int(n), random_state=self.random_state)
            gm.fit(embeddings)
            bics.append(gm.bic(embeddings))
        return int(n_range[np.argmin(bics)])

    def _assign_labels(self, reduced: np.ndarray, n_clusters: int) -> List[int]:
        """GMM 聚类 + 软分配阈值。"""
        if n_clusters == 1:
            return [0] * len(reduced)

        gm = GaussianMixture(n_components=n_clusters, random_state=self.random_state)
        gm.fit(reduced)
        probs = gm.predict_proba(reduced)

        labels: List[int] = []
        for prob in probs:
            candidates = np.where(prob > self.threshold)[0]
            labels.append(int(candidates[0]) if len(candidates) > 0 else 0)
        return labels

    async def _summarize(
        self, pool: List[Tuple[str, np.ndarray]], indices: List[int]
    ) -> List[Tuple[str, np.ndarray]]:
        """对一批 chunk 调用 LLM 生成摘要，并追加到 pool。"""
        if not self.llm_call or not self.embed_call:
            return pool

        texts = [pool[i][0] for i in indices]
        if not texts:
            return pool

        # 控制每个 chunk 的长度，避免超出 LLM 上下文
        max_total = 8192 - self.max_token
        per_chunk = max(1, max_total // len(texts))
        truncated = [t[:per_chunk] if len(t) > per_chunk else t for t in texts]
        cluster_content = "\n".join(truncated)

        try:
            response = await self.llm_call(
                system_prompt="你是一个内容总结助手。",
                messages=[{
                    "role": "user",
                    "content": self.summarization_prompt.format(
                        cluster_content=cluster_content
                    ),
                }],
                max_tokens=self.max_token,
            )
            response = re.sub(
                r"(······\n由于长度的原因，回答被截断了，要继续吗？|For the content length reason, it stopped, continue?)",
                "",
                response or "",
            )
            if not response.strip():
                return pool

            emb_raw = await self.embed_call(response)
            if isinstance(emb_raw, np.ndarray):
                emb_raw = emb_raw.tolist()
            emb = np.array(emb_raw, dtype=np.float32)

            pool.append((response, emb))

        except Exception as e:
            self._error_count += 1
            logger.warning(
                "raptor_summarize_failed",
                chunk_count=len(texts),
                error=str(e),
                error_count=self._error_count,
            )
            if self._error_count >= self.max_errors:
                raise RuntimeError(
                    f"RAPTOR 中止（{self._error_count} 个错误，最大 {self.max_errors}）"
                ) from e

        return pool

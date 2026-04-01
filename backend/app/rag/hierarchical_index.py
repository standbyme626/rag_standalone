from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Literal, Optional, Protocol

import structlog

from app.core.config import settings

logger = structlog.get_logger(__name__)

IndexLevel = Literal["document", "section", "paragraph"]


@dataclass
class HierarchicalHit:
    doc_id: str
    level: IndexLevel
    text: str
    score: float
    metadata: Dict[str, object]

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class HierarchicalIndexBackend(Protocol):
    async def search(
        self,
        query: str,
        level: IndexLevel = "paragraph",
        top_k: int = 3,
        filters: Optional[Dict[str, object]] = None,
    ) -> List[HierarchicalHit]: ...


class MilvusHierarchicalIndex:
    """
    Milvus-backed hierarchical index for legal/civil code search.

    Levels:
        - document: top-level law (e.g. 民法典)
        - section:   编 / chapter
        - paragraph: 条 / article (default)
    """

    def __init__(self, collection_name: str = "huatuo_knowledge"):
        from app.rag.modules.vector import VectorStoreManager

        self.vector_store = VectorStoreManager(collection_name=collection_name)

    async def search(
        self,
        query: str,
        level: IndexLevel = "paragraph",
        top_k: int = 3,
        filters: Optional[Dict[str, object]] = None,
    ) -> List[HierarchicalHit]:
        filter_expr = self._build_filter_expr(level, filters)
        raw_results = await self.vector_store.asearch(
            query, top_k=max(top_k, 5), filter_expr=filter_expr
        )
        hits: List[HierarchicalHit] = []
        for item in raw_results[:top_k]:
            meta = item.get("metadata", {})
            hits.append(
                HierarchicalHit(
                    doc_id=str(meta.get("doc_id", "")),
                    level=level,
                    text=item.get("text", meta.get("chunk_text", "")),
                    score=float(item.get("score", 0.0)),
                    metadata=meta,
                )
            )
        return hits

    def _build_filter_expr(
        self, level: IndexLevel, filters: Optional[Dict[str, object]]
    ) -> Optional[str]:
        parts: List[str] = []
        parts.append(f'metadata.get("hierarchy_level", "paragraph") == "{level}"')
        if filters:
            if "law_code" in filters:
                parts.append(f'metadata.get("law_code") == "{filters["law_code"]}"')
            if "article_number" in filters:
                parts.append(
                    f'metadata.get("article_number") == {filters["article_number"]}'
                )
        return " and ".join(parts) if parts else None


class NoopHierarchicalIndex:
    """Default placeholder backend: keeps behavior unchanged until real index is ready."""

    async def search(
        self,
        query: str,
        level: IndexLevel = "paragraph",
        top_k: int = 3,
        filters: Optional[Dict[str, object]] = None,
    ) -> List[HierarchicalHit]:
        logger.debug(
            "hierarchical_index_noop",
            query=(query or "")[:80],
            level=level,
            top_k=top_k,
        )
        return []


class HierarchicalIndexGateway:
    def __init__(self, backend: Optional[HierarchicalIndexBackend] = None):
        self.backend: HierarchicalIndexBackend = backend or MilvusHierarchicalIndex()

    def set_backend(self, backend: HierarchicalIndexBackend) -> None:
        self.backend = backend

    async def search(
        self,
        query: str,
        level: IndexLevel = "paragraph",
        top_k: int = 3,
        filters: Optional[Dict[str, object]] = None,
        force_enable: bool = False,
    ) -> List[Dict[str, object]]:
        enabled = bool(force_enable or settings.ENABLE_HIERARCHICAL_INDEX)
        if not enabled:
            return []

        normalized_level = (level or "paragraph").lower().strip()
        if normalized_level not in {"document", "section", "paragraph"}:
            normalized_level = "paragraph"

        try:
            hits = await self.backend.search(
                query=query,
                level=normalized_level,  # type: ignore[arg-type]
                top_k=max(1, int(top_k)),
                filters=filters,
            )
            serialized = [
                h.to_dict() if hasattr(h, "to_dict") else dict(h) for h in hits
            ]
            logger.info(
                "hierarchical_index_search",
                level=normalized_level,
                top_k=top_k,
                hit_count=len(serialized),
            )
            return serialized
        except Exception as exc:
            logger.warning("hierarchical_index_search_failed", error=str(exc))
            return []


hierarchical_index_gateway = HierarchicalIndexGateway()

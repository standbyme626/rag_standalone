"""
Built-in tools for the pipeline executor.

These are simple tools used for testing and as reference implementations.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def retrieve(query: str, top_k: int = 3, **kwargs: Any) -> Dict[str, Any]:
    """Simulate document retrieval. Returns a list of chunks."""
    chunks = [
        {"id": i, "text": f"Result for '{query}' part {i}", "score": 0.9 - i * 0.1}
        for i in range(top_k)
    ]
    return {"chunks": chunks, "query": query, "count": len(chunks)}


def rerank(query: str, docs: Optional[List[Dict[str, Any]]] = None,
           top_k: int = 2, **kwargs: Any) -> Dict[str, Any]:
    """Simulate reranking documents. Returns sorted docs."""
    if not docs:
        return {"top_docs": [], "query": query}
    sorted_docs = sorted(docs, key=lambda d: d.get("score", 0), reverse=True)
    return {
        "top_docs": sorted_docs[:top_k],
        "query": query,
        "original_count": len(docs),
    }


def query_expand(query: str, n_variants: int = 2, **kwargs: Any) -> Dict[str, Any]:
    """Simulate query expansion. Returns original + variant queries."""
    variants = [f"{query} (variant {i})" for i in range(1, n_variants + 1)]
    return {"queries": [query] + variants, "original": query}


def format_results(chunks: Optional[List[Dict[str, Any]]] = None,
                   top_docs: Optional[List[Dict[str, Any]]] = None,
                   **kwargs: Any) -> Dict[str, Any]:
    """Format results into a final answer string."""
    sources = top_docs or chunks or []
    texts = [s.get("text", str(s)) for s in sources]
    answer = "\n".join(f"- {t}" for t in texts) if texts else "No results found."
    return {"formatted_answer": answer, "source_count": len(sources)}


def identity(x: Any = None, **kwargs: Any) -> Dict[str, Any]:
    """Pass-through tool for testing data flow."""
    return {"x": x}

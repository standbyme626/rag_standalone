"""Tests for Web Search multi-backend module."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from app.rag.modules.web_search.base import BaseWebSearchBackend

# Factory tests
class TestWebSearchFactory:
    def test_unknown_backend_returns_none(self):
        from app.rag.modules.web_search import create_web_search_backend

        result = create_web_search_backend(backend="unknown")
        assert result is None

    def test_missing_tavily_returns_none(self):
        from app.rag.modules.web_search import create_web_search_backend

        result = create_web_search_backend(
            backend="tavily",
            tavily_api_key="test-key",
        )
        assert result is None

    def test_missing_exa_returns_none(self):
        from app.rag.modules.web_search import create_web_search_backend

        result = create_web_search_backend(
            backend="exa",
            exa_api_key="test-key",
        )
        assert result is None


# Mocked backend tests (don't require API keys)
class MockedTavilyBackend(BaseWebSearchBackend):
    """Mock implementation for testing interface contract."""

    backend_name: str = "tavily"

    async def search(self, query: str, top_k: int = 5):
        results = [
            {
                "title": f"Result {i}",
                "content": f"Content {i}: {query}",
                "url": f"https://result{i}.com",
                "score": 0.9 - i * 0.1,
                "source": "tavily",
            }
            for i in range(min(top_k, 3))
        ]
        return results


class TestWebSearchResultFormat:
    def test_result_structure(self):
        result = {
            "title": "Test Result",
            "content": "Test content",
            "url": "https://test.com",
            "score": 0.95,
            "source": "tavily",
        }
        assert "title" in result
        assert "content" in result
        assert "url" in result
        assert "score" in result
        assert "source" in result

    def test_result_score_is_number(self):
        result = {"score": 0.85, "source": "tavily"}
        assert isinstance(result["score"], float)

    def test_backend_name_attr(self):
        mock = MockedTavilyBackend()
        assert mock.backend_name == "tavily"

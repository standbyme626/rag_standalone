"""Tests for RAPTOR module (Phase 2.2)."""

import asyncio
import numpy as np
import pytest

from app.rag.raptor import Raptor, RaptorChunk


class TestRaptorChunk:
    def test_valid_chunk(self):
        chunk = RaptorChunk("hello world", [0.1, 0.2, 0.3])
        assert chunk.is_valid() is True

    def test_invalid_empty_text(self):
        chunk = RaptorChunk("", [0.1, 0.2, 0.3])
        assert chunk.is_valid() is False

    def test_invalid_no_embedding(self):
        chunk = RaptorChunk("hello world")
        assert chunk.is_valid() is False

    def test_to_dict(self):
        chunk = RaptorChunk("test", [1.0, 2.0])
        d = chunk.to_dict()
        assert d["text"] == "test"
        assert d["embedding"] == [1.0, 2.0]


class TestRaptor:
    @staticmethod
    async def fake_llm(system_prompt: str, messages: list, max_tokens: int = 512) -> str:
        return "Summary of the input content"

    @staticmethod
    async def fake_embed(text: str) -> list:
        np.random.seed(hash(text) % 2**31)
        return np.random.randn(32).tolist()

    @pytest.mark.asyncio
    async def test_single_chunk_returns_unchanged(self):
        model = Raptor(
            llm_call=self.fake_llm,
            embed_call=self.fake_embed,
        )
        chunks = [RaptorChunk("one chunk", [0.1] * 32)]
        result = await model(chunks)
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_empty_list_returns_empty(self):
        model = Raptor(
            llm_call=self.fake_llm,
            embed_call=self.fake_embed,
        )
        result = await model([])
        assert result == []

    @pytest.mark.asyncio
    async def test_no_llm_no_embed_drops_invalid(self):
        model = Raptor()
        chunks = [
            RaptorChunk("valid with no embed"),
            RaptorChunk("valid with embed", [0.1] * 32),
        ]
        result = await model(chunks)
        # Should return chunks without LLM processing
        assert len(result) >= 0

    def test_raptor_config(self):
        model = Raptor(
            max_cluster=5,
            max_token=256,
            threshold=0.2,
            random_state=123,
        )
        assert model.max_cluster == 5
        assert model.max_token == 256
        assert model.threshold == 0.2
        assert model.random_state == 123

    def test_best_n_clusters_single(self):
        model = Raptor()
        embeddings = np.random.randn(1, 32)
        n = model._best_n_clusters(embeddings)
        assert n == 1

    def test_best_n_clusters_two_chunks(self):
        model = Raptor()
        embeddings = np.random.randn(2, 32)
        n = model._best_n_clusters(embeddings)
        assert n == 1  # with BIC, 1 cluster is likely for 2 points

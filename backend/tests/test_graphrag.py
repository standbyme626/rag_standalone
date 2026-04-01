"""Tests for GraphRAG enhancements (Phase 3.2)."""

import networkx as nx
import pytest

from app.rag.graphrag import (
    Entity,
    Relation,
    GraphIndex,
    leiden_community_detection,
    CommunityReporter,
    entity_resolution_similar,
    _tokenize,
    _is_english,
    _jaccard_similarity,
    _edit_distance_similar,
    _similarity,
    _extract_json,
)


class TestTokenize:
    def test_english(self):
        tokens = _tokenize("diabetes treatment")
        assert "diabetes" in tokens
        assert "treatment" in tokens

    def test_chinese(self):
        tokens = _tokenize("糖尿病")
        assert len(tokens) == 3

    def test_mixed(self):
        tokens = _tokenize("糖尿病diabetes")
        assert len(tokens) > 0


class TestIsEnglish:
    def test_english_text(self):
        assert _is_english("Hello World") is True

    def test_chinese_text(self):
        assert _is_english("你好世界") is False


class TestSimilarity:
    def test_jaccard_identical(self):
        assert _jaccard_similarity("abc", "abc") == 1.0

    def test_jaccard_no_overlap(self):
        assert _jaccard_similarity("abc", "xyz") == 0.0

    def test_edit_distance_identical(self):
        assert _edit_distance_similar("hello", "hello") == 1.0

    def test_edit_distance_similar(self):
        assert _edit_distance_similar("hello", "helo") > 0.8

    def test_similarity_english(self):
        assert _similarity("hello", "hello") > 0.9

    def test_similarity_chinese(self):
        assert _similarity("糖尿病", "糖尿病") == 1.0


class TestExtractJson:
    def test_extract(self):
        text = 'Here is the result: {"key": "value"}'
        assert _extract_json(text) == '{"key": "value"}'

    def test_no_json(self):
        assert _extract_json("no json here") is None


class TestGraphIndex:
    def test_add_and_search(self):
        graph = GraphIndex()
        graph.add_entity(Entity("糖尿病", "Disease", "一种代谢疾病"))
        graph.add_entity(Entity("胰岛素", "Drug", "治疗糖尿病的药物"))
        graph.add_relation(Relation("糖尿病", "胰岛素", "治疗"))
        result = graph.search("糖尿病", top_k=2)
        assert len(result) >= 1
        assert result[0]["name"] == "糖尿病"

    def test_search_returns_two(self):
        graph = GraphIndex()
        graph.add_entity(Entity("A", "Type", "desc"))
        graph.add_entity(Entity("B", "Type", "desc"))
        graph.add_relation(Relation("A", "B", "connects"))
        result = graph.search("A B", top_k=5)
        assert len(result) == 2

    def test_n_hop(self):
        graph = GraphIndex()
        graph.add_entity(Entity("A", "T", "d"))
        graph.add_entity(Entity("B", "T", "d"))
        graph.add_entity(Entity("C", "T", "d"))
        graph.add_relation(Relation("A", "B", "l1"))
        graph.add_relation(Relation("B", "C", "l2"))
        neighbors = graph.n_hop_neighbors("B", n=1)
        assert "A" in neighbors
        assert "C" in neighbors
        assert "B" not in neighbors

    def test_serialize_deserialize(self):
        graph = GraphIndex()
        graph.add_entity(Entity("A", "Type", "desc A"))
        graph.add_relation(Relation("A", "B", "connects"))
        data = graph.to_dict()
        new_graph = GraphIndex.from_dict(data)
        assert len(new_graph.get_all_entities()) == len(graph.get_all_entities())

    def test_entity_merge_sep(self):
        graph = GraphIndex()
        graph.add_entity(Entity("糖尿病", "Disease", "一种代谢疾病"))
        graph.add_entity(Entity("糖尿病", "Condition", "慢性代谢性疾病"))
        node_data = graph._graph.nodes["糖尿病"]
        assert "[SEP]" in node_data.get("description", "")

    def test_entity_weight_sum(self):
        graph = GraphIndex()
        graph.add_entity(Entity("A", "Type", "Desc 1"))
        graph.add_entity(Entity("A", "Type", "Desc 2"))
        assert graph._graph.nodes["A"]["weight"] == 2.0

    def test_empty_graph(self):
        graph = GraphIndex()
        results = graph.search("nothing here", top_k=5)
        assert results == []


class TestCommunityDetection:
    def test_single_community(self):
        graph = nx.complete_graph(3)
        communities = leiden_community_detection(graph)
        assert len(communities) == 1

    def test_multi_community(self):
        graph = nx.Graph()
        graph.add_edges_from([(0, 1), (1, 2), (2, 0)])
        graph.add_edges_from([(3, 4), (4, 5), (5, 3)])
        communities = leiden_community_detection(graph)
        assert len(communities) >= 1

    def test_empty_graph(self):
        communities = leiden_community_detection(nx.Graph())
        assert communities == {}


class TestCommunityReporter:
    @pytest.mark.asyncio
    async def test_basic_report_no_llm(self):
        reporter = CommunityReporter()
        report = await reporter.generate_report(
            "Test Community",
            [{"name": "A", "description": "Entity A"}],
        )
        assert "Test Community" in report
        assert "Entity A" in report

    @pytest.mark.asyncio
    async def test_empty_report(self):
        reporter = CommunityReporter()
        report = await reporter.generate_report("Empty", [])
        assert "Entities (0)" in report


class TestEntityResolution:
    def test_identical_entities(self):
        result = entity_resolution_similar(["hello", "hello"])
        assert len(result) == 1
        assert "hello" in result[0]

    def test_similar_english(self):
        result = entity_resolution_similar(["hello", "helo", "world"])
        grouped = [list(g) for g in result]
        assert any("hello" in g for g in grouped)

    def test_distinct(self):
        result = entity_resolution_similar(["abc", "xyz", "123"])
        assert len(result) >= 2

    def test_chinese_similar(self):
        # Similar Chinese phrases should group
        result = entity_resolution_similar(["糖尿病治疗", "糖尿病疗法", "骨折"])
        grouped = [list(g) for g in result]
        # "糖尿病治疗" and "糖尿病疗法" share characters
        assert any(len(g) >= 2 for g in grouped)

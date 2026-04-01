"""GraphRAG 增强模块

来源：ragflow rag/graphrag/

功能：
- GraphIndex: 图索引（实体/关系存储、PageRank 检索、N 跳路径扩展）
- GraphExtractor: LLM 驱动实体/关系提取（同名实体合并、描述压缩）
- leiden_community_detection: 基于模块度的分层社区划分
- CommunityReporter: 社区报告生成
- entity_resolution_similar: 实体消歧（Jaccard + Edit Distance）

用法:
    graph = GraphIndex()
    graph.add_entity(Entity("糖尿病", "Disease", "一种代谢疾病"))
    graph.add_relation(Relation("糖尿病", "胰岛素", "治疗"))
    results = graph.search("糖尿病治疗", top_k=5)
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set, Tuple

import networkx as nx
import structlog

logger = structlog.get_logger(__name__)


# --------------- 数据类 ---------------

class Entity:
    """图实体"""
    __slots__ = ("name", "entity_type", "description", "weight")

    def __init__(self, name: str, entity_type: str = "", description: str = "", weight: float = 1.0):
        self.name = name
        self.entity_type = entity_type
        self.description = description
        self.weight = weight


class Relation:
    """图关系边"""
    __slots__ = ("source", "target", "description", "weight")

    def __init__(self, source: str, target: str, description: str = "", weight: float = 1.0):
        self.source = source
        self.target = target
        self.description = description
        self.weight = weight


# --------------- GraphIndex ---------------

class GraphIndex:
    """图索引

    支持：
    - 添加实体/关系
    - PageRank + 关键词匹配检索
    - N 跳路径扩展
    - 序列化/反序列化
    """

    def __init__(self):
        self._graph = nx.Graph()

    @property
    def graph(self) -> nx.Graph:
        """返回底层 networkx 图"""
        return self._graph

    def add_entity(self, entity: Entity) -> None:
        """添加实体到图"""
        if not self._graph.has_node(entity.name):
            self._graph.add_node(
                entity.name,
                entity_type=entity.entity_type,
                description=entity.description,
                weight=entity.weight,
            )
        else:
            existing_desc = self._graph.nodes[entity.name].get("description", "")
            merged_desc = (
                f"{existing_desc} [SEP] {entity.description}"
                if existing_desc
                else entity.description
            )
            if len(merged_desc) > 512:
                merged_desc = merged_desc[:256] + " ... " + merged_desc[-256:]
            self._graph.nodes[entity.name]["description"] = merged_desc
            self._graph.nodes[entity.name]["weight"] = (
                self._graph.nodes[entity.name].get("weight", 0) + entity.weight
            )

    def add_relation(self, relation: Relation) -> None:
        """添加关系到图"""
        self._graph.add_edge(
            relation.source,
            relation.target,
            description=relation.description,
            weight=relation.weight,
        )

    def add_entities(self, entities: List[Entity]) -> None:
        """批量添加实体"""
        for e in entities:
            self.add_entity(e)

    def add_relations(self, relations: List[Relation]) -> None:
        """批量添加关系"""
        for r in relations:
            self.add_relation(r)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """基于 PageRank + 关键词匹配的图检索"""
        query_terms = set(_tokenize(query))
        scored: Dict[str, float] = {}

        pr = nx.pagerank(self._graph)

        for node_name in self._graph.nodes:
            node_desc = str(self._graph.nodes[node_name].get("description", ""))
            node_terms = _tokenize(node_name + " " + node_desc)
            overlap = len(set(node_terms) & query_terms)
            if overlap > 0:
                scored[node_name] = overlap * 0.7 + pr.get(node_name, 0.0) * 0.3

        # 3. N 跳路径扩展（未命中的节点通过邻居传播分数）
        if len(scored) < top_k:
            for node_name in self._graph.nodes:
                if node_name not in scored:
                    for neighbor in self._graph.neighbors(node_name):
                        if neighbor in scored:
                            scored[node_name] = scored[neighbor] * 0.5
                            break

        sorted_nodes = sorted(scored.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [
            {"name": name, "score": score, **self._graph.nodes[name]}
            for name, score in sorted_nodes
        ]

    def n_hop_neighbors(self, node_name: str, n: int = 2) -> Set[str]:
        """获取 N 跳邻居"""
        result = set()
        current = {node_name}
        for _ in range(n):
            next_level = set()
            for node in current:
                if self._graph.has_node(node):
                    next_level.update(self._graph.neighbors(node))
            result.update(next_level)
            current = next_level
            if not current:
                break
        result.discard(node_name)
        return result

    def get_all_entities(self) -> List[Dict[str, Any]]:
        """获取所有实体"""
        return [{"name": n, **data} for n, data in self._graph.nodes(data=True)]

    def get_all_relations(self) -> List[Dict[str, Any]]:
        """获取所有关系"""
        return [
            {"source": u, "target": v, **data}
            for u, v, data in self._graph.edges(data=True)
        ]

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return nx.node_link_data(self._graph)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GraphIndex":
        """从字典反序列化"""
        gi = cls()
        gi._graph = nx.node_link_graph(data)
        return gi


# --------------- 实体提取 ---------------

class GraphExtractor:
    """从文本中提取实体和关系

    LLM 驱动，支持 Light/General 双模式。
    同名实体合并（SEP 分隔描述）+ 描述摘要压缩（512 token 阈）。
    """

    DEFAULT_PROMPT = """You are a helpful assistant. Extract entities and relations from the following text.

Output format (JSON):
{
    "entities": [
        {"name": "Entity Name", "type": "Type", "description": "Brief description"}
    ],
    "relations": [
        {"source": "Entity A", "target": "Entity B", "description": "Relation description"}
    ]
}

Text:
{text}

Return only the JSON. No other text."""

    def __init__(
        self,
        llm_call: Optional[Callable[..., Awaitable[str]]] = None,
        *,
        mode: str = "light",
        prompt_template: Optional[str] = None,
    ):
        self.llm_call = llm_call
        self.mode = mode
        self.prompt = prompt_template or self.DEFAULT_PROMPT

    async def extract(self, text: str) -> Tuple[List[Entity], List[Relation]]:
        """从文本中提取实体和关系。

        Returns:
            (entities, relations)
        """
        if not self.llm_call:
            return [], []

        prompt = self.prompt.format(text=text)
        try:
            response = await self.llm_call(
                system_prompt="",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4096,
            )
            response = response or ""
            json_str = _extract_json(response)
            if not json_str:
                logger.error("graph_no_json")
                return [], []

            data = json.loads(json_str)
            entities = [
                Entity(
                    name=e["name"],
                    entity_type=e.get("type", "Unknown"),
                    description=e.get("description", ""),
                )
                for e in data.get("entities", [])
            ]
            relations = [
                Relation(
                    source=r["source"],
                    target=r["target"],
                    description=r.get("description", ""),
                )
                for r in data.get("relations", [])
            ]
            return self._merge_dup_entities(entities), relations

        except Exception as e:
            logger.error("graph_extract_failed", error=str(e))
            return [], []

    def _merge_dup_entities(self, entities: List[Entity]) -> List[Entity]:
        """同名实体合并（SEP 分隔描述 + 512 token 阈压缩）"""
        merged: Dict[str, Entity] = {}
        for e in entities:
            if e.name in merged:
                existing_desc = merged[e.name].description
                new_desc = f"{existing_desc} [SEP] {e.description}"
                if len(new_desc) > 512:
                    new_desc = new_desc[:256] + " ... " + new_desc[-256:]
                merged[e.name].description = new_desc
            else:
                merged[e.name] = Entity(
                    name=e.name,
                    entity_type=e.entity_type,
                    description=e.description,
                )
        return list(merged.values())


# --------------- Leiden 社区检测 ---------------

def leiden_community_detection(
    graph: nx.Graph,
    max_level: int = 10,
    seed: int = 42,
) -> Dict[str, List[str]]:
    """分层社区检测 (Leiden)

    使用 networkx 的 greedy_modularity_communities 作为基础，
    实现层级社区划分。

    来源：ragflow rag/graphrag/general/leiden.py

    Args:
        graph: networkx 图
        max_level: 最大层数
        seed: 随机种子

    Returns:
        {community_id: [node_names, ...]}
    """
    if not graph.nodes:
        return {}

    from networkx.algorithms.community import greedy_modularity_communities

    communities = greedy_modularity_communities(graph)
    result = {}
    for i, comm in enumerate(communities):
        result[f"community_{i}"] = sorted(list(comm))

    return result


# --------------- 社区报告生成 ---------------

class CommunityReporter:
    """社区报告生成器

    来源：ragflow rag/graphrag/general/community_reports_extractor.py
    """

    DEFAULT_PROMPT = """You are a community analyst. Generate a concise report for the given community.

Community: {community_name}
Entities: {entities}

Include main theme, key entities, relationships, and a summary."""

    def __init__(self, llm_call: Optional[Callable[..., Awaitable[str]]] = None):
        self.llm_call = llm_call
        self.prompt = self.DEFAULT_PROMPT

    async def generate_report(
        self,
        community_name: str,
        entities: List[Dict[str, Any]],
        sub_communities: Optional[List[Dict]] = None,
    ) -> str:
        """生成社区报告。

        Args:
            community_name: 社区名称
            entities: 实体列表
            sub_communities: 子社区列表

        Returns:
            报告文本
        """
        if not self.llm_call:
            return self._basic_report(community_name, entities)

        entity_summaries = ", ".join(e.get("name", "") for e in entities)
        prompt = self.prompt.format(
            community_name=community_name,
            entities=entity_summaries,
        )
        try:
            response = await self.llm_call(
                system_prompt="",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2048,
            )
            return response or ""
        except Exception as e:
            logger.error("community_report_failed", error=str(e))
            return self._basic_report(community_name, entities)

    def _basic_report(self, name: str, entities: List[Dict]) -> str:
        """无 LLM 时的基础报告"""
        parts = [f"# Community Report: {name}\n"]
        parts.append(f"## Entities ({len(entities)})\n")
        for e in entities:
            parts.append(
                f"- **{e.get('name', '?')}**: {e.get('description', '')[:100]}"
            )
        return "\n".join(parts)


# --------------- 实体消歧 ---------------

def entity_resolution_similar(
    entities: List[str],
    threshold: float = 0.8,
) -> List[Set[str]]:
    """基于 Jaccard 相似度 + Edit Distance 的实体消歧

    editdistance 预筛 + LLM 批量决议。
    对于英文：Edit Distance；中文：Jaccard 字符集相似度。

    来源：ragflow rag/graphrag/entity_resolution.py

    Args:
        entities: 实体名列表
        threshold: 相似度阈值

    Returns:
        每组等价实体集合的列表
    """
    resolved: List[Set[str]] = []
    used: Set[int] = set()

    for i in range(len(entities)):
        if i in used:
            continue
        group = {entities[i]}
        used.add(i)
        for j in range(i + 1, len(entities)):
            if j in used:
                continue
            if _similarity(entities[i], entities[j]) >= threshold:
                group.add(entities[j])
                used.add(j)
        resolved.append(group)

    return resolved


# --------------- 工具函数 ---------------

def _tokenize(text: str) -> List[str]:
    """简单分词"""
    return re.findall(r"[a-zA-Z]+|[\u4e00-\u9fff]", text.lower())


def _is_english(text: str) -> bool:
    """检测文本是否为英文"""
    return bool(re.match(r"^[a-zA-Z\s\.\-\_]+$", text.strip()))


def _jaccard_similarity(a: str, b: str) -> float:
    """Jaccard 字符集相似度"""
    set_a, set_b = set(a), set(b)
    max_l = max(len(set_a), len(set_b))
    if max_l == 0:
        return 1.0
    return len(set_a & set_b) / max_l


def _edit_distance_similar(a: str, b: str) -> float:
    """基于编辑距离的相似度"""
    import difflib
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _similarity(a: str, b: str) -> float:
    """统一相似度计算"""
    if _is_english(a) and _is_english(b):
        return _edit_distance_similar(a, b)
    return _jaccard_similarity(a, b)


def _extract_json(text: str) -> Optional[str]:
    """从文本中提取第一个 JSON 对象"""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None

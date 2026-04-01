# 可吸收内容 + 优先级 (Absorption Roadmap)

> **目标**：基于收益/投入比排序，从高到低吸收 UltraRAG 的核心能力，提升 rag_standalone 的检索质量与评估闭环。

---

## 目录

- [🔴 高收益，低投入（强烈推荐）](#-高收益低投入强烈推荐)
  - [1. Cross-Encoder Reranker](#1-cross-encoder-reranker)
  - [2. Web Search Retriever](#2-web-search-retriever)
  - [3. RAGAS 评估集成](#3-ragas-评估集成)
- [🟡 高收益，中等投入](#-高中等投入)
  - [4. Sub-Query Decomposition](#4-sub-query-decomposition)
  - [5. Corrective RAG（检索反馈回路）](#5-corrective-rag检索反馈回路)
- [🟢 中等收益，较高投入](#-中等收益较高投入)
  - [6. IR 评估指标](#6-ir-评估指标)
  - [7. YAML Pipeline 编排](#7-yaml-pipeline-编排)
- [⚫ 不推荐吸收](#-不推荐吸收)
- [📊 深度横向对比](#-深度横向对比)
- [🗺️ 推荐实施路线](#️-推荐实施路线)
- [修改记录](#修改记录)

---

## 🔴 高收益，低投入（强烈推荐）

### 1. Cross-Encoder Reranker

- **来源**: UltraRAG `servers/reranker/` (SBERT CrossEncoder backend)
- **投入**: ~2天
- **收益**: +3-5% hit rate
- **做法**: 把 `QwenReranker`（相似度差值 rerank）替换为 `sentence_transformers.CrossEncoder`
- **可直接参考 UltraRAG 的实现**：
  ```python
  from sentence_transformers import CrossEncoder
  reranker = CrossEncoder("BAAI/bge-reranker-base")
  scores = reranker.predict([(query, doc) for doc in candidates])
  ```
- **备注**: rag_standalone 已有 SiliconFlow 接入，可改成本地 SBERT CrossEncoder，或保留 SiliconFlow 作为备选。
- **状态**: 🔄 进行中

### 2. Web Search Retriever

- **来源**: UltraRAG `servers/retriever/` (Tavily/Exa backend)
- **投入**: ~1天
- **收益**: 覆盖实时问题，拓展知识边界
- **做法**: 新增 `WebSearchRetriever`，作为 Hybrid Search 的第三路
- **接入 Tavily 作为 fallback / 补充**：
  ```python
  from tavily import TavilyClient
  tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
  results = tavily.search(query, max_results=5)
  ```
- **备注**: rag_standalone 的 `HybridRetriever` 已经是多路检索，扩展成本极低。
- **状态**: ⬜ 未开始

### 3. RAGAS 评估集成

- **来源**: UltraRAG `evaluation/` + 主流 RAGAS framework
- **投入**: ~3天
- **收益**: 量化质量，支撑迭代决策（最大缺口）
- **做法**: 新增 `app/rag/evaluation.py`，集成 RAGAS
  ```python
  from ragas import evaluate
  from ragas.metrics import (
      faithfulness, answer_relevancy, context_precision, context_recall
  )
  from ragas.dataset import Dataset
  
  def evaluate_rag(query: str, answer: str, contexts: list[str]) -> dict:
      ds = Dataset([{"user_input": query, "response": answer, "contexts": contexts}])
      result = evaluate(ds, metrics=[faithfulness, answer_relevancy, context_precision])
      return result
  ```
- **备注**: 可配合现有 Langfuse trace 数据，自动计算 RAGAS 分数。
- **状态**: ⬜ 未开始

---

## 🟡 高收益，中等投入

### 4. Sub-Query Decomposition

- **来源**: UltraRAG prompt + generation (loop 中做 query rewrite)
- **投入**: ~1周
- **收益**: 多跳问题（"民法典第205条和206条有什么关系？"）质量大幅提升
- **做法**: 扩展现有 `query_rewrite`，在 rewrite prompt 中要求生成子问题
  ```python
  # 新增 SubQueryDecomposer
  SUBQUERY_PROMPT = """将复杂问题分解为简单子问题：
  问题: {query}
  要求：每个子问题独立可检索，用分号分隔"""
  
  def decompose(query: str) -> list[str]:
      subqs = llm.predict(SUBQUERY_PROMPT, query=query)
      return [q.strip() for q in subqs.split(";")]
  
  # 检索：对每个子问题分别检索，merge 结果
  sub_results = [hybrid_search(subq) for subq in decompose(query)]
  merged = merge_by_rrf(sub_results)
  ```
- **状态**: ⬜ 未开始

### 5. Corrective RAG（检索反馈回路）

- **来源**: UltraRAG loop 控制流 + Self-RAG 思想
- **投入**: ~1周
- **收益**: 低分检索自动重试，减少幻觉
- **做法**: 在现有 pipeline 加一层"检索质量评估"
  ```python
  async def corrective_rag(query: str) -> list[Document]:
      for attempt in range(3):
          results = await hybrid_search(query)
          relevance = await judge_relevance(query, results)
          
          if relevance.avg_score > 0.7:
              return results
          if attempt < 2:
              # 低分 → 重写 query 再试
              query = await query_rewrite(query, low_score_context=results)
      return results
  
  # judge_relevance 用 LLM 打分
  RELEVANCE_PROMPT = """Query: {query}\n\nDoc: {doc}\n\nRelevance score 1-5:"""
  ```
- **状态**: ⬜ 未开始

---

## 🟢 中等收益，较高投入

### 6. IR 评估指标

- **来源**: UltraRAG `evaluation/` (MAP/MRR/NDCG/Recall)
- **投入**: ~3天
- **收益**: 离线批量评估，离线优化时必须有
- **做法**: 新增 `app/rag/eval/metrics.py`
  ```python
  def compute_map(results: list[list[str]], qrels: dict) -> float:
      """Mean Average Precision"""
      
  def compute_ndcg(results: list[list[str]], qrels: dict, k: int = 10) -> float:
      """NDCG@K"""
      
  def compute_mrr(results: list[list[str]], qrels: dict) -> float:
      """Mean Reciprocal Rank"""
  ```
- **备注**: 配合现有 pytest，写批量评估测试用例。
- **状态**: ⬜ 未开始

### 7. YAML Pipeline 编排

- **来源**: UltraRAG pipeline 编排思想
- **投入**: ~2-3周
- **收益**: pipeline 配置化，支持动态调整（生产调参神器）
- **做法**: 引入 YAML 配置 + 解释器，类似 UltraRAG 但保持 Python 优先
  ```yaml
  # config/pipelines/legal_consultation.yaml
  pipeline:
    - semantic_cache.lookup
    - intent.classify
    - branch:
        medical: [hyde.generate, retriever.search, rerank]
        legal: [retriever.search_legal, rerank]
    - loop:
        condition: "score < 0.6"
        times: 2
        steps: [query_rewrite, retriever.search]
    - evaluation.score
  ```
- **状态**: ⬜ 未开始

---

## ⚫ 不推荐吸收

| 内容 | 原因 |
|------|------|
| MCP Server 架构 | 与现有 FastAPI 架构冲突，改造成本 > 收益 |
| Visual Pipeline Builder | 独立项目，优先级低 |
| Deep Research Loop (3+ 轮) | LLM 调用成本高，医疗/法律场景收益有限 |

---

## 🗺️ 推荐实施路线

### Phase 1（1-2周，高优先级）:
  ✅ Cross-Encoder Reranker          (2天) — 立即提升检索质量  
  ✅ Web Search Retriever           (1天) — 扩展知识边界  
  ✅ RAGAS Evaluation               (3天) — 建立评估闭环

### Phase 2（2-3周，中优先级）:
  🟡 Sub-Query Decomposition        (1周) — 多跳问题  
  🟡 Corrective RAG                (1周) — 低分自动重试

### Phase 3（长期）:
  🟢 IR Metrics + 离线评估          (3天) — 批量优化  
  🟢 YAML Pipeline 编排             (2-3周) — 生产调参

> 最快见效的是 Phase 1：Cross-Encoder + Web Search + RAGAS，三周内可完成，立即建立质量可量化 + 检索质量提升的闭环。

---

## 分析对比

### 1. 提供内容的真实性分析

经过对 UltraRAG 源码的核实，用户提供的内容**基本属实**：

| 项目 | 核实结果 | 证据 |
|------|----------|------|
| Cross-Encoder Reranker | ✅ 属实 | UltraRAG `reranker.py` 明确支持 `sentence_transformers` 后端 |
| Web Search Retriever | ✅ 属实 | UltraRAG 有 `tavily_backend.py`、`exa_backend.py`、`zhipuai_backend.py` |
| RAGAS 评估 | ⚠️ 部分属实 | UltraRAG 使用 `pytrec_eval` 进行 IR 评估，但未集成 RAGAS |
| Sub-Query Decomposition | ⚠️ 待确认 | UltraRAG 支持 `query_rewrite`，但未明确实现子问题分解 |
| Corrective RAG | ⚠️ 待确认 | UltraRAG 支持 loop 控制流，但未明确实现 corrective RAG |
| IR 评估指标 | ✅ 属实 | UltraRAG 有完整的 MAP/MRR/NDCG/Recall 实现 |
| YAML Pipeline 编排 | ✅ 属实 | UltraRAG 的核心特性就是 YAML 配置化 pipeline |

### 2. rag_standalone vs UltraRAG 深度对比

#### 架构差异

| 维度 | rag_standalone | UltraRAG |
|------|----------------|----------|
| **架构模式** | 单体 FastAPI 应用 | 微服务 MCP 架构（每个组件是独立服务器） |
| **编排方式** | 硬编码 Python 代码 | YAML 配置驱动，支持条件分支、循环 |
| **扩展性** | 需要修改代码 | 通过注册新 Tool 无缝集成 |
| **部署** | 单一 Docker 容器 | 多容器编排（每个 MCP Server 独立部署） |

#### 功能差异

| 功能 | rag_standalone | UltraRAG |
|------|----------------|----------|
| **检索器** | Hybrid (Vector + BM25) | Hybrid + Web Search (Tavily/Exa) |
| **重排器** | QwenReranker (CausalLM) | 多后端支持 (Infinity, CrossEncoder, OpenAI) |
| **缓存** | 语义缓存 + 精确缓存 | 语义缓存 + 精确缓存 |
| **评估** | 自定义评估门控 | 完整的 IR 指标 (MAP/MRR/NDCG) + 生成质量评估 |
| **意图分类** | 医疗领域意图分类 | 通用意图分类 |
| **pipeline 配置** | 硬编码 | YAML 配置化 |
| **UI** | 无可视化 | 内置 Pipeline Builder IDE |

#### 技术栈差异

| 技术 | rag_standalone | UltraRAG |
|------|----------------|----------|
| **后端框架** | FastAPI | FastMCP (基于 MCP 协议) |
| **LLM 集成** | LangChain, SiliconFlow | OpenAI API, vLLM |
| **向量数据库** | Milvus | Milvus, FAISS |
| **评估工具** | 自定义 | pytrec_eval, RAGAS (可选) |
| **监控** | Prometheus + Grafana | 内置日志 |

### 3. 吸收建议

#### ✅ 值得吸收的功能

1. **Cross-Encoder Reranker** - 立即提升检索质量
2. **Web Search Retriever** - 扩展知识边界，覆盖实时问题
3. **RAGAS 评估** - 建立量化评估闭环（当前最大缺口）
4. **IR 评估指标** - 离线批量评估，支撑优化决策

#### ⚠️ 选择性吸收

1. **Sub-Query Decomposition** - 医疗场景下多跳问题较多，值得尝试
2. **Corrective RAG** - 可提升低质量检索的容错性
3. **YAML Pipeline 编排** - 长期收益大，但改造成本高

#### ❌ 不建议吸收

1. **MCP Server 架构** - 与现有 FastAPI 架构冲突，改造成本 > 收益
2. **Visual Pipeline Builder** - 独立项目，优先级低
3. **Deep Research Loop** - LLM 调用成本高，医疗场景收益有限

---

## 📊 深度横向对比

### 1. 架构哲学对比

| 维度 | rag_standalone (当前) | UltraRAG | 主流 RAG 框架 (LangChain/LlamaIndex/Haystack) |
|------|----------------------|----------|-----------------------------------------------|
| **设计理念** | 领域驱动的垂直解决方案 | 通用、可复用的 RAG 框架 | 通用库，提供组件但不强制架构 |
| **架构模式** | 单体 FastAPI 应用，模块化设计 | 微服务 MCP 架构，每个组件是独立服务器 | 库/框架模式，由用户组装 |
| **编排方式** | 硬编码 Python 代码，流程固定 | YAML 配置驱动，支持顺序/分支/循环 | 代码编排，提供 Workflow/DAG 抽象 |
| **扩展机制** | 修改代码、插件系统 (plugins/) | 注册新 MCP Server (Tool) | 继承基类、实现接口 |
| **学习曲线** | 中等（需要理解医疗领域逻辑） | 低（YAML 配置即可） | 中等（需要理解框架抽象） |
| **生产就绪度** | 高（已部署生产环境） | 中（研究/原型设计导向） | 高（社区活跃，生产案例多） |

### 2. 技术栈深度对比

| 组件 | rag_standalone | UltraRAG | LangChain RAG | LlamaIndex | Haystack |
|------|----------------|----------|---------------|------------|----------|
| **后端框架** | FastAPI (异步) | FastMCP (基于 MCP 协议) | 无框架限制 | 无框架限制 | FastAPI/Flask |
| **LLM 集成** | LangChain + SiliconFlow | OpenAI API, vLLM | LangChain (多 provider) | LlamaIndex (多 provider) | Transformers, OpenAI |
| **向量数据库** | Milvus (生产级) | Milvus, FAISS | Chroma, Pinecone, Weaviate, etc. | 多种向量数据库 | Milvus, Pinecone, etc. |
| **词法检索** | BM25 (Jieba分词) | BM25s | BM25 via community | BM25 via community | BM25 |
| **重排器** | QwenReranker (CausalLM) | Infinity, CrossEncoder, OpenAI Cohere | Cohere, CrossEncoder | CrossEncoder, Cohere | Cohere, CrossEncoder |
| **缓存** | 语义缓存 (Redis+Milvus) + 精确缓存 | 语义缓存 | 内置缓存抽象 | 可集成外部缓存 | 可集成 Redis |
| **评估** | 自定义评估门控 + 测试数据 | pytrec_eval (IR指标) | RAGAS, 自定义 | RAGAS, 自定义 | 评估管道 |
| **监控** | Prometheus + Grafana + Langfuse | 内置日志 | LangSmith, Langfuse | 无内置 | 无内置 |
| **部署** | Docker Compose (单体+依赖) | Docker (每个MCP Server独立) | 任意部署 | 任意部署 | Docker, 任意 |

### 3. 功能特性详细对比

#### 3.1 检索能力对比

| 功能 | rag_standalone | UltraRAG | 优势分析 |
|------|----------------|----------|----------|
| **向量检索** | ✅ Milvus HNSW索引 | ✅ Milvus/FAISS | rag_standalone 生产级配置更成熟 |
| **词法检索** | ✅ BM25 (Jieba分词) | ✅ BM25s | 功能相当，rag_standalone 医疗分词优化 |
| **混合检索** | ✅ RRF 融合 | ✅ 支持多路融合 | rag_standalone 有自适应权重调整 |
| **Web 搜索** | ❌ 无 | ✅ Tavily, Exa, ZhiPuAI | **UltraRAG 优势**：实时信息获取 |
| **HyDE** | ✅ 有实现 | ✅ 通过 Prompt 实现 | 功能相当 |
| **查询重写** | ✅ 本地 SLM 重写 | ✅ Prompt 重写 | rag_standalone 使用本地模型，成本更低 |
| **意图分类** | ✅ 医疗专用 (0.6B模型) | ✅ 通用分类 | rag_standalone 领域针对性强 |
| **科室过滤** | ✅ 医疗专用门控 | ❌ 无 | rag_standalone 领域特性 |
| **安全拦截** | ✅ 药物相互作用检查 | ❌ 无 | rag_standalone 领域特性 |

#### 3.2 重排能力对比

| 功能 | rag_standalone | UltraRAG | 优势分析 |
|------|----------------|----------|----------|
| **后端多样性** | 2种 (本地QwenReranker, SiliconFlow云端) | 3种 (Infinity, CrossEncoder, OpenAI) | **UltraRAG 优势**：更多选择 |
| **CrossEncoder** | ❌ 无 | ✅ sentence_transformers | **UltraRAG 优势**：更准确的点对点评分 |
| **云端API** | ✅ SiliconFlow | ✅ OpenAI Cohere | rag_standalone 有国内API |
| **自适应跳过** | ✅ 高分时跳过重排 | ❌ 无 | rag_standalone 性能优化 |
| **批量处理** | ✅ 支持 | ✅ 支持 | 功能相当 |

#### 3.3 生成能力对比

| 功能 | rag_standalone | UltraRAG | 优势分析 |
|------|----------------|----------|----------|
| **LLM 后端** | LangChain (多provider) + SiliconFlow | OpenAI API, vLLM, 本地模型 | rag_standalone 集成更灵活 |
| **本地模型** | ✅ 0.6B 模型 (意图/摘要) | ✅ vLLM 本地部署 | rag_standalone 有轻量级本地模型 |
| **流式输出** | ✅ FastAPI StreamingResponse | ✅ 支持 | 功能相当 |
| **多轮对话** | ✅ 通过 API 实现 | ✅ 有专门工具 | UltraRAG 有原生支持 |

#### 3.4 评估能力对比

| 功能 | rag_standalone | UltraRAG | 优势分析 |
|------|----------------|----------|----------|
| **IR 指标** | ⚠️ 定义但未实现 | ✅ MAP, MRR, NDCG, Recall | **UltraRAG 优势**：完整 IR 评估 |
| **生成评估** | ✅ 自定义 (Faithfulness, Citation Coverage) | ✅ 通过自定义实现 | rag_standalone 领域针对性强 |
| **RAGAS** | ❌ 无 | ⚠️ 可集成但未默认 | 均需额外集成 |
| **基准测试** | ✅ 医疗领域测试数据 | ✅ 支持标准基准 (HotpotQA等) | UltraRAG 通用性强，rag_standalone 领域深 |
| **A/B 测试** | ❌ 无 | ❌ 无 | 均需外部实现 |

#### 3.5 缓存能力对比

| 功能 | rag_standalone | UltraRAG | 优势分析 |
|------|----------------|----------|----------|
| **语义缓存** | ✅ Redis + Milvus 实现 | ✅ 支持 | rag_standalone 实现更完整 |
| **精确缓存** | ✅ Redis MD5 key | ✅ 支持 | 功能相当 |
| **缓存验证** | ✅ 二次校验 (term overlap + rerank score) | ❌ 简单验证 | **rag_standalone 优势**：防误命中 |
| **缓存门控** | ✅ 质量门控写入 | ❌ 无 | **rag_standalone 优势**：防低质量污染 |
| **压缩存储** | ✅ ZSTD 压缩 | ❌ 无 | **rag_standalone 优势**：节省存储 |

### 4. 扩展性与维护性对比

| 维度 | rag_standalone | UltraRAG | 说明 |
|------|----------------|----------|------|
| **添加新检索器** | 需修改代码，实现接口 | 新建 MCP Server，注册 Tool | UltraRAG 更灵活 |
| **添加新重排器** | 在 `reranker.py` 添加后端 | 新建或扩展现有 Server | UltraRAG 更模块化 |
| **添加新评估指标** | 修改 `evaluation_gate.py` | 新建 Evaluation Server | UltraRAG 更解耦 |
| **配置管理** | 环境变量 + Settings 类 | YAML 文件 + 参数文件 | 各有优劣 |
| **代码复杂度** | 高（业务逻辑复杂） | 中（框架通用） | rag_standalone 领域逻辑多 |
| **测试覆盖** | 有 pytest 测试 | 示例和基准测试 | rag_standalone 生产级测试 |
| **文档质量** | 内部文档 | 完整公开文档 | UltraRAG 文档更系统 |
| **社区支持** | 无 | OpenBMB 社区 | UltraRAG 有社区支持 |

### 5. 部署与运维对比

| 维度 | rag_standalone | UltraRAG | 主流 RAG 框架 |
|------|----------------|----------|---------------|
| **部署复杂度** | 中等 (Docker Compose) | 高 (多服务编排) | 低-中 (库集成) |
| **资源需求** | 单节点，GPU 推荐 | 多节点，每个 Server 独立 | 取决于实现 |
| **可观测性** | ⭐⭐⭐⭐⭐ (Prometheus+Grafana+Langfuse) | ⭐⭐ (日志) | 取决于集成 |
| **水平扩展** | 有限（单体应用） | 容易（每个 Server 独立扩展） | 取决于架构 |
| **监控告警** | ✅ 完整监控栈 | ❌ 基础日志 | 取决于集成 |
| **CI/CD** | 有 Docker 构建 | 有 Docker 构建 | 取决于项目 |
| **备份恢复** | PostgreSQL + Redis + Milvus | 每个服务独立 | 取决于实现 |

### 6. 与主流 RAG 框架横向对比

| 特性 | rag_standalone | UltraRAG | LangChain RAG | LlamaIndex | Haystack |
|------|----------------|----------|---------------|------------|----------|
| **定位** | 垂直领域解决方案 | 研究/原型框架 | 通用 LLM 应用框架 | 索引和检索框架 | 生产级 RAG 框架 |
| **架构** | 单体应用 | 微服务 MCP | 库 | 库 | 框架+库 |
| **学习曲线** | 中 | 低 | 中-高 | 中 | 中 |
| **生产就绪** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **社区生态** | 无 | 小 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **文档质量** | 内部 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **评估工具** | 自定义 | pytrec_eval | RAGAS 集成 | RAGAS 集成 | 评估管道 |
| **部署难度** | 中 | 高 | 低 | 低 | 中 |
| **扩展性** | 中 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **企业特性** | ⭐⭐⭐⭐⭐ (领域特化) | ⭐⭐ (研究导向) | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **成本控制** | ✅ 本地模型优化 | ⚠️ 多服务开销 | 取决于使用 | 取决于使用 | 取决于使用 |
| **实时性** | ❌ 离线检索 | ✅ Web 搜索 | 取决于集成 | 取决于集成 | 取决于集成 |

### 7. 综合评分 (满分10分)

| 维度 | rag_standalone | UltraRAG | 说明 |
|------|----------------|----------|------|
| **架构先进性** | 6 | 9 | UltraRAG 的 MCP 架构更前沿 |
| **生产就绪度** | 9 | 6 | rag_standalone 已生产部署 |
| **功能完整性** | 8 | 8 | 各有侧重 |
| **领域针对性** | 9 | 5 | rag_standalone 医疗特化 |
| **扩展灵活性** | 6 | 9 | UltraRAG 模块化更佳 |
| **评估体系** | 7 | 8 | UltraRAG IR 指标更完整 |
| **部署运维** | 8 | 6 | rag_standalone 运维体系成熟 |
| **社区生态** | 3 | 6 | UltraRAG 有 OpenBMB 社区 |
| **学习成本** | 6 | 8 | UltraRAG YAML 配置更简单 |
| **总分** | 6.2 | 7.1 | |

### 8. 差距分析与吸收策略

#### 核心差距：
1. **架构灵活性差距**：UltraRAG 的 MCP 微服务架构更灵活，但改造成本高
2. **评估体系差距**：缺乏标准化 IR 指标评估（MAP/MRR/NDCG）
3. **功能缺失**：无 Web 搜索、RAGAS 评估、子问题分解、纠正性 RAG
4. **工具链差距**：无 Pipeline Builder UI，无法可视化编排

#### 吸收优先级：
1. **立即吸收**：Cross-Encoder Reranker、Web Search、RAGAS 评估、IR 指标
2. **选择性吸收**：Sub-Query Decomposition、Corrective RAG
3. **长期考虑**：YAML Pipeline 编排（如需多 Agent 协作）
4. **不建议吸收**：MCP 架构改造、Visual Pipeline Builder

#### 实施建议：
- **保持现有架构**：FastAPI 单体架构在生产环境中更简单可靠
- **增强功能模块**：通过新增模块吸收 UltraRAG 的优秀功能
- **渐进式改进**：先易后难，从 Cross-Encoder 和 Web 搜索开始
- **评估驱动**：建立 RAGAS + IR 指标的量化评估体系

---

## 修改记录

| 日期 | 修改项目 | 修改人 | 备注 |
|------|----------|--------|------|
| 2026-03-31 | 创建文档 | opencode | 核实内容并初始化 |
| 2026-03-31 | Cross-Encoder Reranker | opencode | 标记为进行中，开始分析实现 |
| 2026-03-31 | 添加分析对比部分 | opencode | 深度分析 rag_standalone vs UltraRAG 架构差异 |
| 2026-03-31 | 添加深度横向对比章节 | opencode | 详细对比架构、技术栈、功能、部署，并与主流RAG框架对比 |
| | | | |

---

*最后更新：2026-03-31*
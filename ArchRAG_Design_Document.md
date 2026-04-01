# ArchRAG - 超级 RAG 框架设计文档

> **目标**：融合 rag_standalone、UltraRAG、ragflow 三大系统的全部优点，构建一个**通用性强、可插拔专属能力**的超级 RAG 框架。

---

## 目录

- [1. 架构设计](#1-架构设计)
- [2. 核心模块](#2-核心模块)
- [3. 数据处理能力](#3-数据处理能力)
- [4. 插件系统](#4-插件系统)
- [5. MCP 协议集成](#5-mcp-协议集成)
- [6. 领域切换](#6-领域切换)
- [7. 三系统对比](#7-三系统对比)
- [8. 实施路线](#8-实施路线)
- [9. 预期效果](#9-预期效果)

---

## 1. 架构设计

### 1.1 核心理念

```
┌─────────────────────────────────────────────────────────────────┐
│                        ArchRAG 架构                              │
│                                                                 │
│    ┌─────────────────────────────────────────────────────────┐ │
│    │                    通用 RAG 内核                         │ │
│    │  (领域无关，所有能力均可插拔替换)                         │ │
│    └─────────────────────────────────────────────────────────┘ │
│                              │                                   │
│          ┌───────────────────┼───────────────────┐             │
│          │                   │                   │             │
│          ▼                   ▼                   ▼             │
│    ┌──────────┐        ┌──────────┐        ┌──────────┐        │
│    │ 插件槽位 │        │ 插件槽位 │        │ 插件槽位 │        │
│    │ Intent   │        │ Retrieval│        │ Rerank   │        │
│    └──────────┘        └──────────┘        └──────────┘        │
│          │                   │                   │             │
│          ▼                   ▼                   ▼             │
│    ┌──────────┐        ┌──────────┐        ┌──────────┐        │
│    │医疗意图  │        │向量+BM25 │        │CrossEnc  │        │
│    │分类插件  │        │+知识图谱  │        │重排插件  │        │
│    └──────────┘        └──────────┘        └──────────┘        │
│                                                                 │
│    ┌──────────┐        ┌──────────┐        ┌──────────┐        │
│    │ 插件槽位 │        │ 插件槽位 │        │ 插件槽位 │        │
│    │ Parser   │        │ Security │        │ Eval     │        │
│    └──────────┘        └──────────┘        └──────────┘        │
│          │                   │                   │             │
│          ▼                   ▼                   ▼             │
│    ┌──────────┐        ┌──────────┐        ┌──────────┐        │
│    │DeepDoc   │        │领域规则  │        │RAGAS+IR  │        │
│    │文档解析  │        │安全拦截  │        │评估插件  │        │
│    └──────────┘        └──────────┘        └──────────┘        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 设计原则

| 原则 | 说明 |
|------|------|
| **通用内核** | 领域无关的核心 RAG 能力 |
| **插件插槽** | 7大插槽支持热插拔 |
| **MCP 协议** | 远程/本地插件统一协议 |
| **零代码切换** | 一行配置切换领域 |
| **性能优先** | 本地优先，按需远程 |

---

## 2. 核心模块

### 2.1 六大插件槽位

| 槽位 | 通用实现 | 专属插件示例 | 插槽接口 |
|------|----------|--------------|----------|
| **IntentClassifier** | 通用分类 | 医疗意图、金融意图、法律意图 | `classify(query) → Intent` |
| **Retriever** | Vector+BM25 | 知识图谱、Web搜索、领域检索 | `retrieve(query, top_k)` |
| **Reranker** | CrossEncoder | 领域重排、质量重排 | `rerank(query, docs)` |
| **Parser** | 基础文本 | DeepDoc、OCR、表格识别 | `parse(file) → chunks` |
| **Security** | 无限制 | 医疗安全、金融合规、内容审核 | `check(content) → bool` |
| **Evaluator** | 基础评估 | RAGAS、医疗QA、金融指标 | `evaluate(contexts, answer)` |

### 2.2 插件接口定义

```python
class IntentClassifierPlugin(ABC):
    @abstractmethod
    def classify(self, query: str) -> str:
        """返回意图标签"""
        pass

class RetrieverPlugin(ABC):
    @abstractmethod
    async def retrieve(self, query: str, top_k: int = 10, filters: Dict = None) -> RetrievalResult:
        """执行检索"""
        pass

class RerankerPlugin(ABC):
    @abstractmethod
    async def rerank(self, query: str, results: RetrievalResult, top_k: int = 5) -> RetrievalResult:
        """执行重排"""
        pass

class ParserPlugin(ABC):
    @abstractmethod
    async def parse(self, file_path: str) -> List[Chunk]:
        """解析文档为块"""
        pass

class SecurityPlugin(ABC):
    @abstractmethod
    def check(self, content: str) -> tuple[bool, str]:
        """检查内容安全性"""
        pass

class EvaluatorPlugin(ABC):
    @abstractmethod
    async def evaluate(self, query: str, contexts: List[str], answer: str) -> Dict[str, float]:
        """返回评估指标"""
        pass
```

---

## 3. 数据处理能力

### 3.1 全链路数据处理流程

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        数据处理全链路                                     │
│                                                                         │
│  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐    │
│  │ 数据源  │ → │ 解析    │ → │ 分块   │ → │ 向量化  │ → │ 存储    │    │
│  │Ingestion│   │Parsing │   │Chunking│   │Embedding│   │Storage │    │
│  └─────────┘   └─────────┘   └─────────┘   └─────────┘   └─────────┘    │
│       │             │             │             │              │        │
│       ▼             ▼             ▼             ▼              ▼        │
│  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐    │
│  │30+连接器│   │DeepDoc  │   │语义分块 │   │向量嵌入 │   │Milvus   │    │
│  │S3/GitHub│   │OCR识别  │   │递归分块 │   │稀疏嵌入 │   │ES       │    │
│  │Notion   │   │表格识别 │   │标题分块 │   │混合嵌入 │   │Redis    │    │
│  └─────────┘   └─────────┘   └─────────┘   └─────────┘   └─────────┘    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 文档解析能力

| 格式 | 支持 | 解析方法 | 输出 |
|------|------|----------|------|
| **PDF** | ✅ | DeepDoc / VLM / PlainText | JSON / Markdown |
| **Word** | ✅ | DeepDoc | JSON / Markdown |
| **Excel** | ✅ | DeepDoc | JSON / Markdown / HTML |
| **PPT** | ✅ | DeepDoc | JSON |
| **图片** | ✅ | OCR + VLM | JSON |
| **表格** | ✅ | DeepDoc | JSON / HTML |
| **音频** | ✅ | Whisper | JSON |
| **视频** | ✅ | 截帧 + OCR | JSON |
| **EPUB** | ✅ | 解析器 | Text / JSON |
| **Email** | ✅ | 解析器 | Text / JSON |
| **Text/MD** | ✅ | 解析器 | Text / JSON |
| **HTML** | ✅ | 解析器 | JSON |

### 3.3 分块策略

| 策略 | 说明 | 适用场景 |
|------|------|----------|
| **recursive** | 递归字符分块 | 通用场景 |
| **semantic** | LLM 驱动语义分块 | 长文档 |
| **title** | 按标题层级分块 | 书籍、手册 |
| **paragraph** | 按自然段落分块 | 新闻、文章 |
| **table** | 保持表格结构 | 表格数据 |
| **qa** | QA 对分块 | 问答数据 |
| **dialogue** | 对话单元分块 | 聊天记录 |

### 3.4 向量化模型

| 模型 | 类型 | 向量维度 | 说明 |
|------|------|----------|------|
| **BGE** | 稠密 | 1024/768 | 中英文最优 |
| **M3E** | 稠密 | 1536 | 多语言支持 |
| **BM25** | 稀疏 | - | 词法检索 |
| **Hybrid** | 混合 | - | 稠密+稀疏 |
| **CrossEncoder** | 重排 | - | 精排模型 |

### 3.5 存储层

| 类型 | 支持 | 用途 |
|------|------|------|
| **Milvus** | ✅ | 向量存储 |
| **Elasticsearch** | ✅ | 全文索引 |
| **OceanBase** | ✅ | 混合存储 |
| **Redis** | ✅ | KV 缓存 |
| **Neo4j** | ✅ | 知识图谱 |
| **MySQL** | ✅ | 关系数据 |

---

## 4. 插件系统

### 4.1 插件注册表

```python
class PluginRegistry:
    _slots = {
        "intent_classifiers": {},
        "retrievers": {},
        "rerankers": {},
        "parsers": {},
        "security": {},
        "evaluators": {},
    }

    @classmethod
    def register(cls, slot: str, name: str, plugin):
        """注册插件"""
        if slot not in cls._slots:
            raise ValueError(f"Unknown slot: {slot}")
        cls._slots[slot][name] = plugin

    @classmethod
    def get(cls, slot: str, name: str = "default"):
        """获取插件"""
        return cls._slots[slot].get(name)
```

### 4.2 插件开发模板

```python
class MyMedicalIntentPlugin(BasePlugin):
    metadata = PluginMetadata(
        name="medical-intent",
        version="1.0.0",
        slot="intent_classifier",
        author="Your Name",
        description="医疗领域意图分类插件"
    )

    async def classify(self, query: str) -> str:
        """实现意图分类逻辑"""
        intent = self.model.predict(query)
        return intent

    def get_supported_intents(self):
        return ["diagnosis", "medication", "treatment", "prevention"]

# 注册插件
PluginRegistry.register("intent_classifier", "medical", MyMedicalIntentPlugin())
```

### 4.3 YAML 配置示例

```yaml
# config/profiles/medical_rag.yaml
pipeline:
  name: "医疗RAG"
  version: "1.0"

  plugins:
    intent_classifier:
      type: "medical"
      model: "bce-embedding"

    retriever:
      type: "hybrid"
      retrievers:
        - type: "vector"
          weight: 0.6
        - type: "bm25"
          weight: 0.3
        - type: "knowledge_graph"
          weight: 0.1

    reranker:
      type: "cross_encoder"
      model: "BAAI/bge-reranker-base"

    security:
      type: "medical"
      rules:
        - "drug_interaction"
        - "contraindication"
        - "dosage_limit"

    evaluator:
      type: "ragas"
```

### 4.4 插件市场

| 类别 | 插件 | 说明 |
|------|------|------|
| **意图分类** | medical-intent | 医疗意图识别 |
| | finance-intent | 金融意图识别 |
| | legal-intent | 法律意图识别 |
| | generic-intent | 通用意图分类 |
| **检索器** | vector-retriever | Milvus 向量检索 |
| | bm25-retriever | BM25 词法检索 |
| | kg-retriever | 知识图谱检索 |
| | web-retriever | Tavily Web 搜索 |
| | hybrid-retriever | 混合检索 |
| **重排器** | cross-encoder | CrossEncoder 重排 |
| | bge-reranker | BGE 重排模型 |
| | cohere-reranker | Cohere 云重排 |
| **解析器** | deepdoc-parser | DeepDoc 文档解析 |
| | ocr-parser | OCR 文字识别 |
| | table-parser | 表格解析 |
| **安全** | medical-security | 医疗安全审查 |
| | finance-security | 金融合规审查 |
| | content-moderation | 内容审核 |
| **评估器** | ragas-evaluator | RAGAS 评估 |
| | ir-metrics | IR 指标 (MAP/MRR/NDCG) |
| | medical-qa | 医疗 QA 评估 |

---

## 5. MCP 协议集成

### 5.1 三种插件运行模式

```python
class PluginMode(Enum):
    LOCAL = "local"           # 本地 Python 插件
    REMOTE = "remote"         # MCP 远程服务
    HYBRID = "hybrid"         # 本地优先，失败切换远程
```

### 5.2 MCP 插件适配器

```python
class MCPPluginAdapter:
    """MCP 协议插件适配器"""

    def __init__(self, mcp_server_url: str, plugin_slot: str):
        self.server_url = mcp_server_url
        self.slot = plugin_slot
        self.mcp_client = MCPClient(url=mcp_server_url)

    async def call_tool(self, tool_name: str, **kwargs):
        """通过 MCP 协议调用远程插件"""
        return await self.mcp_client.call_tool(tool_name, kwargs)
```

### 5.3 接入已有 MCP Server

```python
# 接入 UltraRAG MCP Server
PluginRegistry.register_mcp(
    slot="retriever",
    name="ultra_rag_retriever",
    server_url="http://localhost:8001",
    tools=["vector_search", "bm25_search", "hybrid_search"]
)

# 接入 ragflow 知识图谱 MCP Server
PluginRegistry.register_mcp(
    slot="retriever",
    name="ragflow_kg",
    server_url="http://localhost:8002",
    tools=["knowledge_graph.search", "entity_extraction"]
)
```

### 5.4 导出为 MCP Server

```python
# 导出本地插件为独立 MCP Server
if __name__ == "__main__":
    from arch_rag.server import MCPPluginServer
    server = MCPPluginServer(MyMedicalIntentPlugin())
    server.run(transport="stdio")
```

---

## 6. 领域切换

### 6.1 一键切换领域

```python
from arch_rag import RAGEngine

engine = RAGEngine()

# 切换为医疗模式
engine.load_profile("medical_rag")
answer = engine.query("阿司匹林副作用有哪些？")

# 切换为金融模式
engine.load_profile("finance_rag")
answer = engine.query("上市公司并购重组流程？")

# 切换为法律模式
engine.load_profile("legal_rag")
answer = engine.query("合同违约金上限？")
```

### 6.2 领域插件包

```yaml
# 医疗插件包 (medical_plugin_pack/)
medical_plugin_pack/
├── intent_classifier/
│   └── medical_intent.py
├── retriever/
│   └── medical_retriever.py
├── security/
│   └── drug_interaction.py
├── evaluator/
│   └── medical_qa_eval.py
└── config.yaml

# 法律插件包 (legal_plugin_pack/)
legal_plugin_pack/
├── intent_classifier/
│   └── legal_intent.py
├── security/
│   └── confidentiality.py
└── config.yaml
```

---

## 7. 三系统对比

### 7.1 功能对比

| 功能 | rag_standalone | UltraRAG | ragflow | **ArchRAG** |
|------|----------------|----------|---------|-------------|
| **架构模式** | 单体 FastAPI | 微服务 MCP | 分布式服务 | **插件化+MCP** |
| **意图分类** | 医疗专用 | 通用 | 通用 | **通用+领域插件** |
| **向量检索** | ✅ Milvus | ✅ Milvus/FAISS | ✅ Infinity | ✅ **多数据库** |
| **词法检索** | ✅ BM25 | ✅ BM25s | ✅ 自研 | ✅ **标准接口** |
| **混合检索** | ✅ RRF | ✅ 多路融合 | ✅ 多路融合 | ✅ **插件化** |
| **Web 搜索** | ❌ | ✅ Tavily/Exa | ✅ Tavily | ✅ **插件化** |
| **知识图谱** | ❌ | ❌ | ✅ 完整 | ✅ **插件化** |
| **RAPTOR** | ❌ | ❌ | ✅ | ✅ **插件化** |
| **查询分解** | ⚠️ 简单 | ✅ 支持 | ✅ 树状 | ✅ **插件化** |
| **重排器** | 2种 | 3种 | 10+ | ✅ **多后端** |
| **CrossEncoder** | ❌ | ✅ | ✅ | ✅ |
| **PDF解析** | 基础 | 基础 | **DeepDoc** | ✅ **DeepDoc** |
| **表格识别** | ❌ | ❌ | ✅ | ✅ |
| **OCR** | ❌ | ❌ | ✅ | ✅ |
| **分块策略** | 3种 | 3种 | 6+ | ✅ **6+** |
| **语义分块** | ❌ | ❌ | ✅ | ✅ |
| **IR 评估** | ⚠️ 定义未实现 | ✅ MAP/MRR/NDCG | ✅ Benchmark | ✅ **完整** |
| **RAGAS** | ❌ | ⚠️ 可集成 | ❌ | ✅ |
| **数据源** | 本地 | 本地 | **30+** | ✅ **30+** |
| **YAML编排** | ❌ | ✅ | ⚠️ 部分 | ✅ |
| **监控** | ✅ Prometheus | ⚠️ 日志 | ⚠️ 日志 | ✅ **完整** |

### 7.2 技术栈对比

| 组件 | rag_standalone | UltraRAG | ragflow | **ArchRAG** |
|------|----------------|----------|---------|-------------|
| **后端框架** | FastAPI | FastMCP | FastAPI | **FastAPI/MCP** |
| **向量数据库** | Milvus | Milvus/FAISS | Infinity/ES | **多数据库插件** |
| **LLM 集成** | LangChain + SiliconFlow | OpenAI/vLLM | 多 Provider | **统一接口** |
| **缓存** | Redis + Milvus | Redis | Redis | **Redis** |
| **监控** | Prometheus+Grafana | 日志 | 日志 | **可插拔** |

### 7.3 定位对比

| 系统 | 定位 | 最适合场景 |
|------|------|-----------|
| **rag_standalone** | 垂直领域解决方案 | 医疗垂直领域、生产环境 |
| **UltraRAG** | 研究/原型框架 | 快速实验、Pipeline 配置化 |
| **ragflow** | 企业级 RAG 平台 | 多格式文档、知识图谱需求 |
| **ArchRAG** | **通用插件化框架** | **任意领域、快速切换** |

---

## 8. 实施路线

### 8.1 Phase 1：基础增强（Week 1-2）

| 任务 | 来源 | 投入 | 说明 |
|------|------|------|------|
| Cross-Encoder Reranker | ragflow | 1天 | 检索精度 +3-5% |
| Web Search Retriever | UltraRAG | 1天 | 实时信息获取 |
| 插件框架基础 | 新设计 | 3天 | 核心插槽系统 |

### 8.2 Phase 2：核心升级（Week 3-4）

| 任务 | 来源 | 投入 | 说明 |
|------|------|------|------|
| RAPTOR 层次检索 | ragflow | 3天 | 长文档理解提升 |
| 知识图谱基础 | ragflow | 5天 | 多跳问题处理 |
| MCP 协议集成 | UltraRAG | 3天 | 远程插件支持 |

### 8.3 Phase 3：高级特性（Month 2）

| 任务 | 来源 | 投入 | 说明 |
|------|------|------|------|
| DeepDoc 文档解析 | ragflow | 1周 | PDF/表格解析 |
| 树状查询分解 | ragflow | 3天 | 复杂问题分解 |
| 30+ 数据源连接 | ragflow | 按需 | 扩展接入能力 |

### 8.4 Phase 4：完善闭环（Month 3）

| 任务 | 来源 | 投入 | 说明 |
|------|------|------|------|
| IR 评估指标 | UltraRAG | 2天 | MAP/MRR/NDCG |
| RAGAS 集成 | 新设计 | 3天 | 生成质量评估 |
| Benchmark 框架 | ragflow | 2天 | 标准化评估 |

---

## 9. 预期效果

### 9.1 检索质量指标

| 指标 | 当前 (rag_standalone) | 融合后预期 | 提升幅度 |
|------|----------------------|------------|----------|
| **Hit Rate** | ~75% | **90-95%** | +15-20% |
| **MRR@10** | ~0.65 | **0.85+** | +30% |
| **NDCG@10** | ~0.60 | **0.82+** | +35% |
| **多跳问题准确率** | ~50% | **80%+** | +60% |
| **长文档理解** | ~55% | **85%+** | +55% |

### 9.2 功能全景

```
┌─────────────────────────────────────────────────────────────────┐
│                    超级 RAG 系统功能架构                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │ 意图分类    │  │ 语义缓存    │  │ 安全拦截    │              │
│  │ 本地0.6B   │  │ Redis+Milvus│  │ 医疗领域    │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    混合检索引擎                              ││
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐      ││
│  │  │向量检索  │ │ BM25     │ │ Web搜索  │ │知识图谱  │      ││
│  │  │HNSW索引  │ │ Jieba    │ │ Tavily   │ │ 实体关系 │      ││
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘      ││
│  │                        ↓                                   ││
│  │               ┌──────────────────┐                         ││
│  │               │ Cross-Encoder    │                         ││
│  │               │ 多后端重排       │                         ││
│  │               └──────────────────┘                         ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │ RAPTOR      │  │ 查询分解    │  │ Corrective  │              │
│  │ 层次摘要    │  │ 树状多跳    │  │ RAG 自修正  │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    文档解析 (DeepDoc)                         ││
│  │  PDF │ Excel │ Word │ PPT │ 图片OCR │ 表格 │ 音频 │ 视频   ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │ 30+ 数据源  │  │ YAML Pipeline│  │ Benchmark   │              │
│  │ 连接器      │  │ 配置化编排   │  │ 评估框架    │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 9.3 核心优势

| 特性 | 说明 |
|------|------|
| **🔌 完全可插拔** | 任意槽位可替换，无需修改核心代码 |
| **🏭 零代码切换** | 一行配置切换领域，无需重新开发 |
| **📦 插件市场** | 第三方开发者可发布插件 |
| **🧩 自由组合** | 可同时启用多个同类插件（混合检索） |
| **📈 性能可视化** | 每个插件独立耗时统计 |
| **🛡️ 安全隔离** | 插件间无直接依赖，沙箱运行 |
| **🌐 MCP 协议** | 远程/本地插件统一协议 |
| **📊 完整评估** | RAGAS + IR 指标闭环 |

### 9.4 最终目标

```
rag_standalone + UltraRAG + ragflow 优点集合
                    ↓
           ┌───────────────┐
           │   ArchRAG    │
           │  (通用内核)   │
           └───────────────┘
                    │
    ┌───────────────┼───────────────┐
    │               │               │
    ▼               ▼               ▼
┌────────┐    ┌────────┐    ┌────────┐
│医疗插件│    │金融插件│    │法律插件│
└────────┘    └────────┘    └────────┘
    │               │               │
    ▼               ▼               ▼
┌────────────────────────────────────────┐
│         同一内核，任意领域              │
│         一次开发，到处部署              │
│         插件市场，生态共建              │
└────────────────────────────────────────┘
```

---

## 修改记录

| 日期 | 版本 | 修改内容 |
|------|------|----------|
| 2026-04-01 | 1.0 | 初始版本，整合三系统分析 |

---

> **文档版本**: v1.0
> **创建日期**: 2026-04-01
> **维护者**: rag_standalone team

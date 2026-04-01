# ArchRAG 实施清单 — 可打勾 Checklist

> **目标**：以 rag_standalone 为地基，吸收 UltraRAG + ragflow 核心能力，构建 ArchRAG
> **创建日期**: 2026-04-01
> **参考**: `ANALYSIS_THREE_SYSTEMS.md`（深度对比文档）

---

## Phase 1：检索增强（基础能力吸收）

### 1.1 重排器升级

- [ ] **1.1.1** 新增 CrossEncoder Reranker（SentenceTransformers 后端）
  - 来源：UltraRAG `servers/reranker/`
  - 代码：`backend/app/rag/reranker/` 新建目录，抽象 reranker 接口
  - 模型：BAAI/bge-reranker-base 或同级别
  - 现有 QwenReranker 保留作为备选
- [ ] **1.1.2** Reranker 接口统一（多后端抽象）
  - 接口：`RerankerABC` + `RerankerRegistry`
  - 后端：CrossEncoder / Qwen CausalLM / Cloud API
  - 配置：YAML 指定默认后端 + fallback 链

**完成时间**：____  **修改记录**：

| 日期 | 文件 | 说明 |
|------|------|------|

---

### 1.2 Web Search 多后端

- [x] ~~**1.2.1**~~ 吸收 UltraRAG Tavily 后端
  - 来源：UltraRAG `servers/retriever/websearch_backends/tavily_backend.py`
- [x] ~~**1.2.2**~~ 吸收 UltraRAG Exa 后端
  - 来源：UltraRAG `servers/retriever/websearch_backends/exa_backend.py`
- [x] ~~**1.2.3**~~ 统一 Web Search Reranker 接口
  - 作为 Hybrid Search 的第三路（向量+BM25+Web）
  - 配置化启用/禁用

**完成时间**：2026-04-01  **修改记录**：

| 日期 | 文件 | 说明 |
|------|------|------|
| 2026-04-01 | `backend/app/rag/modules/web_search/base.py` | 抽象基类 `BaseWebSearchBackend` |
| 2026-04-01 | `backend/app/rag/modules/web_search/tavily.py` | Tavily 后端（Async TavilyClient） |
| 2026-04-01 | `backend/app/rag/modules/web_search/exa.py` | Exa 后端（AsyncExa） |
| 2026-04-01 | `backend/app/rag/modules/web_search/__init__.py` | 工厂 `create_web_search_backend` |
| 2026-04-01 | `backend/app/core/config.py` | 新增 `WEB_SEARCH_ENABLED` + `WEB_SEARCH_BACKEND` |
| 2026-04-01 | `backend/tests/test_web_search.py` | 6 个新测试（工厂 + 结果格式） |

---

### 1.3 段落智能重排（reflow）

- [x] ~~**1.3.1**~~ 吸收 UltraRAG `reflow_paragraphs()` 函数
  - 来源：UltraRAG `servers/corpus/src/corpus.py`
  - 功能：自动合并被错误换行拆分的段落，处理句末标点、连字符断行
  - 集成：`backend/app/rag/ingestion/cleaners/` 新增 `reflow.py`
  - 作为 ingestion pipeline 的可选预处理步骤

**完成时间**：2026-04-01  **修改记录**：

| 日期 | 文件 | 说明 |
|------|------|------|
| 2026-04-01 | `backend/app/rag/ingestion/cleaners/reflow.py` | `ReflowCleaner` 实现 |
| 2026-04-01 | `backend/app/rag/ingestion/cleaners/__init__.py` | 注册 `reflow` cleaner |
| 2026-04-01 | `backend/tests/test_reflow.py` | 7 个新测试 |

---

### 1.4 评估框架增强（含 p-value）

- [x] ~~**1.4.1**~~ 新增生成评估指标（ROUGE / F1 / EM）
  - 来源：UltraRAG `servers/evaluation/`
  - 指标：accuracy / exact_match / f1 / rouge-1 / rouge-2 / rouge-l
- [x] ~~**1.4.2**~~ TREC 格式 IR 评估支持
  - 加载 qrels + run 文件，支持 MRR / MAP / NDCG@k / Precision@k / Recall@k
- [x] ~~**1.4.3**~~ 统计显著性 p-value 检验
  - 双尾置换检验（10000 次采样）
  - A/B 系统对比，输出 `A_mean / B_mean / Diff / p_value / significant`
  - 新增 CLI 命令：`python -m app.rag.eval compare run_a.json run_b.json qrels.jsonl`
- [x] ~~**1.4.4**~~ 集成到 `evaluation_gate.py`
  - 现有指标保留，新增 ROUGE/F1/p-value

**完成时间**：2026-04-01  **修改记录**：

| 日期 | 文件 | 说明 |
|------|------|------|
| 2026-04-01 | `backend/app/rag/eval.py` | CLI 工具：compare / significance / ir-metrics 子命令 |
| 2026-04-01 | `backend/app/rag/evaluation_gate.py` | 新增 ROUGE/F1/EM + permutation_test 函数 |
| 2026-04-01 | `backend/tests/test_evaluation_enhanced.py` | 12 个新测试 |

---

### 1.5 Chonkie 分块后端

- [x] ~~**1.5.1**~~ 新增 Chonkie 分块器（token/sentence/recursive）
  - 来源：UltraRAG `servers/corpus/src/corpus.py` chunk_documents
  - 现有 IngestionPipeline 增加 `--chunker chonkie` 选项
  - 支持 gpt2/word/character tokenizer

**完成时间**：2026-04-01  **修改记录**：

| 日期 | 文件 | 说明 |
|------|------|------|
| 2026-04-01 | `backend/app/rag/ingestion/chunkers/chonkie.py` | `ChonkieChunker` 实现 |
| 2026-04-01 | `backend/app/rag/ingestion/chunkers/__init__.py` | 注册 `chonkie` chunker |
| 2026-04-01 | `backend/tests/test_chonkie.py` | 4 个新测试 |

---

## Phase 2：深度解析（文档处理升级）

### 2.1 DeepDoc 文档解析

- [ ] **2.1.1** 搬入 DeepDoc vision 模块（OCR）
  - 来源：ragflow `deepdoc/vision/ocr.py` + `deepdoc/vision/`
  - ONNX 推理，CPU/GPU 均可
- [ ] **2.1.2** 搬入 DeepDoc layout 识别
  - 来源：ragflow `deepdoc/vision/layout_recognizer.py`
  - 检测 text/table/figure/title 区域
- [ ] **2.1.3** 搬入 DeepDoc 表格结构识别
  - 来源：ragflow `deepdoc/vision/table_structure_recognizer.py`
- [ ] **2.1.4** 搬入 PDF 解析器（含跨页 XGBoost 合并）
  - 来源：ragflow `deepdoc/parser/pdf_parser.py`
  - 输出带 positions 元数据的 JSON
- [ ] **2.1.5** 新增 ParserPlugin 高级后端
  - 抽象 `ParserPlugin` 接口，支持 DeepDoc / MinerU / PlainText 多后端
  - 配置：`--parse-method deepdoc|plain_text|mineru`
- [ ] **2.1.6** Media Context 上下文绑定
  - 来源：ragflow `rag/flow/splitter/splitter.py` attach_media_context
  - 表格/图片 chunk 自动拼接最近文本 context
- [ ] **2.1.7** PDF 坐标追踪 + 引用溯源
  - 每个 chunk 带 `positions` 元数据（页码+坐标）
  - 回答格式：`[ID:n]` 引用标注，溯源到原文位置
  - Citation 系统（标准 + 后验注入）来源：ragflow `rag/prompts/citation*.md`

**完成时间**：____  **修改记录**：

| 日期 | 文件 | 说明 |
|------|------|------|

---

### 2.2 RAPTOR 层次检索

- [ ] **2.2.1** 实现 RAPTOR
  - 来源：ragflow `rag/raptor.py`
  - UMAP 降维 + GMM 聚类（BIC 自动定簇数）
  - 每层 LLM 递归摘要
- [ ] **2.2.2** 替换现有 hierarchical_index.py
  - 现方案是规则式 3 级，RAPTOR 是算法式多层
  - 保留现方案作为 fallback (`--raptor disabled`)
- [ ] **2.2.3** HierarchicalMerger 吸收
  - 来源：ragflow `rag/flow/hierarchical_merger/hierarchical_merger.py`
  - Regex 匹配层级，树状合并 chunk

**完成时间**：____  **修改记录**：

| 日期 | 文件 | 说明 |
|------|------|------|

---

### 2.3 TOC 语义检索

- [ ] **2.3.1** LLM TOC 提取管线
  - 来源：ragflow `rag/flow/extractor/extractor.py` + `rag/prompts/toc_*.md`
  - 检测→提取→补齐→定级→映射→评分
- [ ] **2.3.2** 检索时 TOC 路由
  - 来源：ragflow `relevant_chunks_with_toc()`
  - 按 TOC 语义层级打分，而非扁平 RRF

**完成时间**：____  **修改记录**：

| 日期 | 文件 | 说明 |
|------|------|------|

---

### 2.4 MinerU 集成

- [x] ~~**2.4.1**~~ MinerU PDF 解析接入
  - 来源：UltraRAG `servers/corpus/src/corpus.py` `mineru_parse()`
  - CLI 调用 MinerU，自动提取文本+图片 corpus
- [x] ~~**2.4.2**~~ 作为 ParserPlugin 的后端之一

**完成时间**：2026-04-01  **修改记录**：

| 日期 | 文件 | 说明 |
|------|------|------|
| 2026-04-01 | `backend/app/rag/ingestion/parsers/mineru_parser.py` | `MinerUParser` 实现 |
| 2026-04-01 | `backend/app/rag/ingestion/parsers/__init__.py` | 引入 MinerUParser |
| 2026-04-01 | `backend/app/rag/ingestion/parsers/base.py` | `register_parser` 改为 `setdefault` |
| 2026-04-01 | `backend/tests/test_mineru_parser.py` | 6 个新测试 |

---

## Phase 3：智能检索（分路 + 图谱）

### 3.1 树状查询分解（核心新功能）

- [ ] **3.1.1** 实现 Tree-Structured Query Decomposition
  - 来源：ragflow `rag/advanced_rag/tree_structured_query_decomposition_retrieval.py`
  - 流程：多路检索 → LLM 充分性判断 → 子查询生成 → 递归研究
- [ ] **3.1.2** 深度控制（depth=3 默认）
- [ ] **3.1.3** 信息汇总（asyncio.Lock thread-safe）
  - 来源：ragflow `_async_update_chunk_info()`
- [ ] **3.1.4** 查询类型识别
  - LLM 判断：对比型 / 多步型 / 混合型 → 分路；简单型 → 快速路径
- [ ] **3.1.5** 子查询 Prompt 模板
  - 来源：ragflow `rag/prompts/sufficiency_check.md` + `multi_queries_gen.md`
  - 充分性判断 + 缺失信息 + 下一批查询

**完成时间**：____  **修改记录**：

| 日期 | 文件 | 说明 |
|------|------|------|

---

### 3.2 GraphRAG 增强

- [ ] **3.2.1** 实体合并逻辑
  - 来源：ragflow `rag/graphrag/general/extractor.py`
  - LLM 驱动 Light/General 双模式
  - 同名实体合并（SEP 分隔描述）+ 描述摘要压缩（512 token 阈）
- [ ] **3.2.2** Leiden 社区检测
  - 来源：ragflow `rag/graphrag/general/leiden.py`
  - 层级社区划分
- [ ] **3.2.3** 社区报告生成
  - 来源：ragflow `rag/graphrag/general/community_reports_extractor.py`
- [ ] **3.2.4** 图检索增强
  - PageRank + 相似度融合 + N 跳路径
  - 来源：ragflow `rag/graphrag/search.py`
- [ ] **3.2.5** 实体消歧
  - 来源：ragflow `rag/graphrag/entity_resolution.py`
  - editdistance 预筛 + LLM 批量决议

**完成时间**：____  **修改记录**：

| 日期 | 文件 | 说明 |
|------|------|------|

---

### 3.3 IRCoT / Search-o1 推理范式

- [x] ~~**3.3.1**~~ IRCoT（迭代检索思维链）
  - 来源：UltraRAG `servers/custom/src/custom.py` `ircot_*` 工具
  - 检索→推理→再检索→推理 循环
- [x] ~~**3.3.2**~~ Search-o1（深度推理搜索）
  - 来源：UltraRAG `servers/custom/src/custom.py` `search_o1_*` 工具
- [x] ~~**3.3.3**~~ 作为 RAG pipeline 的可选模式
  - 新建 `reasoning/` 模块
  - `IRCoTOrchestrator` + `SearchO1Orchestrator`
  - 配置：`--reasoning-mode ircot|search_o1|none`

**完成时间**：2026-04-01  **修改记录**：

| 日期 | 文件 | 说明 |
|------|------|------|
| 2026-04-01 | `backend/app/rag/reasoning/ircot.py` | IRCoT 编排器 |
| 2026-04-01 | `backend/app/rag/reasoning/search_o1.py` | Search-o1 + 全部工具函数 |
| 2026-04-01 | `backend/app/rag/reasoning/_prompts.py` | IRCoT + Search-o1 Prompt 模板 |
| 2026-04-01 | `backend/app/rag/reasoning/__init__.py` | 公开 Orchestrator 类 |
| 2026-04-01 | `backend/tests/test_reasoning.py` | 12 个新测试 |

---

### 3.4 辅助能力

- [ ] **3.4.1** 跨语言查询扩展
  - 来源：ragflow `rag/prompts/cross_languages_*.md`
  - 翻译查询 → 多语言同时检索 → 合并结果
- [ ] **3.4.2** 元数据过滤 LLM 提取
  - 来源：ragflow `rag/prompts/meta_filter.md`
  - LLM 从查询提取 date/author/type → 检索层应用 filter
- [ ] **3.4.3** Model Family Policy Engine
  - 来源：ragflow `rag/llm/chat_model.py` `_apply_model_family_policies`
  - 根据不同模型名自动调整参数（Qwen3/GPT-5/Kimi 等）

**完成时间**：____  **修改记录**：

| 日期 | 文件 | 说明 |
|------|------|------|

---

## Phase 4：管线平台（MCP + YAML）

### 4.1 Pipeline YAML 编排

- [ ] **4.1.1** 定义 Pipeline YAML Schema
  - 来源：UltraRAG pipeline 格式
  - servers 映射（retriever/reranker/generation/evaluation）
  - pipeline 步骤列表（支持 loop / branch）
- [ ] **4.1.2** 实现 Pipeline 执行器
  - 来源：UltraRAG `src/ultrarag/client.py` `UltraData` + `run()`
  - 内存系统（`memory_xxx`）
  - 变量依赖图自动提取
- [ ] **4.1.3** 重写现有检索管线为 Pipeline
  - 逐步把 `MedicalRetriever.search_rag30()` 拆解为 Pipeline 步骤
  - 保留现有入口作为兼容模式

**完成时间**：____  **修改记录**：

| 日期 | 文件 | 说明 |
|------|------|------|

---

### 4.2 MCP Server 生态

- [ ] **4.2.1** 实现 ToolCall 统一路由
  - 来源：UltraRAG `src/ultrarag/api.py` `_Router/_ServerProxy/_CallWrapper`
- [ ] **4.2.2** 拆分 MCP Servers
  - Retriever Server / Reranker Server / Generation Server / Prompt Server / Evaluation Server
- [ ] **4.2.3** Prompt 沙箱管理服务
  - 来源：UltraRAG `servers/prompt/src/prompt.py`
  - `SandboxedEnvironment` + `_safe_render()`
  - Jinja2 模板防 XSS
- [ ] **4.2.4** server.yaml + parameter.yaml 配置热切换
  - 每个 server 独立配置目录

**完成时间**：____  **修改记录**：

| 日期 | 文件 | 说明 |
|------|------|------|

---

### 4.3 语料处理 Server

- [ ] **4.3.1** 吸收 UltraRAG corpus server
  - 来源：UltraRAG `servers/corpus/src/corpus.py`
  - 多格式文本提取 / 图片语料 / MinerU 集成 / Chonkie 分块
- [ ] **4.3.2** 安全文件处理 `_validate_path()`
  - 路径穿越防护

**完成时间**：____  **修改记录**：

| 日期 | 文件 | 说明 |
|------|------|------|

---

## Phase 5：前端升级

### 5.1 前端架构设计

- [ ] **5.1.1** 技术栈决策
  - 保留 Next.js（已确定）
  - 新增组件库：shadcn/ui + Tailwind CSS
  - 新增数据管理：React Query（服务端状态）+ Zustand（客户端状态）
  - 新增工具：Sonner toast / Lucide React icons / date-fns
- [ ] **5.1.2** 路由重构
  - 参考 ragflow 40+ 路由结构
  - 新增：/datasets（知识库管理）/documents（文档管理）/eval（评估面板）
  - 保留：chat（主界面）/rag（检索测试）/settings

**完成时间**：____  **修改记录**：

| 日期 | 文件 | 说明 |
|------|------|------|

---

### 5.2 知识库管理页面

- [ ] **5.2.1** 知识库列表页
  - 来源：ragflow `web/src/pages/datasets/index.tsx`
  - 分页卡片列表 + 搜索 + 筛选
  - 创建/重命名对话框（Zod 校验，选择 embedding 模型和分块策略）
- [ ] **5.2.2** 文档上传与管理
  - 来源：ragflow `web/src/components/file-upload-dialog/`
  - 拖拽上传 + 文件夹上传 + "解析即创建"开关
  - 文件列表 + 解析进度实时显示
- [ ] **5.2.3** 解析/分块配置
  - 来源：ragflow `web/src/components/chunk-method-dialog/`
  - 可视化选择解析方法和分块策略

**完成时间**：____  **修改记录**：

| 日期 | 文件 | 说明 |
|------|------|------|

---

### 5.3 检索测试页面

- [ ] **5.3.1** 分屏检索测试
  - 来源：ragflow `web/src/pages/dataset/testing/`
  - 左表单 + 右结果，支持多路对比
  - 升级现有 `/rag` 页面
- [ ] **5.3.2** 对比栏升级
  - 现有 compare-bar 升级为完整指标面板
  - 显示 MRR/NDCG/F1 等指标

**完成时间**：____  **修改记录**：

| 日期 | 文件 | 说明 |
|------|------|------|

---

### 5.4 聊天界面升级

- [ ] **5.4.1** 会话侧边栏
  - 多会话管理（参考 ragflow / UltraRAG UI）
- [ ] **5.4.2** 知识库选择器
  - 来源：UltraRAG UI chat 输入区域 KB 下拉选择
  - 在聊天输入区选择查询的知识库
- [ ] **5.4.3** 多模型对话对比
  - 来源：ragflow `web/src/pages/next-chats/`
  - 支持 2-3 模型并行回答
- [ ] **5.4.4** 引用溯源显示
  - 支持 `[ID:n]` 点击定位到文档原始位置（如 PDF 页码坐标）

**完成时间**：____  **修改记录**：

| 日期 | 文件 | 说明 |
|------|------|------|

---

### 5.5 评估面板页面

- [ ] **5.5.1** 评估数据可视化
  - 指标面板（MRR/MAP/NDCG/F1/ROUGE）
  - A/B 对比柱状图 + p-value 标注
  - 历史评估结果趋势图
- [ ] **5.5.2** 运行评估
  - 选择数据集 → 选择评估指标 → 运行 → 显示结果
  - 结果导出为 JSON / Markdown 报告

**完成时间**：____  **修改记录**：

| 日期 | 文件 | 说明 |
|------|------|------|

---

### 5.6 共享组件建设

- [ ] **5.6.1** 引入 UI 组件库
  - 推荐：Ant Design 或 shadcn/ui + Tailwind
  - 替换现有的手写 CSS 类
- [ ] **5.6.2** 主题切换（暗黑/明亮模式）
  - 来源：ragflow 的 ThemeProvider
  - CSS 变量驱动
- [ ] **5.6.3** ListFilterBar 组件
  - 来源：ragflow `web/src/components/list-filter-bar`
  - 搜索 + 筛选 + 创建按钮一体化
- [ ] **5.6.4** 全局导航栏
  - 来源：ragflow `web/src/layouts/components/global-navbar.tsx`
  - 胶囊状标签栏，带动画活动指标
- [ ] **5.6.5** i18n 国际化
  - 来源：ragflow i18next + 多语言 JSON 字典

**完成时间**：____  **修改记录**：

| 日期 | 文件 | 说明 |
|------|------|------|

---

## Phase 6：远期（Agent 编排）

### 6.1 DAG Agent Canvas

- [ ] **6.1.1** 基础组件框架
  - 来源：ragflow `agent/` + `@xyflow/react`
  - 节点：Begin / LLM / Retrieval / Categorize / Switch / Message
- [ ] **6.1.2** 节点拖拽与连接
  - 可视化拖拽，自动连线
- [ ] **6.1.3** 节点配置表单
  - 点击节点弹出抽屉配置
- [ ] **6.1.4** DSL JSON 序列化/反序列化
- [ ] **6.1.5** 运行与调试模式
  - 实时日志流 + 节点执行状态

**完成时间**：____  **修改记录**：

| 日期 | 文件 | 说明 |
|------|------|------|

---

### 6.2 更多领域包

- [ ] **6.2.1** 金融领域插件
- [ ] **6.2.2** 通用知识领域插件
- [ ] **6.2.3** 插件市场基础架构

---

## 总体能力矩阵（融合前后对比）

| 能力维度 | 当前 rag_standalone | 融合后 ArchRAG 预期 | 来源 |
|----------|---------------------|----------------------|------|
| 文档解析 | PDF/DOCX/HTML/MD/TXT 基础 | + DeepDoc OCR+布局+表格 / MinerU | ragflow |
| 分块策略 | 5 种 | + reflow / Media Context / Hierarchical | ragflow + UltraRAG |
| 向量检索 | Milvus HNSW | 保留 | — |
| 词法检索 | BM25 jieba | 保留 | — |
| 知识图谱 | 混合搜索基础 | + 实体合并 / Leiden 社区 / PageRank | ragflow |
| 重排器 | Qwen 0.6B + Cloud | + CrossEncoder / Infinity / OpenAI | UltraRAG |
| Web 搜索 | Tavily (部分) | + Exa / ZhipuAI | UltraRAG |
| 查询处理 | 改写+扩展+HyDE | + 树状查询分解 / 跨语言 / IRCoT | ragflow + UltraRAG |
| 语义缓存 | 0.96 交叉验证 | 保留 | — |
| 安全门控 | 医疗专属 | 保留 (插件化) | — |
| 引用输出 | 基础 | + 严格 Citation / 位置溯源 | ragflow |
| 监控 | Prometheus+Grafana+Langfuse | 保留 | — |
| 评估 | MRR/MAP/NDCG/Faithfulness | + ROUGE / F1 / p-value 显著性 | UltraRAG |
| 管线配置 | Python 硬编码→YAML 编排 | + Pipeline YAML + MCP Server | UltraRAG |
| 推理范式 | 无 | IRCoT / Search-o1 | UltraRAG |
| 前端 | 3 页 Next.js 手写 CSS | 完整 KB 管理/文档上传/评估面板/多模型对比 | ragflow + UltraRAG |
| 领域切换 | medical/legal 插件 | + finance / 任意领域配置 | — |

---

## 修改记录

| 日期 | 版本 | 修改内容 |
|------|------|----------|
| 2026-04-01 | 1.0 | 初始版本，基于三系统深度分析创建实施清单 |

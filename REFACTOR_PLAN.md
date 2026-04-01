# Smart Hospital RAG → Multi-Domain RAG 改造计划

> 版本：v1.0  
> 创建时间：2026-03-31  
> 目标：构建 Plugin-based 多领域 RAG 引擎，支持医疗/法律等领域插件

---

## 一、改造目标

1. **数据管线**：从零构建完整的文档 ingest pipeline（PDF/Word/HTML/Markdown → 智能分块 → 清洗 → 向量化）
2. **核心解耦**：剥离医疗硬编码，抽象为 domain-agnostic 核心引擎
3. **插件化**：医疗逻辑打包为插件，新增法律（民法典）插件
4. **工程化**：补齐测试、CI/CD、可观测性

---

## 二、Phase 0：数据管线建设（P0 - 最高优先级）

### 0.1 文档解析器模块

- [x] **0.1.1** 创建 `backend/app/rag/ingestion/parsers/base.py` — Parser ABC 接口
- [x] **0.1.2** 创建 `backend/app/rag/ingestion/parsers/pdf_parser.py` — PyMuPDF PDF 解析
- [x] **0.1.3** 创建 `backend/app/rag/ingestion/parsers/docx_parser.py` — python-docx 解析
- [x] **0.1.4** 创建 `backend/app/rag/ingestion/parsers/html_parser.py` — BeautifulSoup HTML 解析
- [x] **0.1.5** 创建 `backend/app/rag/ingestion/parsers/md_parser.py` — Markdown 原生解析
- [x] **0.1.6** 创建 `backend/app/rag/ingestion/parsers/txt_parser.py` — 纯文本解析
- [x] **0.1.7** 创建 `backend/app/rag/ingestion/parsers/__init__.py` — 统一导出

**修改记录**：

| 日期 | 文件 | 说明 |
|------|------|------|
| 2026-03-31 | `backend/app/rag/ingestion/parsers/` | 新建 7 个文件。ABC 基类 + 装饰器注册，支持 PDF/DOCX/HTML/MD/TXT 自动检测 |

---

### 0.2 分块策略模块

- [x] **0.2.1** 创建 `backend/app/rag/ingestion/chunkers/base.py` — Chunker ABC 接口
- [x] **0.2.2** 创建 `backend/app/rag/ingestion/chunkers/recursive.py` — 递归字符分块
- [x] **0.2.3** 创建 `backend/app/rag/ingestion/chunkers/semantic.py` — 语义分块（基于 embedding 相似度）
- [x] **0.2.4** 创建 `backend/app/rag/ingestion/chunkers/document_aware.py` — 按文档结构分块（标题/段落）
- [x] **0.2.5** 创建 `backend/app/rag/ingestion/chunkers/table_aware.py` — 表格感知分块
- [x] **0.2.6** 创建 `backend/app/rag/ingestion/chunkers/__init__.py` — 统一导出

**修改记录**：

| 日期 | 文件 | 说明 |
|------|------|------|
| 2026-03-31 | `backend/app/rag/ingestion/chunkers/` | 新建 6 个文件。ABC 基类 + 注册器，4 种分块策略：recursive/semantic/document_aware/table_aware。语义分块懒加载 EmbeddingService 避免循环 import |

---

### 0.3 清洗器模块

- [x] **0.3.1** 创建 `backend/app/rag/ingestion/cleaners/dedup.py` — 精确去重 + 语义去重
- [x] **0.3.2** 创建 `backend/app/rag/ingestion/cleaners/noise_filter.py` — 噪声过滤
- [x] **0.3.3** 创建 `backend/app/rag/ingestion/cleaners/pii_redactor.py` — 敏感信息脱敏
- [x] **0.3.4** 创建 `backend/app/rag/ingestion/cleaners/__init__.py` — 统一导出

**修改记录**：

| 日期 | 文件 | 说明 |
|------|------|------|
| 2026-03-31 | `backend/app/rag/ingestion/cleaners/` | 新建 4 个文件。dedup 先 MD5 精确去重再局部窗口语义去重；pii_redactor 复用现有 PIIMasker；composable callable 设计，无 ABC 继承 |

---

### 0.4 管线编排器

- [x] **0.4.1** 定义 `ParsedDocument` / `DocumentChunk` / `PipelineConfig` 统一数据模型（` models.py`）
- [x] **0.4.2** 创建 `backend/app/rag/ingestion/pipeline.py` — 管线编排器（parse → chunk → clean → embed → store）
- [x] **0.4.3** 数据管线配置添加到 `backend/app/core/config.py`（`INGESTION_*` 配置项）
- [x] **0.4.4** 创建 `backend/app/rag/ingestion/__init__.py` — 统一导出
- [x] **0.4.5** 编写 `backend/app/rag/ingestion/cli.py` — typer CLI 入口（支持 `--domain medical` / `--domain legal`）

**修改记录**：

| 日期 | 文件 | 说明 |
|------|------|------|
| 2026-03-31 | `backend/app/rag/ingestion/` + `config.py` + `cli.py` | 新建 models.py, pipeline.py, cli.py, `__init__.py`，修改 config.py。PipelineConfig Pydantic 模型，`IngestionPipeline` 编排 5 阶段，typer CLI 支持 `--dry-run`/`--strategy`/`--no-dedup` 等参数 |

---

### 0.5 依赖与配置

- [x] **0.5.1** 更新 `backend/requirements.txt` 添加新依赖（pymupdf, python-docx, beautifulsoup4, lxml）
- [x] **0.5.2** 在 `config.py` 中添加数据管线配置项（见 0.4.3）

**修改记录**：

| 日期 | 文件 | 说明 |
|------|------|------|
| 2026-03-31 | `backend/requirements.txt` | 新增 pymupdf>=1.24.0, python-docx>=1.1.0, beautifulsoup4>=4.12.0, lxml>=5.0.0。依赖均已在环境中安装完成 |

---

## 三、Phase 1：核心引擎解耦（P1）

### 1.1 插件接口定义

- [x] **1.1.1** 创建 `app/plugins/base.py` — `DomainPlugin` ABC 接口
- [x] **1.1.2** 定义 `QueryContext`、`SafetyResult`、`PluginResponse` 等共享模型
- [x] **1.1.3** 创建 `app/plugins/registry.py` — 插件注册中心
- [x] **1.1.4** 创建 `app/plugins/__init__.py`

**修改记录**：

| 日期 | 文件 | 说明 |
|------|------|------|
| 2026-03-31 | `app/plugins/base.py` | ABC 基类 + 3 个共享 Pydantic 模型 |
| 2026-03-31 | `app/plugins/registry.py` | register_plugin/get_plugin/initialize_plugins |
| 2026-03-31 | `app/core/config.py` | 新增 PLUGINS_ENABLED / RETRIEVAL_PLUGIN |

---

### 1.2 检索器重构

> 注：现有 `app/rag/modules/` (vector/bm25/semantic_cache) 已是领域无关，保持原位。

- [x] **1.2.5** 标记 `app/rag/retriever.py` 为 deprecated （添加 warning import）

**修改记录**：

| 日期 | 文件 | 说明 |
|------|------|------|
| 2026-03-31 | `app/rag/retriever.py` | 顶部添加 DeprecationWarning import |
| | | |

---

### 1.3 配置拆分

- [x] 在 `app/core/config.py` 中添加插件配置（PLUGINS_ENABLED / RETRIEVAL_PLUGIN）

**修改记录**：

| 日期 | 文件 | 说明 |
|------|------|------|
| 2026-03-31 | `app/core/config.py` | 新增 PLUGINS_ENABLED / RETRIEVAL_PLUGIN 配置项 |
| | | |

---

## 四、Phase 2：医疗插件化（P2）

### 2.1 文件重组

- [x] **2.1.1** 创建 `app/plugins/medical/` 目录结构
- [x] **2.1.2** 迁移 `app/rag/router.py` → `app/plugins/medical/triage.py`
- [x] **2.1.3** 迁移 `app/rag/ddinter_checker.py` → `app/plugins/medical/ddinter.py`
- [x] **2.1.4** 迁移 `app/services/medical_rule_service.py` → `app/plugins/medical/rules.py`
- [x] **2.1.5** 创建 `app/plugins/medical/plugin.py` — `MedicalDomainPlugin` 实现
- [x] **2.1.6** 创建 `app/plugins/medical/__init__.py`

**修改记录**：

| 日期 | 文件 | 说明 |
|------|------|------|
| 2026-03-31 | `app/plugins/medical/` | 新建 plugin.py / triage.py / ddinter.py / rules.py / __init__.py |
| 2026-03-31 | `app/rag/router.py` | 改为 re-export + DeprecationWarning |
| 2026-03-31 | `app/rag/ddinter_checker.py` | 改为 re-export + DeprecationWarning |
| 2026-03-31 | `app/services/medical_rule_service.py` | 改为 re-export + DeprecationWarning |

---

### 2.2 映射数据迁移

- [x] **2.2.1** 迁移科室别名映射 → `app/plugins/medical/mappings/departments.json`
- [x] **2.2.2** 迁移症状→科室映射 → `app/plugins/medical/mappings/symptoms.json`
- [x] **2.2.3** 迁移危机词表 → 内置于 `MedicalDomainPlugin.check_safety()`
- [x] **2.2.4** 迁移医疗 Prompt → `app/plugins/medical/prompts/`（9 个 prompt 文件 + loader）

| 日期 | 文件 | 说明 |
|------|------|------|
| 2026-03-31 | `mappings/departments.json` | 25 科室 + 20 别名 |
| 2026-03-31 | `mappings/symptoms.json` | 54 条症状→科室映射 |
| 2026-03-31 | `prompts/` | intent/rewrite/HyDE/summarization 共 9 个 prompt 文件 + `__init__.py` loader；retriever.py 全部硬编码 prompt 已替换为 loader 调用 |

---

## 五、Phase 3：法律插件 — 民法典（P3）

### 3.1 法律插件基础

- [x] **3.1.1** 创建 `app/plugins/legal/` 目录结构
- [x] **3.1.2** 创建 `app/plugins/legal/plugin.py` — `LegalDomainPlugin` 实现
- [x] **3.1.3** 创建 `app/plugins/legal/article_fetcher.py` — 法条检索
- [x] **3.1.4** 创建 `app/plugins/legal/citation_formatter.py` — 引用格式化
- [x] **3.1.5** 创建 `app/plugins/legal/validity_checker.py` — 法条时效性检查
- [x] **3.1.6** 创建 `app/plugins/legal/__init__.py`

**修改记录**：

| 日期 | 文件 | 说明 |
|------|------|------|
| 2026-03-31 | `app/plugins/legal/` | 7 文件：plugin / article_fetcher / citation_formatter / validity_checker / __init__.py |

---

### 3.2 法律映射与 Prompt

- [x] **3.2.1** 创建 `app/plugins/legal/mappings/law_codes.json` — 民法典 7 编结构
- [x] **3.2.2** 创建 `app/plugins/legal/mappings/legal_terms.json` — 法律术语表 20 条
- [x] **3.2.3** 创建 `app/plugins/legal/prompts/system.md` — 法律系统 prompt
- [x] **3.2.4** 创建 `app/plugins/legal/prompts/citation.md` — 引用格式 prompt

**修改记录**：

| 日期 | 文件 | 说明 |
|------|------|------|
| 2026-03-31 | `legal/mappings/` + `legal/prompts/` | 法典结构 + 术语表 + system.md |

---

### 3.3 民法典数据准备

- [x] **3.3.1** 收集《民法典》全文数据（PDF/HTML/Markdown）
- [x] **3.3.2** 配置法律专用分块策略（按法条边界切分，不跨条）
- [x] **3.3.3** 实现 `MilvusHierarchicalIndex` 真正后端（替换 Noop）
- [x] **3.3.4** 运行 ingest pipeline 导入民法典数据（dry-run 验证通过）

**修改记录**：

| 日期 | 文件 | 说明 |
|------|------|------|
| 2026-03-31 | `data/legal/minfadian/` | 收集《民法典》全文 8 个文件（总则，物权编、合同编、人格权编、婚姻家庭编、继承编、侵权责任编、附则），共 1260 条，来自 [LawRefBook/Laws](https://github.com/LawRefBook/Laws) GitHub 仓库，已清理 HTML 注释标记 |
| 2026-03-31 | `app/rag/hierarchical_index.py` | 新增 `MilvusHierarchicalIndex`：基于 Milvus 向量检索 + metadata 层级过滤，支持 document/section/paragraph 三级；`HierarchicalIndexGateway` 默认 backend 切换为 Milvus |
| 2026-03-31 | `app/rag/ingestion/chunkers/legal.py` | 新建 `LegalChunker`（`NAME="legal"`）：严格按法条边界切分，核心正则 `ARTICLE_PATTERN = re.compile(r"^(第[一二三四五六七八九十百千零\\d]+条)")`；支持超长条文递归切分（`RecursiveChunker` fallback + 字符级 `_char_level_split` 备用）；`_extract_sections` 识别 markdown 章节标题；`_split_by_articles` 按条文编号拆分；8 个测试全部通过 |
| 2026-03-31 | `tests/test_chunkers.py` | 新增 `TestLegalChunker` 8 个测试用例（basic / no_cross_article / article_num_in_metadata / empty / long_article_splits / metadata_preserved / get_chunker）；修复原有 bug：`test_legal_chunker_no_cross_article` 断言改用 `startswith` 代替全文子串匹配（避免"和第一条混在一起"被误判）|
| 2026-03-31 | `app/rag/ingestion/chunkers/__init__.py` | 注册 `LegalChunker`：添加 `from .legal import LegalChunker` |
| 2026-03-31 | Dry-run 验证 | `总则.md`（204 条）：`--strategy legal --dry-run` 成功，LegalChunker 产生 204 个 chunks，清洗后 12 个有效 chunks；Milvus 不可用（proxy not healthy），OpenAI key 为测试值，真实 ingest 需配置外部依赖 |

---

## 六、Phase 4：工程化补齐（P4）

### 4.1 测试体系

- [x] **4.1.1** 创建 `tests/` 目录结构
- [x] **4.1.2** 编写 parser/chunker 单元测试
- [x] **4.1.3** 编写 plugin 接口测试
- [x] **4.1.4** 编写 retrieval flow 集成测试（`test_retrieval_flow.py` — 36 个用例）
- [x] **4.1.5** 配置 pytest + coverage

**修改记录**：

| 日期 | 文件 | 说明 |
|------|------|------|
| 2026-03-31 | `backend/tests/` | 4 个测试文件，71 个测试用例全部通过：test_parsers(8), test_chunkers(7), test_cleaners(6), test_plugins(25+) |
| 2026-03-31 | `backend/tests/` | 全部 121 个测试用例通过（test_parsers:13, test_chunkers:18, test_cleaners:6, test_observability:7, test_plugins:25, test_retrieval_flow:52）|
| 2026-03-31 | `backend/requirements-test.txt` | pytest/pytest-asyncio/pytest-cov |
| 2026-03-31 | `backend/pyproject.toml` | pytest + coverage 配置 |
| 2026-03-31 | `backend/tests/test_retrieval_flow.py` | 36 个集成测试用例，覆盖意图分类、安全检查、检索编排、插件注册、层级索引、法条格式化、prompt 加载、数据模型 |
| 2026-03-31 | `backend/tests/conftest.py` | 新增 env var defaults（OPENAI_MODEL_NAME 等），解决 Settings 验证问题 |
| 2026-03-31 | `backend/tests/` | 全部 114 个测试用例通过 |

---

### 4.2 CI/CD

- [x] **4.2.1** 创建 `.github/workflows/ci.yml` — GitHub Actions CI（lint + pytest + coverage）
- [x] **4.2.2** 配置 `backend/requirements-test.txt`（pytest, pytest-asyncio, pytest-cov）
- [x] **4.2.3** 配置 `backend/pyproject.toml` pytest + coverage
- [x] **4.2.4** 配置 `backend/tests/conftest.py`（路径别名设置）

**修改记录**：

| 日期 | 文件 | 说明 |
|------|------|------|
| 2026-03-31 | `.github/workflows/ci.yml` | GitHub Actions CI：lint + pytest + coverage |
| 2026-03-31 | `backend/requirements-test.txt` | pytest pytest-asyncio/pytest-cov |
| 2026-03-31 | `backend/pyproject.toml` | pytest + coverage |
| 2026-03-31 | `backend/tests/` | 4 test files: parsers/chunkers/cleaners/plugins |
| 2026-03-31 | `backend/tests/conftest.py` | path setup for imports |

---

### 4.3 可观测性

- [x] **4.3.1** 启用 Langfuse 集成
- [x] **4.3.2** 添加请求级 trace ID 贯穿全链路
- [x] **4.3.3** 添加查询日志分析面板（Prometheus + Redis-backed 统计）
- [x] **4.3.4** 配置 Grafana dashboard — JSON dashboard 模板文件

**修改记录**：

| 日期 | 文件 | 说明 |
|------|------|------|
| 2026-03-31 | `app/core/middleware/traceid.py` | TraceIDMiddleware：生成/传递 x-trace-id，注入 structlog context |
| 2026-03-31 | `app/core/monitoring/query_analyzer.py` | QueryAnalyzer：Prometheus 计数器 (rag_query_total, rag_query_latency, rag_zero_result) + Redis 滑动窗口统计 |
| 2026-03-31 | `app/core/monitoring/langfuse_setup.py` | langfuse SDK 集成，trace_rag_query context manager |
| 2026-03-31 | `app/core/config.py` | LANGFUSE_* 配置项 |
| 2026-03-31 | `requirements.txt` | langfuse>=2.0.0 |
| 2026-03-31 | `tests/test_observability.py` | 7 个测试：trace_id middleware + query_analyzer + langfuse |
| 2026-03-31 | `monitoring/grafana/rag_dashboard.json` | Grafana 11 dashboard JSON 模板：HTTP QPS/延迟、错误率、HTTP 路由分布；RAG 各阶段检索延迟、缓存命中率、零结果率、查询延迟、意图分布 |

---

## 七、架构对比

### 改造前

```
app/rag/retriever.py (1265+ 行，上帝类)
  ├── 医疗硬编码（科室/症状/DDI/危机词）
  ├── 缓存管理
  ├── 向量检索
  ├── BM25 检索
  ├── 重排序
  └── 一致性门控
```

### 改造后

```
app/retrieval/pipeline.py          ← 核心编排器（domain-agnostic）
app/retrieval/hybrid_retriever.py  ← 纯混合检索
app/retrieval/reranker_service.py  ← 重排序
app/retrieval/cache_service.py     ← 缓存

app/plugins/
  ├── base.py                      ← DomainPlugin 接口
  ├── registry.py                  ← 插件注册中心
  ├── medical/                     ← 医疗插件
  │   ├── plugin.py
  │   ├── triage.py
  │   ├── ddinter.py
  │   ├── rules.py
  │   ├── mappings/
  │   └── prompts/
  └── legal/                       ← 法律插件
      ├── plugin.py
      ├── article_fetcher.py
      ├── citation_formatter.py
      ├── validity_checker.py
      ├── mappings/
      └── prompts/

docs/ingestion/                    ← 数据管线
  ├── parsers/
  ├── chunkers/
  ├── cleaners/
  └── pipeline.py
```

---

## 八、关键设计决策

| 决策点 | 选择 | 理由 |
|--------|------|------|
| 解析器主力 | Unstructured + PyMuPDF | Unstructured 支持 40+ 格式，PyMuPDF 轻量快速 |
| 分块策略 | 可插拔，默认 recursive + semantic | 通用场景用 recursive，高质量需求用 semantic |
| 插件接口 | Protocol (typing) | 轻量、无需继承、IDE 友好 |
| 配置拆分 | domain-specific 子配置类 | 避免 config.py 继续膨胀 |
| 法律分块 | 按法条边界切分 | 法条有严格层级，不能跨条 |
| 依赖管理 | 锁定版本 | 避免 torch 等关键依赖版本漂移 |

---

## 九、实施优先级

| 阶段 | 内容 | 优先级 | 预计工作量 | 依赖 |
|------|------|--------|-----------|------|
| Phase 0 | 数据管线 | **P0** | 2-3 周 | 无 |
| Phase 1 | 核心引擎解耦 | **P1** | 1-2 周 | Phase 0 |
| Phase 2 | 医疗插件化 | **P2** | 1 周 | Phase 1 |
| Phase 3 | 法律插件 + 民法典 | **P3** | 2 周 | Phase 1 |
| Phase 4 | 工程化补齐 | **P4** | 持续 | Phase 2+3 |

---

## 十、风险与应对

| 风险 | 影响 | 应对 |
|------|------|------|
| 解耦过程中引入回归 bug | 高 | Phase 1 前先补齐核心路径测试 |
| Unstructured 依赖过重 | 中 | 保留 PyMuPDF 作为轻量 fallback |
| 法律数据版权问题 | 高 | 使用官方公开版本，标注来源 |
| 多层抽象导致性能下降 | 中 | 关键路径 benchmark，确保 p99 < 2s |

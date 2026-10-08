# 中医AI健康Agent系统 (TCM Health AI Agent)

基于 **RAG + LangGraph Agent** 的中医体质辨识与养生知识智能助手。支持多轮对话、长期记忆、混合检索、多模态舌诊/体检报告解析，并通过 MCP 协议对外暴露检索能力。

> ⚠️ 本系统仅用于技术演示与学习，所有建议不构成医疗诊断或治疗依据。

---

## 一、系统架构

```
┌────────────────────────────────────────────────────────────────┐
│                     前端 Streamlit（frontend/app.py）           │
│   智能问答 · 体质辨识 · 健康画像 · 舌诊/报告上传 · 历史记录        │
└────────────────────────────┬───────────────────────────────────┘
                             │ REST（JSON） + SSE（流式）
┌────────────────────────────▼───────────────────────────────────┐
│                    FastAPI 后端（backend/main.py）              │
│   路由 + CORS + 请求级可观测（middleware 日志 / 耗时追踪）         │
└──────┬──────────────────┬────────────────────┬─────────────────┘
       │                  │                    │
┌──────▼───────┐   ┌──────▼────────┐   ┌───────▼──────────────┐
│  Agent 编排   │   │  多模态视觉     │   │  体质辨识规则引擎      │
│  LangGraph   │   │  qwen3.8-     │   │  9题问卷 + 评分       │
│  Function    │   │  omni-flash   │   │  9种体质映射          │
│  Calling     │   └───────────────┘   └──────────────────────┘
└──────┬───────┘
       │ 并行调用（ThreadPoolExecutor）
┌──────▼─────────────────────────────────────────────────────────┐
│                  工具层（backend/tools.py）                     │
│  search_herbs / search_formulas / search_diet /                │
│  search_constitution / get_user_profile                        │
└──────┬─────────────────────────────────────────────────────────┘
       │
┌──────▼─────────────────────────────────────────────────────────┐
│            RAG 检索管线（backend/rag_pipeline.py）              │
│  混合检索：向量(BM25) → RRF 融合 → Rerank(qwen3-rerank) 精排     │
└──────┬──────────────────────────────┬──────────────────────────┘
       │                              │
┌──────▼──────────┐          ┌────────▼─────────┐
│  Chroma 向量库   │          │  BM25 倒排索引    │
│  bge-large-zh   │          │  字符 n-gram 分词 │
└─────────────────┘          └──────────────────┘
┌────────────────────────────────────────────────────────────────┐
│                     记忆与存储层                                │
│  长期记忆四层：窗口→摘要→向量→持久化DB                           │
│  LangGraph Checkpoint（SqliteSaver）会话持久化                   │
│  业务数据：SQLite（开发默认）/ PostgreSQL（生产，DATABASE_URL 切换）│
└────────────────────────────────────────────────────────────────┘
```

**一次问答的调用链路**：用户提问 → 医疗意图安全拦截 → 长期记忆召回（向量）→ LLM 自主决定调哪些工具（Function Calling，可并行多调）→ 混合检索命中知识库 → RRF 融合 + Rerank 精排 → 溯源引用 → 生成回答 + 免责声明 → 持久化 + 记忆沉淀。

---

## 二、核心特性

| 模块 | 说明 |
|------|------|
| **混合检索** | 向量检索（bge-large-zh）+ BM25 稀疏检索（字符 1-gram/2-gram），RRF 融合 + cross-encoder 精排 |
| **Agent 编排** | LangGraph StateGraph + Function Calling，LLM 按语义自主决定调用工具，支持**多工具并行执行** |
| **长期记忆（四层）** | ① 窗口记忆（最近 N 轮）② 摘要记忆（超阈值 LLM 压缩）③ 向量记忆（LLM 提取记忆点 → 语义去重 → 按用户隔离召回）④ 持久化 DB |
| **多模态** | 舌诊 + 体检报告图片解析（qwen3.8-omni-flash，DashScope 兼容模式） |
| **MCP 协议** | `backend/mcp_server.py` 通过 stdio 暴露 `search_herbs` / `search_all`，供任意 MCP 客户端调用 |
| **医疗风控** | 医疗意图识别 + 确定性拒绝 + 免责声明 + 溯源引用 |
| **流式输出** | SSE 逐 token 推送，打字机效果 |
| **可观测性** | HTTP 请求级日志（method/path/status/耗时）+ Agent 调用链路日志（工具/检索库/耗时） |
| **会话持久化** | LangGraph Checkpoint（SqliteSaver）支持断点续跑 / 时间旅行 |

---

## 三、技术栈

| 层级 | 技术 |
|------|------|
| 前端 | Streamlit 1.60 + Plotly |
| 后端 | FastAPI + Pydantic（REST + SSE） |
| Agent | LangGraph + Function Calling |
| 检索 | Chroma + BM25 + Rerank（qwen3-rerank） |
| 嵌入 | bge-large-zh-v1.5 |
| LLM | DeepSeek-chat（主）/ Ollama qwen2.5（备） |
| 多模态 | qwen3.8-omni-flash（DashScope 兼容模式） |
| 协议 | MCP（模型上下文协议） |
| 数据库 | SQLite（开发）/ PostgreSQL（生产） |
| 测试 | pytest |

---

## 四、快速启动

### 前置要求
- Python 3.11+
- （生产用 PostgreSQL）Docker & Docker Compose
- API Key：`DEEPSEEK_API_KEY`（主 LLM）、`DASHSCOPE_API_KEY`（多模态 + Rerank）

### 方式一：本地开发（推荐）

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置环境变量
cp .env.example .env        # 编辑 .env 填入 DEEPSEEK_API_KEY / DASHSCOPE_API_KEY

# 3. 重建向量库（首次运行必须）
python scripts/rebuild_vectordb.py

# 4. 启动后端（新终端）
python -m uvicorn backend.main:app --reload --port 8000

# 5. 启动前端（另一终端）
python -m streamlit run frontend/app.py

# 6. 访问
# 前端: http://localhost:8501
# API 文档: http://localhost:8000/docs
```

### 方式二：Docker Compose

```bash
cp .env.example .env          # 填入 API Key
docker-compose up -d          # 后端 + 前端 + PostgreSQL
# 前端 http://localhost:8501，API http://localhost:8000/docs
```

### 环境变量（.env）

| 变量 | 说明 | 默认 |
|------|------|------|
| `DATABASE_URL` | 数据库连接串 | `sqlite:///./tcm_health.db` |
| `LLM_PROVIDER` | LLM 提供方 | `deepseek`（或 `ollama`） |
| `DEEPSEEK_API_KEY` | 主 LLM Key | — |
| `OLLAMA_HOST` / `OLLAMA_MODEL` | 本地 Ollama | `localhost:11434` / `qwen2.5:7b` |
| `DASHSCOPE_API_KEY` | 多模态 + Rerank Key | — |

---

## 五、召回率报告

在 **30 条标注测试集**上（中药 12 + 方剂 8 + 食疗 6 + 体质 4），混合检索（向量 + BM25 → RRF 融合 → Rerank）：

```
Hit Rate@5 : 30/30 = 100.00%
MRR        : 1.000        （全部命中文档排在第 1 位）
```

复现方式：

```bash
python scripts/rebuild_vectordb.py   # 重建向量库
python scripts/eval_recall.py        # 输出逐条命中明细 + Hit Rate@5 / MRR
pytest tests/test_recall.py -m slow  # 同测试集以 pytest 形式跑
```

---

## 六、测试

```bash
# 快测试（BM25 单元 + 安全拦截，无需 embedding）
pytest tests/ -m "not slow"

# 慢测试（召回率 + 长期记忆，需向量库与 embedding 模型）
pytest tests/test_recall.py tests/test_memory.py -m slow

# 全量
pytest
```

| 测试文件 | 覆盖 |
|----------|------|
| `tests/test_bm25.py` | 字符 n-gram 分词、BM25 索引构建/检索排序 |
| `tests/test_recall.py` | 30 条混合检索 Hit Rate@5 |
| `tests/test_memory.py` | 长期记忆写入/召回/去重/隔离/阈值过滤 |
| `tests/test_safety.py` | 医疗意图拦截、消息构建、来源提取（18 例） |

---

## 七、项目结构

```
tcm_health_agent/
├── backend/
│   ├── main.py              # FastAPI 入口 + 路由 + 请求级可观测
│   ├── agent_router.py      # LangGraph Agent（Function Calling 编排）
│   ├── rag_pipeline.py      # 混合检索管线（向量+BM25 → RRF → Rerank）
│   ├── bm25.py              # 字符 n-gram 分词 + Okapi BM25
│   ├── rerank.py            # cross-encoder 精排（qwen3-rerank）
│   ├── memory.py            # 长期记忆（提取/摘要/向量召回/去重）
│   ├── llm.py               # LLM 工厂（DeepSeek / Ollama）
│   ├── tools.py             # Agent 工具集（4 检索 + 用户画像）
│   ├── vision.py            # 多模态舌诊/体检报告
│   ├── mcp_server.py        # MCP 协议服务（stdio）
│   ├── safety.py            # 医疗风控 + 免责声明
│   ├── constitution.py      # 体质辨识规则引擎
│   ├── observability.py     # 结构化日志
│   ├── config.py / schemas.py / models.py / database.py
├── frontend/
│   └── app.py               # Streamlit 界面（问答/体质/画像/图片上传）
├── data/                    # 知识库（herbs / formulas / diet / constitution）
├── prompts/system_prompt.txt
├── scripts/
│   ├── rebuild_vectordb.py  # 重建向量库
│   ├── eval_recall.py       # 召回率评测
│   ├── eval_hybrid.py       # 混合检索 vs 纯向量对比
│   ├── gen_herbs.py / gen_formulas.py / gen_diet.py   # 知识库生成
│   └── test_*.py            # 各模块冒烟测试
├── tests/                   # pytest 测试套件
├── docker/  docker-compose.yml  init.sql
└── requirements.txt  pytest.ini
```

---

## 八、API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/users` | 创建用户 |
| GET | `/api/users/{id}` | 获取用户信息 |
| GET | `/api/constitution/questions` | 获取体质问卷题目 |
| POST | `/api/constitution/assess` | 提交问卷，返回体质结果 |
| POST | `/api/profile/update` | 更新健康画像（BMI 等） |
| POST | `/api/chat/query` | 智能问答（核心，非流式） |
| POST | `/api/chat/stream` | 智能问答（SSE 流式） |
| POST | `/api/vision/analyze` | 舌诊/体检报告图片解析 |
| GET | `/api/chat/history` | 获取对话历史 |
| GET | `/api/health` | 健康检查 |

---

## 九、部署说明

### 本地（开发）
见「快速启动 → 方式一」。SQLite 免安装，后端 `--reload` 热更新。

### 生产（PostgreSQL + Docker）
1. 在 `.env` 设置 `DATABASE_URL=postgresql://tcm_user:tcm_pass@db:5432/tcm_health`；
2. `docker-compose up -d`，PostgreSQL 由 `pgvector/pgvector` 镜像提供（含向量扩展预留）；
3. 后端启动时后台线程异步预热 embedding 模型，不阻塞服务就绪；
4. 建议用 gunicorn/uvicorn 多 worker 部署，日志接入 ELK/Loki（已按结构化格式输出）。

### 数据初始化
- 知识库语料在 `data/`，改动后执行 `python scripts/rebuild_vectordb.py`；
- 数据库表在启动时由 `init_db()` 自动建表（`Base.metadata.create_all`）。

---

## 十、安全声明

- 不提供药物处方、用药剂量，不做疾病诊断；
- 医疗意图命中时返回确定性拒绝话术；
- 所有回答末尾附医疗免责声明；
- 多模态舌诊/报告解读同样附「仅供参考，不能替代医生面诊」声明；
- 如有身体不适，请及时前往正规医院就诊。

---

## License

MIT

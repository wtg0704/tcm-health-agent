# 中医AI健康Agent系统 (TCM Health AI Agent)

基于RAG+LangGraph的中医体质辨识与养生知识智能助手。

## 项目概述

本系统是一个完整的AI应用，涵盖：
- 用户健康画像（BMI计算、体征录入）
- 中医体质辨识（9题标准问卷+规则引擎）
- RAG知识问答（4个独立知识库、分域检索、溯源引用）
- Agent智能路由（LangGraph编排、意图识别、多Tool并行调用）
- 安全防护（医疗意图识别、免责声明、内容质量控制）

## 技术栈

| 层级 | 技术 | 说明 |
|------|------|------|
| 前端 | Streamlit + Plotly | 多Tab界面、雷达图、BMI仪表盘 |
| 后端 | FastAPI + Pydantic | RESTful API、异步、自动文档 |
| Agent | LangGraph | StateGraph编排、条件路由、Tool Calling |
| 向量库 | Chroma | 4个独立Collection、分域检索 |
| 嵌入 | bge-large-zh-v1.5 | 中文语料最优嵌入模型 |
| LLM | DeepSeek-V3 (主) / Ollama qwen2.5 (备) | 双模式、环境变量切换 |
| 数据库 | PostgreSQL | 用户画像、体质记录、对话历史 |
| 部署 | Docker Compose | 一键启动全部服务 |

## 快速启动

### 前置要求
- Python 3.11+
- Docker & Docker Compose
- DeepSeek API Key (注册获取) 或 Ollama (本地模型)

### 方式一：Docker Compose（推荐）

```bash
# 1. 配置环境变量
cp .env.example .env
# 编辑 .env 填入 DEEPSEEK_API_KEY

# 2. 启动全部服务
docker-compose up -d

# 3. 访问
# 前端: http://localhost:8501
# 后端API文档: http://localhost:8000/docs
```

### 方式二：本地开发

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 启动PostgreSQL
docker run -d --name tcm_pg -p 5432:5432 \
  -e POSTGRES_USER=tcm_user \
  -e POSTGRES_PASSWORD=tcm_pass \
  -e POSTGRES_DB=tcm_health \
  pgvector/pgvector:pg16

# 3. 启动后端
uvicorn backend.main:app --reload --port 8000

# 4. 启动前端（新终端）
streamlit run frontend/app.py

# 5. 访问 http://localhost:8501
```

## 项目结构

```
tcm_health_agent/
├── backend/
│   ├── main.py              # FastAPI入口 + 路由
│   ├── config.py            # 配置管理
│   ├── schemas.py           # Pydantic数据模型
│   ├── database.py          # 数据库会话
│   ├── models.py            # SQLAlchemy ORM模型
│   ├── rag_pipeline.py      # RAG管线（加载→分块→embedding→检索）
│   ├── agent_router.py      # LangGraph Agent（意图路由→多Tool→汇总）
│   ├── constitution.py      # 体质辨识（9题+规则引擎）
│   └── safety.py            # 安全层（System Prompt+输出校验）
├── frontend/
│   └── app.py               # Streamlit多Tab界面
├── data/
│   ├── herbs/               # 中药知识库
│   ├── formulas/            # 方剂知识库
│   ├── diet/                # 食疗知识库
│   └── constitution/        # 体质知识库
├── prompts/
│   └── system_prompt.txt    # System Prompt模板
├── docker/
│   ├── Dockerfile
│   └── Dockerfile.frontend
├── docker-compose.yml
└── requirements.txt
```

## API接口

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/users` | 创建用户 |
| GET | `/api/users/{id}` | 获取用户信息 |
| GET | `/api/constitution/questions` | 获取体质问卷题目 |
| POST | `/api/constitution/assess` | 提交问卷，获取体质结果 |
| POST | `/api/profile/update` | 更新健康画像 |
| POST | `/api/chat/query` | 智能问答（核心接口） |
| GET | `/api/chat/history` | 获取对话历史 |

## 安全声明

本系统仅供养生知识参考，不构成医疗诊断或治疗建议。
- 不提供药物处方或用药剂量
- 不做出疾病诊断
- 所有建议末尾附医疗免责声明
- 如有身体不适，请前往正规医院就诊

## 后续规划

- [ ] 人形模特可视化（前端Canvas渲染体质状态）
- [ ] 中药图片多模态识别
- [ ] 历史体质趋势追踪（多次问卷对比）
- [ ] 向量库迁移Milvus（分布式检索）
- [ ] 前端升级Vue3 + 移动端适配
- [ ] 添加Rerank提升检索精度
- [ ] 混合检索（BM25 + 向量）

## License

MIT

---

⚠️ 本系统仅用于技术演示和学习，不提供医疗建议。

"""配置管理：环境变量、LLM配置、数据库连接字符串"""
import os
from pathlib import Path
from dotenv import load_dotenv

# 显式指向项目根的 .env（不从当前工作目录隐式查找，避免从别的目录启动时漏加载）
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# 数据库（本地开发默认SQLite，生产用PostgreSQL）
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite:///./tcm_health.db"  # SQLite默认值，方便本地开发
)

# LLM配置
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "deepseek")  # deepseek | ollama

# DeepSeek
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"

# Ollama
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")

# Embedding模型（用bge-large-zh做中文向量化）
EMBEDDING_MODEL = "BAAI/bge-large-zh-v1.5"

# Chroma
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
CHROMA_PERSIST_DIR = "./chroma_db"

# 知识库路径
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "prompts")

# 检索参数
RETRIEVAL_TOP_K = 5
RELEVANCE_THRESHOLD = 0.3  # 低于此阈值的片段不纳入生成

# 多模态（DashScope 兼容模式：舌诊 / 体检报告图片解析）
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DASHSCOPE_VL_MODEL = os.getenv("DASHSCOPE_VL_MODEL", "qwen3.8-omni-flash")

# Rerank 重排（cross-encoder 精排；兼容接口仅支持 qwen3-rerank）
DASHSCOPE_RERANK_MODEL = os.getenv("DASHSCOPE_RERANK_MODEL", "qwen3-rerank")

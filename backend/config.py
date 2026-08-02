"""配置管理：环境变量、LLM配置、数据库连接字符串"""
import os
from dotenv import load_dotenv

load_dotenv()

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

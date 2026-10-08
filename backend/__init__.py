# backend package
# 最先加载 .env（HF_HOME/HF_ENDPOINT 必须在 transformers/huggingface_hub 被 import 之前就绪，
# 否则会固化 huggingface.co 和默认 C 盘缓存路径，导致嵌入模型联网加载失败）
from . import config  # noqa: F401

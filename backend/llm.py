# -*- coding: utf-8 -*-
"""LLM 工厂：根据配置返回 LLM 实例（DeepSeek / Ollama）。

独立成模块的原因：agent_router（Agent 编排）和 memory（长期记忆提取/摘要）
都需要复用 LLM，放 config 里会混入业务，放 agent_router 里会造成循环依赖。
"""
from .config import (
    LLM_PROVIDER, DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL,
    OLLAMA_HOST, OLLAMA_MODEL,
)


def get_llm():
    """根据配置返回 LLM 实例"""
    if LLM_PROVIDER == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_HOST, temperature=0.3)
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        model=DEEPSEEK_MODEL,
        temperature=0.3,
    )

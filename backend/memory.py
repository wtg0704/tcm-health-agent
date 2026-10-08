# -*- coding: utf-8 -*-
"""长期记忆：向量记忆（语义检索）+ 摘要记忆（浓缩压缩）。

四层记忆架构（见 memory/「AI应用记忆管理四层架构」）中的第 2、3 层：

- 第 1 层「窗口记忆」：agent_router 里取最近 N 轮消息进 prompt —— 已实现
- 第 2 层「摘要记忆」：summarize_history() 把超长旧对话压缩成摘要 —— 本模块
- 第 3 层「向量记忆」：extract_memories() 用 LLM 从每轮对话提取「记忆点」，
  embedding 后存 Chroma（按 user_id 隔离），retrieve_memories() 语义召回 —— 本模块
- 第 4 层「持久化 DB」：ChatHistory 表落库 —— 已实现

设计取舍（面试可讲）：
- 向量记忆存的是「LLM 提取出的记忆点」而非对话原文：原文噪音大（寒暄、一次性问答），
  提取后检索更准、存得更省；这是 memory extraction 去噪，比 naive 全量存原文高级。
- 记忆提取是同步的（每次对话后追加调用 LLM + embedding）。生产环境应异步化
  （后台任务/消息队列），否则会拖慢单轮响应——当前为可读性保留同步，try/except 兜底。
- 按 user_id 隔离：一个用户一个 Chroma collection（memory_{user_id}），
  检索天然带隔离，不会串号（这是多租户记忆最朴素也最安全的做法）。
"""
import json
import os
from typing import Dict, List

from langchain_community.vectorstores import Chroma

from .config import CHROMA_PERSIST_DIR
from .rag_pipeline import get_embeddings
from .llm import get_llm

# 记忆向量库缓存：user_id -> Chroma（懒加载，避免每次检索重新建连接/重载 collection）
_memory_stores: Dict[str, Chroma] = {}

# 每次检索召回的记忆条数
MEMORY_TOP_K = 3

# 记忆相关性阈值：低于此相似度的记忆不注入 prompt（与知识库 RELEVANCE_THRESHOLD 一致）
_MEMORY_THRESHOLD = 0.30

# 语义去重阈值：新记忆与已有记忆的余弦相似度超过此值则跳过（不重复存）
_DEDUP_THRESHOLD = 0.90

# 记忆提取 prompt：让 LLM 只吐有长期价值的事实，忽略寒暄
_EXTRACT_PROMPT = """从下面的对话中提取值得长期记住的用户信息。
只提取有长期价值的事实（健康状况、体质、疾病史、过敏、饮食偏好、生活习惯、正在调理的问题等），
忽略寒暄、一次性问答和助手的客套话。

用户：{user_message}
助手：{assistant_answer}

只输出 JSON 数组，每项一个记忆点，例如 [{{"fact":"用户最近失眠"}},{{"fact":"用户是气虚质"}}]。
没有值得记住的就输出 []。不要输出 JSON 以外的任何内容。"""

# 摘要 prompt：把旧对话压缩成摘要，保留关键信息
_SUMMARY_PROMPT = """把下面的对话压缩成简洁摘要，保留用户提到的关键信息（健康问题、体质、偏好、重要问答结论），不要遗漏用户的重要信息。

对话：
{conversation}

摘要："""


def _get_store(user_id: str) -> Chroma:
    """获取（或创建）某用户的记忆向量库，按 user_id 隔离 collection"""
    global _memory_stores
    if user_id not in _memory_stores:
        collection = f"memory_{user_id}"
        persist_dir = os.path.join(CHROMA_PERSIST_DIR, collection)
        _memory_stores[user_id] = Chroma(
            persist_directory=persist_dir,
            embedding_function=get_embeddings(),
            collection_name=collection,
        )
    return _memory_stores[user_id]


def retrieve_memories(user_id: str, query: str, top_k: int = MEMORY_TOP_K) -> List[str]:
    """语义检索用户的历史记忆，返回相关记忆点文本列表（失败时静默降级为空）"""
    try:
        store = _get_store(user_id)
        if store._collection.count() == 0:
            return []
        docs = store.similarity_search_with_relevance_scores(query, k=top_k)
        # 相关性阈值过滤：只保留真正相关的记忆，避免无关记忆污染 prompt
        return [d.page_content for d, score in docs if score >= _MEMORY_THRESHOLD]
    except Exception as e:
        print(f"[memory] 检索记忆失败: {e}")
        return []


def add_memories(user_id: str, facts: List[str]) -> int:
    """把记忆点写入向量库（带语义去重），返回实际新增条数"""
    if not facts:
        return 0
    store = _get_store(user_id)
    added = 0
    for fact in facts:
        fact = fact.strip()
        if not fact:
            continue
        # 语义去重：若已有高度相似记忆则跳过
        try:
            if store._collection.count() > 0:
                dup = store.similarity_search_with_relevance_scores(fact, k=1)
                if dup and dup[0][1] >= _DEDUP_THRESHOLD:
                    continue
        except Exception:
            pass
        store.add_texts([fact])
        added += 1
    return added


def extract_memories(user_message: str, assistant_answer: str) -> List[str]:
    """用 LLM 从一轮对话中提取记忆点（JSON 数组 -> List[str]），失败静默返回空"""
    try:
        llm = get_llm()
        prompt = _EXTRACT_PROMPT.format(
            user_message=user_message, assistant_answer=assistant_answer
        )
        resp = llm.invoke(prompt)
        text = resp.content.strip()
        # 剥离 LLM 可能包裹的 ```json ... ``` 代码块
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:]
        data = json.loads(text)
        if isinstance(data, list):
            return [
                item.get("fact", "")
                for item in data
                if isinstance(item, dict) and item.get("fact")
            ]
        return []
    except Exception as e:
        print(f"[memory] 提取记忆失败: {e}")
        return []


def extract_and_save_memory(user_id: str, user_message: str, assistant_answer: str) -> List[str]:
    """提取并保存记忆（每轮对话结束后调用），返回保存的记忆点"""
    facts = extract_memories(user_message, assistant_answer)
    add_memories(user_id, facts)
    return facts


def summarize_history(messages: List[Dict], keep_recent: int = 6) -> str:
    """把旧对话压缩成摘要。

    messages: [{"role": "user"/"assistant", "message": "..."}]，按时间正序。
    keep_recent: 保留最近 N 条原文（默认 6 条 ≈ 最近 3 轮），更早的压缩成摘要。
    返回摘要文本；消息数不超过 keep_recent 时返回空串（无需压缩）。
    """
    if len(messages) <= keep_recent:
        return ""
    old = messages[:-keep_recent]
    conversation = "\n".join(f"{m['role']}: {m['message']}" for m in old)
    try:
        llm = get_llm()
        resp = llm.invoke(_SUMMARY_PROMPT.format(conversation=conversation))
        return resp.content.strip()
    except Exception as e:
        print(f"[memory] 摘要失败: {e}")
        return ""

# -*- coding: utf-8 -*-
"""Rerank 精排：用 cross-encoder 模型对召回候选逐条判相关性，重排。

两阶段检索架构（工业界 RAG 标准做法）：
- 召回（recall，粗排）：BM25 + 向量混合检索，保证高召回（宁可多召回，不能漏）
- 精排（rerank）：qwen3-rerank 是 cross-encoder，把 query 和每个候选拼接后精确打分，
  精度远高于 bi-encoder 的向量相似度，用于把真正相关的排到最前（提升 MRR）

DashScope 的 OpenAI 兼容 rerank 端点有几个坑（实际踩过，面试可讲）：
1. 路径是 /reranks（复数），单数 /rerank 会 404；
2. 基础路径是 compatible-api/v1，与 embedding 用的 compatible-mode/v1 不同；
3. 请求体是扁平结构 {model, query, documents, top_n}，不是 DashScope 原生嵌套结构。
"""
from typing import List, Dict

import requests

from .config import DASHSCOPE_API_KEY, DASHSCOPE_RERANK_MODEL

# 注意：rerank 的兼容根路径是 compatible-api/v1，不是 compatible-mode/v1
RERANK_BASE_URL = "https://dashscope.aliyuncs.com/compatible-api/v1"


def rerank(query: str, candidates: List[Dict], top_n: int = None) -> List[Dict]:
    """对召回候选做精排，返回按 relevance_score 降序的结果。

    Args:
        query: 用户查询
        candidates: 混合检索结果 [{doc, excerpt, score, collection}, ...]
        top_n: 保留前 N 条（None=全部）

    Returns:
        重排后的候选列表，score 字段替换为 rerank 的 relevance_score。
        未配置 key 或调用失败时降级为原顺序（不影响可用性）。
    """
    if not candidates:
        return []
    if not DASHSCOPE_API_KEY:
        return candidates

    documents = [c.get("excerpt", "") for c in candidates]
    try:
        resp = requests.post(
            f"{RERANK_BASE_URL}/reranks",
            headers={"Authorization": f"Bearer {DASHSCOPE_API_KEY}"},
            json={
                "model": DASHSCOPE_RERANK_MODEL,
                "query": query,
                "documents": documents,
                "top_n": top_n or len(documents),
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"[rerank] 重排失败，降级为原顺序: {e}")
        return candidates

    results = data.get("results", [])
    reordered = []
    seen = set()
    for r in results:
        idx = r.get("index", 0)
        if 0 <= idx < len(candidates):
            c = dict(candidates[idx])
            c["score"] = round(r.get("relevance_score", 0.0), 4)
            reordered.append(c)
            seen.add(idx)

    # 兜底：rerank 未返回的候选（异常情况）按原顺序补上
    if top_n is None and len(reordered) < len(candidates):
        for i, c in enumerate(candidates):
            if i not in seen:
                reordered.append(c)

    return reordered[:top_n] if top_n else reordered


def hybrid_search_with_rerank(query: str, collections: List[str] = None,
                              top_k: int = None, recall_k: int = None) -> List[Dict]:
    """两阶段检索：混合召回（多取）→ rerank 精排（取 top_k）。

    这是完整的 RAG 检索链路：
      阶段1 召回：BM25 + 向量混合，多召回候选（recall_k = top_k * 4），保证不漏
      阶段2 精排：cross-encoder rerank，把候选按 query-doc 相关性重排，取 top_k

    局部 import 避免与 rag_pipeline 的循环依赖。
    """
    from .rag_pipeline import hybrid_search
    from .config import RETRIEVAL_TOP_K

    top_k = top_k or RETRIEVAL_TOP_K
    recall_k = recall_k or top_k * 4
    candidates = hybrid_search(query, collections, recall_k)
    return rerank(query, candidates, top_n=top_k)

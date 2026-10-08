# -*- coding: utf-8 -*-
"""BM25 稀疏检索：字符级 n-gram 分词 + Okapi BM25 打分。

为什么需要 BM25（与向量检索互补）：
- 向量检索擅长「语义相似」——"补气的药"能召回黄芪，即使文本里没出现"黄芪"；
- 但嵌入向量对「专有名词 / 精确词项」可能漂移，比如"四君子汤""血府逐瘀汤"这类长方名，
  查询与文档用词完全一致时，字面精确匹配（BM25）反而最可靠；
- 混合检索 = 两者取长补短，用 RRF 融合排序，召回率更稳。

分词选择「字符 1-gram + 2-gram」而非 jieba 词典分词：中医术语（黄芪、痰湿质、君药）
不在通用词典里，jieba 容易切错；字符 n-gram 无词典依赖，对专有名词天然鲁棒。
"""
import math
import re
from collections import defaultdict
from typing import List, Tuple, Dict

# Okapi BM25 超参数
K1 = 1.5   # 词频饱和参数
B = 0.75   # 文档长度归一化参数


def tokenize(text: str) -> List[str]:
    """中文分词：字符 1-gram + 2-gram（保留中英文数字，去掉标点空白）"""
    text = re.sub(r"[^一-龥A-Za-z0-9]+", "", text)
    if not text:
        return []
    unigrams = list(text)
    bigrams = [text[i:i + 2] for i in range(len(text) - 1)]
    return unigrams + bigrams


class BM25Index:
    """单个知识库的 BM25 索引。

    构建时输入所有 chunk 的 (id, 文本)，检索返回按 BM25 分数降序的 [(chunk_id, score), ...]。
    """

    def __init__(self, ids: List[str], docs: List[str]):
        self.ids = list(ids)
        self.docs = list(docs)
        self.N = len(self.docs)

        self.doc_tokens = [tokenize(d) for d in self.docs]
        self.doc_len = [len(t) for t in self.doc_tokens]
        self.avgdl = (sum(self.doc_len) / self.N) if self.N else 0.0

        # 倒排索引：term -> {doc_idx: term_frequency}
        self.postings: Dict[str, Dict[int, int]] = defaultdict(dict)
        self.df: Dict[str, int] = defaultdict(int)

        for idx, tokens in enumerate(self.doc_tokens):
            tf: Dict[str, int] = defaultdict(int)
            for t in tokens:
                tf[t] += 1
            for t, f in tf.items():
                self.postings[t][idx] = f
                self.df[t] += 1

    def _idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        # 平滑 IDF，恒为正
        return math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """检索，返回 [(chunk_id, bm25_score), ...] 按分数降序"""
        q_tokens = tokenize(query)
        if not q_tokens or self.N == 0:
            return []

        scores: Dict[int, float] = defaultdict(float)
        for t in q_tokens:
            idf = self._idf(t)
            if idf == 0.0:
                continue
            for idx, tf in self.postings.get(t, {}).items():
                dl = self.doc_len[idx]
                denom = tf + K1 * (1 - B + B * dl / self.avgdl)
                scores[idx] += idf * (tf * (K1 + 1)) / denom

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        return [(self.ids[idx], score) for idx, score in ranked]

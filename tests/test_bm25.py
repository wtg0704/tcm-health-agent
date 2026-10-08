# -*- coding: utf-8 -*-
"""BM25 稀疏检索单元测试（纯逻辑，无 embedding，快）"""
from backend.bm25 import tokenize, BM25Index


class TestTokenize:
    """字符 1-gram + 2-gram 分词"""

    def test_chinese_bigram_unigram(self):
        # "黄芪" -> unigram ['黄','芪'] + bigram ['黄芪']
        assert tokenize("黄芪") == ["黄", "芪", "黄芪"]

    def test_strips_punctuation(self):
        # 中文标点被剥离，仅保留中英文数字
        assert tokenize("黄芪，性温！") == ["黄", "芪", "性", "温", "黄芪", "芪性", "性温"]

    def test_mixed_alnum(self):
        tokens = tokenize("vitaminB12")
        assert "vi" in tokens and "B1" in tokens

    def test_empty_after_strip(self):
        assert tokenize("，。！？") == []


class TestBM25Index:
    """Okapi BM25 索引构建与检索"""

    def test_build_and_search_rank(self):
        ids = ["d1", "d2", "d3"]
        docs = ["黄芪补气固表", "当归补血活血", "黄芪当归同用补气血"]
        idx = BM25Index(ids, docs)
        hits = idx.search("黄芪", top_k=2)
        assert hits, "应返回检索结果"
        assert hits[0][0] in ("d1", "d3"), "命中含黄芪的文档"
        scores = [s for _, s in hits]
        assert scores == sorted(scores, reverse=True), "分数应降序"

    def test_exact_term_match(self):
        # 精确词项命中应排第一（BM25 对专有名词可靠）
        idx = BM25Index(["d1", "d2"], ["四君子汤由人参白术茯苓甘草组成", "人参大补元气"])
        hits = idx.search("四君子汤", top_k=1)
        assert hits[0][0] == "d1"

    def test_empty_index(self):
        assert BM25Index([], []).search("黄芪") == []

    def test_no_match_returns_empty(self):
        idx = BM25Index(["d1"], ["黄芪补气"])
        assert idx.search("xyzxyz") == []

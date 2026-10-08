# -*- coding: utf-8 -*-
"""召回率评测：混合检索 Hit Rate@5（依赖向量库，标记 slow）

前置：先跑 python scripts/rebuild_vectordb.py 重建向量库。
测试集与 scripts/eval_recall.py 一致，共 30 条。
"""
import pytest

from backend.rag_pipeline import hybrid_search

pytestmark = pytest.mark.slow

# 测试集：(query, 检索的知识库, 期望命中的 source_label)
TESTSET = [
    # ===== 中药（12条）=====
    ("黄芪有什么功效", "herbs", "herbs/黄芪"),
    ("人参能补气吗", "herbs", "herbs/人参"),
    ("当归治什么", "herbs", "herbs/当归"),
    ("枸杞能明目吗", "herbs", "herbs/枸杞"),
    ("金银花清热解毒吗", "herbs", "herbs/金银花"),
    ("三七能止血吗", "herbs", "herbs/三七"),
    ("酸枣仁治失眠吗", "herbs", "herbs/酸枣仁"),
    ("茯苓祛湿吗", "herbs", "herbs/茯苓"),
    ("菊花清肝明目吗", "herbs", "herbs/菊花"),
    ("丹参活血吗", "herbs", "herbs/丹参"),
    ("陈皮化痰吗", "herbs", "herbs/陈皮"),
    ("百合润肺吗", "herbs", "herbs/百合"),
    # ===== 方剂（8条）=====
    ("四君子汤由哪几味药组成", "formulas", "formulas/四君子汤"),
    ("四物汤治什么", "formulas", "formulas/四物汤"),
    ("六味地黄丸有什么功效", "formulas", "formulas/六味地黄丸"),
    ("补中益气汤治什么", "formulas", "formulas/补中益气汤"),
    ("归脾汤养心吗", "formulas", "formulas/归脾汤"),
    ("逍遥散疏肝解郁吗", "formulas", "formulas/逍遥散"),
    ("二陈汤化痰吗", "formulas", "formulas/二陈汤"),
    ("玉屏风散治什么", "formulas", "formulas/玉屏风散"),
    # ===== 食疗（6条）=====
    ("红枣桂圆茶怎么泡", "diet", "diet/红枣桂圆茶"),
    ("红豆薏米粥祛湿吗", "diet", "diet/红豆薏米粥"),
    ("银耳百合羹润肺吗", "diet", "diet/银耳百合羹"),
    ("酸枣仁茶助眠吗", "diet", "diet/酸枣仁茶"),
    ("玫瑰花茶疏肝吗", "diet", "diet/玫瑰花茶"),
    ("生姜红糖茶驱寒吗", "diet", "diet/生姜红糖茶"),
    # ===== 体质（4条）=====
    ("什么是气虚质", "constitution", "constitution/九种体质详解"),
    ("痰湿质有什么特征", "constitution", "constitution/九种体质详解"),
    ("阴虚质怎么调理", "constitution", "constitution/九种体质详解"),
    ("气虚质吃什么食物", "constitution", "constitution/体质食疗原则"),
]


@pytest.mark.parametrize("query,col,expected", TESTSET)
def test_hybrid_recall_hit_at_5(query, col, expected):
    """每条标注 query 的 top5 混合检索结果应命中期望文档"""
    results = hybrid_search(query, [col], top_k=5)
    docs = [r["doc"] for r in results]
    assert expected in docs, f"「{query}」top5 应命中 {expected}，实际 {docs}"

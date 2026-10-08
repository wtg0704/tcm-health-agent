# -*- coding: utf-8 -*-
"""召回率评测：30 条标注测试集，计算 Hit Rate@5 / MRR，并输出逐条命中明细。

用法：python scripts/eval_recall.py
前置：先跑 python scripts/rebuild_vectordb.py 重建向量库。

指标说明：
- Hit Rate@5：top5 检索结果中命中「期望文档」的 query 占比（0~1，越高越好）
- MRR：命中文档排名的倒数均值（衡量正确结果是否排得靠前）
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# Windows 终端默认 GBK，中文/emoji 会乱码或抛 UnicodeEncodeError，统一输出 UTF-8
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

from backend.rag_pipeline import hybrid_search
from backend.config import RETRIEVAL_TOP_K

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


def evaluate():
    hit = 0
    mrr_sum = 0.0
    rows = []
    for query, col, expected in TESTSET:
        results = hybrid_search(query, [col], top_k=RETRIEVAL_TOP_K)
        docs = [r["doc"] for r in results]
        scores = [r["score"] for r in results]
        hit_flag = expected in docs
        rank = docs.index(expected) + 1 if hit_flag else None
        if hit_flag:
            hit += 1
            mrr_sum += 1.0 / rank
        rows.append((query, col, expected, hit_flag, rank, scores))

    total = len(TESTSET)
    print(f"评测配置：top_k={RETRIEVAL_TOP_K}（混合检索 RRF 融合）\n")
    print(f"{'query':<22} {'库':<12} {'期望文档':<24} {'命中':<4} {'排名':<4} {'top5分数'}")
    print("-" * 100)
    for query, col, expected, hit_flag, rank, scores in rows:
        mark = "✓" if hit_flag else "✗"
        print(f"{query:<22} {col:<12} {expected:<24} {mark:<4} {str(rank or '-'):<4} {scores}")

    hit_rate = hit / total
    mrr = mrr_sum / total
    print("\n" + "=" * 100)
    print(f"Hit Rate@5 : {hit}/{total} = {hit_rate:.2%}")
    print(f"MRR        : {mrr:.3f}")
    print("=" * 100)
    return {"hit_rate": hit_rate, "mrr": mrr, "total": total, "hit": hit}


if __name__ == "__main__":
    evaluate()

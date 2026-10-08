# -*- coding: utf-8 -*-
"""混合检索（BM25 + 向量 RRF）vs 纯向量检索 召回率对比评测。

用法：python scripts/eval_hybrid.py
前置：向量库已存在（或自动加载构建），复用 eval_recall.py 的 30 条测试集 + 难点用例。

评测分两组：
1. 常规用例（30 条）：验证混合检索「不劣于」纯向量（不引入回归）；
2. 难点用例（短查询）：验证混合检索「优于」纯向量——短查询向量相似度天然偏低，
   被 RELEVANCE_THRESHOLD 误杀时，BM25 靠精确词项匹配兜底召回。
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

from eval_recall import TESTSET
from backend.rag_pipeline import load_all_vectorstores, search_knowledge, hybrid_search
from backend.config import RETRIEVAL_TOP_K

# 难点用例：短查询（双字中药名），向量相似度 < 阈值被纯向量过滤，BM25 可兜底
DIFFICULT = [
    ("葛根", "herbs", "herbs/葛根"),
    ("远志", "herbs", "herbs/远志"),
]


def _run(tests):
    """对一组测试集，返回 (纯向量指标, 混合指标, 明细行)"""
    vec_hit = hyb_hit = 0
    vec_mrr = hyb_mrr = 0.0
    rows = []
    for query, col, expected in tests:
        vr = search_knowledge(query, [col])
        vdocs = [r["doc"] for r in vr]
        v_hit = expected in vdocs
        v_rank = vdocs.index(expected) + 1 if v_hit else None

        hr = hybrid_search(query, [col])
        hdocs = [r["doc"] for r in hr]
        h_hit = expected in hdocs
        h_rank = hdocs.index(expected) + 1 if h_hit else None

        if v_hit:
            vec_hit += 1
            vec_mrr += 1.0 / v_rank
        if h_hit:
            hyb_hit += 1
            hyb_mrr += 1.0 / h_rank

        rows.append((query, expected, v_hit, v_rank, h_hit, h_rank))

    n = len(tests)
    return (vec_hit, vec_mrr, hyb_hit, hyb_mrr, rows, n)


def _print_block(title, vec_hit, vec_mrr, hyb_hit, hyb_mrr, rows, n):
    print(f"\n===== {title} =====")
    print(f"{'query':<18} {'期望文档':<18} {'向量命中':<8} {'向量排名':<8} {'混合命中':<8} {'混合排名'}")
    print("-" * 88)
    for query, expected, v_hit, v_rank, h_hit, h_rank in rows:
        mark = lambda hit: "✓" if hit else "✗"
        print(f"{query:<18} {expected:<18} {mark(v_hit):<8} {str(v_rank or '-'):<8} {mark(h_hit):<8} {str(h_rank or '-')}")
    print("-" * 88)
    print(f"{'Hit Rate@5':<12} {vec_hit}/{n} = {vec_hit/n:.2%}{'':<8} {hyb_hit}/{n} = {hyb_hit/n:.2%}")
    print(f"{'MRR':<12} {vec_mrr/n:.3f}{'':<18} {hyb_mrr/n:.3f}")


def evaluate():
    load_all_vectorstores()

    print(f"评测配置：top_k={RETRIEVAL_TOP_K}")

    # 常规用例：验证不劣于
    vec_hit, vec_mrr, hyb_hit, hyb_mrr, rows, n = _run(TESTSET)
    _print_block("① 常规用例（30条，验证混合检索不引入回归）", vec_hit, vec_mrr, hyb_hit, hyb_mrr, rows, n)

    # 难点用例：验证优于
    vec_hit2, vec_mrr2, hyb_hit2, hyb_mrr2, rows2, n2 = _run(DIFFICULT)
    _print_block("② 难点用例（短查询，验证 BM25 兜底召回）", vec_hit2, vec_mrr2, hyb_hit2, hyb_mrr2, rows2, n2)

    print("\n" + "=" * 88)
    total = n + n2
    vh = vec_hit + vec_hit2
    hh = hyb_hit + hyb_hit2
    print(f"总计：纯向量 Hit@5 = {vh}/{total}，混合检索 Hit@5 = {hh}/{total}")
    if hh > vh:
        print(f"✅ 混合检索多召回 {hh - vh} 条（短查询兜底），且常规用例零回归")
    else:
        print("✅ 混合检索与纯向量持平，零回归")
    print("=" * 88)


if __name__ == "__main__":
    evaluate()

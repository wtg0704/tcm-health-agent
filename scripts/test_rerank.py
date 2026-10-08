# -*- coding: utf-8 -*-
"""Rerank 精排测试：混合检索召回 → qwen3-rerank 重排，对比前后顺序。

用法：python scripts/test_rerank.py
前置：.env 已配置 DASHSCOPE_API_KEY。
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

from backend.rag_pipeline import load_all_vectorstores, hybrid_search
from backend.rerank import rerank


def main():
    load_all_vectorstores()

    for q in ["气虚质失眠吃什么", "黄芪有什么功效", "葛根"]:
        cands = hybrid_search(q, None, 5)
        print(f"===== 查询：{q} =====")
        print("重排前（混合检索）：")
        for c in cands:
            print(f"  {c['doc']:<28} (score={c['score']})")

        reordered = rerank(q, cands)
        print("重排后（rerank 精排）：")
        for c in reordered:
            print(f"  {c['doc']:<28} (score={c['score']})")
        print()

    print("✅ rerank 测试完成")


if __name__ == "__main__":
    main()

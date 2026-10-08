# -*- coding: utf-8 -*-
"""重建所有知识库向量库（data/ 目录内容更新后运行一次）。

用法：python scripts/rebuild_vectordb.py
会重新对 data/{herbs,formulas,diet,constitution} 下所有 .md 分块、embedding、写入 chroma_db/。
"""
from backend.rag_pipeline import init_all_vectorstores


def main():
    print("开始重建向量库...")
    vss = init_all_vectorstores()
    print("\n===== 重建结果 =====")
    for col, vs in vss.items():
        print(f"  {col}: {vs._collection.count()} 条")
    print("重建完成")


if __name__ == "__main__":
    main()

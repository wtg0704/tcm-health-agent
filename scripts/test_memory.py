# -*- coding: utf-8 -*-
"""长期记忆（向量记忆 + 摘要）端到端测试。

测试点：
1. 记忆写入 + 语义检索（不依赖 LLM，直接 add_memories 验证 Chroma 存储链路）
2. 语义去重（高度相似记忆不重复存）
3. 按 user_id 隔离（两个用户互不可见）
4. extract_memories 真实 LLM 提取（可选，用 --extract 开启）

运行：python scripts/test_memory.py [--extract]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Windows GBK 控制台 emoji 修复
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

from backend.memory import (
    add_memories, retrieve_memories, extract_memories,
)


def test_store_and_retrieve():
    uid = "test_user_1"
    print("\n== 1. 写入 + 语义检索 ==")
    add_memories(uid, [
        "用户最近失眠，入睡困难",
        "用户是气虚体质，容易疲劳",
        "用户喜欢喝菊花茶",
    ])
    hits = retrieve_memories(uid, "我睡不着", top_k=2)
    print("查询「我睡不着」召回：")
    for h in hits:
        print("  -", h)
    assert any("失眠" in h or "睡" in h for h in hits), "应召回失眠相关记忆"


def test_dedup():
    uid = "test_user_1"
    print("\n== 2. 语义去重 ==")
    n = add_memories(uid, ["用户最近失眠，入睡困难"])  # 与已有记忆高度相似
    print(f"重复写入相似记忆，新增 {n} 条（应为 0）")
    assert n == 0, "语义去重未生效"


def test_isolation():
    uid_a, uid_b = "test_user_1", "test_user_2"
    print("\n== 3. 按 user_id 隔离 ==")
    add_memories(uid_b, ["用户对海鲜过敏"])
    hits_b = retrieve_memories(uid_b, "过敏", top_k=5)
    hits_a = retrieve_memories(uid_a, "过敏", top_k=5)
    print(f"B 查「过敏」: {hits_b}")
    print(f"A 查「过敏」: {hits_a}")
    assert any("海鲜" in h for h in hits_b), "B 应召回自己的过敏记忆"
    assert not any("海鲜" in h for h in hits_a), "A 不应看到 B 的记忆"
    # 阈值过滤：A 没有「过敏」相关记忆，应返回空（而非硬塞不相关记忆）
    assert hits_a == [], f"A 查「过敏」应因阈值过滤返回空，实际 {hits_a}"


if __name__ == "__main__":
    if "--extract" in sys.argv:
        print("\n== 4. LLM 提取记忆（真实 DeepSeek 调用）==")
        facts = extract_memories(
            "我最近老是失眠，还有点气虚，喝点黄芪行吗",
            "黄芪补气升阳，适合气虚体质，可配伍党参、白术。失眠可再考虑酸枣仁。"
        )
        print("提取到的记忆点：", facts)

    test_store_and_retrieve()
    test_dedup()
    test_isolation()
    print("\n全部通过")

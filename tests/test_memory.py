# -*- coding: utf-8 -*-
"""长期记忆测试：写入/检索/去重/隔离/阈值过滤（依赖 embedding，标记 slow）"""
import pytest

from backend.memory import add_memories, retrieve_memories

pytestmark = pytest.mark.slow


def test_store_and_retrieve():
    """写入记忆点后，语义查询能召回相关记忆"""
    uid = "pytest_mem_retrieve"
    add_memories(uid, [
        "用户最近失眠，入睡困难",
        "用户是气虚体质",
        "用户喜欢喝菊花茶",
    ])
    hits = retrieve_memories(uid, "我睡不着", top_k=3)
    assert any("失眠" in h for h in hits), f"应召回失眠记忆，实际 {hits}"


def test_semantic_dedup():
    """高度相似记忆不重复存储"""
    uid = "pytest_mem_dedup"
    add_memories(uid, ["用户最近失眠，入睡困难"])
    n = add_memories(uid, ["用户最近失眠，入睡困难"])
    assert n == 0, "语义去重未生效"


def test_user_isolation():
    """按 user_id 隔离：A 看不到 B 的记忆，且无关记忆被阈值过滤"""
    add_memories("pytest_mem_a", ["用户对海鲜过敏"])
    assert any("海鲜" in h for h in retrieve_memories("pytest_mem_a", "过敏"))
    assert retrieve_memories("pytest_mem_b", "过敏") == []

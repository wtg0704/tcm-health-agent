# -*- coding: utf-8 -*-
"""长期记忆端到端：跑一轮 run_agent，验证（1）主流程不报错（2）记忆被提取落库并可语义召回。

运行：python scripts/test_memory_e2e.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

from backend.agent_router import run_agent
from backend.memory import retrieve_memories

uid = "e2e_memory_demo"

print("== 第1轮：告知个人情况（走 Function Calling + 记忆提取）==")
r1 = run_agent(uid, "我最近晚上总是睡不着，白天没精神，平时还容易累")
print("回答:", (r1["answer"] or "")[:100], "...\n")

print("== 验证：记忆是否被提取落库，且可语义召回 ==")
for q in ["失眠", "疲劳", "我的体质"]:
    hits = retrieve_memories(uid, q, top_k=3)
    print(f"  查「{q}」召回: {hits if hits else '(无)'}")

assert any(hits for q in ["失眠", "疲劳"] for hits in [retrieve_memories(uid, q, top_k=3)]), \
    "至少应召回一条本轮提取的记忆"

print("\n== 第2轮：追问之前的问题（验证注入记忆后主流程仍正常）==")
r2 = run_agent(uid, "我前面说的睡眠问题，有什么食疗建议吗？")
print("回答:", (r2["answer"] or "")[:120], "...\n")

print("端到端通过")

# -*- coding: utf-8 -*-
"""MCP Server（stdio）：把中医知识库检索能力暴露成标准 MCP 工具。

运行：
    python backend/mcp_server.py          # 从项目根
    或 python -m backend.mcp_server

MCP 客户端配置示例（Claude Desktop 的 claude_desktop_config.json / Cursor 的 mcp.json）：
    {
      "mcpServers": {
        "tcm-kb": {
          "command": "python",
          "args": ["backend/mcp_server.py"],
          "cwd": "E:/yuyi-cc/tcm_health_agent"
        }
      }
    }

设计要点：
- 复用 backend.rag_pipeline 的 hybrid_search/get_retrieval_context，检索逻辑与 Agent 内一致；
- stdio 模式下 stdout 是 JSON-RPC 协议通道，所有日志/print 必须走 stderr；
- 知识库在启动时加载一次（embedding 模型 + 4 个向量库）。
"""
import os
import sys
import contextlib

# 允许 `python backend/mcp_server.py` 与 `python -m backend.mcp_server` 两种方式运行
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp.server.mcpserver import MCPServer

from backend.rag_pipeline import load_all_vectorstores, hybrid_search, get_retrieval_context

# 启动时加载知识库（embedding 模型 + 4 个向量库，约 30 秒）。
# 把其内部 print 重定向到 stderr，避免污染 stdio 的协议通道。
print("[mcp-server] 正在加载知识库...", file=sys.stderr)
with contextlib.redirect_stdout(sys.stderr):
    load_all_vectorstores()
print("[mcp-server] 知识库加载完成，MCP Server 就绪", file=sys.stderr)

mcp = MCPServer("中医知识检索")


@mcp.tool()
def search_herbs(query: str) -> str:
    """检索「中药」知识库，查询单味中药的性味归经、功效、禁忌等。适用：用户询问某味中药（如"黄芪有什么功效"）。"""
    return get_retrieval_context(hybrid_search(query, ["herbs"]))


@mcp.tool()
def search_formulas(query: str) -> str:
    """检索「方剂」知识库，查询方剂的组成、功效、适用症等。适用：用户询问某个方剂（如"四君子汤由哪几味药组成"）。"""
    return get_retrieval_context(hybrid_search(query, ["formulas"]))


@mcp.tool()
def search_diet(query: str) -> str:
    """检索「食疗」知识库，查询食疗方案、做法、适合体质。适用：用户问吃什么/食疗推荐（如"气虚质失眠吃什么"）。"""
    return get_retrieval_context(hybrid_search(query, ["diet"]))


@mcp.tool()
def search_constitution(query: str) -> str:
    """检索「体质」知识库，查询九种体质的特征、成因、调理方向。适用：用户询问体质概念或调理（如"什么是痰湿质"）。"""
    return get_retrieval_context(hybrid_search(query, ["constitution"]))


@mcp.tool()
def search_all(query: str) -> str:
    """跨库综合检索：同时检索中药/方剂/食疗/体质四个知识库。适用：问题涉及多个领域或不确定属于哪类。"""
    return get_retrieval_context(
        hybrid_search(query, ["herbs", "formulas", "diet", "constitution"])
    )


if __name__ == "__main__":
    mcp.run()  # stdio 传输

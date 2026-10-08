# -*- coding: utf-8 -*-
"""MCP Server 端到端测试：通过 stdio 协议启动 server → 握手 → 列工具 → 调用工具。

用法：python scripts/test_mcp.py
验证 backend/mcp_server.py 能被标准 MCP 客户端通过协议发现并调用。
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# Windows 终端默认 GBK，中文/emoji 会乱码或抛 UnicodeEncodeError，统一输出 UTF-8
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


async def main():
    params = StdioServerParameters(
        command=sys.executable,
        args=["backend/mcp_server.py"],
        cwd=PROJECT,
        env=None,
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            print("=== MCP Server 注册的工具 ===")
            for t in tools.tools:
                print(f"  - {t.name}")

            print("\n=== 调用 search_herbs('黄芪有什么功效') ===")
            result = await session.call_tool("search_herbs", {"query": "黄芪有什么功效"})
            for c in result.content:
                text = getattr(c, "text", str(c))
                print(text[:300])

            print("\n=== 调用 search_all('气虚质失眠吃什么') ===")
            result = await session.call_tool("search_all", {"query": "气虚质失眠吃什么"})
            for c in result.content:
                text = getattr(c, "text", str(c))
                print(text[:400])

    print("\nMCP 端到端测试通过 ✅")


if __name__ == "__main__":
    asyncio.run(main())

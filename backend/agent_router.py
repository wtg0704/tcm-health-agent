"""LangGraph Agent：Function Calling（LLM 自主调工具）+ 并行工具执行 + Checkpoint 状态管理 + 医疗风控。

架构演进：
- 旧版：intent_classifier 用 if-elif 关键词硬编码路由到不同工具（串行）
- 新版：LLM 通过 bind_tools 拿到工具 schema，根据语义自主决定调用哪些工具（可并行多调）
- 新增：SqliteSaver checkpoint 状态持久化，支持断点续跑 / 时间旅行（thread_id 会话）
"""
import os
import re
import time
from typing import List, Dict, Any, TypedDict, Annotated
from concurrent.futures import ThreadPoolExecutor

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import (
    HumanMessage, AIMessage, SystemMessage, ToolMessage, BaseMessage,
)
from .llm import get_llm
from .memory import retrieve_memories, extract_and_save_memory, summarize_history
from .safety import (
    load_system_prompt, wrap_disclaimer,
    is_medical_query, medical_reject_response,
)
from .tools import build_tools, TOOL_COLLECTIONS
from .database import SessionLocal
from .models import ChatHistory
from .observability import get_logger


# ========== Checkpoint 状态持久化 ==========
# checkpoint 数据库文件（项目根，与 tcm_health.db 同级）
_CHECKPOINT_DB = os.path.join(os.path.dirname(__file__), "..", "agent_checkpoints.db")


def get_checkpointer():
    """返回 SqliteSaver 上下文管理器：每次请求独立连接，状态持久化到文件。

    用途：多轮对话状态持久化（服务重启不丢）、断点续跑、时间旅行回放。
    """
    return SqliteSaver.from_conn_string(_CHECKPOINT_DB)


# ========== State定义 ==========

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    user_id: str
    final_answer: str
    sources: List[Dict]
    agent_trace: Dict


# 工具调用循环最大轮数（防止 LLM 反复调工具导致死循环）
MAX_ITERATIONS = 5

_DISCLAIMER = "⚠️ 以上为养生参考建议，不构成医疗诊断。如有身体不适，请及时前往正规医院就诊。"

# Agent 调用链路日志（结构化，便于追溯每次调用的工具与检索库）
agent_logger = get_logger("tcm.agent")


# ========== 消息构建 ==========

# 摘要触发阈值：历史超过 12 条（6 轮）才把更早的消息压缩成摘要
_SUMMARY_TRIGGER = 12


def build_messages(message: str, conversation_history: List[Dict] = None,
                   summary: str = "") -> List[BaseMessage]:
    """构建对话消息（不含 System Prompt，system 由 call_model / 流式路径动态注入）。

    summary 非空时，作为「摘要记忆」以独立 SystemMessage 注入到窗口历史之前。
    这样 checkpoint 里只累积纯对话，不会重复堆叠 system prompt。
    """
    messages: List[BaseMessage] = []
    if summary:
        messages.append(SystemMessage(content=f"[对话历史摘要]\n{summary}"))
    if conversation_history:
        for h in conversation_history[-6:]:  # 只保留最近3轮（6条消息）——窗口记忆
            if h["role"] == "user":
                messages.append(HumanMessage(content=h["message"]))
            else:
                messages.append(AIMessage(content=h["message"]))
    messages.append(HumanMessage(content=message))
    return messages


def _with_system(messages: List[BaseMessage]) -> List[BaseMessage]:
    """在消息列表前注入 System Prompt"""
    return [SystemMessage(content=load_system_prompt())] + list(messages)


def _system_with_memory(user_id: str, query: str) -> str:
    """构建带「向量记忆」的 System Prompt：基础系统提示 + 语义召回的该用户历史记忆"""
    base = load_system_prompt()
    memories = retrieve_memories(user_id, query, top_k=3)
    if memories:
        base += "\n\n## 用户历史记忆（你与用户过往对话中记住的信息，回答时可自然引用）\n" + \
                "\n".join(f"- {m}" for m in memories)
    return base


def _build_summary(conversation_history: List[Dict]) -> str:
    """对话超过阈值时，把更早的旧消息压缩成摘要（解决长对话 token 膨胀）。

    生产优化点：摘要应在历史变化时才重算并缓存，这里为可读性每次现算，
    仅在超过阈值时触发，避免常规短对话多一次 LLM 调用。
    """
    if not conversation_history or len(conversation_history) <= _SUMMARY_TRIGGER:
        return ""
    return summarize_history(conversation_history, keep_recent=6)


# ========== 工具执行（并行）==========

def execute_tools_parallel(tools, tool_calls) -> List[ToolMessage]:
    """并行执行多个工具调用，返回 ToolMessage 列表。

    LLM 一次返回多个 tool_calls 时（如同时检索食疗+体质+中药），
    用线程池并发执行，而不是串行等待，这就是「多工具链并行编排」。
    """
    tool_map = {t.name: t for t in tools}

    def run_one(tc):
        t = tool_map[tc["name"]]
        result = t.invoke(tc["args"])
        return ToolMessage(content=str(result), tool_call_id=tc["id"], name=tc["name"])

    if len(tool_calls) == 1:
        return [run_one(tool_calls[0])]
    with ThreadPoolExecutor(max_workers=len(tool_calls)) as ex:
        return list(ex.map(run_one, tool_calls))


# ========== sources / trace 提取 ==========

_SOURCE_PATTERN = re.compile(
    r"\[来源\d+\]\s*(.+?)（相关度:\s*[\d.]+\）\n(.+?)(?=\n\n\[来源\d+\]|\Z)",
    re.DOTALL,
)


def extract_sources(messages: List[BaseMessage]) -> List[Dict]:
    """从工具返回结果中提取结构化引用来源（doc + excerpt），供前端展示与持久化"""
    sources = []
    seen = set()
    for m in messages:
        if not isinstance(m, ToolMessage):
            continue
        content = m.content or ""
        for doc, excerpt in _SOURCE_PATTERN.findall(content):
            doc = doc.strip()
            if doc in seen:
                continue
            seen.add(doc)
            sources.append({"doc": doc, "excerpt": excerpt.strip()[:200]})
    return sources


def build_trace(messages: List[BaseMessage]) -> Dict:
    """从消息列表重建 Agent 调用链路（便于调试追溯 + 前端展示）"""
    tools_called = []
    for m in messages:
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
            for tc in m.tool_calls:
                tools_called.append(tc["name"])

    searched_collections = []
    for name in tools_called:
        for col in TOOL_COLLECTIONS.get(name, []):
            if col not in searched_collections:
                searched_collections.append(col)

    return {
        "mode": "function_calling",
        "tools_called": tools_called,
        "searched_collections": searched_collections,
    }


# ========== LangGraph 图（Function Calling 循环）==========

def build_agent(user_id: str, checkpointer=None, system: str = None):
    """构建 Function Calling Agent 图：agent(LLM决策) ⇄ tools(并行执行)

    图结构：
        entry → agent ──有 tool_calls──→ tools → agent（循环）
                  └──无 tool_calls──→ END

    checkpointer 传入时，图按 thread_id 持久化状态，支持断点续跑/时间旅行。
    system 传入时作为 System Prompt（可带长期记忆），否则用默认系统提示。
    """
    tools = build_tools(user_id)
    system = system or load_system_prompt()

    def call_model(state: AgentState) -> Dict:
        """agent 节点：LLM bind_tools，根据语义自主决定调用哪些工具"""
        llm = get_llm().bind_tools(tools)
        response = llm.invoke([SystemMessage(content=system)] + list(state["messages"]))
        return {"messages": [response]}

    def should_continue(state: AgentState) -> str:
        """条件路由：最后一条 AI 消息含 tool_calls 则继续调工具，否则结束"""
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
            return "tools"
        return END

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", ToolNode(tools))
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    workflow.add_edge("tools", "agent")
    return workflow.compile(checkpointer=checkpointer)


# ========== 对话持久化 ==========

def save_chat_history(user_id: str, user_message: str, answer: str,
                      sources: List[Dict], agent_trace: Dict):
    """保存对话记录到数据库"""
    db = SessionLocal()
    try:
        db.add(ChatHistory(user_id=user_id, role="user", message=user_message))
        db.add(ChatHistory(
            user_id=user_id, role="assistant", message=answer,
            sources=sources, agent_trace=agent_trace,
        ))
        db.commit()
    finally:
        db.close()


# ========== 非流式执行（走 LangGraph 图）==========

def run_agent(user_id: str, message: str, conversation_history: List[Dict] = None,
              thread_id: str = None) -> Dict:
    """执行Agent（非流式）：医疗风控前置拦截 → Function Calling 图 → 持久化

    Args:
        user_id: 用户ID
        message: 用户消息
        conversation_history: 传统模式的历史消息（前端传参）
        thread_id: checkpoint 模式会话ID；传入时历史由 checkpointer 恢复，忽略 conversation_history

    Returns:
        {"answer": "...", "sources": [...], "agent_trace": {...}, "disclaimer": "..."}
    """
    start = time.perf_counter()
    # 医疗意图快速拦截（确定性安全过滤，第一道防线）
    if is_medical_query(message):
        answer = wrap_disclaimer(medical_reject_response(message))
        trace = {"mode": "function_calling", "tools_called": ["safety_filter"], "searched_collections": []}
        save_chat_history(user_id, message, answer, [], trace)
        agent_logger.info("agent_call user=%s tools=[safety_filter] collections=[] duration=%.0fms",
                          user_id, (time.perf_counter() - start) * 1000)
        return {"answer": answer, "sources": [], "agent_trace": trace, "disclaimer": _DISCLAIMER}

    initial: AgentState = {
        "messages": [],
        "user_id": user_id,
        "final_answer": "",
        "sources": [],
        "agent_trace": {},
    }

    if thread_id:
        # checkpoint 模式：历史由 checkpointer 恢复，只传当前消息
        with get_checkpointer() as checkpointer:
            agent = build_agent(user_id, checkpointer=checkpointer,
                                system=_system_with_memory(user_id, message))
            initial["messages"] = build_messages(message)
            config = {"configurable": {"thread_id": thread_id}}
            result = agent.invoke(initial, config)
    else:
        # 传统模式：前端传 conversation_history
        summary = _build_summary(conversation_history)
        agent = build_agent(user_id, system=_system_with_memory(user_id, message))
        initial["messages"] = build_messages(message, conversation_history, summary=summary)
        result = agent.invoke(initial)

    all_messages = result["messages"]
    raw_answer = all_messages[-1].content
    final_answer = wrap_disclaimer(raw_answer)
    sources = extract_sources(all_messages)
    agent_trace = build_trace(all_messages)

    save_chat_history(user_id, message, final_answer, sources, agent_trace)

    # 长期记忆：提取并保存本轮记忆点（生产应异步化，这里同步 + 兜底）
    extract_and_save_memory(user_id, message, raw_answer)

    agent_logger.info("agent_call user=%s tools=%s collections=%s duration=%.0fms",
                      user_id, agent_trace["tools_called"], agent_trace["searched_collections"],
                      (time.perf_counter() - start) * 1000)

    return {
        "answer": final_answer,
        "sources": sources,
        "agent_trace": agent_trace,
        "disclaimer": _DISCLAIMER,
    }


# ========== 流式执行（手动 Tool Calling 循环 + 流式生成）==========

def run_agent_stream(user_id: str, message: str, conversation_history: List[Dict] = None):
    """流式执行Agent——逐token返回，供SSE推流使用（打字机效果）。

    与 run_agent 走 LangGraph 图不同，这里手动串联同样的 Function Calling 逻辑，
    目的是让最终答案生成阶段可以用 llm.stream() 逐 token 产出：
      阶段1 工具调用循环：非流式 invoke 快速决策调哪些工具 + 并行执行
      阶段2 最终生成：llm.stream 逐 token 推流

    Yields:
        {"type": "token", "content": "文"}   # 增量token
        {"type": "done", "sources": [...], "agent_trace": {...}}  # 结束标记+元数据
    """
    start = time.perf_counter()
    # 医疗意图快速拦截
    if is_medical_query(message):
        answer = wrap_disclaimer(medical_reject_response(message))
        trace = {"mode": "function_calling", "tools_called": ["safety_filter"], "searched_collections": []}
        save_chat_history(user_id, message, answer, [], trace)
        agent_logger.info("agent_stream user=%s tools=[safety_filter] collections=[] duration=%.0fms",
                          user_id, (time.perf_counter() - start) * 1000)
        yield {"type": "token", "content": answer}
        yield {"type": "done", "sources": [], "agent_trace": trace}
        return

    tools = build_tools(user_id)
    llm_with_tools = get_llm().bind_tools(tools)
    summary = _build_summary(conversation_history)
    messages = [SystemMessage(content=_system_with_memory(user_id, message))] + \
               build_messages(message, conversation_history, summary=summary)

    # 阶段1：工具调用循环（非流式，快速）
    # 推一个状态事件，消除"等待期无反馈"的空窗（用户在 6 秒决策期内看到进度）
    yield {"type": "status", "content": "🧠 正在理解您的问题…\n\n"}
    for _ in range(MAX_ITERATIONS):
        ai_msg = llm_with_tools.invoke(messages)
        if not getattr(ai_msg, "tool_calls", None):
            break
        messages.append(ai_msg)
        yield {"type": "status", "content": "🔍 正在检索中医知识库…\n\n"}
        messages.extend(execute_tools_parallel(tools, ai_msg.tool_calls))

    # 阶段2：流式生成最终答案（不带工具，专注生成）
    llm = get_llm()
    full_response = ""
    try:
        for chunk in llm.stream(messages):
            token = chunk.content
            if token:
                full_response += token
                yield {"type": "token", "content": token}
    except Exception as e:
        err = f"\n\n[生成中断: {str(e)}]"
        full_response += err
        yield {"type": "token", "content": err}

    # 追加免责声明
    final_answer = wrap_disclaimer(full_response)
    disclaimer_suffix = final_answer[len(full_response):]
    if disclaimer_suffix:
        yield {"type": "token", "content": disclaimer_suffix}

    # 提取 sources + 调用链路
    sources = extract_sources(messages)
    agent_trace = build_trace(messages)

    # 持久化
    save_chat_history(user_id, message, final_answer, sources, agent_trace)

    # 长期记忆：提取并保存本轮记忆点
    extract_and_save_memory(user_id, message, full_response)

    agent_logger.info("agent_stream user=%s tools=%s collections=%s duration=%.0fms",
                      user_id, agent_trace["tools_called"], agent_trace["searched_collections"],
                      (time.perf_counter() - start) * 1000)

    yield {"type": "done", "sources": sources, "agent_trace": agent_trace}

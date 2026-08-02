"""LangGraph Agent路由：意图识别 → 多Tool调用 → 结果汇总"""
from typing import List, Dict, Any, TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from .config import (
    LLM_PROVIDER, DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL,
    OLLAMA_HOST, OLLAMA_MODEL
)
from .safety import load_system_prompt, wrap_disclaimer
from .rag_pipeline import search_knowledge, get_retrieval_context
from .database import SessionLocal
from .models import User, HealthProfile, ConstitutionResult, ChatHistory
import json


# ========== LLM初始化 ==========

def get_llm():
    """根据配置返回LLM实例"""
    if LLM_PROVIDER == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_HOST, temperature=0.3)
    else:
        return ChatOpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
            model=DEEPSEEK_MODEL,
            temperature=0.3,
        )


# ========== State定义 ==========

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    user_id: str
    intent: str
    searched_collections: List[str]
    tools_called: List[str]
    retrieval_results: List[Dict]
    user_profile: Dict
    constitution: Dict
    final_answer: str
    sources: List[Dict]
    agent_trace: Dict


# ========== Agent节点 ==========

def intent_classifier(state: AgentState) -> AgentState:
    """意图识别节点：分析用户想干什么"""
    last_message = state["messages"][-1].content

    # 用关键词+简单规则做意图分类（避免多一次LLM调用）
    msg_lower = last_message.lower()

    # 医疗意图检测（安全优先级最高）
    medical_keywords = [
        "吃什么药", "开个方子", "怎么治疗", "确诊", "治病", "处方",
        "剂量", "能治好吗", "多久能好", "不吃药行吗", "手术", "打针",
        "抗生素", "消炎药", "止疼药", "降压药", "降糖药"
    ]
    is_medical = any(kw in msg_lower for kw in medical_keywords)

    if is_medical:
        state["intent"] = "medical_reject"
        state["tools_called"] = ["safety_filter"]
        state["searched_collections"] = []
    elif any(kw in msg_lower for kw in ["体质", "什么质", "辨识"]):
        state["intent"] = "constitution"
        state["tools_called"] = ["get_constitution"]
        state["searched_collections"] = ["constitution"]
    elif any(kw in msg_lower for kw in ["吃", "食谱", "食疗", "推荐", "喝什么", "茶", "汤"]):
        state["intent"] = "diet_recommend"
        state["tools_called"] = ["get_user_profile", "search_diet", "search_herbs"]
        state["searched_collections"] = ["diet", "herbs"]
    elif any(kw in msg_lower for kw in ["方剂", "方子", "配伍", "配方", "汤剂"]):
        state["intent"] = "formula_query"
        state["tools_called"] = ["search_formulas", "search_herbs"]
        state["searched_collections"] = ["formulas", "herbs"]
    elif any(kw in msg_lower for kw in ["中药", "药材", "黄芪", "当归", "人参", "枸杞",
                                          "菊花", "陈皮", "茯苓", "党参", "阿胶"]):
        state["intent"] = "herb_query"
        state["tools_called"] = ["search_herbs"]
        state["searched_collections"] = ["herbs"]
    else:
        state["intent"] = "general_knowledge"
        state["tools_called"] = ["search_all_knowledge"]
        state["searched_collections"] = ["herbs", "formulas", "diet", "constitution"]

    return state


def medical_reject(state: AgentState) -> AgentState:
    """医疗问题拒绝节点"""
    last_message = state["messages"][-1].content
    # 提取可能的养生话题用于引导
    health_topics = {
        "头疼": "头痛的日常调理和穴位按摩",
        "失眠": "改善睡眠的食疗和养生方法",
        "胃": "脾胃调理的饮食建议",
        "湿气": "祛湿的日常方法",
        "感冒": "增强免疫力的养生方式",
        "咳嗽": "润肺的食疗推荐",
    }
    guide = "中医养生知识"
    for keyword, topic in health_topics.items():
        if keyword in last_message:
            guide = topic
            break

    state["final_answer"] = (
        f"关于疾病治疗和用药问题，我无法提供医疗建议。建议您咨询正规医院的中医师。\n\n"
        f"我可以为您介绍相关的养生知识——比如{guide}。您想了解哪方面呢？\n\n"
        f"---\n"
        f"⚠️ 以上为养生参考建议，不构成医疗诊断。如有身体不适，请及时前往正规医院就诊。"
    )
    state["sources"] = []
    return state


def retrieve_user_data(state: AgentState) -> AgentState:
    """获取用户画像和体质信息"""
    user_id = state["user_id"]
    db = SessionLocal()
    try:
        # 获取最近一次体质结果
        constitution = db.query(ConstitutionResult).filter(
            ConstitutionResult.user_id == user_id
        ).order_by(ConstitutionResult.created_at.desc()).first()

        # 获取健康画像
        profile = db.query(HealthProfile).filter(
            HealthProfile.user_id == user_id
        ).first()

        if constitution:
            state["constitution"] = {
                "type": constitution.constitution_type,
                "scores": constitution.scores,
            }
        else:
            state["constitution"] = {}

        if profile:
            state["user_profile"] = {
                "height_cm": profile.height_cm,
                "weight_kg": profile.weight_kg,
                "bmi": profile.bmi,
                "sleep_quality": profile.sleep_quality,
                "exercise_frequency": profile.exercise_frequency,
            }
        else:
            state["user_profile"] = {}
    finally:
        db.close()
    return state


def search_tools(state: AgentState) -> AgentState:
    """执行知识库检索"""
    collections = state.get("searched_collections", [])
    last_message = state["messages"][-1].content
    results = search_knowledge(last_message, collections)
    state["retrieval_results"] = results

    # 格式化sources用于返回
    state["sources"] = [
        {"doc": r["doc"], "excerpt": r["excerpt"][:200]}
        for r in results
    ]
    return state


def generate_answer(state: AgentState) -> AgentState:
    """调用LLM生成回答"""
    intent = state["intent"]
    results = state.get("retrieval_results", [])
    constitution = state.get("constitution", {})
    profile = state.get("user_profile", {})

    # 构建上下文
    context = get_retrieval_context(results)

    # 构建用户信息提示
    user_info = ""
    if constitution:
        user_info += f"\n用户体质类型：{constitution.get('type', '未知')}"
    if profile:
        bmi = profile.get("bmi", "")
        bmi_str = f"{bmi:.1f}" if bmi else "未录入"
        user_info += f"\n身高：{profile.get('height_cm', '未录入')}cm"
        user_info += f"\n体重：{profile.get('weight_kg', '未录入')}kg"
        user_info += f"\nBMI：{bmi_str}"
    if not user_info:
        user_info = "\n（用户尚未完成体质辨识和健康画像，可建议其先完成问卷）"

    system_prompt = load_system_prompt() + f"\n\n## 当前用户信息{user_info}"

    # 构建消息列表
    messages = [HumanMessage(content=system_prompt)]
    messages.extend(state["messages"])

    llm = get_llm()
    # 附加上下文
    if context and context != "（知识库中未找到相关信息）":
        augmented_message = HumanMessage(content=f"请基于以下知识库内容回答问题：\n\n{context}\n\n用户问题：{state['messages'][-1].content}")
        messages = [HumanMessage(content=system_prompt), augmented_message]
    else:
        messages = [HumanMessage(content=system_prompt), state["messages"][-1]]

    response = llm.invoke(messages)
    answer = response.content

    # 包装免责声明
    state["final_answer"] = wrap_disclaimer(answer)
    state["agent_trace"] = {
        "intent": intent,
        "searched_collections": state.get("searched_collections", []),
        "tools_called": state.get("tools_called", []),
    }
    return state


def save_chat_history(state: AgentState) -> AgentState:
    """保存对话记录到数据库"""
    user_id = state["user_id"]
    db = SessionLocal()
    try:
        # 保存用户消息
        user_msg = ChatHistory(
            user_id=user_id,
            role="user",
            message=state["messages"][-1].content,
        )
        db.add(user_msg)

        # 保存AI回复
        ai_msg = ChatHistory(
            user_id=user_id,
            role="assistant",
            message=state["final_answer"],
            sources=state.get("sources"),
            agent_trace=state.get("agent_trace"),
        )
        db.add(ai_msg)
        db.commit()
    finally:
        db.close()
    return state


# ========== 构建Agent图 ==========

def build_agent():
    """构建LangGraph Agent"""
    workflow = StateGraph(AgentState)

    # 添加节点
    workflow.add_node("classify", intent_classifier)
    workflow.add_node("medical_reject", medical_reject)
    workflow.add_node("retrieve_user", retrieve_user_data)
    workflow.add_node("search", search_tools)
    workflow.add_node("generate", generate_answer)
    workflow.add_node("save", save_chat_history)

    # 设置入口
    workflow.set_entry_point("classify")

    # 条件路由：医疗意图直接拒绝
    def route_after_classify(state: AgentState) -> str:
        if state["intent"] == "medical_reject":
            return "medical_reject"
        return "retrieve_user"

    workflow.add_conditional_edges("classify", route_after_classify, {
        "medical_reject": "medical_reject",
        "retrieve_user": "retrieve_user",
    })

    # 正常流程
    workflow.add_edge("retrieve_user", "search")
    workflow.add_edge("search", "generate")
    workflow.add_edge("generate", "save")
    workflow.add_edge("save", END)

    # 拒绝流程直接结束
    workflow.add_edge("medical_reject", "save")
    # 注意：medical_reject → save 然后 save → END

    return workflow.compile()


# 全局Agent实例
_agent = None


def get_agent():
    """获取Agent实例（懒加载）"""
    global _agent
    if _agent is None:
        _agent = build_agent()
    return _agent


def run_agent(user_id: str, message: str, conversation_history: List[Dict] = None) -> Dict:
    """执行Agent

    Args:
        user_id: 用户ID
        message: 用户消息
        conversation_history: [{"role": "user/assistant", "message": "..."}]

    Returns:
        {"answer": "...", "sources": [...], "agent_trace": {...}, "disclaimer": "..."}
    """
    agent = get_agent()

    # 构建历史消息
    messages = []
    if conversation_history:
        for h in conversation_history[-6:]:  # 只保留最近3轮（6条消息）
            if h["role"] == "user":
                messages.append(HumanMessage(content=h["message"]))
            else:
                messages.append(AIMessage(content=h["message"]))
    messages.append(HumanMessage(content=message))

    # 执行Agent
    initial_state: AgentState = {
        "messages": messages,
        "user_id": user_id,
        "intent": "",
        "searched_collections": [],
        "tools_called": [],
        "retrieval_results": [],
        "user_profile": {},
        "constitution": {},
        "final_answer": "",
        "sources": [],
        "agent_trace": {},
    }

    result = agent.invoke(initial_state)

    return {
        "answer": result["final_answer"],
        "sources": result["sources"],
        "agent_trace": result["agent_trace"],
        "disclaimer": "⚠️ 以上为养生参考建议，不构成医疗诊断。如有身体不适，请及时前往正规医院就诊。",
    }

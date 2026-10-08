"""Agent 工具定义：知识库检索 + 用户画像查询，封装为标准 Tool 供 LLM Function Calling 调用。

与旧版关键词路由的区别：
- 旧版：intent_classifier 用 if-elif 关键词硬编码决定调哪个工具
- 新版：LLM 通过 bind_tools 拿到这些工具的 schema，根据语义自主决定调用哪些工具（可并行多调）
"""
from typing import List
from langchain_core.tools import tool

from .rag_pipeline import hybrid_search, get_retrieval_context
from .database import SessionLocal
from .models import ConstitutionResult, HealthProfile


# ========== 知识库检索工具（静态，全局复用）==========

@tool
def search_herbs(query: str) -> str:
    """检索「中药」知识库，查询单味中药的功效、性味归经、使用禁忌等。

    适用场景：用户询问某味中药，例如"黄芪有什么功效""枸杞能天天吃吗"。
    """
    results = hybrid_search(query, ["herbs"])
    return get_retrieval_context(results)


@tool
def search_formulas(query: str) -> str:
    """检索「方剂」知识库，查询方剂的组成、功效、适用症等。

    适用场景：用户询问某个方剂，例如"四君子汤由哪几味药组成"。
    """
    results = hybrid_search(query, ["formulas"])
    return get_retrieval_context(results)


@tool
def search_diet(query: str) -> str:
    """检索「食疗」知识库，查询食疗方案、做法、适合体质等。

    适用场景：用户问吃什么/喝什么/食疗推荐，例如"气虚质失眠吃什么""湿气重喝什么茶"。
    """
    results = hybrid_search(query, ["diet"])
    return get_retrieval_context(results)


@tool
def search_constitution(query: str) -> str:
    """检索「体质」知识库，查询九种体质的特征、成因、调理方向等。

    适用场景：用户询问体质概念或调理，例如"什么是痰湿质""阴虚质怎么调理"。
    """
    results = hybrid_search(query, ["constitution"])
    return get_retrieval_context(results)


# 静态检索工具（可全局复用）
RETRIEVAL_TOOLS: List = [search_herbs, search_formulas, search_diet, search_constitution]

# 工具名 -> 检索的知识库，供调用链路追溯使用
TOOL_COLLECTIONS = {
    "search_herbs": ["herbs"],
    "search_formulas": ["formulas"],
    "search_diet": ["diet"],
    "search_constitution": ["constitution"],
    "get_user_profile": [],
}


# ========== 用户画像工具（闭包绑定 user_id，每次请求动态生成）==========

def make_profile_tool(user_id: str):
    """生成「查询用户画像」工具，闭包绑定当前 user_id。

    user_id 通过闭包注入，不暴露给 LLM（工具的 schema 中无此参数）。
    """
    @tool
    def get_user_profile() -> str:
        """查询当前用户的体质辨识结果和健康画像（身高/体重/BMI/睡眠/运动）。

        适用场景：用户问题涉及"我"的体质、个人身体情况、或需要个性化建议时，先调用本工具获取用户信息。
        """
        db = SessionLocal()
        try:
            constitution = db.query(ConstitutionResult).filter(
                ConstitutionResult.user_id == user_id
            ).order_by(ConstitutionResult.created_at.desc()).first()

            profile = db.query(HealthProfile).filter(
                HealthProfile.user_id == user_id
            ).first()

            parts = []
            if constitution:
                parts.append(f"体质类型：{constitution.constitution_type}")
                if constitution.scores:
                    top3 = sorted(constitution.scores.items(), key=lambda kv: -kv[1])[:3]
                    parts.append("体质评分(前三)： " + "、".join(f"{k}{v}分" for k, v in top3))
            else:
                parts.append("体质类型：未辨识（用户还没做体质问卷）")

            if profile:
                if profile.height_cm:
                    parts.append(f"身高：{profile.height_cm}cm")
                if profile.weight_kg:
                    parts.append(f"体重：{profile.weight_kg}kg")
                if profile.bmi:
                    parts.append(f"BMI：{profile.bmi:.1f}")
                if profile.sleep_quality:
                    parts.append(f"睡眠质量：{profile.sleep_quality}/5")
                if profile.exercise_frequency:
                    parts.append(f"每周运动：{profile.exercise_frequency}次")
            else:
                parts.append("健康画像：未录入")

            return "\n".join(parts) if parts else "用户尚未录入任何健康信息"
        finally:
            db.close()

    return get_user_profile


def build_tools(user_id: str) -> List:
    """构建某次请求的工具集：4 个检索工具 + 1 个用户画像工具（绑定 user_id）"""
    return RETRIEVAL_TOOLS + [make_profile_tool(user_id)]

"""安全层：医疗风控（关键词黑名单 + LLM语义兜底）+ System Prompt模板 + 输出免责声明"""
import os
from .config import PROMPTS_DIR


# ========== 医疗风控 ==========
# 关键词黑名单：确定性快速拦截诊疗/用药类请求（第一道防线，与LLM语义判断形成双重保险）
MEDICAL_KEYWORDS = [
    "吃什么药", "开个方子", "怎么治疗", "确诊", "治病", "处方",
    "剂量", "能治好吗", "多久能好", "不吃药行吗", "手术", "打针",
    "抗生素", "消炎药", "止疼药", "降压药", "降糖药",
]

# 命中关键词后，把拒绝转化成合规的养生话题引导
_HEALTH_TOPIC_MAP = {
    "头疼": "头痛的日常调理和穴位按摩",
    "失眠": "改善睡眠的食疗和养生方法",
    "胃": "脾胃调理的饮食建议",
    "湿气": "祛湿的日常方法",
    "感冒": "增强免疫力的养生方式",
    "咳嗽": "润肺的食疗推荐",
}


def is_medical_query(text: str) -> bool:
    """医疗意图快速检测：确定性拦截诊疗/用药类请求"""
    if not text:
        return False
    return any(kw in text for kw in MEDICAL_KEYWORDS)


def medical_reject_response(text: str) -> str:
    """医疗问题拒绝回复：拒绝 + 引导到合规养生话题"""
    guide = "中医养生知识"
    for keyword, topic in _HEALTH_TOPIC_MAP.items():
        if keyword in text:
            guide = topic
            break
    return (
        f"关于疾病治疗和用药问题，我无法提供医疗建议。建议您咨询正规医院的中医师。\n\n"
        f"我可以为您介绍相关的养生知识——比如{guide}。您想了解哪方面呢？"
    )


# ========== System Prompt ==========

def load_system_prompt() -> str:
    """加载System Prompt模板"""
    prompt_path = os.path.join(PROMPTS_DIR, "system_prompt.txt")
    if os.path.exists(prompt_path):
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()
    return _default_system_prompt()


def _default_system_prompt() -> str:
    """默认System Prompt（内嵌备用）"""
    return """你是"中医养生顾问"，专注于中医体质调理和养生知识科普。

## 你的能力
1. 基于用户体质类型给出个性化养生建议
2. 回答中医知识问题（中药功效、方剂组成、食疗方案、穴位按摩）
3. 解释中医概念（如"湿气""气虚""经络"等）
4. 推荐适合用户体质的日常饮食和生活习惯

## 工具使用（重要）
你可以调用工具获取准确信息，不要凭空编造知识库内容：
- get_user_profile：查询当前用户的体质类型、健康画像（身高/体重/BMI/睡眠/运动）。当用户问题涉及"我"的体质、个人身体情况、需要个性化建议时，先调用它。
- search_herbs / search_formulas / search_diet / search_constitution：分别检索中药、方剂、食疗、体质四类知识库。回答专业知识前，先调用对应工具检索，基于检索结果回答；同一问题可并行调用多个工具（如"气虚质失眠吃什么"应同时检索食疗和体质）。

## 回答规则
1. 引用知识库内容时标注来源
2. 使用通俗易懂的语言，必要时解释专业术语
3. 如果知识库没有相关信息，诚实说明
4. 如果存在多种观点，并列说明

## 安全约束（绝对不可违反）
你绝对不能：
- 提供药物处方、药品推荐、具体用药剂量
- 做出疾病诊断（"你这是XX病"）
- 建议用户停用现有治疗方案
- 暗示你可以替代医生

你可以：
- 提供养生知识科普
- 推荐食疗方案和日常饮食建议
- 介绍穴位按摩、八段锦等养生方法
- 解释体质特征和调理方向

## 医疗意图识别
如果用户问题涉及疾病治疗、用药、诊断（如"我XX病吃什么药""怎么治疗XX""XX病能治好吗"），使用以下模板回复：
"关于疾病治疗和用药问题，我无法提供医疗建议。建议您咨询正规医院的中医师。我可以为您介绍相关的养生知识——比如[关联的养生话题]。您想了解哪方面呢？"

## 输出格式
每次回答末尾必须包含：
---
⚠️ 以上为养生参考建议，不构成医疗诊断。如有身体不适，请及时前往正规医院就诊。"""


def wrap_disclaimer(answer: str) -> str:
    """确保回答末尾有免责声明"""
    disclaimer = "\n\n---\n⚠️ 以上为养生参考建议，不构成医疗诊断。如有身体不适，请及时前往正规医院就诊。"
    if "不构成医疗诊断" in answer[-100:]:
        return answer
    return answer + disclaimer

# -*- coding: utf-8 -*-
"""多模态视觉分析：舌诊 + 体检报告解析。

基于 DashScope 兼容模式（OpenAI 接口）调用 qwen3.8-omni-flash 全模态模型：
- 舌诊（tongue）：用户上传舌头照片 → 观察舌色/舌苔/舌形 → 推测体质倾向 + 调理建议
- 体检报告（report）：用户上传报告照片 → 解析关键指标 → 通俗解读 + 异常提示

与 RAG 问答的区别：问答是「文本 → 检索知识库 → 生成」，这里是「图片 + 文本 → 视觉模型直接生成」，
两者互补，让系统从纯文本升级为可理解图片的 Agent。
"""
from openai import OpenAI

from .config import DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL, DASHSCOPE_VL_MODEL

# 不同任务的 System/Prompt 指令
_TASK_PROMPTS = {
    "tongue": (
        "你是资深中医师。请根据用户上传的舌象照片做舌诊分析，分点输出：\n"
        "1) 舌象观察（舌色、舌苔、舌形/舌态）；\n"
        "2) 据此推测的体质倾向及可能的不适；\n"
        "3) 对应的饮食、作息、情志调理建议。\n"
        "语气专业但通俗，避免堆砌术语。"
    ),
    "report": (
        "你是健康管理助手。请解析用户上传的体检报告图片，分点输出：\n"
        "1) 能辨认出的关键指标及数值；\n"
        "2) 逐项用通俗语言说明是否处于参考范围、可能代表什么；\n"
        "3) 对异常/临界项给出生活建议。\n"
        "看不清的数值要如实说明『图片中无法辨认』，严禁编造数据。"
    ),
}

# 统一的图像分析免责声明（与文本问答的免责声明分开，语义更贴切）
VISION_DISCLAIMER = "⚠️ 以上为 AI 图像分析结果，仅供参考，不构成医疗诊断。舌诊/报告解读不能替代医生面诊，如有不适请及时就医。"


def analyze_image(image_data_url: str, task: str = "tongue", user_message: str = "") -> str:
    """调用多模态模型分析图片。

    Args:
        image_data_url: 图片的 data URL，形如 ``data:image/png;base64,xxxx``
        task: 任务类型，``tongue``（舌诊）| ``report``（体检报告）
        user_message: 用户补充说明（可选），拼接进提示词

    Returns:
        模型输出的分析文本
    """
    if not DASHSCOPE_API_KEY:
        raise RuntimeError("未配置 DASHSCOPE_API_KEY，请在项目根 .env 中设置")

    prompt = _TASK_PROMPTS.get(task, _TASK_PROMPTS["tongue"])
    if user_message:
        prompt = f"{prompt}\n\n用户补充说明：{user_message}"

    client = OpenAI(api_key=DASHSCOPE_API_KEY, base_url=DASHSCOPE_BASE_URL)
    resp = client.chat.completions.create(
        model=DASHSCOPE_VL_MODEL,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ],
        }],
        temperature=0.3,
    )
    return resp.choices[0].message.content

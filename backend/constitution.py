"""体质辨识：9题评分 + 规则引擎 + LLM校验"""
from typing import Dict, List, Tuple

# 九种体质的中文名称
CONSTITUTION_TYPES = [
    "平和质", "气虚质", "阳虚质", "阴虚质",
    "痰湿质", "湿热质", "血瘀质", "气郁质", "特禀质"
]

# 9道体质辨识题目（基于中华中医药学会《中医体质分类与判定》标准简化版）
# 每题按1-5分作答，分数映射到各体质维度
CONSTITUTION_QUESTIONS = [
    {
        "id": 1,
        "text": "您精力充沛吗？（是否容易疲劳）",
        "reverse": True,  # 反向计分
        "mapping": {"气虚质": 1.0}  # 此题主要贡献到气虚质
    },
    {
        "id": 2,
        "text": "您容易疲乏吗？",
        "mapping": {"气虚质": 1.0, "阳虚质": 0.5}
    },
    {
        "id": 3,
        "text": "您容易气短（呼吸短促、上气不接下气）吗？",
        "mapping": {"气虚质": 1.0}
    },
    {
        "id": 4,
        "text": "您手脚发凉吗？",
        "mapping": {"阳虚质": 1.0}
    },
    {
        "id": 5,
        "text": "您怕冷、衣服比别人穿得多吗？",
        "mapping": {"阳虚质": 1.0}
    },
    {
        "id": 6,
        "text": "您感到手脚心发热或身体、脸上发热吗？",
        "mapping": {"阴虚质": 1.0, "湿热质": 0.5}
    },
    {
        "id": 7,
        "text": "您面部或鼻部有油腻感或者油亮发光吗？",
        "mapping": {"湿热质": 1.0, "痰湿质": 0.5}
    },
    {
        "id": 8,
        "text": "您容易生痤疮或疮疖吗？",
        "mapping": {"湿热质": 0.8, "血瘀质": 0.5}
    },
    {
        "id": 9,
        "text": "您感到闷闷不乐、情绪低沉吗？",
        "mapping": {"气郁质": 1.0, "气虚质": 0.3}
    },
]


def calculate_constitution(answers: List[Dict]) -> Dict:
    """根据9题答案计算体质得分

    Args:
        answers: [{"question_id": 1, "score": 3}, ...] 每题1-5分

    Returns:
        {"constitution_type": "气虚质", "scores": {...}, "description": "..."}
    """
    # 初始化各体质得分
    scores = {t: 0.0 for t in CONSTITUTION_TYPES}

    # 累积各题分数到对应体质
    for answer in answers:
        qid = answer["question_id"]
        raw_score = answer["score"]

        # 找到对应题目
        q = next((q for q in CONSTITUTION_QUESTIONS if q["id"] == qid), None)
        if q is None:
            continue

        score = raw_score
        if q.get("reverse"):
            score = 6 - raw_score  # 反向计分

        # 映射到体质
        for ctype, weight in q["mapping"].items():
            scores[ctype] += score * weight

    # 平和质综合计算（基于其他体质得分的反向指标）
    deviation = sum(max(0, s - 3) for t, s in scores.items() if t != "平和质")
    scores["平和质"] = max(1, 15 - deviation * 0.5)

    # 进行归一化处理（每项满分约15分）
    max_possible = 15.0
    normalized = {t: min(round(s / max_possible * 100, 1), 100) for t, s in scores.items()}

    # 判断主体质（得分最高的非平和质，或平和质得分最高）
    sorted_types = sorted(normalized.items(), key=lambda x: x[1], reverse=True)
    primary_type = sorted_types[0][0]

    # 区分平和质与偏颇质
    if primary_type == "平和质" and normalized["平和质"] >= 60:
        primary_type = "平和质"
    elif primary_type == "平和质":
        # 平和质不是最高时，取最高偏颇质
        primary_type = sorted_types[1][0]

    return {
        "constitution_type": primary_type,
        "scores": normalized,
        "description": get_constitution_description(primary_type),
        "health_tips": get_health_tips(primary_type),
    }


def get_constitution_description(ctype: str) -> str:
    """获取体质描述"""
    descriptions = {
        "平和质": "阴阳气血调和，体态适中，面色润泽，精力充沛。是健康的理想状态，继续保持良好的生活习惯即可。",
        "气虚质": "元气不足，以疲乏、气短、自汗等气虚表现为主要特征。常见于过度劳累、久病不愈的人群。调养重在补气健脾。",
        "阳虚质": "阳气不足，以畏寒怕冷、手足不温等虚寒表现为主要特征。常见于久居寒冷环境、喜食生冷的人群。调养重在温阳补肾。",
        "阴虚质": "阴液亏少，以口燥咽干、手足心热等虚热表现为主要特征。常见于熬夜、过度用脑的人群。调养重在滋阴润燥。",
        "痰湿质": "痰湿凝聚，以形体肥胖、腹部肥满、口黏苔腻等痰湿表现为主要特征。常见于饮食不节、缺乏运动的人群。调养重在健脾利湿。",
        "湿热质": "湿热内蕴，以面垢油光、口苦口干、大便黏滞等湿热表现为主要特征。常见于嗜食辛辣油腻、长期饮酒的人群。调养重在清热祛湿。",
        "血瘀质": "血行不畅，以肤色晦暗、舌质紫暗等血瘀表现为主要特征。常见于久坐少动、情志不畅的人群。调养重在活血化瘀。",
        "气郁质": "气机郁滞，以神情抑郁、忧虑脆弱等气郁表现为主要特征。常见于压力大、情绪波动多的人群。调养重在疏肝理气。",
        "特禀质": "先天失常，以过敏反应、先天性疾病等为主要特征。常见于有家族过敏史的人群。调养重在益气固表，避免接触过敏原。",
    }
    return descriptions.get(ctype, "暂无描述")


def get_health_tips(ctype: str) -> str:
    """获取体质对应的养生方向提示"""
    tips = {
        "平和质": "饮食有节，起居有常，劳逸结合。坚持适度运动即可，无需特别调理。建议每年体检一次，防患于未然。",
        "气虚质": "饮食宜选益气健脾的食物，如山药、红枣、小米、鸡肉等。避免过度劳累，保证充足睡眠。适合太极拳、八段锦等温和运动。常用穴位：足三里、气海。",
        "阳虚质": "饮食宜选温阳益气的食物，如羊肉、韭菜、生姜、核桃等。注意保暖，尤其腰腹部。适合在阳光充足时进行户外活动。常用穴位：关元、命门。",
        "阴虚质": "饮食宜选滋阴清热的食物，如百合、银耳、梨、鸭肉等。避免熬夜，保持充足睡眠。适合游泳、瑜伽等偏静的运动。常用穴位：太溪、三阴交。",
        "痰湿质": "饮食宜选健脾利湿的食物，如薏米、冬瓜、赤小豆、山药等。控制甜食和油腻，坚持有氧运动如跑步、跳绳。常用穴位：丰隆、阴陵泉。",
        "湿热质": "饮食宜选清热祛湿的食物，如绿豆、苦瓜、薏米、莲藕等。避免辛辣油腻和饮酒。适合大强度运动帮助排汗。常用穴位：曲池、阴陵泉。",
        "血瘀质": "饮食宜选活血化瘀的食物，如山楂、醋、黑豆、茄子等。避免久坐，保持适度运动促进血液循环。常用穴位：血海、合谷。",
        "气郁质": "饮食宜选行气解郁的食物，如玫瑰花茶、柑橘、香菜等。多参加集体活动，培养兴趣爱好，保持心情舒畅。常用穴位：太冲、膻中。",
        "特禀质": "饮食宜清淡均衡，注意避开已知过敏原。增强体质，适当锻炼提高免疫力。常用穴位：迎香、足三里（增强免疫力）。",
    }
    return tips.get(ctype, "请保持健康的生活方式，定期体检。")


def get_questions() -> List[Dict]:
    """获取体质辨识题目列表（只返回题目，不含mapping信息）"""
    return [
        {"id": q["id"], "text": q["text"]}
        for q in CONSTITUTION_QUESTIONS
    ]

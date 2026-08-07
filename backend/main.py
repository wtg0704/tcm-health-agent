"""FastAPI入口：路由注册、CORS、启动事件"""
import json
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from typing import List
from datetime import datetime

from .config import DATA_DIR
from .database import get_db, init_db, SessionLocal
from .models import User, HealthProfile, ConstitutionResult, ChatHistory
from .schemas import (
    ConstitutionAssessRequest, ConstitutionAssessResponse, RadarData,
    ProfileUpdateRequest, ProfileResponse,
    ChatQueryRequest, ChatQueryResponse, ChatMessage,
    ChatHistoryResponse, HistoryItem,
    UserCreateRequest, UserResponse,
)
from .constitution import calculate_constitution, get_questions
from .rag_pipeline import init_all_vectorstores
from .agent_router import run_agent, run_agent_stream

from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    print("正在初始化数据库...")
    init_db()
    print("正在加载知识库（后台进行）...")
    try:
        init_all_vectorstores()
        print("知识库加载完成！")
    except Exception as e:
        print(f"知识库加载失败（不影响API启动）: {e}")
    print("启动完成！")
    yield  # 应用运行中
    print("应用关闭")


app = FastAPI(
    title="中医AI健康Agent系统",
    description="基于RAG+Agent的中医养生知识智能助手",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== 用户管理 ====================

@app.post("/api/users", response_model=UserResponse)
def create_user(req: UserCreateRequest, db: Session = Depends(get_db)):
    """创建新用户"""
    user = User(
        name=req.name,
        gender=req.gender,
        birth_date=datetime.strptime(req.birth_date, "%Y-%m-%d").date() if req.birth_date else None,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    return UserResponse(
        id=user.id,
        name=user.name,
        has_profile=False,
        has_constitution=False,
    )


@app.get("/api/users/{user_id}", response_model=UserResponse)
def get_user(user_id: str, db: Session = Depends(get_db)):
    """获取用户信息"""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="用户不存在")

    profile = db.query(HealthProfile).filter(HealthProfile.user_id == user_id).first()
    constitution = db.query(ConstitutionResult).filter(
        ConstitutionResult.user_id == user_id
    ).order_by(ConstitutionResult.created_at.desc()).first()

    return UserResponse(
        id=user.id,
        name=user.name,
        has_profile=profile is not None,
        has_constitution=constitution is not None,
        constitution_type=constitution.constitution_type if constitution else None,
    )


# ==================== 体质辨识 ====================

@app.get("/api/constitution/questions")
def get_constitution_questions():
    """获取体质辨识题目"""
    return {"questions": get_questions()}


@app.post("/api/constitution/assess")
def assess_constitution(req: ConstitutionAssessRequest, user_id: str = None, db: Session = Depends(get_db)):
    """提交体质辨识答案，返回体质结果"""
    answers = [{"question_id": a.question_id, "score": a.score} for a in req.answers]

    if len(answers) != 9:
        raise HTTPException(status_code=400, detail="必须回答全部9道题")

    # 规则引擎计算
    result = calculate_constitution(answers)

    # 保存到数据库（需要user_id）
    if user_id:
        user = db.query(User).filter(User.id == user_id).first()
        if user:
            record = ConstitutionResult(
                user_id=user_id,
                constitution_type=result["constitution_type"],
                scores=result["scores"],
                answers=answers,
            )
            db.add(record)
            db.commit()

    return ConstitutionAssessResponse(
        constitution_type=result["constitution_type"],
        type_description=result["description"],
        sub_scores=result["scores"],
        health_tips=result["health_tips"],
        radar_data=RadarData(
            labels=list(result["scores"].keys()),
            values=list(result["scores"].values()),
        ),
    )


# ==================== 健康画像 ====================

@app.post("/api/profile/update", response_model=ProfileResponse)
def update_profile(req: ProfileUpdateRequest, user_id: str = None, db: Session = Depends(get_db)):
    """更新用户健康画像"""
    if not user_id:
        raise HTTPException(status_code=400, detail="需要user_id")

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="用户不存在")

    # 查找或创建画像
    profile = db.query(HealthProfile).filter(HealthProfile.user_id == user_id).first()
    if not profile:
        profile = HealthProfile(user_id=user_id)
        db.add(profile)

    # 更新字段
    if req.height_cm is not None:
        profile.height_cm = req.height_cm
    if req.weight_kg is not None:
        profile.weight_kg = req.weight_kg
    if req.sleep_quality is not None:
        profile.sleep_quality = req.sleep_quality
    if req.exercise_frequency is not None:
        profile.exercise_frequency = req.exercise_frequency

    # 计算BMI
    if profile.height_cm and profile.weight_kg:
        profile.bmi = profile.weight_kg / ((profile.height_cm / 100) ** 2)

    db.commit()
    db.refresh(profile)

    # 获取体质信息
    constitution = db.query(ConstitutionResult).filter(
        ConstitutionResult.user_id == user_id
    ).order_by(ConstitutionResult.created_at.desc()).first()

    # BMI等级
    bmi = profile.bmi
    bmi_level = "未录入"
    if bmi:
        if bmi < 18.5:
            bmi_level = "偏瘦"
        elif bmi < 24:
            bmi_level = "正常"
        elif bmi < 28:
            bmi_level = "偏胖"
        else:
            bmi_level = "肥胖"

    # 生成今日建议
    constitution_type = constitution.constitution_type if constitution else None
    tips_map = {
        "气虚质": "今日建议：早餐来一碗红枣小米粥，午后散步15分钟补充阳气。避免过度劳累。",
        "阳虚质": "今日建议：晨起喝杯生姜红糖水，注意腰腹保暖。适当晒晒太阳补充阳气。",
        "阴虚质": "今日建议：泡一杯枸杞菊花茶，晚上11点前入睡。避免辛辣燥热食物。",
        "痰湿质": "今日建议：早餐用薏米煮粥，饭后散步半小时帮助排湿。少喝冰饮。",
        "湿热质": "今日建议：喝杯绿豆汤清热，晚餐清淡少油。适合傍晚跑步出汗。",
        "血瘀质": "今日建议：用山楂片泡水，工作时每小时站起来活动5分钟。保持心情舒畅。",
        "气郁质": "今日建议：泡杯玫瑰花茶，听听舒缓音乐。约朋友聊聊天散散心。",
        "特禀质": "今日建议：注意佩戴口罩防过敏原，饮食清淡均衡。适当锻炼增强免疫力。",
        "平和质": "今日建议：保持规律作息，营养均衡。今天也很健康！",
    }
    daily_tip = tips_map.get(constitution_type, "今日建议：保持良好的生活习惯，健康快乐每一天！")

    # 雷达图数据
    radar_data = None
    if constitution:
        radar_data = RadarData(
            labels=list(constitution.scores.keys()),
            values=list(constitution.scores.values()),
        )

    return ProfileResponse(
        user_id=user_id,
        height_cm=profile.height_cm,
        weight_kg=profile.weight_kg,
        bmi=round(bmi, 1) if bmi else None,
        bmi_level=bmi_level,
        sleep_quality=profile.sleep_quality,
        exercise_frequency=profile.exercise_frequency,
        constitution_type=constitution_type,
        daily_tip=daily_tip,
        radar_data=radar_data,
    )


# ==================== 知识问答（核心） ====================

@app.post("/api/chat/query")
def chat_query(req: ChatQueryRequest):
    """知识问答接口：Agent路由+多轮对话"""
    try:
        # 检查用户是否存在
        db = SessionLocal()
        user = db.query(User).filter(User.id == req.user_id).first()
        db.close()
        if not user:
            raise HTTPException(status_code=404, detail="用户不存在，请先创建用户")

        # 构建对话历史
        history = []
        if req.conversation_history:
            history = [
                {"role": h.role, "message": h.message}
                for h in req.conversation_history
            ]

        # 执行Agent
        result = run_agent(req.user_id, req.message, history)

        return ChatQueryResponse(
            answer=result["answer"],
            sources=result["sources"],
            agent_trace=result["agent_trace"],
            disclaimer=result["disclaimer"],
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理请求失败: {str(e)}")


# ==================== 流式知识问答（SSE） ====================

@app.post("/api/chat/stream")
def chat_stream(req: ChatQueryRequest):
    """流式知识问答接口——逐token推送SSE事件，实现打字机效果"""
    # 检查用户是否存在
    db = SessionLocal()
    user = db.query(User).filter(User.id == req.user_id).first()
    db.close()
    if not user:
        raise HTTPException(status_code=404, detail="用户不存在，请先创建用户")

    # 构建对话历史
    history = []
    if req.conversation_history:
        history = [
            {"role": h.role, "message": h.message}
            for h in req.conversation_history
        ]

    def generate():
        """SSE事件生成器"""
        for event in run_agent_stream(req.user_id, req.message, history):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # 禁用nginx缓冲
        },
    )


# ==================== 对话历史 ====================

@app.get("/api/chat/history")
def get_chat_history(user_id: str, limit: int = 20, db: Session = Depends(get_db)):
    """获取用户对话历史"""
    records = db.query(ChatHistory).filter(
        ChatHistory.user_id == user_id
    ).order_by(ChatHistory.created_at.desc()).limit(limit).all()

    messages = []
    for r in reversed(records):
        messages.append(HistoryItem(
            role=r.role,
            message=r.message,
            sources=r.sources,
            agent_trace=r.agent_trace,
            created_at=r.created_at,
        ))

    return ChatHistoryResponse(messages=messages)


# ==================== 健康检查 ====================

@app.get("/api/health")
def health_check():
    return {"status": "ok", "service": "中医AI健康Agent系统", "version": "1.0.0"}

"""Pydantic数据模型定义"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from datetime import datetime


# ========== 体质辨识 ==========

class AnswerItem(BaseModel):
    question_id: int = Field(..., ge=1, le=9, description="题目编号 1-9")
    score: int = Field(..., ge=1, le=5, description="分数 1-5")


class ConstitutionAssessRequest(BaseModel):
    answers: List[AnswerItem] = Field(..., min_length=9, max_length=9, description="9题答案")


class RadarData(BaseModel):
    labels: List[str]
    values: List[float]


class ConstitutionAssessResponse(BaseModel):
    constitution_type: str
    type_description: str
    sub_scores: Dict[str, float]
    health_tips: str
    radar_data: RadarData


# ========== 健康画像 ==========

class ProfileUpdateRequest(BaseModel):
    height_cm: Optional[float] = Field(None, ge=50, le=300, description="身高(cm)")
    weight_kg: Optional[float] = Field(None, ge=10, le=500, description="体重(kg)")
    sleep_quality: Optional[int] = Field(None, ge=1, le=5, description="睡眠质量 1-5")
    exercise_frequency: Optional[int] = Field(None, ge=0, description="每周运动次数")


class ProfileResponse(BaseModel):
    user_id: str
    height_cm: Optional[float] = None
    weight_kg: Optional[float] = None
    bmi: Optional[float] = None
    bmi_level: Optional[str] = None
    sleep_quality: Optional[int] = None
    exercise_frequency: Optional[int] = None
    constitution_type: Optional[str] = None
    daily_tip: str
    radar_data: Optional[RadarData] = None


# ========== 知识问答 ==========

class ChatMessage(BaseModel):
    role: str  # user | assistant
    message: str


class ChatQueryRequest(BaseModel):
    user_id: str
    message: str = Field(..., min_length=1, description="用户问题")
    conversation_history: Optional[List[ChatMessage]] = Field(default=[], description="对话历史")


class SourceItem(BaseModel):
    doc: str
    excerpt: str


class AgentTrace(BaseModel):
    intent: str
    searched_collections: List[str]
    tools_called: List[str]


class ChatQueryResponse(BaseModel):
    answer: str
    sources: List[SourceItem]
    agent_trace: AgentTrace
    disclaimer: str


# ========== 对话历史 ==========

class HistoryItem(BaseModel):
    role: str
    message: str
    sources: Optional[List[SourceItem]] = None
    agent_trace: Optional[AgentTrace] = None
    created_at: datetime


class ChatHistoryResponse(BaseModel):
    messages: List[HistoryItem]


# ========== 通用 ==========

class UserCreateRequest(BaseModel):
    name: Optional[str] = None
    gender: Optional[str] = None
    birth_date: Optional[str] = None


class UserResponse(BaseModel):
    id: str
    name: Optional[str] = None
    has_profile: bool = False
    has_constitution: bool = False
    constitution_type: Optional[str] = None

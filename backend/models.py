"""SQLAlchemy ORM模型（兼容SQLite和PostgreSQL）"""
import uuid
from sqlalchemy import Column, String, Integer, Float, Date, Text, DateTime, ForeignKey, CheckConstraint, Index
from sqlalchemy.types import JSON
from sqlalchemy.orm import relationship
from datetime import datetime
from .database import Base, engine


def generate_uuid():
    return str(uuid.uuid4())


# 根据数据库类型选择字段类型
_is_sqlite = "sqlite" in str(engine.url)

if _is_sqlite:
    # SQLite使用String存储UUID和JSON
    UUIDType = String(36)
    JSONType = JSON
else:
    from sqlalchemy.dialects.postgresql import UUID as PG_UUID, JSONB
    UUIDType = PG_UUID(as_uuid=False)
    JSONType = JSONB


class User(Base):
    __tablename__ = "users"

    id = Column(UUIDType, primary_key=True, default=generate_uuid)
    name = Column(String(50), nullable=True)
    gender = Column(String(10), nullable=True)
    birth_date = Column(Date, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    health_profile = relationship("HealthProfile", back_populates="user", uselist=False, cascade="all, delete-orphan")
    constitution_results = relationship("ConstitutionResult", back_populates="user", cascade="all, delete-orphan")
    chat_history = relationship("ChatHistory", back_populates="user", cascade="all, delete-orphan")


class ConstitutionResult(Base):
    __tablename__ = "constitution_results"

    id = Column(UUIDType, primary_key=True, default=generate_uuid)
    user_id = Column(UUIDType, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    constitution_type = Column(String(20), nullable=False)
    scores = Column(JSONType, nullable=False)
    answers = Column(JSONType, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="constitution_results")

    __table_args__ = (
        Index("idx_constitution_user", "user_id", "created_at"),
    )


class HealthProfile(Base):
    __tablename__ = "health_profiles"

    id = Column(UUIDType, primary_key=True, default=generate_uuid)
    user_id = Column(UUIDType, ForeignKey("users.id", ondelete="CASCADE"), unique=True, nullable=False)
    height_cm = Column(Float, nullable=True)
    weight_kg = Column(Float, nullable=True)
    bmi = Column(Float, nullable=True)
    sleep_quality = Column(Integer, nullable=True)
    exercise_frequency = Column(Integer, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="health_profile")


class ChatHistory(Base):
    __tablename__ = "chat_history"

    id = Column(UUIDType, primary_key=True, default=generate_uuid)
    user_id = Column(UUIDType, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    role = Column(String(10), nullable=False)
    message = Column(Text, nullable=False)
    sources = Column(JSONType, nullable=True)
    agent_trace = Column(JSONType, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="chat_history")

    __table_args__ = (
        Index("idx_chat_user", "user_id", "created_at"),
    )

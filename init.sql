-- 初始化数据库表结构
-- 由 docker-compose 自动执行

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- 用户表
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(50),
    gender VARCHAR(10),
    birth_date DATE,
    created_at TIMESTAMP DEFAULT NOW()
);

-- 体质辨识结果
CREATE TABLE IF NOT EXISTS constitution_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    constitution_type VARCHAR(20) NOT NULL,
    scores JSONB NOT NULL,
    answers JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- 健康画像
CREATE TABLE IF NOT EXISTS health_profiles (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID UNIQUE REFERENCES users(id) ON DELETE CASCADE,
    height_cm DECIMAL(5,1),
    weight_kg DECIMAL(5,1),
    bmi DECIMAL(4,1) GENERATED ALWAYS AS (
        weight_kg / ((height_cm/100) * (height_cm/100))
    ) STORED,
    sleep_quality INTEGER CHECK (sleep_quality BETWEEN 1 AND 5),
    exercise_frequency INTEGER,
    updated_at TIMESTAMP DEFAULT NOW()
);

-- 对话历史
CREATE TABLE IF NOT EXISTS chat_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    role VARCHAR(10) NOT NULL CHECK (role IN ('user', 'assistant')),
    message TEXT NOT NULL,
    sources JSONB,
    agent_trace JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- 索引
CREATE INDEX IF NOT EXISTS idx_chat_user ON chat_history(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_constitution_user ON constitution_results(user_id, created_at DESC);

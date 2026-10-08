"""数据库连接与会话管理"""
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker, declarative_base
from .config import DATABASE_URL

# SQLite不支持pool_size/pool_pre_ping
connect_args = {}
engine_kwargs = {}
if "sqlite" in DATABASE_URL:
    connect_args["check_same_thread"] = False
else:
    engine_kwargs["pool_size"] = 5
    engine_kwargs["pool_pre_ping"] = True

engine = create_engine(DATABASE_URL, connect_args=connect_args, **engine_kwargs)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """FastAPI依赖注入：获取数据库会话"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """初始化数据库表（开发用）"""
    Base.metadata.create_all(bind=engine)
    _migrate_users_password()


def _migrate_users_password():
    """轻量迁移：给已存在的 users 表补 password 列（兼容旧库，避免删库）"""
    insp = inspect(engine)
    if "users" not in insp.get_table_names():
        return
    cols = [c["name"] for c in insp.get_columns("users")]
    if "password" not in cols:
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE users ADD COLUMN password VARCHAR(128)"))

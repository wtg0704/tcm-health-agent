# -*- coding: utf-8 -*-
"""可观测性：结构化日志 + 请求级追踪。

埋点覆盖（统一走标准库 logging，方便后续接入 ELK / Loki 等日志平台）：
1. HTTP 请求 middleware（main.py）：method / path / status / 耗时；
2. Agent 调用链路（agent_router.py）：user_id / tools_called / searched_collections / 耗时；
3. 长期记忆（memory.py，可选）：召回条数 / 写入条数。
"""
import logging
import time

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_configured = False


def setup_logging(level: int = logging.INFO) -> None:
    """初始化根日志（幂等，避免重复添加 handler）"""
    global _configured
    if _configured:
        return
    logging.basicConfig(level=level, format=_LOG_FORMAT)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """获取带统一格式的 logger"""
    setup_logging()
    return logging.getLogger(name)


def elapsed_ms(start: float) -> float:
    """计算从 start（time.perf_counter() 结果）到现在的耗时（毫秒）"""
    return (time.perf_counter() - start) * 1000

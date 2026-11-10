"""I1 最小循环：消息模型、工具注册、循环状态机与退出原因。

对应文章：https://github.com/QingyaFan/blog/issues/260
"""

from enum import StrEnum


class StopReason(StrEnum):
    """Why a run finished."""

    COMPLETED = "completed"
    MAX_TURNS = "max_turns"
    NO_PROGRESS = "no_progress"
    FAILED = "failed"
    CANCELLED = "cancelled"
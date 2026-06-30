"""
领域实体 — Session（会话）及其值对象
========================================

对应 session_state 表的一行记录，以及 chat_history 表中的单轮记录。

实体：
  - Session — 会话聚合根，包含立场、情绪、轮次、悬案线程等状态

值对象：
  - TurnRecord — 单轮对话记录（不可变）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from domain.entities.memory import SessionId, UserId


# ── 实体 ──


@dataclass
class Session:
    """会话实体。

    对应 session_state 表的一行记录。
    追踪单次对话的生命周期状态：
      - 全局状态：stance（会话立场）、mood（会话氛围分数）
      - 交互统计：turn_count（轮次）、stance_turns（当前立场持续轮数）
      - 参与度：engagement_level / momentum_depth / momentum_energy
      - 时序：last_active（最后活跃时间）
      - 未完成话题：dangling_threads（JSON，格式 {"threads": [...], "turn": N}）
      - 蒸馏进度：last_distill_turn（已蒸馏轮次）、last_distill_at（最后蒸馏时间）

    注意：v2.0 重构后 stance / momentum / engagement 的自动分析逻辑已移除，
          这些字段保留在实体中以兼容现有数据库 schema。
    """

    session_id: SessionId
    user_id: UserId | None = None

    # ── 会话立场与情绪 ──
    stance: str = "open"
    mood: float = 0.0
    stance_turns: int = 0

    # ── 交互统计 ──
    turn_count: int = 0

    # ── 参与度（v2.0 后无自动分析，保留 schema 兼容）──
    engagement_level: str = "coasting"
    momentum_depth: float = 0.0
    momentum_energy: float = 0.0

    # ── 时序 ──
    last_active: datetime | None = None

    # ── 未完成话题与蒸馏 ──
    dangling_threads: dict[str, Any] = field(default_factory=dict)
    embedding_updated: int = 0
    last_distill_turn: int = 0
    last_distill_at: datetime | None = None


# ── 值对象 ──


@dataclass(frozen=True)
class TurnRecord:
    """单轮对话记录值对象。

    对应 chat_history 表的 user+assistant 两条记录中的用户侧或助手侧。
    注意：一次对话轮次会产生两条 TurnRecord（role='user' 和 role='assistant'）。

    turn_num 在用户侧和助手侧共享同一个值，用于关联同一轮中的输入与回复。
    """

    session_id: SessionId
    role: str  # 'user' | 'assistant'
    content: str
    turn_num: int

    user_id: str = ""
    sender_name: str = ""
    importance: float = 0.3
    mood: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        """Validate field values."""
        if self.role not in ("user", "assistant"):
            raise ValueError(f"role must be 'user' or 'assistant', got {self.role!r}")
        if not self.content:
            raise ValueError("content cannot be empty")
        if not isinstance(self.importance, (int, float)) or not 0.0 <= self.importance <= 1.0:
            raise ValueError(f"importance must be in [0.0, 1.0], got {self.importance}")

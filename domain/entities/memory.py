"""
领域实体 — Memory（记忆）及其值对象
========================================

对应 cognitive_distill 表的核心记忆记录，以及支撑的值对象。

值对象：
  - MemoryId / UserId / SessionId — 标识类型
  - Importance / Mood — 带验证的语义类型

实体：
  - Memory — 认知蒸馏记忆聚合根，支持时间衰减和过期判断。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import math


# ── 标识值对象 ──


@dataclass(frozen=True)
class MemoryId:
    """记忆记录 ID 值对象。"""

    value: int

    def __post_init__(self) -> None:
        if not isinstance(self.value, int):
            raise TypeError(f"MemoryId 必须是 int，收到 {type(self.value)}")


@dataclass(frozen=True)
class UserId:
    """用户 ID 值对象。"""

    value: str

    def __post_init__(self) -> None:
        if not isinstance(self.value, str) or not self.value.strip():
            raise ValueError("UserId 必须是非空字符串")


@dataclass(frozen=True)
class SessionId:
    """会话 ID 值对象。"""

    value: str

    def __post_init__(self) -> None:
        if not isinstance(self.value, str) or not self.value.strip():
            raise ValueError("SessionId 必须是非空字符串")


# ── 语义值对象（带验证）──


@dataclass(frozen=True)
class Importance:
    """重要性值对象，合法范围 [0.0, 1.0]。"""

    value: float

    def __post_init__(self) -> None:
        if not isinstance(self.value, (int, float)):
            raise TypeError(f"Importance 必须是数值，收到 {type(self.value)}")
        if self.value < 0.0 or self.value > 1.0:
            raise ValueError(f"Importance 必须在 [0.0, 1.0] 范围内，收到 {self.value}")


@dataclass(frozen=True)
class Mood:
    """情绪标签值对象，非空字符串（允许空串表示无情绪）。"""

    value: str

    def __post_init__(self) -> None:
        if not isinstance(self.value, str):
            raise TypeError(f"Mood 必须是 str，收到 {type(self.value)}")


# ── 实体 ──


@dataclass
class Memory:
    """认知蒸馏记忆实体。

    对应 cognitive_distill 表的一行记录。
    存储经 LLM 蒸馏后的关键事实、图标签摘要、情绪标注、重要性评分、
    向量嵌入元数据和过期时间。

    可通过 decayed_importance() 计算时间衰减后的重要性，
    通过 is_expired() 判断是否已到过期时间。
    """

    memory_id: MemoryId
    user_id: UserId
    content: str
    created_at: datetime
    keylabel: str = ""
    summary: str = ""

    importance: Importance = field(default_factory=lambda: Importance(0.3))
    mood: Mood = field(default_factory=lambda: Mood(""))
    mood_intensity: float = 0.0

    session_id: SessionId | None = None
    entities: str = "[]"
    turn_num: int = 0

    expires_at: datetime | None = None

    embedding: bytes | None = None
    embedding_dim: int | None = None

    def is_expired(self, reference: datetime | None = None) -> bool:
        """判断记忆是否已过期。

        在未设置 expires_at 时返回 False。
        可传入 reference 作为"当前时间"基准，默认使用 UTC now。

        Args:
            reference: 判断过期的时间基准

        Returns:
            如果 reference 晚于等于 expires_at 则返回 True
        """
        if self.expires_at is None:
            return False
        ref = reference or datetime.now(timezone.utc)
        if self.expires_at.tzinfo is None and ref.tzinfo is not None:
            return ref.replace(tzinfo=None) >= self.expires_at
        if ref.tzinfo is None and self.expires_at.tzinfo is not None:
            return ref >= self.expires_at.replace(tzinfo=None)
        return ref >= self.expires_at

    def decayed_importance(self, halflife_days: int = 30) -> float:
        """计算时间衰减后的重要性分数。

        使用指数衰减模型：imp * exp(-ln(2) * age_days / halflife_days)。
        半衰期默认 30 天，可配置。

        Args:
            halflife_days: 半衰期天数，默认 30

        Returns:
            [0.0, 1.0] 范围内的衰减后重要性
        """
        imp = self.importance.value
        now = datetime.now(timezone.utc)
        ref_now = now.replace(tzinfo=None) if now.tzinfo else now
        ref_created = self.created_at.replace(tzinfo=None) if self.created_at.tzinfo else self.created_at
        age_days = (ref_now - ref_created).total_seconds() / 86400.0
        if age_days <= 0:
            return imp
        lam = math.log(2) / max(halflife_days, 1)
        decayed = imp * math.exp(-lam * age_days)
        return max(0.0, min(1.0, decayed))

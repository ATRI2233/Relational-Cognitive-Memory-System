"""
领域实体 — Identity（身份）及其值对象
============================================

映射数据库表 `identity_memory`：
  - traits       → Trait 列表   (JSON: [{"t": str, "s": int, "c": int}])
  - preferences  → Preferences  (JSON: {"likes": [...], "dislikes": [...]})
  - self_identity → List[str]   (JSON: ["..."])
  - boundaries   → Boundary 列表 (JSON: [{"description": "..."}])
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import List


# ── 值对象（Value Objects）：不可变、自验证 ──


@dataclass(frozen=True)
class Trait:
    """人格特质值对象。

    对应 LLM 分析产出的 traits_updates，每条特质含文本、强度、确认次数。
    强度 strength 与 `_apply_analysis` 中的衰减/确认逻辑对应。
    """

    text: str
    strength: int = 0
    count: int = 0

    def __post_init__(self) -> None:
        if not self.text or not self.text.strip():
            raise ValueError("特质文本不能为空")
        if self.strength < 0:
            raise ValueError(f"特质强度不能为负: {self.strength}")
        if self.count < 0:
            raise ValueError(f"特质出现次数不能为负: {self.count}")


@dataclass
class Preferences:
    """偏好值对象。

    对应 `identity_memory.preferences` 字段中的 JSON 结构。
    """

    likes: List[str] = field(default_factory=list)
    dislikes: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        # 过滤无效空串
        cleaned_likes = [s for s in self.likes if s and s.strip()]
        cleaned_dislikes = [s for s in self.dislikes if s and s.strip()]
        if len(cleaned_likes) != len(self.likes):
            logging.getLogger("rcms").warning("Preferences.likes contained empty strings that were cleaned")
            self.likes = cleaned_likes
        if len(cleaned_dislikes) != len(self.dislikes):
            logging.getLogger("rcms").warning("Preferences.dislikes contained empty strings that were cleaned")
            self.dislikes = cleaned_dislikes


@dataclass(frozen=True)
class Boundary:
    """边界/雷区值对象。

    对应 `identity_memory.boundaries` 中的条目，表示用户不希望触及的话题。
    """

    description: str

    def __post_init__(self) -> None:
        if not self.description or not self.description.strip():
            raise ValueError("边界描述不能为空")


# ── 实体（Entity）：可变的聚合根 ──


@dataclass
class Identity:
    """用户身份实体。

    作为用户画像的聚合根，包含特质、偏好、自我认知和边界。
    对应 `identity_memory` 表的一行记录。
    """

    user_id: str
    created_at: datetime
    updated_at: datetime
    traits: List[Trait] = field(default_factory=list)
    preferences: Preferences = field(default_factory=Preferences)
    self_identity: List[str] = field(default_factory=list)
    boundaries: List[Boundary] = field(default_factory=list)

"""
Identity 仓储接口
===================

定义 IIdentityRepository Protocol，作为 Identity 聚合根的持久化协约。
"""

from __future__ import annotations

from typing import List, Optional, Protocol, runtime_checkable

from domain.entities.identity import Boundary, Identity, Preferences, Trait


@runtime_checkable
class IIdentityRepository(Protocol):
    """用户身份仓储接口。

    负责 Identity 聚合根的持久化操作。
    所有方法均为异步，返回领域实体类型。
    """

    async def get(self, user_id: str) -> Optional[Identity]:
        """获取用户的完整身份信息。

        Args:
            user_id: 用户标识

        Returns:
            如果用户存在则返回 Identity 实体，否则返回 None
        """
        ...

    async def save_traits(self, user_id: str, traits: List[Trait]) -> None:
        """保存用户特质列表（覆盖写入）。

        对应 `_apply_analysis` 中的 traits_updates 处理逻辑：
          新特质以 strength=5 写入，已有特质 strength 置 5 且 count+1，
          未被确认的特质 strength 衰减。

        Args:
            user_id: 用户标识
            traits: 特质值对象列表
        """
        ...

    async def save_preferences(self, user_id: str, prefs: Preferences) -> None:
        """保存用户偏好信息（覆盖写入）。

        对应 `identity_memory.preferences` 字段的写入操作。

        Args:
            user_id: 用户标识
            prefs: 偏好值对象
        """
        ...

    async def save_self_identity(self, user_id: str, identities: List[str]) -> None:
        """保存用户自我认知（覆盖写入）。

        对应 `identity_memory.self_identity` 字段的写入操作。

        Args:
            user_id: 用户标识
            identities: 自我认知描述字符串列表
        """
        ...

    async def save_boundaries(self, user_id: str, boundaries: List[Boundary]) -> None:
        """保存用户边界/雷区（覆盖写入）。

        对应 _apply_analysis 第 6 步的 boundaries 写入操作。

        Args:
            user_id: 用户标识
            boundaries: 边界值对象列表
        """
        ...

    async def update_identity(self, user_id: str, identity: Identity) -> None:
        """更新完整的用户身份信息。

        一次性写入 Identity 实体的所有字段。
        当需要原子更新整个聚合根时使用此方法。

        Args:
            user_id: 用户标识
            identity: 完整的 Identity 实体
        """
        ...

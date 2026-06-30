"""
PostUpdateService — 对话后处理服务。

从 core.py post_update_rules（行 192-221）+ _init_identity（行 145-148）提取。
职责：
1. 确保 identity_memory 行存在（_init_identity）
2. 更新 session_state.last_active
3. 检查 dangling_threads 过期并归档
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from domain.entities.identity import Identity
from domain.entities.memory import Importance, Memory, MemoryId, SessionId, UserId
from domain.ports.clock import IClock
from domain.ports.identity_repo import IIdentityRepository
from domain.ports.repositories import IMemoryRepository, ISessionRepository


class PostUpdateService:
    """对话后处理服务

    每轮对话后的纯管理操作，不涉及 LLM 调用。
    """

    def __init__(
        self,
        session_repo: ISessionRepository,
        identity_repo: IIdentityRepository,
        memory_repo: IMemoryRepository,
        clock: IClock,
        dangling_expire_turns: int = 15,
    ) -> None:
        self._session_repo = session_repo
        self._identity_repo = identity_repo
        self._memory_repo = memory_repo
        self._clock = clock
        self._dangling_expire_turns = dangling_expire_turns

    async def run(
        self,
        user_id: str,
        session_id: str,
        user_input: str,
        stance: str = "open",
    ) -> None:
        """执行对话后处理

        步骤（对应 core.py 行 192-221）：
        1. 确保 identity_memory 行存在（对应行 145-148 _init_identity）
        2. 更新 session_state.last_active（对应行 200）
        3. 读取 dangling_threads（对应行 201-203）
        4. 检查是否过期：current_turn - since_turn >= expire_turns（对应行 208-211）
        5. 过期的 -> 归档到 cognitive_distill -> 清空 session_state（对应行 212）

        Args:
            user_id: 用户标识
            session_id: 会话标识
            user_input: 用户输入
            stance: 会话立场
        """
        now_dt = self._clock.now()

        # 1. 初始化身份行（core.py:145-148 _init_identity）
        await self._ensure_identity(user_id, now_dt)

        # 2. 更新 last_active（core.py:200）
        await self._session_repo.update_last_active(SessionId(session_id), now_dt)

        # 3. 读取 dangling_threads（core.py:201-203）
        session = await self._session_repo.get(SessionId(session_id))
        if session is None:
            return

        dangling = session.dangling_threads
        if not dangling or not dangling.get("threads"):
            return

        # 4. 检查过期（core.py:208-211）
        since_turn = dangling.get("turn", 0)
        current_turn = session.turn_count
        if current_turn - since_turn >= self._dangling_expire_turns:
            # 5. 归档（core.py:212）
            await self._archive_dangling(
                user_id, session_id, dangling["threads"], now_dt,
            )

    async def _ensure_identity(self, user_id: str, now: datetime) -> None:
        """确保 identity_memory 行存在（core.py:145-148 _init_identity）"""
        identity = await self._identity_repo.get(user_id)
        if identity is not None:
            return
        fresh = Identity(user_id=user_id, created_at=now, updated_at=now)
        await self._identity_repo.update_identity(user_id, fresh)

    async def _archive_dangling(
        self,
        user_id: str,
        session_id: str,
        threads: list,
        now: datetime,
    ) -> None:
        """归档过期悬案 — 只清空 session_state，不再写入 cognitive_distill。

        `[悬案归档·过期]` 是系统内部运维噪音，不应写入 cognitive_distill，
        否则时间/重要性通道会误将其作为用户记忆召回。
        """
        # 不再写入 cognitive_distill，只清空 dangling_threads
        await self._session_repo.update_dangling_threads(SessionId(session_id), {"threads": [], "turn": 0})

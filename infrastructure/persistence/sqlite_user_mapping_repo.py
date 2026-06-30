"""
SQLite 用户映射仓储实现
========================

实现 IUserMappingRepository Protocol 的全部 4 个方法。
负责 user_mappings 表的读写操作，用于跨会话用户身份追踪。

匹配原 rcms_core/session.py 中 SessionMixin 的：
  - save_turn — 自动注册 sender_name（行 33-37）
  - find_mentioned_users — 三级匹配（行 57-100）
  - bind_user_label — INSERT OR REPLACE（行 102-109）

依赖：
  - sqlite3.Connection — 数据库连接
  - domain.ports.clock.IClock — 可注入的时间源
"""

from __future__ import annotations

import logging
import sqlite3
from typing import Optional

from domain.ports.clock import IClock
from domain.ports.repositories import IUserMappingRepository

logger = logging.getLogger("rcms")

_TIMESTAMP_FMT = "%Y-%m-%d %H:%M:%S"


class SQLiteUserMappingRepository(IUserMappingRepository):
    """SQLite 实现的用户映射仓储。

    负责 user_mappings 表的增/查操作：
      - upsert_mapping:     INSERT OR IGNORE 注册映射
      - find_mentioned:     三级扫描文本中被提及的用户
      - get_labels:         获取 session 内用户的标签列表
      - bind_user_label:    INSERT OR REPLACE 绑定标签

    Args:
        conn: SQLite 数据库连接
        clock: 可注入的时间源
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        clock: IClock,
    ) -> None:
        self._conn = conn
        self._clock = clock

    # ── 公有接口 ──────────────────────────────────────────────

    async def upsert_mapping(
        self,
        session_id: str,
        user_id: str,
        label: str,
        source: str = "",
    ) -> None:
        """注册或更新用户映射（INSERT OR IGNORE）。

        同 (session_id, user_id, label) 的记录仅写入一次，
        不会覆盖已有数据。

        Args:
            session_id: 会话标识
            user_id: 用户标识
            label: 显示名称/标签
            source: 来源（如 'nickname'），默认空字符串
        """
        if not label:
            return
        try:
            self._conn.execute(
                "INSERT OR IGNORE INTO user_mappings "
                "(session_id, user_id, label, source, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    session_id,
                    user_id,
                    label,
                    source or "nickname",
                    self._clock.strftime(_TIMESTAMP_FMT),
                ),
            )
            self._conn.commit()
            logger.debug(
                "用户映射已注册 session=%s user=%s label=%s",
                session_id, user_id, label,
            )
        except sqlite3.Error as e:
            self._conn.rollback()
            logger.error(
                "注册用户映射失败 session=%s user=%s label=%s: %s",
                session_id, user_id, label, e,
            )
            raise

    async def find_mentioned(
        self,
        session_id: str,
        text: str,
        speaker_id: str = "",
    ) -> list[tuple[str, str]]:
        """三级查找被提及的用户。

        匹配原 SessionMixin.find_mentioned_users（session.py:57-100）。

        匹配策略：
          1. 查当前 session 的 user_mappings，匹配 text 中包含的 label
          2. 查 speaker 参与过的其它 session 中的映射
          3. 全局 user_mappings 兜底匹配

        同 user_id 的标签只返回第一次匹配的结果。

        Args:
            session_id: 当前会话标识
            text: 消息文本，用于 label 匹配
            speaker_id: 发言者标识，提供时可查跨 session 映射

        Returns:
            [(user_id, label), ...] 被提及的用户标识和显示名列表
        """
        if not text:
            return []

        result: list[tuple[str, str]] = []
        seen: set[str] = set()

        # ── 第 1 级：当前 session 的 user_mappings ──
        rows = self._conn.execute(
            "SELECT user_id, label FROM user_mappings WHERE session_id = ?",
            (session_id,),
        ).fetchall()
        for uid, label in rows:
            if label and label.lower() in text.lower() and uid not in seen:
                seen.add(uid)
                result.append((uid, label))

        # ── 第 2 级：查发言者参与过的其他 session ──
        # 不判断 not result，因为发言者的 label 可能在 text 中（如格式前缀），
        # 导致 result 非空但真正的被提及者未被发现
        if speaker_id:
            rows = self._conn.execute(
                """
                SELECT DISTINCT um.user_id, um.label
                FROM user_mappings um
                WHERE um.session_id IN (
                    SELECT session_id FROM user_mappings WHERE user_id = ?
                )
                AND um.user_id != ?
                """,
                (speaker_id, speaker_id),
            ).fetchall()
            for uid, label in rows:
                if label and label.lower() in text.lower() and uid not in seen:
                    seen.add(uid)
                    result.append((uid, label))

        # ── 第 3 级：全局 user_mappings 标签匹配 ──
        # 被提及者可能从未和发言者同 session（如仅私聊过 bot）
        rows = self._conn.execute(
            """
            SELECT DISTINCT user_id, label FROM user_mappings
            WHERE label != '' AND label IS NOT NULL
            LIMIT 200
            """,
        ).fetchall()
        if len(rows) >= 200:
            logger.warning("user_mappings fallback scan returned >=200 rows, may be slow")
        for uid, label in rows:
            if label and label.lower() in text.lower() and uid not in seen:
                seen.add(uid)
                result.append((uid, label))

        return result

    async def get_labels(
        self,
        session_id: str,
        user_id: str,
    ) -> list[str]:
        """获取某用户在指定 session 中的所有标签。

        Args:
            session_id: 会话标识
            user_id: 用户标识

        Returns:
            标签字符串列表，无匹配时返回空列表
        """
        if not session_id or not user_id:
            return []
        try:
            rows = self._conn.execute(
                "SELECT label FROM user_mappings "
                "WHERE session_id = ? AND user_id = ?",
                (session_id, user_id),
            ).fetchall()
            return [row[0] for row in rows if row[0]]
        except sqlite3.Error as e:
            logger.error(
                "获取用户标签失败 session=%s user=%s: %s",
                session_id, user_id, e,
            )
            raise

    async def bind_user_label(
        self,
        session_id: str,
        user_id: str,
        label: str,
        source: str = "",
    ) -> None:
        """绑定用户的显示名称/标签到当前 session。

        使用 INSERT OR REPLACE，覆盖已有同（session_id, user_id, label）的记录。
        匹配原 SessionMixin.bind_user_label（session.py:102-109）。

        Args:
            session_id: 会话标识
            user_id: 用户标识
            label: 显示名称/标签
            source: 来源（如 'nickname'、'custom'），默认空字符串
        """
        if not label:
            return
        try:
            self._conn.execute(
                "INSERT OR REPLACE INTO user_mappings "
                "(session_id, user_id, label, source, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    session_id,
                    user_id,
                    label,
                    source or "nickname",
                    self._clock.strftime(_TIMESTAMP_FMT),
                ),
            )
            self._conn.commit()
            logger.debug(
                "用户标签已绑定 session=%s user=%s label=%s",
                session_id, user_id, label,
            )
        except sqlite3.Error as e:
            self._conn.rollback()
            logger.error(
                "绑定用户标签失败 session=%s user=%s label=%s: %s",
                session_id, user_id, label, e,
            )
            raise

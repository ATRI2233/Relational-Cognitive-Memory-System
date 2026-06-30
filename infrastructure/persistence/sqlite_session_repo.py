"""
SQLite 会话状态仓储实现
========================

实现 ISessionRepository Protocol 的全部 8 个方法。
仅操作 session_state 一张表，不跨 cognitive_distill。

依赖：
  - sqlite3.Connection — 数据库连接
  - domain.ports.clock.IClock — 可注入的时间源
  - domain.entities.session.Session — 会话实体
  - domain.entities.memory.SessionId — 会话标识
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime
from typing import Any, Optional

from domain.entities.memory import SessionId, UserId
from domain.entities.session import Session
from domain.ports.clock import IClock
from domain.ports.repositories import ISessionRepository

logger = logging.getLogger("rcms")

# session_state 表全部列的常量，保证所有读写操作列顺序一致
_SESSION_COLUMNS = (
    "session_id, user_id, stance, mood, turn_count, stance_turns, "
    "engagement_level, momentum_depth, momentum_energy, last_active, "
    "dangling_threads, embedding_updated, last_distill_turn, last_distill_at"
)

_TIMESTAMP_FMT = "%Y-%m-%d %H:%M:%S"


class SQLiteSessionRepository(ISessionRepository):
    """SQLite 会话状态仓储。

    实现 ISessionRepository Protocol，仅操作 session_state 表。
    所有数据库写入使用 IClock 获取时间，消除对 datetime.now() 的直接依赖。

    Args:
        conn: SQLite 数据库连接
        clock: 可注入的时间源
    """

    def __init__(self, conn: sqlite3.Connection, clock: IClock) -> None:
        self._conn = conn
        self._clock = clock

    # ── 内部辅助方法 ──────────────────────────────────────────────

    def _now_str(self) -> str:
        """返回格式化的当前时间字符串。"""
        return self._clock.strftime(_TIMESTAMP_FMT)

    @staticmethod
    def _format_dt(val: datetime | None) -> str | None:
        """将 datetime 格式化为 SQLite 时间戳字符串。

        Args:
            val: 待格式化的 datetime

        Returns:
            格式化后的时间字符串，或 None
        """
        if val is None:
            return None
        return val.strftime(_TIMESTAMP_FMT)

    @staticmethod
    def _parse_dt(val: object) -> datetime | None:
        """将数据库返回的时间戳值解析为 datetime。

        SQLite 无原生 datetime 类型，值可能为字符串或已由
        connection 工厂转换为 datetime 对象。
        对于 None 或无法解析的值返回 None。

        Args:
            val: 数据库返回的时间戳值

        Returns:
            datetime 对象，或 None
        """
        if val is None:
            return None
        if isinstance(val, datetime):
            return val
        if isinstance(val, str):
            for fmt in (_TIMESTAMP_FMT, "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
                try:
                    return datetime.strptime(val, fmt)
                except ValueError:
                    continue
        return None

    @staticmethod
    def _parse_dangling(raw: object) -> dict:
        """将 dangling_threads 字段解析为 dict。

        处理 NULL、空字符串、JSON 解析失败等情况，
        统一返回空 dict 以保证调用方安全使用 .get("threads") 等方法。

        Args:
            raw: 数据库返回的 dangling_threads 值

        Returns:
            解析后的 dict，或空 dict
        """
        if raw is None:
            return {}
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str) and raw.strip():
            try:
                parsed = json.loads(raw)
                return parsed if isinstance(parsed, dict) else {}
            except (json.JSONDecodeError, ValueError):
                logger.warning("无法解析 dangling_threads JSON: %s", raw[:80])
                return {}
        return {}

    @classmethod
    def _row_to_session(cls, row: sqlite3.Row) -> Session:
        """将 session_state 表的一行转换为 Session 实体。

        Args:
            row: sqlite3.Row 查询结果

        Returns:
            构造完成的 Session 实体
        """
        return Session(
            session_id=SessionId(row[0]),
            user_id=UserId(row[1]) if row[1] else None,
            stance=row[2] or "open",
            mood=row[3] if row[3] is not None else 0.0,
            turn_count=row[4] if row[4] is not None else 0,
            stance_turns=row[5] if row[5] is not None else 0,
            engagement_level=row[6] or "coasting",
            momentum_depth=row[7] if row[7] is not None else 0.0,
            momentum_energy=row[8] if row[8] is not None else 0.0,
            last_active=cls._parse_dt(row[9]),
            dangling_threads=cls._parse_dangling(row[10]),
            embedding_updated=row[11] if row[11] is not None else 0,
            last_distill_turn=row[12] if row[12] is not None else 0,
            last_distill_at=cls._parse_dt(row[13]),
        )

    # ── ISessionRepository 接口实现 ─────────────────────────────

    async def get(self, session_id: SessionId) -> Optional[Session]:
        """获取会话状态。

        查询 session_state 表，返回完整的 Session 实体。
        若会话不存在则返回 None。

        Args:
            session_id: 会话标识

        Returns:
            Session 实体，或 None
        """
        try:
            row = self._conn.execute(
                f"SELECT {_SESSION_COLUMNS} FROM session_state WHERE session_id = ?",
                (session_id.value,),
            ).fetchone()
        except sqlite3.Error as e:
            logger.error("获取会话状态失败 session_id=%s: %s", session_id.value, e)
            raise
        return self._row_to_session(row) if row else None

    async def save(self, session: Session) -> None:
        """持久化会话状态（覆盖写入）。

        使用 INSERT OR REPLACE 将 Session 实体全字段写入 session_state 表。

        Args:
            session: 待持久化的 Session 实体
        """
        try:
            self._conn.execute(
                f"INSERT OR REPLACE INTO session_state ({_SESSION_COLUMNS}) "
                f"VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    session.session_id.value,
                    session.user_id.value if session.user_id else None,
                    session.stance,
                    session.mood,
                    session.turn_count,
                    session.stance_turns,
                    session.engagement_level,
                    session.momentum_depth,
                    session.momentum_energy,
                    self._format_dt(session.last_active),
                    json.dumps(session.dangling_threads, ensure_ascii=False),
                    session.embedding_updated,
                    session.last_distill_turn,
                    self._format_dt(session.last_distill_at),
                ),
            )
            self._conn.commit()
        except sqlite3.Error as e:
            self._conn.rollback()
            logger.error(
                "保存会话状态失败 session_id=%s: %s",
                session.session_id.value,
                e,
            )
            raise

    async def increment_turn(self, session_id: SessionId) -> int:
        """原子递增会话轮次计数器。

        若会话尚不存在，自动初始化为 turn_count=1。
        三步操作在同一事务中执行以保证原子性。

        Args:
            session_id: 会话标识

        Returns:
            递增后的 turn_count
        """
        try:
            # 1. 确保会话行存在（INSERT OR IGNORE 并发安全）
            self._conn.execute(
                "INSERT OR IGNORE INTO session_state (session_id, turn_count, last_active) "
                "VALUES (?, 0, ?)",
                (session_id.value, self._now_str()),
            )
            # 2. 原子递增
            self._conn.execute(
                "UPDATE session_state SET turn_count = turn_count + 1 WHERE session_id = ?",
                (session_id.value,),
            )
            # 3. 读取新值
            row = self._conn.execute(
                "SELECT turn_count FROM session_state WHERE session_id = ?",
                (session_id.value,),
            ).fetchone()
            self._conn.commit()
            return row[0] if row and row[0] is not None else 1
        except sqlite3.Error as e:
            self._conn.rollback()
            logger.error("递增轮次失败 session_id=%s: %s", session_id.value, e)
            raise

    async def update_last_active(
        self,
        session_id: SessionId,
        now: datetime,
    ) -> None:
        """更新会话的最后活跃时间。

        Args:
            session_id: 会话标识
            now: 当前时间
        """
        try:
            self._conn.execute(
                "UPDATE session_state SET last_active = ? WHERE session_id = ?",
                (self._format_dt(now), session_id.value),
            )
            self._conn.commit()
        except sqlite3.Error as e:
            self._conn.rollback()
            logger.error(
                "更新 last_active 失败 session_id=%s: %s",
                session_id.value,
                e,
            )
            raise

    async def get_dangling_threads(
        self,
        session_id: SessionId,
    ) -> dict:
        """获取会话的未完成话题。

        读取 dangling_threads JSON 字段并解析为 dict。
        若会话不存在或字段为空/解析失败，均返回空 dict。

        Args:
            session_id: 会话标识

        Returns:
            未完成话题字典，格式为 {"threads": [...], "turn": N} 或 {}
        """
        try:
            row = self._conn.execute(
                "SELECT dangling_threads FROM session_state WHERE session_id = ?",
                (session_id.value,),
            ).fetchone()
        except sqlite3.Error as e:
            logger.error(
                "获取 dangling_threads 失败 session_id=%s: %s",
                session_id.value,
                e,
            )
            raise
        return self._parse_dangling(row[0]) if row else {}

    async def update_dangling_threads(
        self,
        session_id: SessionId,
        threads: dict,
    ) -> None:
        """写入会话的未完成话题。

        threads dict 被序列化为 JSON 字符串后写入 dangling_threads 字段。

        Args:
            session_id: 会话标识
            threads: 未完成话题字典，格式应与 {"threads": [...], "turn": N} 一致
        """
        try:
            self._conn.execute(
                "UPDATE session_state SET dangling_threads = ? WHERE session_id = ?",
                (json.dumps(threads, ensure_ascii=False), session_id.value),
            )
            self._conn.commit()
        except sqlite3.Error as e:
            self._conn.rollback()
            logger.error(
                "更新 dangling_threads 失败 session_id=%s: %s",
                session_id.value,
                e,
            )
            raise

    async def get_last_distill(
        self,
        session_id: SessionId,
    ) -> tuple[int, datetime | None]:
        """获取已蒸馏的最后轮次和时间。

        读取 last_distill_turn 和 last_distill_at 字段。
        若会话不存在，返回 (0, None)。

        Args:
            session_id: 会话标识

        Returns:
            (last_distill_turn, last_distill_at) 元组
        """
        try:
            row = self._conn.execute(
                "SELECT last_distill_turn, last_distill_at "
                "FROM session_state WHERE session_id = ?",
                (session_id.value,),
            ).fetchone()
        except sqlite3.Error as e:
            logger.error(
                "获取 last_distill 失败 session_id=%s: %s",
                session_id.value,
                e,
            )
            raise
        if not row:
            return (0, None)
        return (row[0] if row[0] is not None else 0, self._parse_dt(row[1]))

    async def update_last_distill(
        self,
        session_id: SessionId,
        turn: int,
        now: datetime,
    ) -> None:
        """更新蒸馏进度。

        同时更新 last_distill_turn 和 last_distill_at 字段。

        Args:
            session_id: 会话标识
            turn: 当前蒸馏到的轮次
            now: 蒸馏完成时间
        """
        try:
            self._conn.execute(
                "UPDATE session_state SET last_distill_turn = ?, last_distill_at = ? "
                "WHERE session_id = ?",
                (turn, self._format_dt(now), session_id.value),
            )
            self._conn.commit()
        except sqlite3.Error as e:
            self._conn.rollback()
            logger.error(
                "更新 last_distill 失败 session_id=%s: %s",
                session_id.value,
                e,
            )
            raise

    # ── ISessionQueryRepository（session_warmup 所需）────────────────

    async def get_most_recent_excluding(
        self, exclude_session_id: str,
    ) -> Optional[Session]:
        """获取最近活跃的、非指定 session 的会话。

        查询 session_state 表中除 exclude_session_id 外最近活跃的会话，
        用于新 session 预热（session_warmup）— 展示上一个 session 的未完成话题。

        Args:
            exclude_session_id: 排除的会话标识（通常是当前 session）

        Returns:
            最近活跃的 Session 实体，如果没有任何其他 session 则返回 None
        """
        try:
            row = self._conn.execute(
                f"SELECT {_SESSION_COLUMNS} FROM session_state "
                "WHERE session_id != ? AND last_active IS NOT NULL "
                "ORDER BY last_active DESC LIMIT 1",
                (exclude_session_id,),
            ).fetchone()
        except sqlite3.Error as e:
            logger.error(
                "获取最近会话失败 exclude=%s: %s", exclude_session_id, e,
            )
            raise
        return self._row_to_session(row) if row else None

"""
SQLiteSharedContextRepository — 共享上下文 SQLite 实现
"""
from __future__ import annotations

import logging
import sqlite3
from domain.ports.clock import IClock

logger = logging.getLogger("rcms")


class SQLiteSharedContextRepository:
    def __init__(self, conn: sqlite3.Connection, clock: IClock):
        self._conn = conn
        self._clock = clock

    async def upsert_joke(self, user_id: str, trigger: str, context: str) -> None:
        try:
            row = self._conn.execute(
                "SELECT context_id FROM shared_context WHERE user_id=? AND context_body LIKE ?",
                (user_id, f'%{trigger}%')
            ).fetchone()
            if row:
                self._conn.execute(
                    "UPDATE shared_context SET omission_count=omission_count+1 WHERE context_id=?",
                    (row[0],)
                )
            else:
                self._conn.execute(
                    "INSERT INTO shared_context (user_id, context_body, omission_count, confirmed) VALUES (?, ?, 1, 1)",
                    (user_id, f'[梗] {trigger} → {context}')
                )
            self._conn.commit()
        except sqlite3.Error as e:
            self._conn.rollback()
            logger.error("upsert_joke 失败 user=%s trigger=%s: %s", user_id, trigger, e)
            raise

    async def get_recent(self, user_id: str, limit: int = 4) -> list[str]:
        try:
            rows = self._conn.execute(
                "SELECT context_body FROM shared_context WHERE user_id=? ORDER BY context_id DESC LIMIT ?",
                (user_id, limit)
            ).fetchall()
            return [r[0] for r in rows]
        except sqlite3.Error as e:
            logger.error("get_recent 失败 user=%s: %s", user_id, e)
            raise

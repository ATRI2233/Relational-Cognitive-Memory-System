"""
SQLite 认知记忆仓储实现
=======================

实现 IMemoryRepository Protocol 的全部 10 个方法。
主要负责 cognitive_distill 表（核心蒸馏记忆）的增删改查、
chat_history 的对话记录写入、session_state 的轮次更新，
以及向量嵌入的存储/检索。

依赖：
  - sqlite3.Connection — 数据库连接
  - domain.ports.clock.IClock — 可注入的时间源
  - domain.entities.memory — Memory, MemoryId, UserId 等实体/值对象
"""

from __future__ import annotations

import array
import logging
import math
import sqlite3
from datetime import datetime
from typing import Optional

from domain.entities.memory import Importance, Memory, MemoryId, Mood, SessionId, UserId
from domain.ports.clock import IClock
from domain.ports.repositories import IMemoryRepository
from infrastructure.persistence.ddl import wal_checkpoint

logger = logging.getLogger("rcms")

_TIMESTAMP_FMT = "%Y-%m-%d %H:%M:%S"

# cognitive_distill 表所有列的常量，保证读写操作的列序一致
_COGNITIVE_COLUMNS = (
    "id, user_id, session_id, content, keylabel, summary, "
    "mood, mood_intensity, importance, entities, embedding, "
    "turn_num, created_at, embedding_dim, expires_at"
)


class SQLiteMemoryRepository(IMemoryRepository):
    """SQLite 认知记忆仓储。

    实现 IMemoryRepository Protocol，主要操作 cognitive_distill 表，
    辅助操作 chat_history 和 session_state 表以支持 save_turn 方法。

    Args:
        conn: SQLite 数据库连接
        clock: 可注入的时间源
    """

    def __init__(self, conn: sqlite3.Connection, clock: IClock) -> None:
        self._conn = conn
        self._clock = clock
        # 嵌入向量缓存：user_id -> {"vectors": list[list[float]], "meta": [(id, content)]}
        self._emb_cache: dict[str, dict] = {}

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
    def _embed_to_blob(vec: list[float]) -> bytes:
        """将浮点向量序列化为 BLOB（array.array('f') 二进制）。

        Args:
            vec: 浮点向量

        Returns:
            序列化后的 bytes
        """
        return array.array("f", vec).tobytes()

    @staticmethod
    def _blob_to_embed(blob: bytes) -> list[float]:
        """将 BLOB 反序列化为浮点向量。

        Args:
            blob: 序列化的向量 bytes

        Returns:
            浮点向量列表
        """
        return array.array("f", blob).tolist()

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """计算两个向量的余弦相似度。

        Args:
            a: 向量 A
            b: 向量 B

        Returns:
            [-1.0, 1.0] 范围内的余弦相似度，维度不匹配时返回 0.0
        """
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    @classmethod
    def _row_to_memory(cls, row: sqlite3.Row) -> Memory:
        """将 cognitive_distill 表的一行转换为 Memory 实体。

        Args:
            row: sqlite3.Row 查询结果

        Returns:
            构造完成的 Memory 实体

        Note:
            _COGNITIVE_COLUMNS 定义的列序必须与 SELECT 一致:
            id, user_id, session_id, content, keylabel, summary,
            mood, mood_intensity, importance, entities, embedding,
            turn_num, created_at, embedding_dim, expires_at
        """
        return Memory(
            memory_id=MemoryId(row[0]),
            user_id=UserId(row[1]) if row[1] else UserId(""),
            session_id=SessionId(row[2]) if row[2] else None,
            content=row[3] or "",
            keylabel=row[4] or "",
            summary=row[5] or "",
            mood=Mood(row[6] or ""),
            mood_intensity=row[7] if row[7] is not None else 0.0,
            importance=Importance(row[8] if row[8] is not None else 0.3),
            entities=row[9] or "[]",
            embedding=row[10],
            turn_num=row[11] if row[11] is not None else 0,
            created_at=cls._parse_dt(row[12]) or datetime.now(),
            embedding_dim=row[13],
            expires_at=cls._parse_dt(row[14]),
        )

    # ── 查询构建辅助 ─────────────────────────────────────────────

    @staticmethod
    def _build_kw_clauses(
        keywords: list[str],
    ) -> tuple[list[str], list[str]]:
        """为关键词检索构建 WHERE 子句片段和参数。

        每个关键词需要在 content、keylabel、summary 三列中分别做
        LIKE 匹配，同一关键词的三列条件用 OR 连接，
        不同关键词之间用 AND 连接。

        Args:
            keywords: 搜索关键词列表

        Returns:
            (clauses, params) 元组，clauses 为 WHERE 片段列表，
            params 为对应的参数列表
        """
        clauses: list[str] = []
        params: list[str] = []
        for kw in keywords:
            pattern = f"%{kw}%"
            clauses.append(
                "(content LIKE ? OR keylabel LIKE ? OR summary LIKE ?)"
            )
            params.extend([pattern, pattern, pattern])
        return clauses, params

    # ── IMemoryRepository 接口实现 ──────────────────────────────

    async def save(self, memory: Memory) -> MemoryId:
        """持久化一条记忆记录。

        将 Memory 实体的各字段写入 cognitive_distill 表。
        memory_id 由 SQLite 自增主键生成，以 MemoryId 形式返回。

        Args:
            memory: 待持久化的 Memory 实体

        Returns:
            由数据库生成的 MemoryId
        """
        try:
            cur = self._conn.execute(
                "INSERT INTO cognitive_distill "
                "(user_id, session_id, content, keylabel, summary, "
                " mood, mood_intensity, importance, entities, embedding, "
                " turn_num, created_at, embedding_dim, expires_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    memory.user_id.value if memory.user_id else None,
                    memory.session_id.value if memory.session_id else None,
                    memory.content,
                    memory.keylabel,
                    memory.summary,
                    memory.mood.value,
                    memory.mood_intensity,
                    memory.importance.value,
                    memory.entities,
                    memory.embedding,
                    memory.turn_num,
                    self._format_dt(memory.created_at),
                    memory.embedding_dim,
                    self._format_dt(memory.expires_at),
                ),
            )
            self._conn.commit()
            new_id = MemoryId(cur.lastrowid)
            logger.debug("记忆已保存 id=%s user=%s", new_id.value, memory.user_id.value)
            return new_id
        except sqlite3.Error as e:
            self._conn.rollback()
            logger.error(
                "保存记忆失败 user=%s: %s",
                memory.user_id.value if memory.user_id else "?",
                e,
            )
            raise

    async def save_turn(
        self,
        session_id: SessionId,
        user_input: str,
        reply: str,
        user_id: UserId | None = None,
        sender_name: str = "",
        importance: float = 0.3,
        mood: str = "",
    ) -> None:
        """写入一轮对话到聊天历史。

        同时写入 user 和 assistant 两条记录（共享 turn_num），
        并更新 session_state 的 turn_count 和 last_active。

        turn_num 通过查询 chat_history 中该会话的 MAX(turn_num) + 1 得到，
        保证同一轮 user 与 assistant 记录共享编号。

        Args:
            session_id: 会话标识
            user_input: 用户输入
            reply: AI 回复
            user_id: 用户标识
            sender_name: 发送者昵称（用于 user_mappings 自动注册）
            importance: 本轮对话的重要性评分
            mood: 情绪标签
        """
        uid_str = user_id.value if user_id else ""
        now_str = self._now_str()
        try:
            # 1. 开启 IMMEDIATE 事务，保证 turn_num 互斥读取
            if not self._conn.in_transaction:
                self._conn.execute("BEGIN IMMEDIATE")

            # 2. 在事务内获取下一个 turn_num（避免并发 race condition）
            row = self._conn.execute(
                "SELECT COALESCE(MAX(turn_num), 0) + 1 FROM chat_history WHERE session_id = ?",
                (session_id.value,),
            ).fetchone()
            turn_num = row[0] if row else 1

            # 3. 写入 user 记录
            self._conn.execute(
                "INSERT INTO chat_history "
                "(session_id, role, content, turn_num, importance, mood, user_id, sender_name, created_at) "
                "VALUES (?, 'user', ?, ?, ?, ?, ?, ?, ?)",
                (
                    session_id.value,
                    user_input,
                    turn_num,
                    importance,
                    mood,
                    uid_str,
                    sender_name,
                    now_str,
                ),
            )

            # 4. 写入 assistant 记录（共享 turn_num）
            self._conn.execute(
                "INSERT INTO chat_history "
                "(session_id, role, content, turn_num, importance, mood, user_id, sender_name, created_at) "
                "VALUES (?, 'assistant', ?, ?, ?, ?, ?, ?, ?)",
                (
                    session_id.value,
                    reply,
                    turn_num,
                    importance,
                    mood,
                    uid_str,
                    sender_name,
                    now_str,
                ),
            )

            # 5. 确保 session_state 行存在并更新 turn_count 和 last_active
            self._conn.execute(
                "INSERT OR IGNORE INTO session_state (session_id, turn_count, last_active) "
                "VALUES (?, 0, ?)",
                (session_id.value, now_str),
            )
            self._conn.execute(
                "UPDATE session_state SET turn_count = turn_count + 1, last_active = ? "
                "WHERE session_id = ?",
                (now_str, session_id.value),
            )

            self._conn.commit()
            # WAL checkpoint after each turn (from old session.py:41-49)
            wal_checkpoint(self._conn, force_truncate=False)
            logger.debug(
                "对话轮次已保存 session=%s turn=%d", session_id.value, turn_num
            )
        except sqlite3.Error as e:
            self._conn.rollback()
            logger.error(
                "保存对话轮次失败 session=%s: %s", session_id.value, e
            )
            raise

    async def search_by_keywords(
        self,
        user_id: UserId,
        keywords: list[str],
        limit: int = 5,
        min_importance: float = 0.0,
        time_filter: tuple[int, int] | None = None,
    ) -> list[Memory]:
        """按关键词模糊检索记忆。

        用于通道 2（语义共振）的非向量保底路径。
        在 cognitive_distill 表的 content、keylabel、summary 三列中
        执行 LIKE 匹配，支持多关键词联合检索（AND 逻辑）。
        结果默认按重要性降序、创建时间降序排列。

        Args:
            user_id: 用户标识
            keywords: 搜索关键词列表（LIKE '%kw%' 匹配）
            limit: 最大返回条数
            min_importance: 最低重要性阈值
            time_filter: 可选时间范围 (min_days_ago, max_days_ago)，
                        仅返回该天数范围内的记忆

        Returns:
            按重要性降序排列的 Memory 实体列表
        """
        if not keywords:
            return []

        kw_clauses, kw_params = self._build_kw_clauses(keywords)
        params: list = [user_id.value]
        time_clause = ""

        if time_filter is not None:
            min_days, max_days = time_filter
            time_clause = (
                "AND created_at >= datetime('now', ?) "
                "AND created_at <= datetime('now', ?) "
            )
            params.append(f"-{max_days} days")
            params.append(f"-{min_days} days")

        expired_clause = "AND (expires_at IS NULL OR expires_at > datetime('now'))"

        params.extend(kw_params)
        params.extend([min_importance, limit])

        try:
            rows = self._conn.execute(
                f"SELECT {_COGNITIVE_COLUMNS} FROM cognitive_distill "
                "WHERE user_id = ? "
                f"{time_clause}"
                f"{expired_clause} "
                f"AND ({' AND '.join(kw_clauses)}) "
                "AND importance >= ? "
                "ORDER BY importance DESC, created_at DESC "
                "LIMIT ?",
                params,
            ).fetchall()
        except sqlite3.Error as e:
            logger.error(
                "关键词检索失败 user=%s: %s", user_id.value, e
            )
            raise

        return [self._row_to_memory(r) for r in rows]

    async def search_by_embedding(
        self,
        user_id: UserId,
        query_vec: list[float],
        limit: int = 5,
    ) -> list[tuple[Memory, float]]:
        """按向量嵌入余弦相似度检索记忆。

        用于通道 2（语义共振）的主路径。
        从嵌入缓存中加载用户的所有向量，计算每个向量与 query_vec
        的余弦相似度，取 top-limit 返回 (Memory, cosine_similarity) 元组列表。

        Args:
            user_id: 用户标识
            query_vec: 查询文本的向量嵌入
            limit: 最大返回条数

        Returns:
            (Memory, 余弦相似度) 元组列表，按相似度降序
        """
        uid = user_id.value

        # 确保缓存已加载
        if uid not in self._emb_cache:
            await self.load_emb_cache(user_id)

        cache = self._emb_cache.get(uid, {})
        vectors = cache.get("vectors", [])
        meta = cache.get("meta", [])
        if not vectors:
            return []

        # 计算每条缓存向量与 query_vec 的余弦相似度
        scored: list[tuple[int, float]] = []
        for i, emb_vec in enumerate(vectors):
            sim = self._cosine_similarity(query_vec, emb_vec)
            if sim > 0.0:
                scored.append((meta[i][0], sim))

        # 按相似度降序排序，取 top limit
        scored.sort(key=lambda x: -x[1])
        top = scored[:limit]

        if not top:
            return []

        # 批量获取匹配记录对应的完整 Memory 实体
        record_ids = [rid for rid, _ in top]
        placeholders = ",".join("?" * len(record_ids))
        try:
            rows = self._conn.execute(
                f"SELECT {_COGNITIVE_COLUMNS} FROM cognitive_distill "
                f"WHERE id IN ({placeholders})",
                record_ids,
            ).fetchall()
        except sqlite3.Error as e:
            logger.error(
                "向量检索后获取记忆失败 user=%s: %s", uid, e
            )
            raise

        # 构建 id -> Memory 映射
        mem_map: dict[int, Memory] = {}
        for r in rows:
            mem = self._row_to_memory(r)
            mem_map[mem.memory_id.value] = mem

        # 按相似度排序返回
        result: list[tuple[Memory, float]] = []
        for rid, sim in top:
            mem = mem_map.get(rid)
            if mem is not None:
                result.append((mem, sim))

        logger.debug(
            "向量检索完成 user=%s candidates=%d returned=%d",
            uid,
            len(cache),
            len(result),
        )
        return result

    async def get_by_user(
        self,
        user_id: UserId,
        limit: int = 10,
        offset: int = 0,
        min_importance: float = 0.0,
    ) -> list[Memory]:
        """获取某用户的所有记忆（分页）。

        从 cognitive_distill 表中查询指定用户、重要性阈值以上的记忆，
        按 created_at 降序排列，支持分页。

        Args:
            user_id: 用户标识
            limit: 每页条数
            offset: 偏移量
            min_importance: 最低重要性过滤

        Returns:
            Memory 实体列表，按 created_at 降序
        """
        try:
            rows = self._conn.execute(
                f"SELECT {_COGNITIVE_COLUMNS} FROM cognitive_distill "
                "WHERE user_id = ? AND importance >= ? "
                "ORDER BY created_at DESC "
                "LIMIT ? OFFSET ?",
                (user_id.value, min_importance, limit, offset),
            ).fetchall()
        except sqlite3.Error as e:
            logger.error(
                "获取用户记忆失败 user=%s: %s", user_id.value, e
            )
            raise

        return [self._row_to_memory(r) for r in rows]

    async def get_unembedded(
        self,
        user_id: UserId,
        limit: int = 100,
    ) -> list[Memory]:
        """获取尚未生成向量嵌入的记忆记录。

        对应 embedding IS NULL 的查询，按 id 升序排列。
        用于后台嵌入任务批量处理。

        Args:
            user_id: 用户标识
            limit: 最大返回条数

        Returns:
            未嵌入的 Memory 实体列表（按 id 升序）
        """
        try:
            rows = self._conn.execute(
                f"SELECT {_COGNITIVE_COLUMNS} FROM cognitive_distill "
                "WHERE user_id = ? AND embedding IS NULL "
                "ORDER BY id ASC "
                "LIMIT ?",
                (user_id.value, limit),
            ).fetchall()
        except sqlite3.Error as e:
            logger.error(
                "获取未嵌入记忆失败 user=%s: %s", user_id.value, e
            )
            raise

        return [self._row_to_memory(r) for r in rows]

    async def store_embedding(
        self,
        record_id: MemoryId,
        embedding: list[float],
    ) -> None:
        """为指定记忆记录存储向量嵌入。

        幂等写入：仅在 embedding IS NULL 时写入，
        避免并发场景下的重复覆盖。
        同时记录嵌入维度到 embedding_dim 字段。

        Args:
            record_id: 记忆记录 ID
            embedding: 向量嵌入列表
        """
        blob = self._embed_to_blob(embedding)
        dim = len(embedding)
        try:
            cur = self._conn.execute(
                "UPDATE cognitive_distill SET embedding = ?, embedding_dim = ? "
                "WHERE id = ? AND embedding IS NULL",
                (blob, dim, record_id.value),
            )
            self._conn.commit()
            if cur.rowcount > 0:
                # 更新内存缓存，确保后续 search_by_embedding 可命中
                uid_row = self._conn.execute(
                    "SELECT user_id, content FROM cognitive_distill WHERE id = ?",
                    (record_id.value,),
                ).fetchone()
                if uid_row:
                    uid = uid_row[0]
                    content = uid_row[1] or ""
                    if uid in self._emb_cache:
                        self._emb_cache[uid]["vectors"].append(embedding)
                        self._emb_cache[uid]["meta"].append((record_id.value, content))
                logger.debug(
                    "嵌入已存储 id=%s dim=%d", record_id.value, dim
                )
            else:
                logger.debug(
                    "嵌入跳过（已存在）id=%s", record_id.value
                )
        except sqlite3.Error as e:
            self._conn.rollback()
            logger.error(
                "存储嵌入失败 id=%s: %s", record_id.value, e
            )
            raise

    async def delete_expired(self, user_id: UserId) -> int:
        """删除所有已过期的记忆记录。

        对应 expires_at <= now 的条件清理。

        Args:
            user_id: 用户标识

        Returns:
            被删除的记录数
        """
        now_str = self._now_str()
        try:
            cur = self._conn.execute(
                "DELETE FROM cognitive_distill "
                "WHERE user_id = ? AND expires_at IS NOT NULL AND expires_at <= ?",
                (user_id.value, now_str),
            )
            self._conn.commit()
            count = cur.rowcount
            if count > 0:
                logger.info(
                    "过期记忆已清理 user=%s count=%d", user_id.value, count
                )
            return count
        except sqlite3.Error as e:
            self._conn.rollback()
            logger.error(
                "删除过期记忆失败 user=%s: %s", user_id.value, e
            )
            raise

    async def load_emb_cache(self, user_id: UserId) -> None:
        """加载用户所有嵌入向量到内存缓存。

        从 cognitive_distill 读取所有 embedding IS NOT NULL 的记录，
        反序列化后存入 _emb_cache（dict 格式含 vectors + meta）。
        处理维度不匹配的条目时自动调用 mark_rebuild 标记重建。

        Args:
            user_id: 用户标识
        """
        uid = user_id.value
        try:
            rows = self._conn.execute(
                "SELECT id, content, embedding, embedding_dim FROM cognitive_distill "
                "WHERE user_id = ? AND embedding IS NOT NULL "
                "ORDER BY id ASC",
                (uid,),
            ).fetchall()
        except sqlite3.Error as e:
            logger.error(
                "加载嵌入缓存失败 user=%s: %s", uid, e
            )
            raise

        empty_cache: dict = {"vectors": [], "meta": []}

        if not rows:
            self._emb_cache[uid] = empty_cache
            logger.debug("嵌入缓存已清空 user=%s（无嵌入数据）", uid)
            return

        vectors: list[list[float]] = []
        meta: list[tuple[int, str]] = []
        expected_dim: int | None = None

        for row in rows:
            rec_id = row[0]
            content = row[1] or ""
            blob = row[2]
            db_dim = row[3]

            if blob is None:
                continue

            try:
                vec = self._blob_to_embed(blob)
            except Exception as e:
                logger.warning(
                    "嵌入反序列化失败 id=%s: %s，标记重建", rec_id, e
                )
                await self.mark_rebuild(user_id, MemoryId(rec_id), "deserialize_failed")
                continue

            # 确定标准维度
            if expected_dim is None:
                expected_dim = db_dim if db_dim else len(vec)

            # 维度检查
            if len(vec) == expected_dim:
                vectors.append(vec)
                meta.append((rec_id, content))
            else:
                logger.warning(
                    "维度不匹配 id=%s dim=%d (expected=%d)，标记重建",
                    rec_id,
                    len(vec),
                    expected_dim,
                )
                self._conn.execute(
                    "UPDATE cognitive_distill SET embedding = NULL, embedding_dim = NULL WHERE id = ?",
                    (rec_id,),
                )
                self._conn.commit()
                await self.mark_rebuild(user_id, MemoryId(rec_id), "dim_mismatch")

        self._emb_cache[uid] = {"vectors": vectors, "meta": meta}
        logger.info(
            "嵌入缓存已加载 user=%s vectors=%d dim=%s",
            uid,
            len(vectors),
            expected_dim or "N/A",
        )

    async def mark_rebuild(
        self,
        user_id: UserId,
        record_id: MemoryId,
        reason: str = "",
    ) -> None:
        """标记一条记录需要重新生成向量嵌入。

        将 record_id 加入 embedding_rebuild_queue，
        供后台重建任务消费。

        Args:
            user_id: 用户标识
            record_id: 需要重建的记忆记录 ID
            reason: 重建原因（如 'dim_mismatch'）
        """
        try:
            self._conn.execute(
                "INSERT OR IGNORE INTO embedding_rebuild_queue "
                "(user_id, record_id, reason) "
                "VALUES (?, ?, ?)",
                (user_id.value, record_id.value, reason),
            )
            self._conn.commit()
            logger.debug(
                "嵌入重建已标记 user=%s id=%s reason=%s",
                user_id.value,
                record_id.value,
                reason,
            )
        except sqlite3.Error as e:
            self._conn.rollback()
            logger.error(
                "标记嵌入重建失败 user=%s id=%s: %s",
                user_id.value,
                record_id.value,
                e,
            )
            raise

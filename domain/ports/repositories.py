"""
仓储接口（Repository Protocols）
====================================

定义 IMemoryRepository 和 ISessionRepository 的 Protocol 接口，
作为领域层与基础设施层的持久化协约。

依赖：
  - domain.entities.memory — Memory, MemoryId, UserId, SessionId
  - domain.entities.session — Session, TurnRecord

所有方法均为异步，返回领域实体类型。
不含任何 try/except 或第三方库依赖。
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional, Protocol, runtime_checkable

from domain.entities.memory import Importance, Memory, MemoryId, Mood, UserId, SessionId
from domain.entities.session import Session, TurnRecord


@runtime_checkable
class IMemoryRepository(Protocol):
    """认知记忆仓储接口。

    负责 cognitive_distill 表（核心蒸馏记忆）的
    增删改查、向量嵌入存储、关键词/向量/用户级检索、过期清理与重建。

    所有方法均为异步协程。
    """

    async def save(self, memory: Memory) -> MemoryId:
        """持久化一条记忆记录。

        新增记录时 memory_id 由仓储生成，以返回值形式返回。

        Args:
            memory: 待持久化的 Memory 实体

        Returns:
            由仓储生成的 MemoryId
        """
        ...

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
        更新 session_state 的 turn_count 和 last_active。

        Args:
            session_id: 会话标识
            user_input: 用户输入
            reply: AI 回复
            user_id: 用户标识
            sender_name: 发送者昵称（用于 user_mappings 自动注册）
            importance: 本轮对话的重要性评分
            mood: 情绪标签
        """
        ...

    async def search_by_keywords(
        self,
        user_id: UserId,
        keywords: list[str],
        limit: int = 5,
        min_importance: float = 0.0,
        time_filter: tuple[int, int] | None = None,
    ) -> list[Memory]:
        """按关键词模糊检索记忆。

        用于通道 2（语义共振）的非向量保底路径，
        以及通道 1（时间重要性）的时间衰减排序。

        Args:
            user_id: 用户标识
            keywords: 搜索关键词列表（LIKE '%kw%' 匹配）
            limit: 最大返回条数
            min_importance: 最低重要性阈值
            time_filter: 可选时间范围 (min_days_ago, max_days_ago)，
                        仅返回该天数范围内的记忆

        Returns:
            按相关度降序排列的 Memory 实体列表
        """
        ...

    async def search_by_embedding(
        self,
        user_id: UserId,
        query_vec: list[float],
        limit: int = 5,
    ) -> list[tuple[Memory, float]]:
        """按向量嵌入余弦相似度检索记忆。

        用于通道 2（语义共振）的主路径。
        返回 (Memory, cosine_similarity) 元组列表。

        Args:
            user_id: 用户标识
            query_vec: 查询文本的向量嵌入
            limit: 最大返回条数

        Returns:
            (Memory, 余弦相似度) 元组列表，按相似度降序
        """
        ...

    async def get_by_user(
        self,
        user_id: UserId,
        limit: int = 10,
        offset: int = 0,
        min_importance: float = 0.0,
    ) -> list[Memory]:
        """获取某用户的所有记忆（分页）。

        Args:
            user_id: 用户标识
            limit: 每页条数
            offset: 偏移量
            min_importance: 最低重要性过滤

        Returns:
            Memory 实体列表，按 created_at 降序
        """
        ...

    async def get_unembedded(
        self,
        user_id: UserId,
        limit: int = 100,
    ) -> list[Memory]:
        """获取尚未生成向量嵌入的记忆记录。

        用于后台嵌入任务批量处理。
        对应 embedding IS NULL 的查询。

        Args:
            user_id: 用户标识
            limit: 最大返回条数

        Returns:
            未嵌入的 Memory 实体列表（按 id 升序）
        """
        ...

    async def store_embedding(
        self,
        record_id: MemoryId,
        embedding: list[float],
    ) -> None:
        """为指定记忆记录存储向量嵌入。

        幂等写入：仅在 embedding IS NULL 时写入，
        避免并发场景下的重复覆盖。

        Args:
            record_id: 记忆记录 ID
            embedding: 向量嵌入列表
        """
        ...

    async def delete_expired(self, user_id: UserId) -> int:
        """删除所有已过期的记忆记录。

        对应 expires_at <= now 的条件清理。

        Args:
            user_id: 用户标识

        Returns:
            被删除的记录数
        """
        ...

    async def load_emb_cache(self, user_id: UserId) -> None:
        """加载用户所有嵌入向量到内存缓存。

        供通道 2（语义共振）中的批量余弦相似度计算使用。
        处理维度不匹配时自动标记重建。

        Args:
            user_id: 用户标识
        """
        ...

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
        ...


@runtime_checkable
class ISessionRepository(Protocol):
    """会话状态仓储接口。

    负责 session_state 表的增删改查、
    轮次管理、悬案线程追踪、蒸馏进度追踪。

    所有方法均为异步协程。
    """

    async def get(self, session_id: SessionId) -> Optional[Session]:
        """获取会话状态。

        Args:
            session_id: 会话标识

        Returns:
            如果会话存在则返回 Session 实体，否则返回 None
        """
        ...

    async def save(self, session: Session) -> None:
        """持久化会话状态（覆盖写入）。

        用于 session_state 表的新增与全字段更新。

        Args:
            session: 待持久化的 Session 实体
        """
        ...

    async def increment_turn(self, session_id: SessionId) -> int:
        """使会话轮次计数器 +1。

        原子递增 turn_count，返回递增后的新值。
        如果 session 尚不存在，自动初始化为 turn_count=1。

        Args:
            session_id: 会话标识

        Returns:
            递增后的 turn_count
        """
        ...

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
        ...

    async def get_dangling_threads(
        self,
        session_id: SessionId,
    ) -> dict:
        """获取会话的未完成话题。

        返回的 dict 格式为 {"threads": [...], "turn": N}。

        Args:
            session_id: 会话标识

        Returns:
            未完成话题字典，如果没有则返回空 dict
        """
        ...

    async def update_dangling_threads(
        self,
        session_id: SessionId,
        threads: dict,
    ) -> None:
        """写入会话的未完成话题。

        threads dict 格式应与 {"threads": [...], "turn": N} 一致。

        Args:
            session_id: 会话标识
            threads: 未完成话题字典
        """
        ...

    async def get_last_distill(
        self,
        session_id: SessionId,
    ) -> tuple[int, datetime | None]:
        """获取已蒸馏的最后轮次和时间。

        对应 last_distill_turn 和 last_distill_at 字段。

        Args:
            session_id: 会话标识

        Returns:
            (last_distill_turn, last_distill_at) 元组
        """
        ...

    async def update_last_distill(
        self,
        session_id: SessionId,
        turn: int,
        now: datetime,
    ) -> None:
        """更新蒸馏进度。

        同时更新 last_distill_turn 和 last_distill_at。

        Args:
            session_id: 会话标识
            turn: 当前蒸馏到的轮次
            now: 蒸馏完成时间
        """
        ...


@runtime_checkable
class IUserMappingRepository(Protocol):
    """用户映射仓储接口。

    对应 user_mappings 表的读写操作。
    记录 session 中各用户的昵称映射关系，
    用于跨会话用户身份追踪和多用户场景的身份识别。

    所有方法均为异步协程。
    """

    async def upsert_mapping(
        self,
        session_id: str,
        user_id: str,
        label: str,
        source: str = "",
    ) -> None:
        """注册或更新用户映射。

        使用 INSERT OR IGNORE 写入，避免重复覆盖。
        同 (session_id, user_id, label) 的记录仅写入一次。

        Args:
            session_id: 会话标识
            user_id: 用户标识
            label: 显示名称/标签
            source: 来源（如 'nickname'、'custom'），默认空字符串
        """
        ...

    async def find_mentioned(
        self,
        session_id: str,
        text: str,
        speaker_id: str = "",
    ) -> list[tuple[str, str]]:
        """扫描消息文本，返回被提及的用户。

        三级匹配策略：
          1. 当前 session 中的已知用户映射
          2. 发言者参与过的其他 session 中的映射
          3. 全局映射兜底

        Args:
            session_id: 当前会话标识
            text: 消息文本
            speaker_id: 发言者标识，用于排除自己和跨 session 搜索

        Returns:
            [(user_id, label), ...] 被提及的用户标识和显示名列表
        """
        ...

    async def get_labels(
        self,
        session_id: str,
        user_id: str,
    ) -> list[str]:
        """获取某用户在指定 session 中的所有昵称/标签。

        Args:
            session_id: 会话标识
            user_id: 用户标识

        Returns:
            标签字符串列表
        """
        ...

    async def bind_user_label(
        self,
        session_id: str,
        user_id: str,
        label: str,
        source: str = "",
    ) -> None:
        """绑定用户的显示名称/标签到当前 session。

        使用 INSERT OR REPLACE，覆盖已有同源标签。

        Args:
            session_id: 会话标识
            user_id: 用户标识
            label: 显示名称/标签
            source: 来源（如 'nickname'、'custom'），默认空字符串
        """
        ...

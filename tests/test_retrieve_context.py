"""
test_retrieve_context.py — RetrieveContextUseCase 专项测试
========================================================

覆盖场景：
  1. test_retrieve_memories_accepts_session_id_keyword
     — 回归测试：session_id 和 stance 作为关键字参数传递时不崩溃
  2. test_retrieve_memories_with_different_stances
     — casual / engaged 两种 stance 均正常工作
  3. test_graph_paths_populated_when_data_exists /
     test_graph_paths_empty_when_no_data
     — 图数据存在/不存在时 graph_paths 返回值正确
  4. test_retrieve_memories_empty_input
     — 空 user_input 不崩溃，优雅返回空列表

设计：
  - 使用纯 Mock 桩件，不依赖真实数据库
  - Settings() 提供 IRetrievalConfig / IWordListConfig 实现
  - FrozenClock 固定时间基准
  - asyncio.run() 执行协程方法
"""
from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pytest

# ── 将项目根目录加入 sys.path ──────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── 基础设施层 ──────────────────────────────────────────────────────
from infrastructure.clock import FrozenClock
from infrastructure.config.settings import Settings

# ── 应用层 ──────────────────────────────────────────────────────────
from application.use_cases.retrieve_context_use_case import (
    RetrieveContextUseCase,
    IEmbeddingService,
    ISharedContextRepository,
    IUserMappingRepository,
    ITextTemplateProvider,
    IRetrievalConfig,
    ISessionQueryRepository,
)

# ── 领域层 ──────────────────────────────────────────────────────────
from domain.entities.memory import Memory, MemoryId, UserId, SessionId, Importance, Mood
from domain.entities.session import Session
from domain.entities.graph import GraphNode, GraphEdge, DiffusionResult
from domain.entities.identity import Identity, Trait, Preferences, Boundary
from domain.services.fusion_service import FusionService, IFusionConfig
from domain.services.time_service import TimeService
from domain.services.keyword_service import KeywordService, ITokenizer, IWordListConfig


# ===================================================================
# Helper
# ===================================================================


def _run_async(coro):
    """同步包装器：在同步测试函数中安全执行协程。"""
    return asyncio.run(coro)


# ===================================================================
# Mock 分词器 & 词表配置
# ===================================================================


class MockTokenizer:
    """按空格和中文标点简单切分。"""

    def cut(self, text: str) -> list[str]:
        import re
        tokens: list[str] = []
        for t in re.split(r'[\s,，。！？、；：""''（）()—\n]+', text):
            if t:
                tokens.append(t)
        return tokens


@dataclass
class MockWordListConfig:
    """实现 IWordListConfig protocol。"""

    trivial_markers: list[str] = field(
        default_factory=lambda: ["吃", "喝", "睡", "饭"]
    )
    stop_words: list[str] = field(
        default_factory=lambda: ["今天", "什么", "怎么", "可以"]
    )
    time_words: dict[str, list[int]] = field(
        default_factory=lambda: {"今天": [0, 0], "最近": [0, 7]}
    )


# ===================================================================
# Mock Embedding 服务
# ===================================================================


class MockEmbeddingService:
    """实现 IEmbeddingService protocol — 返回固定维度假向量。"""

    async def embed(self, text: str) -> list[float]:
        return [0.001] * 256


# ===================================================================
# Mock Repository 桩件
# ===================================================================


class MockMemoryRepository:
    """实现 IMemoryRepository protocol — 所有检索返回空列表。"""

    async def get_by_user(
        self,
        user_id: UserId,
        limit: int = 10,
        offset: int = 0,
        min_importance: float = 0.0,
    ) -> list[Memory]:
        return []

    async def search_by_keywords(
        self,
        user_id: UserId,
        keywords: list[str],
        limit: int = 5,
        min_importance: float = 0.0,
        time_filter: tuple[int, int] | None = None,
    ) -> list[Memory]:
        return []

    async def search_by_embedding(
        self,
        user_id: UserId,
        query_vec: list[float],
        limit: int = 5,
    ) -> list[tuple[Memory, float]]:
        return []

    async def save(self, memory: Memory) -> MemoryId:
        return MemoryId(1)

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
        pass

    async def get_unembedded(
        self, user_id: UserId, limit: int = 100
    ) -> list[Memory]:
        return []

    async def store_embedding(
        self, record_id: MemoryId, embedding: list[float]
    ) -> None:
        pass

    async def delete_expired(self, user_id: UserId) -> int:
        return 0

    async def load_emb_cache(self, user_id: UserId) -> None:
        pass

    async def mark_rebuild(
        self, user_id: UserId, record_id: MemoryId, reason: str = ""
    ) -> None:
        pass


class MockIdentityRepository:
    """实现 IIdentityRepository protocol — 恒返回 None。"""

    async def get(self, user_id: str) -> Optional[Identity]:
        return None

    async def save_traits(
        self, user_id: str, traits: list[Trait]
    ) -> None:
        pass

    async def save_preferences(
        self, user_id: str, prefs: Preferences
    ) -> None:
        pass

    async def save_self_identity(
        self, user_id: str, identities: list[str]
    ) -> None:
        pass

    async def save_boundaries(
        self, user_id: str, boundaries: list[Boundary]
    ) -> None:
        pass

    async def update_identity(
        self, user_id: str, identity: Identity
    ) -> None:
        pass


class MockSessionRepository:
    """实现 ISessionRepository protocol — 返回默认值。"""

    async def get(self, session_id: SessionId) -> Optional[Session]:
        return None

    async def save(self, session: Session) -> None:
        pass

    async def increment_turn(self, session_id: SessionId) -> int:
        return 1

    async def update_last_active(
        self, session_id: SessionId, now: datetime
    ) -> None:
        pass

    async def get_dangling_threads(
        self, session_id: SessionId
    ) -> dict:
        return {}

    async def update_dangling_threads(
        self, session_id: SessionId, threads: dict
    ) -> None:
        pass

    async def get_last_distill(
        self, session_id: SessionId
    ) -> tuple[int, datetime | None]:
        return (0, None)

    async def update_last_distill(
        self, session_id: SessionId, turn: int, now: datetime
    ) -> None:
        pass


class MockSharedContextRepository:
    """实现 ISharedContextRepository protocol。"""

    async def get_recent(
        self, user_id: str, limit: int = 4
    ) -> list[str]:
        return []


class MockUserMappingRepository:
    """实现 IUserMappingRepository protocol。"""

    async def find_mentioned(
        self, session_id: str, text: str, speaker_id: str = ""
    ) -> list[tuple[str, str]]:
        return []

    async def get_labels(
        self, session_id: str, user_id: str
    ) -> list[str]:
        return []

    async def upsert_mapping(
        self,
        session_id: str,
        user_id: str,
        label: str,
        source: str = "",
    ) -> None:
        pass

    async def bind_user_label(
        self,
        session_id: str,
        user_id: str,
        label: str,
        source: str = "",
    ) -> None:
        pass


class MockSessionQueryRepository:
    """实现 ISessionQueryRepository protocol。"""

    async def get_most_recent_excluding(
        self, exclude_session_id: str
    ) -> Optional[Session]:
        return None


class MockTemplateProvider:
    """实现 ITextTemplateProvider protocol — 返回空/默认模板。"""

    def narrative_templates(self) -> dict:
        return {}

    def channel_labels(self) -> dict:
        return {
            "recent": "时间·重要性",
            "resonance": "语义检索",
            "skeleton": "图谱关联",
        }

    def memories_display_order(self) -> list:
        return ["resonance", "skeleton", "recent"]

    def prompt_compressor_templates(self) -> dict:
        return {}


# ===================================================================
# Mock Graph Repository（可配置返回数据开关）
# ===================================================================


class MockGraphRepository:
    """实现 IGraphRepository protocol — return_data 控制是否返回图数据。

    用于 TestGraphPaths 测试类：
      - return_data=True  第一次调用产生图路径
      - return_data=False 第二次调用无图路径，验证路径已清空
    """

    def __init__(self, return_data: bool = False):
        self.return_data = return_data

    async def search_nodes(
        self, user_id: str, keyword: str
    ) -> list[GraphNode]:
        if self.return_data:
            return [GraphNode(
                node_id=1, user_id=user_id, label=keyword,
                freq=1, last_seen=None, entity_type="auto",
            )]
        return []

    async def get_nodes_by_user(self, user_id: str) -> list[GraphNode]:
        if self.return_data:
            return [
                GraphNode(
                    node_id=1, user_id=user_id, label="哲学",
                    freq=1, last_seen=None, entity_type="auto",
                ),
                GraphNode(
                    node_id=2, user_id=user_id, label="编程",
                    freq=1, last_seen=None, entity_type="auto",
                ),
            ]
        return []

    async def get_edges_by_node(self, node_id: int) -> list[GraphEdge]:
        if self.return_data:
            return [
                GraphEdge(
                    from_node_id=1,
                    to_node_id=2,
                    weight=1.0,
                    encounter_count=1,
                    relation="喜欢",
                    last_seen=None,
                    created_at=None,
                )
            ]
        return []

    async def get_chain_paths(
        self, user_id: str, labels: list[str]
    ) -> list[str]:
        if self.return_data:
            return ["哲学 --[喜欢]--> 编程"]
        return []

    async def bfs_diffuse(
        self, user_id: str, seed_ids: list[int], depth: int = 2
    ) -> list[DiffusionResult]:
        if self.return_data:
            return [DiffusionResult(label="哲学", score=0.8)]
        return []

    # ── 以下方法在 retrieve_memories 中不会被调用 ──
    async def upsert_node(
        self, user_id: str, label: str, entity_type: str = "auto"
    ) -> int:
        return 1

    async def upsert_edge(
        self, from_id: int, to_id: int, relation: str = ""
    ) -> None:
        pass

    async def maintain(self, user_id: str) -> None:
        pass


# ===================================================================
# 测试夹具
# ===================================================================


def _build_use_case(
    graph_repo: MockGraphRepository | None = None,
) -> RetrieveContextUseCase:
    """组装 RetrieveContextUseCase，所有依赖注入 Mock 桩件。

    Args:
        graph_repo: 可选的自定义图仓储 mock（默认无数据返回）。

    Returns:
        配置完成的 RetrieveContextUseCase 实例
    """
    clock = FrozenClock(datetime(2026, 6, 15, 10, 0, 0))
    settings = Settings()

    memory_repo = MockMemoryRepository()
    identity_repo = MockIdentityRepository()
    session_repo = MockSessionRepository()
    shared_context_repo = MockSharedContextRepository()
    user_mapping_repo = MockUserMappingRepository()
    session_query_repo = MockSessionQueryRepository()
    template_provider = MockTemplateProvider()

    graph_repo = graph_repo or MockGraphRepository(return_data=False)

    fusion_service = FusionService(settings.retrieval)
    time_service = TimeService(clock)
    keyword_service = KeywordService(
        MockTokenizer(), MockWordListConfig()
    )
    embed_service = MockEmbeddingService()

    return RetrieveContextUseCase(
        memory_repo=memory_repo,
        identity_repo=identity_repo,
        graph_repo=graph_repo,
        session_repo=session_repo,
        shared_context_repo=shared_context_repo,
        user_mapping_repo=user_mapping_repo,
        fusion_service=fusion_service,
        time_service=time_service,
        keyword_service=keyword_service,
        embed_service=embed_service,
        clock=clock,
        config=settings.retrieval,
        session_query_repo=session_query_repo,
        template_provider=template_provider,
    )


# ===================================================================
# 测试 1：session_id / stance 关键字参数
# ===================================================================


class TestRetrieveMemoriesKeywordArgs:
    """场景 1：回归测试 — 防止 session_id 和 stance 参数顺序错误崩溃。"""

    def test_retrieve_memories_accepts_session_id_keyword(self) -> None:
        """使用 session_id= 和 stance= 关键字参数不应崩溃且返回 tuple。"""
        uc = _build_use_case()

        result, gp = _run_async(
            uc.retrieve_memories(
                "test_user",
                "hello",
                session_id="test_session",
                stance="engaged",
            )
        )

        assert isinstance(result, list)


# ===================================================================
# 测试 2：不同 stance 值
# ===================================================================


class TestRetrieveMemoriesStances:
    """场景 2：casual 和 engaged 两种 stance 均正常工作。"""

    def test_retrieve_memories_with_different_stances(self) -> None:
        """'casual' 返回空列表，'engaged' 执行完整召回返回列表。"""
        uc = _build_use_case()

        # ── casual：跳过所有召回通道 ──
        casual_result, casual_gp = _run_async(
            uc.retrieve_memories("u1", "hello", stance="casual")
        )
        assert isinstance(casual_result, list)
        assert len(casual_result) == 0

        # ── engaged：执行完整三通道召回 ──
        engaged_result, engaged_gp = _run_async(
            uc.retrieve_memories("u1", "hello", stance="engaged")
        )
        assert isinstance(engaged_result, list)


# ===================================================================
# 测试 3：图路径返回
# ===================================================================


class TestGraphPaths:
    """场景 3：图数据存在/不存在时 retrieve_memories 的 graph_paths 返回值。"""

    def test_graph_paths_populated_when_data_exists(self) -> None:
        """当图数据存在时，retrieve_memories 应返回非空 graph_paths。"""
        graph_repo = MockGraphRepository(return_data=True)
        uc = _build_use_case(graph_repo=graph_repo)

        result, gp = _run_async(
            uc.retrieve_memories(
                "u1", "哲学", session_id="s1", stance="engaged"
            )
        )
        assert isinstance(result, list)
        assert len(gp) > 0, (
            "图数据存在时应产生图路径"
        )

    def test_graph_paths_empty_when_no_data(self) -> None:
        """当图数据不存在时，retrieve_memories 应返回空 graph_paths。"""
        graph_repo = MockGraphRepository(return_data=False)
        uc = _build_use_case(graph_repo=graph_repo)

        result, gp = _run_async(
            uc.retrieve_memories(
                "u1", "nothing", session_id="s2", stance="engaged"
            )
        )
        assert isinstance(result, list)
        assert len(gp) == 0, (
            "无图数据时应返回空图路径"
        )


# ===================================================================
# 测试 4：空输入不崩溃
# ===================================================================


class TestRetrieveMemoriesEmptyInput:
    """场景 4：空 user_input 应优雅处理不崩溃。"""

    def test_retrieve_memories_empty_input(self) -> None:
        """空字符串 user_input 不应引发异常，应返回空列表。"""
        uc = _build_use_case()

        # 空字符串
        result, gp = _run_async(
            uc.retrieve_memories(
                "u1", "", session_id="s1", stance="engaged"
            )
        )
        assert isinstance(result, list)
        assert len(result) == 0, (
            f"空输入应返回空列表，收到 {len(result)} 条"
        )

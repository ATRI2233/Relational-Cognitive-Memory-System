"""
test_event_handlers.py — Event Handler 集成测试
================================================

覆盖场景:
  1. EmbeddingDone / TurnSaved 事件的 occurred_at 字段正确性
  2. PostUpdateHandler 接收 TurnSaved 事件不崩溃
  3. GraphMaintenanceHandler 接收 MemoryDistilled 事件不崩溃
  4. EventBus 向多个 Handler 发布事件，全部触发
  5. 一个 Handler 抛出异常不影响其他 Handler

约定:
  - 每个测试函数独立 tmp_path，互不干扰
  - 使用 FrozenClock 固定时间
  - 手动组装依赖，不依赖 adapter/rcms_factory
  - 使用 asyncio.run() 执行协程方法
"""
from __future__ import annotations

import asyncio
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

# ── 将项目根目录加入 sys.path ──────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── 基础设施层 ──────────────────────────────────────────────────────
from infrastructure.clock import FrozenClock
from infrastructure.persistence.sqlite_memory_repo import SQLiteMemoryRepository
from infrastructure.persistence.sqlite_session_repo import SQLiteSessionRepository
from infrastructure.persistence.sqlite_identity_repo import SQLiteIdentityRepository
from infrastructure.persistence.sqlite_graph_repo import SQLiteGraphRepository

# ── 应用层 ──────────────────────────────────────────────────────────
from application.event_bus import EventBus
from application.post_update_service import PostUpdateService
from application.handlers.post_update_handler import PostUpdateHandler
from application.handlers.delayed_embed_handler import DelayedEmbedHandler
from application.handlers.graph_maintenance_handler import GraphMaintenanceHandler

# ── 领域事件 ────────────────────────────────────────────────────────
from domain.events.memory_events import TurnSaved, MemoryDistilled, EmbeddingDone

# ── 领域实体 ─────────────────────────────────────────────────────────
from domain.entities.memory import UserId


# ===================================================================
# DDL — 各 Handler 及其依赖 Repository 所需的数据表
# ===================================================================

_DDL_STATEMENTS = """
CREATE TABLE IF NOT EXISTS cognitive_distill (
    id INTEGER PRIMARY KEY,
    user_id TEXT,
    session_id TEXT,
    content TEXT NOT NULL,
    keylabel TEXT,
    summary TEXT DEFAULT '',
    mood TEXT DEFAULT '',
    mood_intensity REAL DEFAULT 0.0,
    importance REAL DEFAULT 0.3,
    entities TEXT DEFAULT '[]',
    embedding BLOB,
    turn_num INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    embedding_dim INTEGER DEFAULT NULL,
    expires_at TIMESTAMP DEFAULT NULL
);
CREATE INDEX IF NOT EXISTS idx_cd_user ON cognitive_distill(user_id, created_at);
CREATE INDEX IF NOT EXISTS idx_cd_embed ON cognitive_distill(user_id) WHERE embedding IS NOT NULL;

CREATE TABLE IF NOT EXISTS session_state (
    session_id TEXT PRIMARY KEY,
    user_id TEXT,
    stance TEXT DEFAULT 'open',
    mood REAL DEFAULT 0,
    turn_count INTEGER DEFAULT 0,
    stance_turns INTEGER DEFAULT 0,
    engagement_level TEXT DEFAULT 'coasting',
    momentum_depth REAL DEFAULT 0.0,
    momentum_energy REAL DEFAULT 0.0,
    last_active TIMESTAMP,
    dangling_threads TEXT DEFAULT '[]',
    embedding_updated INTEGER DEFAULT 0,
    last_distill_turn INTEGER DEFAULT 0,
    last_distill_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS identity_memory (
    user_id TEXT PRIMARY KEY,
    traits TEXT DEFAULT '[]',
    preferences TEXT DEFAULT '{}',
    self_identity TEXT DEFAULT '[]',
    boundaries TEXT DEFAULT '[]',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS memory_graph_nodes (
    node_id INTEGER PRIMARY KEY,
    user_id TEXT,
    label TEXT,
    node_type TEXT DEFAULT 'keyword',
    freq INTEGER DEFAULT 1,
    last_seen TIMESTAMP,
    entity_type TEXT DEFAULT 'auto'
);
CREATE INDEX IF NOT EXISTS idx_mgn_user_label ON memory_graph_nodes(user_id, label);

CREATE TABLE IF NOT EXISTS memory_graph_edges (
    from_node_id INTEGER,
    to_node_id INTEGER,
    weight REAL DEFAULT 1.0,
    encounter_count INTEGER DEFAULT 1,
    last_seen TIMESTAMP,
    relation TEXT DEFAULT '',
    created_at TEXT DEFAULT '',
    PRIMARY KEY (from_node_id, to_node_id)
);
CREATE INDEX IF NOT EXISTS idx_mge_from ON memory_graph_edges(from_node_id);
CREATE INDEX IF NOT EXISTS idx_mge_to ON memory_graph_edges(to_node_id);
"""


# ===================================================================
# 测试 Helper
# ===================================================================


def _init_db(db_path: str) -> sqlite3.Connection:
    """创建并初始化临时 SQLite 数据库，返回连接。"""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(_DDL_STATEMENTS)
    conn.commit()
    return conn


def _run_async(coro):
    """同步包装器：在同步测试函数中安全执行协程。"""
    return asyncio.run(coro)


# ===================================================================
# Mock 实现
# ===================================================================


class MockEmbedding:
    """Mock 向量嵌入服务 — 返回固定维度假向量。"""

    async def embed(self, text: str) -> list[float]:
        return [0.001] * 256


# ===================================================================
# Test 1：EmbeddingDone / TurnSaved 事件字段验证
# ===================================================================


class TestEventFields:
    """事件字段正确性测试"""

    def test_embedding_done_event_has_occurred_at(self) -> None:
        """EmbeddingDone 和 TurnSaved 事件应包含 occurred_at 字段，且为 datetime。"""
        # ── EmbeddingDone ──
        event = EmbeddingDone(
            user_id="test_user",
            record_id=1,
            embedding_dim=256,
            occurred_at=datetime.now(),
            source="test",
        )
        assert hasattr(event, "occurred_at"), (
            "EmbeddingDone 缺少 occurred_at 字段"
        )
        assert isinstance(event.occurred_at, datetime), (
            f"occurred_at 应为 datetime, got {type(event.occurred_at)}"
        )

        # ── TurnSaved ──
        event2 = TurnSaved(
            user_id="test_user",
            session_id="test_session",
            turn_number=1,
            user_input="hello",
            reply="hi",
            occurred_at=datetime.now(),
        )
        assert hasattr(event2, "occurred_at"), (
            "TurnSaved 缺少 occurred_at 字段"
        )
        assert isinstance(event2.occurred_at, datetime), (
            f"occurred_at 应为 datetime, got {type(event2.occurred_at)}"
        )


# ===================================================================
# Test 2：PostUpdateHandler 集成
# ===================================================================


class TestPostUpdateHandlerIntegration:
    """PostUpdateHandler 集成测试"""

    def test_post_update_handler_receives_turn_saved(self, tmp_path: Path) -> None:
        """PostUpdateHandler 应能处理 TurnSaved 事件不崩溃。"""
        clock = FrozenClock(datetime(2026, 6, 15, 10, 0, 0))
        conn = _init_db(str(tmp_path / "post_update.db"))

        try:
            # ── Repos ──
            memory_repo = SQLiteMemoryRepository(conn, clock)
            session_repo = SQLiteSessionRepository(conn, clock)
            identity_repo = SQLiteIdentityRepository(conn, clock)

            # ── PostUpdateService + Handler ──
            post_update_service = PostUpdateService(
                session_repo=session_repo,
                identity_repo=identity_repo,
                memory_repo=memory_repo,
                clock=clock,
            )
            handler = PostUpdateHandler(post_update_service)

            bus = EventBus()
            handler.register(bus, handler)

            # ── 发布 TurnSaved 事件 ──
            _run_async(bus.publish(TurnSaved(
                occurred_at=clock.now(),
                user_id="test_user",
                session_id="test_session",
                turn_number=1,
                user_input="你好",
                reply="你好！",
            )))

            # ── 验证 PostUpdateHandler 确实执行了 ──
            # _ensure_identity 应创建 identity_memory 行
            id_row = conn.execute(
                "SELECT user_id FROM identity_memory WHERE user_id = ?",
                ("test_user",),
            ).fetchone()
            assert id_row is not None, (
                "PostUpdateHandler 未执行：identity_memory 行未被创建"
            )

        finally:
            conn.close()


# ===================================================================
# Test 3：GraphMaintenanceHandler 集成
# ===================================================================


class TestGraphMaintenanceHandlerIntegration:
    """GraphMaintenanceHandler 集成测试"""

    def test_graph_maintenance_handler_receives_memory_distilled(self, tmp_path: Path) -> None:
        """GraphMaintenanceHandler 应能处理 MemoryDistilled 事件不崩溃。"""
        clock = FrozenClock(datetime(2026, 6, 15, 10, 0, 0))
        conn = _init_db(str(tmp_path / "graph_maintenance.db"))

        try:
            graph_repo = SQLiteGraphRepository(conn, clock)
            handler = GraphMaintenanceHandler(graph_repo)

            bus = EventBus()
            handler.register(bus, handler)

            # ── 发布 MemoryDistilled 事件 ──
            # 空数据库上 maintain 应为安全无操作，不应崩溃
            _run_async(bus.publish(MemoryDistilled(
                occurred_at=clock.now(),
                user_id="test_user",
                session_id="test_session",
                memory_id=1,
                content="测试记忆内容",
                keylabel="test",
                importance=0.8,
                mood="平静",
                mood_intensity=0.5,
            )))

            # 额外验证：数据库在此之后仍可用
            row = conn.execute("SELECT COUNT(*) AS cnt FROM memory_graph_nodes").fetchone()
            assert row is not None
            assert row["cnt"] == 0

        finally:
            conn.close()


# ===================================================================
# Test 4：EventBus 多 Handler 发布
# ===================================================================


class TestEventBusMultiHandler:
    """EventBus 多 Handler 集成测试"""

    def test_event_bus_publishes_to_all_handlers(self, tmp_path: Path) -> None:
        """注册多个 Handler 后 publish 应全部触发。"""
        clock = FrozenClock(datetime(2026, 6, 15, 10, 0, 0))
        conn = _init_db(str(tmp_path / "multi_handler.db"))

        try:
            # ── Repos ──
            memory_repo = SQLiteMemoryRepository(conn, clock)
            session_repo = SQLiteSessionRepository(conn, clock)
            identity_repo = SQLiteIdentityRepository(conn, clock)

            # 预创建一条未向量化的记忆，供 DelayedEmbedHandler 处理
            conn.execute(
                "INSERT INTO cognitive_distill "
                "(user_id, session_id, content, keylabel, importance) "
                "VALUES (?, ?, ?, ?, ?)",
                ("test_user", "test_session",
                 "需要向量化的测试记忆内容", "test_embed", 0.5),
            )
            conn.commit()

            bus = EventBus()

            # ── Handler 1：PostUpdateHandler（真实） ──
            post_svc = PostUpdateService(
                session_repo=session_repo,
                identity_repo=identity_repo,
                memory_repo=memory_repo,
                clock=clock,
            )
            handler_post = PostUpdateHandler(post_svc)
            handler_post.register(bus, handler_post)

            # ── Handler 2：DelayedEmbedHandler（真实，MockEmbedding） ──
            mock_embed = MockEmbedding()
            handler_embed = DelayedEmbedHandler(
                memory_repo=memory_repo,
                embedding=mock_embed,
                event_bus=bus,
                min_text_length=0,  # 绕过长度检查确保触发
            )
            handler_embed.register(bus, handler_embed)

            # ── Handler 3：跟踪回调 ──
            callback_hits: list[str] = []

            async def tracking_handler(event: TurnSaved) -> None:
                callback_hits.append(
                    f"收到 {event.user_id} 轮次 {event.turn_number}"
                )

            bus.register(TurnSaved, tracking_handler)

            # ── 发布 TurnSaved ──
            _run_async(bus.publish(TurnSaved(
                occurred_at=clock.now(),
                user_id="test_user",
                session_id="test_session",
                turn_number=1,
                user_input="今天心情不太好，想找人聊聊天，最近工作压力很大",
                reply="我理解你的感受，工作压力确实需要适当释放。",
            )))

            # ── 验证全部三个 Handler 都被调用 ──

            # Handler 1：identity 已创建
            id_row = conn.execute(
                "SELECT user_id FROM identity_memory WHERE user_id = ?",
                ("test_user",),
            ).fetchone()
            assert id_row is not None, "PostUpdateHandler 未被调用"

            # Handler 2：embedding 已生成
            # 使用 get_unembedded() 验证 — 与 DelayedEmbedHandler 内部使用相同方法
            unembedded = _run_async(memory_repo.get_unembedded(UserId("test_user")))
            assert len(unembedded) == 0, (
                f"DelayedEmbedHandler 应处理所有未嵌入记录，仍有 {len(unembedded)} 条未嵌入"
            )

            # Handler 3：回调已执行
            assert len(callback_hits) == 1, (
                f"tracking_handler 未被调用, hits={callback_hits}"
            )
            assert "test_user" in callback_hits[0]

        finally:
            conn.close()


# ===================================================================
# Test 5：EventBus Handler 异常隔离
# ===================================================================


class TestEventBusErrorIsolation:
    """EventBus Handler 异常隔离测试"""

    def test_event_bus_handler_error_isolation(self, tmp_path: Path) -> None:
        """一个 Handler 抛出异常不应阻止其他 Handler 执行。"""
        clock = FrozenClock(datetime(2026, 6, 15, 10, 0, 0))
        conn = _init_db(str(tmp_path / "error_isolation.db"))

        try:
            memory_repo = SQLiteMemoryRepository(conn, clock)
            session_repo = SQLiteSessionRepository(conn, clock)
            identity_repo = SQLiteIdentityRepository(conn, clock)

            bus = EventBus()

            # ── Handler A：抛出异常 ──
            async def throwing_handler(event: TurnSaved) -> None:
                raise RuntimeError("handler_a 预期错误")

            # ── Handler B：PostUpdateHandler（正常执行） ──
            post_svc = PostUpdateService(
                session_repo=session_repo,
                identity_repo=identity_repo,
                memory_repo=memory_repo,
                clock=clock,
            )
            handler_normal = PostUpdateHandler(post_svc)

            bus.register(TurnSaved, throwing_handler)
            handler_normal.register(bus, handler_normal)

            # ── 发布 — publish 本身不应抛出异常 ──
            # EventBus._safe_dispatch 会捕获 handler 内的异常
            _run_async(bus.publish(TurnSaved(
                occurred_at=clock.now(),
                user_id="test_user",
                session_id="test_session",
                turn_number=1,
                user_input="你好",
                reply="你好！",
            )))

            # ── 验证正常 Handler 仍然执行了 ──
            # _ensure_identity 应创建 identity_memory 行
            id_row = conn.execute(
                "SELECT user_id FROM identity_memory WHERE user_id = ?",
                ("test_user",),
            ).fetchone()
            assert id_row is not None, (
                "异常 handler 不应阻止 PostUpdateHandler 执行"
            )

        finally:
            conn.close()

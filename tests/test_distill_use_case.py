"""
test_distill_use_case.py — DistillUseCase 单元/集成测试
=======================================================

覆盖场景：
  1. Memory 对象 created_at 字段验证（回归测试：4 个 Memory() 调用缺少 created_at）
  2. check_and_run 方法签名验证（turn_count 参数）
  3. 空输入边界处理

约定：
  - 继承 test_integration.py 的测试模式
  - 使用 FrozenClock 固定时间
  - 使用 on-disk SQLite（tmp_path）
  - 使用 asyncio.run() 执行协程
  - 手动组装依赖，不依赖 adapter/rcms_factory
"""
from __future__ import annotations

import asyncio
import json
import sqlite3
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
from infrastructure.persistence.sqlite_memory_repo import SQLiteMemoryRepository
from infrastructure.persistence.sqlite_session_repo import SQLiteSessionRepository
from infrastructure.persistence.sqlite_identity_repo import SQLiteIdentityRepository
from infrastructure.persistence.sqlite_graph_repo import SQLiteGraphRepository
from infrastructure.config.settings import Settings

# ── 应用层 ──────────────────────────────────────────────────────────
from application.event_bus import EventBus
from application.use_cases.distill_use_case import (
    DistillUseCase,
    DistillResult,
    IAnalysisSettings,
    IMaintenanceSettings,
    DefaultDistillPromptBuilder,
)

# ── 领域实体 ────────────────────────────────────────────────────────
from domain.entities.memory import (
    Importance,
    Memory,
    MemoryId,
    Mood,
    SessionId,
    UserId,
)
from domain.entities.session import Session, TurnRecord

# ── 领域事件 ────────────────────────────────────────────────────────
from domain.events.memory_events import MemoryDistilled


# ===================================================================
# DDL — 与 test_integration.py _DDL_STATEMENTS 保持一致
# ===================================================================

_DDL_STATEMENTS = """
CREATE TABLE IF NOT EXISTS analysis_raw (
    id INTEGER PRIMARY KEY,
    user_id TEXT,
    session_id TEXT,
    content TEXT,
    parsed INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS chat_history (
    id INTEGER PRIMARY KEY,
    session_id TEXT,
    role TEXT,
    content TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    turn_num INTEGER DEFAULT 0,
    importance REAL DEFAULT 0.3,
    mood TEXT DEFAULT '',
    user_id TEXT DEFAULT '',
    sender_name TEXT DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_ch_session ON chat_history(session_id);

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
CREATE INDEX IF NOT EXISTS idx_cd_user_imp ON cognitive_distill(user_id, importance DESC, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_cd_mood ON cognitive_distill(user_id, mood) WHERE mood IS NOT NULL;

CREATE TABLE IF NOT EXISTS embedding_rebuild_queue (
    id INTEGER PRIMARY KEY,
    user_id TEXT NOT NULL,
    record_id INTEGER NOT NULL,
    reason TEXT DEFAULT '',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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

CREATE TABLE IF NOT EXISTS memory_links (
    id INTEGER PRIMARY KEY,
    user_id TEXT NOT NULL,
    from_memory_id INTEGER NOT NULL,
    to_memory_id INTEGER NOT NULL,
    link_type TEXT DEFAULT 'related',
    reason TEXT DEFAULT '',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(from_memory_id, to_memory_id)
);

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

CREATE TABLE IF NOT EXISTS shared_context (
    context_id INTEGER PRIMARY KEY,
    user_id TEXT,
    context_body TEXT,
    omission_count INTEGER DEFAULT 0,
    confirmed INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_sc_user ON shared_context(user_id);

CREATE TABLE IF NOT EXISTS user_mappings (
    id INTEGER PRIMARY KEY,
    session_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    label TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'nickname',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(session_id, user_id, label)
);

CREATE INDEX IF NOT EXISTS idx_um_session ON user_mappings(session_id);
CREATE INDEX IF NOT EXISTS idx_um_label ON user_mappings(session_id, label);
"""


# ===================================================================
# 测试 Helper
# ===================================================================


def _run_async(coro):
    """同步包装器：在同步测试函数中安全执行协程。"""
    return asyncio.run(coro)


def _init_db(db_path: str) -> sqlite3.Connection:
    """创建并初始化临时 SQLite 数据库，返回连接。"""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(_DDL_STATEMENTS)
    conn.commit()
    return conn


# ===================================================================
# Mock 实现
# ===================================================================


class MockLLM:
    """Mock LLM 服务 — 实现 ILLMService protocol。"""

    def __init__(self, response: str = "这是一个模拟回复。"):
        self._response = response
        self.last_prompt: str = ""

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        self.last_prompt = prompt
        return self._response


class MockEmbedding:
    """Mock 向量嵌入服务 — 返回固定维度假向量。"""

    async def embed(self, text: str) -> list[float]:
        return [0.001] * 256


@dataclass
class MockMaintenanceSettings:
    """实现 IMaintenanceSettings protocol。"""
    keep_rule_summary: int = 100
    max_trait_count: int = 30


@dataclass
class MockAnalysisSettings:
    """实现 IAnalysisSettings protocol 的测试配置。

    同时包含协议未声明但运行时需要的额外属性，
    如 dangling_fallback_importance（_save_dangling_threads 中使用）。
    """
    personality_type: str = "default"
    max_turns: int = 5
    max_minutes: int = 60
    distill_min_turns: int = 1
    max_snapshot_lines: int = 50
    distill_entry_importance: float = 0.5
    archived_dangling_importance: float = 0.3
    dangling_fallback_importance: float = 0.3
    keep_rule_summary: int = 100
    maintenance: MockMaintenanceSettings = field(default_factory=MockMaintenanceSettings)
    permanent_fact_max: int = 3
    transient_fact_max: int = 5


# ===================================================================
# 测试 1：Memory 对象 created_at 字段验证
# ===================================================================


class TestMemoryCreatedAt:
    """场景 1：Memory 对象 created_at 字段回归测试。

    验证 _write_distill_summary 中构造 Memory 时传入 created_at，
    确保 SQLite 中的 created_at 与 FrozenClock 一致。
    """

    def test_memory_construction_with_created_at(self, tmp_path: Path) -> None:
        """Memory 被持久化后，cognitive_distill.created_at 应与 clock.now() 一致。"""
        frozen_time = datetime(2026, 6, 15, 10, 0, 0)
        clock = FrozenClock(frozen_time)
        conn = _init_db(str(tmp_path / "distill_created_at.db"))

        try:
            # ── Repository ──
            session_repo = SQLiteSessionRepository(conn, clock)
            memory_repo = SQLiteMemoryRepository(conn, clock)
            identity_repo = SQLiteIdentityRepository(conn, clock)
            graph_repo = SQLiteGraphRepository(conn, clock)

            # ── EventBus ──
            bus = EventBus()

            # ── Settings ──
            settings = MockAnalysisSettings()

            # ── 创建 Session（turn_count 足够大以触发蒸馏）──
            session = Session(
                session_id=SessionId("s1"),
                user_id=UserId("u1"),
                turn_count=10,
                last_active=clock.now(),
            )
            _run_async(session_repo.save(session))

            # ── chat_history_provider：返回足够多的 TurnRecord ──
            async def chat_history_provider(
                sid: SessionId, last_turn: int, turn_count: int,
            ) -> list[TurnRecord]:
                return [
                    TurnRecord(
                        session_id=SessionId("s1"),
                        role="user",
                        content=f"这是第 {i} 轮对话",
                        turn_num=i,
                        user_id="u1",
                        sender_name="小明",
                    )
                    for i in range(1, 9)  # 8 条记录 > distill_min_turns=1
                ]

            # ── Mock LLM：返回合法 JSON ──
            llm_response = json.dumps({
                "content": "用户在此轮对话中表现出对哲学的浓厚兴趣",
                "keylabel": "哲学兴趣",
                "summary": "用户对哲学表现出浓厚兴趣",
                "analysis": {
                    "mood": "curious",
                    "mood_intensity": 0.6,
                },
            }, ensure_ascii=False)
            llm = MockLLM(response=llm_response)

            # ── Mock Embedding ──
            embedding = MockEmbedding()

            # ── Prompt Builder ──
            prompt_builder = DefaultDistillPromptBuilder()

            # ── DistillUseCase ──
            # 以下可选参数使用默认值，验证默认路径的正确性：
            #   persona_name="Bot"          — 群聊检测中排除自身
            #   analysis_writer=None         — 9 维分析走内部实现
            #   save_shared_context_fn=None  — 共享梗/上下文跳过
            #   trim_low_importance_fn=None  — 低重要性条目不做截断
            #   save_raw_analysis_fn=None    — 不写入 analysis_raw 审计日志
            #   mark_raw_parsed_fn=None      — 不标记原始分析为已解析
            distill_uc = DistillUseCase(
                session_repo=session_repo,
                memory_repo=memory_repo,
                identity_repo=identity_repo,
                graph_repo=graph_repo,
                llm=llm,
                embedding=embedding,
                clock=clock,
                event_bus=bus,
                settings=settings,
                prompt_builder=prompt_builder,
                chat_history_provider=chat_history_provider,
            )

            # ── 执行 ──
            result = _run_async(distill_uc.check_and_run(
                user_id="u1",
                session_id="s1",
            ))

            # ── 验证返回值 ──
            assert result.triggered, (
                f"蒸馏应该被触发，但 triggered={result.triggered}"
            )
            assert result.memory_id > 0, f"memory_id 应 > 0，收到 {result.memory_id}"
            assert result.keylabel != "", "keylabel 不应为空"

            # ── 验证 created_at 字段 ──
            rows = conn.execute(
                "SELECT id, created_at, content, keylabel FROM cognitive_distill WHERE id = ?",
                (result.memory_id,),
            ).fetchall()
            assert len(rows) == 1, f"应找到 1 条蒸馏记录，但找到 {len(rows)} 条"
            row = rows[0]
            raw_created = row["created_at"]
            assert raw_created is not None, "created_at 不应为 NULL"

            # SQLite TIMESTAMP → datetime 转换后比较
            if isinstance(raw_created, str):
                created_dt = datetime.fromisoformat(raw_created)
            else:
                created_dt = raw_created

            assert created_dt == frozen_time, (
                f"created_at ({created_dt}) 应与 clock.now() ({frozen_time}) 一致"
            )

            # ── 额外验证：cognitive_distill 中的关键字段 ──
            assert row["content"] == "用户在此轮对话中表现出对哲学的浓厚兴趣"
            assert row["keylabel"] == "哲学兴趣"

        finally:
            conn.close()


# ===================================================================
# 测试 2：check_and_run 返回 DistillResult
# ===================================================================


class TestDistillResultReturned:
    """场景 2：check_and_run 方法签名与返回值验证。

    即使蒸馏未实际执行（条件不满足），也应返回 DistillResult。
    """

    def test_distill_result_returned(self, tmp_path: Path) -> None:
        """check_and_run 应返回 DistillResult，不抛异常。"""
        clock = FrozenClock(datetime(2026, 6, 15, 10, 0, 0))
        conn = _init_db(str(tmp_path / "distill_result.db"))

        try:
            # ── Repository ──
            session_repo = SQLiteSessionRepository(conn, clock)
            memory_repo = SQLiteMemoryRepository(conn, clock)
            identity_repo = SQLiteIdentityRepository(conn, clock)
            graph_repo = SQLiteGraphRepository(conn, clock)

            bus = EventBus()
            settings = MockAnalysisSettings()

            llm = MockLLM()
            embedding = MockEmbedding()
            prompt_builder = DefaultDistillPromptBuilder()

            distill_uc = DistillUseCase(
                session_repo=session_repo,
                memory_repo=memory_repo,
                identity_repo=identity_repo,
                graph_repo=graph_repo,
                llm=llm,
                embedding=embedding,
                clock=clock,
                event_bus=bus,
                settings=settings,
                prompt_builder=prompt_builder,
            )

            # 未创建任何 session → _check_distill_needed 返回 None
            result = _run_async(distill_uc.check_and_run(
                user_id="u1",
                session_id="nonexistent_session",
            ))

            assert isinstance(result, DistillResult), (
                f"返回值应为 DistillResult，收到 {type(result)}"
            )
            assert result.triggered is False, (
                f"无 session 时应返回 triggered=False，收到 {result.triggered}"
            )
            assert result.memory_id == 0
            assert result.keylabel == ""

        finally:
            conn.close()


# ===================================================================
# 测试 3：空输入边界处理
# ===================================================================


class TestInvalidInputs:
    """场景 3：调用 check_and_run 时空输入不应导致未处理的异常。

    - 空 user_id + 有效 session_id → 应返回 DistillResult（无 session 时 triggered=False）
    - 空 session_id → SessionId 值对象会抛出 ValueError（领域层验证）
    """

    def test_empty_user_id_handled_gracefully(self, tmp_path: Path) -> None:
        """空 user_id 且 session 不存在时应返回 DistillResult。"""
        clock = FrozenClock(datetime(2026, 6, 15, 10, 0, 0))
        conn = _init_db(str(tmp_path / "distill_empty_user.db"))

        try:
            session_repo = SQLiteSessionRepository(conn, clock)
            memory_repo = SQLiteMemoryRepository(conn, clock)
            identity_repo = SQLiteIdentityRepository(conn, clock)
            graph_repo = SQLiteGraphRepository(conn, clock)

            bus = EventBus()
            settings = MockAnalysisSettings()

            llm = MockLLM()
            embedding = MockEmbedding()
            prompt_builder = DefaultDistillPromptBuilder()

            distill_uc = DistillUseCase(
                session_repo=session_repo,
                memory_repo=memory_repo,
                identity_repo=identity_repo,
                graph_repo=graph_repo,
                llm=llm,
                embedding=embedding,
                clock=clock,
                event_bus=bus,
                settings=settings,
                prompt_builder=prompt_builder,
            )

            # user_id 为空字符串，session_id 指向不存在的 session
            # → 应返回 DistillResult(triggered=False)，不抛异常
            result = _run_async(distill_uc.check_and_run(
                user_id="",
                session_id="test_session",
            ))
            assert isinstance(result, DistillResult)
            assert result.triggered is False

        finally:
            conn.close()

    def test_empty_session_id_raises_value_error(self, tmp_path: Path) -> None:
        """空 session_id 应抛出 ValueError（SessionId 值对象验证）。"""
        clock = FrozenClock(datetime(2026, 6, 15, 10, 0, 0))
        conn = _init_db(str(tmp_path / "distill_empty_session.db"))

        try:
            session_repo = SQLiteSessionRepository(conn, clock)
            memory_repo = SQLiteMemoryRepository(conn, clock)
            identity_repo = SQLiteIdentityRepository(conn, clock)
            graph_repo = SQLiteGraphRepository(conn, clock)

            bus = EventBus()
            settings = MockAnalysisSettings()

            llm = MockLLM()
            embedding = MockEmbedding()
            prompt_builder = DefaultDistillPromptBuilder()

            distill_uc = DistillUseCase(
                session_repo=session_repo,
                memory_repo=memory_repo,
                identity_repo=identity_repo,
                graph_repo=graph_repo,
                llm=llm,
                embedding=embedding,
                clock=clock,
                event_bus=bus,
                settings=settings,
                prompt_builder=prompt_builder,
            )

            # SessionId("") 在 __post_init__ 中抛出 ValueError
            with pytest.raises(ValueError, match="SessionId"):
                _run_async(distill_uc.check_and_run(
                    user_id="test_user",
                    session_id="",
                ))

        finally:
            conn.close()

    def test_both_empty_raises_value_error(self, tmp_path: Path) -> None:
        """user_id 和 session_id 皆空时应抛出 ValueError（SessionId 验证优先）。"""
        clock = FrozenClock(datetime(2026, 6, 15, 10, 0, 0))
        conn = _init_db(str(tmp_path / "distill_both_empty.db"))

        try:
            session_repo = SQLiteSessionRepository(conn, clock)
            memory_repo = SQLiteMemoryRepository(conn, clock)
            identity_repo = SQLiteIdentityRepository(conn, clock)
            graph_repo = SQLiteGraphRepository(conn, clock)

            bus = EventBus()
            settings = MockAnalysisSettings()

            llm = MockLLM()
            embedding = MockEmbedding()
            prompt_builder = DefaultDistillPromptBuilder()

            distill_uc = DistillUseCase(
                session_repo=session_repo,
                memory_repo=memory_repo,
                identity_repo=identity_repo,
                graph_repo=graph_repo,
                llm=llm,
                embedding=embedding,
                clock=clock,
                event_bus=bus,
                settings=settings,
                prompt_builder=prompt_builder,
            )

            with pytest.raises(ValueError, match="SessionId"):
                _run_async(distill_uc.check_and_run(
                    user_id="",
                    session_id="",
                ))

        finally:
            conn.close()


# ===================================================================
# 测试 4：LLM 错误响应处理
# ===================================================================


class TestLLMErrorHandling:
    """场景 4：LLM 返回异常响应时的降级处理。

    验证 _call_llm 和 _parse_llm_response 的异常路径：
      - 空响应 → check_and_run 返回 triggered=False
      - 非 JSON 响应 → _parse_llm_response 返回 None → triggered=False
      - JSON 缺少 content 字段 → _parse_llm_response 返回 None → triggered=False
    """

    @staticmethod
    def _build_use_case(
        tmp_path: Path, llm: MockLLM,
    ) -> tuple[sqlite3.Connection, DistillUseCase]:
        """组装带指定 MockLLM 的 DistillUseCase。"""
        clock = FrozenClock(datetime(2026, 6, 15, 10, 0, 0))
        conn = _init_db(str(tmp_path / "llm_error.db"))

        session_repo = SQLiteSessionRepository(conn, clock)
        memory_repo = SQLiteMemoryRepository(conn, clock)
        identity_repo = SQLiteIdentityRepository(conn, clock)
        graph_repo = SQLiteGraphRepository(conn, clock)

        bus = EventBus()
        settings = MockAnalysisSettings()
        embedding = MockEmbedding()
        prompt_builder = DefaultDistillPromptBuilder()

        # 预创建 Session，turn_count 足够大以触发蒸馏检查
        session = Session(
            session_id=SessionId("s1"),
            user_id=UserId("u1"),
            turn_count=10,
            last_active=clock.now(),
        )
        _run_async(session_repo.save(session))

        # chat_history_provider 返回足够记录
        async def chat_history_provider(
            sid: SessionId, last_turn: int, turn_count: int,
        ) -> list[TurnRecord]:
            return [
                TurnRecord(
                    session_id=SessionId("s1"),
                    role="user",
                    content=f"第 {i} 轮",
                    turn_num=i,
                    user_id="u1",
                    sender_name="小明",
                )
                for i in range(1, 9)
            ]

        distill_uc = DistillUseCase(
            session_repo=session_repo,
            memory_repo=memory_repo,
            identity_repo=identity_repo,
            graph_repo=graph_repo,
            llm=llm,
            embedding=embedding,
            clock=clock,
            event_bus=bus,
            settings=settings,
            prompt_builder=prompt_builder,
            chat_history_provider=chat_history_provider,
        )
        return conn, distill_uc

    def test_empty_llm_response(self, tmp_path: Path) -> None:
        """LLM 返回空字符串时应降级返回 triggered=False。"""
        conn, distill_uc = self._build_use_case(tmp_path, MockLLM(response=""))
        try:
            result = _run_async(distill_uc.check_and_run(
                user_id="u1", session_id="s1",
            ))
            assert isinstance(result, DistillResult)
            assert result.triggered is False, (
                "空 LLM 响应应触发降级"
            )
            assert result.memory_id == 0
            assert result.keylabel == ""
        finally:
            conn.close()

    def test_malformed_json_response(self, tmp_path: Path) -> None:
        """LLM 返回非 JSON 文本时应降级返回 triggered=False。"""
        conn, distill_uc = self._build_use_case(
            tmp_path, MockLLM(response="这不是 JSON 数据，我是纯文本回复。"),
        )
        try:
            result = _run_async(distill_uc.check_and_run(
                user_id="u1", session_id="s1",
            ))
            assert isinstance(result, DistillResult)
            assert result.triggered is False, (
                "非 JSON 响应应触发降级"
            )
            assert result.memory_id == 0
        finally:
            conn.close()

    def test_json_missing_content_field(self, tmp_path: Path) -> None:
        """LLM 返回合法 JSON 但缺少 content 字段时应降级。"""
        conn, distill_uc = self._build_use_case(
            tmp_path,
            MockLLM(response='{"keylabel": "test", "summary": "test summary"}'),
        )
        try:
            result = _run_async(distill_uc.check_and_run(
                user_id="u1", session_id="s1",
            ))
            assert isinstance(result, DistillResult)
            assert result.triggered is False, (
                "缺少 content 的 JSON 应触发降级"
            )
            assert result.memory_id == 0
            assert result.keylabel == ""
        finally:
            conn.close()

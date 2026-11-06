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
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import pytest

# ── 将项目根目录加入 sys.path ──────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── 基础设施层 ──────────────────────────────────────────────────────
from infrastructure.clock import FrozenClock
from infrastructure.config.settings import StorageSettings
from infrastructure.persistence.ddl import ensure_schema
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
# Schema 直接复用生产 DDL（infrastructure.persistence.ddl），
# 避免测试里再维护一份副本 —— 之前正是副本导致废弃表删漏。
# ===================================================================

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
    ensure_schema(conn, StorageSettings())
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


# ===================================================================
# 测试 5：时间型蒸馏条件（回归）
#
# 旧实现写的是 `last_at is not None` 才计时，而新会话的 last_distill_at
# 一直是 NULL —— 结果「攒够 60 分钟也蒸一次」这条规则在第一次蒸馏之前
# 永远不成立，短会话永远产不出记忆。这里的用例锁住修复后的行为。
# ===================================================================


class MutableClock:
    """可推进的测试时钟（FrozenClock 冻结后无法前进）。"""

    def __init__(self, dt: datetime) -> None:
        self._dt = dt

    def now(self) -> datetime:
        return self._dt

    def strftime(self, fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
        return self._dt.strftime(fmt)

    def advance(self, minutes: float) -> None:
        self._dt += timedelta(minutes=minutes)


class TestTimeBasedDistillGate:
    """校验「距上次蒸馏超过 max_minutes」这条条件。"""

    NOW = datetime(2026, 6, 15, 12, 0, 0)

    @classmethod
    def _build(
        cls,
        tmp_path: Path,
        first_seen_minutes_ago: float,
        turn_count: int = 6,
        last_distill_minutes_ago: float | None = None,
        llm: MockLLM | None = None,
    ):
        clock = MutableClock(cls.NOW)
        conn = _init_db(str(tmp_path / "gate.db"))
        session_repo = SQLiteSessionRepository(conn, clock)
        memory_repo = SQLiteMemoryRepository(conn, clock)
        identity_repo = SQLiteIdentityRepository(conn, clock)
        graph_repo = SQLiteGraphRepository(conn, clock)

        settings = MockAnalysisSettings()
        settings.max_turns = 30          # 让轮次条件不参与
        settings.max_minutes = 60

        session = Session(
            session_id=SessionId("s1"),
            user_id=UserId("u1"),
            turn_count=turn_count,
            last_active=clock.now(),
            first_seen_at=clock.now() - timedelta(minutes=first_seen_minutes_ago),
            last_distill_at=(
                clock.now() - timedelta(minutes=last_distill_minutes_ago)
                if last_distill_minutes_ago is not None else None
            ),
        )
        _run_async(session_repo.save(session))

        async def chat_history_provider(
            sid: SessionId, last_turn: int, turn_count: int,
        ) -> list[TurnRecord]:
            return [
                TurnRecord(
                    session_id=SessionId("s1"), role="user" if i % 2 else "assistant",
                    content=f"第 {i} 条", turn_num=i, user_id="u1", sender_name="小明",
                )
                for i in range(1, 9)
            ]

        uc = DistillUseCase(
            session_repo=session_repo,
            memory_repo=memory_repo,
            identity_repo=identity_repo,
            graph_repo=graph_repo,
            llm=llm or MockLLM(response=""),
            embedding=MockEmbedding(),
            clock=clock,
            event_bus=EventBus(),
            settings=settings,
            prompt_builder=DefaultDistillPromptBuilder(),
            chat_history_provider=chat_history_provider,
        )
        return conn, uc, clock, session_repo

    def test_never_distilled_session_uses_first_seen(self, tmp_path: Path) -> None:
        """从没蒸过、但会话已经开了 3 小时 → 时间条件成立。"""
        conn, uc, _clock, _repo = self._build(tmp_path, first_seen_minutes_ago=180)
        try:
            needed = _run_async(uc._check_distill_needed("s1"))
            assert needed is not None, "从未蒸馏过的会话应能按 first_seen_at 触发"
            last_turn, turn_count, snapshot, _senders = needed
            assert last_turn == 0 and turn_count == 6
            assert snapshot
        finally:
            conn.close()

    def test_fresh_session_does_not_trigger(self, tmp_path: Path) -> None:
        """刚开 10 分钟的会话两条条件都不满足 → 不触发。"""
        conn, uc, _clock, _repo = self._build(tmp_path, first_seen_minutes_ago=10)
        try:
            assert _run_async(uc._check_distill_needed("s1")) is None
        finally:
            conn.close()

    def test_last_distill_at_takes_precedence(self, tmp_path: Path) -> None:
        """蒸过一次之后，起点换成 last_distill_at，而不是会话起点。"""
        conn, uc, _clock, _repo = self._build(
            tmp_path, first_seen_minutes_ago=600, last_distill_minutes_ago=5,
        )
        try:
            assert _run_async(uc._check_distill_needed("s1")) is None, (
                "5 分钟前刚蒸过，不该因为会话开得久而重复蒸馏"
            )
        finally:
            conn.close()

    def test_failed_attempt_backs_off(self, tmp_path: Path) -> None:
        """蒸馏失败不写 last_distill_at，但不能每条消息都重试 LLM。"""
        conn, uc, clock, _repo = self._build(
            tmp_path, first_seen_minutes_ago=180, llm=MockLLM(response=""),
        )
        try:
            first = _run_async(uc.check_and_run(user_id="u1", session_id="s1"))
            assert first.triggered is False          # LLM 空响应 → 未产出
            assert _run_async(uc._check_distill_needed("s1")) is None

            clock.advance(61)                        # 冷却结束
            assert _run_async(uc._check_distill_needed("s1")) is not None
        finally:
            conn.close()

    def test_first_seen_at_roundtrips(self, tmp_path: Path) -> None:
        """仓储必须原样存取 first_seen_at，否则起点会每次落库丢一次。"""
        conn, _uc, _clock, repo = self._build(tmp_path, first_seen_minutes_ago=120)
        try:
            session = _run_async(repo.get(SessionId("s1")))
            assert session is not None
            assert session.first_seen_at is not None
            assert abs(
                (session.first_seen_at - (self.NOW - timedelta(minutes=120)))
                .total_seconds()
            ) < 2
        finally:
            conn.close()

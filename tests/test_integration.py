"""
test_integration.py — 新架构端到端集成冒烟测试
=================================================

覆盖场景：
  1. 全链路初始化        — 创建所有 Repository 和 Use Case，验证成功
  2. ChatUseCase 完整流程 — 用 Mock LLM 执行 execute，验证返回 ChatResponse
  3. Save → Retrieve 循环 — 写入记忆后能通过 keyword search 找回
  4. Session 生命周期     — increment_turn → get → save 循环正常
  5. Identity 读写         — save_traits → get 循环正常
  6. Graph 操作            — upsert_node → search_nodes → maintain 循环正常
  7. EventBus 集成         — 注册 Handler → publish → Handler 被调用
  8. FusionService 融合    — 三通道输入 → 保底去重加权排序正常

约定：
  - 每个测试函数独立 tmp_path，互不干扰
  - 使用 MockBackend 而非真实 LLM
  - 使用 FrozenClock 固定时间
  - 手动组装依赖，不依赖 adapter/rcms_factory
  - 使用 asyncio.run() 执行协程方法
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
from unittest.mock import MagicMock

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
from infrastructure.config.settings import Settings, EmotionalWordsSettings

# ── 应用层 ──────────────────────────────────────────────────────────
from application.event_bus import EventBus
from application.use_cases.chat_use_case import (
    ChatUseCase,
    ChatRequest,
    ChatResponse,
    ILLMService,
    IRetrievalStrategy,
    IPromptBuilder,
    IDistillRunner,
    ISettings,
)

# ── 领域服务 ────────────────────────────────────────────────────────
from domain.services.fusion_service import FusionService, IFusionConfig, ChannelTag
from domain.services.time_service import TimeService
from domain.services.keyword_service import KeywordService

# ── 领域实体 ────────────────────────────────────────────────────────
from domain.entities.memory import (
    Importance,
    Memory,
    MemoryId,
    Mood,
    SessionId,
    UserId,
)
from domain.entities.identity import Boundary, Identity, Preferences, Trait
from domain.entities.session import Session
from domain.entities.graph import GraphNode, GraphEdge
from domain.events.memory_events import TurnSaved, MemoryDistilled


# ===================================================================
# DDL — 与 schema_snapshot.sql 保持一致
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


class MockRetrievalStrategy:
    """Mock 三通道召回策略 — 实现 IRetrievalStrategy protocol。"""

    def __init__(
        self,
        memories: list[tuple[str, str]] | None = None,
        graph_paths: list[str] | None = None,
    ):
        self._memories = memories or [("最近心情不好", "recent"), ("喜欢哲学", "resonance")]
        self._graph_paths = graph_paths or ["小明 --[喜欢]--> 哲学"]

    async def retrieve(
        self, user_id: str, user_input: str, session_id: str
    ) -> tuple[list[tuple[str, str]], list[str]]:
        return self._memories, self._graph_paths


class MockPromptBuilder:
    """Mock Prompt 构建器 — 实现 IPromptBuilder protocol。"""

    async def build(
        self,
        user_id: str,
        session_id: str,
        user_input: str,
        memories: list[tuple[str, str]],
        long_term: dict[str, Any],
        graph_paths: list[str],
    ) -> str:
        return f"[RCMS 上下文]\n用户说: {user_input}\n请根据上下文回复。"


class MockDistillRunner:
    """Mock 蒸馏检查执行器 — 实现 IDistillRunner protocol。"""

    def __init__(self):
        self.called_with: tuple[str, str, int] | None = None

    async def check_and_run(self, user_id: str, session_id: str, turn_count: int) -> None:
        self.called_with = (user_id, session_id, turn_count)


class MockEmbedding:
    """Mock 向量嵌入服务 — 返回固定维度假向量。"""

    async def embed(self, text: str) -> list[float]:
        return [0.001] * 256


@dataclass
class MockTokenFilter:
    """Mock 词表配置 — 匹配 IWordListConfig protocol。"""

    trivial_markers: list[str] = field(default_factory=lambda: ["吃", "喝", "睡", "饭"])
    stop_words: list[str] = field(default_factory=lambda: ["今天", "什么", "怎么", "可以"])
    time_words: dict[str, list[int]] = field(default_factory=lambda: {"今天": [0, 0], "最近": [0, 7]})


class MockTokenizer:
    """Mock 分词器 — 按空格和中文字符简单切分。"""

    def cut(self, text: str) -> list[str]:
        import re
        tokens: list[str] = []
        for t in re.split(r'[\s,，。！？、；：""''（）()—\n]+', text):
            if t:
                tokens.append(t)
        return tokens


# ===================================================================
# 测试 1：全链路初始化
# ===================================================================


class TestFullInit:
    """场景 1：创建所有 Repository 和 Use Case，验证成功。"""

    def test_all_repositories_instantiate(self, tmp_path: Path) -> None:
        """所有 Repository 和 Use Case 应能成功实例化。"""
        clock = FrozenClock(datetime(2026, 6, 15, 10, 0, 0))
        conn = _init_db(str(tmp_path / "full_init.db"))

        try:
            # ── Repository ──
            memory_repo = SQLiteMemoryRepository(conn, clock)
            session_repo = SQLiteSessionRepository(conn, clock)
            identity_repo = SQLiteIdentityRepository(conn, clock)
            graph_repo = SQLiteGraphRepository(conn, clock)

            assert memory_repo is not None
            assert session_repo is not None
            assert identity_repo is not None
            assert graph_repo is not None

            # ── EventBus ──
            bus = EventBus()
            assert bus.handler_count == 0

            # ── Settings ──
            settings = Settings()
            assert len(settings.emotional_words.emotional_words) > 0

            # ── FusionService ──
            fusion = FusionService(settings.retrieval)
            assert fusion is not None

            # ── TimeService ──
            time_svc = TimeService(clock)
            assert time_svc is not None

            # ── KeywordService ──
            kw_svc = KeywordService(MockTokenizer(), MockTokenFilter())
            assert kw_svc is not None

            # ── Mock 依赖 ──
            llm = MockLLM()
            retrieval = MockRetrievalStrategy()
            prompt_builder = MockPromptBuilder()
            distill = MockDistillRunner()

            # ── ChatUseCase ──
            chat_uc = ChatUseCase(
                memory_repo=memory_repo,
                session_repo=session_repo,
                identity_repo=identity_repo,
                graph_repo=graph_repo,
                llm=llm,
                clock=clock,
                event_bus=bus,
                settings=settings,
                retrieval_strategy=retrieval,
                prompt_builder=prompt_builder,
                distill_runner=distill,
            )
            assert chat_uc is not None

            # ── 数据库表验证 ──
            tables = {
                row[0] for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            expected = {
                "cognitive_distill", "session_state", "chat_history",
                "identity_memory", "memory_graph_nodes", "memory_graph_edges",
                "shared_context", "user_mappings", "embedding_rebuild_queue",
            }
            missing = expected - tables
            assert not missing, f"缺少表: {missing}"

        finally:
            conn.close()


# ===================================================================
# 测试 2：ChatUseCase 完整流程
# ===================================================================


class TestChatUseCase:
    """场景 2：用 Mock LLM 执行 ChatUseCase.execute，验证完整管线。"""

    def test_execute_returns_chat_response(self, tmp_path: Path) -> None:
        """ChatUseCase.execute 应返回 ChatResponse 并走通完整 11 步管线。"""
        clock = FrozenClock(datetime(2026, 6, 15, 10, 30, 0))
        conn = _init_db(str(tmp_path / "chat_flow.db"))

        try:
            # ── 组装依赖 ──
            memory_repo = SQLiteMemoryRepository(conn, clock)
            session_repo = SQLiteSessionRepository(conn, clock)
            identity_repo = SQLiteIdentityRepository(conn, clock)
            graph_repo = SQLiteGraphRepository(conn, clock)

            bus = EventBus()
            settings = Settings()

            llm = MockLLM(response="别太难过，我陪你聊聊。")
            retrieval = MockRetrievalStrategy(
                memories=[("他说最近工作很累", "recent"), ("喜欢哲学和编程", "resonance")],
                graph_paths=["小明 --[喜欢]--> 哲学"],
            )
            prompt_builder = MockPromptBuilder()
            distill = MockDistillRunner()

            chat_uc = ChatUseCase(
                memory_repo=memory_repo,
                session_repo=session_repo,
                identity_repo=identity_repo,
                graph_repo=graph_repo,
                llm=llm,
                clock=clock,
                event_bus=bus,
                settings=settings,
                retrieval_strategy=retrieval,
                prompt_builder=prompt_builder,
                distill_runner=distill,
            )

            # ── 执行 ──
            request = ChatRequest(
                user_id="test_user",
                session_id="test_session",
                user_input="今天心情不太好",
            )
            response = _run_async(chat_uc.execute(request))

            # ── 验证返回值 ──
            assert isinstance(response, ChatResponse)
            assert response.reply == "别太难过，我陪你聊聊。"
            assert response.turn_number == 1

            # ── 验证 LLM 确实被调用了 ──
            assert "心情不太好" in llm.last_prompt

            # ── 验证蒸馏检查改为由事件总线异步触发 ──
            # （不再同步调用 distill_runner.check_and_run）

            # ── 验证 chat_history 已写入 ──
            rows = conn.execute(
                "SELECT role, content FROM chat_history WHERE session_id = ? ORDER BY id",
                ("test_session",),
            ).fetchall()
            assert len(rows) == 2
            assert rows[0]["role"] == "user"
            assert rows[1]["role"] == "assistant"

            # ── 验证 session_state 已更新 ──
            # 每次 execute 中 increment_turn + save_turn 各至少递增一次，
            # 使用 >= 1 而非硬编码 2，避免改动步骤数时失效
            srow = conn.execute(
                "SELECT turn_count, last_active FROM session_state WHERE session_id = ?",
                ("test_session",),
            ).fetchone()
            assert srow is not None
            assert srow["turn_count"] >= 1

        finally:
            conn.close()

    def test_execute_validates_input(self, tmp_path: Path) -> None:
        """空输入时应抛出 ValueError。"""
        clock = FrozenClock(datetime(2026, 6, 15, 10, 30, 0))
        conn = _init_db(str(tmp_path / "chat_validate.db"))

        try:
            memory_repo = SQLiteMemoryRepository(conn, clock)
            session_repo = SQLiteSessionRepository(conn, clock)
            identity_repo = SQLiteIdentityRepository(conn, clock)
            graph_repo = SQLiteGraphRepository(conn, clock)

            chat_uc = ChatUseCase(
                memory_repo=memory_repo,
                session_repo=session_repo,
                identity_repo=identity_repo,
                graph_repo=graph_repo,
                llm=MockLLM(),
                clock=clock,
                event_bus=EventBus(),
                settings=Settings(),
                retrieval_strategy=MockRetrievalStrategy(),
                prompt_builder=MockPromptBuilder(),
                distill_runner=MockDistillRunner(),
            )

            with pytest.raises(ValueError, match="user_id 不能为空"):
                _run_async(chat_uc.execute(ChatRequest(
                    user_id="", session_id="s", user_input="hi",
                )))
            with pytest.raises(ValueError, match="session_id 不能为空"):
                _run_async(chat_uc.execute(ChatRequest(
                    user_id="u", session_id="", user_input="hi",
                )))
            with pytest.raises(ValueError, match="user_input 不能为空"):
                _run_async(chat_uc.execute(ChatRequest(
                    user_id="u", session_id="s", user_input="  ",
                )))
        finally:
            conn.close()

    def test_core_veto_replaces_didactic_phrases(self, tmp_path: Path) -> None:
        """_core_veto 应将说教用语替换为温和建议。"""
        clock = FrozenClock(datetime(2026, 6, 15, 10, 30, 0))
        conn = _init_db(str(tmp_path / "veto.db"))

        try:
            memory_repo = SQLiteMemoryRepository(conn, clock)
            session_repo = SQLiteSessionRepository(conn, clock)
            identity_repo = SQLiteIdentityRepository(conn, clock)
            graph_repo = SQLiteGraphRepository(conn, clock)

            chat_uc = ChatUseCase(
                memory_repo=memory_repo,
                session_repo=session_repo,
                identity_repo=identity_repo,
                graph_repo=graph_repo,
                llm=MockLLM(),
                clock=clock,
                event_bus=EventBus(),
                settings=Settings(),
                retrieval_strategy=MockRetrievalStrategy(),
                prompt_builder=MockPromptBuilder(),
                distill_runner=MockDistillRunner(),
            )

            result = chat_uc._core_veto("你应该多运动")
            assert "或许可以试试" in result

            result2 = chat_uc._core_veto("这样不对，你应该这样做")
            assert "或许可以试试" in result2

            result3 = chat_uc._core_veto("今天天气真好")
            assert result3 == "今天天气真好"  # 无说教用语，原样返回

        finally:
            conn.close()

    def test_calc_importance(self) -> None:
        """_calc_importance 应按情绪词和输入长度正确计算。

        该方法仅依赖 self._settings.emotional_words.emotional_words，
        无需真实的数据库仓库，使用 MagicMock 替代。"""
        chat_uc = ChatUseCase(
            memory_repo=MagicMock(),
            session_repo=MagicMock(),
            identity_repo=MagicMock(),
            graph_repo=MagicMock(),
            llm=MagicMock(),
            clock=FrozenClock(datetime(2026, 6, 15, 10, 30, 0)),
            event_bus=MagicMock(),
            settings=Settings(),
            retrieval_strategy=MagicMock(),
            prompt_builder=MagicMock(),
            distill_runner=MagicMock(),
        )

        # 无情绪词 → 0.3
        imp1 = chat_uc._calc_importance("今天天气真好")
        assert imp1 == 0.3

        # 一个情绪词 → 0.4
        imp2 = chat_uc._calc_importance("我很难过")
        assert imp2 == 0.4

        # 多个情绪词 → 上限 0.8
        imp3 = chat_uc._calc_importance("又累又烦又焦虑又迷茫又失望又生气又孤独又崩溃")
        assert imp3 == 0.8

        # 超 50 字 + 0.1
        long_text = (
            "今天天气真的非常好适合出去散步活动一下身体呼吸新鲜空气"
            "感受大自然的美好风景这样的生活真是让人感到无比舒适和惬意"
            "希望每天都能保持这样的好心情去面对生活中的每一个挑战"
        )
        assert len(long_text) > 50, f"len={len(long_text)}"
        imp4 = chat_uc._calc_importance(long_text)
        assert imp4 == 0.4  # 0.3 + 0.1


# ===================================================================
# 测试 3：Save → Retrieve 循环
# ===================================================================


class TestSaveRetrieveCycle:
    """场景 3：写入记忆后能通过 keyword search 找回。"""

    def test_save_and_search_by_keywords(self, tmp_path: Path) -> None:
        """保存记忆后应能通过关键词检索到。"""
        clock = FrozenClock(datetime(2026, 6, 15, 10, 0, 0))
        conn = _init_db(str(tmp_path / "sr_keyword.db"))

        try:
            repo = SQLiteMemoryRepository(conn, clock)

            # ── 保存记忆 ──
            mem = Memory(created_at=clock.now(),
                memory_id=MemoryId(0),
                user_id=UserId("u1"),
                content="用户今天谈到喜欢哲学和编程",
                keylabel="哲学编程",
                summary="喜欢哲学和编程",
                importance=Importance(0.7),
                mood=Mood("开心"),
            )
            mem_id = _run_async(repo.save(mem))
            assert mem_id.value > 0

            # ── 关键词检索 ──
            results = _run_async(repo.search_by_keywords(
                UserId("u1"), ["哲学"], limit=5,
            ))
            assert len(results) == 1
            assert results[0].content == "用户今天谈到喜欢哲学和编程"

            # ── 保存更多记忆 ──
            mem2 = Memory(created_at=clock.now(),
                memory_id=MemoryId(0),
                user_id=UserId("u1"),
                content="用户最近工作压力很大",
                keylabel="工作压力",
                summary="工作压力",
                importance=Importance(0.6),
            )
            _run_async(repo.save(mem2))

            mem3 = Memory(created_at=clock.now(),
                memory_id=MemoryId(0),
                user_id=UserId("u1"),
                content="用户喜欢跑步和运动",
                keylabel="跑步运动",
                summary="喜欢运动",
                importance=Importance(0.5),
            )
            _run_async(repo.save(mem3))

            # ── 多关键词检索 ──
            results2 = _run_async(repo.search_by_keywords(
                UserId("u1"), ["工作", "压力"], limit=5,
            ))
            assert len(results2) >= 1
            assert any("工作压力" in r.content for r in results2)

            # ── 不存在的关键词应返回空列表 ──
            results3 = _run_async(repo.search_by_keywords(
                UserId("u1"), ["不存在的话题"], limit=5,
            ))
            assert len(results3) == 0

            # ── 跨用户隔离 ──
            results4 = _run_async(repo.search_by_keywords(
                UserId("u2"), ["哲学"], limit=5,
            ))
            assert len(results4) == 0

        finally:
            conn.close()

    def test_save_turn_and_query_chat_history(self, tmp_path: Path) -> None:
        """save_turn 应写入 chat_history 并可查询。"""
        clock = FrozenClock(datetime(2026, 6, 15, 10, 0, 0))
        conn = _init_db(str(tmp_path / "sr_turn.db"))

        try:
            repo = SQLiteMemoryRepository(conn, clock)

            _run_async(repo.save_turn(
                SessionId("s1"), "你好", "你好！有什么可以帮你的？",
                user_id=UserId("u1"), sender_name="小明",
            ))

            rows = conn.execute(
                "SELECT role, content, user_id, sender_name, turn_num FROM chat_history "
                "WHERE session_id = ? ORDER BY id",
                ("s1",),
            ).fetchall()
            assert len(rows) == 2
            assert rows[0]["role"] == "user"
            assert rows[0]["content"] == "你好"
            assert rows[1]["role"] == "assistant"
            assert rows[1]["content"] == "你好！有什么可以帮你的？"
            assert rows[0]["turn_num"] == 1
            assert rows[1]["turn_num"] == 1  # 共享 turn_num
            assert rows[0]["sender_name"] == "小明"

        finally:
            conn.close()


# ===================================================================
# 测试 4：Session 生命周期
# ===================================================================


class TestSessionLifecycle:
    """场景 4：increment_turn → get → save 循环正常。"""

    def test_increment_turn_and_get(self, tmp_path: Path) -> None:
        """increment_turn 应自动初始化 session 并递增轮次。"""
        clock = FrozenClock(datetime(2026, 6, 15, 10, 0, 0))
        conn = _init_db(str(tmp_path / "session_incr.db"))

        try:
            repo = SQLiteSessionRepository(conn, clock)

            # 首次递增应返回 1
            turn1 = _run_async(repo.increment_turn(SessionId("s1")))
            assert turn1 == 1

            # 再次递增应返回 2
            turn2 = _run_async(repo.increment_turn(SessionId("s1")))
            assert turn2 == 2

            # 第三次递增应返回 3
            turn3 = _run_async(repo.increment_turn(SessionId("s1")))
            assert turn3 == 3

            # get 应返回 Session 实体
            session = _run_async(repo.get(SessionId("s1")))
            assert session is not None
            assert session.turn_count == 3
            assert session.session_id.value == "s1"

        finally:
            conn.close()

    def test_save_and_get_session(self, tmp_path: Path) -> None:
        """save 完整 Session 后应能通过 get 读回。"""
        clock = FrozenClock(datetime(2026, 6, 15, 10, 0, 0))
        conn = _init_db(str(tmp_path / "session_save.db"))

        try:
            repo = SQLiteSessionRepository(conn, clock)
            now = clock.now()

            session = Session(
                session_id=SessionId("s2"),
                user_id=UserId("u1"),
                stance="engaged",
                turn_count=5,
                last_active=now,
                dangling_threads={"threads": ["工作话题", "旅行计划"], "turn": 3},
            )
            _run_async(repo.save(session))

            read_back = _run_async(repo.get(SessionId("s2")))
            assert read_back is not None
            assert read_back.session_id.value == "s2"
            assert read_back.user_id is not None
            assert read_back.user_id.value == "u1"
            assert read_back.stance == "engaged"
            assert read_back.turn_count == 5
            assert read_back.dangling_threads.get("threads") == ["工作话题", "旅行计划"]

        finally:
            conn.close()

    def test_update_last_active(self, tmp_path: Path) -> None:
        """update_last_active 应更新会话的最后活跃时间。"""
        clock = FrozenClock(datetime(2026, 6, 15, 10, 0, 0))
        conn = _init_db(str(tmp_path / "session_active.db"))

        try:
            repo = SQLiteSessionRepository(conn, clock)

            # 先确保 session 行存在
            _run_async(repo.increment_turn(SessionId("s3")))

            _run_async(repo.update_last_active(
                SessionId("s3"), clock.now(),
            ))
            session = _run_async(repo.get(SessionId("s3")))
            assert session is not None
            assert session.last_active is not None
            # 经过 strftime → strptime 往返，精确到秒
            assert session.last_active.strftime("%Y-%m-%d %H:%M:%S") == clock.now().strftime("%Y-%m-%d %H:%M:%S")

        finally:
            conn.close()

    def test_dangling_threads(self, tmp_path: Path) -> None:
        """update 和 get dangling_threads 应 round-trip 正常。"""
        clock = FrozenClock(datetime(2026, 6, 15, 10, 0, 0))
        conn = _init_db(str(tmp_path / "session_dangling.db"))

        try:
            repo = SQLiteSessionRepository(conn, clock)

            # 先确保 session 行存在
            _run_async(repo.increment_turn(SessionId("s4")))

            # 初始应返回空 dict
            empty = _run_async(repo.get_dangling_threads(SessionId("s4")))
            assert empty == {} or empty.get("threads") is None

            # 写入悬案
            threads = {"threads": ["未完成的话题A"], "turn": 2}
            _run_async(repo.update_dangling_threads(SessionId("s4"), threads))

            # 读回
            read = _run_async(repo.get_dangling_threads(SessionId("s4")))
            assert read.get("threads") == ["未完成的话题A"]
            assert read.get("turn") == 2

            # 清空
            _run_async(repo.update_dangling_threads(SessionId("s4"), {}))
            cleared = _run_async(repo.get_dangling_threads(SessionId("s4")))
            assert cleared == {}

        finally:
            conn.close()

    def test_last_distill(self, tmp_path: Path) -> None:
        """get_last_distill 和 update_last_distill 应 round-trip 正常。"""
        clock = FrozenClock(datetime(2026, 6, 15, 10, 0, 0))
        conn = _init_db(str(tmp_path / "session_distill.db"))

        try:
            repo = SQLiteSessionRepository(conn, clock)

            # 不存在的 session 应返回 (0, None)
            turn, dt = _run_async(repo.get_last_distill(SessionId("s5")))
            assert turn == 0
            assert dt is None

            # 先创建 session 行
            _run_async(repo.increment_turn(SessionId("s5")))

            # 更新蒸馏进度
            now = clock.now()
            _run_async(repo.update_last_distill(SessionId("s5"), 5, now))

            turn2, dt2 = _run_async(repo.get_last_distill(SessionId("s5")))
            assert turn2 == 5
            assert dt2 is not None
            assert dt2.strftime("%Y-%m-%d %H:%M:%S") == now.strftime("%Y-%m-%d %H:%M:%S")

        finally:
            conn.close()


# ===================================================================
# 测试 5：Identity 读写
# ===================================================================


class TestIdentityReadWrite:
    """场景 5：save_traits → get 循环正常。"""

    def test_save_traits_and_get(self, tmp_path: Path) -> None:
        """保存特质后应能通过 get 读回完整的 Identity。"""
        clock = FrozenClock(datetime(2026, 6, 15, 10, 0, 0))
        conn = _init_db(str(tmp_path / "identity_traits.db"))

        try:
            repo = SQLiteIdentityRepository(conn, clock)

            # 新用户 get 应返回 None
            identity = _run_async(repo.get("u1"))
            assert identity is None

            # 写入特质
            traits = [
                Trait(text="喜欢哲学", strength=5, count=1),
                Trait(text="性格内向", strength=3, count=2),
            ]
            _run_async(repo.save_traits("u1", traits))

            # 读回
            identity = _run_async(repo.get("u1"))
            assert identity is not None
            assert identity.user_id == "u1"
            trait_texts = [t.text for t in identity.traits]
            assert "喜欢哲学" in trait_texts
            assert "性格内向" in trait_texts

        finally:
            conn.close()

    def test_save_preferences_and_self_identity(self, tmp_path: Path) -> None:
        """save_preferences 和 save_self_identity 应覆盖写入。"""
        clock = FrozenClock(datetime(2026, 6, 15, 10, 0, 0))
        conn = _init_db(str(tmp_path / "identity_prefs.db"))

        try:
            repo = SQLiteIdentityRepository(conn, clock)

            # 写入偏好
            prefs = Preferences(likes=["哲学", "编程"], dislikes=["吵闹"])
            _run_async(repo.save_preferences("u2", prefs))

            # 写入自我认知
            self_id = ["程序员", "内向者"]
            _run_async(repo.save_self_identity("u2", self_id))

            # 读回
            identity = _run_async(repo.get("u2"))
            assert identity is not None
            assert "哲学" in identity.preferences.likes
            assert "吵闹" in identity.preferences.dislikes
            assert "程序员" in identity.self_identity
            assert "内向者" in identity.self_identity

            # 覆盖写入偏好
            prefs2 = Preferences(likes=["运动"], dislikes=[])
            _run_async(repo.save_preferences("u2", prefs2))

            identity2 = _run_async(repo.get("u2"))
            assert identity2 is not None
            assert "运动" in identity2.preferences.likes
            assert "哲学" not in identity2.preferences.likes  # 已覆盖

        finally:
            conn.close()

    def test_save_boundaries(self, tmp_path: Path) -> None:
        """save_boundaries 应覆盖写入。"""
        clock = FrozenClock(datetime(2026, 6, 15, 10, 0, 0))
        conn = _init_db(str(tmp_path / "identity_bounds.db"))

        try:
            repo = SQLiteIdentityRepository(conn, clock)

            boundaries = [
                Boundary(description="不要催我"),
                Boundary(description="不要问工资"),
            ]
            _run_async(repo.save_boundaries("u3", boundaries))

            identity = _run_async(repo.get("u3"))
            assert identity is not None
            assert len(identity.boundaries) == 2
            descs = [b.description for b in identity.boundaries]
            assert "不要催我" in descs
            assert "不要问工资" in descs

        finally:
            conn.close()

    def test_trait_merge_decay(self, tmp_path: Path) -> None:
        """save_traits 的确认合并和衰减逻辑。"""
        clock = FrozenClock(datetime(2026, 6, 15, 10, 0, 0))
        conn = _init_db(str(tmp_path / "identity_merge.db"))

        try:
            repo = SQLiteIdentityRepository(conn, clock)

            # 第 1 轮：写入两个特质
            traits1 = [Trait(text="喜欢哲学", strength=5, count=1)]
            _run_async(repo.save_traits("u4", traits1))

            # 第 2 轮：确认一个旧特质 + 一个新特质
            traits2 = [
                Trait(text="喜欢哲学", strength=5, count=1),
                Trait(text="性格内向", strength=5, count=1),
            ]
            _run_async(repo.save_traits("u4", traits2))

            identity = _run_async(repo.get("u4"))
            assert identity is not None
            trait_map = {t.text: t for t in identity.traits}

            # "喜欢哲学" 被确认 2 次 → strength=5, count=2
            assert trait_map["喜欢哲学"].strength == 5
            assert trait_map["喜欢哲学"].count == 2

            # "性格内向" 被确认 1 次 → strength=5, count=1
            assert trait_map["性格内向"].strength == 5
            assert trait_map["性格内向"].count == 1

        finally:
            conn.close()


# ===================================================================
# 测试 6：Graph 操作
# ===================================================================


class TestGraphOperations:
    """场景 6：upsert_node → search_nodes → maintain 循环正常。"""

    def test_upsert_node_and_search(self, tmp_path: Path) -> None:
        """upsert_node 后应能通过 search_nodes 搜索到。"""
        clock = FrozenClock(datetime(2026, 6, 15, 10, 0, 0))
        conn = _init_db(str(tmp_path / "graph_nodes.db"))

        try:
            repo = SQLiteGraphRepository(conn, clock)

            nid = _run_async(repo.upsert_node("u1", "哲学", entity_type="concept"))
            assert nid >= 0

            _run_async(repo.upsert_node("u1", "跑步", entity_type="activity"))
            _run_async(repo.upsert_node("u1", "编程"))

            # 搜索
            results = _run_async(repo.search_nodes("u1", "哲"))
            assert len(results) == 1
            assert results[0].label == "哲学"
            assert results[0].entity_type == "concept"

            results2 = _run_async(repo.search_nodes("u1", "跑"))
            assert len(results2) == 1
            assert results2[0].entity_type == "activity"

            # 不存在的关键词
            results3 = _run_async(repo.search_nodes("u1", "不存在"))
            assert len(results3) == 0

        finally:
            conn.close()

    def test_upsert_node_freq_increments(self, tmp_path: Path) -> None:
        """重复 upsert 同一节点时 freq 递增且返回相同 node_id。"""
        clock = FrozenClock(datetime(2026, 6, 15, 10, 0, 0))
        conn = _init_db(str(tmp_path / "graph_freq.db"))

        try:
            repo = SQLiteGraphRepository(conn, clock)

            nid1 = _run_async(repo.upsert_node("u2", "读书"))
            nid2 = _run_async(repo.upsert_node("u2", "读书"))
            nid3 = _run_async(repo.upsert_node("u2", "读书"))

            assert nid1 == nid2 == nid3

            nodes = _run_async(repo.get_nodes_by_user("u2"))
            assert len(nodes) == 1
            assert nodes[0].freq == 3

        finally:
            conn.close()

    def test_upsert_edge_and_get_edges(self, tmp_path: Path) -> None:
        """upsert_edge 后应能通过 get_edges_by_node 查找到。"""
        clock = FrozenClock(datetime(2026, 6, 15, 10, 0, 0))
        conn = _init_db(str(tmp_path / "graph_edges.db"))

        try:
            repo = SQLiteGraphRepository(conn, clock)

            n1 = _run_async(repo.upsert_node("u3", "小明", entity_type="person"))
            n2 = _run_async(repo.upsert_node("u3", "哲学", entity_type="concept"))

            _run_async(repo.upsert_edge(n1, n2, relation="喜欢"))

            edges = _run_async(repo.get_edges_by_node(n1))
            assert len(edges) >= 1
            assert edges[0].relation == "喜欢"
            assert edges[0].weight == 1.0
            assert edges[0].encounter_count == 1

            # 重复 upsert 应 weight+0.5
            _run_async(repo.upsert_edge(n1, n2, relation="喜欢"))
            edges2 = _run_async(repo.get_edges_by_node(n1))
            edge = next(e for e in edges2 if e.from_node_id == n1 and e.to_node_id == n2)
            assert edge.weight == 1.5
            assert edge.encounter_count == 2

        finally:
            conn.close()

    def test_self_loop_skipped(self, tmp_path: Path) -> None:
        """from_id == to_id 时应跳过不插入。"""
        clock = FrozenClock(datetime(2026, 6, 15, 10, 0, 0))
        conn = _init_db(str(tmp_path / "graph_self.db"))

        try:
            repo = SQLiteGraphRepository(conn, clock)

            nid = _run_async(repo.upsert_node("u4", "孤独"))
            _run_async(repo.upsert_edge(nid, nid, relation="自指"))

            edges = _run_async(repo.get_edges_by_node(nid))
            self_loops = [
                e for e in edges
                if e.from_node_id == nid and e.to_node_id == nid
            ]
            assert len(self_loops) == 0

        finally:
            conn.close()

    def test_label_cleaning(self, tmp_path: Path) -> None:
        """标签前后非文字字符应被清洗。"""
        clock = FrozenClock(datetime(2026, 6, 15, 10, 0, 0))
        conn = _init_db(str(tmp_path / "graph_clean.db"))

        try:
            repo = SQLiteGraphRepository(conn, clock)

            nid = _run_async(repo.upsert_node("u5", " - 测试标签 · "))
            assert nid >= 0

            nodes = _run_async(repo.get_nodes_by_user("u5"))
            assert nodes[0].label == "测试标签"

        finally:
            conn.close()

    def test_maintain_removes_low_weight_edges(self, tmp_path: Path) -> None:
        """maintain 应删除低权重边和孤立节点。"""
        clock = FrozenClock(datetime(2026, 6, 15, 10, 0, 0))
        conn = _init_db(str(tmp_path / "graph_maintain.db"))

        try:
            repo = SQLiteGraphRepository(conn, clock)

            n1 = _run_async(repo.upsert_node("u6", "节点A"))
            n2 = _run_async(repo.upsert_node("u6", "节点B"))

            # 直接插入低权重边 (weight < 0.4)
            conn.execute(
                "INSERT INTO memory_graph_edges "
                "(from_node_id, to_node_id, weight, encounter_count, last_seen, relation) "
                "VALUES (?, ?, 0.2, 1, ?, '')",
                (n1, n2, clock.strftime()),
            )
            conn.commit()

            _run_async(repo.maintain("u6"))

            # 低权重边应被删除
            remaining = conn.execute(
                "SELECT count(*) FROM memory_graph_edges "
                "WHERE from_node_id = ? AND to_node_id = ?",
                (n1, n2),
            ).fetchone()[0]
            assert remaining == 0

        finally:
            conn.close()


# ===================================================================
# 测试 7：EventBus 集成
# ===================================================================


class TestEventBusIntegration:
    """场景 7：注册 Handler → publish → Handler 被调用。"""

    @pytest.mark.asyncio
    async def test_register_and_publish(self) -> None:
        """注册 Handler 后 publish 应触发 Handler。"""
        bus = EventBus()
        results: list[str] = []

        async def on_turn_saved(event: TurnSaved) -> None:
            results.append(f"收到事件: {event.user_id} 轮次 {event.turn_number}")

        async def on_memory_distilled(event: MemoryDistilled) -> None:
            results.append(f"蒸馏完成: {event.keylabel}")

        bus.register(TurnSaved, on_turn_saved)
        bus.register(MemoryDistilled, on_memory_distilled)

        assert bus.handler_count == 2

        # 发布 TurnSaved
        await bus.publish(TurnSaved(
            occurred_at=datetime.now(),
            user_id="test_user",
            session_id="test_session",
            turn_number=3,
            user_input="你好",
            reply="你好！",
        ))
        assert len(results) == 1
        assert "test_user" in results[0]
        assert "3" in results[0]

        # 发布 MemoryDistilled
        await bus.publish(MemoryDistilled(
            occurred_at=datetime.now(),
            user_id="test_user",
            session_id="test_session",
            memory_id=1,
            content="蒸馏内容",
            keylabel="distill_test",
            importance=0.8,
            mood="平静",
            mood_intensity=0.5,
        ))
        assert len(results) == 2
        assert "distill_test" in results[1]

    @pytest.mark.asyncio
    async def test_unregister_stops_handler(self) -> None:
        """注销 Handler 后 publish 不应再触发该 Handler。"""
        bus = EventBus()
        results: list[str] = []

        async def handler(event: TurnSaved) -> None:
            results.append("被调用了")

        bus.register(TurnSaved, handler)
        await bus.publish(TurnSaved(
            occurred_at=datetime.now(),
            user_id="u", session_id="s", turn_number=1,
            user_input="hi", reply="hi",
        ))
        assert len(results) == 1

        bus.unregister(TurnSaved, handler)
        await bus.publish(TurnSaved(
            occurred_at=datetime.now(),
            user_id="u", session_id="s", turn_number=2,
            user_input="hi", reply="hi",
        ))
        assert len(results) == 1  # 未增加

    @pytest.mark.asyncio
    async def test_error_in_handler_does_not_block_others(self) -> None:
        """Handler 抛出异常不应影响其他 Handler。"""
        bus = EventBus()
        results: list[str] = []

        async def handler_a(event: TurnSaved) -> None:
            raise RuntimeError("handler_a 出错")

        async def handler_b(event: TurnSaved) -> None:
            results.append("handler_b 正常执行")

        bus.register(TurnSaved, handler_a)
        bus.register(TurnSaved, handler_b)

        await bus.publish(TurnSaved(
            occurred_at=datetime.now(),
            user_id="u", session_id="s", turn_number=1,
            user_input="hi", reply="hi",
        ))
        assert len(results) == 1
        assert "handler_b 正常执行" in results[0]

    @pytest.mark.asyncio
    async def test_multiple_events_multiple_handlers(self) -> None:
        """多种事件类型各自触发对应 Handler。"""
        bus = EventBus()
        turn_saved_count = 0
        distilled_count = 0

        async def count_turn(event: TurnSaved) -> None:
            nonlocal turn_saved_count
            turn_saved_count += 1

        async def count_distill(event: MemoryDistilled) -> None:
            nonlocal distilled_count
            distilled_count += 1

        bus.register(TurnSaved, count_turn)
        bus.register(MemoryDistilled, count_distill)

        for i in range(3):
            await bus.publish(TurnSaved(
            occurred_at=datetime.now(),
                user_id="u", session_id="s", turn_number=i + 1,
                user_input="hi", reply="hi",
            ))
        await bus.publish(MemoryDistilled(
            occurred_at=datetime.now(),
            user_id="u", session_id="s", memory_id=1,
            content="c", keylabel="k", importance=0.8,
            mood="平静", mood_intensity=0.5,
        ))

        assert turn_saved_count == 3
        assert distilled_count == 1

    @pytest.mark.asyncio
    async def test_clear_removes_all_handlers(self) -> None:
        """clear 应清空所有注册的 Handler。"""
        bus = EventBus()

        async def handler(event: TurnSaved) -> None:
            pass

        bus.register(TurnSaved, handler)
        bus.register(MemoryDistilled, handler)
        assert bus.handler_count == 2

        bus.clear()
        assert bus.handler_count == 0


# ===================================================================
# 测试 8：FusionService 融合
# ===================================================================


@dataclass
class _FusionTestConfig:
    """实现 IFusionConfig protocol 的测试配置。"""
    total_cap: int = 5
    channel_min: list[int] = (1, 1, 1)
    channel_weights: list[float] = (0.5, 1.0, 0.6)


class TestFusionService:
    """场景 8：三通道输入 → 保底去重加权排序正常。"""

    def test_fusion_respects_total_cap(self) -> None:
        """融合结果总数不超过 total_cap。"""
        config = _FusionTestConfig(total_cap=3)
        svc = FusionService(config)

        channels = {
            ChannelTag.RECENT: [("记忆A", 1.0), ("记忆B", 1.0), ("记忆C", 1.0)],
            ChannelTag.RESONANCE: [("记忆D", 1.0), ("记忆E", 1.0)],
            ChannelTag.SKELETON: [("记忆F", 1.0)],
        }
        result = svc.fuse(channels)
        assert len(result) <= 3

    def test_fusion_deduplicates(self) -> None:
        """相同内容的记忆不应重复出现。"""
        config = _FusionTestConfig(total_cap=5)
        svc = FusionService(config)

        channels = {
            ChannelTag.RECENT: [("相同内容", 1.0)],
            ChannelTag.RESONANCE: [("相同内容", 1.0)],
            ChannelTag.SKELETON: [("不同内容", 1.0)],
        }
        result = svc.fuse(channels)
        contents = [c for c, _ in result]
        assert contents.count("相同内容") == 1

    def test_fusion_channel_minimums(self) -> None:
        """每通道至少返回 ch_min 条。"""
        config = _FusionTestConfig(
            total_cap=6, channel_min=[2, 2, 2], channel_weights=[0.5, 1.0, 0.6],
        )
        svc = FusionService(config)

        channels = {
            ChannelTag.RECENT: [("时间记忆1", 1.0), ("时间记忆2", 1.0), ("时间记忆3", 1.0)],
            ChannelTag.RESONANCE: [("语义记忆1", 1.0), ("语义记忆2", 1.0), ("语义记忆3", 1.0)],
            ChannelTag.SKELETON: [("图谱记忆1", 1.0), ("图谱记忆2", 1.0), ("图谱记忆3", 1.0)],
        }
        result = svc.fuse(channels)
        tags = [t for _, t in result]
        assert tags.count(ChannelTag.RECENT) >= 2
        assert tags.count(ChannelTag.RESONANCE) >= 2
        assert tags.count(ChannelTag.SKELETON) >= 2

    def test_fusion_weighted_order(self) -> None:
        """高权重通道的条目应排在前面。"""
        config = _FusionTestConfig(
            total_cap=6,
            channel_min=[0, 0, 0],
            channel_weights=[0.1, 1.0, 0.3],
        )
        svc = FusionService(config)

        channels = {
            ChannelTag.RECENT: [("时间记忆", 1.0)],
            ChannelTag.RESONANCE: [("语义记忆", 1.0)],
            ChannelTag.SKELETON: [("图谱记忆", 1.0)],
        }
        result = svc.fuse(channels)

        # Resonance 权重最高 (1.0)，应排在首位
        assert len(result) >= 3
        assert result[0][1] == ChannelTag.RESONANCE

    def test_fusion_returns_correct_tag(self) -> None:
        """返回的每个条目都应携带正确的 source tag。"""
        config = _FusionTestConfig(total_cap=5)
        svc = FusionService(config)

        channels = {
            ChannelTag.RECENT: [("时间记忆", 1.0)],
            ChannelTag.RESONANCE: [("语义记忆", 1.0)],
            ChannelTag.SKELETON: [("图谱记忆", 1.0)],
        }
        result = svc.fuse(channels)

        tag_map = {content: tag for content, tag in result}
        assert tag_map["时间记忆"] == ChannelTag.RECENT
        assert tag_map["语义记忆"] == ChannelTag.RESONANCE
        assert tag_map["图谱记忆"] == ChannelTag.SKELETON

    def test_fusion_empty_channels(self) -> None:
        """所有通道为空时应返回空列表。"""
        config = _FusionTestConfig(total_cap=5)
        svc = FusionService(config)

        channels = {
            ChannelTag.RECENT: [],
            ChannelTag.RESONANCE: [],
            ChannelTag.SKELETON: [],
        }
        result = svc.fuse(channels)
        assert result == []

    def test_fusion_partial_channels(self) -> None:
        """部分通道数据不足时不影响其他通道。"""
        config = _FusionTestConfig(
            total_cap=3, channel_min=[1, 0, 0], channel_weights=[1.0, 1.0, 1.0],
        )
        svc = FusionService(config)

        channels = {
            ChannelTag.RECENT: [("唯一记忆", 1.0)],
            ChannelTag.RESONANCE: [],
            ChannelTag.SKELETON: [],
        }
        result = svc.fuse(channels)
        assert len(result) == 1
        assert result[0][0] == "唯一记忆"

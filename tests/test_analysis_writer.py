"""
test_analysis_writer.py — AnalysisWriter 回归测试
===================================================

覆盖场景：
  1. Empty key_facts      — write_all 返回空列表
  2. String importance    — 字符串 vs 浮点数比较不抛 TypeError
  3. Valid data           — 完整数据写入（不含 key_facts / dangling_threads 到 cognitive_distill）
  4. Missing fields       — 最小数据写入
  5. Unused variable      — now_str 回归检测

约定：
  - 使用 FrozenClock 固定时间
  - 使用 in-memory SQLite 避免文件残留
  - 使用 ensure_schema 初始化表结构
  - 每个测试独立数据库互不干扰
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
from infrastructure.persistence.ddl import ensure_schema
from infrastructure.config.settings import Settings, StorageSettings

# ── 应用层 ──────────────────────────────────────────────────────────
from application.analysis_writer import AnalysisWriter

# ── 领域实体 ────────────────────────────────────────────────────────
from domain.entities.memory import SessionId


# ===================================================================
# 测试 Helper
# ===================================================================


def _run_async(coro):
    """同步包装器：在同步测试函数中安全执行协程。"""
    return asyncio.run(coro)


def _init_in_memory_db() -> sqlite3.Connection:
    """创建并初始化内存 SQLite 数据库，返回连接。"""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    ensure_schema(conn, StorageSettings())
    return conn


# ===================================================================
# 共享分析数据常量
# ===================================================================

_VALID_ANALYSIS: dict[str, Any] = {
    "mood": "开心",
    "mood_intensity": 0.8,
    "user_state": "engaged",
    "traits_updates": ["喜欢哲学", "性格开朗"],
    "speech_quirks": ["喜欢用'嗯...'开头"],
    "preferences": {
        "likes": ["哲学", "编程"],
        "dislikes": ["吵闹"],
    },
    "self_identity": ["程序员", "思考者"],
    "boundaries": [
        {"description": "不要催我"},
        {"description": "不要问工资"},
    ],
    "shared_jokes": [
        {"trigger": "哲学", "context": "用户喜欢讨论哲学话题"},
    ],
    "dangling_threads": [
        "关于工作的深层次讨论",
        "哲学话题需要继续探讨",
    ],
    "entities": [
        {
            "name": "小明",
            "type": "person",
            "relations": [
                {"target": "哲学", "relation": "喜欢"},
            ],
        },
        {
            "name": "哲学",
            "type": "concept",
            "relations": [],
        },
    ],
    "key_facts": [
        {
            "content": "用户是一名程序员，喜欢哲学和编程",
            "temporal": "permanent",
        },
        {
            "content": "用户最近在找工作，感到焦虑",
            "temporal": "transient",
            "expires_after_days": 30,
        },
    ],
    "importance": 0.7,
}

_STRING_IMPORTANCE_ANALYSIS: dict[str, Any] = {
    "mood": "平静",
    "mood_intensity": 0.3,
    "importance": "0.5",
    "key_facts": [
        {"content": "字符串重要性事实", "temporal": "permanent"},
    ],
}

_EMPTY_KEY_FACTS_ANALYSIS: dict[str, Any] = {
    "mood": "平静",
    "mood_intensity": 0.0,
    "key_facts": [],
}

_MINIMAL_ANALYSIS: dict[str, Any] = {
    "mood": "平静",
    "mood_intensity": 0.0,
}


# ===================================================================
# 特征化测试
# ===================================================================


class TestAnalysisWriter:
    """AnalysisWriter 回归测试套件。"""

    def setup_method(self) -> None:
        """每个测试方法前执行：创建基础设施。"""
        self.clock = FrozenClock(datetime(2026, 6, 15, 10, 0, 0))
        self.conn = _init_in_memory_db()

        self.memory_repo = SQLiteMemoryRepository(self.conn, self.clock)
        self.session_repo = SQLiteSessionRepository(self.conn, self.clock)
        self.identity_repo = SQLiteIdentityRepository(self.conn, self.clock)
        self.graph_repo = SQLiteGraphRepository(self.conn, self.clock)
        self.settings = Settings()

        # shared_context upsert 调用记录
        self.upserted_shared: list[tuple[str, str, str]] = []

        async def _mock_upsert_shared(user_id: str, trigger: str, context: str) -> None:
            self.upserted_shared.append((user_id, trigger, context))

        self.writer = AnalysisWriter(
            memory_repo=self.memory_repo,
            session_repo=self.session_repo,
            identity_repo=self.identity_repo,
            graph_repo=self.graph_repo,
            clock=self.clock,
            settings=self.settings,
            upsert_shared_context=_mock_upsert_shared,
        )

    def teardown_method(self) -> None:
        """每个测试方法后执行：关闭数据库连接。"""
        self.conn.close()

    # ── 测试 1：write_all 不再返回子条目 ──

    def test_write_all_returns_empty_list(self) -> None:
        """key_facts 和 dangling_threads 不再写入 cognitive_distill，
        write_all 应始终返回空列表。"""
        result = _run_async(self.writer.write_all(
            user_id="test_user",
            session_id="test_session",
            analysis=_EMPTY_KEY_FACTS_ANALYSIS,
        ))
        assert result == [], (
            f"write_all 应返回空列表，收到 {result}"
        )

    # ── 测试 2：String importance（仅验证不崩溃）──

    def test_importance_float_conversion_safe(self) -> None:
        """Regression: importance 为字符串时不应引发 TypeError。

        LLM JSON 有时返回字符串格式的 "0.5"。
        _log_summary 中的 try/except 应安全转换。"""
        try:
            result = _run_async(self.writer.write_all(
                user_id="test_user",
                session_id="test_session",
                analysis=_STRING_IMPORTANCE_ANALYSIS,
            ))
            # write_all 不再写入事实，仅验证不崩溃
            assert isinstance(result, list)
        except TypeError as e:
            pytest.fail(f"字符串 importance 不应引发 TypeError: {e}")

    # ── 测试 3：Valid data ─────────────────────────────────────────

    def test_write_all_with_valid_data(self) -> None:
        """完整 9 维分析数据应成功写入所有 Repository。"""
        # 预创建 session 行，使 update_last_active / dangling_threads 生效
        _run_async(self.session_repo.increment_turn(SessionId("test_session")))

        result = _run_async(self.writer.write_all(
            user_id="test_user",
            session_id="test_session",
            analysis=_VALID_ANALYSIS,
        ))

        # 验证返回空列表（不再写入子条目到 cognitive_distill）
        assert isinstance(result, list), (
            f"write_all 应返回 list，收到 {type(result)}"
        )
        assert result == [], (
            f"write_all 应返回空列表，收到 {result}"
        )

        # 验证 cognitive_distill 无记录（AnalysisWriter 不再写入子条目）
        distill_rows = self.conn.execute(
            "SELECT COUNT(*) FROM cognitive_distill WHERE user_id = ?",
            ("test_user",),
        ).fetchone()[0]
        assert distill_rows == 0, (
            f"AnalysisWriter 不应写入 cognitive_distill，实为 {distill_rows}"
        )

        # 验证 identity_memory 有 traits
        identity_row = self.conn.execute(
            "SELECT traits FROM identity_memory WHERE user_id = ?",
            ("test_user",),
        ).fetchone()
        assert identity_row is not None, "identity_memory 应存在"
        assert len(identity_row[0]) > 0, "traits 不应为空"

        # 验证 graph 有节点
        node_count = self.conn.execute(
            "SELECT COUNT(*) FROM memory_graph_nodes WHERE user_id = ?",
            ("test_user",),
        ).fetchone()[0]
        assert node_count >= 2, (
            f"图谱节点应 >= 2 (小明 + 哲学)，实际 {node_count}"
        )

        # 验证 shared_context 调用了 callback
        assert len(self.upserted_shared) >= 1, (
            "shared_jokes 应触发 upsert 回调"
        )
        # 验证回调内容正确性：_VALID_ANALYSIS.shared_jokes[0]
        if self.upserted_shared:
            _, trigger, context = self.upserted_shared[0]
            assert trigger == "哲学", (
                f"trigger 应为 '哲学'，收到 {trigger!r}"
            )
            assert "哲学" in context, (
                f"context 应包含哲学，收到 {context!r}"
            )

        # 验证 session_state 的 dangling_threads 已更新
        session_row = self.conn.execute(
            "SELECT dangling_threads FROM session_state WHERE session_id = ?",
            ("test_session",),
        ).fetchone()
        assert session_row is not None, "session_state 应存在"
        assert '"关于工作的深层次讨论"' in session_row[0], (
            "dangling_threads 应包含悬案内容"
        )

    # ── 测试 4：Missing fields ─────────────────────────────────────

    def test_write_all_with_missing_fields(self) -> None:
        """最小/部分数据应优雅降级，不崩溃。"""
        result = _run_async(self.writer.write_all(
            user_id="test_user",
            session_id="test_session",
            analysis=_MINIMAL_ANALYSIS,
        ))

        assert isinstance(result, list), (
            f"write_all 应返回 list，收到 {type(result)}"
        )
        # 无 key_facts 和 dangling_threads → 空列表
        assert result == [], (
            f"无 facts/threads 时应返回空列表，收到 {result}"
        )

        # 验证不应有残留数据库写入
        distill_count = self.conn.execute(
            "SELECT COUNT(*) FROM cognitive_distill WHERE user_id = ?",
            ("test_user",),
        ).fetchone()[0]
        assert distill_count == 0, (
            f"不应写入 cognitive_distill，实为 {distill_count}"
        )

        identity_row = self.conn.execute(
            "SELECT COUNT(*) FROM identity_memory WHERE user_id = ?",
            ("test_user",),
        ).fetchone()[0]
        assert identity_row == 0, (
            f"不应写入 identity_memory，实为 {identity_row}"
        )

    def test_write_all_with_missing_mood(self) -> None:
        """mood 缺失时不应崩溃。"""
        partial = {
            "key_facts": [
                {"content": "无 mood 的事实", "temporal": "permanent"},
            ],
        }
        result = _run_async(self.writer.write_all(
            user_id="test_user",
            session_id="test_session",
            analysis=partial,
        ))
        # key_facts 不再写入，仅验证不崩溃
        assert isinstance(result, list)

    def test_write_all_with_missing_session_state(self) -> None:
        """session_state 行不存在时应优雅处理。

        write_all 中的 update_last_active / _write_session_stance /
        _write_dangling_threads 都应能正确处理 session 不存在的情况。"""
        result = _run_async(self.writer.write_all(
            user_id="test_user",
            session_id="uninitialized_session",
            analysis=_VALID_ANALYSIS,
        ))
        assert isinstance(result, list)
        # 不再写入子条目 → 空列表
        assert result == [], (
            f"write_all 应返回空列表，收到 {len(result)}"
        )

        # 验证 session_state 不应有行（update_last_active 只是空 UPDATE）
        session_row = self.conn.execute(
            "SELECT COUNT(*) FROM session_state WHERE session_id = ?",
            ("uninitialized_session",),
        ).fetchone()[0]
        assert session_row == 0, (
            "未预创建 session 时 session_state 应仍为空"
        )

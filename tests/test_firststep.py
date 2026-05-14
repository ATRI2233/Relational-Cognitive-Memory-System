"""测试 FirstStep 最小流程 — 跑通 4 步 Pipeline"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from datetime import datetime, timedelta
from minimal_rcms import MinimalRCMS
from backends import MockBackend

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_memory.db")


@pytest.fixture
def rcms():
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    r = MinimalRCMS(db_path=DB_PATH)
    yield r
    r.close()
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)


@pytest.fixture
def backend():
    return MockBackend()


@pytest.mark.asyncio
async def test_table_creation(rcms):
    tables = [t[0] for t in rcms.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()]
    for t in ('long_term_memory', 'session_state', 'chat_history'):
        assert t in tables, f"缺少表: {t}"


def test_stance_detection(rcms):
    assert rcms.detect_stance("你好") == 'casual'
    assert rcms.detect_stance("今天天气不错") == 'casual'
    assert rcms.detect_stance("我好累啊") == 'engaged'
    assert rcms.detect_stance("你觉得我应该怎么办？") == 'engaged'
    assert rcms.detect_stance("我今天遇到一件非常有意思的事情想跟你聊聊") == 'engaged'


@pytest.mark.asyncio
async def test_casual_conversation(rcms, backend):
    reply = await rcms.chat("u1", "s1", "今天天气不错", backend)
    state = rcms.conn.execute(
        "SELECT stance, turn_count FROM session_state WHERE session_id = ?", ("s1",)
    ).fetchone()
    assert state[0] == 'casual'
    assert state[1] == 1
    assert reply == "嗯，我在听，你继续说。"


@pytest.mark.asyncio
async def test_engaged_with_memory_write(rcms, backend):
    await rcms.chat("u1", "s2", "最近好累，工作压力太大了天天加班", backend)
    state = rcms.conn.execute(
        "SELECT stance, turn_count FROM session_state WHERE session_id = ?", ("s2",)
    ).fetchone()
    assert state[0] == 'engaged'
    assert state[1] == 1

    memories = rcms.conn.execute(
        "SELECT content, memory_type FROM long_term_memory"
    ).fetchall()
    assert len(memories) >= 1


@pytest.mark.asyncio
async def test_memory_retrieval(rcms, backend):
    # 先写入一条记忆
    rcms.conn.execute(
        "INSERT INTO long_term_memory (user_id, content, memory_type, session_id, created_at) VALUES (?, ?, ?, ?, ?)",
        ("u1", "曾经为了一个项目熬夜三个月", "event", "s3", datetime.now().isoformat())
    )
    rcms.conn.commit()

    prompt, stance = rcms.build_prompt("u1", "s3", "熬夜项目")
    assert "熬夜三个月" in prompt or "熬夜" in prompt


def test_time_fuzzing(rcms):
    assert "前两天" in rcms._fuzz_time(datetime.now().isoformat())
    assert "不久前" in rcms._fuzz_time((datetime.now() - timedelta(days=7)).isoformat())
    assert "前段时间" in rcms._fuzz_time((datetime.now() - timedelta(days=30)).isoformat())
    assert "很久以前" in rcms._fuzz_time((datetime.now() - timedelta(days=200)).isoformat())


@pytest.mark.asyncio
async def test_history_tracking(rcms, backend):
    await rcms.chat("u1", "s4", "你好", backend)
    await rcms.chat("u1", "s4", "今天怎么样", backend)
    history = rcms.conn.execute(
        "SELECT role FROM chat_history WHERE session_id = ? ORDER BY created_at", ("s4",)
    ).fetchall()
    assert len(history) == 4
    assert history[0][0] == 'user'
    assert history[2][0] == 'user'


@pytest.mark.asyncio
async def test_cross_session_memory(rcms, backend):
    # Session 1: 写入记忆
    await rcms.chat("u1", "s5", "我最近在学习机器学习，感觉很有意思", backend)
    # Session 2: 应该能检索到
    prompt, _ = rcms.build_prompt("u1", "s6", "机器学习")
    assert "机器学习" in prompt


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

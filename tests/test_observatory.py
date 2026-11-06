"""观测站只读接口测试 —— 面板的五组主接口 + 原始表接口。

观测站是纯同步的 SQLite 只读封装，不需要事件循环。
这里的用例主要防两件事：
  1. schema 改动后接口静默返回空（列名对不上不会报错，只会少数据）
  2. 历史脏数据（final_tags 里塞字符串、sender_name 是哈希）导致整个接口 500
"""

from __future__ import annotations

import datetime as dt
import json
import sqlite3

import pytest

from infrastructure.config.settings import StorageSettings
from infrastructure.persistence.ddl import ensure_schema
from observatory.server import _Store

SESSION = "S1"
USER = "U1"

# 时间戳一律相对今天生成，否则用例会在日期翻篇后失效
_TODAY = dt.date.today()
D0 = f"{_TODAY} 10:00:00"
D1 = f"{_TODAY - dt.timedelta(days=1)} 10:00:00"
D5 = f"{_TODAY - dt.timedelta(days=5)} 10:00:00"


@pytest.fixture()
def store(tmp_path):
    db = tmp_path / "rcms_memory.db"
    conn = sqlite3.connect(str(db))
    ensure_schema(conn, StorageSettings())
    with conn:
        conn.executemany(
            "INSERT INTO cognitive_distill (user_id, session_id, content, keylabel, mood, "
            "mood_intensity, importance, entities, created_at, embedding_dim) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            [
                (USER, SESSION, "他很着急，四五十天后就要考研", "考研", "焦虑", 0.7, 0.9,
                 json.dumps(["考研", "着急"], ensure_ascii=False), D0, 1536),
                (USER, SESSION, "他常常熬到凌晨两点", "作息", "疲惫", 0.4, 0.5,
                 json.dumps(["熬夜"], ensure_ascii=False), D1, 1536),
                (USER, SESSION, "他喜欢直接给结论", "偏好", "", 0.0, 0.3,
                 json.dumps([], ensure_ascii=False), D5, 1536),
            ],
        )
        for label, ntype, freq in (("考研", "event", 5), ("焦虑", "emotion", 3)):
            conn.execute(
                "INSERT INTO memory_graph_nodes (user_id, label, node_type, freq, last_seen) "
                "VALUES (?,?,?,?,?)",
                (USER, label, ntype, freq, D0),
            )
        conn.execute(
            "INSERT INTO memory_graph_edges (from_node_id, to_node_id, weight, "
            "encounter_count, relation, last_seen) VALUES (1, 2, 2.5, 4, '引发', ?)",
            (D0,),
        )
        conn.execute(
            "INSERT INTO chat_history (session_id, role, content, created_at, turn_num, "
            "user_id, sender_name) VALUES (?,?,?,?,?,?,?)",
            (SESSION, "user", "你好", D0, 1, USER, USER),
        )
        conn.execute(
            "INSERT INTO chat_history (session_id, role, content, created_at, turn_num, "
            "user_id, sender_name) VALUES (?,?,?,?,?,?,?)",
            (SESSION, "assistant", "在的", D0, 1, USER, USER),
        )
        conn.execute(
            "INSERT INTO session_state (session_id, user_id, turn_count, last_active, "
            "last_distill_turn) VALUES (?,?,?,?,?)",
            (SESSION, USER, 12, D0, 0),
        )
        good = [{"tag": "recent", "content": "他很着急"}, {"tag": "resonance", "content": "考研"}]
        conn.executemany(
            "INSERT INTO retrieval_trace (session_id, user_id, user_input, final_tags, "
            "injected, embedding_ok, embedding_dim) VALUES (?,?,?,?,?,?,?)",
            [
                (SESSION, USER, "着急", json.dumps(good, ensure_ascii=False), "注入内容", 1, 1536),
                # 历史脏数据：final_tags 是纯字符串列表
                (SESSION, USER, "旧记录", json.dumps(["recent", "skeleton"]), "", 0, 0),
            ],
        )
    conn.close()
    return _Store(
        str(db),
        hints={"distill_max_turns": 30, "distill_max_minutes": 60, "total_cap": 5},
    )


def test_stats_counts_and_progress(store):
    d = store.stats()
    assert d["available"] is True
    assert d["counts"]["memories"] == 3
    assert d["counts"]["entities"] == 2
    assert d["counts"]["edges"] == 1
    assert d["counts"]["messages"] == 2
    assert d["counts"]["traces"] == 2
    assert d["sessions"] == 1
    # 蒸馏进度：12 轮、门槛 30，还差 18
    assert d["distill"]["gap"] == 12
    assert d["distill"]["remaining"] == 18
    assert d["distill"]["ever_distilled"] is False
    # 近 14 天补零
    assert len(d["daily"]) == 14
    assert d["daily"][-1]["d"] == _TODAY.isoformat()
    moods = {m["mood"] for m in d["moods"]}
    assert moods == {"焦虑", "疲惫"}  # 空情绪不进分布


def test_stats_on_missing_db(tmp_path):
    d = _Store(str(tmp_path / "nope.db")).stats()
    assert d["available"] is False
    assert d["counts"] == {}
    assert d["daily"] == []


def test_memories_orders_and_filters(store):
    recent = store.memories()
    assert [m["id"] for m in recent["items"]] == [1, 2, 3]  # created_at DESC
    assert recent["total"] == 3
    assert recent["items"][0]["entities"] == ["考研", "着急"]

    by_imp = store.memories(order="importance")
    assert [m["id"] for m in by_imp["items"]] == [1, 2, 3]

    oldest = store.memories(order="oldest")
    assert [m["id"] for m in oldest["items"]] == [3, 2, 1]

    assert store.memories(mood="焦虑")["total"] == 1
    assert store.memories(query="熬夜")["total"] == 1
    assert store.memories(query="不存在的东西")["total"] == 0
    # days 过滤按自然日算，1 天只剩今天的
    assert store.memories(days=1)["total"] == 1

    page = store.memories(limit=1, offset=1)
    assert len(page["items"]) == 1 and page["total"] == 3


def test_graph_joins_labels(store):
    g = store.graph()
    assert [n["label"] for n in g["nodes"]] == ["考研", "焦虑"]  # freq DESC
    assert g["edges"][0]["from_label"] == "考研"
    assert g["edges"][0]["to_label"] == "焦虑"
    assert g["edges"][0]["weight"] == 2.5
    assert {t["type"] for t in g["types"]} == {"event", "emotion"}


def test_sessions_and_dialogue(store):
    rows = store.sessions()
    assert len(rows) == 1
    assert rows[0]["messages"] == 2
    assert rows[0]["turn_count"] == 12

    d = store.dialogue(SESSION)
    assert d["total"] == 2
    assert [m["role"] for m in d["items"]] == ["user", "assistant"]
    # 不传 session 时落到最近一个会话
    assert store.dialogue()["session_id"] == SESSION


def test_traces_tolerates_legacy_final_tags(store):
    rows = store.traces()
    assert len(rows) == 2
    by_input = {r["user_input"]: r for r in rows}
    assert by_input["着急"]["hit_count"] == 2
    assert by_input["着急"]["channels"] == ["recent", "resonance"]
    # 纯字符串的老数据不能让接口炸，标签照样取出来
    assert by_input["旧记录"]["hit_count"] == 2
    assert by_input["旧记录"]["channels"] == ["recent", "skeleton"]


def test_overview_and_rows(store):
    ov = store.overview()
    names = [t["name"] for t in ov["tables"]]
    assert "cognitive_distill" in names
    assert len(names) == 9  # sqlite_sequence 已被过滤
    assert all(t["title"] != t["name"] for t in ov["tables"])  # 每张表都有中文标题

    d = store.rows("cognitive_distill", limit=2)
    assert d["total"] == 3
    assert len(d["rows"]) == 2
    assert d["columns"][0]["name"] == "id"

    # 图谱边自动联查实体名
    e = store.rows("memory_graph_edges")
    assert "from_label" in e["rows"][0]
    assert e["rows"][0]["from_label"] == "考研"
    # 搜索可以命中联查出来的名称
    assert store.rows("memory_graph_edges", query="焦虑")["total"] == 1

    with pytest.raises(ValueError):
        store.rows("no_such_table")


def test_trace_detail_parses_json(store):
    t = store.trace(1)
    assert t["embedding_ok"] is True
    assert t["fusion"] is None or isinstance(t["fusion"], dict)
    with pytest.raises(ValueError):
        store.trace(999)

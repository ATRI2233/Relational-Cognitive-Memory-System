"""RCMS 本地观测站 —— 插件自带的 localhost HTTP 服务。

只做只读展示，不参与对话链路：
  1. 各张 SQLite 表里到底存了什么
  2. 每次「三路召回」的来源与融合细节

实现约束：
  - 只用标准库（http.server），不引入额外依赖
  - 跑在独立守护线程里，不占用 AstrBot 的事件循环
  - 每次请求单独开只读连接（``mode=ro``），避免跨线程共享 sqlite3 连接
  - 只监听 127.0.0.1（默认），不做鉴权，因此不允许改到公网地址
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import os
import re
import sqlite3
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse

logger = logging.getLogger("rcms.webui")

_STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
_TABLE_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# 表的用途说明 —— 「这个表到底在存什么」比列名本身更重要
TABLE_META: dict[str, dict[str, str]] = {
    "cognitive_distill": {
        "title": "认知记忆",
        "desc": "蒸馏产出的记忆条目，长期记忆的主体。每条含情绪、重要性、向量、关键标签。",
    },
    "chat_history": {
        "title": "对话历史",
        "desc": "逐轮原始对话记录（用户输入 + 回复），蒸馏的原料。",
    },
    "analysis_raw": {
        "title": "分析原文",
        "desc": "每次蒸馏时 LLM 返回的原始 JSON，用于排查解析失败。",
    },
    "retrieval_trace": {
        "title": "三路召回追踪",
        "desc": "每次构建提示词时三条通道的候选、分数与融合过程。",
    },
    "identity_memory": {
        "title": "身份画像",
        "desc": "从对话中沉淀的用户特质、喜好、边界。",
    },
    "memory_graph_nodes": {
        "title": "图谱节点",
        "desc": "记忆里的实体（人、事、物）及其出现频次。",
    },
    "memory_graph_edges": {
        "title": "图谱边",
        "desc": "实体之间的关系边，带权重与最近出现时间（已联查两端实体名）。",
    },
    "session_state": {
        "title": "会话状态",
        "desc": "每个会话的轮次、蒸馏进度、悬案线索。",
    },
    "user_mappings": {"title": "用户映射", "desc": "昵称/别名 → 用户 ID 的映射。"},
}

# 单次返回的最大行数，防止误点把整库拉爆
_MAX_LIMIT = 500
# 单元格文本截断长度（完整内容可在详情里看）
_CELL_MAX = 1200

# 默认排序：优先按自增 id 倒序（最新的在上）；没有 id 的表退到时间/频次列
_ORDER_FALLBACK = ("id", "created_at", "last_seen", "updated_at", "freq")

# 有些表只存 ID（图谱边的两端是 node_id），直接展示成数字等于没展示。
# 这里给这类表补一条联查，把可读名称带出来，并让搜索能按名称命中。
_DERIVED_SOURCES: dict[str, dict] = {
    "memory_graph_edges": {
        # 名称紧跟各自的 ID，避免用户为了看懂一行而横向滚到最右
        "select": (
            "SELECT e.from_node_id, n1.label AS from_label, "
            "e.to_node_id, n2.label AS to_label, e.weight, e.encounter_count, "
            "e.last_seen, e.relation, e.created_at"
        ),
        "from": (
            "FROM memory_graph_edges e "
            "LEFT JOIN memory_graph_nodes n1 ON n1.node_id = e.from_node_id "
            "LEFT JOIN memory_graph_nodes n2 ON n2.node_id = e.to_node_id"
        ),
        "alias": "e",
        "columns": [
            {"name": "from_label", "type": "TEXT", "pk": False},
            {"name": "to_label", "type": "TEXT", "pk": False},
        ],
        "display": [
            "from_node_id",
            "from_label",
            "to_node_id",
            "to_label",
            "weight",
            "encounter_count",
            "last_seen",
            "relation",
            "created_at",
        ],
        "search": ["n1.label LIKE ?", "n2.label LIKE ?"],
    },
}


def _cell_value(value: object) -> object:
    """把 SQLite 值转成可 JSON 序列化的形式。"""
    if isinstance(value, bytes):
        return f"<BLOB {len(value)} bytes>"
    if isinstance(value, str) and len(value) > _CELL_MAX:
        return value[:_CELL_MAX] + f"…（共 {len(value)} 字符）"
    return value


class _Store:
    """对 SQLite 的只读访问封装。"""

    def __init__(self, db_path: str, hints: dict | None = None) -> None:
        self.db_path = db_path
        # 面板拿不到的运行期参数（蒸馏门槛、召回上限…）由插件注入，
        # 用来在总览里给出「为什么记忆还是空的」这类判断。
        self.hints = hints or {}

    def _connect(self) -> sqlite3.Connection:
        uri = f"file:{self.db_path.replace(os.sep, '/')}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=3.0)
        conn.row_factory = sqlite3.Row
        return conn

    @property
    def available(self) -> bool:
        return os.path.isfile(self.db_path)

    def table_names(self) -> list[str]:
        if not self.available:
            return []
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%' ORDER BY name"
            ).fetchall()
        return [r["name"] for r in rows]

    def overview(self) -> dict:
        if not self.available:
            return {
                "db_path": self.db_path,
                "available": False,
                "size_bytes": 0,
                "tables": [],
            }
        tables = []
        with self._connect() as conn:
            for name in self.table_names():
                try:
                    count = conn.execute(
                        f'SELECT count(*) AS c FROM "{name}"'
                    ).fetchone()["c"]
                except sqlite3.Error:
                    count = -1
                cols = [
                    r["name"]
                    for r in conn.execute(f'PRAGMA table_info("{name}")').fetchall()
                ]
                meta = TABLE_META.get(name, {})
                tables.append(
                    {
                        "name": name,
                        "title": meta.get("title", name),
                        "desc": meta.get("desc", ""),
                        "rows": count,
                        "columns": cols,
                    }
                )
        return {
            "db_path": self.db_path,
            "available": True,
            "size_bytes": os.path.getsize(self.db_path),
            "tables": tables,
        }

    def rows(
        self,
        table: str,
        limit: int = 50,
        offset: int = 0,
        query: str = "",
        order: str = "",
    ) -> dict:
        if not _TABLE_NAME_RE.match(table or ""):
            raise ValueError("非法表名")
        if not self.available:
            raise ValueError("记忆库尚未创建")
        derived = _DERIVED_SOURCES.get(table)
        with self._connect() as conn:
            exists = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                (table,),
            ).fetchone()
            if not exists:
                raise ValueError(f"表 {table} 不存在")

            base_cols = [
                {"name": r["name"], "type": r["type"] or "", "pk": bool(r["pk"])}
                for r in conn.execute(f'PRAGMA table_info("{table}")').fetchall()
            ]
            extra_cols = list(derived["columns"]) if derived else []
            cols = base_cols + extra_cols
            if derived and derived.get("display"):
                by_name = {c["name"]: c for c in cols}
                cols = [by_name[n] for n in derived["display"] if n in by_name]
            col_names = {c["name"] for c in base_cols}
            prefix = f"{derived['alias']}." if derived else ""

            params: list[object] = []
            search_terms: list[str] = []
            if query.strip():
                like = f"%{query.strip()}%"
                for c in base_cols:
                    if "TEXT" in (c["type"] or "").upper() or c["type"] in (
                        "",
                        "TIMESTAMP",
                    ):
                        search_terms.append(f'{prefix}"{c["name"]}" LIKE ?')
                        params.append(like)
                for extra in (derived or {}).get("search", []):
                    search_terms.append(extra)
                    params.append(like)
            where = (" WHERE " + " OR ".join(search_terms)) if search_terms else ""

            order_clause = ""
            order_name = (order or "").lstrip("-")
            if order_name:
                direction = "ASC" if order.startswith("-") else "DESC"
                if order_name in col_names:
                    order_clause = f' ORDER BY {prefix}"{order_name}" {direction}'
                elif order_name in {c["name"] for c in extra_cols}:
                    # 联查出来的列是输出别名，不能带表前缀
                    order_clause = f' ORDER BY "{order_name}" {direction}'
            if not order_clause:
                for candidate in _ORDER_FALLBACK:
                    if candidate in col_names:
                        order_clause = f' ORDER BY {prefix}"{candidate}" DESC'
                        break

            if derived:
                sql = (
                    f'{derived["select"]} {derived["from"]}{where}{order_clause}'
                    " LIMIT ? OFFSET ?"
                )
                count_sql = f'SELECT count(*) AS c {derived["from"]}{where}'
            else:
                sql = f'SELECT * FROM "{table}"{where}{order_clause} LIMIT ? OFFSET ?'
                count_sql = f'SELECT count(*) AS c FROM "{table}"{where}'

            total = conn.execute(count_sql, params).fetchone()["c"]
            cur = conn.execute(sql, [*params, limit, offset])
            items = [
                {k: _cell_value(row[k]) for k in row.keys()} for row in cur.fetchall()
            ]
        return {
            "table": table,
            "columns": cols,
            "rows": items,
            "total": total,
            "limit": limit,
            "offset": offset,
        }

    def traces(self, limit: int = 50) -> list[dict]:
        if not self.available:
            return []
        with self._connect() as conn:
            exists = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='retrieval_trace'"
            ).fetchone()
            if not exists:
                return []
            rows = conn.execute(
                "SELECT id, created_at, user_input, display_name, user_id, role, "
                "elapsed_ms, embedding_ok, embedding_dim, final_tags "
                "FROM retrieval_trace ORDER BY id DESC LIMIT ?",
                (max(1, min(limit, _MAX_LIMIT)),),
            ).fetchall()
        out = []
        for r in rows:
            try:
                final = json.loads(r["final_tags"] or "[]")
            except json.JSONDecodeError:
                final = []
            if not isinstance(final, list):
                final = []
            # 正常写入的是 dict（含 tag/content）；老数据或手写调试行可能是纯字符串，
            # 不能因为一行形状不对就让整个列表接口 500。
            tags = set()
            for item in final:
                tag = item.get("tag", "") if isinstance(item, dict) else str(item)
                if tag:
                    tags.add(tag)
            out.append(
                {
                    "id": r["id"],
                    "created_at": r["created_at"],
                    "user_input": r["user_input"],
                    "display_name": r["display_name"],
                    "user_id": r["user_id"],
                    "role": r["role"],
                    "elapsed_ms": r["elapsed_ms"],
                    "embedding_ok": bool(r["embedding_ok"]),
                    "embedding_dim": r["embedding_dim"],
                    "hit_count": len(final),
                    "channels": sorted(tags),
                }
            )
        return out

    def trace(self, trace_id: int) -> dict:
        if not self.available:
            raise ValueError("记忆库尚未创建")
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM retrieval_trace WHERE id = ?", (trace_id,)
            ).fetchone()
        if row is None:
            raise ValueError("该追踪记录不存在")
        data = {k: row[k] for k in row.keys()}
        for field in ("keywords", "channels", "fusion", "final_tags",
                      "graph_paths", "extra"):
            try:
                data[field] = json.loads(data.get(field) or "null")
            except json.JSONDecodeError:
                data[field] = None
        data["embedding_ok"] = bool(data.get("embedding_ok"))
        return data

    # ── 面向「看懂了没」的聚合接口 ────────────────────────────────
    #
    # 原始表接口回答的是「库里有什么」，这几个接口回答「它记住了什么、
    # 记得对不对、为什么这次没记住」。前端的主视图全部基于它们。

    @staticmethod
    def _has(conn: sqlite3.Connection, table: str) -> bool:
        return conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone() is not None

    def _count(self, conn: sqlite3.Connection, table: str) -> int:
        try:
            return conn.execute(f'SELECT count(*) AS c FROM "{table}"').fetchone()["c"]
        except sqlite3.Error:
            return 0

    def stats(self) -> dict:
        """总览：各类实体数量、最近活动、近 14 天曲线、蒸馏进度。"""
        out: dict = {
            "db_path": self.db_path,
            "available": self.available,
            "size_bytes": os.path.getsize(self.db_path) if self.available else 0,
            "counts": {},
            "moods": [],
            "daily": [],
            "last": {},
            "sessions": 0,
            "distill": {},
            "hints": self.hints,
        }
        if not self.available:
            return out

        with self._connect() as conn:
            out["counts"] = {
                "memories": self._count(conn, "cognitive_distill"),
                "entities": self._count(conn, "memory_graph_nodes"),
                "edges": self._count(conn, "memory_graph_edges"),
                "messages": self._count(conn, "chat_history"),
                "traces": self._count(conn, "retrieval_trace"),
                "identities": self._count(conn, "identity_memory"),
            }
            out["sessions"] = (
                conn.execute(
                    "SELECT count(DISTINCT session_id) AS c FROM chat_history"
                ).fetchone()["c"]
                if self._has(conn, "chat_history")
                else 0
            )

            def last_at(table: str, col: str) -> str | None:
                if not self._has(conn, table):
                    return None
                try:
                    row = conn.execute(
                        f'SELECT max("{col}") AS m FROM "{table}"'
                    ).fetchone()
                    return row["m"] if row else None
                except sqlite3.Error:
                    return None

            out["last"] = {
                "memory_at": last_at("cognitive_distill", "created_at"),
                "message_at": last_at("chat_history", "created_at"),
                "trace_at": last_at("retrieval_trace", "created_at"),
                "distill_at": last_at("session_state", "last_distill_at"),
            }

            # 情绪分布 —— 记忆条目上真正有区分度的一维
            if self._has(conn, "cognitive_distill"):
                rows = conn.execute(
                    "SELECT mood, count(*) AS n FROM cognitive_distill "
                    "WHERE mood IS NOT NULL AND mood <> '' "
                    "GROUP BY mood ORDER BY n DESC LIMIT 8"
                ).fetchall()
                out["moods"] = [{"mood": r["mood"], "n": r["n"]} for r in rows]

            out["daily"] = self._daily(conn)
            out["distill"] = self._distill_progress(conn)
        return out

    def _daily(self, conn: sqlite3.Connection) -> list[dict]:
        """近 14 天的对话消息数与新增记忆数（补零，保证曲线不断）。"""
        days = [(_dt.date.today() - _dt.timedelta(days=i)).isoformat()
                for i in range(13, -1, -1)]
        bucket = {d: {"d": d, "messages": 0, "memories": 0} for d in days}
        for table, key in (("chat_history", "messages"),
                           ("cognitive_distill", "memories")):
            if not self._has(conn, table):
                continue
            try:
                rows = conn.execute(
                    f"SELECT date(created_at) AS d, count(*) AS n FROM \"{table}\" "
                    "WHERE date(created_at) >= date('now', '-13 days') "
                    "GROUP BY d"
                ).fetchall()
            except sqlite3.Error:
                continue
            for r in rows:
                if r["d"] in bucket:
                    bucket[r["d"]][key] = r["n"]
        return [bucket[d] for d in days]

    def _distill_progress(self, conn: sqlite3.Connection) -> dict:
        """最近活跃会话离下一次蒸馏还差多远。

        蒸馏有两条并列条件：攒够 max_turns 轮，或距离上次蒸馏
        max_minutes 分钟（从没蒸过就算会话首次出现的时间）。
        两条都没到就不会蒸馏，所以这里两个都算出来。
        """
        info = {
            "max_turns": self.hints.get("distill_max_turns"),
            "max_minutes": self.hints.get("distill_max_minutes"),
            "session_id": None,
            "turn_count": 0,
            "last_distill_turn": 0,
            "gap": 0,
            "remaining": None,
            "ever_distilled": False,
            "first_seen_at": None,
            "elapsed_minutes": None,
            "wait_minutes": None,
        }
        if not self._has(conn, "session_state"):
            return info
        try:
            row = conn.execute(
                # 库里所有时间戳都是本地时间（应用侧用 datetime.now() 写），
                # 所以这里必须用 'now','localtime'，否则相差一个时区会算出负数。
                "SELECT session_id, turn_count, last_distill_turn, last_distill_at, "
                "       first_seen_at, "
                "       CAST((julianday('now', 'localtime') - julianday("
                "         coalesce(last_distill_at, first_seen_at, last_active))) "
                "         * 24 * 60 AS INTEGER) AS elapsed "
                "FROM session_state ORDER BY last_active DESC LIMIT 1"
            ).fetchone()
        except sqlite3.Error:
            return info
        if row is None:
            return info
        gap = max(0, (row["turn_count"] or 0) - (row["last_distill_turn"] or 0))
        info.update(
            session_id=row["session_id"],
            turn_count=row["turn_count"] or 0,
            last_distill_turn=row["last_distill_turn"] or 0,
            gap=gap,
            ever_distilled=row["last_distill_at"] is not None,
            first_seen_at=row["first_seen_at"],
            elapsed_minutes=max(0, int(row["elapsed"] or 0)),
        )
        if info["max_turns"]:
            info["remaining"] = max(0, int(info["max_turns"]) - gap)
        if info["max_minutes"] and info["elapsed_minutes"] is not None:
            info["wait_minutes"] = max(0, int(info["max_minutes"]) - info["elapsed_minutes"])
        return info

    def memories(
        self,
        query: str = "",
        order: str = "recent",
        mood: str = "",
        days: int = 0,
        limit: int = 30,
        offset: int = 0,
    ) -> dict:
        """记忆条目（cognitive_distill）—— 面板的主视图。"""
        if not self.available:
            raise ValueError("记忆库尚未创建")
        order_sql = {
            "importance": "importance DESC, created_at DESC",
            "oldest": "created_at ASC",
            "recent": "created_at DESC",
        }.get(order, "created_at DESC")

        where, params = [], []
        if query.strip():
            like = f"%{query.strip()}%"
            where.append("(content LIKE ? OR summary LIKE ? OR keylabel LIKE ? "
                         "OR entities LIKE ? OR mood LIKE ?)")
            params.extend([like] * 5)
        if mood.strip():
            where.append("mood = ?")
            params.append(mood.strip())
        if days and days > 0:
            where.append("date(created_at) >= date('now', ?)")
            params.append(f"-{int(days) - 1} days")
        clause = (" WHERE " + " AND ".join(where)) if where else ""

        with self._connect() as conn:
            if not self._has(conn, "cognitive_distill"):
                return {"items": [], "total": 0, "limit": limit, "offset": offset}
            total = conn.execute(
                f"SELECT count(*) AS c FROM cognitive_distill{clause}", params
            ).fetchone()["c"]
            rows = conn.execute(
                "SELECT id, user_id, session_id, content, keylabel, summary, mood, "
                "mood_intensity, importance, entities, turn_num, created_at, "
                "embedding_dim, expires_at, "
                "CASE WHEN embedding IS NULL THEN 0 ELSE 1 END AS has_vector "
                f"FROM cognitive_distill{clause} ORDER BY {order_sql} LIMIT ? OFFSET ?",
                [*params, limit, offset],
            ).fetchall()

        items = []
        for r in rows:
            try:
                ents = json.loads(r["entities"] or "[]")
            except json.JSONDecodeError:
                ents = []
            items.append(
                {
                    "id": r["id"],
                    "content": _cell_value(r["content"]),
                    "summary": r["summary"] or "",
                    "keylabel": r["keylabel"] or "",
                    "mood": r["mood"] or "",
                    "mood_intensity": r["mood_intensity"],
                    "importance": r["importance"],
                    "entities": ents if isinstance(ents, list) else [],
                    "session_id": r["session_id"] or "",
                    "user_id": r["user_id"] or "",
                    "turn_num": r["turn_num"],
                    "created_at": r["created_at"],
                    "embedding_dim": r["embedding_dim"],
                    "has_vector": bool(r["has_vector"]),
                    "expires_at": r["expires_at"],
                }
            )
        return {"items": items, "total": total, "limit": limit, "offset": offset}

    def graph(self, limit: int = 300) -> dict:
        """实体与关系 —— 只返回有连接价值的节点（按频次倒序）。"""
        out = {"nodes": [], "edges": [], "types": []}
        if not self.available:
            return out
        with self._connect() as conn:
            if not self._has(conn, "memory_graph_nodes"):
                return out
            nodes = conn.execute(
                "SELECT node_id, label, node_type, freq, last_seen, entity_type "
                "FROM memory_graph_nodes ORDER BY freq DESC, node_id LIMIT ?",
                (max(1, min(limit, _MAX_LIMIT)),),
            ).fetchall()
            edges = conn.execute(
                "SELECT e.from_node_id, n1.label AS from_label, e.to_node_id, "
                "       n2.label AS to_label, e.weight, e.encounter_count, "
                "       e.relation, e.last_seen "
                "FROM memory_graph_edges e "
                "LEFT JOIN memory_graph_nodes n1 ON n1.node_id = e.from_node_id "
                "LEFT JOIN memory_graph_nodes n2 ON n2.node_id = e.to_node_id "
                "ORDER BY e.weight DESC LIMIT ?",
                (max(1, min(limit, _MAX_LIMIT)),),
            ).fetchall()
            types = conn.execute(
                "SELECT node_type, count(*) AS n FROM memory_graph_nodes "
                "GROUP BY node_type ORDER BY n DESC"
            ).fetchall()
        out["nodes"] = [
            {
                "id": r["node_id"],
                "label": r["label"],
                "node_type": r["node_type"] or "",
                "entity_type": r["entity_type"] or "",
                "freq": r["freq"],
                "last_seen": r["last_seen"],
            }
            for r in nodes
        ]
        out["edges"] = [
            {
                "from": r["from_node_id"],
                "from_label": r["from_label"] or f"#{r['from_node_id']}",
                "to": r["to_node_id"],
                "to_label": r["to_label"] or f"#{r['to_node_id']}",
                "weight": r["weight"],
                "encounter_count": r["encounter_count"],
                "relation": r["relation"] or "",
                "last_seen": r["last_seen"],
            }
            for r in edges
        ]
        out["types"] = [{"type": r["node_type"] or "?", "n": r["n"]} for r in types]
        return out

    def sessions(self) -> list[dict]:
        """会话列表 —— 以 chat_history 为准，左连接蒸馏进度。"""
        if not self.available:
            return []
        with self._connect() as conn:
            if not self._has(conn, "chat_history"):
                return []
            rows = conn.execute(
                "SELECT h.session_id, count(*) AS messages, max(h.turn_num) AS last_turn, "
                "       max(h.created_at) AS last_at, "
                "       (SELECT user_id FROM chat_history t WHERE t.session_id = h.session_id "
                "        AND t.role = 'user' ORDER BY t.id DESC LIMIT 1) AS user_id, "
                "       (SELECT sender_name FROM chat_history t WHERE t.session_id = h.session_id "
                "        AND t.sender_name <> '' ORDER BY t.id DESC LIMIT 1) AS display_name, "
                "       s.turn_count, s.last_distill_turn, s.last_distill_at "
                "FROM chat_history h "
                "LEFT JOIN session_state s ON s.session_id = h.session_id "
                "GROUP BY h.session_id ORDER BY last_at DESC LIMIT ?",
                (_MAX_LIMIT,),
            ).fetchall()
        return [
            {
                "session_id": r["session_id"],
                "messages": r["messages"],
                "turn_count": r["turn_count"] if r["turn_count"] is not None
                else r["last_turn"],
                "last_distill_turn": r["last_distill_turn"] or 0,
                "last_distill_at": r["last_distill_at"],
                "last_at": r["last_at"],
                "user_id": r["user_id"] or "",
                "display_name": r["display_name"] or "",
            }
            for r in rows
        ]

    def dialogue(
        self, session_id: str = "", limit: int = 200, offset: int = 0
    ) -> dict:
        """按会话读取对话流（role 交替），供前端做聊天式展示。"""
        if not self.available:
            raise ValueError("记忆库尚未创建")
        with self._connect() as conn:
            if not self._has(conn, "chat_history"):
                return {"items": [], "total": 0, "session_id": session_id}
            if not session_id:
                row = conn.execute(
                    "SELECT session_id FROM chat_history ORDER BY id DESC LIMIT 1"
                ).fetchone()
                session_id = (row["session_id"] if row else "") or ""
            total = conn.execute(
                "SELECT count(*) AS c FROM chat_history WHERE session_id = ?",
                (session_id,),
            ).fetchone()["c"]
            rows = conn.execute(
                "SELECT id, role, content, created_at, turn_num, sender_name, mood, "
                "importance, user_id FROM chat_history WHERE session_id = ? "
                "ORDER BY id ASC LIMIT ? OFFSET ?",
                (session_id, limit, offset),
            ).fetchall()
        return {
            "session_id": session_id,
            "total": total,
            "items": [
                {
                    "id": r["id"],
                    "role": r["role"] or "",
                    "content": _cell_value(r["content"]),
                    "created_at": r["created_at"],
                    "turn_num": r["turn_num"],
                    "sender_name": r["sender_name"] or "",
                    "mood": r["mood"] or "",
                    "importance": r["importance"],
                    "user_id": r["user_id"] or "",
                }
                for r in rows
            ],
        }


class _Handler(BaseHTTPRequestHandler):
    server_version = "RCMS-WebUI/2.0"
    protocol_version = "HTTP/1.1"

    # 注入自 _Server
    store: _Store

    def log_message(self, fmt: str, *args: object) -> None:  # noqa: A003
        logger.debug("webui %s - %s", self.address_string(), fmt % args)

    # ── 响应工具 ────────────────────────────────────────────────

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _json(self, payload: object, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        self._send(status, body, "application/json; charset=utf-8")

    def _error(self, message: str, status: int = 400) -> None:
        self._json({"ok": False, "error": message}, status)

    def _static(self, name: str) -> None:
        safe = os.path.basename(name)
        path = os.path.join(_STATIC_DIR, safe)
        if not os.path.isfile(path):
            self._error("文件不存在", 404)
            return
        ext = os.path.splitext(safe)[1].lower()
        ctype = {
            ".html": "text/html; charset=utf-8",
            ".js": "application/javascript; charset=utf-8",
            ".css": "text/css; charset=utf-8",
            ".svg": "image/svg+xml",
        }.get(ext, "application/octet-stream")
        with open(path, "rb") as f:
            self._send(200, f.read(), ctype)

    # ── 路由 ────────────────────────────────────────────────────

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        qs = parse_qs(parsed.query)

        def q(name: str, default: str = "") -> str:
            return (qs.get(name) or [default])[0]

        try:
            if path in ("/", "/index.html"):
                self._static("index.html")
                return
            if path.startswith("/static/"):
                self._static(path[len("/static/"):])
                return
            if path == "/api/health":
                self._json({"ok": True, "db": self.store.db_path})
                return
            if path == "/api/overview":
                self._json({"ok": True, "data": self.store.overview()})
                return
            if path == "/api/stats":
                self._json({"ok": True, "data": self.store.stats()})
                return
            if path == "/api/memories":
                try:
                    limit = max(1, min(int(q("limit", "30")), _MAX_LIMIT))
                    offset = max(0, int(q("offset", "0")))
                    days = max(0, int(q("days", "0")))
                except ValueError:
                    self._error("limit/offset/days 必须是整数")
                    return
                self._json({"ok": True, "data": self.store.memories(
                    query=q("q"), order=q("order", "recent"), mood=q("mood"),
                    days=days, limit=limit, offset=offset,
                )})
                return
            if path == "/api/graph":
                try:
                    limit = max(1, min(int(q("limit", "300")), _MAX_LIMIT))
                except ValueError:
                    limit = 300
                self._json({"ok": True, "data": self.store.graph(limit)})
                return
            if path == "/api/sessions":
                self._json({"ok": True, "data": self.store.sessions()})
                return
            if path == "/api/dialogue":
                try:
                    limit = max(1, min(int(q("limit", "200")), _MAX_LIMIT))
                    offset = max(0, int(q("offset", "0")))
                except ValueError:
                    self._error("limit/offset 必须是整数")
                    return
                self._json({"ok": True, "data": self.store.dialogue(
                    session_id=q("session"), limit=limit, offset=offset,
                )})
                return
            if path == "/api/tables":
                self._json({"ok": True, "data": self.store.overview()["tables"]})
                return
            if path.startswith("/api/tables/"):
                name = path[len("/api/tables/"):]
                try:
                    limit = int(q("limit", "50"))
                    offset = int(q("offset", "0"))
                except ValueError:
                    self._error("limit/offset 必须是整数")
                    return
                limit = max(1, min(limit, _MAX_LIMIT))
                offset = max(0, offset)
                data = self.store.rows(
                    name, limit=limit, offset=offset,
                    query=q("q"), order=q("order"),
                )
                self._json({"ok": True, "data": data})
                return
            if path == "/api/traces":
                try:
                    limit = int(q("limit", "50"))
                except ValueError:
                    limit = 50
                self._json({"ok": True, "data": self.store.traces(limit)})
                return
            if path.startswith("/api/traces/"):
                raw = path[len("/api/traces/"):]
                try:
                    trace_id = int(raw)
                except ValueError:
                    self._error("追踪 ID 必须是整数")
                    return
                self._json({"ok": True, "data": self.store.trace(trace_id)})
                return
            self._error("未找到该路由", 404)
        except ValueError as exc:
            self._error(str(exc))
        except sqlite3.Error as exc:
            logger.warning("webui 查询失败: %s", exc)
            self._error(f"数据库查询失败: {exc}", 500)
        except Exception as exc:  # noqa: BLE001
            logger.exception("webui 处理异常: %s", exc)
            self._error("服务内部错误", 500)


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class RcmsWebServer:
    """插件自带的只读观测站。"""

    def __init__(
        self,
        db_path: str,
        host: str = "127.0.0.1",
        port: int = 8765,
        hints: dict | None = None,
    ):
        self.db_path = db_path
        self.host = host or "127.0.0.1"
        self.port = int(port)
        self.hints = hints or {}
        self._httpd: _Server | None = None
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        if self._httpd is None:
            return ""
        host, port = self._httpd.server_address[:2]
        shown = "127.0.0.1" if host in ("0.0.0.0", "") else host
        return f"http://{shown}:{port}/"

    def start(self) -> bool:
        """启动服务；端口被占用时自动顺延。返回是否成功。"""
        if self._httpd is not None:
            return True
        handler = type(
            "_BoundHandler",
            (_Handler,),
            {"store": _Store(self.db_path, self.hints)},
        )
        last_error: Exception | None = None
        for candidate in range(self.port, self.port + 10):
            try:
                httpd = _Server((self.host, candidate), handler)
            except OSError as exc:
                last_error = exc
                continue
            self._httpd = httpd
            self.port = candidate
            self._thread = threading.Thread(
                target=httpd.serve_forever,
                name="rcms-webui",
                daemon=True,
            )
            self._thread.start()
            logger.info("RCMS 观测站已启动: %s", self.url)
            return True
        logger.warning(
            "RCMS 观测站启动失败（%s:%d 起 10 个端口均不可用）: %s",
            self.host,
            self.port,
            last_error,
        )
        return False

    def stop(self) -> None:
        httpd = self._httpd
        if httpd is None:
            return
        self._httpd = None
        try:
            httpd.shutdown()
            httpd.server_close()
        except Exception as exc:  # noqa: BLE001
            logger.debug("关闭观测站时出错: %s", exc)
        thread = self._thread
        self._thread = None
        if thread is not None and thread.is_alive():
            thread.join(timeout=3.0)
        logger.info("RCMS 观测站已停止")

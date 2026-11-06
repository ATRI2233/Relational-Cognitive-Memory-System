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
    "shared_context": {"title": "共享语境", "desc": "双方共享的约定、共识类信息。"},
    "user_mappings": {"title": "用户映射", "desc": "昵称/别名 → 用户 ID 的映射。"},
    "relationship_arc": {"title": "关系弧", "desc": "关系阶段与走向的时间线记录。"},
    "emotional_trace": {"title": "情绪轨迹", "desc": "按时间记录的情绪变化。"},
    "entity_relations": {"title": "实体关系", "desc": "旧版三元组关系表，新版本用图谱边。"},
    "memory_links": {"title": "记忆关联", "desc": "记忆条目之间的直接关联。"},
    "embedding_rebuild_queue": {
        "title": "向量重建队列",
        "desc": "待重新计算向量的记忆（换 Embedding 模型后会入队）。",
    },
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

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path

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
                    "channels": sorted({i.get("tag", "") for i in final}),
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


class _Handler(BaseHTTPRequestHandler):
    server_version = "RCMS-WebUI/1.0"
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

    def __init__(self, db_path: str, host: str = "127.0.0.1", port: int = 8765):
        self.db_path = db_path
        self.host = host or "127.0.0.1"
        self.port = int(port)
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
        handler = type("_BoundHandler", (_Handler,), {"store": _Store(self.db_path)})
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

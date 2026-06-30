"""
SQLite 记忆图谱仓储实现
=========================

实现 IGraphRepository Protocol 的全部 8 个方法。
仅操作 memory_graph_nodes / memory_graph_edges 两张表。

依赖：
  - sqlite3.Connection — 数据库连接
  - domain.ports.clock.IClock — 可注入的时间源
  - domain.entities.graph.GraphNode / GraphEdge / DiffusionResult — 领域实体/值对象
"""

from __future__ import annotations

import logging
import re
import sqlite3
from collections import deque
from datetime import datetime
from typing import Dict, List, Set, Tuple

from domain.entities.graph import DiffusionResult, GraphEdge, GraphNode
from domain.ports.clock import IClock
from domain.ports.graph_repo import IGraphRepository

logger = logging.getLogger("rcms")

# ── 图相关常量 ──────────────────────────────────────────────────

_OPPOSITE_RELATIONS: Dict[str, str] = {
    "喜欢": "讨厌",
    "讨厌": "喜欢",
    "使用": "放弃",
    "放弃": "使用",
    "朋友": "敌人",
    "敌人": "朋友",
}

_GRAPH_ACTIVATION_DECAY = 0.5
_TIMESTAMP_FMT = "%Y-%m-%d %H:%M:%S"
_CHINESE_RANGE = "一-鿿"


class SQLiteGraphRepository(IGraphRepository):
    """SQLite 记忆图谱仓储。

    实现 IGraphRepository Protocol，负责 memory_graph_nodes 和
    memory_graph_edges 两张表的 CRUD、BFS 扩散激活、连通路径检索
    以及衰减维护。

    Args:
        conn: SQLite 数据库连接
        clock: 可注入的时间源
    """

    def __init__(self, conn: sqlite3.Connection, clock: IClock) -> None:
        self._conn = conn
        self._clock = clock

    # ── 内部辅助方法 ──────────────────────────────────────────────

    def _now_str(self) -> str:
        """返回格式化的当前时间字符串。"""
        return self._clock.strftime(_TIMESTAMP_FMT)

    @classmethod
    def _parse_dt(cls, val: object) -> datetime | None:
        """将数据库返回的时间戳值解析为 datetime。

        SQLite 无原生 datetime 类型，值可能为字符串或已由
        connection 工厂转换为 datetime 对象。
        对于 None 或无法解析的值返回 None。

        Args:
            val: 数据库返回的时间戳值

        Returns:
            datetime 对象，或 None
        """
        if val is None:
            return None
        if isinstance(val, datetime):
            return val
        if isinstance(val, str):
            for fmt in (_TIMESTAMP_FMT, "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
                try:
                    return datetime.strptime(val, fmt)
                except ValueError:
                    continue
        return None

    @classmethod
    def _row_to_graph_node(cls, row: sqlite3.Row) -> GraphNode:
        """将 memory_graph_nodes 表的一行转换为 GraphNode 实体。

        Args:
            row: sqlite3.Row 查询结果

        Returns:
            构造完成的 GraphNode 实体
        """
        return GraphNode(
            node_id=row[0],
            user_id=row[1],
            label=row[2],
            freq=row[3] if row[3] is not None else 1,
            last_seen=cls._parse_dt(row[4]),
            entity_type=row[5] or "auto",
        )

    @classmethod
    def _row_to_graph_edge(cls, row: sqlite3.Row) -> GraphEdge:
        """将 memory_graph_edges 表的一行转换为 GraphEdge 实体。

        Args:
            row: sqlite3.Row 查询结果

        Returns:
            构造完成的 GraphEdge 实体
        """
        return GraphEdge(
            from_node_id=row[0],
            to_node_id=row[1],
            weight=row[2] if row[2] is not None else 1.0,
            encounter_count=row[3] if row[3] is not None else 1,
            relation=row[4] or "",
            last_seen=cls._parse_dt(row[5]),
            created_at=cls._parse_dt(row[6]) if row[6] else None,
        )

    @staticmethod
    def _clean_label(label: str) -> str:
        """清洗标签：去掉前导/末尾非文字字符。

        Args:
            label: 原始标签字符串

        Returns:
            清洗后的标签字符串（若为空则返回空字符串）
        """
        cleaned = re.sub(
            rf'^[^a-zA-Z0-9{_CHINESE_RANGE}]+|[^a-zA-Z0-9{_CHINESE_RANGE}]+$',
            "",
            label,
        )
        return cleaned

    # ── IGraphRepository 接口实现 ────────────────────────────────

    async def upsert_node(
        self, user_id: str, label: str, entity_type: str = "auto"
    ) -> int:
        """插入或更新图节点。

        清洗 label（去除非文字字符），如果 label 已存在则 freq +1、
        更新 last_seen 和 entity_type，否则创建新节点。
        使用 SELECT-then-INSERT/UPDATE 模式，兼容无 (user_id,label) UNIQUE 约束的旧表。

        Args:
            user_id: 用户标识
            label: 节点标签（关键词/实体名）
            entity_type: 实体类型，默认 "auto"

        Returns:
            节点的 node_id（新插入或已存在的），清洗后为空则返回 -1
        """
        label = self._clean_label(label)
        if not label:
            logger.warning("Graph: skip empty label after cleaning user=%s", user_id)
            return -1

        now_str = self._now_str()
        try:
            row = self._conn.execute(
                "SELECT node_id, entity_type FROM memory_graph_nodes "
                "WHERE user_id = ? AND label = ?",
                (user_id, label),
            ).fetchone()

            if row:
                old_type = row[1] or "auto"
                new_type = entity_type if entity_type != "auto" else old_type
                self._conn.execute(
                    "UPDATE memory_graph_nodes SET freq = freq + 1, last_seen = ?, entity_type = ? WHERE node_id = ?",
                    (now_str, new_type, row[0]),
                )
                self._conn.commit()
                return row[0]

            cursor = self._conn.execute(
                "INSERT INTO memory_graph_nodes (user_id, label, entity_type, freq, last_seen) VALUES (?, ?, ?, 1, ?)",
                (user_id, label, entity_type, now_str),
            )
            self._conn.commit()
            return cursor.lastrowid
        except sqlite3.Error as e:
            self._conn.rollback()
            logger.error(
                "upsert_node 失败 user=%s label=%s: %s", user_id, label, e
            )
            raise

    async def upsert_edge(
        self, from_id: int, to_id: int, relation: str = ""
    ) -> None:
        """插入或更新图边。

        若 from_id == to_id 则跳过。
        若存在对立关系则先删除旧边。
        边已存在则 weight +0.5、encounter_count +1、更新 last_seen；
        不存在则创建新边（weight=1.0, encounter_count=1）。

        Args:
            from_id: 源节点 ID
            to_id: 目标节点 ID
            relation: 关系描述，空字符串表示共现关系
        """
        if from_id == to_id:
            return

        now_str = self._now_str()
        try:
            # ── 矛盾检测 ──
            opposite = _OPPOSITE_RELATIONS.get(relation)
            if opposite:
                conflict = self._conn.execute(
                    "SELECT from_node_id, to_node_id, relation "
                    "FROM memory_graph_edges "
                    "WHERE ((from_node_id = ? AND to_node_id = ?) OR "
                    "       (from_node_id = ? AND to_node_id = ?)) "
                    "AND relation = ?",
                    (from_id, to_id, to_id, from_id, opposite),
                ).fetchone()
                if conflict:
                    self._conn.execute(
                        "DELETE FROM memory_graph_edges "
                        "WHERE from_node_id = ? AND to_node_id = ?",
                        (conflict[0], conflict[1]),
                    )
                    logger.warning(
                        "Graph: 矛盾关系替换 from=%d to=%d %s -> %s",
                        from_id,
                        to_id,
                        conflict[2],
                        relation,
                    )

            # ── 边 upsert ──
            existing = self._conn.execute(
                "SELECT weight FROM memory_graph_edges "
                "WHERE from_node_id = ? AND to_node_id = ?",
                (from_id, to_id),
            ).fetchone()

            if existing:
                self._conn.execute(
                    "UPDATE memory_graph_edges SET "
                    "weight = weight + 0.5, "
                    "encounter_count = encounter_count + 1, "
                    "last_seen = ?, "
                    "relation = CASE WHEN ? != '' THEN ? ELSE relation END "
                    "WHERE from_node_id = ? AND to_node_id = ?",
                    (now_str, relation, relation, from_id, to_id),
                )
            else:
                self._conn.execute(
                    "INSERT INTO memory_graph_edges "
                    "(from_node_id, to_node_id, weight, encounter_count, "
                    " last_seen, relation, created_at) "
                    "VALUES (?, ?, 1.0, 1, ?, ?, ?)",
                    (from_id, to_id, now_str, relation, now_str),
                )

            self._conn.commit()
        except sqlite3.Error as e:
            self._conn.rollback()
            logger.error(
                "upsert_edge 失败 from=%d to=%d: %s", from_id, to_id, e
            )
            raise

    async def bfs_diffuse(
        self, user_id: str, seed_ids: List[int], depth: int = 2
    ) -> List[DiffusionResult]:
        """BFS 图扩散激活。

        以 seed_ids 为起点按 BFS 逐层扩散，激活分数 = 边权 *
        decay**(当前层数+1)。无 relation 的共现边降权为 0.1 倍。
        边的时间衰减按 (0.95 ** 距今天数) 计算，保底 0.3。

        Args:
            user_id: 用户标识（用于日志）
            seed_ids: 种子节点 ID 列表
            depth: BFS 最大深度，默认 2

        Returns:
            按激活分数降序排列的 DiffusionResult 列表
        """
        if not seed_ids:
            return []

        now_dt = self._clock.now()
        try:
            visited: Set[int] = set()
            activation_map: Dict[int, float] = {}

            for nid in seed_ids:
                activation_map[nid] = 1.0
                visited.add(nid)

            # queue: (node_id, current_depth)
            queue: deque[Tuple[int, int]] = deque([(nid, 0) for nid in seed_ids])

            while queue:
                cid, cur_depth = queue.popleft()
                if cur_depth >= depth:
                    continue

                edges = self._conn.execute(
                    "SELECT from_node_id, to_node_id, weight, relation, last_seen "
                    "FROM memory_graph_edges "
                    "WHERE from_node_id = ? OR to_node_id = ?",
                    (cid, cid),
                ).fetchall()

                for frm, to, w, relation, last_seen in edges:
                    # 时间衰减
                    if last_seen is not None:
                        try:
                            if isinstance(last_seen, str):
                                last_dt = datetime.strptime(
                                    str(last_seen)[:19], _TIMESTAMP_FMT
                                )
                            else:
                                last_dt = last_seen  # type: ignore[assignment]
                            days = (now_dt - last_dt).days
                            w = max(w * (0.95 ** days), 0.3)
                        except (ValueError, TypeError):
                            pass

                    # 无 relation 的共现边严重降权
                    w = w * (1.0 if relation else 0.1)

                    nid = frm if frm != cid else to
                    decayed = w * (_GRAPH_ACTIVATION_DECAY ** (cur_depth + 1))

                    if nid in visited:
                        activation_map[nid] += decayed
                        continue

                    visited.add(nid)
                    activation_map[nid] = decayed
                    queue.append((nid, cur_depth + 1))

            sorted_nodes = sorted(
                activation_map.items(), key=lambda x: -x[1]
            )

            results: List[DiffusionResult] = []
            for nid, score in sorted_nodes:
                row = self._conn.execute(
                    "SELECT label FROM memory_graph_nodes WHERE node_id = ?",
                    (nid,),
                ).fetchone()
                if row:
                    results.append(DiffusionResult(label=row[0], score=score))

            logger.info(
                "GraphDiffusion: seeds=%d depth=%d returned=%d user=%s",
                len(seed_ids),
                depth,
                len(results),
                user_id,
            )
            return results
        except sqlite3.Error as e:
            logger.error(
                "bfs_diffuse 失败 user=%s: %s", user_id, e
            )
            raise

    async def get_edges_by_node(self, node_id: int) -> List[GraphEdge]:
        """获取与指定节点相连的所有边。

        Args:
            node_id: 节点 ID

        Returns:
            GraphEdge 列表（包括出边和入边）
        """
        try:
            rows = self._conn.execute(
                "SELECT from_node_id, to_node_id, weight, encounter_count, "
                "       relation, last_seen, created_at "
                "FROM memory_graph_edges "
                "WHERE from_node_id = ? OR to_node_id = ?",
                (node_id, node_id),
            ).fetchall()
            return [self._row_to_graph_edge(r) for r in rows]
        except sqlite3.Error as e:
            logger.error(
                "get_edges_by_node 失败 node_id=%d: %s", node_id, e
            )
            raise

    async def get_nodes_by_user(self, user_id: str) -> List[GraphNode]:
        """获取用户的所有图节点。

        Args:
            user_id: 用户标识

        Returns:
            该用户的所有 GraphNode 列表
        """
        try:
            rows = self._conn.execute(
                "SELECT node_id, user_id, label, freq, last_seen, entity_type "
                "FROM memory_graph_nodes WHERE user_id = ?",
                (user_id,),
            ).fetchall()
            return [self._row_to_graph_node(r) for r in rows]
        except sqlite3.Error as e:
            logger.error(
                "get_nodes_by_user 失败 user=%s: %s", user_id, e
            )
            raise

    async def maintain(self, user_id: str) -> None:
        """执行图衰减与清理维护。

        低频边快衰减，高频边慢衰减，语义边额外保护。
        衰减率 = MIN(0.80 + 0.15 * encounter_count/20 + 是否语义边*0.05, 0.95)。
        权重低于 0.4 的边被删除，无关联边的孤立节点被删除。

        Args:
            user_id: 用户标识
        """
        try:
            self._conn.execute(
                """
                UPDATE memory_graph_edges SET weight = ROUND(weight * MIN(
                    0.80 + 0.15 * CAST(MIN(encounter_count, 20) AS REAL) / 20.0
                    + CASE WHEN relation != '' THEN 0.05 ELSE 0.0 END,
                    0.95
                ), 2)
                WHERE from_node_id IN (
                    SELECT node_id FROM memory_graph_nodes WHERE user_id = ?
                )
                """,
                (user_id,),
            )

            dead_edges = self._conn.execute(
                "DELETE FROM memory_graph_edges WHERE weight < 0.4 "
                "AND (from_node_id IN (SELECT node_id FROM memory_graph_nodes WHERE user_id = ?) "
                "OR to_node_id IN (SELECT node_id FROM memory_graph_nodes WHERE user_id = ?))",
                (user_id, user_id),
            ).rowcount

            orphan_nodes = self._conn.execute(
                """
                DELETE FROM memory_graph_nodes
                WHERE user_id = ? AND node_id NOT IN (
                    SELECT from_node_id FROM memory_graph_edges
                    UNION
                    SELECT to_node_id FROM memory_graph_edges
                )
                """,
                (user_id,),
            ).rowcount

            self._conn.commit()

            if dead_edges or orphan_nodes:
                logger.info(
                    "Graph: 图维护 user=%s deleted_edges=%d orphan_nodes=%d",
                    user_id,
                    dead_edges,
                    orphan_nodes,
                )
        except sqlite3.Error as e:
            self._conn.rollback()
            logger.error("maintain 失败 user=%s: %s", user_id, e)
            raise

    async def search_nodes(
        self, user_id: str, keyword: str
    ) -> List[GraphNode]:
        """按关键词模糊搜索图节点。

        Args:
            user_id: 用户标识
            keyword: 搜索关键词（LIKE 模糊匹配）

        Returns:
            匹配的 GraphNode 列表
        """
        try:
            rows = self._conn.execute(
                "SELECT node_id, user_id, label, freq, last_seen, entity_type "
                "FROM memory_graph_nodes "
                "WHERE user_id = ? AND label LIKE ?",
                (user_id, f"%{keyword}%"),
            ).fetchall()
            return [self._row_to_graph_node(r) for r in rows]
        except sqlite3.Error as e:
            logger.error(
                "search_nodes 失败 user=%s keyword=%s: %s",
                user_id,
                keyword,
                e,
            )
            raise

    async def get_chain_paths(
        self, user_id: str, labels: List[str]
    ) -> List[str]:
        """获取节点标签之间的连通路径。

        对节点标签集合查询之间的语义边（relation != ''），
        DFS 找最长链，返回最多 3 条路径描述字符串。

        Args:
            user_id: 用户标识
            labels: 节点标签列表

        Returns:
            路径描述字符串列表，最多 3 条，格式如 "A [关系] B -> B [关系] C"
        """
        if not labels:
            return []

        label_set = set(labels)
        try:
            ph = ",".join("?" * len(label_set))
            params = list(label_set) + list(label_set) + [user_id, user_id]
            rows = self._conn.execute(
                f"""
                SELECT n1.label AS a, n2.label AS b, e.relation
                FROM memory_graph_edges e
                JOIN memory_graph_nodes n1 ON e.from_node_id = n1.node_id
                JOIN memory_graph_nodes n2 ON e.to_node_id = n2.node_id
                WHERE n1.label IN ({ph}) AND n2.label IN ({ph})
                  AND n1.user_id = ? AND n2.user_id = ?
                  AND e.relation != ''
                """,
                params,
            ).fetchall()
        except sqlite3.Error as e:
            logger.error(
                "get_chain_paths 查询失败 user=%s: %s", user_id, e
            )
            raise

        # 构建邻接表: label -> [(relation, target_label)]
        adj: Dict[str, List[Tuple[str, str]]] = {}
        for a, b, relation in rows:
            adj.setdefault(a, []).append((relation, b))

        if not adj:
            return []

        # DFS 找以 start 为起点的最长链
        def _longest_from(start: str, visited: Set[str]) -> list:
            best: list = [start]
            for rel, nxt in adj.get(start, []):
                if nxt in visited:
                    continue
                path = _longest_from(nxt, visited | {nxt})
                if len(path) + 1 > len(best):
                    best = [start] + [(rel, nxt)] + path[1:]
            return best

        chains: List[str] = []
        used_labels: Set[str] = set()

        for label in sorted(
            label_set, key=lambda x: -len(adj.get(x, []))
        ):
            if label in used_labels:
                continue

            path = _longest_from(label, {label})
            if len(path) >= 2:
                segments: List[str] = []
                cur = path[0]
                for i in range(1, len(path)):
                    if isinstance(path[i], tuple):
                        rel, nxt = path[i]
                        segments.append(f"{cur} [{rel}] {nxt}")
                        cur = nxt
                if len(segments) >= 2:
                    chains.append(" → ".join(segments))
                    for item in path:
                        if not isinstance(item, tuple):
                            used_labels.add(item)
                    if len(chains) >= 3:
                        break

        if not chains:
            # 退化为单段边
            for a, b, relation in rows:
                chains.append(f"{a} [{relation}] {b}")
                if len(chains) >= 3:
                    break

        logger.info(
            "GraphPaths: labels=%d paths=%d user=%s",
            len(labels),
            len(chains),
            user_id,
        )
        return chains[:3]

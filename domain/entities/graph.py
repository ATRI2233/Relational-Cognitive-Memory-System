"""
领域实体 — Graph（图谱）及其值对象
============================================

映射数据库表：
  - memory_graph_nodes → GraphNode
  - memory_graph_edges → GraphEdge

值对象 EntityRelation / DiffusionResult 用于 Repository 接口的返回类型。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


# ── 实体（Entity）：有身份可变 ──


@dataclass
class GraphNode:
    """图谱节点实体。

    对应 `memory_graph_nodes` 表，表示一个关键词/实体节点。
    node_id 为数据库自增主键，作为实体标识。
    """

    node_id: int
    user_id: str
    label: str
    freq: int = 1
    last_seen: Optional[datetime] = None
    entity_type: str = "auto"


@dataclass
class GraphEdge:
    """图谱边实体。

    对应 `memory_graph_edges` 表，表示两个节点间的关系。
    (from_node_id, to_node_id) 构成联合主键。
    """

    from_node_id: int
    to_node_id: int
    weight: float = 1.0
    encounter_count: int = 1
    relation: str = ""
    last_seen: Optional[datetime] = None
    created_at: Optional[datetime] = None


# ── 值对象（Value Objects）：不可变 ──


@dataclass(frozen=True)
class EntityRelation:
    """实体关系值对象。

    用于图谱检索和实体分析的输出，表示两个实体间的语义关系。
    """

    source: str
    target: str
    relation: str
    weight: float = 1.0


@dataclass(frozen=True)
class DiffusionResult:
    """图扩散激活结果值对象。

    对应 `_graph_activation_diffusion` 方法的输出，
    表示从种子节点 BFS 扩散到的节点及其激活分数。
    """

    label: str
    score: float
    depth: int = 0

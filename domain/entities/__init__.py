"""
RCMS 领域实体 — 值对象与实体定义

核心领域类型：
  - memory.py:   Memory（记忆实体）、MemoryId / UserId / SessionId / Importance / Mood
  - session.py:  Session（会话实体）、TurnRecord（轮次记录）
  - graph.py:    GraphNode / GraphEdge（图谱实体）、EntityRelation / DiffusionResult
  - identity.py: Identity（身份实体）、Trait / Preferences / Boundary
"""

from domain.entities.graph import (
    DiffusionResult,
    EntityRelation,
    GraphEdge,
    GraphNode,
)
from domain.entities.identity import (
    Boundary,
    Identity,
    Preferences,
    Trait,
)
from domain.entities.memory import (
    Importance,
    Memory,
    MemoryId,
    Mood,
    SessionId,
    UserId,
)
from domain.entities.session import (
    Session,
    TurnRecord,
)

__all__ = [
    # memory
    "Memory",
    "MemoryId",
    "UserId",
    "SessionId",
    "Importance",
    "Mood",
    # session
    "Session",
    "TurnRecord",
    # graph
    "GraphNode",
    "GraphEdge",
    "EntityRelation",
    "DiffusionResult",
    # identity
    "Identity",
    "Trait",
    "Preferences",
    "Boundary",
]

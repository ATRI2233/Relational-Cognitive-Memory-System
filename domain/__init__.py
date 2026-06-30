"""
领域层 — 系统的核心抽象与业务规则，不依赖外部框架或基础设施实现。

包含端口（Port，即协议接口）和领域模型。
"""

from domain.entities.identity import Boundary, Identity, Preferences, Trait
from domain.entities.graph import DiffusionResult, EntityRelation, GraphEdge, GraphNode

__all__ = [
    # Identity
    "Boundary",
    "Identity",
    "Preferences",
    "Trait",
    # Graph
    "DiffusionResult",
    "EntityRelation",
    "GraphEdge",
    "GraphNode",
]

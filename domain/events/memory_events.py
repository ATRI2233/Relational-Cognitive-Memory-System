"""
记忆相关的领域事件。

每个事件对应 RCMS 管线中的一个关键节点：
  - TurnSaved:      一轮对话写入后触发 → PostUpdateHandler / DelayedEmbedHandler
  - MemoryDistilled: LLM 蒸馏分析完成后触发 → GraphMaintenanceHandler
  - EmbeddingDone:  向量化完成后触发 → 缓存刷新
  - EmbeddingRebuildNeeded: 维度不匹配时触发 → 重建队列
  - MemoryExpired:  过期记忆清理后触发 → 可选通知
  - GraphUpdated:   图谱节点/边更新后触发 → 可选旁路
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class TurnSaved:
    """一轮对话已保存 — save_turn 完成后发布

    携带完整的对话轮次信息，供后处理 Handler 消费：
      - PostUpdateHandler：更新 last_active、检查 dangling 过期
      - DelayedEmbedHandler：为新条目生成向量
      - DistillCheckerHandler：检查是否满足蒸馏条件
    """
    user_id: str
    session_id: str
    turn_number: int
    user_input: str
    reply: str
    occurred_at: datetime
    sender_name: str = ""


@dataclass(frozen=True)
class MemoryDistilled:
    """记忆蒸馏分析完成 — 一次蒸馏分析（LLM 调用+写入）完成

    携带蒸馏出的记忆内容和 9 维分析结果，供 GraphMaintenanceHandler
    和 EmbeddingHandler 消费。
    """
    user_id: str
    session_id: str
    memory_id: int
    content: str
    keylabel: str
    importance: float
    mood: str
    mood_intensity: float
    occurred_at: datetime
    raw_analysis: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EmbeddingDone:
    """记忆向量化完成 — 一条记忆的 embedding 计算并写入完毕

    触发 EmbeddingCacheHandler 刷新内存中的 _emb_cache。
    """
    user_id: str
    record_id: int
    embedding_dim: int
    occurred_at: datetime
    source: str = ""


@dataclass(frozen=True)
class EmbeddingRebuildNeeded:
    """向量需要重建 — 模型变更导致维度不匹配

    触发 RebuildQueueHandler 将记录加入 embedding_rebuild_queue。
    """
    user_id: str
    record_id: int
    occurred_at: datetime
    reason: str = "dim_mismatch"


@dataclass(frozen=True)
class MemoryExpired:
    """记忆已过期并被清理 — 蒸馏时触发的过期清理

    用于可选的过期通知或统计数据更新。
    """
    user_id: str
    expired_count: int
    occurred_at: datetime


@dataclass(frozen=True)
class GraphUpdated:
    """图谱已更新 — 节点或边被添加/修改

    在 _upsert_graph_node / _upsert_graph_edge 后发布。
    用于可选的图缓存刷新或旁路分析。
    """
    user_id: str
    occurred_at: datetime
    node_count: int = 0
    edge_count: int = 0
    operation: str = "upsert"

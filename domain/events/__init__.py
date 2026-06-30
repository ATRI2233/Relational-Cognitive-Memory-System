"""
领域事件模块 — 记录架构中发生的具有业务意义的事件。

领域事件是 DDD 的核心概念，用于：
  1. 解耦 Use Case 与后处理逻辑（PostUpdateHandler / EmbeddingHandler 等通过事件驱动）
  2. 提供审计轨迹（可持久化事件流用于重建状态）
  3. 支持异步消息模式（fire-and-forget）

所有事件类均为冻结 dataclass，保证不可变性。
事件命名采用过去时（Distilled / Saved / Updated），表示已发生的事实。
"""

from domain.events.memory_events import (
    MemoryDistilled,
    TurnSaved,
    EmbeddingDone,
    EmbeddingRebuildNeeded,
    MemoryExpired,
    GraphUpdated,
)

__all__ = [
    "MemoryDistilled",
    "TurnSaved",
    "EmbeddingDone",
    "EmbeddingRebuildNeeded",
    "MemoryExpired",
    "GraphUpdated",
]

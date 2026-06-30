"""
DelayedEmbedHandler — TurnSaved 事件处理器。

在每轮对话保存后，查找最新一条未向量化的记忆并生成 embedding。
对应 plugins/rcms-astrbot/main.py 中 _delayed_embed 方法的逻辑。
"""
from __future__ import annotations

from typing import Callable, Protocol, runtime_checkable

from domain.entities.memory import MemoryId, UserId
from domain.events.memory_events import EmbeddingDone, TurnSaved
from domain.ports.repositories import IMemoryRepository


@runtime_checkable
class IEmbeddingService(Protocol):
    """Embedding 服务接口"""
    async def embed(self, text: str) -> list[float]: ...


class DelayedEmbedHandler:
    """TurnSaved 事件处理器 — 延迟向量化

    每轮对话后查找最近一条未向量化的记忆并生成 embedding，
    然后发布 EmbeddingDone 事件。
    """

    def __init__(self,
                 memory_repo: IMemoryRepository,
                 embedding: IEmbeddingService,
                 event_bus=None,
                 min_text_length: int = 15):
        self._memory_repo = memory_repo
        self._embedding = embedding
        self._event_bus = event_bus
        self._min_text_length = min_text_length

    async def handle(self, event: TurnSaved) -> None:
        """处理 TurnSaved 事件

        只对长度超过 min_text_length 的输入生成向量。

        Args:
            event: TurnSaved 事件实例
        """
        if len(event.user_input) <= self._min_text_length:
            return

        memories = await self._memory_repo.get_unembedded(
            UserId(event.user_id), limit=1
        )
        if not memories:
            return

        memory = memories[0]
        vec = await self._embedding.embed(memory.content[:512])
        if vec:
            await self._memory_repo.store_embedding(memory.memory_id, vec)
            if self._event_bus:
                await self._event_bus.publish(EmbeddingDone(
                    user_id=event.user_id,
                    record_id=memory.memory_id.value,
                    embedding_dim=len(vec),
                    occurred_at=event.occurred_at,
                    source="delayed",
                ))

    @staticmethod
    def register(event_bus, handler: DelayedEmbedHandler) -> None:
        """在 EventBus 上注册此处理器"""
        event_bus.register(TurnSaved, handler.handle)

"""
EventBus — 事件总线。

发布/订阅模式，解耦 Use Case 与后处理逻辑：
  - TurnSaved → PostUpdateHandler / DelayedEmbedHandler / DistillCheckerHandler
  - MemoryDistilled → GraphMaintenanceHandler
  - EmbeddingDone → EmbeddingCacheHandler
  - EmbeddingRebuildNeeded → RebuildQueueHandler
"""
from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from collections.abc import Callable, Coroutine
from typing import Any


class EventBus:
    """事件总线 — 注册、注销、发布事件"""

    def __init__(self):
        self._handlers: dict[type, list[Callable[..., Coroutine[Any, Any, None]]]] = defaultdict(list)

    def register(self, event_type: type, handler: Callable) -> None:
        """注册事件处理器

        Args:
            event_type: 领域事件类（如 TurnSaved）
            handler: 异步回调函数，接受事件实例作为唯一参数
        """
        self._handlers[event_type].append(handler)

    def unregister(self, event_type: type, handler: Callable) -> None:
        """注销事件处理器"""
        handlers = self._handlers.get(event_type, [])
        if handler in handlers:
            handlers.remove(handler)

    async def publish(self, event: object) -> None:
        """发布事件，同步调用所有已注册的处理器

        每个处理器独立执行，异常不会阻止其他处理器的执行。

        Args:
            event: 领域事件实例
        """
        handlers = list(self._handlers.get(type(event), []))
        if not handlers:
            return
        tasks = []
        for handler in handlers:
            tasks.append(self._safe_dispatch(handler, event))
        await asyncio.gather(*tasks)

    @staticmethod
    async def _safe_dispatch(handler: Callable, event: object) -> None:
        """安全分发单个事件给单个处理器"""
        try:
            result = handler(event)
            if asyncio.iscoroutine(result):
                await result
        except Exception as exc:
            _logger = logging.getLogger("rcms.event_bus")
            _logger.exception(
                "事件处理器 %s.%s 在处理 %s 时异常: %s",
                handler.__module__
                if hasattr(handler, "__module__")
                else "?",
                handler.__name__
                if hasattr(handler, "__name__")
                else str(handler),
                type(event).__name__,
                exc,
            )

    def clear(self) -> None:
        """清空所有注册的处理器（测试用）"""
        self._handlers.clear()

    @property
    def handler_count(self) -> int:
        """返回所有事件类型的处理器总数"""
        return sum(len(h) for h in self._handlers.values())

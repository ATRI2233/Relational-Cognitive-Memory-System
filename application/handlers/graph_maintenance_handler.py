"""
GraphMaintenanceHandler — MemoryDistilled 事件处理器。

在蒸馏分析完成后执行图衰减与清理：
1. 低频边快衰减，高频边慢衰减
2. 语义边额外保护
3. 权重低于阈值的边删除
4. 无关联边的孤立节点清理

对应 retrieval.py 中 _maintain_graph 方法的逻辑。
"""
from __future__ import annotations

from domain.events.memory_events import MemoryDistilled
from domain.ports.graph_repo import IGraphRepository


class GraphMaintenanceHandler:
    """MemoryDistilled 事件处理器 — 图谱衰减维护"""

    def __init__(self, graph_repo: IGraphRepository):
        self._graph_repo = graph_repo

    async def handle(self, event: MemoryDistilled) -> None:
        """处理 MemoryDistilled 事件

        Args:
            event: MemoryDistilled 事件实例
        """
        await self._graph_repo.maintain(event.user_id)

    @staticmethod
    def register(event_bus, handler) -> None:
        """在 EventBus 上注册此处理器"""
        event_bus.register(MemoryDistilled, handler.handle)

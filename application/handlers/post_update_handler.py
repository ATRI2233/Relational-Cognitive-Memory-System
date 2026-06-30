"""
PostUpdateHandler — TurnSaved 事件处理器。

在每轮对话保存后执行后处理管理操作：
1. 确保 identity_memory 行存在
2. 更新 session_state.last_active
3. 检查 dangling_threads 过期并归档
"""
from __future__ import annotations

from domain.events.memory_events import TurnSaved
from application.post_update_service import PostUpdateService


class PostUpdateHandler:
    """TurnSaved 事件处理器 — 后处理管理操作"""

    def __init__(self, post_update_service: PostUpdateService):
        self._post_update_service = post_update_service

    async def handle(self, event: TurnSaved) -> None:
        """处理 TurnSaved 事件

        Args:
            event: TurnSaved 事件实例
        """
        await self._post_update_service.run(
            user_id=event.user_id,
            session_id=event.session_id,
            user_input=event.user_input,
        )

    @staticmethod
    def register(event_bus, handler) -> None:
        """快捷：在 EventBus 上注册此处理器"""
        event_bus.register(TurnSaved, handler.handle)

"""
DistillCheckerHandler — TurnSaved 事件处理器。

每轮对话后检查是否满足蒸馏触发条件：
1. 轮数条件：current_turn - last_distill_turn >= max_turns
2. 时间条件：elapsed_minutes >= max_minutes
3. 最少消息条数：snapshot 至少包含 min_turns 轮对话

满足条件时触发 DistillUseCase.check_and_run()。
对应 plugins/rcms-astrbot/main.py 中 _check_and_distill 方法的逻辑。
"""
from __future__ import annotations

from domain.events.memory_events import TurnSaved
from application.use_cases.distill_use_case import DistillUseCase


class DistillCheckerHandler:
    """TurnSaved 事件处理器 — 蒸馏条件检查与触发"""

    def __init__(self, distill_use_case: DistillUseCase):
        self._distill_use_case = distill_use_case

    async def handle(self, event: TurnSaved) -> None:
        """处理 TurnSaved 事件

        检查蒸馏条件，满足则执行蒸馏分析。

        Args:
            event: TurnSaved 事件实例
        """
        await self._distill_use_case.check_and_run(
            user_id=event.user_id,
            session_id=event.session_id,
        )

    @staticmethod
    def register(event_bus, handler) -> None:
        """在 EventBus 上注册此处理器"""
        event_bus.register(TurnSaved, handler.handle)

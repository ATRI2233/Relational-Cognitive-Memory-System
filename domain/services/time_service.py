"""
TimeService — 时间衰减和模糊时间描述服务。

提取自 retrieval.py 的 _time_decay / _fuzz_time，职责：
1. 指数衰减计算
2. datetime 转模糊中文时间描述（"刚刚"/"昨天"/"上周"/...）
"""
from __future__ import annotations

import logging
import math
from datetime import datetime

from domain.ports.clock import IClock


class TimeService:
    """时间相关业务逻辑 — 衰减计算 + 模糊时间格式化"""

    def __init__(self, clock: IClock, halflife_days: int = 30):
        self._clock = clock
        self._halflife_days = halflife_days

    def time_decay(self, created_at: datetime | None, halflife_days: int | None = None) -> float:
        """指数衰减：importance * exp(-ln(2) * days / halflife)

        当 created_at 为 None 时返回 0（表示无法计算衰减）。
        halflife_days 可覆盖默认值。

        Args:
            created_at: 记忆创建时间
            halflife_days: 半衰期天数（可选，覆盖构造函数默认值）

        Returns:
            [0.0, 1.0] 范围的衰减系数
        """
        if created_at is None:
            return 0.0
        half = halflife_days if halflife_days is not None else self._halflife_days
        now = self._clock.now()
        days = max(0, (now - created_at).days)
        lam = math.log(2) / max(half, 1)
        return math.exp(-lam * days)

    def time_decay_weighted(self, created_at: datetime | None, importance: float,
                            halflife_days: int | None = None) -> float:
        """时间衰减 × importance 的加权值

        用于记忆召回时的评分计算。
        当 created_at 为 None 时使用 importance 原值。

        Args:
            created_at: 记忆创建时间
            importance: [0.0, 1.0] 的重要性评分
            halflife_days: 半衰期天数（可选）

        Returns:
            衰减后的加权值
        """
        if created_at is None:
            return importance
        t = self.time_decay(created_at, halflife_days)
        return min(t * (0.5 + importance), 1.0)

    def fuzz_time(self, dt: datetime | str | None) -> str:
        """将时间戳转为模糊中文时间描述

        映射规则（与 retrieval.py _fuzz_time 一致）：
          - None 或解析失败 → 空字符串
          - 0 小时内 → "刚刚"
          - 今天内 → "上午/下午/晚上的时候"
          - 1 天前 → "昨天"
          - 2 天前 → "前天"
          - 3-6 天 → "前几天"
          - 7-14 天 → "上周"
          - 15-60 天 → "前段时间"
          - 60+ 天 → "很久以前"

        Args:
            dt: datetime 对象、ISO 格式字符串或 None

        Returns:
            模糊时间字符串，解析失败返回空字符串
        """
        if dt is None:
            return ""

        parsed: datetime | None = None
        if isinstance(dt, datetime):
            parsed = dt
        elif isinstance(dt, str):
            try:
                parsed = datetime.fromisoformat(str(dt))
            except (ValueError, TypeError):
                return ""

        if parsed is None:
            return ""

        now = self._clock.now()
        delta = now - parsed
        days = delta.days

        if days < 0:
            logger = logging.getLogger("rcms")
            logger.warning("format_time_ago received future timestamp: %s", dt)
            return "刚刚"
        if days == 0:
            hours = delta.total_seconds() / 3600
            if hours < 1:
                return "刚刚"
            if parsed.hour < 12:
                return "上午的时候"
            if parsed.hour < 18:
                return "下午的时候"
            return "晚上的时候"
        if days == 1:
            return "昨天"
        if days == 2:
            return "前天"
        if days <= 6:
            return "前几天"
        if days <= 14:
            return "上周"
        if days <= 60:
            return "前段时间"
        return "很久以前"

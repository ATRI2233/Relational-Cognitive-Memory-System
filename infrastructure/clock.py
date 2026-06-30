"""
IClock 的具体实现：SystemClock（生产用）和 FrozenClock（测试用）。
"""
from datetime import datetime

from domain.ports.clock import IClock


class SystemClock:
    """生产环境时间源 — 直接调用 datetime.now() 返回实时时间。"""

    def now(self) -> datetime:
        """返回当前系统时间。"""
        return datetime.now()

    def strftime(self, fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
        """返回格式化的当前时间字符串。"""
        return self.now().strftime(fmt)


class FrozenClock:
    """测试用时间源 — 冻结在指定时间点，所有调用返回相同时间。"""

    def __init__(self, dt: datetime):
        """使用固定的时间点初始化。

        Args:
            dt: 冻结的时间点，所有 now() 调用均返回此值。
        """
        self._dt = dt

    def now(self) -> datetime:
        """返回构造时指定的冻结时间。"""
        return self._dt

    def strftime(self, fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
        """返回冻结时间的格式化字符串。"""
        return self._dt.strftime(fmt)

"""
IClock — 时间抽象接口，消除对 datetime.now() 的直接依赖。

所有需要获取当前时间的业务代码应依赖此接口而非直接调用 datetime.now()。
"""
from datetime import datetime
from typing import Protocol, runtime_checkable


@runtime_checkable
class IClock(Protocol):
    """可注入的时间源，抽象了当前时间的获取方式。"""

    def now(self) -> datetime:
        """返回当前时间。"""
        ...

    def strftime(self, fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
        """返回格式化的当前时间字符串。"""
        ...

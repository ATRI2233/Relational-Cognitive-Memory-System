"""
RCMS × AstrBot 适配器 — v2 新架构入口

转发到 adapter/astrbot_plugin.py 中的新实现。
旧版备份在 main.py.bak。
"""
import sys
import os

_self = os.path.dirname(os.path.abspath(__file__))
_root = os.path.abspath(os.path.join(_self, "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

# 委托给新架构的插件类
from adapter.astrbot_plugin import RcmsPlugin

__all__ = ["RcmsPlugin"]

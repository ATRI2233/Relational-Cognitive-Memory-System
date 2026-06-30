"""
领域服务 — 无状态的业务逻辑封装，依赖端口（Protocol）而非具体实现。

当前包含：
  - importance_service.py: ImportanceService（记忆重要性评分计算）
"""

from domain.services.importance_service import ImportanceService

__all__ = [
    "ImportanceService",
]

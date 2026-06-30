"""
持久化层 — SQLite 仓储实现。

包含 IRepository 接口的 SQLite 具体实现。
所有仓储类依赖 domain.ports 中定义的抽象接口。
"""

from infrastructure.persistence.sqlite_user_mapping_repo import (
    SQLiteUserMappingRepository,
)

__all__ = [
    "SQLiteUserMappingRepository",
]

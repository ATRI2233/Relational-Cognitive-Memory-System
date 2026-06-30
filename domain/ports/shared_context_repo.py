"""
ISharedContextRepository — 共享上下文仓储接口
"""
from __future__ import annotations
from typing import Protocol, runtime_checkable


@runtime_checkable
class ISharedContextRepository(Protocol):
    async def upsert_joke(self, user_id: str, trigger: str, context: str) -> None: ...
    async def get_recent(self, user_id: str, limit: int = 4) -> list[str]: ...

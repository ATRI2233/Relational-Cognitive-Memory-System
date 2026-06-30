"""
端口（Ports）— 定义系统边界上的抽象接口（Protocol）。

端口层只声明协约（接口），不包含具体实现。
具体实现在 infrastructure 层中提供。

当前可用的仓储接口：
  - IGraphRepository       — 图谱（memory_graph_nodes / memory_graph_edges）
  - IIdentityRepository    — 用户身份（identity_memory）
  - IMemoryRepository      — 认知蒸馏记忆（cognitive_distill）
  - ISessionRepository     — 会话状态（session_state / chat_history）
"""

from domain.ports.graph_repo import IGraphRepository
from domain.ports.identity_repo import IIdentityRepository
from domain.ports.repositories import IMemoryRepository, ISessionRepository

__all__ = [
    "IGraphRepository",
    "IIdentityRepository",
    "IMemoryRepository",
    "ISessionRepository",
]

"""
Graph 仓储接口
================

定义 IGraphRepository Protocol，作为图谱持久化与检索的协约。
"""

from __future__ import annotations

from typing import List, Protocol, runtime_checkable

from domain.entities.graph import DiffusionResult, GraphEdge, GraphNode


@runtime_checkable
class IGraphRepository(Protocol):
    """记忆图谱仓储接口。

    负责记忆图谱（memory_graph_nodes / memory_graph_edges）的
    增删改查、BFS 扩散激活、连通路径检索及衰减维护。
    所有方法均为异步，返回领域实体类型。
    """

    async def upsert_node(
        self, user_id: str, label: str, entity_type: str = "auto"
    ) -> int:
        """插入或更新图节点。

        如果 label 已存在则 freq +1、更新 last_seen，
        否则创建新节点，freq 初始化为 1。
        对应 `_upsert_graph_node` 方法的行为。

        Args:
            user_id: 用户标识
            label: 节点标签（关键词/实体名）
            entity_type: 实体类型，默认 "auto"

        Returns:
            节点的 node_id（新插入或已存在的）
        """
        ...

    async def upsert_edge(
        self, from_id: int, to_id: int, relation: str = ""
    ) -> None:
        """插入或更新图边。

        如果边已存在则 weight +0.5、encounter_count +1、更新 last_seen，
        并可选覆盖 relation。若存在对立关系则先删除旧边。
        对应 `_upsert_graph_edge` 方法的行为。

        Args:
            from_id: 源节点 ID
            to_id: 目标节点 ID
            relation: 关系描述，空字符串表示共现关系
        """
        ...

    async def bfs_diffuse(
        self, user_id: str, seed_ids: List[int], depth: int = 2
    ) -> List[DiffusionResult]:
        """BFS 图扩散激活。

        以 seed_ids 为起点，按 BFS 深度优先扩散，激活分数沿边递减。
        无 relation 的共现边降权为 0.1 倍。
        对应 `_graph_activation_diffusion` 方法的行为。

        Args:
            user_id: 用户标识
            seed_ids: 种子节点 ID 列表
            depth: BFS 最大深度，默认 2

        Returns:
            按激活分数降序排列的 DiffusionResult 列表
        """
        ...

    async def get_edges_by_node(self, node_id: int) -> List[GraphEdge]:
        """获取与指定节点相连的所有边。

        对应图谱查询中筛选某节点关联边的操作。

        Args:
            node_id: 节点 ID

        Returns:
            GraphEdge 列表（包括出边和入边）
        """
        ...

    async def get_nodes_by_user(self, user_id: str) -> List[GraphNode]:
        """获取用户的所有图节点。

        Args:
            user_id: 用户标识

        Returns:
            该用户的所有 GraphNode 列表
        """
        ...

    async def maintain(self, user_id: str) -> None:
        """执行图衰减与清理维护。

        低频边快衰减，高频边慢衰减，语义边额外保护。
        权重低于 0.4 的边被删除，无关联边的孤立节点被清理。
        对应 `_maintain_graph` 方法的行为。

        Args:
            user_id: 用户标识
        """
        ...

    async def search_nodes(
        self, user_id: str, keyword: str
    ) -> List[GraphNode]:
        """按关键词模糊搜索图节点。

        用于通道 3 图骨架查询和通道 2 关键词检索时的种子节点查找。

        Args:
            user_id: 用户标识
            keyword: 搜索关键词（LIKE 模糊匹配）

        Returns:
            匹配的 GraphNode 列表
        """
        ...

    async def get_chain_paths(
        self, user_id: str, labels: List[str]
    ) -> List[str]:
        """获取节点标签之间的连通路径。

        对节点标签集合，查询之间的语义边，DFS 找最长链。
        格式如："A [关系] B → B [关系] C"。
        对应 `_build_graph_paths` 方法的行为。

        Args:
            user_id: 用户标识
            labels: 节点标签列表

        Returns:
            路径描述字符串列表，最多 3 条
        """
        ...

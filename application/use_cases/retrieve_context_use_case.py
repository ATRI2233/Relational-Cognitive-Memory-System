"""
RetrieveContextUseCase — 上下文检索与格式化 Use Case。

对应以下源文件的管线整合：
  - retrieval.py:         三通道融合召回（行 49-67 入口，行 77-103 通道1，
                          行 121-241 通道2，行 245-306 通道3，行 374-421 融合）
  - core.py:              _load_long_term_context（行 150-190）
                          build_multi_user_context（行 114-143）
  - context.py:           narrative_context（行 51-202）
                          prompt_compressor（行 204-267）
                          _session_warmup（行 11-49）

本 Use Case 的职责：
  1. 三通道融合召回（时间重要性 / 语义共振 / 图谱骨架）
  2. FusionService 融合去重排序
  3. 加载长期语境（身份 / 实体 / 共享上下文）
  4. 构建 narrative context 文本
  5. 构建 prompt compressor 文本
  6. 多用户上下文构建

设计原则：
  - 零 try/except：异常向上抛给调用方
  - 零第三方库依赖：向量运算、分词等由基础设施层实现
  - 所有依赖通过构造函数显式注入（Protocol 接口）
  - 单次 retrieve 周期内的运行时状态（_graph_chain_labels / _graph_paths）
    通过局部变量传递，不存储在实例变量中，确保并发安全
"""
from __future__ import annotations

from dataclasses import dataclass, field
import logging
import re
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from domain.entities.memory import Memory, MemoryId, UserId, SessionId
from domain.entities.identity import Identity, Trait, Preferences, Boundary
from domain.entities.graph import GraphEdge, GraphNode
from domain.entities.session import Session
from domain.ports.repositories import IMemoryRepository, ISessionRepository
from domain.ports.identity_repo import IIdentityRepository
from domain.ports.graph_repo import IGraphRepository
from domain.ports.clock import IClock
from domain.services.fusion_service import FusionService, ChannelTag
from domain.services.time_service import TimeService
from domain.services.keyword_service import KeywordService

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
# 依赖接口协议（应用层协议，尚未纳入 domain.ports）
# ═══════════════════════════════════════════════════════════════


@runtime_checkable
class IEmbeddingService(Protocol):
    """文本向量化服务。

    将文本转为浮点数向量，供语义相似度检索使用。
    具体实现使用 OpenAI / 本地模型等，位于 infrastructure 层。
    """

    async def embed(self, text: str) -> list[float]:
        """将文本转为向量。

        Args:
            text: 输入文本（建议不超过 512 token）

        Returns:
            浮点数列表，维度由具体模型决定

        Raises:
            Exception: API 调用失败时向上抛出
        """
        ...


@runtime_checkable
class ISharedContextRepository(Protocol):
    """共享语境仓储。

    对应 shared_context 表的读写操作。
    记录用户与 AI 之间积累的共同背景知识。
    """

    async def get_recent(self, user_id: str, limit: int = 4) -> list[str]:
        """获取最近的共享语境。

        Args:
            user_id: 用户标识
            limit: 最大返回条数

        Returns:
            context_body 字符串列表，按 context_id 降序
        """
        ...


@runtime_checkable
class IUserMappingRepository(Protocol):
    """用户映射仓储。

    对应 user_mappings 表的读写操作。
    记录 session 中各用户的昵称映射关系。
    用于多用户场景的身份识别和别名管理。
    """

    async def find_mentioned(
        self, session_id: str, text: str, speaker_id: str = ""
    ) -> list[tuple[str, str]]:
        """扫描消息文本，返回被提及的用户。

        三级匹配策略（对应 SessionMixin.find_mentioned_users）：
          1. 当前 session 中的已知用户映射
          2. 发言者参与过的其他 session 中的映射
          3. 全局映射兜底

        Args:
            session_id: 当前会话标识
            text: 消息文本
            speaker_id: 发言者标识，排除自己

        Returns:
            [(user_id, label), ...] 被提及的用户标识和显示名列表
        """
        ...

    async def get_labels(self, session_id: str, user_id: str) -> list[str]:
        """获取某用户在指定 session 中的所有昵称。

        Args:
            session_id: 会话标识
            user_id: 用户标识

        Returns:
            昵称字符串列表
        """
        ...


@runtime_checkable
class ITextTemplateProvider(Protocol):
    """文案模板提供者。

    封装 prompts.json 的读取逻辑。
    测试时可以注入内存模板，无需依赖 JSON 文件。
    """

    def narrative_templates(self) -> dict:
        """获取 narrative_context 模板节。

        Returns:
            prompts.json["narrative_context"] 的全部键值对，
            包含 prefix / footer / memories_title / profile_title 等模板字符串
        """
        ...

    def channel_labels(self) -> dict:
        """获取通道标签文本。

        Returns:
            {"recent": "时间·重要性", "resonance": "语义检索", "skeleton": "图谱关联"}
        """
        ...

    def memories_display_order(self) -> list:
        """获取记忆通道显示顺序。

        Returns:
            ["resonance", "skeleton", "recent"]
        """
        ...

    def prompt_compressor_templates(self) -> dict:
        """获取 prompt_compressor 模板节。

        Returns:
            prompts.json["prompt_compressor"] 的全部键值对，
            包含 mood_title / memory_title / graph_chain_title 等模板字符串
        """
        ...


@runtime_checkable
class IRetrievalConfig(Protocol):
    """检索配置接口。

    从 config.json analysis.retrieval 节读取，
    控制三通道融合的参数和行为。
    """

    total_cap: int
    channel_min: list[int]
    channel_weights: list[float]
    time_decay_halflife: int
    emotional_resonance_bonus: float


@runtime_checkable
class ISessionQueryRepository(Protocol):
    """会话查询仓储扩展接口。

    提供跨会话查询能力，用于 session_warmup 等场景。
    """

    async def get_most_recent_excluding(
        self, exclude_session_id: str
    ) -> Optional[Session]:
        """获取最近活跃的、非指定 session 的会话。

        Args:
            exclude_session_id: 排除的会话标识（通常是当前 session）

        Returns:
            最近活跃的 Session 实体，如果没有任何其他 session 则返回 None
        """
        ...


# ═══════════════════════════════════════════════════════════════
# Use Case
# ═══════════════════════════════════════════════════════════════


class RetrieveContextUseCase:
    """上下文检索与格式化 Use Case。

    处理三通道召回、长期语境加载、上下文文本构建。
    不持有 DB 连接，所有依赖通过构造函数注入。
    遵循依赖倒置原则，所有依赖均为 Protocol 接口。

    Usage::

        use_case = RetrieveContextUseCase(
            memory_repo=memory_repo,
            identity_repo=identity_repo,
            graph_repo=graph_repo,
            session_repo=session_repo,
            shared_context_repo=shared_context_repo,
            user_mapping_repo=user_mapping_repo,
            fusion_service=fusion_service,
            time_service=time_service,
            keyword_service=keyword_service,
            embed_service=embed_service,
            clock=system_clock,
            config=retrieval_config,
            session_query_repo=session_query_repo,
            template_provider=template_provider,
        )

        memories = await use_case.retrieve_memories("user_1", "最近心情不好")
        long_term = await use_case.load_long_term_context("user_1")
        context = await use_case.build_narrative_context(
            memories, long_term, "user_1", "session_1"
        )
    """

    def __init__(
        self,
        memory_repo: IMemoryRepository,
        identity_repo: IIdentityRepository,
        graph_repo: IGraphRepository,
        session_repo: ISessionRepository,
        shared_context_repo: ISharedContextRepository,
        user_mapping_repo: IUserMappingRepository,
        fusion_service: FusionService,
        time_service: TimeService,
        keyword_service: KeywordService,
        embed_service: IEmbeddingService,
        clock: IClock,
        config: IRetrievalConfig,
        session_query_repo: ISessionQueryRepository,
        template_provider: ITextTemplateProvider,
    ) -> None:
        """初始化 RetrieveContextUseCase。

        Args:
            memory_repo: 认知记忆仓储（cognitive_distill 检索）
            identity_repo: 用户身份仓储（identity_memory 查询）
            graph_repo: 图谱仓储（memory_graph_nodes / edges 检索）
            session_repo: 会话状态仓储（session_state 查询）
            shared_context_repo: 共享语境仓储（shared_context 查询）
            user_mapping_repo: 用户映射仓储（user_mappings 查询）
            fusion_service: 三通道融合服务（去重 / 加权排序 / 截断）
            time_service: 时间服务（衰减计算 / 模糊时间描述）
            keyword_service: 关键词提取和时间词解析
            embed_service: 文本向量化服务
            clock: 时间源（消除对 datetime.now() 的直接依赖）
            config: 检索配置（channel_min / total_cap / 半衰期等）
            session_query_repo: 会话查询仓储扩展（跨会话查询）
            template_provider: 文案模板提供者（prompts.json 抽象）
        """
        self._memory_repo = memory_repo
        self._identity_repo = identity_repo
        self._graph_repo = graph_repo
        self._session_repo = session_repo
        self._shared_context_repo = shared_context_repo
        self._user_mapping_repo = user_mapping_repo
        self._fusion_service = fusion_service
        self._time_service = time_service
        self._keyword_service = keyword_service
        self._embed_service = embed_service
        self._clock = clock
        self._config = config
        self._session_query_repo = session_query_repo
        self._template_provider = template_provider

        # ── 最后一次检索的图谱路径（供 graph_paths 属性向后兼容）──
        self._graph_paths: list[str] = []

    @property
    def graph_paths(self) -> list[str]:
        """获取最近一次 retrieve_memories 调用产生的图谱关系链路径。

        Returns:
            图谱路径描述字符串列表，供 prompt 构建使用。
            在 retrieve_memories 调用之间可能为空列表。

        Note: 仅用于向后兼容。新代码应当从 retrieve_memories 的返回值获取。
        """
        return self._graph_paths

    # ═══════════════════════════════════════════════════════════
    # 公有入口：三通道融合召回
    # ═══════════════════════════════════════════════════════════

    async def retrieve_memories(
        self,
        user_id: str,
        user_input: str,
        session_id: Optional[str] = None,
        stance: str = "engaged",
    ) -> tuple[list[tuple[str, str]], list[str]]:
        """三通道融合召回。

        对应 retrieval.py retrieve_memories 方法（行 49-67）。
        当 stance == 'casual' 时返回空列表。

        Args:
            user_id: 用户标识
            user_input: 用户输入文本，用作语义共振和关键词提取的源
            session_id: 会话标识，用于通道 1 的时间重要性锚定
            stance: 会话立场（'engaged' 执行完整召回，'casual' 跳过召回）

        Returns:
            (memories, graph_paths):
              memories:    [(content, channel_tag), ...]
                content:     格式化的记忆文本（已含模糊时间前缀等）
                channel_tag: 'recent' / 'resonance' / 'skeleton'
              按加权分降序，总数不超过 config.total_cap
              graph_paths: 图谱关系链描述字符串列表（供 prompt 注入用）
        """
        if stance == "casual":
            self._graph_paths = []
            return [], []

        config = self._config

        # ── 三通道并行召回 ──
        channel_time = await self._channel_time(
            user_id, session_id=session_id, limit=config.channel_min[0] + 1
        )
        channel_resonance = await self._channel_resonance(
            user_id, user_input, limit=config.channel_min[1] + 2
        )
        channel_graph_result = await self._channel_graph(
            user_id, user_input, limit=config.channel_min[2] + 1
        )
        channel_graph, chain_labels = channel_graph_result

        # ── FusionService 融合 ──
        result = self._fusion_service.fuse({
            ChannelTag.RECENT: channel_time,
            ChannelTag.RESONANCE: channel_resonance,
            ChannelTag.SKELETON: channel_graph,
        })

        # ── 构建图谱关系链（从 _channel_graph 收集的链标签查询连通路径）─
        graph_paths: list[str] = []
        if chain_labels:
            graph_paths = await self._build_graph_paths(user_id, chain_labels)

        self._graph_paths = graph_paths
        return result, graph_paths

    async def load_long_term_context(self, user_id: str) -> dict:
        """加载用户长期语境。

        对应 core.py _load_long_term_context 方法（行 150-190）。
        使用 IIdentityRepository + IGraphRepository 替代裸 SQL。

        Args:
            user_id: 用户标识

        Returns:
            dict 包含以下键：
              - identity_traits: [str, ...]          用户特质文本列表
              - trait_details:   [dict, ...]          特质详情（含 strength / count）
              - preferences:     {"likes": [...], "dislikes": [...]}
              - self_identity:   [str, ...]           自我认知描述
              - boundaries:      [{"description": str}, ...]  边界/雷区
              - entities:        [{"name", "type", "relation", "fact"}, ...]  图谱实体
              - shared_contexts: [str, ...]           共享语境文本列表
        """
        # ── 身份特质（IIdentityRepository）──
        identity = await self._identity_repo.get(user_id)

        if identity is not None:
            trait_details = [
                {"text": t.text, "strength": t.strength, "count": t.count}
                for t in identity.traits
                if t.text and t.strength > 0
            ]
            identity_traits = [t["text"] for t in trait_details]
            preferences = {
                "likes": identity.preferences.likes,
                "dislikes": identity.preferences.dislikes,
            }
            self_identity = identity.self_identity[:]
            boundaries = [
                {"description": b.description} for b in identity.boundaries
            ]
        else:
            trait_details = []
            identity_traits = []
            preferences = {}
            self_identity = []
            boundaries = []

        # ── 图谱实体关系（IGraphRepository）──
        entities = await self._load_entities(user_id)

        # ── 共享语境 ──
        shared_contexts = await self._shared_context_repo.get_recent(
            user_id, limit=4
        )

        return {
            "identity_traits": identity_traits,
            "trait_details": trait_details,
            "preferences": preferences,
            "self_identity": self_identity,
            "boundaries": boundaries,
            "entities": entities,
            "shared_contexts": shared_contexts,
        }

    async def build_narrative_context(
        self,
        memories: list[tuple[str, str]],
        long_term: dict,
        user_id: str,
        session_id: Optional[str] = None,
        user_input: str = "",
        graph_paths: Optional[list[str]] = None,
    ) -> str:
        """构建 narrative context 文本。

        对应 context.py narrative_context 方法（行 51-202）。
        文案模板从 ITextTemplateProvider 读取。

        Args:
            memories: 三通道召回结果 [(content, tag), ...]
            long_term: load_long_term_context 返回的长期语境字典
            user_id: 用户标识
            session_id: 会话标识（可选，用于读取会话统计和预热）
            user_input: 用户输入文本
            graph_paths: 图谱关系链路径列表，来自 retrieve_memories 返回值

        Returns:
            格式化的 narrative context 字符串
        """
        tmpl = self._template_provider.narrative_templates()
        channel_labels = self._template_provider.channel_labels()
        display_order = self._template_provider.memories_display_order()

        parts: list[str] = []

        # ── 三通道记忆 ──
        if memories:
            grouped: dict[str, list[str]] = {}
            for content, tag in memories:
                grouped.setdefault(tag, []).append(content)
            ch_lines: list[str] = []
            for key in display_order:
                items = grouped.get(key)
                if items:
                    label = channel_labels.get(key, key)
                    # 图谱通道用 chains 替代单条三元组，增加信息量
                    if key == "skeleton" and graph_paths:
                        items = graph_paths
                    count_info = f"（共 {len(items)} 条）"
                    ch_lines.append(
                        f"【{label}】{count_info}\n"
                        + "\n".join(f"  · {c}" for c in items)
                    )
            if ch_lines:
                memories_title = tmpl.get("memories_title", "相关记忆")
                parts.append(f"{memories_title}:\n" + "\n\n".join(ch_lines))

        # ── 会话统计 + 预热 ──
        turn_count = 0
        dangling = ""
        if session_id is not None:
            session_obj = await self._session_repo.get(SessionId(session_id))
            if session_obj is not None:
                turn_count = session_obj.turn_count
                dangling = session_obj.dangling_threads or {}

        # ── 新 session 预热 ──
        warmup = await self._session_warmup(user_id, session_id, turn_count)
        if warmup:
            warmup_title = tmpl.get("session_warmup_title", "[上次聊到]")
            parts.append(f"{warmup_title}\n{warmup}")

        # ── 轮数 ──
        if turn_count:
            turn_template = tmpl.get("turn_count_template", "聊了 {count} 轮")
            parts.append(turn_template.format(count=turn_count))

        # ── 用户画像：traits + quirks ──
        remaining_traits_tpl = tmpl.get(
            "remaining_traits_template", "及其他 {count} 条特质"
        )
        quirk_label = tmpl.get("quirk_prefix", "小细节小习惯")
        profile_lines: list[str] = []
        trait_details = long_term.get("trait_details", [])
        trait_details.sort(key=lambda x: x.get("strength", 0), reverse=True)
        all_traits = [
            td
            for td in trait_details
            if not td["text"].startswith("[小细节小习惯]")
        ]
        max_show = 5
        for td in all_traits[:max_show]:
            strength = td.get("strength", 0)
            prefix = "" if strength >= 5 else "↘ " if strength <= 2 else "· "
            profile_lines.append(f"{prefix}{td['text']}")
        remaining = len(all_traits) - max_show
        if remaining > 0:
            profile_lines.append(remaining_traits_tpl.format(count=remaining))
        quirks = [
            (td.get("strength", 0), td["text"].replace("[小细节小习惯] ", ""))
            for td in trait_details
            if td["text"].startswith("[小细节小习惯]")
        ]
        quirks.sort(key=lambda x: x[0], reverse=True)
        if quirks:
            q_mark = "↘ " if any(q[0] <= 2 for q in quirks) else ""
            profile_lines.append(
                f"{q_mark}{quirk_label}: {'、'.join(q[1] for q in quirks[:2])}"
            )
        if profile_lines:
            profile_title = tmpl.get("profile_title", "他是什么样的")
            parts.append(
                profile_title + ":\n"
                + "\n".join(f"  · {t}" for t in profile_lines)
            )

        # ── 结构化画像 ──
        struct_lines: list[str] = []
        prefs = long_term.get("preferences", {})
        if prefs.get("likes"):
            likes_label = tmpl.get("pref_likes", "喜好")
            struct_lines.append(
                likes_label + ": " + "、".join(prefs["likes"][:5])
            )
        if prefs.get("dislikes"):
            dislikes_label = tmpl.get("pref_dislikes", "不喜欢")
            struct_lines.append(
                dislikes_label + ": " + "、".join(prefs["dislikes"][:3])
            )
        si = long_term.get("self_identity", [])
        if si:
            identity_label = tmpl.get("self_identity", "自我认同")
            struct_lines.append(
                identity_label + ": " + "、".join(si[:3])
            )
        bounds = long_term.get("boundaries", [])
        if bounds:
            boundary_label = tmpl.get("boundaries", "雷区")
            bound_texts = [
                b["description"] if isinstance(b, dict) else str(b)
                for b in bounds[:3]
            ]
            struct_lines.append(
                boundary_label + ": " + "、".join(bound_texts)
            )
        if struct_lines:
            struct_title = tmpl.get("struct_title", "结构化画像")
            parts.append(
                struct_title + ":\n"
                + "\n".join(f"  · {s}" for s in struct_lines)
            )

        # ── 共同语境 ──
        context_title = tmpl.get("context_title", "共同语境")
        joke_label = tmpl.get("joke_label", "梗")
        entity_label_tpl = tmpl.get(
            "entity_label_template", "他提过的{type}"
        )
        ctx_lines: list[str] = []
        shared = long_term.get("shared_contexts", [])
        jokes = [s.replace("[梗] ", "") for s in shared if s.startswith("[梗]")][:2]
        other = [s for s in shared if not s.startswith("[梗]")][:2]
        ctx_lines.extend(f"{joke_label}: {j}" for j in jokes)
        ctx_lines.extend(other)

        entities = long_term.get("entities", [])
        if entities:
            grouped_ents: dict[str, list[str]] = {}
            seen_names: set[str] = set()
            for e in entities:
                name = e.get("name", "")
                if not name or name in seen_names:
                    continue
                seen_names.add(name)
                etype = e.get("type", "auto")
                if etype not in grouped_ents:
                    grouped_ents[etype] = []
                tag = f" ({e.get('relation', '')})" if e.get("relation") else ""
                grouped_ents[etype].append(f"{name}{tag}")
            type_labels: dict[str, str] = {
                "person": "人",
                "place": "地方",
                "concept": "概念",
                "activity": "活动",
                "auto": "相关",
            }
            for etype, items in grouped_ents.items():
                label = type_labels.get(etype, etype)
                ctx_lines.append(
                    entity_label_tpl.format(type=label) + ": " + "、".join(items[:4])
                )
        if ctx_lines:
            parts.append(
                context_title + ":\n"
                + "\n".join(f"  · {c}" for c in ctx_lines)
            )

        # ── 图谱关系链（已删除，保留【图谱关联】通道即可）──
        # ── 未完成话题 ──
        stale_prefix = tmpl.get("dangling_stale_prefix", "↘ ")
        if dangling:
            if isinstance(dangling, dict):
                dt_list = dangling.get("threads", [])
                since_turn = dangling.get("turn", 0)
                if dt_list and turn_count - since_turn <= 10:
                    stale = turn_count - since_turn > 5
                    prefix = stale_prefix if stale else ""
                    dangling_display = prefix + "、".join(dt_list[:3])
                    dangling_label = tmpl.get("dangling_label", "未完成")
                    parts.append(f"{dangling_label}: {dangling_display}")
            elif isinstance(dangling, list) and dangling:
                parts.append(
                    tmpl.get("dangling_label", "未完成")
                    + ": " + "、".join(dangling[:3])
                )

        footer = tmpl.get(
            "footer",
            "→ 以上是你通过长期对话积累的对他的了解，用来更好地理解他的意图。",
        )
        parts.append(footer)

        prefix = tmpl.get(
            "prefix",
            "[RCMS 关系上下文,这里面放置了你对他的了解,按需使用]",
        )
        return prefix + "\n" + "\n\n".join(parts)

    def build_prompt(
        self,
        memories: list[tuple[str, str]],
        long_term: dict,
        user_input: str,
        graph_paths: Optional[list[str]] = None,
    ) -> str:
        """构建 LLM prompt。

        对应 context.py prompt_compressor 方法（行 204-267）。
        与 narrative_context 使用不同的模板节（prompt_compressor），
        输出更紧凑的 prompt 格式。

        Args:
            memories: 三通道召回结果 [(content, tag), ...]
            long_term: load_long_term_context 返回的长期语境字典
            user_input: 用户输入文本
            graph_paths: 图谱关系链路径列表，来自 retrieve_memories 返回值

        Returns:
            组装后的 prompt 字符串
        """
        pc = self._template_provider.prompt_compressor_templates()
        channel_labels = self._template_provider.channel_labels()

        # ── 记忆块 ──
        grouped: dict[str, list[str]] = {}
        order: list[str] = []
        for content, tag in memories:
            if tag not in grouped:
                grouped[tag] = []
                order.append(tag)
            grouped[tag].append(content)
        mem_lines: list[str] = []
        for key in order:
            if key not in grouped:
                continue
            label = channel_labels.get(key, key)
            items = grouped[key][:2]
            count_info = f"（共 {len(items)} 条）"
            mem_lines.append(
                f"【{label}】{count_info}\n" + "\n".join(f"  · {c}" for c in items)
            )
        mem_block = "\n\n".join(mem_lines) if mem_lines else ""

        # ── 图谱关系链 ──
        gp_block = ""
        graph_chain_title = pc.get("graph_chain_title", "【图谱关系链】")
        paths = graph_paths or self._graph_paths or []
        if paths:
            gp_block = (
                "\n" + graph_chain_title + "\n"
                + "\n".join(f"  · {pth}" for pth in paths)
            )

        # ── 长期语境块 ──
        lt_block = ""
        shared_ctx = long_term.get("shared_contexts", [])
        if shared_ctx:
            ctx = "、".join(shared_ctx[:3])
            shared_title = pc.get("shared_context_title", "【共同语境】")
            lt_block += "\n" + shared_title + ctx
        traits = long_term.get("identity_traits", [])
        if traits:
            trait_strs = [
                t for t in traits if not t.startswith("[小细节小习惯]")
            ][:3]
            if trait_strs:
                trait_title = pc.get("trait_title", "【用户特质】")
                lt_block += "\n" + trait_title + "；".join(trait_strs)
            quirks = [
                t for t in traits if t.startswith("[小细节小习惯]")
            ][:2]
            if quirks:
                quirk_title = pc.get("quirk_title", "【小细节小习惯】")
                lt_block += (
                    "\n" + quirk_title
                    + "；".join(
                        q.replace("[小细节小习惯] ", "") for q in quirks
                    )
                )
        si = long_term.get("self_identity", [])
        if si:
            identity_title = pc.get("identity_title", "【自我认同】")
            lt_block += "\n" + identity_title + "、".join(si[:2])
        bounds = long_term.get("boundaries", [])
        if bounds:
            bound_title = pc.get("boundary_title", "【雷区】")
            bound_texts = [
                b["description"] if isinstance(b, dict) else str(b)
                for b in bounds[:2]
            ]
            lt_block += "\n" + bound_title + "、".join(bound_texts)

        # ── 组装 ──
        mood_title = pc.get("mood_title", "【当前心理状态】")
        mood_default = pc.get("mood_default", "自然地聊")
        prompt = f"{mood_title}\n{mood_default}"

        if mem_block:
            memory_title = pc.get("memory_title", "【相关记忆】")
            prompt += "\n\n" + memory_title + "\n" + mem_block
        if gp_block:
            prompt += gp_block
        if lt_block:
            prompt += lt_block

        bottom_label = pc.get("bottom_line_label", "【底线】")
        bottom_text = pc.get(
            "bottom_line_text",
            "不主动说教。不假装完全理解。疲惫时简短但不冷漠。",
        )
        user_tmpl = pc.get("user_template", "用户: {user_input}\n你:")
        prompt += (
            "\n\n" + bottom_label + "\n" + bottom_text
            + "\n\n" + user_tmpl.format(user_input=user_input)
        )
        return prompt

    async def build_multi_user_context(
        self,
        session_id: str,
        user_input: str,
        speaker_id: str,
        speaker_name: str,
    ) -> str:
        """多用户上下文构建。

        对应 core.py build_multi_user_context 方法（行 114-143）。
        为发言者和被提及用户分别构建标注了姓名的 narrative context 块。

        Args:
            session_id: 会话标识
            user_input: 用户输入文本
            speaker_id: 当前发言者标识
            speaker_name: 当前发言者显示名

        Returns:
            多用户上下文字符串，各用户块用换行分隔
        """
        user_entries: list[tuple[str, str, str]] = [
            (speaker_id, speaker_name, "当前发言")
        ]
        mentioned = await self._user_mapping_repo.find_mentioned(
            session_id, user_input, speaker_id
        )
        for mid, label in mentioned:
            if mid != speaker_id:
                user_entries.append((mid, label, "被提及"))

        blocks: list[str] = []
        for uid, display_name, role in user_entries:
            name_parts = await self._user_mapping_repo.get_labels(
                session_id, uid
            )
            main_name = (
                display_name
                if display_name in name_parts
                else (name_parts[0] if name_parts else display_name)
            )
            aliases = [n for n in name_parts if n != main_name]
            alias_str = (
                f"，也被叫做：{'、'.join(aliases)}" if aliases else ""
            )
            header = (
                f"[RCMS 关系上下文: {main_name}（{role}{alias_str}）]"
            )

            mems, graph_paths = await self.retrieve_memories(
                uid, user_input, session_id=session_id, stance="engaged"
            )
            lt = await self.load_long_term_context(uid)
            ctx = await self.build_narrative_context(
                mems, lt, uid, session_id, graph_paths=graph_paths
            )
            blocks.append(f"{header}\n" + ctx)

        return "\n\n".join(blocks)

    # ═══════════════════════════════════════════════════════════
    # 通道 1：时间重要性锚点
    # ═══════════════════════════════════════════════════════════

    async def _channel_time(
        self,
        user_id: str,
        session_id: Optional[str] = None,
        limit: int = 2,
    ) -> list[tuple[str, float]]:
        """通道 1：时间衰减 × importance，当前 session 条目额外推高。

        对应 retrieval.py _channel_time_importance（行 77-103）。

        逻辑：
          1. 从 IMemoryRepository 获取最近 50 条重要性 > 0.1 的记忆
          2. 过滤已过期的条目
          3. 每条计算：score = (time_decay + session_boost) * (0.5 + importance)
             session_boost = 0.3（当前 session 匹配时）
          4. 格式化：fuzz_time + "，" + summary
          5. 按 score 降序，截取 limit 条

        Args:
            user_id: 用户标识
            session_id: 会话标识（可选，用于 session_boost）
            limit: 最大返回条数

        Returns:
            [(formatted_content, score), ...] 按分数降序，最多 limit 条
        """
        uid = UserId(user_id)
        memories = await self._memory_repo.get_by_user(
            uid, limit=200, min_importance=0.0
        )
        now = self._clock.now()

        scored: list[tuple[str, float]] = []
        for mem in memories:
            if mem.is_expired(now):
                continue

            t = self._time_service.time_decay(mem.created_at)
            session_boost = 0.0
            if session_id is not None and mem.session_id is not None:
                if mem.session_id.value == session_id:
                    session_boost = 0.3
            score = (t + session_boost) * (0.5 + mem.importance.value)

            # 保护：高重要性记忆的最低分数保障
            importance_floor = 0.05
            min_score = importance_floor * mem.importance.value
            score = max(score, min_score)

            # 格式化：模糊时间 + "，" + 摘要
            fuzz = self._time_service.fuzz_time(mem.created_at)
            text = mem.summary if mem.summary else mem.content
            formatted = f"{fuzz}，{text}" if fuzz else text
            scored.append((formatted, score))

        scored.sort(key=lambda x: -x[1])
        return scored[:limit]

    # ═══════════════════════════════════════════════════════════
    # 通道 2：语义共振
    # ═══════════════════════════════════════════════════════════

    async def _channel_resonance(
        self,
        user_id: str,
        user_input: str,
        limit: int = 3,
    ) -> list[tuple[str, float]]:
        """通道 2：时间词过滤 → 图扩散扩词 → 向量余弦 → 情绪共振 → 重要性兜底。

        对应 retrieval.py _channel_multi_resonance（行 121-241）。

        流程：
          1. 解析时间范围（"最近"/"昨天"等）
          2. 提取关键词 → 图扩散扩充
          3. 构造扩展查询文本 → 向量化 → IMemoryRepository.search_by_embedding
          4. IMemoryRepository.search_by_keywords（补充 vec 未覆盖的条目）
          5. 无结果时降级为重要性兜底查询
          6. 评分：余弦 × 0.6 + 时间衰减重要性 × 0.4，情绪共振加成 15%
          7. 格式化内容字符串，按分数降序截取

        Args:
            user_id: 用户标识
            user_input: 用户输入文本
            limit: 最大返回条数

        Returns:
            [(content, score), ...] 按分数降序，最多 limit 条
        """
        config = self._config
        uid = UserId(user_id)

        # 1. 时间范围解析
        time_range = self._keyword_service.parse_time_filter(user_input)

        # 2. 关键词提取 + 图扩散
        kws = self._keyword_service.extract_keywords(user_input, max_kw=4)
        diffused = await self._graph_diffuse(user_id, kws)
        all_keywords = list(set(kws + [label for label, _ in diffused[:3]]))

        # 3. 向量检索（用原查询 + 扩散词）
        query_text = user_input
        if all_keywords:
            query_text = user_input + " " + " ".join(all_keywords)

        query_vec = await self._embed_service.embed(query_text[:512])
        vec_results: dict[int, tuple[float, Memory]] = {}
        if query_vec:
            embedding_results = await self._memory_repo.search_by_embedding(
                uid, query_vec, limit=50
            )
            for mem, cos_sim in embedding_results:
                if cos_sim > 0.3:
                    vec_results[mem.memory_id.value] = (cos_sim, mem)

        # 4. 关键词 SQL 候选
        kw_memories: list[Memory] = []
        if all_keywords:
            kw_memories = await self._memory_repo.search_by_keywords(
                uid, all_keywords, limit=30, time_filter=time_range
            )

        # 5. 无向量 + 无关键词结果 → 重要性兜底
        if not vec_results and not kw_memories:
            kw_memories = await self._memory_repo.get_by_user(
                uid, limit=5, min_importance=0.5
            )

        # 6. 获取当前情绪（最近一条记忆的情绪标签）
        recent = await self._memory_repo.get_by_user(uid, limit=1)
        current_mood = recent[0].mood.value if recent else ""

        # 7. 评分融合
        now = self._clock.now()
        scored: list[tuple[str, float]] = []
        kw_mem_map: dict[int, Memory] = {
            m.memory_id.value: m for m in kw_memories
        }

        if vec_results:
            for mid, mem in kw_mem_map.items():
                cos_sim = vec_results.get(mid, (0.0, None))[0]
                t = self._time_service.time_decay(mem.created_at)
                imp_decay = mem.importance.value * t

                if cos_sim > 0:
                    score = cos_sim * 0.6 + imp_decay * 0.4
                else:
                    score = imp_decay * 0.5

                if current_mood and mem.mood.value and mem.mood.value == current_mood:
                    score *= 1 + config.emotional_resonance_bonus

                scored.append((mem.content, score))

            for mid, (cos_sim, mem) in vec_results.items():
                if mid not in kw_mem_map:
                    score = cos_sim * 0.6
                    scored.append((mem.content, score))
        else:
            for mem in kw_memories:
                t = self._time_service.time_decay(mem.created_at)
                score = mem.importance.value * t
                if current_mood and mem.mood.value and mem.mood.value == current_mood:
                    score *= 1 + config.emotional_resonance_bonus
                scored.append((mem.content, score))

        scored.sort(key=lambda x: -x[1])
        return scored[:limit]

    # ═══════════════════════════════════════════════════════════
    # 通道 3：图谱骨架事实
    # ═══════════════════════════════════════════════════════════

    async def _channel_graph(
        self,
        user_id: str,
        user_input: str,
        limit: int = 2,
    ) -> tuple[list[tuple[str, float]], set[str]]:
        """通道 3：纯图边查询（relation 语义优先），输出「A」--[关系]--> 「B」。

        对应 retrieval.py _channel_graph_skeleton（行 245-306）。

        流程：
          1. 提取关键词
          2. 搜索图节点（IGraphRepository.search_nodes）
          3. 遍历种子节点，查询相连边（IGraphRepository.get_edges_by_node）
          4. 优先带 relation 的边，relation 为空则视为共现边
          5. 时间衰减：weight * 0.95^days, 最低 0.3
          6. 格式化语句，边对去重
          7. 收集链标签供后续 _build_graph_paths 使用

        Args:
            user_id: 用户标识
            user_input: 用户输入文本
            limit: 最大返回条数

        Returns:
            (results, chain_labels):
              results:      [(formatted_statement, score), ...] 按分数降序，最多 limit 条
              chain_labels: 图中涉及的节点标签集合，供 _build_graph_paths 使用
        """
        kws = self._keyword_service.extract_keywords(user_input, max_kw=4)

        # 1a. 先尝试直接关键词匹配
        seed_nodes: dict[int, GraphNode] = {}
        for kw in kws:
            nodes = await self._graph_repo.search_nodes(user_id, kw)
            for n in nodes:
                seed_nodes[n.node_id] = n

        # 1b. 再用图扩散扩展关键词（无论直接匹配有没有结果）
        #     这样既有直接匹配的节点，也有语义关联的节点
        diffusion_seed = kws if kws else [user_input[:20]]
        diffused = await self._graph_diffuse(user_id, diffusion_seed)
        if diffused:
            expanded_kws = list(set(
                (kws if kws else [])
                + [label for label, _ in diffused[:8]]
            ))
            for kw in expanded_kws:
                nodes = await self._graph_repo.search_nodes(user_id, kw)
                for n in nodes:
                    seed_nodes[n.node_id] = n

        # 1c. 如果还是没有种子节点，降级取用户最常出现的节点
        if not seed_nodes:
            user_nodes = await self._graph_repo.get_nodes_by_user(user_id)
            if user_nodes:
                user_nodes.sort(key=lambda n: n.freq, reverse=True)
                seed_nodes = {n.node_id: n for n in user_nodes[:3]}
            if not seed_nodes:
                logger.debug("_channel_graph: no nodes matched for user %s", user_id)
                return [], set()

        # 获取用户所有节点标签映射（供后续查 other node label）
        all_nodes = await self._graph_repo.get_nodes_by_user(user_id)
        node_label_map: dict[int, str] = {n.node_id: n.label for n in all_nodes}

        # 遍历种子节点的边
        now = self._clock.now()
        results: list[tuple[str, float]] = []
        seen_pairs: set[tuple[str, str]] = set()
        chain_labels: set[str] = set()

        for node_id in seed_nodes:
            edges = await self._graph_repo.get_edges_by_node(node_id)
            for edge in edges:
                # 确定另一端节点 ID
                other_id = (
                    edge.to_node_id
                    if edge.from_node_id == node_id
                    else edge.from_node_id
                )
                src_label = seed_nodes[node_id].label
                tgt_label = node_label_map.get(other_id, "?")

                # 时间衰减：对 datetime 做规范化减法
                w = edge.weight
                if edge.last_seen is not None:
                    ref = now.replace(tzinfo=None) if now.tzinfo else now
                    ls = edge.last_seen.replace(tzinfo=None) if edge.last_seen.tzinfo else edge.last_seen
                    days = max(0, (ref - ls).days)
                    w = max(w * (0.95 ** days), 0.3)

                # 边对去重
                pair = (min(src_label, tgt_label), max(src_label, tgt_label))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)

                chain_labels.add(src_label)
                chain_labels.add(tgt_label)

                if edge.relation:
                    stmt = (
                        f"「{src_label}」--[{edge.relation}]--> 「{tgt_label}」"
                    )
                else:
                    stmt = (
                        f"话题「{src_label}」与「{tgt_label}」"
                        f"常被一起提及（相关度 {w:.1f}）"
                    )
                score = w + 2.0 if edge.relation else w
                results.append((stmt, score))

        results.sort(key=lambda x: -x[1])
        return results[:limit], chain_labels

    # ═══════════════════════════════════════════════════════════
    # 私有辅助方法
    # ═══════════════════════════════════════════════════════════

    async def _graph_diffuse(
        self, user_id: str, seed_keywords: list[str]
    ) -> list[tuple[str, float]]:
        """图扩散：从关键词出发 BFS 激活相邻节点。

        对应 retrieval.py _graph_activation_diffusion（行 565-621）。

        使用 IGraphRepository.search_nodes 查找种子节点，
        再通过 IGraphRepository.bfs_diffuse 扩散。

        Args:
            user_id: 用户标识
            seed_keywords: 种子关键词列表

        Returns:
            [(label, activation_score), ...] 按激活分数降序
        """
        if not seed_keywords:
            return []

        seed_nodes: dict[int, GraphNode] = {}
        for kw in seed_keywords:
            nodes = await self._graph_repo.search_nodes(user_id, kw)
            for n in nodes:
                seed_nodes[n.node_id] = n

        if not seed_nodes:
            return []

        seed_ids = list(seed_nodes.keys())
        diffused = await self._graph_repo.bfs_diffuse(user_id, seed_ids)

        return [(d.label, d.score) for d in diffused]

    async def _build_graph_paths(self, user_id: str, chain_labels: set[str]) -> list[str]:
        """构建图谱关系链路径。

        对应 retrieval.py _build_graph_paths（行 308-370）。
        使用 IGraphRepository.get_chain_paths 替代手动 DFS。

        Args:
            user_id: 用户标识
            chain_labels: 从 _channel_graph 收集的节点标签集合

        Returns:
            路径描述字符串列表，最多 3 条
        """
        if not chain_labels:
            return []
        labels = list(chain_labels)
        return await self._graph_repo.get_chain_paths(user_id, labels)

    async def _session_warmup(
        self,
        user_id: str,
        session_id: Optional[str],
        turn_count: int,
    ) -> str:
        """新 session 预热：读取上一个 session 的 dangling + 最近活跃时间。

        对应 context.py _session_warmup（行 11-49）。
        仅在首次轮次时执行。

        Args:
            user_id: 用户标识
            session_id: 当前会话标识
            turn_count: 当前会话轮次

        Returns:
            预热文本字符串，无条件不满足时返回空字符串
        """
        if turn_count > 1:
            return ""
        if not user_id or not session_id:
            return ""

        prev_session = await self._session_query_repo.get_most_recent_excluding(
            session_id
        )
        if prev_session is None:
            return ""

        dangling = prev_session.dangling_threads
        last_active = prev_session.last_active
        if not dangling:
            return ""

        tmpl = self._template_provider.narrative_templates()
        dangling_label = tmpl.get("dangling_label", "未完成")
        parts: list[str] = []

        if dangling:
            if isinstance(dangling, dict):
                threads = dangling.get("threads", [])
                if threads:
                    parts.append(
                        dangling_label + "：" + "、".join(threads[:3])
                    )
            elif isinstance(dangling, list) and dangling:
                parts.append(
                    dangling_label + "：" + "、".join(dangling[:3])
                )

        if last_active is not None:
            ref = self._clock.now().replace(tzinfo=None) if self._clock.now().tzinfo else self._clock.now()
            la = last_active.replace(tzinfo=None) if last_active.tzinfo else last_active
            days = max(0, (ref - la).days)
            if days == 0:
                parts.append(tmpl.get("warmup_days_today", "距上次对话：今天"))
            elif days == 1:
                parts.append(tmpl.get("warmup_days_yesterday", "距上次对话：昨天"))
            else:
                parts.append(
                    tmpl.get("warmup_days_ago", "距上次对话：{days} 天前").format(days=days)
                )

        if not parts:
            return ""
        return "\n".join(f"  · {part}" for part in parts)

    async def _load_entities(self, user_id: str) -> list[dict]:
        """加载图谱实体关系。

        对应 core.py _load_long_term_context（行 155-162）中
        entities 部分的查询逻辑。

        从 IGraphRepository 获取用户节点和边，
        筛选带 relation 的边，构建实体条目。

        Returns:
            [{"name": str, "type": str, "relation": str, "fact": str}, ...]
            按 weight 降序，最多 10 条
        """
        all_nodes = await self._graph_repo.get_nodes_by_user(user_id)
        node_label_map: dict[int, str] = {n.node_id: n.label for n in all_nodes}
        node_type_map: dict[int, str] = {n.node_id: n.entity_type for n in all_nodes}

        collected: list[dict] = []
        seen_relations: set[tuple[str, str, str]] = set()

        for node in all_nodes:
            edges = await self._graph_repo.get_edges_by_node(node.node_id)
            for edge in edges:
                if not edge.relation:
                    continue

                target_id = (
                    edge.to_node_id
                    if edge.from_node_id == node.node_id
                    else edge.from_node_id
                )
                target_label = node_label_map.get(target_id)
                if target_label is None:
                    continue

                rel_key = (node.label, edge.relation, target_label)
                if rel_key in seen_relations:
                    continue
                seen_relations.add(rel_key)

                collected.append({
                    "name": node.label,
                    "type": node_type_map.get(node.node_id, "auto"),
                    "relation": edge.relation,
                    "fact": target_label,
                })
                if len(collected) >= 10:
                    break
            if len(collected) >= 10:
                break

        return collected

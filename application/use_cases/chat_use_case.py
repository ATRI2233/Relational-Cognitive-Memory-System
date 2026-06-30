"""
ChatUseCase — 对话流程编排 Use Case。

对应 core.py chat() 方法的完整流程（行 225-245）：
  1. 检索记忆（委托给 IRetrievalStrategy）
  2. 加载长期语境（identity_repo + graph_repo）
  3. 构建 prompt（委托给 IPromptBuilder）
  4. 调用 LLM（通过 ILLMService）
  5. 保存对话轮次（session_repo + memory_repo）
  6. 后处理（由 PostUpdateHandler 通过 EventBus 异步处理）
  7. 检查蒸馏条件（委托给 IDistillRunner）
  8. 发布 TurnSaved 事件

设计原则：
  - 零 try/except：异常向上抛给调用方
  - 纯编排：不持有任何数据库连接
  - 所有依赖通过构造函数显式注入（Protocol 接口）
  - 返回值使用 DTO，不暴露领域实体给调用层
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional, Protocol, runtime_checkable

from domain.entities.memory import SessionId, UserId
from domain.ports.clock import IClock
from domain.ports.graph_repo import IGraphRepository
from domain.ports.identity_repo import IIdentityRepository
from domain.ports.repositories import IMemoryRepository, ISessionRepository, IUserMappingRepository
from domain.events.memory_events import TurnSaved
from application.event_bus import EventBus


# ==============================================================
# 依赖接口协议（应用层协议，定义 Use Case 需要的外部能力）
# ==============================================================


@runtime_checkable
class ILLMService(Protocol):
    """LLM 服务接口 — 可注入 OpenAI / Mock / 其他后端。

    对应 core.py 中 LLMBackend.generate（行 231）。
    """

    async def generate(self, prompt: str) -> str:
        """生成回复。

        Args:
            prompt: 完整 prompt 字符串

        Returns:
            LLM 生成的文本回复

        Raises:
            Exception: LLM 调用失败时向上抛出，由调用方处理降级策略
        """
        ...


@runtime_checkable
class IRetrievalStrategy(Protocol):
    """三通道召回策略。

    对应 retrieval.py retrieve_memories 入口（行 49-67）。
    封装三通道融合召回 + 图谱关系链构建的完整流程。
    """

    async def retrieve(
        self,
        user_id: str,
        user_input: str,
        session_id: str,
    ) -> tuple[list[tuple[str, str]], list[str]]:
        """执行三通道融合召回。

        通道 1（时间重要性）+ 通道 2（语义共振）+ 通道 3（图骨架）
        叠加 fusion 去重加权排序。

        Args:
            user_id: 用户标识
            user_input: 用户输入文本，用作语义共振和关键词提取的源
            session_id: 会话标识，用于通道 1 的时间重要性锚定

        Returns:
            (memories, graph_paths):
              memories:      [(content, channel_tag), ...] 按加权分降序
                             channel_tag 为 'recent' / 'resonance' / 'skeleton'
              graph_paths:   图谱关系链描述字符串列表（供 prompt 注入用）
        """
        ...


@runtime_checkable
class IPromptBuilder(Protocol):
    """Prompt 构建器。

    对应 context.py prompt_compressor（行 204-267）。
    将召回的记忆、长期语境、用户输入组装为 LLM 可理解的最终 prompt。
    """

    async def build(
        self,
        user_id: str,
        session_id: str,
        user_input: str,
        memories: list[tuple[str, str]],
        long_term: dict[str, Any],
        graph_paths: list[str],
    ) -> str:
        """组装最终 prompt。

        格式结构（来自 context.py:259-266）：
          【当前心理状态】 + 【相关记忆】 + 【图谱关系链】
          + 【共同语境/用户特质/自我认同/雷区】 + 【底线】 + 用户输入模板

        Args:
            user_id: 用户标识
            session_id: 会话标识
            user_input: 用户输入文本
            memories: 三通道召回的记忆列表 [(content, tag), ...]
            long_term: 长期语境字典，含 identity_traits / preferences /
                       self_identity / boundaries / shared_contexts 等
            graph_paths: 图谱关系链描述列表

        Returns:
            组装后的完整 prompt 字符串
        """
        ...


@runtime_checkable
class IDistillRunner(Protocol):
    """蒸馏检查与执行器。

    对应 analysis.py 的：
      - check_distill_needed（行 491-529）：检查是否达到蒸馏触发条件
      - _run_distill_analysis（行 317-423）：执行 LLM 蒸馏分析并写库

    封装了读取 chat_history 构建快照、调用 LLM、解析 JSON 响应、
    写入 cognitive_distill / identity_memory / emotional_trace 的完整流程。
    """

    async def check_and_run(
        self,
        user_id: str,
        session_id: str,
        turn_count: int,
    ) -> None:
        """检查蒸馏条件，满足时执行 LLM 蒸馏分析。

        触发条件（对应 analysis.py:502-512）：
          - 距上次蒸馏超过 max_turns 轮（默认 30）
          - 距上次蒸馏超过 max_minutes 分钟（默认 60）

        满足条件后执行：
          1. 从 chat_history 读取快照文本
          2. 构建蒸馏 prompt 并调用 LLM
          3. 解析 JSON（摘要 + 9 维分析）
          4. 写入 cognitive_distill + identity_memory + emotional_trace
          5. 更新 session_state 中的 last_distill_turn/last_distill_at
          6. 触发记忆过期清理和图维护

        Args:
            user_id: 用户标识
            session_id: 会话标识
            turn_count: 当前对话轮次（由 ISessionRepository.increment_turn 返回）
        """
        ...


@runtime_checkable
class IEmotionalWordsSettings(Protocol):
    """情绪词表配置子接口。

    匹配 infrastructure.config.settings.EmotionalWordsSettings 的形状。
    """

    emotional_words: list[str]


@runtime_checkable
class ISettings(Protocol):
    """配置接口 — 缩减版，仅声明 ChatUseCase 需要的属性。

    对应 infrastructure.config.settings.Settings。
    实际注入时可以通过适配器或直接传递 Settings 实例（如果结构匹配）。

    仅需要 emotional_words 用于重要性计算。
    """

    emotional_words: IEmotionalWordsSettings


@runtime_checkable
class ISharedContextRepository(Protocol):
    """共享语境仓储协议。

    对应 shared_context 表的读写操作。
    记录用户与 AI 之间积累的共同背景知识。
    """

    async def get_recent(self, user_id: str, limit: int = 4) -> list[str]:
        """获取最近的共享语境列表。

        Args:
            user_id: 用户标识
            limit: 最大返回条数

        Returns:
            context_body 字符串列表，按时间降序
        """
        ...


# ==============================================================
# 数据传输对象（DTO）
# ==============================================================


@dataclass
class ChatRequest:
    """对话请求 DTO。

    Attributes:
        user_id: 用户标识
        session_id: 会话标识
        user_input: 用户本次输入的文本
        sender_name: 发送者昵称（可选），用于自动注册用户映射
    """

    user_id: str
    session_id: str
    user_input: str
    sender_name: str = ""


@dataclass
class ChatResponse:
    """对话响应 DTO。

    Attributes:
        reply: LLM 生成的回复文本
        turn_number: 当前对话轮次（1-based）
    """

    reply: str
    turn_number: int


# ==============================================================
# Use Case
# ==============================================================


class ChatUseCase:
    """对话流程编排 Use Case。

    职责：编排完整对话管线（pipeline），不持有任何数据库连接。
    所有依赖通过构造函数显式注入（依赖倒置原则）。

    管线步骤:
      1. 三通道召回记忆 → IRetrievalStrategy.retrieve()
      2. 加载长期语境 → _build_long_term_context()（使用 IIdentityRepository + IGraphRepository）
      3. 构建 prompt → IPromptBuilder.build()
      4. 核心否决 → _core_veto()（替换说教/命令式表述）
      5. LLM 生成回复 → ILLMService.generate()
      6. 原子递增轮次 → ISessionRepository.increment_turn()
      7. 计算重要性 → _calc_importance()
      8. 写入对话记录 → IMemoryRepository.save_turn()
      9. 发布 TurnSaved 事件 → EventBus.publish()
         （后处理 + 蒸馏由 PostUpdateHandler / DistillCheckerHandler 异步执行）

    异常处理：零 try/except — 所有异常向上抛给调用方，
    由外层适配器（如 CLI / AstrBot 插件 / FastAPI 路由）负责捕获与降级。

    Usage::

        use_case = ChatUseCase(
            memory_repo=memory_repo,
            session_repo=session_repo,
            identity_repo=identity_repo,
            graph_repo=graph_repo,
            llm=llm_service,
            clock=system_clock,
            event_bus=event_bus,
            settings=settings,
            retrieval_strategy=three_channel_strategy,
            prompt_builder=narrative_compressor,
            distill_runner=distill_checker,
        )
        response = await use_case.execute(ChatRequest(
            user_id="user_123",
            session_id="session_456",
            user_input="今天心情不太好",
        ))
    """

    def __init__(
        self,
        memory_repo: IMemoryRepository,
        session_repo: ISessionRepository,
        identity_repo: IIdentityRepository,
        graph_repo: IGraphRepository,
        llm: ILLMService,
        clock: IClock,
        event_bus: EventBus,
        settings: ISettings,
        retrieval_strategy: IRetrievalStrategy,
        prompt_builder: IPromptBuilder,
        distill_runner: IDistillRunner,
        shared_context_repo: Optional[ISharedContextRepository] = None,
        user_mapping_repo: Optional[IUserMappingRepository] = None,
    ) -> None:
        """初始化 ChatUseCase。

        Args:
            memory_repo: 认知记忆仓储（cognitive_distill / chat_history 写入）
            session_repo: 会话状态仓储（session_state 管理）
            identity_repo: 用户身份仓储（identity_memory 查询）
            graph_repo: 图谱仓储（memory_graph_nodes / edges 查询）
            llm: LLM 服务（负责 prompt → reply 的生成）
            clock: 时间源（消除对 datetime.now() 的直接依赖）
            event_bus: 事件总线（发布 TurnSaved 触发后处理）
            settings: 配置（提供情绪词表等业务参数）
            retrieval_strategy: 三通道召回策略
            prompt_builder: Prompt 构建器
            distill_runner: 蒸馏检查与执行器
            shared_context_repo: 共享语境仓储（可选，提供后可在
                                 long_term_context 中填入 shared_contexts）
            user_mapping_repo: 用户映射仓储（可选，提供后自动注册 sender_name
                               到 user_mappings 表）
        """
        self._memory_repo = memory_repo
        self._session_repo = session_repo
        self._identity_repo = identity_repo
        self._graph_repo = graph_repo
        self._llm = llm
        self._clock = clock
        self._event_bus = event_bus
        self._settings = settings
        self._retrieval_strategy = retrieval_strategy
        self._prompt_builder = prompt_builder
        self._distill_runner = distill_runner
        self._shared_context_repo = shared_context_repo
        self._user_mapping_repo = user_mapping_repo

    # ── 公有入口 ──────────────────────────────────────────────

    async def save_turn_only(self, request: ChatRequest, reply: str) -> int:
        """仅持久化对话记录，不调用 LLM。

        用于 AstrBot 等适配器场景 —— LLM 回复已由外部生成，
        execute() 中的 LLM 调用步骤需要跳过。

        执行步骤（对应 execute() 中的步骤 6-9）：
          1. 原子递增轮次
          2. 计算重要性
          3. 写入对话记录
          4. 自动注册 sender_name 到 user_mappings
          5. 发布 TurnSaved 事件（触发后处理 Handler 链）

        Args:
            request: 对话请求 DTO（含 user_id/session_id/user_input/sender_name）
            reply: 已有 LLM 回复文本（由外部生成，此处直接持久化）

        Returns:
            当前对话轮次号

        Raises:
            ValueError: 参数校验失败
        """
        # ── 参数校验 ──
        if not request.user_id:
            raise ValueError("user_id 不能为空")
        if not request.session_id:
            raise ValueError("session_id 不能为空")
        if not request.user_input or not request.user_input.strip():
            raise ValueError("user_input 不能为空")

        session_id = SessionId(request.session_id)

        # Step 6: 原子递增轮次 —— core.py:238 / session.py:21
        turn_number = await self._session_repo.increment_turn(session_id)

        # Step 7: 计算本轮重要性 —— session.py:22-29
        importance = self._calc_importance(request.user_input)

        # Step 8: 写入对话记录 —— session.py:11-55
        await self._memory_repo.save_turn(
            session_id,
            request.user_input,
            reply,
            user_id=UserId(request.user_id) if request.user_id else None,
            sender_name=request.sender_name,
            importance=importance,
        )

        # Step 8b: 自动注册 sender_name 到 user_mappings
        if request.sender_name and self._user_mapping_repo is not None:
            await self._user_mapping_repo.upsert_mapping(
                request.session_id,
                request.user_id,
                request.sender_name,
                source="nickname",
            )

        # Step 9: 发布 TurnSaved 事件
        # 触发后处理 Handler 链：
        #   PostUpdateHandler → DistillCheckerHandler → DelayedEmbedHandler
        # 蒸馏检查由 DistillCheckerHandler 在事件中触发，不再同步执行。
        await self._event_bus.publish(TurnSaved(
            user_id=request.user_id,
            session_id=request.session_id,
            turn_number=turn_number,
            user_input=request.user_input,
            reply=reply,
            sender_name=request.sender_name,
            occurred_at=self._clock.now(),
        ))

        return turn_number

    async def execute(self, request: ChatRequest) -> ChatResponse:
        """执行完整对话流程。

        步骤（对应 core.py:225-245 的完整管线）：
          1. 三通道召回记忆（retrieval.py:49）
          2. 加载长期语境（core.py:150）
          3. 构建 prompt（context.py:204）
          4. 核心否决（utils.py:53）
          5. LLM 生成回复（core.py:231）
          6. 原子递增轮次（session.py:21 → ISessionRepository）
          7. 计算重要性（session.py:22-29）
          8. 写入对话记录（session.py:11 → IMemoryRepository）
          9. 后处理 + 蒸馏由 EventBus Handler 异步触发

        Args:
            request: 对话请求 DTO

        Returns:
            ChatResponse 包含 LLM 回复和当前轮次号

        Raises:
            ValueError: 参数校验失败（空 user_id / session_id / user_input）
            Exception: LLM 调用或其他基础设施异常，不在此处捕获
        """
        # ── 参数校验 ──
        if not request.user_id:
            raise ValueError("user_id 不能为空")
        if not request.session_id:
            raise ValueError("session_id 不能为空")
        if not request.user_input or not request.user_input.strip():
            raise ValueError("user_input 不能为空")

        session_id = SessionId(request.session_id)

        # Step 1: 三通道召回记忆 —— retrieval.py:49
        # retrieve() 返回 (memories, graph_paths)
        # memories: [(content, channel_tag), ...] 三通道融合去重后的结果
        memories, graph_paths = await self._retrieval_strategy.retrieve(
            request.user_id,
            request.user_input,
            request.session_id,
        )

        # Step 2: 加载长期语境 —— core.py:150
        # 从 identity_repo（特质/偏好/自我认知/边界）和 graph_repo 构建
        long_term = await self._build_long_term_context(request.user_id)

        # Step 3: 构建 prompt —— context.py:204
        # prompt_builder 将 memory_block + graph_paths + long_term_block
        # + mood + bottom_line 组装为 LLM 输入
        prompt = await self._prompt_builder.build(
            request.user_id,
            request.session_id,
            request.user_input,
            memories,
            long_term,
            graph_paths,
        )

        # Step 4: 核心否决 —— utils.py:53
        # 将 prompt 中的说教/命令式表述替换为温和建议
        # 如 "你应该" → "或许可以试试"
        prompt = self._core_veto(prompt)

        # Step 5: LLM 生成回复 —— core.py:231
        # 直接调用，异常向上抛给调用方处理降级
        reply = await self._llm.generate(prompt)

        # Step 6: 原子递增轮次 —— core.py:238 / session.py:21
        # increment_turn 原子递增 turn_count 并返回新值
        # 如果 session 尚不存在则自动初始化为 turn_count=1
        turn_number = await self._session_repo.increment_turn(session_id)

        # Step 7: 计算本轮重要性 —— session.py:22-29
        # 基于情绪词命中数和输入长度计算 [0.3, 0.8] 范围的重要性
        importance = self._calc_importance(request.user_input)

        # Step 8: 写入对话记录 —— session.py:11-55
        # 同时写入 user 和 assistant 两条 chat_history 记录
        # 自动注册 sender_name 到 user_mappings
        await self._memory_repo.save_turn(
            session_id,
            request.user_input,
            reply,
            user_id=UserId(request.user_id) if request.user_id else None,
            sender_name=request.sender_name,
            importance=importance,
        )

        # Step 8b: 自动注册 sender_name 到 user_mappings（从 session.py:33-37 移入）
        # 对应原 save_turn 中 INSERT OR IGNORE INTO user_mappings
        if request.sender_name and self._user_mapping_repo is not None:
            await self._user_mapping_repo.upsert_mapping(
                request.session_id,
                request.user_id,
                request.sender_name,
                source="nickname",
            )

        # 后处理（更新 last_active / 确保 identity / dangling 过期归档）
        # 由 PostUpdateHandler 通过 EventBus 异步处理，此处不再内联执行。

        # Step 9: 发布 TurnSaved 事件
        # 触发后处理 Handler 链：
        #   PostUpdateHandler → DistillCheckerHandler → DelayedEmbedHandler
        # 蒸馏检查由 DistillCheckerHandler 在事件中触发，不再同步执行。
        await self._event_bus.publish(TurnSaved(
            user_id=request.user_id,
            session_id=request.session_id,
            turn_number=turn_number,
            user_input=request.user_input,
            reply=reply,
            sender_name=request.sender_name,
            occurred_at=self._clock.now(),
        ))

        return ChatResponse(reply=reply, turn_number=turn_number)

    # ── 私有方法 ──────────────────────────────────────────────

    async def _build_long_term_context(self, user_id: str) -> dict[str, Any]:
        """加载长期语境 — 对应 core.py _load_long_term_context（行 150-190）。

        从 identity_repo、graph_repo 和 shared_context_repo 构建长期语境字典，
        供 prompt_builder 和 distill_runner 使用。

        Returns:
            {
                'identity_traits':  [str, ...],       # 用户特质文本列表
                'trait_details':    [dict, ...],       # 特质详情（含 strength / count）
                'preferences':      {'likes': [...], 'dislikes': [...]},
                'self_identity':    [str, ...],        # 自我认知描述
                'boundaries':       [{"description": str}, ...],  # 边界/雷区描述
                'entities':         [dict, ...],       # 图谱实体（label / entity_type / freq）
                'shared_contexts':  [str, ...],        # 共享语境文本列表
            }
        """
        # ── 图谱实体 —— 从 graph_repo 加载所有节点 ──
        nodes = await self._graph_repo.get_nodes_by_user(user_id)
        entities = [
            {"label": n.label, "entity_type": n.entity_type, "freq": n.freq}
            for n in nodes
        ]

        # ── 共享语境 —— 从 shared_context_repo 加载最新条目 ──
        if self._shared_context_repo is not None:
            shared_contexts = await self._shared_context_repo.get_recent(user_id, limit=4)
        else:
            shared_contexts = []

        # ── 身份信息 —— 从 identity_repo 加载 ──
        identity = await self._identity_repo.get(user_id)
        if identity is None:
            return {
                "identity_traits": [],
                "trait_details": [],
                "preferences": {},
                "self_identity": [],
                "boundaries": [],
                "entities": entities,
                "shared_contexts": shared_contexts,
            }

        trait_details = [
            {"text": t.text, "strength": t.strength, "count": t.count}
            for t in identity.traits
            if t.text and t.strength > 0
        ]

        return {
            "identity_traits": [t["text"] for t in trait_details],
            "trait_details": trait_details,
            "preferences": {
                "likes": identity.preferences.likes,
                "dislikes": identity.preferences.dislikes,
            },
            "self_identity": identity.self_identity[:],
            "boundaries": [{"description": b.description} for b in identity.boundaries],
            "entities": entities,
            "shared_contexts": shared_contexts,
        }

    def _calc_importance(self, user_input: str) -> float:
        """计算本轮重要性 — 对应 session.py save_turn 中的重要性逻辑（行 22-29）。

        规则：
          - 基础值 0.3
          - 每命中一个情绪词 +0.1，上限 0.8
          - 输入超过 50 字额外 +0.1，上限 0.8

        Args:
            user_input: 用户输入文本

        Returns:
            [0.3, 0.8] 范围内的浮点数
        """
        emotional_words: list[str] = self._settings.emotional_words.emotional_words
        importance = 0.3
        hits = sum(1 for w in emotional_words if w in user_input)
        if hits:
            importance = min(0.3 + hits * 0.1, 0.8)
        if len(user_input) > 50:
            importance = min(importance + 0.1, 0.8)
        return importance

    @staticmethod
    def _core_veto(prompt: str) -> str:
        """核心否决 — 对应 utils.py _core_veto（行 53-58）。

        将 prompt 中的说教/命令式表述替换为温和建议，
        每轮只替换第一种命中的模式，避免过度改写。

        替换映射：
          "你应该" → "或许可以试试"
          "你必须" → "或许可以试试"
          "我教你" → "或许可以试试"
          "听我说" → "或许可以试试"
          "你这样不对" → "或许可以试试"

        Args:
            prompt: 原始 prompt 字符串

        Returns:
            替换后的 prompt 字符串（未命中则返回原串）
        """
        _VETO_PAIRS: list[tuple[str, str]] = [
            ("你应该", "或许可以试试"),
            ("你必须", "或许可以试试"),
            ("我教你", "或许可以试试"),
            ("听我说", "或许可以试试"),
            ("你这样不对", "或许可以试试"),
        ]
        for old, new in _VETO_PAIRS:
            if old in prompt:
                prompt = prompt.replace(old, new, 1)
                break
        return prompt

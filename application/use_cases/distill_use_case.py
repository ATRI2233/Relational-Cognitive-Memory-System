"""
DistillUseCase — 蒸馏分析编排 Use Case。

对应 analysis.py 的完整蒸馏流程（check_distill_needed → _run_distill_analysis）：
  1. 检查是否满足蒸馏条件（轮数阈值 / 时间间隔阈值）
  2. 构建对话快照
  3. 加载长期记忆数据（身份、特质、偏好、雷区）
  4. 构建 prompt（群聊/私聊双模板, 三种人格风格）
  5. 调用 LLM 做分析（响应格式 JSON）
  6. 清洗并解析 LLM 输出（移除 <think> 块和代码围栏）
  7. 写入蒸馏摘要（cognitive_distill, 带 mood/mood_intensity）
  8. 写入 9 维分析结果：
       - 会话状态 & 用户立场
       - 身份特质合并/衰减/截断
       - 结构化身份字段（偏好、自我认知）
       - 共享梗/上下文
       - 边界/雷区（覆盖写）
       - 悬案线程归档
       - 实体图谱（节点保底 + 语义边双向 + 同轮共现）
       - 关键事实（permanent 保底 0.5 上限 3, transient 无保底上限 5）
  9. 悬案归档
  10. 过期记忆清理 & 低重要性条目截断
  11. 图谱衰减维护
  12. 向量嵌入（蒸馏摘要 + 新事实 + 悬案）
  13. 发布 MemoryDistilled / EmbeddingDone / MemoryExpired 事件

本 Use Case 不持有 DB 连接, 所有依赖通过构造函数注入。
不包含任何 try/except, 异常由调用方处理。
"""
from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, Optional, Protocol, runtime_checkable

from domain.entities.identity import Boundary, Identity, Preferences, Trait
from domain.entities.memory import Importance, Memory, MemoryId, Mood, SessionId, UserId
from domain.entities.session import Session, TurnRecord
from domain.entities.graph import GraphNode, GraphEdge
from domain.events.memory_events import (
    EmbeddingDone,
    MemoryDistilled,
    MemoryExpired,
)
from domain.ports.clock import IClock
from domain.ports.graph_repo import IGraphRepository
from domain.ports.identity_repo import IIdentityRepository
from domain.ports.repositories import IMemoryRepository, ISessionRepository
from application.analysis_writer import AnalysisWriter


# ════════════════════════════════════════════════════════════════════
# Protocols — 端口抽象（含分析设置协议，替代 infrastructure 层直接引用）
# ════════════════════════════════════════════════════════════════════


class IMaintenanceSettings(Protocol):
    """Protocol for maintenance sub-settings used by DistillUseCase."""
    keep_rule_summary: int
    max_trait_count: int


class IAnalysisSettings(Protocol):
    """Protocol for analysis settings used by DistillUseCase.

    替代直接从 infrastructure.config.settings 导入 AnalysisSettings，
    遵循依赖倒置原则。
    """
    personality_type: str
    max_turns: int
    max_minutes: int
    distill_min_turns: int
    max_snapshot_lines: int
    distill_entry_importance: float
    archived_dangling_importance: float
    keep_rule_summary: int
    maintenance: IMaintenanceSettings
    permanent_fact_max: int
    transient_fact_max: int


# ════════════════════════════════════════════════════════════════════
# DTOs
# ════════════════════════════════════════════════════════════════════


@dataclass
class DistillRequest:
    """蒸馏请求数据传输对象。

    Attributes:
        user_id:   用户标识
        session_id: 会话标识
    """

    user_id: str
    session_id: str


@dataclass
class DistillResult:
    """蒸馏结果数据传输对象。

    Attributes:
        triggered: 是否实际触发了蒸馏
        memory_id: 蒸馏摘要记录 ID（触发时有效）
        keylabel:  核心主题标签（触发时有效）
    """

    triggered: bool = False
    memory_id: int = 0
    keylabel: str = ""


# ════════════════════════════════════════════════════════════════════
# Protocols — 端口抽象
# ════════════════════════════════════════════════════════════════════


@runtime_checkable
class ILLMService(Protocol):
    """LLM 调用接口。

    封装对 LLM chat/completions 的调用, 返回纯文本响应。
    不限定底层实现（OpenAI / AstrBot / Mock）。
    """

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        """发送 prompt 并获取 LLM 回复。

        Args:
            prompt: 完整 prompt 文本
            **kwargs: 额外参数（如 model 覆盖）

        Returns:
            LLM 回复文本（空字符串表示调用失败或无内容）
        """
        ...


@runtime_checkable
class IEmbeddingService(Protocol):
    """文本向量化接口。

    将文本转换为 float 向量, 供余弦相似度检索。
    """

    async def embed(self, text: str) -> list[float]:
        """将文本转换为向量嵌入。

        Args:
            text: 待转换文本

        Returns:
            嵌入向量, 长度为 0 表示失败
        """
        ...


@runtime_checkable
class IEventBus(Protocol):
    """事件总线接口。

    发布/订阅模式, 解耦 Use Case 与后处理 Handler。
    """

    async def publish(self, event: object) -> None:
        """发布领域事件。

        Args:
            event: 领域事件实例（如 TurnSaved, MemoryDistilled）
        """
        ...


@runtime_checkable
class IDistillPromptBuilder(Protocol):
    """蒸馏 prompt 构建器接口。

    封装 analysis.py _build_distill_prompt 的逻辑, 将对话快照和长期记忆
    组装为 LLM 蒸馏分析 prompt。支持私聊/群聊双模板和三种人格风格。
    """

    def build(
        self,
        snapshot_text: str,
        long_term: dict,
        persona_name: str = "Bot",
        personality_style: str = "",
        is_group: bool = False,
        personality_type: str = "default",
    ) -> str:
        """构建完整蒸馏 prompt。

        Args:
            snapshot_text: 对话快照文本
            long_term: 长期记忆字典（identity_traits, preferences, self_identity, boundaries）
            persona_name: Bot 名称
            personality_style: 人格风格描述文本
            is_group: 是否为群聊
            personality_type: 人格类型标识（default / cute / professional）

        Returns:
            完整 prompt 字符串
        """
        ...


# ════════════════════════════════════════════════════════════════════
# 纯函数工具
# ════════════════════════════════════════════════════════════════════


def strip_llm_fences(content: str) -> str:
    """移除 LLM 响应中的 <think> 推理块和 markdown 代码围栏。

    对应 analysis.py _strip_llm_fences（行 607-615）。

    Args:
        content: 原始 LLM 响应文本

    Returns:
        清洗后的纯文本（移除 <think>...</think> 和 ``` 围栏）
    """
    content = re.sub(r"<think>[\s\S]*?</think>\s*", "", content)
    content = re.sub(r"```(?:json)?\s*", "", content)
    return content.strip()


def normalize_analysis_data(data: dict) -> dict:
    """将可能含 dict 的字段归一化为统一字符串列表。

    对应 analysis.py _apply_analysis 行 54-96 的数据兼容逻辑。
    处理的字段：traits_updates, speech_quirks, dangling_threads, key_facts。

    Args:
        data: 原始分析数据字典

    Returns:
        归一化后的数据字典（所有列表字段均为纯字符串列表）
    """
    traits_updates: list[str] = []
    for t in data.get("traits_updates", []) or []:
        if isinstance(t, dict):
            trait = t.get("trait") or t.get("t") or None
            if trait:
                traits_updates.append(trait)
        elif isinstance(t, str):
            traits_updates.append(t)

    speech_quirks: list[str] = []
    for q in data.get("speech_quirks", []) or []:
        if isinstance(q, dict):
            quirk = q.get("quirk") or q.get("q") or q.get("text") or None
            if quirk:
                speech_quirks.append(quirk)
        elif isinstance(q, str):
            speech_quirks.append(q)

    dangling_threads: list[str] = []
    for dt in data.get("dangling_threads", []) or []:
        if isinstance(dt, dict):
            d_content = dt.get("content") or dt.get("text") or None
            if d_content:
                dangling_threads.append(d_content)
        elif isinstance(dt, str):
            dangling_threads.append(dt)

    key_facts_list: list[str] = []
    for kf in data.get("key_facts", []) or []:
        if isinstance(kf, dict):
            c = kf.get("content") or kf.get("text") or None
            if c:
                key_facts_list.append(c)
        elif isinstance(kf, str):
            key_facts_list.append(kf)

    norm = dict(data)
    norm["traits_updates"] = traits_updates
    norm["speech_quirks"] = speech_quirks
    norm["dangling_threads"] = dangling_threads
    norm["key_facts"] = key_facts_list
    return norm


# ════════════════════════════════════════════════════════════════════
# DefaultDistillPromptBuilder — 默认 prompt 构建器
# ════════════════════════════════════════════════════════════════════


class DefaultDistillPromptBuilder:
    """默认蒸馏 prompt 构建器。

    对应 analysis.py _build_distill_prompt（行 424-489）的逻辑。
    从 prompts.json 加载人格变体和输出格式模板, 组装完整 prompt。

    使用方式：
        builder = DefaultDistillPromptBuilder({"distill_prompt": {...}})
        prompt = builder.build(snapshot_text, long_term, ...)
    """

    def __init__(self, prompts: dict | None = None):
        self._prompts = prompts or {}

    def build(
        self,
        snapshot_text: str,
        long_term: dict,
        persona_name: str = "Bot",
        personality_style: str = "",
        is_group: bool = False,
        personality_type: str = "default",
    ) -> str:
        """构建完整蒸馏 prompt。

        对应 analysis.py _build_distill_prompt 行 424-489。
        """
        # ── lt_hint: 已有信息提示 ──  对应行 426-435
        lt_hint = ""
        if long_term:
            if long_term.get("identity_traits"):
                lt_hint += f"\n已知特质: {json.dumps(long_term['identity_traits'], ensure_ascii=False)}"
            if long_term.get("preferences"):
                lt_hint += f"\n已知喜好: {json.dumps(long_term['preferences'], ensure_ascii=False)}"
            if long_term.get("self_identity"):
                lt_hint += f"\n自我认同: {json.dumps(long_term['self_identity'], ensure_ascii=False)}"
            if long_term.get("boundaries"):
                lt_hint += f"\n已知雷区: {json.dumps(long_term['boundaries'], ensure_ascii=False)}"

        # ── 从 prompts 加载人格变体 ──  对应行 437-445
        dp = self._prompts.get("distill_prompt", {})
        ptype_key = personality_type if personality_type in ("cute", "professional") else "default"
        mode_key = "group" if is_group else "private"
        tpl = dp.get(ptype_key, {}).get(mode_key, {}) or dp.get("default", {}).get("private", {})
        preamble = tpl.get("preamble", "").format(persona_name=persona_name)
        content_instruction = tpl.get("content_instruction", "").format(persona_name=persona_name)
        first_stage = tpl.get("first_stage", "")
        extra_rules = tpl.get("extra_rules", "")

        # ── 输出格式模板 ──  对应行 447-470
        intro = dp.get("output_intro", "").replace("{first_stage}", first_stage)
        schema = dict(dp.get("output_schema", {}))
        rules = list(dp.get("output_rules", []))
        footer = dp.get("output_footer", "只输出 JSON。")
        lt_hint_label = dp.get("lt_hint_label", "已了解的信息（后续字段需与之去重）：")

        # 群聊才保留 participants 字段 ──  对应行 454-458
        if not is_group:
            analysis_schema = schema.get("analysis", {})
            if isinstance(analysis_schema, dict):
                analysis_schema.pop("participants", None)

        # schema → 格式化 JSON ──  对应行 460-465
        schema_str = json.dumps(schema, ensure_ascii=False, indent=2)
        schema_str = schema_str.replace("{content_instruction}", content_instruction)
        schema_str = schema_str.replace("{persona_name}", persona_name)

        # 组装规则 ──  对应行 467-470
        rules_str = "\n".join(r.replace("{persona_name}", persona_name) for r in rules)
        if extra_rules:
            rules_str += "\n" + extra_rules

        return f"""[SYSTEM]
{preamble}

角色风格：
{personality_style or '（无特殊设定）'}

[USER]
对话快照：
{snapshot_text}

{lt_hint_label}
{lt_hint}

{intro}{schema_str}

规则：
{rules_str}
{footer}"""


# ════════════════════════════════════════════════════════════════════
# DistillUseCase — 蒸馏分析编排
# ════════════════════════════════════════════════════════════════════


class DistillUseCase:
    """蒸馏分析编排 Use Case。

    对应 analysis.py 的 check_distill_needed → _run_distill_analysis 管线。
    不持有 DB 连接, 所有依赖通过构造函数注入。

    典型管线（check_and_run 内部步骤）：
        check_distill_needed → 构建快照 → 构建 prompt → LLM 分析
        → 解析 JSON → 写入蒸馏摘要 → 写入 9 维分析 → 图谱维护
        → 向量嵌入 → 发布事件
    """

    # 对应 analysis.py _INVERSE_RELATIONS（行 15-29）
    _INVERSE_RELATIONS: dict[str, str] = {
        "朋友": "朋友",
        "同事": "同事",
        "喜欢": "被喜欢",
        "喜欢玩": "被喜欢玩",
        "讨论过": "被讨论过",
        "讨厌": "被讨厌",
        "居住": "居住地于",
        "属于": "包含",
        "对立": "对立",
        "同类": "同类",
        "使用": "被使用",
        "提及": "被提及",
        "养了": "主人是",
    }

    def __init__(
        self,
        session_repo: ISessionRepository,
        memory_repo: IMemoryRepository,
        identity_repo: IIdentityRepository,
        graph_repo: IGraphRepository,
        llm: ILLMService,
        embedding: IEmbeddingService,
        clock: IClock,
        event_bus: IEventBus,
        settings: IAnalysisSettings,
        prompt_builder: IDistillPromptBuilder | None = None,
        persona_name: str = "Bot",
        # ── 可选回调: 补齐现有 Repository Protocol 未覆盖的操作 ──
        chat_history_provider: Callable[
            [SessionId, int, int], Awaitable[list[TurnRecord]]
        ] | None = None,
        save_shared_context_fn: Callable[
            [str, str, str], Awaitable[None]
        ] | None = None,
        trim_low_importance_fn: Callable[
            [UserId, int], Awaitable[int]
        ] | None = None,
        # ── AnalysisWriter: 9 维分析写入服务（替代内部 _apply_nine_dim_analysis） ──
        analysis_writer: AnalysisWriter | None = None,
        # ── analysis_raw 审计日志回调 ──
        save_raw_analysis_fn: Callable[
            [str, str, str], Awaitable[int]
        ] | None = None,
        mark_raw_parsed_fn: Callable[
            [int], Awaitable[None]
        ] | None = None,
    ):
        self._session_repo = session_repo
        self._memory_repo = memory_repo
        self._identity_repo = identity_repo
        self._graph_repo = graph_repo
        self._llm = llm
        self._embedding = embedding
        self._clock = clock
        self._event_bus = event_bus
        self._settings = settings
        self._prompt_builder = prompt_builder or DefaultDistillPromptBuilder()
        self._persona_name = persona_name

        # 可选回调（当 Repository Protocol 尚缺对应方法时注入）
        self._chat_history_provider = chat_history_provider
        self._save_shared_context_fn = save_shared_context_fn
        self._trim_low_importance_fn = trim_low_importance_fn

        # AnalysisWriter 和 raw analysis 审计（Task 2 & 3）
        self._analysis_writer = analysis_writer
        self._save_raw_analysis_fn = save_raw_analysis_fn
        self._mark_raw_parsed_fn = mark_raw_parsed_fn

    # ────────────────────────────────────────────────────────────
    # 公共入口
    # ────────────────────────────────────────────────────────────

    async def check_and_run(self, user_id: str, session_id: str) -> DistillResult:
        """检查条件并执行蒸馏。

        对应 analysis.py 的 check_distill_needed + _run_distill_analysis 完整管线。

        步骤：
          1. 检查蒸馏条件（_check_distill_needed）
          2. 不满足 → 返回 DistillResult(triggered=False)
          3. 加载长期记忆（_load_long_term）
          4. 构建 prompt → 调用 LLM（_call_llm）
          5. 解析 JSON（_parse_llm_response）
          6. 更新会话蒸馏进度（session_repo）
          7. 写入蒸馏摘要（_write_distill_summary）
          8. 写入 9 维分析（_apply_nine_dim_analysis）
          9. 发布事件（_publish_events）

        Args:
            user_id:   用户标识
            session_id: 会话标识

        Returns:
            DistillResult(triggered=False) 表示未触发；
            DistillResult(triggered=True, memory_id=..., keylabel=...) 表示蒸馏完成
        """
        # Step 1-2: 检查蒸馏条件 — 对应 check_distill_needed（行 491-529）
        needed = await self._check_distill_needed(session_id)
        if needed is None:
            return DistillResult(triggered=False)

        last_turn, turn_count, snapshot_text, senders = needed
        now = self._clock.now()

        # Step 3: 加载长期记忆 — 对应 _run_distill_analysis 行 327-331
        long_term = await self._load_long_term(user_id)
        personality_style = self._build_personality_style(long_term)

        # 群聊检测 — 对应行 333-335
        other_senders = [s for s in senders if s != self._persona_name]
        is_group = len(other_senders) > 1

        # Step 4: 构建 prompt → 调用 LLM — 对应行 337-356
        prompt = self._prompt_builder.build(
            snapshot_text=snapshot_text,
            long_term=long_term,
            persona_name=self._persona_name,
            personality_style=personality_style,
            is_group=is_group,
            personality_type=self._settings.personality_type,
        )
        llm_response = await self._call_llm(prompt)
        if not llm_response:
            return DistillResult(triggered=False)

        # Step 4b: 保存原始 LLM 响应到 analysis_raw 审计日志（Task 2）
        raw_id = await self._save_raw_analysis(user_id, session_id, llm_response)

        # Step 5: 解析 JSON — 对应行 374-402
        parsed = self._parse_llm_response(llm_response)
        if parsed is None:
            return DistillResult(triggered=False)
        content, keylabel, summary, analysis_data = parsed

        # Step 5b: 标记原始响应为已解析（Task 2）
        if raw_id > 0 and self._mark_raw_parsed_fn is not None:
            await self._mark_raw_parsed_fn(raw_id)

        # Step 7: 写入蒸馏摘要 — 对应 _apply_distill（行 531-578）
        memory_result = await self._write_distill_summary(
            user_id, session_id, content, keylabel,
            summary, analysis_data, now, turn_count,
        )
        memory_id, new_entries = memory_result

        # Step 8: 写入 9 维分析 — 对应 _apply_analysis（行 42-313）
        analysis_entries = await self._apply_nine_dim_analysis(
            user_id, session_id, analysis_data, now,
        )

        # 合并所有待嵌入条目
        all_entries = new_entries + analysis_entries

        # Step 9: 发布事件
        await self._publish_events(
            user_id, session_id, memory_id, content,
            keylabel, analysis_data, all_entries, now,
        )

        return DistillResult(
            triggered=True,
            memory_id=memory_id.value if isinstance(memory_id, MemoryId) else int(memory_id),
            keylabel=keylabel,
        )

    # ────────────────────────────────────────────────────────────
    # 内部方法：条件检查 + 快照构建
    # ────────────────────────────────────────────────────────────

    async def _check_distill_needed(
        self, session_id: str,
    ) -> tuple[int, int, str, list[str]] | None:
        """检查是否触发蒸馏。

        对应 analysis.py check_distill_needed（行 491-529）。
        条件（满足任一即可）：
          - 从上次蒸馏起新增轮次 >= max_turns（默认 30）
          - 从上次蒸馏起经过的分钟数 >= max_minutes（默认 60）
        附加条件：快照至少 6 条消息。

        Args:
            session_id: 会话标识

        Returns:
            None 表示不满足条件；
            否则返回 (last_turn, turn_count, snapshot_text, sender_names)
        """
        # 行 493-496: 查询会话蒸馏进度
        session = await self._session_repo.get(SessionId(session_id))
        if session is None:
            return None

        now = self._clock.now()
        turn_count = session.turn_count
        last_turn = session.last_distill_turn
        last_at = session.last_distill_at

        # 行 500-501: 无轮次不蒸馏
        if turn_count == 0:
            return None

        # 行 502-510: 条件判断
        triggered = False
        if turn_count - last_turn >= self._settings.max_turns:
            triggered = True
        if not triggered and last_at is not None:
            elapsed = (now - last_at).total_seconds() / 60.0
            if elapsed >= self._settings.max_minutes:
                triggered = True

        if not triggered:
            return None

        # 行 513-528: 构建快照
        snapshot_text, senders = await self._build_snapshot(
            session_id, last_turn, turn_count,
        )
        if snapshot_text is None:
            return None

        return (last_turn, turn_count, snapshot_text, list(senders))

    async def _build_snapshot(
        self, session_id: str, last_turn: int, turn_count: int,
    ) -> tuple[str | None, set[str]]:
        """构建对话快照文本。

        对应 analysis.py check_distill_needed 行 513-528。
        从 chat_history 查询 last_turn 之后、turn_count 之前的对话记录,
        按 [昵称] 内容 格式组装, 最多取 max_snapshot_lines 行。

        如果未注入 chat_history_provider, 则跳过快照构建（返回 None）。

        Args:
            session_id: 会话标识
            last_turn:  上次蒸馏截止轮次
            turn_count: 当前总轮次

        Returns:
            (snapshot_text, senders_set) — 快照文本不足 6 条时返回 (None, set())
        """
        if self._chat_history_provider is None:
            return (None, set())

        # 对应行 513-516: 查询聊天历史
        rows = await self._chat_history_provider(
            SessionId(session_id), last_turn, turn_count,
        )

        # 行 517-518: 至少 6 条消息
        if len(rows) < self._settings.distill_min_turns:
            return (None, set())

        senders: set[str] = set()
        lines: list[str] = []
        max_lines = self._settings.max_snapshot_lines

        for record in rows:
            nick = record.sender_name or (
                "用户" if record.role == "user" else self._persona_name
            )
            if nick:
                senders.add(nick)
            lines.append(f"[{nick}] {record.content[:200]}")
            if len(lines) >= max_lines:
                break

        snapshot_text = "\n".join(lines[:max_lines])
        return (snapshot_text, senders)

    # ────────────────────────────────────────────────────────────
    # 内部方法：长期记忆加载
    # ────────────────────────────────────────────────────────────

    async def _load_long_term(self, user_id: str) -> dict:
        """加载用户的长期记忆数据。

        对应 _run_distill_analysis 行 327-331 的 long_term 构建逻辑。

        Returns:
            dict 包含 identity_traits, preferences, self_identity, boundaries 等
        """
        identity = await self._identity_repo.get(user_id)
        if identity is None:
            return {}

        result: dict = {}

        # 特质
        if identity.traits:
            result["identity_traits"] = [t.text for t in identity.traits]

        # 偏好
        if identity.preferences:
            prefs = identity.preferences
            result["preferences"] = {
                "likes": prefs.likes,
                "dislikes": prefs.dislikes,
            }

        # 自我认知
        if identity.self_identity:
            result["self_identity"] = identity.self_identity

        # 雷区
        if identity.boundaries:
            result["boundaries"] = [b.description for b in identity.boundaries]

        return result

    @staticmethod
    def _build_personality_style(long_term: dict) -> str:
        """构建人格风格描述文本。

        对应 _run_distill_analysis 行 326-331 的 personality_style 逻辑。

        Args:
            long_term: 长期记忆字典

        Returns:
            风格描述字符串
        """
        style_parts: list[str] = []
        traits = long_term.get("identity_traits", [])
        if traits:
            style_parts.append("特质：" + "、".join(traits[:3]))
        return "。".join(style_parts)

    # ────────────────────────────────────────────────────────────
    # 内部方法：LLM 调用 + 响应解析
    # ────────────────────────────────────────────────────────────

    async def _call_llm(self, prompt: str) -> str | None:
        """调用 LLM 并返回清洗后的响应文本。

        对应 _run_distill_analysis 行 340-356。
        返回 None 表示 LLM 调用失败或返回空内容。

        Args:
            prompt: 完整 prompt

        Returns:
            清洗后的 LLM 响应文本, 或 None
        """
        response = await self._llm.generate(prompt)
        if not response or not response.strip():
            return None
        return response

    def _parse_llm_response(
        self, raw: str,
    ) -> tuple[str, str, str, dict] | None:
        """清洗并解析 LLM JSON 响应。

        对应 _run_distill_analysis 行 374-402。

        Args:
            raw: 原始 LLM 响应文本

        Returns:
            (content, keylabel, summary, analysis) 或 None（解析失败）
        """
        # 行 375: 清洗
        cleaned = strip_llm_fences(raw)

        # 行 377-382: JSON 解析
        try:
            result = json.loads(cleaned)
        except json.JSONDecodeError:
            return None
        content = result.get("content", "")
        summary = result.get("summary", "") or content
        analysis_data = result.get("analysis", {})

        if not content:
            return None

        # 行 387-392: 向后兼容 keylabel
        keylabel = result.get("keylabel", "") or result.get("summary", "")
        if len(keylabel) > 30:
            kfs = (analysis_data or {}).get("key_facts", [])
            kf_strings = [f for f in kfs if isinstance(f, str)]
            if kf_strings:
                keylabel = "·".join(f[:10] for f in kf_strings[:3])[:20]
            else:
                keylabel = content[:20]
        if not keylabel:
            keylabel = content[:20]

        return (content, keylabel, summary, analysis_data)

    # ────────────────────────────────────────────────────────────
    # 内部方法：analysis_raw 审计日志（Task 2）
    # ────────────────────────────────────────────────────────────

    async def _save_raw_analysis(
        self, user_id: str, session_id: str, content: str,
    ) -> int:
        """保存 LLM 原始响应到 analysis_raw 审计日志。

        对应 _run_distill_analysis 行 358-368。
        通过注入的 save_raw_analysis_fn 回调持久化，
        返回 analysis_raw 记录的自增 ID（未注入时返回 0）。

        Args:
            user_id:    用户标识
            session_id: 会话标识
            content:    LLM 原始响应文本

        Returns:
            analysis_raw.id（审计记录主键），0 表示未注入或无持久化
        """
        if self._save_raw_analysis_fn is None:
            return 0
        return await self._save_raw_analysis_fn(user_id, session_id, content)

    # ────────────────────────────────────────────────────────────
    # 内部方法：蒸馏摘要写入（含图谱维护 + 过期清理）
    # ────────────────────────────────────────────────────────────

    async def _write_distill_summary(
        self,
        user_id: str,
        session_id: str,
        content: str,
        keylabel: str,
        summary: str,
        analysis: dict,
        now: datetime,
        turn_count: int,
    ) -> tuple[MemoryId, list[tuple[int, str]]]:
        """写入 LLM 蒸馏摘要 + 归档 + 过期清理 + 图谱维护 + 向量嵌入。

        对应 _apply_distill（行 531-578）。

        Args:
            user_id:    用户标识
            session_id: 会话标识
            content:    蒸馏正文
            keylabel:   核心标签
            summary:    概要
            analysis:   9 维分析数据（仅取 mood 和 mood_intensity）
            now:        当前时间
            turn_count: 当前蒸馏到的轮次

        Returns:
            (memory_id, [(entry_id, text_for_embedding), ...])
        """
        # 行 534-537: 保底 keylabel / summary
        if not keylabel:
            keylabel = content[:20]
        if not summary:
            summary = content

        mood = analysis.get("mood", "") if isinstance(analysis, dict) else ""
        mood_intensity = analysis.get("mood_intensity", 0.0) if isinstance(analysis, dict) else 0.0

        # 行 542-545: 写入蒸馏摘要到 cognitive_distill
        memory = Memory(
            memory_id=MemoryId(0),
            user_id=UserId(user_id),
            content=content,
            keylabel=keylabel,
            summary=summary,
            importance=Importance(self._settings.distill_entry_importance),
            mood=Mood(mood),
            mood_intensity=mood_intensity,
            session_id=SessionId(session_id),
            created_at=self._clock.now(),
        )
        memory_id = await self._memory_repo.save(memory)

        # 行 547-550: 更新会话蒸馏进度
        await self._session_repo.update_last_distill(
            SessionId(session_id), turn_count, now,
        )

        # 行 551: 归档悬案 — 对应 _archive_dangling（行 580-604）
        await self._archive_dangling(user_id, session_id, now, reason="蒸馏")

        # 行 552-555: 删除过期记忆
        expired_count = await self._memory_repo.delete_expired(UserId(user_id))

        # 行 558-566: 保留低重要性条目上限
        await self._discard_excess_low_importance(user_id)

        # 行 567: 图谱衰减维护 — 对应 _maintain_graph（retrieval.py 行 486-507）
        await self._graph_repo.maintain(user_id)

        # 行 575-577: 蒸馏摘要向量化
        new_entries: list[tuple[int, str]] = []
        vec = await self._embedding.embed(keylabel[:512])
        if vec:
            await self._memory_repo.store_embedding(
                memory_id, vec,
            )
            new_entries.append(
                (
                    int(memory_id.value) if isinstance(memory_id, MemoryId) else int(memory_id),
                    keylabel,
                ),
            )

        # 发布过期事件
        if expired_count > 0:
            await self._event_bus.publish(MemoryExpired(
                user_id=user_id,
                expired_count=expired_count,
                occurred_at=now,
            ))

        return (memory_id, new_entries)

    async def _archive_dangling(
        self, user_id: str, session_id: str, now: datetime, reason: str = "",
    ) -> None:
        """清空 session_state 中的 dangling_threads，不再写入 cognitive_distill。

        `[悬案归档]` 类记录是系统运维噪音，不应写入 cognitive_distill，
        否则时间/重要性通道会误将其作为用户记忆召回。

        Args:
            user_id:    用户标识
            session_id: 会话标识
            now:        当前时间
            reason:     归档原因描述（保留参数，不再用于生成标签）
        """
        dangling = await self._session_repo.get_dangling_threads(SessionId(session_id))
        if not dangling:
            return
        if not isinstance(dangling, dict) or not dangling.get("threads"):
            return

        # 不再写入 cognitive_distill，只清空 session_state
        await self._session_repo.update_dangling_threads(
            SessionId(session_id), {"threads": [], "turn": 0},
        )

    async def _discard_excess_low_importance(self, user_id: str) -> None:
        """保留下限 importance=0.3 的条目中最近的 N 条, 丢弃超出的。

        对应 _apply_distill 行 558-566。
        如果注入了 trim_low_importance_fn 则委托给该回调,
        否则通过 memory_repo.get_by_user 检查量级但不执行删除。

        Args:
            user_id: 用户标识
        """
        low_imp_memories = await self._memory_repo.get_by_user(
            UserId(user_id),
            limit=self._settings.maintenance.keep_rule_summary,
            min_importance=0.3,
        )
        if len(low_imp_memories) < self._settings.keep_rule_summary:
            return
        if self._trim_low_importance_fn is not None:
            await self._trim_low_importance_fn(
                UserId(user_id), self._settings.keep_rule_summary,
            )

    # ────────────────────────────────────────────────────────────
    # 内部方法：9 维分析写入
    # ────────────────────────────────────────────────────────────

    async def _apply_nine_dim_analysis(
        self,
        user_id: str,
        session_id: str,
        data: dict,
        now: datetime,
    ) -> list[tuple[int, str]]:
        """写入 9 维分析结果。

        对应 _apply_analysis（行 42-313）的完整写入逻辑：
            1. 归一化数据
            2. 更新会话状态（last_active + user_state → stance）
            3. 身份特质合并/衰减/截断
            4. 结构化身份字段（偏好, 自我认知）
            5. 共享梗/上下文
            6. 边界/雷区
            7. 悬案线程 → cognitive_distill
            8. 实体 → 图谱（节点 + 语义边 + 同轮共现）
            9. 关键事实（permanent / transient）
           10. 批量向量化

        Args:
            user_id:    用户标识
            session_id: 会话标识
            data:       归一化前的 LLM 分析数据
            now:        当前时间

        Returns:
            [(entry_id, text_for_embedding), ...] 新写入的待嵌入条目列表
        """
        new_entries: list[tuple[int, str]] = []

        # ── 当注入 AnalysisWriter 时委托给它（Task 3）────────────────
        if self._analysis_writer is not None:
            new_entries = await self._analysis_writer.write_all(
                user_id, session_id, data,
            )
            await self._batch_embed_new_entries(user_id, new_entries)
            return new_entries

        # ── 未注入 AnalysisWriter 时使用内部实现 ─────────────────────

        # Step 1: 数据归一化 — 对应行 54-96
        data = normalize_analysis_data(data)

        # Step 2: 更新会话最后活跃时间 — 对应行 50-52
        await self._session_repo.update_last_active(SessionId(session_id), now)

        # Step 3: 更新用户立场 — 对应行 102-107
        user_state = data.get("user_state")
        if user_state:
            session = await self._session_repo.get(SessionId(session_id))
            if session is not None:
                session.stance = user_state
                await self._session_repo.save(session)

        # Step 4: 身份特质合并/衰减/截断 — 对应行 111-158
        await self._merge_identity_traits(user_id, data, now)

        # Step 5: 结构化身份字段 — 对应行 160-178
        await self._save_identity_structured(user_id, data)

        # Step 6: 共享梗/上下文 — 对应行 180-198
        await self._save_shared_jokes(user_id, data)

        # Step 7: 边界/雷区 — 对应行 200-206
        await self._save_boundaries(user_id, data)

        # Step 8: 悬案线程归档 — 对应行 208-221
        dangling_entries = await self._save_dangling_threads(
            user_id, session_id, data, now,
        )
        new_entries.extend(dangling_entries)

        # Step 9: 实体 → 图谱 — 对应行 223-252
        await self._sync_entities_to_graph(user_id, data)

        # Step 10: 关键事实 — 对应行 254-290
        fact_entries = await self._save_key_facts(user_id, data, now)
        new_entries.extend(fact_entries)

        # Step 11: 批量向量嵌入 — 对应行 309-313
        await self._batch_embed_new_entries(user_id, new_entries)

        return new_entries

    # ── Step 4: 身份特质合并/衰减/截断 ──

    async def _merge_identity_traits(
        self, user_id: str, data: dict, now: datetime,
    ) -> None:
        """合并新特质 + 衰减未确认特质 + 截断容量上限。

        对应 _apply_analysis 行 111-158。
        逻辑：
          - 新特质以 strength=5 写入
          - 已有特质 strength 置 5 且 count +1
          - 未确认特质 strength 衰减（-1, 下限 min(c//2, 2)）
          - 容量上限 30 条（按 s*2 + c 排序）

        Args:
            user_id: 用户标识
            data:    LLM 分析数据
            now:     当前时间
        """
        # 行 112-120: 加载现有特质
        identity = await self._identity_repo.get(user_id)
        if identity is None:
            return

        raw = identity.traits
        trait_map: dict[str, dict[str, int]] = {}
        for item in raw:
            if isinstance(item, str):
                trait_map[item] = {"s": 3, "c": 0}
            elif isinstance(item, dict):
                t_text = item.get("t", "")
                trait_map[t_text] = {"s": item.get("s", 0), "c": item.get("c", 0)}
            elif isinstance(item, Trait):
                trait_map[item.text] = {"s": item.strength, "c": item.count}

        # 行 122-138: 合并新特质 + 小细节归入特质池
        confirmed: set[str] = set()
        for t in data.get("traits_updates", []):
            if t not in trait_map:
                trait_map[t] = {"s": 5, "c": 1}
            else:
                trait_map[t]["s"] = 5
                trait_map[t]["c"] += 1
            confirmed.add(t)

        for q in data.get("speech_quirks", []):
            q_entry = f"[小细节小习惯] {q}"
            if q_entry not in trait_map:
                trait_map[q_entry] = {"s": 0, "c": 0}
            trait_map[q_entry]["s"] = 5
            trait_map[q_entry]["c"] += 1
            confirmed.add(q_entry)

        # 行 140-146: 衰减未确认特质
        for t in list(trait_map.keys()):
            if t not in confirmed:
                floor = min(trait_map[t]["c"] // 2, 2)
                trait_map[t]["s"] = max(trait_map[t]["s"] - 1, floor)
                if trait_map[t]["s"] <= 0:
                    del trait_map[t]

        # 行 148-152: 容量上限截断
        if len(trait_map) > self._settings.maintenance.max_trait_count:
            sorted_t = sorted(
                trait_map.items(),
                key=lambda x: x[1]["s"] * 2 + x[1]["c"],
                reverse=True,
            )[:self._settings.maintenance.max_trait_count]
            trait_map = dict(sorted_t)

        # 行 153-158: 写回
        new_traits = [
            Trait(text=t, strength=v["s"], count=v["c"])
            for t, v in trait_map.items()
        ]
        old_traits = identity.traits
        if new_traits != old_traits:
            identity.traits = new_traits
            identity.updated_at = now
            await self._identity_repo.update_identity(user_id, identity)

    # ── Step 5: 结构化身份字段 ──

    async def _save_identity_structured(self, user_id: str, data: dict) -> None:
        """写入结构化身份字段（偏好, 自我认知）。

        对应 _apply_analysis 行 160-178。
        覆盖写, LLM 每次产出完整快照。

        Args:
            user_id: 用户标识
            data:    LLM 分析数据
        """
        # 偏好
        prefs_raw = data.get("preferences")
        if prefs_raw is not None:
            prefs = Preferences(
                likes=prefs_raw.get("likes", []),
                dislikes=prefs_raw.get("dislikes", []),
            )
            await self._identity_repo.save_preferences(user_id, prefs)

        # 自我认知
        self_id_raw = data.get("self_identity")
        if self_id_raw is not None:
            self_id_list = self_id_raw if isinstance(self_id_raw, list) else [str(self_id_raw)]
            await self._identity_repo.save_self_identity(user_id, self_id_list)

    # ── Step 6: 共享梗/上下文 ──

    async def _save_shared_jokes(self, user_id: str, data: dict) -> None:
        """写入共享梗/上下文。

        对应 _apply_analysis 行 180-198。
        shared_context 表操作通过注入的 save_shared_context_fn 委托,
        未注入时跳过写入。

        Args:
            user_id: 用户标识
            data:    LLM 分析数据, 含 shared_jokes 字段
        """
        jokes = data.get("shared_jokes", [])
        if not jokes or self._save_shared_context_fn is None:
            return

        for joke in jokes:
            trigger = joke.get("trigger", "")
            ctx = joke.get("context", "")
            if not trigger:
                continue
            await self._save_shared_context_fn(user_id, trigger, ctx)

    # ── Step 7: 边界/雷区 ──

    async def _save_boundaries(self, user_id: str, data: dict) -> None:
        """写入边界/雷区（覆盖写）。

        对应 _apply_analysis 行 200-206。
        LLM 已参考现有雷区, 产出即完整快照。

        Args:
            user_id: 用户标识
            data:    LLM 分析数据, 含 boundaries 字段
        """
        boundaries_raw = data.get("boundaries")
        if boundaries_raw is not None and isinstance(boundaries_raw, list):
            boundaries = [Boundary(description=b) for b in boundaries_raw]
            await self._identity_repo.save_boundaries(user_id, boundaries)

    # ── Step 8: 悬案线程归档 ──

    async def _save_dangling_threads(
        self,
        user_id: str,
        session_id: str,
        data: dict,
        now: datetime,
    ) -> list[tuple[int, str]]:
        """将悬案线程写入 cognitive_distill + 更新 session_state。

        对应 _apply_analysis 行 208-221。

        Args:
            user_id:    用户标识
            session_id: 会话标识
            data:       LLM 分析数据, 含 dangling_threads 字段
            now:        当前时间

        Returns:
            新创建的 (entry_id, text_for_embedding) 列表
        """
        entries: list[tuple[int, str]] = []
        threads = data.get("dangling_threads", [])

        for dt in threads:
            memory = Memory(
                memory_id=MemoryId(0),
                user_id=UserId(user_id),
                content=dt,
                keylabel=dt,
                summary=dt,
                importance=Importance(self._settings.dangling_fallback_importance),
                session_id=SessionId(session_id),
                created_at=now,
            )
            memory_id = await self._memory_repo.save(memory)
            entries.append(
                (
                    int(memory_id.value) if isinstance(memory_id, MemoryId) else int(memory_id),
                    dt,
                ),
            )

        if session_id and threads:
            session = await self._session_repo.get(SessionId(session_id))
            current_turn = session.turn_count if session else 0
            await self._session_repo.update_dangling_threads(
                SessionId(session_id),
                {"threads": threads, "turn": current_turn},
            )

        return entries

    # ── Step 9: 实体 → 图谱 ──

    async def _sync_entities_to_graph(
        self, user_id: str, data: dict,
    ) -> None:
        """将实体写入图谱（节点保底 + 语义边双向 + 同轮共现）。

        对应 _apply_analysis 行 223-252。
        包括：
          - 节点 upsert（去重 + freq +1）
          - 语义边（含反向关系）
          - 同轮共现边（空 relation, 权重累积）

        Args:
            user_id: 用户标识
            data:    LLM 分析数据, 含 entities 字段
        """
        ent_ids: dict[str, int] = {}

        for ent in data.get("entities", []):
            name = ent.get("name", "")
            entity_type = ent.get("type", "auto")
            if not name:
                continue

            # 对应 _upsert_graph_node（retrieval.py 行 425-447）
            node_id = await self._graph_repo.upsert_node(
                user_id, name, entity_type=entity_type,
            )
            if node_id < 0:
                continue
            ent_ids[name] = node_id

            # 语义边 — 对应行 234-245
            relations = ent.get("relations", [])
            for rel in relations:
                target = rel.get("target", "")
                relation = rel.get("relation", "")
                if not target or not relation:
                    continue
                to_id = await self._graph_repo.upsert_node(user_id, target)
                await self._graph_repo.upsert_edge(node_id, to_id, relation=relation)
                inverse = self._INVERSE_RELATIONS.get(relation)
                if inverse:
                    await self._graph_repo.upsert_edge(to_id, node_id, relation=inverse)

        # 同轮共现边 — 对应行 247-252
        names = list(ent_ids.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a_id = ent_ids[names[i]]
                b_id = ent_ids[names[j]]
                await self._graph_repo.upsert_edge(a_id, b_id, relation="")

    # ── Step 10: 关键事实 ──

    async def _save_key_facts(
        self,
        user_id: str,
        data: dict,
        now: datetime,
    ) -> list[tuple[int, str]]:
        """将关键事实写入 cognitive_distill。

        对应 _apply_analysis 行 254-290。
        区分 permanent（保底 importance 0.5, 上限 3）和
        transient（无保底, 上限 5, 可设置过期天数）。

        Args:
            user_id: 用户标识
            data:    LLM 分析数据, 含 importance, key_facts, key_facts_structured
            now:     当前时间

        Returns:
            新创建的 (entry_id, text_for_embedding) 列表
        """
        entries: list[tuple[int, str]] = []

        # 行 255-256: 重要性保底
        importance_raw = data.get("importance", 0.0) or 0.0
        kf_imp = max(float(importance_raw), 0.5)

        kfs = data.get("key_facts", []) or data.get("key_facts_structured", [])
        perm_count = 0
        tran_count = 0

        for kf in kfs:
            if isinstance(kf, str):
                fact_content = kf
                temporal = "permanent"
                expires_at = None
            elif isinstance(kf, dict):
                fact_content = kf.get("content", "")
                temporal = kf.get("temporal", "permanent")
                if temporal == "transient" and kf.get("expires_after_days"):
                    expires_at = now + timedelta(days=int(kf["expires_after_days"]))
                else:
                    expires_at = None
            else:
                continue

            if not fact_content:
                continue

            if temporal == "permanent":
                if perm_count >= self._settings.permanent_fact_max:
                    continue
                imp_val = kf_imp
                perm_count += 1
            else:
                if tran_count >= self._settings.transient_fact_max:
                    continue
                imp_val = float(importance_raw)
                tran_count += 1

            memory = Memory(
                memory_id=MemoryId(0),
                user_id=UserId(user_id),
                content=fact_content,
                keylabel=fact_content,
                summary=fact_content,
                importance=Importance(imp_val),
                expires_at=expires_at,
                created_at=now,
            )
            memory_id = await self._memory_repo.save(memory)
            entries.append(
                (
                    int(memory_id.value) if isinstance(memory_id, MemoryId) else int(memory_id),
                    fact_content[:512],
                ),
            )

        return entries

    # ── Step 11: 批量向量嵌入 ──

    async def _batch_embed_new_entries(
        self,
        user_id: str,
        entries: list[tuple[int, str]],
    ) -> None:
        """为新创建的记录批量生成并存储向量嵌入。

        对应 _apply_analysis 行 309-313 + _apply_distill 行 575-577。

        Args:
            user_id: 用户标识（保留参数, 供未来日志使用）
            entries: [(record_id, text), ...]
        """
        for eid, text in entries:
            vec = await self._embedding.embed(text[:512])
            if vec:
                await self._memory_repo.store_embedding(MemoryId(eid), vec)

    # ────────────────────────────────────────────────────────────
    # 内部方法：事件发布
    # ────────────────────────────────────────────────────────────

    async def _publish_events(
        self,
        user_id: str,
        session_id: str,
        memory_id: MemoryId,
        content: str,
        keylabel: str,
        analysis: dict,
        entries: list[tuple[int, str]],
        now: datetime,
    ) -> None:
        """发布蒸馏完成后的一系列事件。

        发布：
          - MemoryDistilled: 触发 GraphMaintenanceHandler 等 Handler
          - EmbeddingDone × N: 每条嵌入完成后触发缓存刷新

        Args:
            user_id:    用户标识
            session_id: 会话标识
            memory_id:  蒸馏摘要记录 ID
            content:    蒸馏正文
            keylabel:   核心标签
            analysis:   9 维分析数据
            entries:    [(record_id, text)] 所有已嵌入条目
            now:        事件发生时间
        """
        mid = int(memory_id.value) if isinstance(memory_id, MemoryId) else int(memory_id)
        mood = analysis.get("mood", "")
        mood_intensity = analysis.get("mood_intensity", 0.0)

        # MemoryDistilled
        await self._event_bus.publish(MemoryDistilled(
            user_id=user_id,
            session_id=session_id,
            memory_id=mid,
            content=content,
            keylabel=keylabel,
            importance=self._settings.distill_entry_importance,
            mood=mood,
            mood_intensity=mood_intensity,
            raw_analysis=analysis,
            occurred_at=now,
        ))

        # EmbeddingDone — 每条已嵌入条目各发布一个事件
        for eid, _ in entries:
            await self._event_bus.publish(EmbeddingDone(
                user_id=user_id,
                record_id=eid,
                embedding_dim=0,
                source="distill",
                occurred_at=now,
            ))

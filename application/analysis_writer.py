"""
AnalysisWriter — 9 维蒸馏分析结果写入服务。

从 analysis.py _apply_analysis 提取（行 42-313）。
职责：将 LLM 蒸馏产出的结构化分析数据写入各个 Repository。
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Awaitable, Callable, Protocol, runtime_checkable

from domain.ports.identity_repo import IIdentityRepository
from domain.ports.graph_repo import IGraphRepository
from domain.ports.repositories import IMemoryRepository, ISessionRepository
from domain.ports.clock import IClock
from domain.entities.memory import SessionId
from domain.entities.identity import Boundary, Preferences, Trait

# ═══════════════════════════════════════════════════════════════
# 设置接口：替代 infrastructure 层 Settings 的直接导入
# ═══════════════════════════════════════════════════════════════


@runtime_checkable
class IAnalysisWriterSettings(Protocol):
    """最小设置接口，供 AnalysisWriter 使用。

    替代从 infrastructure 层直接导入的 Settings 具体类，
    遵循依赖倒置原则，消除应用层对基础设施层的依赖。

    AnalysisWriter 只使用以下属性：
      - inverse_relations.inverse_relations
    """

    inverse_relations: Any
    analysis: Any

logger = logging.getLogger("rcms")


class AnalysisWriter:
    """分析结果写入服务

    处理 LLM 蒸馏产出的多维度持久化（不包含 key_facts 和 dangling_threads 的
    cognitive_distill 写入）：
    mood / traits / preferences / self_identity / boundaries /
    shared_context / dangling_threads(session_state) / entities

    原始逻辑在 analysis.py _apply_analysis（行 42-313）。
    """

    def __init__(
        self,
        memory_repo: IMemoryRepository,
        session_repo: ISessionRepository,
        identity_repo: IIdentityRepository,
        graph_repo: IGraphRepository,
        clock: IClock,
        settings: IAnalysisWriterSettings,
        upsert_shared_context: Callable[[str, str, str], Awaitable[None]] | None = None,
    ):
        """初始化 AnalysisWriter

        Args:
            memory_repo: 认知蒸馏记忆仓储（cognitive_distill 表）
            session_repo: 会话状态仓储（session_state 表）
            identity_repo: 用户身份仓储（identity_memory 表）
            graph_repo: 图谱仓储（memory_graph_nodes / memory_graph_edges 表）
            clock: 时间源抽象
            settings: 全局配置（反向关系映射/事实上限等）
            upsert_shared_context: 可选回调，用于 shared_context 表的 upsert。
                                   原型 (user_id, trigger, context) -> None。
        """
        self._memory_repo = memory_repo
        self._session_repo = session_repo
        self._identity_repo = identity_repo
        self._graph_repo = graph_repo
        self._clock = clock
        self._settings = settings
        self._upsert_shared_context = upsert_shared_context

    async def write_all(
        self, user_id: str, session_id: str, analysis: dict[str, Any]
    ) -> list[tuple[int, str]]:
        """写入分析结果。

        不再写入 key_facts 和 dangling_threads 到 cognitive_distill，
        保留以下维度：
          - 会话状态 & 用户立场
          - 身份特质合并/衰减/截断
          - 结构化身份字段（偏好、自我认同）
          - 共享梗/上下文
          - 边界/雷区
          - 悬案线程 → session_state（不写 cognitive_distill）
          - 实体 → 图谱
          - 日志

        Args:
            user_id: 用户标识
            session_id: 会话标识
            analysis: LLM 产出的结构化分析数据

        Returns:
            始终返回空列表（不再产生待嵌入子条目）
        """
        now_dt = self._clock.now()
        data = self._normalize_analysis(analysis)

        # 1. 更新 session 活跃时间
        await self._session_repo.update_last_active(SessionId(session_id), now_dt)

        # 2. 更新 user_state → session_state.stance
        await self._write_session_stance(session_id, data)

        # 3. 写入 traits_updates + speech_quirks
        await self._write_traits(user_id, data)

        # 4. 写入 preferences / self_identity（覆盖写）
        await self._write_preferences(user_id, data)

        # 5. 写入 boundaries（覆盖写）
        await self._write_boundaries(user_id, data)

        # 6. 写入 shared_context（upsert 梗）
        await self._write_shared_context(user_id, data)

        # 7. 处理 dangling_threads → session_state（不写入 cognitive_distill）
        await self._write_dangling_threads(user_id, session_id, data)

        # 8. 处理 entities → 图谱
        await self._write_entities(user_id, data)

        # 日志
        self._log_summary(user_id, data)

        return []

    # ================================================================
    # 归一化（行 55-96）
    # ================================================================

    def _normalize_analysis(self, raw: dict) -> dict:
        """归一化 LLM 输出的格式兼容性（dict/str 混合格式）

        行 55-96：将新输出格式中可能的 dict 项归一化为 string 列表，
        避免后续逻辑出错。key_facts 保留 dict 结构以保护 temporal 信息。

        Args:
            raw: 原始 LLM 分析输出

        Returns:
            归一化后的分析字典
        """
        data = dict(raw)

        # traits_updates: dict → str（行 55-61）
        traits = []
        for t in data.get("traits_updates", []) or []:
            if isinstance(t, dict):
                trait = t.get("trait") or t.get("t") or None
                if trait:
                    traits.append(trait)
            elif isinstance(t, str):
                traits.append(t)
        data["traits_updates"] = traits

        # speech_quirks: dict → str（行 64-71）
        quirks = []
        for q in data.get("speech_quirks", []) or []:
            if isinstance(q, dict):
                quirk = q.get("quirk") or q.get("q") or q.get("text") or None
                if quirk:
                    quirks.append(quirk)
            elif isinstance(q, str):
                quirks.append(q)
        data["speech_quirks"] = quirks

        # dangling_threads: dict → str（行 73-80）
        threads = []
        for dt in data.get("dangling_threads", []) or []:
            if isinstance(dt, dict):
                content = dt.get("content") or dt.get("text") or None
                if content:
                    threads.append(content)
            elif isinstance(dt, str):
                threads.append(dt)
        data["dangling_threads"] = threads

        # key_facts: 保留 dict 结构以保护 temporal/expires 信息
        # 原始代码（行 82-89）将 dict flatten 为 str 丢失了关键信息，
        # 此处改进为规范化字段名但保留 dict 结构
        if "key_facts" in data:
            normalized_kfs: list[dict[str, Any] | str] = []
            for kf in data["key_facts"] or []:
                if isinstance(kf, dict):
                    content = kf.get("content") or kf.get("text") or ""
                    normalized_kfs.append({
                        "content": content,
                        "temporal": kf.get("temporal", "permanent"),
                        "expires_after_days": kf.get("expires_after_days"),
                    })
                elif isinstance(kf, str):
                    normalized_kfs.append(kf)
            data["key_facts"] = normalized_kfs
            # 移除 key_facts_structured 避免歧义（行 257 的原始 fallback）
            data.pop("key_facts_structured", None)

        return data

    # ================================================================
    # 维度 3：session stance（行 103-107）
    # ================================================================

    async def _write_session_stance(self, session_id: str, data: dict) -> None:
        """更新 session_state.stance（行 103-107）

        从 LLM 产出的 user_state 字段更新会话立场。

        Args:
            session_id: 会话标识
            data: 归一化后的分析数据
        """
        user_state = data.get("user_state")
        if user_state and session_id:
            session = await self._session_repo.get(SessionId(session_id))
            if session:
                session.stance = str(user_state)
                await self._session_repo.save(session)

    # ================================================================
    # 维度 4：traits + quirks（行 112-158）
    # ================================================================

    async def _write_traits(self, user_id: str, data: dict) -> None:
        """写入 traits_updates + speech_quirks（行 112-158）

        将归一化后的特质列表和语言习惯合并为 Trait 列表，
        委托 IIdentityRepository 执行衰减-确认合并逻辑：
          - 新特质以 strength=5 写入
          - 已有特质 strength 置 5 且 count+1
          - 未被本轮确认的特质 strength 衰减

        speech_quirks 以 "[小细节小习惯] {q}" 格式加入特质池（行 131-138）。

        Args:
            user_id: 用户标识
            data: 归一化后的分析数据
        """
        traits_updates: list[str] = data.get("traits_updates", [])
        speech_quirks: list[str] = data.get("speech_quirks", [])

        if not traits_updates and not speech_quirks:
            return

        # 去重合并
        seen: set[str] = set()
        confirmed: list[Trait] = []

        for t in traits_updates:
            if t and t not in seen:
                seen.add(t)
                confirmed.append(Trait(text=t, strength=5, count=1))

        for q in speech_quirks:
            text = f"[小细节小习惯] {q}"
            if text not in seen:
                seen.add(text)
                confirmed.append(Trait(text=text, strength=5, count=1))

        if confirmed:
            await self._identity_repo.save_traits(user_id, confirmed)

    # ================================================================
    # 维度 5：preferences / self_identity（行 161-178）
    # ================================================================

    async def _write_preferences(self, user_id: str, data: dict) -> None:
        """写入 preferences / self_identity（覆盖写，行 161-178）

        LLM 每次产出完整快照，直接覆盖 identity_memory 中的已有内容。
        对应行 163-178 的 col/val 循环，原始 JSON 处理兼容 str/dict 格式。

        Args:
            user_id: 用户标识
            data: 归一化后的分析数据
        """
        # preferences（行 163-165）
        prefs_data = data.get("preferences")
        if prefs_data is not None:
            if isinstance(prefs_data, str):
                prefs_data = json.loads(prefs_data)  # 异常向上传播
            if isinstance(prefs_data, dict):
                preferences = Preferences(
                    likes=prefs_data.get("likes", []),
                    dislikes=prefs_data.get("dislikes", []),
                )
                await self._identity_repo.save_preferences(user_id, preferences)

        # self_identity（行 166）
        si_data = data.get("self_identity")
        if si_data is not None:
            if isinstance(si_data, str):
                si_data = json.loads(si_data)  # 异常向上传播
            if isinstance(si_data, list):
                identities = [str(item) for item in si_data]
                await self._identity_repo.save_self_identity(user_id, identities)

    # ================================================================
    # 维度 6：boundaries（行 200-206）
    # ================================================================

    async def _write_boundaries(self, user_id: str, data: dict) -> None:
        """写入 boundaries（覆盖写，行 200-206）

        LLM 已参考现有雷区，产出即完整快照，直接覆盖 identity_memory。

        Args:
            user_id: 用户标识
            data: 归一化后的分析数据
        """
        raw = data.get("boundaries")
        if raw is None or not isinstance(raw, list):
            return

        boundaries: list[Boundary] = []
        for item in raw:
            if isinstance(item, str):
                if item.strip():
                    boundaries.append(Boundary(description=item))
            elif isinstance(item, dict):
                desc = (item.get("description") or item.get("desc") or "").strip()
                if desc:
                    boundaries.append(Boundary(description=desc))

        if boundaries:
            await self._identity_repo.save_boundaries(user_id, boundaries)

    # ================================================================
    # 维度 7：shared_context（行 180-198）
    # ================================================================

    async def _write_shared_context(self, user_id: str, data: dict) -> None:
        """写入 shared_context（upsert 梗，行 180-198）

        使用注入的回调 upsert_shared_context 进行持久化。
        未配置回调时跳过并记录警告。

        每个梗包含 trigger（触发词）和 context（上下文），
        已存在的 trigger 通过 omission_count +1 累积（行 185-193），
        新 trigger 插入新行（行 194-198）。

        Args:
            user_id: 用户标识
            data: 归一化后的分析数据
        """
        jokes = data.get("shared_jokes", [])
        if not jokes:
            return

        if self._upsert_shared_context is None:
            logger.warning(
                "shared_context: upsert_shared_context 未配置，共 %d 条梗跳过写入",
                len(jokes),
            )
            return

        for joke in jokes:
            trigger = joke.get("trigger", "")
            context = joke.get("context", "")
            if trigger:
                await self._upsert_shared_context(user_id, trigger, context)

    # ================================================================
    # 维度 8：dangling_threads（行 208-221）
    # ================================================================

    async def _write_dangling_threads(
        self,
        user_id: str,
        session_id: str,
        data: dict,
    ) -> None:
        """将悬案线程更新到 session_state（不再写入 cognitive_distill）。

        仅更新 session_state.dangling_threads 字段供 PromptBuilder 使用，
        不移除旧的 session_state 更新逻辑。

        Args:
            user_id: 用户标识
            session_id: 会话标识
            data: 归一化后的分析数据
        """
        threads: list[str] = data.get("dangling_threads", [])
        if not threads:
            return

        # 更新 session 的悬案字段
        if session_id:
            session = await self._session_repo.get(SessionId(session_id))
            current_turn = session.turn_count if session else 0
            await self._session_repo.update_dangling_threads(
                SessionId(session_id),
                {"threads": threads, "turn": current_turn},
            )

    # ================================================================
    # 维度 9：entities（行 223-252）
    # ================================================================

    async def _write_entities(self, user_id: str, data: dict) -> None:
        """处理 entities → 图谱（行 223-252）

        三阶段写入：
          1. 节点保底 upsert（行 225-233）：每个实体名确保图节点存在
          2. 语义边优先插入（行 234-245）：实体间的关系边 + 反向边
          3. 同轮共现兜底（行 247-252）：本轮出现的实体之间建空 relation 边

        反向关系映射从 settings.inverse_relations.inverse_relations 读取，
        对应 analysis.py _INVERSE_RELATIONS（行 15-29）。

        Args:
            user_id: 用户标识
            data: 归一化后的分析数据
        """
        inverse_map = self._settings.inverse_relations.inverse_relations
        ent_ids: dict[str, int] = {}

        for ent in data.get("entities", []):
            name = ent.get("name", "")
            entity_type = ent.get("type", "auto")
            if not name:
                continue

            # 阶段 1：节点 upsert（行 230）
            from_id = await self._graph_repo.upsert_node(user_id, name, entity_type)
            if from_id < 0:
                continue
            ent_ids[name] = from_id

            # 阶段 2：语义关系边 + 反向边（行 234-245）
            for rel in ent.get("relations", []):
                target = rel.get("target", "")
                relation = rel.get("relation", "")
                if not target or not relation:
                    continue
                to_id = await self._graph_repo.upsert_node(user_id, target)
                await self._graph_repo.upsert_edge(from_id, to_id, relation)
                # 只在有明确反向映射时插入反向边（行 243-245）
                if relation in inverse_map:
                    inv = inverse_map[relation]
                    await self._graph_repo.upsert_edge(to_id, from_id, inv)

        # 阶段 3：同轮共现边（行 247-252）
        names = list(ent_ids.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a_id = ent_ids[names[i]]
                b_id = ent_ids[names[j]]
                await self._graph_repo.upsert_edge(a_id, b_id, "")

    # ================================================================
    # 日志
    # ================================================================

    @staticmethod
    def _log_summary(user_id: str, data: dict) -> None:
        """输出分析写入日志摘要。

        Args:
            user_id: 用户标识
            data: 归一化后的分析数据
        """
        log_parts: list[str] = []
        if data.get("traits_updates"):
            log_parts.append(f"traits+{len(data['traits_updates'])}")
        if data.get("shared_jokes"):
            log_parts.append(f"jokes+{len(data['shared_jokes'])}")
        if data.get("boundaries"):
            log_parts.append(f"bounds+{len(data['boundaries'])}")
        if data.get("entities"):
            log_parts.append(f"ents+{len(data['entities'])}")
        try:
            importance = float(data.get("importance", 0))
        except (TypeError, ValueError):
            importance = 0.0
        if importance >= 0.5:
            log_parts.append("event")
        logger.info(
            f"ANALYSIS: write user={user_id} "
            f"{' | '.join(log_parts) if log_parts else 'no-updates'}",
        )

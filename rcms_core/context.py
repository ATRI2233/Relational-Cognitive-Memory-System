import json
import logging
from datetime import datetime

logger = logging.getLogger("rcms")


class ContextMixin:
    """Narrative Context / Prompt Compressor — 供 AstrBot 注入 / standalone chat 使用"""

    def _session_warmup(self, user_id: str, session_id: str, turn_count: int) -> str:
        """新 session 预热：读取上一个 session 的 focus_topic + dangling + 最近蒸馏摘要"""
        if turn_count > 1 or not user_id or not session_id:
            return ""
        row = self.conn.execute("""
            SELECT focus_topic, dangling_threads, last_active
            FROM session_state
            WHERE session_id != ? AND last_active IS NOT NULL
            ORDER BY last_active DESC LIMIT 1
        """, (session_id,)).fetchone()
        if not row:
            return ""
        topic, dangling, last_active = row
        if not topic and not dangling:
            return ""
        parts = []
        if topic:
            parts.append(f"话题：{topic}")
        if dangling:
            try:
                dt = json.loads(dangling)
                if isinstance(dt, dict) and dt.get("threads"):
                    parts.append("未完成：" + "、".join(dt["threads"][:3]))
            except (json.JSONDecodeError, ValueError):
                pass
        if last_active:
            try:
                days = (datetime.now() - datetime.fromisoformat(str(last_active))).days
                if days == 0:
                    parts.append("距上次对话：今天")
                elif days == 1:
                    parts.append("距上次对话：昨天")
                else:
                    parts.append(f"距上次对话：{days} 天前")
            except (ValueError, TypeError):
                pass
        if not parts:
            return ""
        return "\n".join(f"  · {p}" for p in parts)

    def narrative_context(self, stance: str, session_id: str | None = None,
                           memories: list | None = None, long_term: dict | None = None,
                           user_input: str = "", user_id: str = "") -> str:
        parts = []

        # ── 会话统计 ──
        turn_count = 0
        dangling = ""
        focus = ""
        if session_id:
            try:
                row = self.conn.execute(
                    "SELECT turn_count, focus_topic, dangling_threads "
                    "FROM session_state WHERE session_id = ?", (session_id,)
                ).fetchone()
                if row:
                    turn_count = row[0] or 0
                    focus = row[1] or ""
                    dangling = row[2] or ""
            except Exception:
                logger.exception(f"RCMS: 读取 session_state 失败 session={session_id}")

        # ── 新 session 预热 ──
        warmup = self._session_warmup(user_id, session_id, turn_count)
        if warmup:
            parts.append("[上次聊到]\n" + warmup)

        # ── 轮数 ──
        if turn_count:
            parts.append(f"聊了 {turn_count} 轮")

        # ── 用户画像: traits + quirks（强度排序，展示 top5 + 剩余汇总） ──
        profile_lines = []
        if long_term:
            trait_details = long_term.get('trait_details', [])
            trait_details.sort(key=lambda x: x.get("strength", 0), reverse=True)
            all_traits = [td for td in trait_details if not td["text"].startswith("[口癖]")]
            max_show = 5
            for td in all_traits[:max_show]:
                strength = td.get("strength", 0)
                prefix = "" if strength >= 5 else "↘ " if strength <= 2 else "· "
                profile_lines.append(f"{prefix}{td['text']}")
            remaining = len(all_traits) - max_show
            if remaining > 0:
                profile_lines.append(f"及其他 {remaining} 条特质")
            quirks = [(td.get("strength", 0), td["text"].replace("[口癖] ", ""))
                      for td in trait_details if td["text"].startswith("[口癖]")]
            quirks.sort(key=lambda x: x[0], reverse=True)
            if quirks:
                q_mark = "↘ " if any(q[0] <= 2 for q in quirks) else ""
                profile_lines.append(f"{q_mark}口癖: {'、'.join(q[1] for q in quirks[:2])}")
        if profile_lines:
            parts.append("他是什么样的:\n" + '\n'.join(f'  · {t}' for t in profile_lines))

        # ── 结构化画像: 喜好 / 沟通风格 / 自我认同 / 雷区 / 核心身份 ──
        struct_lines = []
        if long_term:
            prefs = long_term.get('preferences', {})
            if prefs.get('likes'):
                struct_lines.append(f"喜好: {'、'.join(prefs['likes'][:5])}")
            if prefs.get('dislikes'):
                struct_lines.append(f"不喜欢: {'、'.join(prefs['dislikes'][:3])}")
            cs = long_term.get('communication_style', '')
            if cs:
                struct_lines.append(f"沟通风格: {cs}")
            si = long_term.get('self_identity', [])
            if si:
                struct_lines.append(f"自我认同: {'、'.join(si[:3])}")
            ci = long_term.get('core_identity', {})
            if ci:
                ci_parts = [v for v in ci.values() if v]
                if ci_parts:
                    struct_lines.append(f"身份: {'·'.join(ci_parts)}")
            bounds = long_term.get('boundaries', [])
            if bounds:
                struct_lines.append(f"雷区: {'、'.join(bounds[:3])}")
        if struct_lines:
            parts.append("结构化画像:\n" + '\n'.join(f'  · {s}' for s in struct_lines))

        # ── 共同语境: 梗 / 上下文 / 实体 / 话题 ──
        ctx_lines = []
        if long_term:
            shared = long_term.get('shared_contexts', [])
            jokes = [s.replace('[梗] ', '') for s in shared if s.startswith('[梗]')][:2]
            other = [s for s in shared if not s.startswith('[梗]')][:2]
            ctx_lines.extend(f"梗: {j}" for j in jokes)
            ctx_lines.extend(other)

            entities = long_term.get('entities', [])
            if entities:
                grouped_ents = {}
                seen_names = set()
                for e in entities:
                    name = e.get('name', '')
                    if not name or name in seen_names:
                        continue
                    seen_names.add(name)
                    etype = e.get('type', 'auto')
                    if etype not in grouped_ents:
                        grouped_ents[etype] = []
                    tag = f" ({e.get('relation', '')})" if e.get('relation') else ""
                    grouped_ents[etype].append(f"{name}{tag}")
                type_labels = {'person': '人', 'place': '地方', 'concept': '概念', 'activity': '活动', 'auto': '相关'}
                for etype, items in grouped_ents.items():
                    label = type_labels.get(etype, etype)
                    ctx_lines.append(f"他提过的{label}: {'、'.join(items[:4])}")
            if focus:
                ctx_lines.append(f"最近总聊: {focus}")
        if ctx_lines:
            parts.append("共同语境:\n" + '\n'.join(f'  · {c}' for c in ctx_lines))

        # ── 最近事件 ──
        ev_lines = []
        if long_term:
            for ev in long_term.get('events', [])[:2]:
                hint = ev.get('hint', '')
                if hint:
                    delta = ev.get('delta', 0)
                    tag = {1: ' ✓', -1: ' ✗'}.get(delta, '')
                    ev_lines.append(f"{hint}{tag}")
        if ev_lines:
            parts.append("最近事件:\n" + '\n'.join(f'  · {e}' for e in ev_lines))

        # ── 三通道记忆（按融合分数排序，最高分通道在前） ──
        if memories:
            channel_map = {'recent': '时间·重要性', 'resonance': '语义检索', 'skeleton': '图谱关联'}
            grouped = {}
            order = []
            for content, tag in memories:
                if tag not in grouped:
                    grouped[tag] = []
                    order.append(tag)
                grouped[tag].append(content)
            ch_lines = []
            for key in order:
                label = channel_map.get(key, key)
                items = grouped[key]
                ch_lines.append(f"【{label}】\n" + "\n".join(f"  · {c}" for c in items))
            if ch_lines:
                parts.append("相关记忆:\n" + "\n\n".join(ch_lines))

        # ── 图谱关系链 ──
        graph_paths = getattr(self, '_graph_paths', [])
        if graph_paths:
            parts.append("图谱关系链:\n" + '\n'.join(f'  · {p}' for p in graph_paths))

        # ── 未完成话题（超 10 轮自动过期） ──
        if dangling:
            try:
                dt_data = json.loads(dangling)
                if isinstance(dt_data, dict):
                    dt_list = dt_data.get("threads", [])
                    since_turn = dt_data.get("turn", 0)
                    if dt_list and turn_count - since_turn <= 10:
                        stale = turn_count - since_turn > 5
                        prefix = "↘ " if stale else ""
                        dangling_display = prefix + "、".join(dt_list[:3])
                        parts.append(f"未完成: {dangling_display}")
                elif isinstance(dt_data, list) and dt_data:
                    parts.append(f"未完成: {'、'.join(dt_data[:3])}")
            except (json.JSONDecodeError, ValueError):
                pass

        parts.append("→ 以上是你通过长期对话积累的对他的了解，用来更好地理解他的意图。人格设定始终优先。")

        return "[RCMS 关系上下文]\n" + "\n\n".join(parts)

    async def prompt_compressor(self, user_id: str, session_id: str, user_input: str,
                           memories: list | None = None,
                           long_term: dict | None = None) -> str:
        if memories is None:
            memories = await self.retrieve_memories(user_id, user_input, 'engaged', session_id=session_id)
        channel_map = {'recent': '时间·重要性', 'resonance': '语义检索', 'skeleton': '图谱关联'}
        grouped = {}
        order = []
        for content, tag in memories:
            if tag not in grouped:
                grouped[tag] = []
                order.append(tag)
            grouped[tag].append(content)
        mem_lines = []
        for key in order:
            if key not in grouped:
                continue
            label = channel_map.get(key, key)
            items = grouped[key][:2]
            mem_lines.append(f"【{label}】\n" + "\n".join(f"  · {c}" for c in items))
        mem_block = "\n\n".join(mem_lines) if mem_lines else ""
        # 图谱关系链
        graph_paths = getattr(self, '_graph_paths', [])
        gp_block = ""
        if graph_paths:
            gp_block = "\n【图谱关系链】\n" + "\n".join(f"  · {p}" for p in graph_paths)
        lt_block = ""
        if long_term:
            shared_ctx = long_term.get('shared_contexts', [])
            if shared_ctx:
                ctx = '、'.join(shared_ctx[:3])
                lt_block += f"\n【共同语境】{ctx}"
            traits = long_term.get('identity_traits', [])
            if traits:
                trait_strs = [t for t in traits if not t.startswith('[口癖]')][:3]
                if trait_strs:
                    lt_block += f"\n【用户特质】{'；'.join(trait_strs)}"
                quirks = [t for t in traits if t.startswith('[口癖]')][:2]
                if quirks:
                    lt_block += f"\n【说话特点】{'；'.join(q.replace('[口癖] ', '') for q in quirks)}"
            cs = long_term.get('communication_style', '')
            if cs:
                lt_block += f"\n【沟通风格】{cs}"
            si = long_term.get('self_identity', [])
            if si:
                lt_block += f"\n【自我认同】{'、'.join(si[:2])}"
            bounds = long_term.get('boundaries', [])
            if bounds:
                lt_block += f"\n【雷区】{'、'.join(bounds[:2])}"
        prompt = "【当前心理状态】\n自然地聊"
        if mem_block:
            prompt += f"\n\n【相关记忆】\n{mem_block}"
        if gp_block:
            prompt += gp_block
        if lt_block:
            prompt += lt_block
        prompt += f"\n\n【底线】\n不主动说教。不假装完全理解。疲惫时简短但不冷漠。\n\n用户: {user_input}\n你:"
        return prompt

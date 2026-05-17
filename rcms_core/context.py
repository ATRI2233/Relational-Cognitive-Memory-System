import json
import logging
from datetime import datetime

logger = logging.getLogger("rcms")


class ContextMixin:
    """Narrative Context / Prompt Compressor — 供 AstrBot 注入 / standalone chat 使用"""

    def narrative_context(self, stance: str, session_id: str | None = None,
                           memories: list | None = None, long_term: dict | None = None,
                           user_input: str = "") -> str:
        parts = []

        # ── 会话统计 ──
        turn_count = 0
        dangling = ""
        focus = ""
        warmth = 0.0
        tension = 0.0
        if session_id:
            try:
                row = self.conn.execute(
                    "SELECT turn_count, focus_topic, dangling_threads, residue_warmth, residue_tension "
                    "FROM session_state WHERE session_id = ?", (session_id,)
                ).fetchone()
                if row:
                    turn_count = row[0] or 0
                    focus = row[1] or ""
                    dangling = row[2] or ""
                    warmth = row[3] or 0.0
                    tension = row[4] or 0.0
            except Exception:
                pass

        # ── 关系 ──
        arc_line = ""
        if long_term:
            arc = long_term.get('arc_stage', 'stranger')
            score = long_term.get('arc_score', 0.0)
            label = {'familiar': '认识一阵了', 'rapport': '算熟了',
                     'history': '老熟人', 'drift': '冷淡过一阵',
                     'reconnect': '重新联系上'}.get(arc, '初识')
            arc_line = f"关系: {label} (分 {score:.1f})"
            if turn_count:
                arc_line += f"，聊了 {turn_count} 轮"
            parts.append(arc_line)

        # ── 当前氛围 ──
        mood_map = {'reflective': '他在回想', 'guarded': '他话里有话',
                    'playful': '气氛轻松带调侃', 'analytical': '他在理性分析',
                    'distant': '他不太想深入', 'intimate': '他在敞开了说'}
        mood = mood_map.get(stance, '气氛平静')
        mood_suffix = ""
        if abs(warmth) > 0.1:
            mood_suffix += f" (warmth {warmth:.1f}"
            mood_suffix += f" / tension {tension:.1f}" if tension > 0.1 else ""
            mood_suffix += ")"
        parts.append(f"当前: {mood}{mood_suffix}")

        # ── 用户画像: traits + quirks + voice（强度排序，展示 top5 + 剩余汇总） ──
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
            voice = long_term.get('voice_hint', '')
            if voice and not all_traits:
                profile_lines.append(voice)
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
                ent_strs = []
                for e in entities[:4]:
                    if not e.get('name'):
                        continue
                    tag = ""
                    if e.get('relation') or e.get('fact'):
                        tag = " (" + "·".join(filter(None, [e.get('relation', ''), e.get('fact', '')])) + ")"
                    ent_strs.append(f"{e['name']}{tag}")
                if ent_strs:
                    ctx_lines.append(f"他提过的人/事: {'、'.join(ent_strs)}")
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

        # ── 相关记忆 ──
        if memories:
            lines = [f'  · {m[0]}' for m in memories[:2]]
            parts.append("相关记忆:\n" + '\n'.join(lines))

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
            except Exception:
                pass

        parts.append("→ 以上是你通过长期对话积累的对他的了解，用来更好地理解他的意图。人格设定始终优先。")

        return "[RCMS 关系上下文]\n" + "\n\n".join(parts)

    async def prompt_compressor(self, user_id: str, session_id: str, user_input: str,
                           memories: list | None = None,
                           long_term: dict | None = None) -> str:
        if memories is None:
            memories = await self.retrieve_memories(user_id, user_input, 'engaged')
        mem_lines = [f"- {m[0]}" for m in memories[:2]]
        mem_block = "\n".join(mem_lines) if mem_lines else ""
        lt_block = ""
        if long_term:
            arc = long_term.get('arc_stage', '')
            if arc and arc != 'stranger':
                stage_map = {'familiar': '已经认识一阵了', 'rapport': '已经很熟了',
                             'history': '是老朋友了', 'drift': '有一阵没联系',
                             'reconnect': '又重新联系上了'}
                lt_block = f"\n【关系】{stage_map.get(arc, '')}"
            if long_term.get('shared_contexts'):
                ctx = '、'.join(long_term['shared_contexts'][:3])
                lt_block += f"\n【共同语境】{ctx}"
            traits = long_term.get('identity_traits', [])
            if traits:
                trait_strs = [t for t in traits if not t.startswith('[口癖]')][:3]
                if trait_strs:
                    lt_block += f"\n【用户特质】{'；'.join(trait_strs)}"
                quirks = [t for t in traits if t.startswith('[口癖]')][:2]
                if quirks:
                    lt_block += f"\n【说话特点】{'；'.join(q.replace('[口癖] ', '') for q in quirks)}"
            prefs = long_term.get('preferences', {})
            if prefs.get('likes') or prefs.get('dislikes'):
                likes = '、'.join(prefs['likes'][:3]) if prefs.get('likes') else ''
                dislikes = '、'.join(prefs['dislikes'][:2]) if prefs.get('dislikes') else ''
                parts = [f"喜欢{likes}" if likes else '', f"不喜欢{dislikes}" if dislikes else '']
                lt_block += f"\n【喜好】{'，'.join(p for p in parts if p)}"
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
        if lt_block:
            prompt += lt_block
        prompt += f"\n\n【底线】\n不主动说教。不假装完全理解。疲惫时简短但不冷漠。\n\n用户: {user_input}\n你:"
        return prompt

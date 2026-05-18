import json
import logging
import os
from datetime import datetime, timedelta

from openai import AsyncOpenAI

logger = logging.getLogger("rcms")


class AnalysisMixin:
    """LLM 事后分析：配置 / prompt / 运行 / 写入"""

    def _get_post_analysis_config(self) -> dict:
        pa = self.analysis_config.get("post_analysis", {})
        return {
            "source": pa.get("source", "astrbot"),
            "api_key": pa.get("custom_api_key", "") or pa.get("api_key", os.environ.get("OPENAI_API_KEY", "")) or pa.get("custom_token", ""),
            "base_url": pa.get("custom_base_url", "") or pa.get("base_url", os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")) or pa.get("custom_url", ""),
            "model": pa.get("custom_model", "") if pa.get("source") == "custom" else "",
            "astrbot_source_id": pa.get("astrbot_source_id", ""),
        }

    async def _apply_analysis(self, user_id: str, session_id: str, user_input: str, reply: str, data: dict):
        now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # 0. Ensure session_state row exists for dangling_threads
        if session_id:
            self.conn.execute("INSERT OR IGNORE INTO session_state (session_id, stance, turn_count, last_active) VALUES (?, 'open', 0, ?)", (session_id, now_str))
            self.conn.execute("UPDATE session_state SET last_active = ? WHERE session_id = ?", (now_str, session_id))

        # 1. Topic tracking — focus_topic from LLM analysis, 比关键词猜测精准
        if data.get("topic_shift") and data.get("key_points"):
            new_topic = data["key_points"][0][:60]
            self.conn.execute(
                "UPDATE session_state SET focus_topic = ? WHERE session_id = ?",
                (new_topic, session_id),
            )

        # 2. Mood & intensity
        mood = data.get("mood", "")
        intensity = data.get("mood_intensity", 0.0)

        # 2b. User state → session_state.stance
        if data.get("user_state") and session_id:
            self.conn.execute(
                "UPDATE session_state SET stance = ? WHERE session_id = ?",
                (data["user_state"], session_id),
            )

        # 3. (relationship arc removed)

        # 4. Identity traits + quirks — 单次 LLM 已产出语义去重，无需额外 embedding API
        identity = self.conn.execute("SELECT traits FROM identity_memory WHERE user_id = ?", (user_id,)).fetchone()
        if identity:
            raw = json.loads(identity[0]) if identity[0] else []
            trait_map = {}
            for item in raw:
                if isinstance(item, str):
                    trait_map[item] = {"s": 3, "c": 0}
                elif isinstance(item, dict):
                    trait_map[item.get("t", "")] = {"s": item.get("s", 0), "c": item.get("c", 0)}

            confirmed = set()
            for t in data.get("traits_updates", []):
                if t not in trait_map:
                    trait_map[t] = {"s": 5, "c": 1}
                else:
                    trait_map[t]["s"] = 5
                    trait_map[t]["c"] += 1
                confirmed.add(t)

            # Quirks join the pool
            for q in data.get("speech_quirks", []):
                q_entry = f"[口癖] {q}"
                if q_entry not in trait_map:
                    trait_map[q_entry] = {"s": 0, "c": 0}
                trait_map[q_entry]["s"] = 5
                trait_map[q_entry]["c"] += 1
                confirmed.add(q_entry)

            # Decay unconfirmed: strength -= 1, floor = min(c, 3)
            for t in list(trait_map.keys()):
                if t not in confirmed:
                    floor = min(trait_map[t]["c"], 3)
                    trait_map[t]["s"] = max(trait_map[t]["s"] - 1, floor)
                    if trait_map[t]["s"] <= 0:
                        del trait_map[t]

            new_traits_json = [{"t": t, "s": v["s"], "c": v["c"]} for t, v in trait_map.items()]
            if new_traits_json != raw:
                self.conn.execute(
                    "UPDATE identity_memory SET traits = ?, updated_at = ? WHERE user_id = ?",
                    (json.dumps(new_traits_json, ensure_ascii=False), now_str, user_id),
                )

        # 4b. 结构化身份字段（覆盖写，LLM 每次产出完整快照）
        id_updates = []
        id_params = []
        for col, key, default in [
            ("preferences", "preferences", "{}"),
            ("communication_style", "communication_style", ""),
            ("self_identity", "self_identity", "[]"),
            ("core_identity", "core_identity", "{}"),
        ]:
            val = data.get(key)
            if val is not None:
                if isinstance(val, (dict, list)):
                    val = json.dumps(val, ensure_ascii=False)
                id_updates.append(f"{col} = ?")
                id_params.append(val)
        if id_updates:
            id_params.extend([now_str, user_id])
            self.conn.execute(
                f"UPDATE identity_memory SET {', '.join(id_updates)}, updated_at = ? WHERE user_id = ?",
                id_params,
            )

        # 5. Shared jokes/context
        for joke in data.get("shared_jokes", []):
            trigger = joke.get("trigger", "")
            ctx = joke.get("context", "")
            if trigger:
                existing = self.conn.execute(
                    "SELECT context_id FROM shared_context WHERE user_id = ? AND context_body LIKE ?",
                    (user_id, f"%{trigger}%"),
                ).fetchone()
                if existing:
                    self.conn.execute(
                        "UPDATE shared_context SET omission_count = omission_count + 1 WHERE context_id = ?",
                        (existing[0],),
                    )
                else:
                    self.conn.execute(
                        "INSERT INTO shared_context (user_id, context_body, omission_count, confirmed) VALUES (?, ?, 1, 1)",
                        (user_id, f"[梗] {trigger} → {ctx}"),
                    )

        # 6. Boundaries — 覆盖写（LLM 已参考现有雷区，产出即完整快照）
        boundaries = data.get("boundaries")
        if boundaries is not None and isinstance(boundaries, list):
            self.conn.execute(
                "UPDATE identity_memory SET boundaries = ?, updated_at = ? WHERE user_id = ?",
                (json.dumps(boundaries, ensure_ascii=False), now_str, user_id),
            )

        # 7. Dangling threads → cognitive_distill
        for dt in data.get("dangling_threads", []):
            self.conn.execute(
                "INSERT INTO cognitive_distill (user_id, content, summary, importance, created_at) VALUES (?, ?, ?, ?, ?)",
                (user_id, dt, dt[:40] + "..." if len(dt) > 40 else dt, 0.5, now_str),
            )
        if session_id and data.get("dangling_threads"):
            row = self.conn.execute("SELECT turn_count FROM session_state WHERE session_id = ?", (session_id,)).fetchone()
            current_turn = row[0] if row else 0
            self.conn.execute(
                "UPDATE session_state SET dangling_threads = ? WHERE session_id = ?",
                (json.dumps({"threads": data["dangling_threads"], "turn": current_turn}, ensure_ascii=False), session_id),
            )

        # 8. Entities → 图谱边（entity_relations 已废弃，统一由图谱带 relation 的边承载）
        for ent in data.get("entities", []):
            from_name = ent.get("name", "")
            rel = ent.get("relation", "")
            to_name = ent.get("fact", "")
            if not from_name or not rel or not to_name:
                continue
            from_id = self._upsert_graph_node(user_id, from_name, now_str)
            to_id = self._upsert_graph_node(user_id, to_name, now_str)
            if from_id != to_id:
                self._upsert_graph_edge(from_id, to_id, now_str, relation=rel)

        # 9. Event memory (if important enough) → cognitive_distill
        importance = data.get("importance", 0.0)
        if importance >= 0.5:
            summary = user_input[:80] + "..." if len(user_input) > 80 else user_input
            existing = self.conn.execute(
                "SELECT id FROM cognitive_distill WHERE user_id = ? AND content = ?", (user_id, summary)
            ).fetchone()
            if not existing:
                self.conn.execute(
                    "INSERT INTO cognitive_distill (user_id, content, summary, mood, mood_intensity, importance, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (user_id, summary, summary[:40] + "...", mood, intensity, importance, now_str),
                )

        # 10. Key facts → cognitive_distill（保底 importance 0.5 防止被碎片清理删除）
        kf_imp = max(importance, 0.5)
        kfs = data.get("key_facts", []) or data.get("key_facts_structured", [])
        for kf in kfs[:3]:
            if isinstance(kf, str):
                content = kf
                expires_at = None
            elif isinstance(kf, dict):
                content = kf.get("content", "")
                temporal = kf.get("temporal", "permanent")
                if temporal == "transient" and kf.get("expires_after_days"):
                    expires_at = (datetime.now() + timedelta(days=int(kf["expires_after_days"]))).strftime('%Y-%m-%d %H:%M:%S')
                else:
                    expires_at = None
            else:
                continue
            if content:
                self.conn.execute(
                    "INSERT INTO cognitive_distill (user_id, content, summary, importance, expires_at, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                    (user_id, content, content[:60], kf_imp, expires_at, now_str),
                )

        log_parts = []
        if data.get("traits_updates"): log_parts.append(f"traits+{len(data['traits_updates'])}")
        if data.get("shared_jokes"): log_parts.append(f"jokes+{len(data['shared_jokes'])}")
        if data.get("boundaries"): log_parts.append(f"bounds+{len(data['boundaries'])}")
        if data.get("key_facts"): log_parts.append(f"facts+{len(data['key_facts'])}")
        if data.get("entities"): log_parts.append(f"ents+{len(data['entities'])}")
        if data.get("importance", 0) >= 0.5: log_parts.append("event")
        logger.info(f"ANALYSIS: write user={user_id} {' | '.join(log_parts) if log_parts else 'no-updates'}")

        self.conn.commit()

    # ── 蒸馏版 LLM 分析（单次调用产出摘要 + 9 维 JSON） ──

    async def _run_distill_analysis(self, user_id: str, session_id: str, snapshot_text: str, long_term: dict, last_turn: int, turn_count: int):
        """蒸馏触发的 LLM 分析：一次调用产出摘要 + 9 维 JSON，然后写入"""
        cfg = self._get_post_analysis_config()
        if not cfg["api_key"] and not self._llm_call:
            logger.warning("DISTILL: no LLM configured, skipping")
            return

        logger.info(f"DISTILL: start user={user_id} turns={last_turn}→{turn_count}")
        prompt = self._build_distill_prompt(snapshot_text, long_term)
        content = None
        try:
            if self._llm_call:
                content = await self._llm_call(prompt, model=cfg["model"])
            else:
                client = AsyncOpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"])
                try:
                    resp = await client.chat.completions.create(
                        model=cfg["model"],
                        messages=[{"role": "user", "content": prompt}],
                        response_format={"type": "json_object"},
                    )
                    content = resp.choices[0].message.content or "{}"
                finally:
                    await client.close()
        except Exception as e:
            logger.warning(f"DISTILL: LLM call failed ({e})")
            return

        if not content:
            logger.warning("DISTILL: empty response")
            return

        try:
            result = json.loads(content)
            summary = result.get("summary", "")
            analysis = result.get("analysis", {})
            if not summary:
                logger.warning("DISTILL: no summary in response")
                return
            logger.info(f"DISTILL: ok summary={summary[:60]} mood={analysis.get('mood','?')}")
        except json.JSONDecodeError:
            logger.warning(f"DISTILL: invalid JSON: {content[:200]}")
            return

        # 写入蒸馏摘要 + 清理碎片
        self._apply_distill(user_id, session_id, last_turn, turn_count, summary)

        # 写入 9 维分析（情感/特质/实体等）
        await self._apply_analysis(user_id, session_id, summary[:80], "", analysis)

    def _build_distill_prompt(self, snapshot_text: str, long_term: dict) -> str:
        """两阶段蒸馏分析 prompt：先理解对话脉络，再精确提取信息"""
        lt_hint = ""
        if long_term:
            if long_term.get("identity_traits"):
                lt_hint += f"\n已知特质: {json.dumps(long_term['identity_traits'], ensure_ascii=False)}"
            if long_term.get("preferences"):
                lt_hint += f"\n已知喜好: {json.dumps(long_term['preferences'], ensure_ascii=False)}"
            if long_term.get("communication_style"):
                lt_hint += f"\n沟通风格: {long_term['communication_style']}"
            if long_term.get("self_identity"):
                lt_hint += f"\n自我认同: {json.dumps(long_term['self_identity'], ensure_ascii=False)}"
            if long_term.get("core_identity"):
                lt_hint += f"\n核心身份: {json.dumps(long_term['core_identity'], ensure_ascii=False)}"
            if long_term.get("boundaries"):
                lt_hint += f"\n已知雷区: {json.dumps(long_term['boundaries'], ensure_ascii=False)}"

        return f"""你是一个对话分析系统。以下是最近多轮对话的完整记录：

{snapshot_text}
{lt_hint}

请按两阶段分析：

第一阶段：理解对话脉络
通读整段对话，理解：
· 发生了什么事——谁说了什么、做了什么、事件顺序
· 情绪基调——整体氛围轻松/紧张/热烈/冷淡
· 人物关系——参与者之间的互动模式

第二阶段：精确提取
基于对对话的理解，产出以下 JSON：

{{
  "summary": "像人复述一样概括这段对话。不要干巴巴的要点罗列，而是连贯叙述：谁做了什么、说了什么、气氛如何。保留对话中的生动细节和转折。",
  "analysis": {{
    "key_facts": [
      "从对话中提取的精确事实列表。每一条是一个独立、完整的陈述：主语+事件+细节。例如「攒抽进行中360沉迷MC搞建筑，尝试先复刻后创作」而非「有人在玩MC」"
    ],
    "key_facts_structured": [
      {"content": "完整可独立理解的事实", "temporal": "permanent"},
      {"content": "临时性事件如面试计划等", "temporal": "transient", "expires_after_days": 14}
    ],
    "mood": "温暖|低落|焦虑|平静|兴奋|防御|疏远",
    "mood_intensity": 0.0~1.0,
    "topic_shift": true/false,
    "key_points": ["事件脉络的简要概括（2-4条）"],
    "user_state": "open|reflective|guarded|playful|analytical|distant|intimate",
    "traits_updates": ["从对话中发现的用户特质"],
    "speech_quirks": ["说话特点"],
    "preferences": {{"likes": ["喜欢的事物"], "dislikes": ["不喜欢的事物"]}},
    "communication_style": "总结用户的说话方式",
    "self_identity": ["用户如何看待自己"],
    "core_identity": {{"职业": "", "角色": "", "标签": ""}},
    "boundaries": ["避免做的事、雷区（完整列表，参考已有雷区增删）"],
    "dangling_threads": ["未完成的话题"],
    "importance": 0.0~1.0,
    "entities": [
      {{"name": "人物/事物名", "relation": "与用户的关系", "fact": "关键事实"}}
    ]
  }}
}}

要求：
· summary 要像人聊天时复述事情一样，有叙事感
· key_facts 与 key_facts_structured 任选一种输出，后者可指定时效性
· key_facts_structured[].temporal 为 permanent 永久保留，transient 到期自动清理
· entities 优先提取反复提及或带有强烈情感的人物/事物
· 只输出 JSON，不要其他文字"""

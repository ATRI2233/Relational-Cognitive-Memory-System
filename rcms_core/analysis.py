import json
import logging
import os
from datetime import datetime, timedelta

from openai import AsyncOpenAI

logger = logging.getLogger("rcms")


class AnalysisMixin:
    """LLM 事后分析：配置 / prompt / 运行 / 写入"""

    _INVERSE_RELATIONS = {
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
        new_entries = []  # (entry_id, text_for_embedding) 待写回向量
        if session_id:
            self.conn.execute("INSERT OR IGNORE INTO session_state (session_id, stance, turn_count, last_active) VALUES (?, 'open', 0, ?)", (session_id, now_str))
            self.conn.execute("UPDATE session_state SET last_active = ? WHERE session_id = ?", (now_str, session_id))

        # 兼容新/旧输出格式：将可能的 dict 项归一化为旧版简单类型（string 列表）以免后续逻辑出错
        traits_updates = []
        for t in data.get("traits_updates", []) or []:
            if isinstance(t, dict):
                trait = t.get("trait") or t.get("t") or None
                if trait:
                    traits_updates.append(trait)
            elif isinstance(t, str):
                traits_updates.append(t)

        speech_quirks = []
        for q in data.get("speech_quirks", []) or []:
            if isinstance(q, dict):
                quirk = q.get("quirk") or q.get("q") or q.get("text") or None
                if quirk:
                    speech_quirks.append(quirk)
            elif isinstance(q, str):
                speech_quirks.append(q)

        dangling_threads = []
        for dt in data.get("dangling_threads", []) or []:
            if isinstance(dt, dict):
                content = dt.get("content") or dt.get("text") or None
                if content:
                    dangling_threads.append(content)
            elif isinstance(dt, str):
                dangling_threads.append(dt)

        key_facts_list = []
        for kf in data.get("key_facts", []) or []:
            if isinstance(kf, dict):
                c = kf.get("content") or kf.get("text") or None
                if c:
                    key_facts_list.append(c)
            elif isinstance(kf, str):
                key_facts_list.append(kf)

        _norm_data = dict(data)
        _norm_data["traits_updates"] = traits_updates
        _norm_data["speech_quirks"] = speech_quirks
        _norm_data["dangling_threads"] = dangling_threads
        _norm_data["key_facts"] = key_facts_list
        data = _norm_data

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
            new_entries.append((self.conn.execute("SELECT last_insert_rowid()").fetchone()[0], dt))
        if session_id and data.get("dangling_threads"):
            row = self.conn.execute("SELECT turn_count FROM session_state WHERE session_id = ?", (session_id,)).fetchone()
            current_turn = row[0] if row else 0
            self.conn.execute(
                "UPDATE session_state SET dangling_threads = ? WHERE session_id = ?",
                (json.dumps({"threads": data["dangling_threads"], "turn": current_turn}, ensure_ascii=False), session_id),
            )

        # 8. Entities → 图谱（带 type 的多关系节点 + 反向边）
        for ent in data.get("entities", []):
            name = ent.get("name", "")
            entity_type = ent.get("type", "auto")
            relations = ent.get("relations", [])
            if not name or not relations:
                continue
            from_id = self._upsert_graph_node(user_id, name, now_str, entity_type=entity_type)
            if from_id < 0:
                continue
            for rel in relations:
                target = rel.get("target", "")
                relation = rel.get("relation", "")
                if not target or not relation:
                    continue
                to_id = self._upsert_graph_node(user_id, target, now_str)
                self._upsert_graph_edge(from_id, to_id, now_str, relation=relation, created_at=now_str)
                # 只在有明确反向映射时插入反向边，不兜底"相关于"
                if relation in self._INVERSE_RELATIONS:
                    inv = self._INVERSE_RELATIONS[relation]
                    self._upsert_graph_edge(to_id, from_id, now_str, relation=inv, created_at=now_str)

        # 10. Key facts → cognitive_distill（保底 importance 0.5 防止被碎片清理删除）
        importance = data.get("importance", 0.0)
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
                new_entries.append((self.conn.execute("SELECT last_insert_rowid()").fetchone()[0], content[:512]))

        log_parts = []
        if data.get("traits_updates"): log_parts.append(f"traits+{len(data['traits_updates'])}")
        if data.get("shared_jokes"): log_parts.append(f"jokes+{len(data['shared_jokes'])}")
        if data.get("boundaries"): log_parts.append(f"bounds+{len(data['boundaries'])}")
        if data.get("key_facts"): log_parts.append(f"facts+{len(data['key_facts'])}")
        if data.get("entities"): log_parts.append(f"ents+{len(data['entities'])}")
        if data.get("importance", 0) >= 0.5: log_parts.append("event")
        logger.info(f"ANALYSIS: write user={user_id} {' | '.join(log_parts) if log_parts else 'no-updates'}")

        self.conn.commit()

        # 10b. 批量写回 key_facts / events / dangling_threads 的向量
        for eid, text in new_entries:
            vec = await self._get_embedding(text)
            if vec:
                self._store_embedding(user_id, eid, vec)

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

        # 写入蒸馏摘要 + 清理碎片（带 mood，让通道 2 情绪共振真正工作）
        mood = analysis.get("mood", "")
        mood_intensity = analysis.get("mood_intensity", 0.0)
        await self._apply_distill(user_id, session_id, last_turn, turn_count, summary, mood, mood_intensity)

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

        return f"""[SYSTEM]
你是结构化对话蒸馏器。两阶段分析：先理解脉络，再提取 JSON。只输出 JSON，不要额外文字。

[USER]
对话快照：
{snapshot_text}
{lt_hint}

第一阶段：理解对话脉络
· 发生了什么事——谁说了什么、做了什么、事件顺序
· 情绪基调——整体氛围轻松/紧张/热烈/冷淡
· 人物关系——参与者之间的互动模式

第二阶段：精确提取。返回如下 JSON：

{{
  "summary": "用第二人称「你」概括这段对话中对方身上发生的事和情绪。不要第三人称「用户/助手」。保留具体细节和情绪转折。",
  "analysis": {{
    "key_facts": ["完整可独立理解的事实，保留时间/原因/经过/结果等具体细节"],
    "key_facts_structured": [
      {{"content": "永久特质如用户是 INFJ", "temporal": "permanent", "expires_after_days": null}},
      {{"content": "临时事件如下周面试", "temporal": "transient", "expires_after_days": 14}}
    ],
    "mood": "温暖|低落|焦虑|平静|兴奋|防御|疏远",
    "mood_intensity": 0.0~1.0,
    "topic_shift": true|false,
    "key_points": ["事件脉络简要概括"],
    "user_state": "open|reflective|guarded|playful|analytical|distant|intimate",
    "traits_updates": ["从对话中发现的用户特质（字符串列表）"],
    "speech_quirks": ["说话特点（字符串列表）"],
    "preferences": {{"likes": ["事物"], "dislikes": ["事物"]}},
    "communication_style": "总结用户的说话方式（纯文本）",
    "self_identity": ["用户如何看待自己"],
    "core_identity": {{"职业": "", "角色": "", "标签": ""}},
    "boundaries": ["雷区列表"],
    "dangling_threads": ["未完成话题（字符串列表）"],
    "importance": 0.0~1.0,
    "entities": [
      {{
        "name": "实体名", "canonical_name": "标准名", "type": "person|place|concept|activity",
        "relations": [
          {{"target": "目标标准名", "relation": "关系类型"}}
        ]
      }}
    ]
  }},
  "meta": {{"truncated": false}}
}}

规则：
1. summary 用第二人称「你」，不要「用户/助手」
2. key_facts 保留具体细节，不要干巴巴的一句话
3. 同一实体的不同表述用 canonical_name 统一消歧，不同 name 指向同一个 canonical_name
4. 关系提取：
   · 优先提取实体之间的客观关系（概念层级/归属/同类并列/空间关联/人物关联）
   · 不提取纯态度关系（如喜欢/讨厌），除非该态度被反复讨论上升为概念
   · 鼓励链式关系：若文本隐含 A→B 和 B→C，即使未明说也要提取，让图谱连成 A→B→C
   · 非人节点（concept/activity）必须参与关系提取，不要只输出人物节点
   · 关系方向：A 是 B 的上位词时，写成 A 属于 B（A→B），不要反向
   · 只提取文本内实体，禁止外部知识补充文本未提及的实体或关系
5. 若无法提取字段用 null 或空数组，不要占位文本或解释
6. 输入过长时优先保留最近对话，设置 meta.truncated = true
7. 遇到可能的敏感信息请掩码处理

只输出 JSON。"""

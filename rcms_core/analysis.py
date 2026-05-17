import json
import logging
import os
from datetime import datetime

import numpy as np
from openai import AsyncOpenAI

logger = logging.getLogger("rcms")


class AnalysisMixin:
    """LLM 事后分析：配置 / prompt / 运行 / 写入"""

    def _get_post_analysis_config(self) -> dict:
        pa = self.analysis_config.get("post_analysis", {})
        return {
            "mode": pa.get("mode", "rule"),
            "sampling": pa.get("sampling", 0.0),
            "source": pa.get("source", "astrbot"),
            "api_key": pa.get("custom_api_key", "") or pa.get("api_key", os.environ.get("OPENAI_API_KEY", "")),
            "base_url": pa.get("custom_base_url", "") or pa.get("base_url", os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")),
            "model": pa.get("custom_model", "") if pa.get("source") == "custom" else "",
            "astrbot_source_id": pa.get("astrbot_source_id", ""),
        }

    async def _run_analysis(self, user_id: str, session_id: str, user_input: str, reply: str, long_term: dict):
        cfg = self._get_post_analysis_config()
        if cfg["mode"] != "llm":
            return
        if cfg["sampling"] < 1.0 and np.random.random() > cfg["sampling"]:
            logger.debug(f"ANALYSIS: user={user_id} skipped by sampling (rate={cfg['sampling']})")
            return

        logger.info(f"ANALYSIS: start user={user_id} model={cfg['model']}")
        prompt = self._build_analysis_prompt(user_id, user_input, reply, long_term)
        content = None
        try:
            if self._llm_call:
                content = await self._llm_call(prompt, model=cfg["model"])
                logger.debug(f"ANALYSIS: via callback len={len(content or '')}")
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
            logger.warning(f"ANALYSIS: LLM call failed ({e})")
            return

        if not content:
            logger.warning("ANALYSIS: empty response")
            return
        try:
            data = json.loads(content)
            logger.info(f"ANALYSIS: ok mood={data.get('mood','?')} delta={data.get('relationship_delta',0)} importance={data.get('importance',0)}")
        except json.JSONDecodeError:
            logger.warning(f"ANALYSIS: invalid JSON: {content[:200]}")
            return

        await self._apply_analysis(user_id, session_id, user_input, reply, data)

    def _build_analysis_prompt(self, user_id: str, user_input: str, reply: str, long_term: dict) -> str:
        lt_hint = ""
        if long_term:
            arc = long_term.get("arc_stage", "stranger")
            traits = long_term.get("identity_traits", [])
            if traits:
                lt_hint += f"\n已知特质: {json.dumps(traits, ensure_ascii=False)}"
            if arc != "stranger":
                lt_hint += f"\n关系阶段: {arc}"
        return f"""你是一个对话分析器。分析以下对话，输出 JSON。

用户说: {user_input}
你回: {reply}{lt_hint}

输出 JSON 格式（请严格按此结构）:
{{
  "mood": "温暖|低落|焦虑|平静|兴奋|防御|疏远",
  "mood_intensity": 0.0~1.0,
  "topic_shift": true/false,
  "key_points": ["摘要1", "摘要2"],
  "relationship_delta": -1|0|1,
  "user_state": "open|reflective|guarded|playful|analytical|distant|intimate",
  "traits_updates": ["新观察到的用户特质（避免和已知特质重复）"],
  "speech_quirks": ["说话特点"],
  "shared_jokes": [{{"trigger": "关键词", "context": "梗/黑话的描述"}}],
  "boundary_hits": ["避免做的事"],
  "dangling_threads": ["未完成的话题"],
  "importance": 0.0~1.0,
  "entities": [{{"name": "人名", "relation": "关系", "fact": "相关事实"}}]
}}

只输出 JSON，不要其他文字。"""

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

        # 2. Emotional trace
        mood = data.get("mood", "")
        intensity = data.get("mood_intensity", 0.0)
        warmth_map = {"温暖": 0.5, "低落": -0.3, "焦虑": -0.4, "平静": 0.1, "兴奋": 0.6, "防御": -0.2, "疏远": -0.5}
        tension_map = {"温暖": 0.0, "低落": 0.1, "焦虑": 0.7, "平静": 0.0, "兴奋": 0.3, "防御": 0.6, "疏远": 0.4}
        warmth = warmth_map.get(mood, 0.0) * intensity
        tension = tension_map.get(mood, 0.0) * intensity
        self.conn.execute(
            "INSERT INTO emotional_trace (user_id, warmth, tension, uncertainty, distance, prose_hint, created_at) VALUES (?, ?, ?, 0.0, 0.0, ?, ?)",
            (user_id, warmth, tension, mood, now_str),
        )

        # 3. Relationship arc
        rd = data.get("relationship_delta", 0)
        if rd != 0:
            row = self.conn.execute("SELECT stage, stage_score FROM relationship_arc WHERE user_id = ?", (user_id,)).fetchone()
            if row:
                new_score = max(0.0, row[1] + rd * 0.5)
                stage = row[0]
                thresholds = {"stranger": 4.0, "familiar": 10.0, "rapport": 20.0, "history": 35.0}
                for s, th in thresholds.items():
                    if new_score >= th and ["stranger", "familiar", "rapport", "history"].index(s) > ["stranger", "familiar", "rapport", "history"].index(stage):
                        stage = s
                self.conn.execute(
                    "UPDATE relationship_arc SET stage = ?, stage_score = ?, updated_at = ? WHERE user_id = ?",
                    (stage, new_score, now_str, user_id),
                )

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

        # 6. Boundary hits
        for bh in data.get("boundary_hits", []):
            existing = self.conn.execute(
                "SELECT context_id FROM shared_context WHERE user_id = ? AND context_body LIKE ?",
                (user_id, f"%{bh}%"),
            ).fetchone()
            if not existing:
                self.conn.execute(
                    "INSERT INTO shared_context (user_id, context_body, omission_count, confirmed) VALUES (?, ?, 1, 1)",
                    (user_id, f"[边界] {bh}"),
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

        # 8. Entities → entity_relations
        for ent in data.get("entities", []):
            name = ent.get("name", "")
            if not name:
                continue
            self.conn.execute(
                """INSERT INTO entity_relations (user_id, entity_name, relation_type, property, mention_count, last_mentioned, sentiment)
                   VALUES (?, ?, ?, ?, 1, ?, 0.0)
                   ON CONFLICT(user_id, entity_name) DO UPDATE SET
                       mention_count = mention_count + 1,
                       last_mentioned = excluded.last_mentioned,
                       relation_type = CASE WHEN excluded.relation_type != '' THEN excluded.relation_type ELSE entity_relations.relation_type END,
                       property = CASE WHEN excluded.property != '' THEN excluded.property ELSE entity_relations.property END""",
                (user_id, name, ent.get("relation", ""), ent.get("fact", ""), now_str),
            )

        # 8b. Entities → 图谱边（带 relation 语义）
        for ent in data.get("entities", []):
            from_name = ent.get("name", "")
            rel = ent.get("relation", "")
            to_name = ent.get("fact", "")
            if not from_name or not rel or not to_name:
                continue
            from_id = self._upsert_graph_node(user_id, from_name, now_str)
            to_id = self._upsert_graph_node(user_id, to_name, now_str)
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

        log_parts = []
        if data.get("traits_updates"): log_parts.append(f"traits+{len(data['traits_updates'])}")
        if data.get("shared_jokes"): log_parts.append(f"jokes+{len(data['shared_jokes'])}")
        if data.get("boundary_hits"): log_parts.append(f"bounds+{len(data['boundary_hits'])}")
        if data.get("entities"): log_parts.append(f"ents+{len(data['entities'])}")
        if data.get("importance", 0) >= 0.5: log_parts.append("event")
        logger.info(f"ANALYSIS: write user={user_id} {' | '.join(log_parts) if log_parts else 'no-updates'}")

        self.conn.commit()

import json
import logging
import os
from datetime import datetime, timedelta
from contextlib import nullcontext

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
            "personality_type": pa.get("personality_type", "default"),
        }

    async def _apply_analysis(self, user_id: str, session_id: str, user_input: str, reply: str, data: dict):
        now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        new_entries = []  # (entry_id, text_for_embedding) 待写回向量
        # 使用实例级 DB 锁序列化写入（短期阻塞）
        lock = getattr(self, '_db_lock', None)
        if lock:
            lock.acquire()
        try:
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
                    q_entry = f"[小细节小习惯] {q}"
                    if q_entry not in trait_map:
                        trait_map[q_entry] = {"s": 0, "c": 0}
                    trait_map[q_entry]["s"] = 5
                    trait_map[q_entry]["c"] += 1
                    confirmed.add(q_entry)
    
                # Decay unconfirmed: strength -= 1, floor = min(c // 2, 2)
                for t in list(trait_map.keys()):
                    if t not in confirmed:
                        floor = min(trait_map[t]["c"] // 2, 2)
                        trait_map[t]["s"] = max(trait_map[t]["s"] - 1, floor)
                        if trait_map[t]["s"] <= 0:
                            del trait_map[t]
    
                # 容量上限：超过30条时按 s*2+c 排序保留前30
                if len(trait_map) > 30:
                    sorted_t = sorted(trait_map.items(), key=lambda x: x[1]["s"] * 2 + x[1]["c"], reverse=True)[:30]
                    trait_map = dict(sorted_t)
    
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
                ("self_identity", "self_identity", "[]"),
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
                cur = self.conn.execute(
                    "INSERT INTO cognitive_distill (user_id, content, keylabel, summary, importance, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                    (user_id, dt, dt, dt, 0.5, now_str),
                )
                new_entries.append((cur.lastrowid, dt))
            if session_id and data.get("dangling_threads"):
                row = self.conn.execute("SELECT turn_count FROM session_state WHERE session_id = ?", (session_id,)).fetchone()
                current_turn = row[0] if row else 0
                self.conn.execute(
                    "UPDATE session_state SET dangling_threads = ? WHERE session_id = ?",
                    (json.dumps({"threads": data["dangling_threads"], "turn": current_turn}, ensure_ascii=False), session_id),
                )
    
            # 8. Entities → 图谱（节点保底 + 语义边优先 + 同轮共现兜底）
            ent_ids = {}  # name -> node_id，用于后面建共现边
            for ent in data.get("entities", []):
                name = ent.get("name", "")
                entity_type = ent.get("type", "auto")
                if not name:
                    continue
                from_id = self._upsert_graph_node(user_id, name, now_str, entity_type=entity_type)
                if from_id < 0:
                    continue
                ent_ids[name] = from_id
                relations = ent.get("relations", [])
                for rel in relations:
                    target = rel.get("target", "")
                    relation = rel.get("relation", "")
                    if not target or not relation:
                        continue
                    to_id = self._upsert_graph_node(user_id, target, now_str)
                    self._upsert_graph_edge(from_id, to_id, now_str, relation=relation, created_at=now_str)
                    # 只在有明确反向映射时插入反向边
                    if relation in self._INVERSE_RELATIONS:
                        inv = self._INVERSE_RELATIONS[relation]
                        self._upsert_graph_edge(to_id, from_id, now_str, relation=inv, created_at=now_str)
            # 同轮共现：本轮出现的实体之间建空 relation 边（权重累积，后续语义边会覆盖）
            names = list(ent_ids.keys())
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    a_id = ent_ids[names[i]]
                    b_id = ent_ids[names[j]]
                    self._upsert_graph_edge(a_id, b_id, now_str, relation="", created_at=now_str)
    
            # 10. Key facts → cognitive_distill（permanent 保底 0.5，transient 无保底可被清理）
            importance = data.get("importance", 0.0)
            kf_imp = max(importance, 0.5)
            kfs = data.get("key_facts", []) or data.get("key_facts_structured", [])
            perm_count = 0
            tran_count = 0
            for kf in kfs:
                if isinstance(kf, str):
                    content = kf
                    temporal = "permanent"
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
                if not content:
                    continue
                if temporal == "permanent":
                    if perm_count >= 3:
                        continue
                    imp_val = kf_imp
                    perm_count += 1
                else:
                    if tran_count >= 5:
                        continue
                    imp_val = importance  # transient 无保底，可被正常衰减清理
                    tran_count += 1
                cur = self.conn.execute(
                    "INSERT INTO cognitive_distill (user_id, content, keylabel, summary, importance, expires_at, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (user_id, content, content, content, imp_val, expires_at, now_str),
                )
                new_entries.append((cur.lastrowid, content[:512]))
    
            log_parts = []
            if data.get("traits_updates"): log_parts.append(f"traits+{len(data['traits_updates'])}")
            if data.get("shared_jokes"): log_parts.append(f"jokes+{len(data['shared_jokes'])}")
            if data.get("boundaries"): log_parts.append(f"bounds+{len(data['boundaries'])}")
            if data.get("key_facts"): log_parts.append(f"facts+{len(data['key_facts'])}")
            if data.get("entities"): log_parts.append(f"ents+{len(data['entities'])}")
            if data.get("importance", 0) >= 0.5: log_parts.append("event")
            logger.info(f"ANALYSIS: write user={user_id} {' | '.join(log_parts) if log_parts else 'no-updates'}")
    
            self.conn.commit()
        finally:
            if lock:
                try:
                    lock.release()
                except Exception:
                    pass

        # 10b. 批量写回 key_facts / events / dangling_threads 的向量
        for eid, text in new_entries:
            vec = await self._get_embedding(text)
            if vec:
                self._store_embedding(user_id, eid, vec)

    # ── 蒸馏版 LLM 分析（单次调用产出摘要 + 9 维 JSON） ──

    async def _run_distill_analysis(self, user_id: str, session_id: str, snapshot_text: str, long_term: dict, last_turn: int, turn_count: int, persona_name: str = "Bot", senders: list = None):
        """蒸馏触发的 LLM 分析：一次调用产出摘要 + 9 维 JSON，然后写入"""
        cfg = self._get_post_analysis_config()
        if not cfg["api_key"] and not self._llm_call:
            logger.warning("DISTILL: no LLM configured, skipping")
            return

        logger.info(f"DISTILL: start user={user_id} turns={last_turn}→{turn_count} persona={persona_name}")

        # Build personality style from long_term for Bot first-person narration
        style_parts = []
        traits = long_term.get('identity_traits', [])
        if traits:
            style_parts.append('特质：' + '、'.join(traits[:3]))
        personality_style = '。'.join(style_parts)

        # Detect group: more than 1 sender excluding Bot itself
        other_senders = [s for s in (senders or []) if s != persona_name]
        is_group = len(other_senders) > 1

        personality_type = cfg.get("personality_type", "default")
        prompt = self._build_distill_prompt(snapshot_text, long_term, persona_name=persona_name, personality_style=personality_style, is_group=is_group, personality_type=personality_type)
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

        # 保存原始 LLM 响应以便审计与后续人工回滚
        try:
            cur = self.conn.execute(
                "INSERT INTO analysis_raw (user_id, session_id, content) VALUES (?, ?, ?)",
                (user_id, session_id, content),
            )
            raw_id = cur.lastrowid
            self.conn.commit()
        except Exception:
            logger.exception(f"DISTILL: failed to persist raw response user={user_id}")
            raw_id = None

        if not content:
            logger.warning("DISTILL: empty response")
            return

        try:
            result = json.loads(content)
            distill_content = result.get("content", "")
            distill_keylabel = result.get("keylabel", "") or result.get("summary", "")
            distill_summary = result.get("summary", "") or distill_content
            analysis = result.get("analysis", {})
            if not distill_content:
                logger.warning("DISTILL: no content in response")
                return
            # 向后兼容：旧格式 summary 是长叙事，新格式是短标签
            if len(distill_keylabel) > 30:
                kfs = (analysis or {}).get("key_facts", [])
                kf_strings = [f for f in kfs if isinstance(f, str)]
                distill_keylabel = "·".join([f[:10] for f in kf_strings[:3]])[:20] if kf_strings else distill_content[:20]
            if not distill_keylabel:
                logger.warning("DISTILL: no keylabel in response")
                return
            logger.info(f"DISTILL: ok keylabel={distill_keylabel[:30]} content_len={len(distill_content)}")
            # 标记原始响应为已解析
            if raw_id:
                try:
                    self.conn.execute("UPDATE analysis_raw SET parsed = 1 WHERE id = ?", (raw_id,))
                    self.conn.commit()
                except Exception:
                    logger.exception(f"DISTILL: failed to mark raw response parsed id={raw_id} user={user_id}")
        except json.JSONDecodeError:
            logger.warning(f"DISTILL: invalid JSON: {content[:200]}")
            return

        # 写入蒸馏摘要 + 清理碎片（带 mood，让通道 2 情绪共振真正工作）
        mood = analysis.get("mood", "")
        mood_intensity = analysis.get("mood_intensity", 0.0)
        await self._apply_distill(user_id, session_id, last_turn, turn_count, distill_content, distill_keylabel, distill_summary, mood, mood_intensity)

        # 写入 9 维分析（情感/特质/实体等）
        await self._apply_analysis(user_id, session_id, distill_content[:80], "", analysis)

    def _build_distill_prompt(self, snapshot_text: str, long_term: dict, persona_name: str = "Bot", personality_style: str = "", is_group: bool = False, personality_type: str = "default") -> str:
        """两阶段蒸馏分析 prompt：私聊/群聊双模板，支持三种人格风格"""
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

        # ── 从 prompts.json 加载人格变体 ──
        dp = self._load_prompts().get("distill_prompt", {})
        ptype_key = personality_type if personality_type in ("cute", "professional") else "default"
        mode_key = "group" if is_group else "private"
        tpl = dp.get(ptype_key, {}).get(mode_key, {}) or dp.get("default", {}).get("private", {})
        preamble = tpl.get("preamble", "").format(persona_name=persona_name)
        content_instruction = tpl.get("content_instruction", "").format(persona_name=persona_name)
        first_stage = tpl.get("first_stage", "")
        extra_rules = tpl.get("extra_rules", "")

        # ── 从 prompts.json 加载输出格式模板 ──
        intro = dp.get("output_intro", "").replace("{first_stage}", first_stage)
        schema = dict(dp.get("output_schema", {}))
        rules = list(dp.get("output_rules", []))
        footer = dp.get("output_footer", "只输出 JSON。")

        # 群聊才保留 participants 字段
        if not is_group:
            analysis_schema = schema.get("analysis", {})
            if isinstance(analysis_schema, dict):
                analysis_schema.pop("participants", None)

        # schema -> 格式化的 JSON 字符串
        schema_str = json.dumps(schema, ensure_ascii=False, indent=2)

        # 替换占位符
        schema_str = schema_str.replace("{content_instruction}", content_instruction)
        schema_str = schema_str.replace("{persona_name}", persona_name)

        # 组装规则
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

已了解的信息（后续字段需与之去重）：
{lt_hint}

{intro}{schema_str}

规则：
{rules_str}
{footer}"""

    def check_distill_needed(self, session_id: str, persona_name: str = "Bot") -> tuple:
        """检查是否需要触发蒸馏。返回 (triggered, last_turn, turn_count, snapshot_text, sender_names)"""
        row = self.conn.execute(
            "SELECT turn_count, last_distill_turn, last_distill_at FROM session_state WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if not row:
            return (False, 0, 0, "", [])
        turn_count, last_turn, last_at = row[0] or 0, row[1] or 0, row[2]
        if turn_count == 0:
            return (False, 0, 0, "", [])
        max_turns = getattr(self, '_DISTILL_MAX_TURNS', 30)
        max_minutes = getattr(self, '_DISTILL_MAX_MINUTES', 60)
        triggered = False
        if turn_count - last_turn >= max_turns:
            triggered = True
        if not triggered and last_at:
            elapsed = (datetime.now() - datetime.fromisoformat(str(last_at))).total_seconds() / 60
            if elapsed >= max_minutes:
                triggered = True
        if not triggered:
            return (False, last_turn, turn_count, "", [])
        rows = self.conn.execute(
            "SELECT role, content, sender_name FROM chat_history WHERE session_id = ? AND turn_num > ? AND turn_num <= ? ORDER BY turn_num, id",
            (session_id, last_turn, turn_count),
        ).fetchall()
        if len(rows) < 6:
            return (False, last_turn, turn_count, "", [])
        senders = set()
        lines = []
        for role, content, sender_name in rows:
            nick = sender_name or (role if role == 'assistant' else '用户')
            if role == 'assistant':
                nick = persona_name
            if nick:
                senders.add(nick)
            lines.append(f"[{nick}] {content[:200]}")
        snapshot_text = "\n".join(lines[:30])
        return (True, last_turn, turn_count, snapshot_text, list(senders))

    async def _apply_distill(self, user_id: str, session_id: str, last_turn: int, turn_count: int, content: str, keylabel: str = "", summary: str = "", mood: str = "", mood_intensity: float = 0.0):
        """写入 LLM 蒸馏摘要（带 mood，供通道 2 情绪共振）+ 过期清理 + 图谱维护"""
        now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        if not keylabel:
            keylabel = content[:20]
        if not summary:
            summary = content
        lock = getattr(self, '_db_lock', None)
        if lock:
            lock.acquire()
        try:
            cur = self.conn.execute(
                "INSERT INTO cognitive_distill (user_id, session_id, content, keylabel, summary, importance, mood, mood_intensity, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (user_id, session_id, content, keylabel, summary, 0.8, mood, mood_intensity, now_str),
            )
            distill_id = cur.lastrowid
            self.conn.execute(
                "UPDATE session_state SET last_distill_turn = ?, last_distill_at = ? WHERE session_id = ?",
                (turn_count, now_str, session_id),
            )
            self._archive_dangling(user_id, session_id, now_str, reason="蒸馏")
            expired = self.conn.execute(
                "DELETE FROM cognitive_distill WHERE user_id = ? AND expires_at IS NOT NULL AND expires_at <= ?",
                (user_id, now_str),
            ).rowcount
            if expired:
                logger.info(f"RCMS: 已清理 {expired} 条过期记忆 user={user_id}")
            KEEP_RULE_SUMMARY = 10
            self.conn.execute("""
                DELETE FROM cognitive_distill WHERE user_id = ? AND importance = 0.3
                AND id NOT IN (
                    SELECT id FROM cognitive_distill
                    WHERE user_id = ? AND importance = 0.3
                    ORDER BY created_at DESC LIMIT ?
                )
            """, (user_id, user_id, KEEP_RULE_SUMMARY))
            self._maintain_graph(user_id)
            self.conn.commit()
        finally:
            if lock:
                try:
                    lock.release()
                except Exception:
                    pass
        vec = await self._get_embedding(keylabel[:512])
        if vec:
            self._store_embedding(user_id, distill_id, vec)
        logger.info(f"RCMS: 蒸馏完成 user={user_id} session={session_id} turn={turn_count} keylabel={keylabel[:60]}")

    def _archive_dangling(self, user_id: str, session_id: str, now_str: str, reason: str = ""):
        """将未结悬案归档到 cognitive_distill 并清空 session_state"""
        row = self.conn.execute(
            "SELECT dangling_threads FROM session_state WHERE session_id = ?", (session_id,)
        ).fetchone()
        if not row or not row[0]:
            return
        try:
            data = json.loads(row[0])
            if not isinstance(data, dict) or not data.get("threads"):
                return
            threads = data["threads"]
            tag = f"[悬案归档·{reason}]" if reason else "[悬案归档]"
            content = f"{tag} " + "、".join(threads[:3])
            self.conn.execute(
                "INSERT INTO cognitive_distill (user_id, session_id, content, keylabel, summary, importance, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (user_id, session_id, content, content.replace(tag, "").strip()[:20], content, 0.7, now_str),
            )
            self.conn.execute(
                "UPDATE session_state SET dangling_threads = ? WHERE session_id = ?",
                ('[]', session_id),
            )
            logger.info(f"RCMS: dangling_threads archived ({reason}) user={user_id}")
        except (json.JSONDecodeError, ValueError):
            logger.debug(f"RCMS: 归档时解析 dangling_threads JSON 失败 user={user_id}")


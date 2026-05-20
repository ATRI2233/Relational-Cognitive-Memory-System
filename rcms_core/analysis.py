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
                cur = self.conn.execute(
                    "INSERT INTO cognitive_distill (user_id, content, summary, importance, created_at) VALUES (?, ?, ?, ?, ?)",
                    (user_id, dt, dt, 0.5, now_str),
                )
                new_entries.append((cur.lastrowid, dt))
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
                    "INSERT INTO cognitive_distill (user_id, content, summary, importance, expires_at, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                    (user_id, content, content, imp_val, expires_at, now_str),
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
        if long_term.get('communication_style'):
            style_parts.append(long_term['communication_style'])
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
            distill_content = result.get("content", "") or result.get("summary", "")
            distill_summary = result.get("summary", "")
            analysis = result.get("analysis", {})
            if not distill_content:
                logger.warning("DISTILL: no content in response")
                return
            # 向后兼容：旧格式 summary 是长叙事，新格式是短标签
            if len(distill_summary) > 30:
                kfs = (analysis or {}).get("key_facts", [])
                kf_strings = [f for f in kfs if isinstance(f, str)]
                distill_summary = "·".join([f[:10] for f in kf_strings[:3]])[:20] if kf_strings else distill_content[:20]
            if not distill_summary:
                logger.warning("DISTILL: no summary in response")
                return
            logger.info(f"DISTILL: ok summary={distill_summary[:30]} content_len={len(distill_content)}")
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
        await self._apply_distill(user_id, session_id, last_turn, turn_count, distill_content, distill_summary, mood, mood_intensity)

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
            if long_term.get("communication_style"):
                lt_hint += f"\n沟通风格: {long_term['communication_style']}"
            if long_term.get("self_identity"):
                lt_hint += f"\n自我认同: {json.dumps(long_term['self_identity'], ensure_ascii=False)}"
            if long_term.get("core_identity"):
                lt_hint += f"\n核心身份: {json.dumps(long_term['core_identity'], ensure_ascii=False)}"
            if long_term.get("boundaries"):
                lt_hint += f"\n已知雷区: {json.dumps(long_term['boundaries'], ensure_ascii=False)}"

        # ── 人格风格分支 ──
        if personality_type == "cute":
            if is_group:
                preamble = f"你是 {persona_name}，一个活泼的群聊观察员~ 分析群聊对话快照，语气由消息内容决定。\n\n对话快照采用 [昵称] 内容 格式，[昵称] 代表消息发送者。"
                participants_field = '"participants": ["参与对话的所有昵称（不含 ' + persona_name + ' 自己）"],\n    '
                content_instruction = f"用 {persona_name} 的第一人称「我」来讲今天的故事，语气由消息内容决定，像记录生活片段一样。。"
                first_stage = (
                    "· 参与者有哪些（看看谁在群里说话啦）\n"
                    "· 发生了什么有趣的事、谁说了什么\n"
                    "· 气氛怎么样——开心/平淡/火药味\n"
                    "· 大家之间什么关系"
                )
            else:
                preamble = f"你是 {persona_name}，一个活泼可爱的小助手~ 分析私聊对话快照，语气由消息内容决定。\n\n对话快照采用 [昵称] 内容 格式，[昵称] 代表消息发送者。"
                participants_field = ""
                content_instruction = f"用 {persona_name} 的第一人称「我」来讲今天的故事，语气由消息内容决定，像记录和朋友的聊天。可以用「呀」「呢」「哦」「~」但别太过。"
                first_stage = (
                    "· 今天发生了什么\n"
                    "· 用户心情怎么样\n"
                    "· 你们聊得开不开心"
                )
            extra_rules = (
                "9. content 语气温暖轻松，朋友聊天感，适当用「呀」「呢」「哦」「~」但不要堆砌\n"
                "10. 保留情绪细节，让摘要读起来有温度\n"
                "11. 可以用「今天」「刚刚」开头讲故事"
            )

        elif personality_type == "professional":
            if is_group:
                preamble = f"你是 {persona_name} 的高级认知分析引擎。以专业分析师视角，客观结构化地分析群聊对话快照。\n\n对话快照采用 [昵称] 内容 格式，[昵称] 代表消息发送者。"
                participants_field = '"participants": ["参与对话的所有参与者昵称（不含 ' + persona_name + '）"],\n    '
                content_instruction = "以第三人称客观记录群聊中的关键事件、参与者行为模式和关系变化。保持精炼、结构化。"
                first_stage = (
                    "· 参与者识别\n"
                    "· 关键事件时序\n"
                    "· 情绪基调量化\n"
                    "· 参与者互动模式分析"
                )
            else:
                preamble = f"你是 {persona_name} 的高级认知分析引擎。以专业分析师视角，客观结构化地分析私聊对话快照。\n\n对话快照采用 [昵称] 内容 格式，[昵称] 代表消息发送者。"
                participants_field = ""
                content_instruction = "以第三人称客观记录本次对话的关键信息、用户状态变化和行为模式。保持专业、精炼。"
                first_stage = (
                    "· 事件时序重建\n"
                    "· 用户情绪状态变化\n"
                    "· 交互模式分析"
                )
            extra_rules = (
                "9. content 用第三人称，保持客观专业，不使用语气词\n"
                "10. 按「事件→影响→模式」结构组织摘要\n"
                "11. 避免主观评价，优先记录可验证的事实"
            )

        else:  # default
            if is_group:
                preamble = f"你是 {persona_name} 的后台认知分析模块。分析群聊对话快照，以 {persona_name} 的第一人称视角产出分析。\n\n对话快照采用 [昵称] 内容 格式，[昵称] 代表消息发送者。你需要理解谁说了什么。"
                participants_field = '"participants": ["参与对话的所有昵称（不含 ' + persona_name + ' 自己）"],\n    '
                content_instruction = f"以 {persona_name} 的第一人称「我」叙事，像日记一样记录观察到的事情。"
                first_stage = (
                    "· 参与者有哪些（通过 [昵称] 区分）\n"
                    "· 发生了什么事、谁说了什么、事件顺序\n"
                    "· 情绪基调\n"
                    "· 人物之间的关系和互动模式"
                )
            else:
                preamble = f"你是 {persona_name} 的后台认知分析模块。分析私聊对话快照，以 {persona_name} 的第一人称视角产出分析。\n\n对话快照采用 [昵称] 内容 格式，[昵称] 代表消息发送者。"
                participants_field = ""
                content_instruction = f"以 {persona_name} 的第一人称「我」叙事，像日记一样记录对用户的观察。"
                first_stage = (
                    "· 发生了什么、事件顺序\n"
                    "· 用户情绪变化\n"
                    "· 你和用户之间的互动节奏"
                )
            extra_rules = ""

        return f"""[SYSTEM]
{preamble}

角色风格：
{personality_style or '（无特殊设定）'}

[USER]
对话快照：
{snapshot_text}

已了解的信息（后续字段需与之去重）：
{lt_hint}

分析要求——两阶段：
第一阶段：理解对话脉络
{first_stage}

第二阶段：提取 JSON。返回格式：

{{
  "content": "{content_instruction} 优先捕捉新变化，不要重复已知信息。保留具体细节和情绪转折。没有意义的事情可以不记；但要记就必须记细节，总字数控制在 200 字以内。",
  "summary": "10-20 字的核心主题标签，必须包含 2-4 个具体实体/主题词，不能是 content 的缩写。如「游戏对比与考研拖延自省」。不要叙事连接词（今天、然后、接着）。",
  "analysis": {{
    {participants_field}"key_facts": [
      "按 importance 降序排列的关键事实，保留时间/原因/经过/结果等具体细节，最多5条"
    ],
    "key_facts_structured": [
      {{"content": "永久特质", "temporal": "permanent", "expires_after_days": null}},
      {{"content": "临时事件", "temporal": "transient", "expires_after_days": 14}}
    ],
    "mood": "用两个词描述情绪，如「轻松好奇」「焦虑疲惫」",
    "mood_intensity": 0.0~1.0,
    "topic_shift": true|false,
    "key_points": ["事件脉络简要概括"],
    "user_state": "open|reflective|guarded|playful|analytical|distant|intimate",
    "traits_updates": ["有具体行为证据支撑的参与者特质，每条≤15字，与已知特质去重后只输出新发现的"],
    "speech_quirks": ["说话特点，去重后只输出新发现的内容"],
    "preferences": {{"likes": ["事物"], "dislikes": ["事物"]}},
    "communication_style": "总结用户的说话方式（纯文本）",
    "self_identity": ["用户如何看待自己"],
    "core_identity": {{"职业": "", "角色": "", "标签": ""}},
    "boundaries": ["雷区列表"],
    "dangling_threads": ["未完成话题"],
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
1. content 用第一人称（default/cute）或第三人称（professional），不要用第二人称「你」
2. traits_updates 禁止提取「善于表达/喜欢思考/善于沟通」等无具体行为支撑的泛化特质；每条≤15字；必须与已知特质去重
3. speech_quirks 与 traits_updates 各自去重，与 lt_hint 对比后只输出新发现的
4. key_facts 最多5条，按 importance 降序，保留具体细节
5. 同一实体的不同表述用 canonical_name 统一消歧
6. 关系提取：优先客观关系；鼓励链式 A→B→C；只提取文本内实体
7. 无法提取的字段用 null 或空数组
8. 输入过长时设置 meta.truncated = true
9. summary 是 10-20 字的核心主题标签，必须包含具体实体/主题词，不能是 content 的缩写。如「游戏对比与考研拖延自省」
{extra_rules}
只输出 JSON。"""

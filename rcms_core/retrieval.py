import hashlib
import json
import logging
import math
import os
import re
from datetime import datetime

import jieba
import numpy as np

logger = logging.getLogger("rcms")


class RetrievalMixin:
    """三通道融合召回引擎"""

    _TIME_WORDS = {
        '今天': (0, 0), '今日': (0, 0),
        '昨天': (1, 1), '昨日': (1, 1),
        '前天': (2, 2),
        '最近': (0, 7), '近来': (0, 7), '近期': (0, 7),
        '上周': (7, 13), '上星期': (7, 13),
    }

    # ── 可调参数（硬编码默认，可被 config.json analysis.retrieval 覆盖） ──

    def _get_retrieval_params(self) -> dict:
        if not hasattr(self, '__retrieval_params'):
            rc = self.analysis_config.get("retrieval", {})
            self.__retrieval_params = {
                "total_cap": rc.get("total_cap", 5),
                "channel_min": rc.get("channel_min", [1, 1, 1]),
                "channel_weights": rc.get("channel_weights", [1.0, 1.0, 0.4]),
                "time_decay_halflife": rc.get("time_decay_halflife", 30),
                "emotional_resonance_bonus": rc.get("emotional_resonance_bonus", 0.15),
            }
        return self.__retrieval_params

    # ── 公共入口 ──

    async def retrieve_memories(self, user_id: str, user_input: str, stance: str, total_cap: int | None = None, session_id: str | None = None):
        """三通道融合，参数从 config.json analysis.retrieval 读取，可被调用方 total_cap 覆盖"""
        if stance == 'casual':
            return []

        p = self._get_retrieval_params()
        total_cap = total_cap or p["total_cap"]
        ch_min = p["channel_min"]
        ch_weights = p["channel_weights"]

        ch1 = self._channel_time_importance(user_id, session_id=session_id, limit=ch_min[0] + 1)
        ch2 = await self._channel_multi_resonance(user_id, user_input, limit=ch_min[1] + 2)
        ch3 = self._channel_graph_skeleton(user_id, user_input, limit=ch_min[2] + 1)

        result = self._fusion([ch1, ch2, ch3], total_cap, ch_min, ch_weights)
        # 确保至少一条完整叙事摘要不被 key_facts 挤掉
        NARRATIVE_MIN_LEN = 150
        if not any(len(c) > NARRATIVE_MIN_LEN for c, _ in result):
            row = self.conn.execute(
                "SELECT id, content FROM cognitive_distill "
                "WHERE user_id = ? AND importance >= 0.7 AND length(content) > ? "
                "ORDER BY created_at DESC LIMIT 1",
                (user_id, NARRATIVE_MIN_LEN),
            ).fetchone()
            if row:
                result[-1] = (row[1], 'recent')
        return result

    # ── 通道 1：时间重要性锚点 ──

    def _time_decay(self, days_ago: int) -> float:
        """指数衰减，半衰期从 config 读取（默认 30 天）"""
        p = self._get_retrieval_params()
        lam = math.log(2) / p["time_decay_halflife"]
        return math.exp(-lam * max(0, days_ago))

    def _channel_time_importance(self, user_id: str, session_id: str | None = None, limit: int = 2):
        """通道 1：时间衰减 × 恒定的 importance 加成，当前 session 条目额外推高"""
        rows = self.conn.execute("""
            SELECT content, created_at, importance, session_id
            FROM cognitive_distill
            WHERE user_id = ? AND importance > 0.1
              AND (expires_at IS NULL OR expires_at > datetime('now'))
            ORDER BY created_at DESC LIMIT 50
        """, (user_id,)).fetchall()

        now = datetime.now()
        scored = []
        for content, created_at, importance, row_sid in rows:
            days = 0
            if created_at:
                try:
                    days = (now - datetime.fromisoformat(str(created_at))).days
                except (ValueError, TypeError):
                    days = 999
            t = self._time_decay(days)
            # 当前 session 条目加 session_boost，避免被旧高 importance 条目压过
            session_boost = 0.3 if session_id and row_sid == session_id else 0.0
            score = (t + session_boost) * (0.5 + importance)
            scored.append((self._fuzz_time(created_at) + '，' + content, score, 'recent'))

        scored.sort(key=lambda x: -x[1])
        return scored[:limit]

    # ── 通道 2：多维共振 ──

    def _parse_time_filter(self, user_input: str):
        """解析时间词 → (min_days_ago, max_days_ago) 或 None"""
        for word, (min_d, max_d) in self._TIME_WORDS.items():
            if word in user_input:
                return (min_d, max_d)
        return None

    def _get_current_mood(self, user_id: str) -> str:
        row = self.conn.execute(
            "SELECT mood FROM cognitive_distill WHERE user_id = ? AND mood != '' ORDER BY created_at DESC LIMIT 1",
            (user_id,),
        ).fetchone()
        return row[0] if row else ''

    async def _channel_multi_resonance(self, user_id: str, user_input: str, limit: int = 3):
        """通道 2：时间词过滤 → 图扩散扩词 → 向量余弦相似度 → 情绪共振 → 重要性兜底"""
        p = self._get_retrieval_params()
        resonance_bonus = p["emotional_resonance_bonus"]
        # ── 1. 时间范围硬过滤 ──
        time_range = self._parse_time_filter(user_input)

        # ── 2. 图扩散扩充关键词 ──
        kws = self._extract_keywords(user_input)[:4]
        diffused = self._graph_activation_diffusion(user_id, kws)
        all_keywords = list(set(kws + [label for label, _ in diffused[:3]]))

        # ── 3. 向量检索：用原查询 + 扩散词构造 query，算余弦相似度 ──
        emb_query = user_input
        if all_keywords:
            emb_query = user_input + " " + " ".join(all_keywords)

        vec_results = {}   # content → cosine_sim
        try:
            if user_id not in self._emb_cache:
                self._load_emb_cache(user_id)
            cache = self._emb_cache[user_id]
            if cache["vectors"].shape[0] > 0:
                q_vec = await self._get_embedding(emb_query[:512])
                if q_vec and len(q_vec) == cache["vectors"].shape[1]:
                    q = np.array(q_vec, dtype=np.float32)
                    norms = cache["vectors"] / (np.linalg.norm(cache["vectors"], axis=1, keepdims=True) + 1e-12)
                    nq = q / (np.linalg.norm(q) + 1e-12)
                    scores = norms @ nq
                    for idx in np.argsort(-scores):
                        if scores[idx] > 0.3:
                            rid, content = cache["meta"][idx]
                            vec_results[content[:80]] = float(scores[idx])
                    logger.info(f"Resonance: user={user_id} vec_candidates={len(vec_results)}")
        except Exception as e:
            logger.warning(f"Resonance: vec search failed ({e}), fallback to kw")

        # ── 4. 关键词 SQL 候选（用于补充 vec_results +
        #        vec 无结果时的兜底） ──
        clauses = ["user_id = ?"]
        params = [user_id]

        if time_range:
            min_d, max_d = time_range
            if max_d is not None:
                clauses.append("CAST(julianday('now') - julianday(created_at) AS INTEGER) <= ?")
                params.append(max_d)
            clauses.append("CAST(julianday('now') - julianday(created_at) AS INTEGER) >= ?")
            params.append(min_d)

        kw_clauses = []
        for kw in all_keywords:
            kw_clauses.append("content LIKE ?")
            params.append(f'%{kw}%')
        if kw_clauses:
            clauses.append(f"({' OR '.join(kw_clauses)})")
        clauses.append("(expires_at IS NULL OR expires_at > datetime('now'))")

        kw_rows = self.conn.execute(
            f"SELECT id, content, created_at, importance, mood FROM cognitive_distill WHERE {' AND '.join(clauses)} ORDER BY created_at DESC LIMIT 30",
            params,
        ).fetchall()

        # 无向量 + 无关键词结果 → 重要性兜底
        if not vec_results and not kw_rows:
            kw_rows = self.conn.execute(
                "SELECT id, content, created_at, importance, mood FROM cognitive_distill WHERE user_id = ? AND importance >= 0.5 AND (expires_at IS NULL OR expires_at > datetime('now')) ORDER BY created_at DESC LIMIT 5",
                (user_id,),
            ).fetchall()

        # ── 5. 评分融合: vec 余弦 × 0.6 + 时间衰减重要性 × 0.4，情绪共振加成 ──
        current_mood = self._get_current_mood(user_id)
        now = datetime.now()
        scored = []

        # vec_results 中有命中 → 用余弦为主分
        if vec_results:
            for rid, content, created_at, importance, mood in kw_rows:
                key = content[:80]
                cos_sim = vec_results.get(key, 0.0)
                days = 0
                if created_at:
                    try:
                        days = (now - datetime.fromisoformat(str(created_at))).days
                    except (ValueError, TypeError):
                        days = 999
                imp_decay = importance * self._time_decay(days)

                if cos_sim > 0:
                    score = cos_sim * 0.6 + imp_decay * 0.4
                else:
                    # vec 无此条但 kw 匹配到 → 靠关键词保底
                    score = imp_decay * 0.5

                if current_mood and mood and mood == current_mood:
                    score *= (1 + resonance_bonus)

                scored.append((self._fuzz_time(created_at) + '，' + content, score, 'resonance'))

            # vec 中还有 kw_rows 未覆盖的条目
            for content, cos_sim in vec_results.items():
                if not any(content == c[:80] for _, c, _, _, _ in kw_rows):
                    score = cos_sim * 0.6
                    scored.append((content, score, 'resonance'))
        else:
            # 无向量 → 纯重要性 + 时间 + 情绪
            for rid, content, created_at, importance, mood in kw_rows:
                days = 0
                if created_at:
                    try:
                        days = (now - datetime.fromisoformat(str(created_at))).days
                    except (ValueError, TypeError):
                        days = 999
                score = importance * self._time_decay(days)
                if current_mood and mood and mood == current_mood:
                    score *= (1 + resonance_bonus)
                scored.append((self._fuzz_time(created_at) + '，' + content, score, 'resonance'))

        scored.sort(key=lambda x: -x[1])
        return scored[:limit]

    # ── 通道 3：图谱骨架事实 ──

    def _channel_graph_skeleton(self, user_id: str, user_input: str, limit: int = 2):
        """通道 3：纯图边查询（relation 语义优先），输出「A」--[关系]--> 「B」"""
        kws = self._extract_keywords(user_input)[:4]
        logger.info(f"GraphSkeleton: keywords={kws} user={user_id}")
        if not kws:
            return []

        kw_clauses = []
        params = [user_id]
        for kw in kws:
            kw_clauses.append("(label LIKE ? OR ? LIKE '%' || label || '%')")
            params.append(f'%{kw}%')
            params.append(kw)
        seed = self.conn.execute(
            f"SELECT node_id FROM memory_graph_nodes WHERE user_id = ? AND ({' OR '.join(kw_clauses)})",
            params,
        ).fetchall()
        seed_ids = {r[0] for r in seed}
        logger.info(f"GraphSkeleton: seed_nodes={len(seed_ids)} user={user_id}")
        if not seed_ids:
            return []

        ph2 = ','.join('?' * len(seed_ids))
        # 优先取带 relation 的边（LLM 分析出的逻辑关系），再取高权重共现边
        edges = self.conn.execute(f"""
            SELECT n1.label AS a, n2.label AS b, e.weight, e.relation
            FROM memory_graph_edges e
            JOIN memory_graph_nodes n1 ON e.from_node_id = n1.node_id
            JOIN memory_graph_nodes n2 ON e.to_node_id = n2.node_id
            WHERE (e.from_node_id IN ({ph2}) OR e.to_node_id IN ({ph2}))
              AND n1.user_id = ? AND n2.user_id = ?
            ORDER BY CASE WHEN e.relation != '' THEN 0 ELSE 1 END, e.weight DESC
            LIMIT ?
        """, (*seed_ids, *seed_ids, user_id, user_id, limit * 4)).fetchall()

        seen = set()
        results = []
        for a, b, weight, relation in edges:
            pair = (min(a, b), max(a, b))
            if pair in seen:
                continue
            seen.add(pair)
            if relation:
                stmt = f"「{a}」--[{relation}]--> 「{b}」"
            else:
                stmt = f"话题「{a}」与「{b}」常被一起提及（相关度 {weight:.1f}）"
            results.append((stmt, weight if not relation else weight + 2.0, 'skeleton'))

        logger.info(f"GraphSkeleton: returned={len(results[:limit])} user={user_id}")
        return results[:limit]

    # ── 融合 ──

    def _fusion(self, channels: list[list], total_cap: int = 5, ch_min: list | None = None, ch_weights: list | None = None):
        """三通道融合：每通道保底 ch_min[i] 条 → 内容 hash 去重 → 加权排序 → 截断"""
        ch_min = ch_min or [1, 1, 1]
        ch_weights = ch_weights or [1.0, 1.0, 0.4]
        tag_to_idx = {'recent': 0, 'resonance': 1, 'skeleton': 2}
        seen = set()
        merged = []

        def _hash_key(content: str) -> str:
            return hashlib.md5(content.strip().encode('utf-8')).hexdigest()

        def _weighted(item: tuple) -> float:
            content, score, tag = item
            return score * ch_weights[tag_to_idx.get(tag, 0)]

        # Phase 1：每通道保底（通道内原序，不受权重影响）
        for i, ch in enumerate(channels):
            taken = 0
            for item in ch:
                if taken >= ch_min[i]:
                    break
                key = _hash_key(item[0])
                if key not in seen:
                    seen.add(key)
                    merged.append(item)
                    taken += 1

        # Phase 2：填剩余名额（按加权分排序）
        all_items = []
        for ch in channels:
            all_items.extend(ch)
        all_items.sort(key=_weighted, reverse=True)

        for item in all_items:
            if len(merged) >= total_cap:
                break
            key = _hash_key(item[0])
            if key not in seen:
                seen.add(key)
                merged.append(item)

        merged.sort(key=_weighted, reverse=True)
        return [(content, tag) for content, score, tag in merged[:total_cap]]

    # ── 图操作 helper（消除 duplication） ──

    def _upsert_graph_node(self, user_id: str, label: str, now_str: str, entity_type: str = 'auto') -> int:
        row = self.conn.execute(
            "SELECT node_id, entity_type FROM memory_graph_nodes WHERE user_id = ? AND label = ?",
            (user_id, label),
        ).fetchone()
        if row:
            old_type = row[1] or 'auto'
            new_type = entity_type if entity_type != 'auto' else old_type
            self.conn.execute(
                "UPDATE memory_graph_nodes SET freq = freq + 1, last_seen = ?, entity_type = ? WHERE node_id = ?",
                (now_str, new_type, row[0]),
            )
            return row[0]
        cur = self.conn.execute(
            "INSERT INTO memory_graph_nodes (user_id, label, entity_type, freq, last_seen) VALUES (?, ?, ?, 1, ?)",
            (user_id, label, entity_type, now_str),
        )
        return cur.lastrowid

    def _upsert_graph_edge(self, from_id: int, to_id: int, now_str: str, relation: str = "", created_at: str = ""):
        if from_id == to_id:
            return
        existing = self.conn.execute(
            "SELECT weight FROM memory_graph_edges WHERE from_node_id = ? AND to_node_id = ?",
            (from_id, to_id),
        ).fetchone()
        if existing:
            self.conn.execute(
                """UPDATE memory_graph_edges SET weight = weight + 0.5,
                   encounter_count = encounter_count + 1, last_seen = ?,
                   relation = CASE WHEN ? != '' THEN ? ELSE relation END
                   WHERE from_node_id = ? AND to_node_id = ?""",
                (now_str, relation, relation, from_id, to_id),
            )
        else:
            self.conn.execute(
                "INSERT INTO memory_graph_edges (from_node_id, to_node_id, weight, encounter_count, last_seen, relation, created_at) VALUES (?, ?, 1.0, 1, ?, ?, ?)",
                (from_id, to_id, now_str, relation, created_at or now_str),
            )

    # ── 辅助：原始工具 ──

    def _fuzz_time(self, dt_str: str) -> str:
        if not dt_str:
            return ''
        try:
            dt = datetime.fromisoformat(str(dt_str)) if isinstance(dt_str, str) else datetime.strptime(str(dt_str)[:19], '%Y-%m-%d %H:%M:%S')
        except (ValueError, TypeError):
            return ''
        days = (datetime.now() - dt).days
        if days <= 2:
            return "前两天"
        if days <= 14:
            return "不久前"
        if days <= 60:
            return "前段时间"
        return "很久以前"

    def _extract_keywords(self, text: str, max_kw: int = 5) -> list[str]:
        # 先用空格/标点分割（英文关键词）
        tokens = re.split(r'[\s,，。！？、；：""''（）()—\n]+', text)
        # 对每个 segment 做 jieba 分词，产出中文词汇
        result = []
        for t in tokens:
            if not t:
                continue
            if re.search(r'[一-鿿]', t):
                # 含中文 → jieba 分词
                segs = jieba.lcut(t)
                for s in segs:
                    s = s.strip()
                    if len(s) > 1 and s not in self._TRIVIAL_MARKERS and s not in self._STOP_WORDS:
                        result.append(s)
            else:
                # 纯英文/数字 → 直接保留
                if len(t) > 1 and t not in self._TRIVIAL_MARKERS and t not in self._STOP_WORDS:
                    result.append(t)
        return result[:max_kw]

    def _graph_activation_diffusion(self, user_id: str, seed_keywords: list[str]) -> list:
        logger.info(f"GraphDiffusion: keywords={seed_keywords} user={user_id}")
        if not seed_keywords:
            return []
        now_dt = datetime.now()
        kw_clauses = []
        params = [user_id]
        for kw in seed_keywords:
            kw_clauses.append("(label LIKE ? OR ? LIKE '%' || label || '%')")
            params.append(f'%{kw}%')
            params.append(kw)
        seed_nodes = self.conn.execute(
            f"SELECT node_id, label, freq FROM memory_graph_nodes WHERE user_id = ? AND ({' OR '.join(kw_clauses)})",
            params,
        ).fetchall()
        logger.info(f"GraphDiffusion: seed_nodes={len(seed_nodes)} user={user_id}")
        if not seed_nodes:
            return []
        visited = set()
        activation_map = {}
        for nid, label, freq in seed_nodes:
            activation_map[nid] = 1.0
            visited.add(nid)
        queue = [(nid, 0) for nid, _, _ in seed_nodes]
        while queue:
            cid, depth = queue.pop(0)
            if depth >= self._GRAPH_BFS_DEPTH:
                continue
            edges = self.conn.execute(
                "SELECT from_node_id, to_node_id, weight, relation FROM memory_graph_edges WHERE from_node_id = ? OR to_node_id = ?",
                (cid, cid)
            ).fetchall()
            for frm, to, w, relation in edges:
                # 无 relation 的共现边是噪音，扩散时严重降权
                w = w * (1.0 if relation else 0.1)
                nid = frm if frm != cid else to
                if nid in visited:
                    activation_map[nid] += w * (self._GRAPH_ACTIVATION_DECAY ** (depth + 1))
                    continue
                visited.add(nid)
                activation_map[nid] = w * (self._GRAPH_ACTIVATION_DECAY ** (depth + 1))
                queue.append((nid, depth + 1))
        sorted_nodes = sorted(activation_map.items(), key=lambda x: -x[1])
        results = []
        for nid, score in sorted_nodes[:4]:
            row = self.conn.execute("SELECT label FROM memory_graph_nodes WHERE node_id = ?", (nid,)).fetchone()
            if row:
                results.append((row[0], score))
        logger.info(f"GraphDiffusion: returned={len(results)} user={user_id}")
        return results

    def _graph_recall(self, user_id: str, user_input: str, engagement_level: str) -> dict:
        kws = self._extract_keywords(user_input)[:4]
        if not kws:
            return {}
        diffused = self._graph_activation_diffusion(user_id, kws)
        surfaced = []
        silent = []
        for label, score in diffused:
            if score >= self._SURFACED_THRESHOLD:
                surfaced.append(label)
            elif score >= self._SILENT_THRESHOLD:
                silent.append(label)
        return {"surfaced": surfaced, "silent": silent, "activated": diffused}

    # ── Embedding 相关 ──

    def _get_retrieval_config(self) -> dict:
        rc = self.analysis_config.get("retrieval", {})
        return {
            "enabled": rc.get("embedding_enabled", rc.get("enabled", False)),
            "source": rc.get("source", "astrbot"),
            "api_key": rc.get("custom_api_key", "") or rc.get("api_key", os.environ.get("OPENAI_API_KEY", "")) or rc.get("custom_token", ""),
            "base_url": rc.get("custom_base_url", "") or rc.get("base_url", os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")) or rc.get("custom_url", ""),
            "model": rc.get("custom_model", "") or rc.get("model", "text-embedding-3-small"),
            "astrbot_source_id": rc.get("astrbot_source_id", ""),
        }

    async def _ensure_emb_client(self):
        cfg = self._get_retrieval_config()
        if self._emb_client is None and cfg["api_key"]:
            self._emb_client = AsyncOpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"])

    async def _get_embedding(self, text: str) -> list[float]:
        if self._embed_call:
            try:
                vec = await self._embed_call(text)
                if vec:
                    logger.debug(f"Embedding: via callback dim={len(vec)}")
                    return vec
            except Exception as e:
                logger.warning(f"Embedding: callback failed ({e})")
        await self._ensure_emb_client()
        if not self._emb_client:
            logger.warning("Embedding: no client (api_key not configured)")
            return []
        try:
            resp = await self._emb_client.embeddings.create(
                model=self._emb_model,
                input=text.replace("\n", " "),
            )
            vec = resp.data[0].embedding
            logger.debug(f"Embedding: ok dim={len(vec)} model={self._emb_model}")
            return vec
        except Exception as e:
            logger.warning(f"Embedding: API call failed ({e})")
            return []

    def _store_embedding(self, user_id: str, record_id: int, embedding: list[float]):
        blob = np.array(embedding, dtype=np.float32).tobytes()
        self.conn.execute(
            "UPDATE cognitive_distill SET embedding = ? WHERE id = ? AND user_id = ?",
            (blob, record_id, user_id),
        )
        self.conn.commit()

    def _load_emb_cache(self, user_id: str):
        rows = self.conn.execute(
            "SELECT id, content, embedding FROM cognitive_distill WHERE user_id = ? AND embedding IS NOT NULL ORDER BY id",
            (user_id,),
        ).fetchall()
        if not rows:
            self._emb_cache[user_id] = {"vectors": np.empty((0, 0), dtype=np.float32), "meta": []}
            logger.debug(f"EmbedCache: user={user_id} no vectors")
            return
        vecs = []
        meta = []
        expected_dim = None
        for row_id, content, blob in rows:
            vec = np.frombuffer(blob, dtype=np.float32)
            if expected_dim is None:
                expected_dim = len(vec)
            if len(vec) == expected_dim:
                vecs.append(vec)
                meta.append((row_id, content))
            else:
                logger.warning(f"EmbedCache: skip vector dim={len(vec)} (expected {expected_dim}) id={row_id}")
        self._emb_cache[user_id] = {
            "vectors": np.array(vecs, dtype=np.float32) if vecs else np.empty((0, expected_dim or 0), dtype=np.float32),
            "meta": meta,
        }
        logger.info(f"EmbedCache: user={user_id} loaded {len(vecs)} vectors (dim={expected_dim})")

    async def retrieve_by_embedding(self, user_id: str, query: str, limit: int = 3):
        if user_id not in self._emb_cache:
            self._load_emb_cache(user_id)
        cache = self._emb_cache[user_id]
        if cache["vectors"].shape[0] == 0:
            logger.debug(f"Retrieve: user={user_id} no vectors yet")
            return [], "no_vectors"
        q_vec = await self._get_embedding(query)
        if not q_vec or len(q_vec) != cache["vectors"].shape[1]:
            logger.debug(f"Retrieve: user={user_id} dim mismatch or empty")
            return [], "no_vectors"
        q = np.array(q_vec, dtype=np.float32)
        norm_cache = cache["vectors"] / (np.linalg.norm(cache["vectors"], axis=1, keepdims=True) + 1e-12)
        norm_q = q / (np.linalg.norm(q) + 1e-12)
        scores = norm_cache @ norm_q
        top_idx = np.argsort(-scores)[:limit]
        results = []
        for idx in top_idx:
            if scores[idx] > 0.3:
                rec_id, content = cache["meta"][idx]
                results.append((content, float(scores[idx])))
        logger.info(f"Retrieve: user={user_id} query='{query[:30]}' found={len(results)}/{cache['vectors'].shape[0]} source=embedding")
        return results, "embedding"

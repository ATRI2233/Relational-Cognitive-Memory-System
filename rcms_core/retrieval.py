import json
import logging
import math
import os
import re
from datetime import datetime

import numpy as np

logger = logging.getLogger("rcms")


class RetrievalMixin:
    """三通道融合召回引擎"""

    _TIME_DECAY_HALFLIFE = 30     # 重要性半衰期（天）
    _EMOTIONAL_RESONANCE_BONUS = 0.15

    _TIME_WORDS = {
        '今天': (0, 0), '今日': (0, 0),
        '昨天': (1, 1), '昨日': (1, 1),
        '前天': (2, 2),
        '最近': (0, 7), '近来': (0, 7), '近期': (0, 7),
        '上周': (7, 13), '上星期': (7, 13),
    }

    # ── 公共入口 ──

    async def retrieve_memories(self, user_id: str, user_input: str, stance: str, total_cap: int = 5):
        """三通道融合，每通道保底 1 条，总数不超过 total_cap"""
        if stance == 'casual':
            return []

        ch1 = self._channel_time_importance(user_id, limit=2)
        ch2 = await self._channel_multi_resonance(user_id, user_input, limit=3)
        ch3 = self._channel_graph_skeleton(user_id, user_input, limit=2)

        return self._fusion([ch1, ch2, ch3], total_cap)

    # ── 通道 1：时间重要性锚点 ──

    def _time_decay(self, days_ago: int) -> float:
        """指数衰减，半衰期 _TIME_DECAY_HALFLIFE 天"""
        lam = math.log(2) / self._TIME_DECAY_HALFLIFE
        return math.exp(-lam * max(0, days_ago))

    def _channel_time_importance(self, user_id: str, limit: int = 2):
        """通道 1：importance × 时间衰减，不查向量"""
        rows = self.conn.execute("""
            SELECT content, created_at, importance
            FROM cognitive_distill
            WHERE user_id = ? AND importance > 0.1 AND content NOT LIKE '[蒸馏]%'
            ORDER BY created_at DESC LIMIT 50
        """, (user_id,)).fetchall()

        now = datetime.now()
        scored = []
        for content, created_at, importance in rows:
            days = 0
            if created_at:
                try:
                    days = (now - datetime.fromisoformat(str(created_at))).days
                except (ValueError, TypeError):
                    days = 999
            score = importance * self._time_decay(days)
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
            "SELECT prose_hint FROM emotional_trace WHERE user_id = ? ORDER BY created_at DESC LIMIT 1",
            (user_id,),
        ).fetchone()
        return row[0] if row else ''

    async def _channel_multi_resonance(self, user_id: str, user_input: str, limit: int = 3):
        """通道 2：时间词过滤 → 图扩散扩词 → 向量余弦相似度 → 情绪共振 → 重要性兜底"""
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
        clauses = ["user_id = ?", "content NOT LIKE '[蒸馏]%'"]
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

        kw_rows = self.conn.execute(
            f"SELECT id, content, created_at, importance, mood FROM cognitive_distill WHERE {' AND '.join(clauses)} ORDER BY created_at DESC LIMIT 30",
            params,
        ).fetchall()

        # 无向量 + 无关键词结果 → 重要性兜底
        if not vec_results and not kw_rows:
            kw_rows = self.conn.execute(
                "SELECT id, content, created_at, importance, mood FROM cognitive_distill WHERE user_id = ? AND importance >= 0.5 AND content NOT LIKE '[蒸馏]%' ORDER BY created_at DESC LIMIT 5",
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
                    score *= (1 + self._EMOTIONAL_RESONANCE_BONUS)

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
                    score *= (1 + self._EMOTIONAL_RESONANCE_BONUS)
                scored.append((self._fuzz_time(created_at) + '，' + content, score, 'resonance'))

        scored.sort(key=lambda x: -x[1])
        return scored[:limit]

    # ── 通道 3：图谱骨架事实 ──

    def _channel_graph_skeleton(self, user_id: str, user_input: str, limit: int = 2):
        """通道 3：纯图边查询（relation 语义优先），输出「A」--[关系]--> 「B」"""
        kws = self._extract_keywords(user_input)[:4]
        if not kws:
            return []

        ph = ','.join('?' * len(kws))
        seed = self.conn.execute(
            f"SELECT node_id FROM memory_graph_nodes WHERE user_id = ? AND label IN ({ph})",
            (user_id, *kws),
        ).fetchall()
        seed_ids = {r[0] for r in seed}
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
                stmt = f"[图谱] 「{a}」--[{relation}]--> 「{b}」"
            else:
                stmt = f"[图谱] 话题「{a}」与「{b}」常被一起提及（相关度 {weight:.1f}）"
            results.append((stmt, weight if not relation else weight + 2.0, 'skeleton'))

        return results[:limit]

    # ── 融合 ──

    def _fusion(self, channels: list[list], total_cap: int = 5):
        """三通道融合：每通道保底 1 条 → 去重 → 排序 → 截断"""
        seen = set()
        merged = []

        # Phase 1：每通道保底 1 条
        for ch in channels:
            for item in ch:
                key = item[0][:25]
                if key not in seen:
                    seen.add(key)
                    merged.append(item)
                    break

        # Phase 2：填剩余名额
        for ch in channels:
            for item in ch:
                if len(merged) >= total_cap:
                    break
                key = item[0][:25]
                if key not in seen:
                    seen.add(key)
                    merged.append(item)
            if len(merged) >= total_cap:
                break

        merged.sort(key=lambda x: -x[1])
        return [(content, tag) for content, score, tag in merged[:total_cap]]

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
        tokens = re.split(r'[\s,，。！？、；：""''（）()—\n]+', text)
        return [w for w in tokens if len(w) > 1 and w not in self._TRIVIAL_MARKERS][:max_kw]

    def _graph_activation_diffusion(self, user_id: str, seed_keywords: list[str]) -> list:
        if not seed_keywords:
            return []
        now_dt = datetime.now()
        placeholders = ','.join('?' * len(seed_keywords))
        seed_nodes = self.conn.execute(
            f"SELECT node_id, label, freq FROM memory_graph_nodes WHERE user_id = ? AND label IN ({placeholders})",
            (user_id, *seed_keywords)
        ).fetchall()
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
                "SELECT from_node_id, to_node_id, weight FROM memory_graph_edges WHERE from_node_id = ? OR to_node_id = ?",
                (cid, cid)
            ).fetchall()
            for frm, to, w in edges:
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
            "enabled": rc.get("enabled", False),
            "source": rc.get("source", "astrbot"),
            "api_key": rc.get("custom_api_key", "") or rc.get("api_key", os.environ.get("OPENAI_API_KEY", "")),
            "base_url": rc.get("custom_base_url", "") or rc.get("base_url", os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")),
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

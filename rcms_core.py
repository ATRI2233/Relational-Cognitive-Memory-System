"""MinimalRCMS — 关系认知记忆系统核心（单文件）"""
import asyncio
import json
import logging
import os
import re
import sqlite3
from datetime import datetime
from typing import Optional

import numpy as np
from openai import AsyncOpenAI

from backends import LLMBackend

logger = logging.getLogger("rcms")


class MinimalRCMS:

    # ── 常量 ──

    _EMOTIONAL_WORDS = [
        '累', '烦', '难过', '开心', '怕', '为什么', '怎么办',
        '焦虑', '迷茫', '失望', '生气', '感动', '孤独', '压力',
        '崩溃', '痛苦', '幸福', '委屈', '愤怒', '绝望', '不安',
        '愧疚', '后悔', '感激', '羡慕', '厌倦', '疲惫', '心累',
        '纠结', '无助', '温暖', '讽刺', '荒谬', '崩溃', '心碎',
        '气死', '受不了', '撑不住', '扛不住', '熬不下去',
        '舍不得', '放不下', '不甘心',
    ]

    _TRIVIAL_MARKERS = ['吃', '喝', '睡', '饭', '菜', '外卖', '快递', '天气',
                        '价格', '多少钱', '购物', '买了', '电影', '追剧',
                        '洗澡', '起床', '睡觉', '游戏']

    _ARC_STAGES = ['stranger', 'familiar', 'rapport', 'history', 'drift', 'reconnect']
    _RESIDUE_DECAY = 0.6
    _GRAPH_BFS_DEPTH = 2
    _GRAPH_ACTIVATION_DECAY = 0.5
    _SURFACED_THRESHOLD = 0.6
    _SILENT_THRESHOLD = 0.25

    # ── 初始化 ──

    def __init__(self, db_path="memory.db", analysis_config: dict | None = None,
                 llm_call=None, embed_call=None):
        """llm_call: async (prompt: str, model: str) -> str | None — LLM 回调
           embed_call: async (text: str) -> list[float] | None — Embedding 回调"""
        self.db_path = db_path
        self.analysis_config = analysis_config or {}
        self._llm_call = llm_call
        self._embed_call = embed_call
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_db()
        self._last_silent_recall = []
        # Embedding 缓存
        self._emb_cache: dict[str, dict] = {}
        self._emb_client: Optional[AsyncOpenAI] = None
        rc = self.analysis_config.get("retrieval", {})
        self._emb_model = rc.get("custom_model", "") or rc.get("model", "text-embedding-3-small")
        logger.info(f"RCMS init: db={db_path}, retrieval={self.analysis_config.get('retrieval', {}).get('enabled', False)}, post_analysis={self.analysis_config.get('post_analysis', {}).get('mode', 'rule')}")

    def _init_db(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS long_term_memory (
                id INTEGER PRIMARY KEY, user_id TEXT, content TEXT,
                memory_type TEXT CHECK(memory_type IN ('event','impression')),
                session_id TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS session_state (
                session_id TEXT PRIMARY KEY, user_id TEXT, stance TEXT DEFAULT 'open',
                mood REAL DEFAULT 0, focus_topic TEXT, turn_count INTEGER DEFAULT 0,
                stance_turns INTEGER DEFAULT 0, engagement_level TEXT DEFAULT 'coasting',
                momentum_depth REAL DEFAULT 0.0, momentum_energy REAL DEFAULT 0.0,
                last_active TIMESTAMP, residue_warmth REAL DEFAULT 0.0,
                residue_tension REAL DEFAULT 0.0, dangling_threads TEXT DEFAULT '[]',
                embedding_updated INTEGER DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY, session_id TEXT, role TEXT, content TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS identity_memory (
                user_id TEXT PRIMARY KEY, traits TEXT DEFAULT '[]',
                voice_hint TEXT DEFAULT '', created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS event_memory (
                event_id INTEGER PRIMARY KEY, user_id TEXT, content TEXT,
                relationship_delta REAL DEFAULT 0.0, emotional_weight REAL DEFAULT 0.0,
                novelty REAL DEFAULT 0.0, compressed_hint TEXT DEFAULT '',
                created_at TIMESTAMP, last_recalled TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS emotional_trace (
                trace_id INTEGER PRIMARY KEY, user_id TEXT, warmth REAL DEFAULT 0.0,
                tension REAL DEFAULT 0.0, uncertainty REAL DEFAULT 0.0,
                distance REAL DEFAULT 0.0, prose_hint TEXT DEFAULT '',
                created_at TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS shared_context (
                context_id INTEGER PRIMARY KEY, user_id TEXT, context_body TEXT,
                omission_count INTEGER DEFAULT 0, confirmed INTEGER DEFAULT 0,
                created_at TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS relationship_arc (
                arc_id INTEGER PRIMARY KEY, user_id TEXT,
                stage TEXT DEFAULT 'stranger', stage_score REAL DEFAULT 0.0,
                updated_at TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS memory_graph_nodes (
                node_id INTEGER PRIMARY KEY, user_id TEXT, label TEXT,
                node_type TEXT DEFAULT 'keyword', freq INTEGER DEFAULT 1,
                last_seen TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS memory_graph_edges (
                from_node_id INTEGER, to_node_id INTEGER, weight REAL DEFAULT 1.0,
                encounter_count INTEGER DEFAULT 1, last_seen TIMESTAMP,
                PRIMARY KEY (from_node_id, to_node_id)
            );
            CREATE INDEX IF NOT EXISTS idx_mgn_user_label ON memory_graph_nodes(user_id, label);
            CREATE INDEX IF NOT EXISTS idx_mge_from ON memory_graph_edges(from_node_id);
            CREATE INDEX IF NOT EXISTS idx_mge_to ON memory_graph_edges(to_node_id);
            CREATE TABLE IF NOT EXISTS memory_embeddings (
                id INTEGER PRIMARY KEY, user_id TEXT, memory_id INTEGER,
                content TEXT, embedding BLOB, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS entity_relations (
                id INTEGER PRIMARY KEY, user_id TEXT, entity_name TEXT,
                relation_type TEXT DEFAULT '', property TEXT DEFAULT '',
                mention_count INTEGER DEFAULT 1, last_mentioned TIMESTAMP,
                sentiment REAL DEFAULT 0.0,
                UNIQUE(user_id, entity_name)
            );
            CREATE INDEX IF NOT EXISTS idx_emb_user ON memory_embeddings(user_id);
            CREATE INDEX IF NOT EXISTS idx_er_user ON entity_relations(user_id, entity_name);
        """)
        for col in [
            "ADD COLUMN stance_turns INTEGER DEFAULT 0",
            "ADD COLUMN engagement_level TEXT DEFAULT 'coasting'",
            "ADD COLUMN momentum_depth REAL DEFAULT 0.0",
            "ADD COLUMN momentum_energy REAL DEFAULT 0.0",
            "ADD COLUMN last_active TIMESTAMP",
            "ADD COLUMN residue_warmth REAL DEFAULT 0.0",
            "ADD COLUMN residue_tension REAL DEFAULT 0.0",
            "ADD COLUMN dangling_threads TEXT DEFAULT ''",
        ]:
            try:
                self.conn.execute(f"ALTER TABLE session_state {col}")
            except Exception:
                pass
        self.conn.commit()

    def close(self):
        self.conn.close()

    # ── 通用文本工具 ──

    def _get_history(self, session_id: str, limit: int = 3):
        rows = self.conn.execute("""
            SELECT role, content FROM chat_history
            WHERE session_id = ? ORDER BY created_at DESC LIMIT ?
        """, (session_id, limit)).fetchall()
        rows.reverse()
        return rows

    @staticmethod
    def _chinese_bigrams(text: str) -> set:
        chars = re.findall(r'[一-鿿]', text)
        return {''.join(chars[i:i + 2]) for i in range(len(chars) - 1)}

    @staticmethod
    def _precise_kw_match(text: str, kw: str) -> bool:
        return kw in text

    @staticmethod
    def _score_markers(text: str, markers: list, per_hit: float = 0.3) -> float:
        count = sum(1 for m in markers if m in text)
        return min(count * per_hit, 1.0)

    # ── 记忆检索 ──

    def _fuzz_time(self, dt_str: str) -> str:
        dt = datetime.fromisoformat(dt_str) if isinstance(dt_str, str) else datetime.strptime(dt_str[:19], '%Y-%m-%d %H:%M:%S')
        days = (datetime.now() - dt).days
        if days <= 2:
            return "前两天"
        if days <= 14:
            return "不久前"
        if days <= 60:
            return "前段时间"
        return "很久以前"

    def retrieve_memories(self, user_id: str, user_input: str, stance: str, limit: int = 2):
        if stance == 'casual':
            logger.debug(f"Retrieve: user={user_id} stance=casual skip")
            return []
        tokens = re.split(r'[\s,，。！？、；：""''（）()—\n]+', user_input)
        keywords = [w for w in tokens if len(w) > 1][:3]
        if not keywords:
            logger.debug(f"Retrieve: user={user_id} no keywords from input")
            return []
        conditions = ' OR '.join(['content LIKE ?'] * len(keywords))
        params = [f'%{k}%' for k in keywords] + [user_id]
        cursor = self.conn.execute(f"""
            SELECT content, memory_type, created_at FROM long_term_memory
            WHERE ({conditions}) AND user_id = ?
            ORDER BY created_at DESC LIMIT ?
        """, params + [limit])
        rows = cursor.fetchall()
        logger.info(f"Retrieve: user={user_id} kws={keywords} found={len(rows)} source=keyword")
        return [(self._fuzz_time(r[2]) + '，' + r[0], r[1]) for r in rows]

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
            nid, depth = queue.pop(0)
            if depth >= self._GRAPH_BFS_DEPTH:
                continue
            edges = self.conn.execute(
                "SELECT from_node_id, to_node_id, weight FROM memory_graph_edges WHERE from_node_id = ? OR to_node_id = ?",
                (nid, nid)
            ).fetchall()
            for from_id, to_id, weight in edges:
                neighbor = to_id if from_id == nid else from_id
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                decay = self._GRAPH_ACTIVATION_DECAY ** (depth + 1)
                activation_map[neighbor] = weight * decay
                queue.append((neighbor, depth + 1))
        sorted_nodes = sorted(activation_map.items(), key=lambda x: -x[1])
        results = []
        seen_content = set()
        for nid, activation in sorted_nodes:
            if len(results) >= 4:
                break
            node_label = self.conn.execute(
                "SELECT label FROM memory_graph_nodes WHERE node_id = ?", (nid,)
            ).fetchone()
            if not node_label:
                continue
            kw = node_label[0]
            memories = self.conn.execute(
                "SELECT content, created_at FROM long_term_memory WHERE user_id = ? AND content LIKE ? ORDER BY created_at DESC LIMIT 2",
                (user_id, f'%{kw}%')
            ).fetchall()
            for content, created_at in memories:
                if content not in seen_content:
                    seen_content.add(content)
                    fuzz_time = self._fuzz_time(created_at)
                    results.append((fuzz_time + '，' + content, activation, created_at))
        results.sort(key=lambda x: -x[1])
        return results[:4]

    def _graph_recall(self, user_id: str, user_input: str, engagement_level: str) -> dict:
        if engagement_level == 'coasting':
            return {'surfaced': [], 'silent': [], 'status': 'skip'}
        seed_kws = self._extract_keywords(user_input, max_kw=4)
        if not seed_kws:
            return {'surfaced': [], 'silent': [], 'status': 'skip'}
        activated = self._graph_activation_diffusion(user_id, seed_kws)
        items = [(a[0], a[1]) for a in activated]
        surfaced = []
        silent = []
        for content, activation in items:
            if activation >= self._SURFACED_THRESHOLD:
                surfaced.append((content, activation))
            elif activation >= self._SILENT_THRESHOLD:
                silent.append((content, activation))
        return {
            'surfaced': surfaced[:2],
            'silent': silent[:3],
            'status': 'graph' if (surfaced or silent) else 'fallback',
        }

    async def _recall(self, user_id: str, user_input: str, engagement_level: str) -> tuple:
        try:
            graph_result = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None, self._graph_recall, user_id, user_input, engagement_level
                ),
                timeout=0.3
            )
        except asyncio.TimeoutError:
            graph_result = {'surfaced': [], 'silent': [], 'status': 'timeout'}
        if graph_result['surfaced']:
            return [(m[0], 'graph') for m in graph_result['surfaced']], 'graph'
        if graph_result['silent']:
            self._last_silent_recall = graph_result['silent']
        memories = self.retrieve_memories(user_id, user_input, 'engaged')
        if memories:
            return memories, 'keyword_fallback'
        return [], 'timeout'

    # ── Embedding 检索层 ──

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
        # 优先使用外部回调（AstrBot 模式）
        if self._embed_call:
            try:
                vec = await self._embed_call(text)
                if vec:
                    logger.debug(f"Embedding: via callback dim={len(vec)}")
                    return vec
            except Exception as e:
                logger.warning(f"Embedding: callback failed ({e})")
        # 回退：直接 OpenAI API
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

    def _store_embedding(self, user_id: str, memory_id: int, content: str, embedding: list[float]):
        blob = np.array(embedding, dtype=np.float32).tobytes()
        self.conn.execute(
            "INSERT INTO memory_embeddings (user_id, memory_id, content, embedding) VALUES (?, ?, ?, ?)",
            (user_id, memory_id, content, blob),
        )
        self.conn.commit()

    def _load_emb_cache(self, user_id: str):
        rows = self.conn.execute(
            "SELECT id, memory_id, content, embedding FROM memory_embeddings WHERE user_id = ? ORDER BY id",
            (user_id,),
        ).fetchall()
        if not rows:
            self._emb_cache[user_id] = {"vectors": np.empty((0, 0), dtype=np.float32), "meta": []}
            logger.debug(f"EmbedCache: user={user_id} no vectors")
            return
        vecs = []
        meta = []
        expected_dim = None
        for row_id, mem_id, content, blob in rows:
            vec = np.frombuffer(blob, dtype=np.float32)
            if expected_dim is None:
                expected_dim = len(vec)
            if len(vec) == expected_dim:
                vecs.append(vec)
                meta.append((mem_id, content))
            else:
                logger.warning(f"EmbedCache: skip vector dim={len(vec)} (expected {expected_dim}) id={row_id}")
        self._emb_cache[user_id] = {
            "vectors": np.array(vecs, dtype=np.float32) if vecs else np.empty((0, expected_dim or 0), dtype=np.float32),
            "meta": meta,
        }
        logger.info(f"EmbedCache: user={user_id} loaded {len(vecs)} vectors (dim={expected_dim})")

    async def retrieve_by_embedding(self, user_id: str, query: str, limit: int = 3):
        # 先检查缓存，没向量就不调 API
        if user_id not in self._emb_cache:
            self._load_emb_cache(user_id)
        cache = self._emb_cache[user_id]
        if cache["vectors"].shape[0] == 0:
            logger.debug(f"Retrieve: user={user_id} no vectors yet")
            return [], "no_vectors"
        # 缓存有向量，才 embedding query 做语义检索
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
                mem_id, content = cache["meta"][idx]
                results.append((content, float(scores[idx])))
        logger.info(f"Retrieve: user={user_id} query='{query[:30]}' found={len(results)}/{cache['vectors'].shape[0]} source=embedding")
        return results, "embedding"

    # ── Narrative Context（供 AstrBot 注入）──

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

        # ── 收束 ──
        parts.append("→ 以上是你通过长期对话积累的对他的了解，用来更好地理解他的意图。人格设定始终优先。")

        return "[RCMS 关系上下文]\n" + "\n\n".join(parts)

    def prompt_compressor(self, user_id: str, session_id: str, user_input: str,
                           memories: list | None = None,
                           long_term: dict | None = None) -> str:
        if memories is None:
            memories = self.retrieve_memories(user_id, user_input, 'engaged')
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
        prompt = "【当前心理状态】\n自然地聊"
        if mem_block:
            prompt += f"\n\n【相关记忆】\n{mem_block}"
        if lt_block:
            prompt += lt_block
        prompt += f"\n\n【底线】\n不主动说教。不假装完全理解。疲惫时简短但不冷漠。\n\n用户: {user_input}\n你:"
        return prompt

    # ── Silent Recall Residue ──

    def _load_residue(self, session_id: str) -> tuple:
        row = self.conn.execute(
            "SELECT residue_warmth, residue_tension FROM session_state WHERE session_id = ?",
            (session_id,)
        ).fetchone()
        return (row[0] or 0.0, row[1] or 0.0) if row else (0.0, 0.0)

    def _decay_residue(self, session_id: str):
        warmth, tension = self._load_residue(session_id)
        warmth *= self._RESIDUE_DECAY
        tension *= self._RESIDUE_DECAY
        if abs(warmth) < 0.01:
            warmth = 0.0
        if abs(tension) < 0.01:
            tension = 0.0
        self.conn.execute(
            "UPDATE session_state SET residue_warmth = ?, residue_tension = ? WHERE session_id = ?",
            (warmth, tension, session_id)
        )

    def _write_residue(self, session_id: str, warmth_delta: float, tension_delta: float):
        cw, ct = self._load_residue(session_id)
        self.conn.execute(
            "UPDATE session_state SET residue_warmth = ?, residue_tension = ? WHERE session_id = ?",
            (max(-1.0, min(1.0, cw + warmth_delta)), max(-1.0, min(1.0, ct + tension_delta)), session_id)
        )

    def _apply_residue(self, depth: float, energy: float, session_id: str) -> tuple:
        rw, rt = self._load_residue(session_id)
        if abs(rw) > 0.01:
            energy += rw * 0.15
        if abs(rt) > 0.01:
            depth += rt * 0.10
        return (max(0.0, min(1.0, depth)), max(-1.0, min(1.0, energy)))

    # ── 长期记忆 ──

    def _init_identity(self, user_id: str):
        now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.conn.execute("INSERT OR IGNORE INTO identity_memory (user_id, traits, voice_hint, updated_at) VALUES (?, '[]', '', ?)", (user_id, now_str))
        self.conn.execute("INSERT OR IGNORE INTO relationship_arc (user_id, stage, stage_score, updated_at) VALUES (?, 'stranger', 0.0, ?)", (user_id, now_str))
        self.conn.commit()

    def _write_event_memory(self, user_id: str, session_id: str, content: str,
                             relationship_delta: float, emotional_weight: float):
        now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        compressed = content[:40] + '...' if len(content) > 40 else content
        self.conn.execute(
            "INSERT INTO event_memory (user_id, content, relationship_delta, emotional_weight, novelty, compressed_hint, created_at) VALUES (?, ?, ?, ?, 0.0, ?, ?)",
            (user_id, content, relationship_delta, emotional_weight, compressed, now_str)
        )
        self.conn.commit()

    def _update_relationship_arc(self, user_id: str, level: str):
        row = self.conn.execute("SELECT stage, stage_score FROM relationship_arc WHERE user_id = ?", (user_id,)).fetchone()
        if not row:
            return
        stage, score = row
        new_score = score + (0.05 if level == 'engaged_candidate' else 0.02)
        new_stage = stage
        thresholds = {'stranger': 4.0, 'familiar': 10.0, 'rapport': 20.0, 'history': 35.0}
        if stage == 'stranger' and new_score >= thresholds['stranger']:
            new_stage = 'familiar'
        elif stage == 'familiar' and new_score >= thresholds['familiar']:
            new_stage = 'rapport'
        elif stage == 'rapport' and new_score >= thresholds['rapport']:
            new_stage = 'history'
        self.conn.execute("UPDATE relationship_arc SET stage = ?, stage_score = ?, updated_at = ? WHERE user_id = ?",
                          (new_stage, new_score, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), user_id))
        self.conn.commit()

    def _build_shared_context(self, user_id: str, user_input: str, reply: str):
        tokens = re.split(r'[\s,，。！？、；：""''（）()—\n]+', user_input)
        kws = [w for w in tokens if len(w) > 2 and w not in self._TRIVIAL_MARKERS]
        if not kws:
            return
        kw = kws[0]
        existing = self.conn.execute("SELECT context_id, omission_count FROM shared_context WHERE user_id = ? AND context_body LIKE ?",
                                      (user_id, f'%{kw}%')).fetchone()
        if existing:
            self.conn.execute("UPDATE shared_context SET omission_count = omission_count + 1 WHERE context_id = ?", (existing[0],))
        else:
            self.conn.execute("INSERT INTO shared_context (user_id, context_body, omission_count, confirmed) VALUES (?, ?, 1, 0)", (user_id, kw))
        self.conn.commit()

    def _load_long_term_context(self, user_id: str) -> dict:
        identity = self.conn.execute("SELECT traits, voice_hint FROM identity_memory WHERE user_id = ?", (user_id,)).fetchone()
        recent_events = self.conn.execute("SELECT compressed_hint, relationship_delta FROM event_memory WHERE user_id = ? ORDER BY created_at DESC LIMIT 2", (user_id,)).fetchall()
        recent_trace = self.conn.execute("SELECT prose_hint, warmth, tension FROM emotional_trace WHERE user_id = ? ORDER BY created_at DESC LIMIT 1", (user_id,)).fetchone()
        arc = self.conn.execute("SELECT stage, stage_score FROM relationship_arc WHERE user_id = ?", (user_id,)).fetchone()
        shared = self.conn.execute("SELECT context_body FROM shared_context WHERE user_id = ? AND confirmed = 1 ORDER BY omission_count DESC LIMIT 4", (user_id,)).fetchall()
        entities = self.conn.execute("SELECT entity_name, relation_type, property FROM entity_relations WHERE user_id = ? ORDER BY mention_count DESC LIMIT 5", (user_id,)).fetchall()
        # Parse traits with strength, filter expired
        raw_traits = json.loads(identity[0]) if identity and identity[0] else []
        trait_details = []
        for item in raw_traits:
            if isinstance(item, str):
                trait_details.append({"text": item, "strength": 3})  # old format
            elif isinstance(item, dict):
                trait_details.append({"text": item.get("t", ""), "strength": item.get("s", 0), "count": item.get("c", 0)})
        trait_details = [p for p in trait_details if p["text"] and p["strength"] > 0]
        return {
            'identity_traits': [p["text"] for p in trait_details],
            'trait_details': trait_details,
            'voice_hint': identity[1] if identity else '',
            'events': [{'hint': r[0], 'delta': r[1]} for r in recent_events],
            'trace': {'prose': recent_trace[0] if recent_trace else '', 'warmth': recent_trace[1] if recent_trace else 0.0, 'tension': recent_trace[2] if recent_trace else 0.0},
            'arc_stage': arc[0] if arc else 'stranger', 'arc_score': arc[1] if arc else 0.0,
            'shared_contexts': [r[0] for r in shared],
            'entities': [{'name': r[0], 'relation': r[1], 'fact': r[2]} for r in entities],
        }

    # ── Post-Update ──

    def _post_update(self, user_id: str, session_id: str, user_input: str, stance: str, reply: str = ""):
        now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self._init_identity(user_id)
        self.conn.execute("UPDATE session_state SET last_active = ? WHERE session_id = ?", (now_str, session_id))
        self._decay_residue(session_id)
        self._update_relationship_arc(user_id, 'attentive')
        if reply:
            self._build_shared_context(user_id, user_input, reply)
        if len(user_input) > 15:
            recent = self.conn.execute("SELECT content FROM long_term_memory WHERE session_id = ? ORDER BY created_at DESC LIMIT 1", (session_id,)).fetchone()
            summary = user_input[:50] + "..." if len(user_input) > 50 else user_input
            if not recent or recent[0] != summary:
                self.conn.execute("INSERT INTO long_term_memory (user_id, content, memory_type, session_id, created_at) VALUES (?, ?, ?, ?, ?)",
                                  (user_id, summary, 'event', session_id, now_str))
                self._build_graph_from_memory(user_id, summary)
        self.conn.commit()

    # ── ANALYSIS LLM 事后分析 ──

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
        """LLM 事后分析：产出 JSON → 写入五张表 + traits/quirk/jokes + entities"""
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
  "traits_updates": ["新观察到的用户特质"],
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

        # 1. Emotional trace
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

        # 2. Relationship arc
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

        # 3. Identity traits + quirks — similarity merge + strength decay + confirmation floor
        identity = self.conn.execute("SELECT traits FROM identity_memory WHERE user_id = ?", (user_id,)).fetchone()
        if identity:
            raw = json.loads(identity[0]) if identity[0] else []
            trait_map = {}  # text -> {s: strength, c: confirm_count}
            for item in raw:
                if isinstance(item, str):
                    trait_map[item] = {"s": 3, "c": 0}
                elif isinstance(item, dict):
                    trait_map[item.get("t", "")] = {"s": item.get("s", 0), "c": item.get("c", 0)}

            # Similarity merge: embed new traits, compare with existing
            new_traits = list(data.get("traits_updates", []))
            confirmed = []
            if new_traits:
                existing_texts = list(trait_map.keys())
                all_texts = new_traits + existing_texts
                embs = []
                for t in all_texts:
                    emb = await self._get_embedding(t)
                    embs.append(np.array(emb, dtype=np.float32) if emb else None)
                n_new = len(new_traits)
                for i, new_t in enumerate(new_traits):
                    if embs[i] is None:
                        confirmed.append(new_t)
                        continue
                    found = False
                    for j, existing_t in enumerate(existing_texts):
                        if embs[n_new + j] is not None:
                            sim = np.dot(embs[i], embs[n_new + j]) / (np.linalg.norm(embs[i]) * np.linalg.norm(embs[n_new + j]) + 1e-12)
                            if sim > 0.82:
                                confirmed.append(existing_t)
                                found = True
                                break
                    if not found:
                        confirmed.append(new_t)

            # Quirks join the pool
            for q in data.get("speech_quirks", []):
                q_entry = f"[口癖] {q}"
                if q_entry not in trait_map:
                    trait_map[q_entry] = {"s": 0, "c": 0}
                confirmed.append(q_entry)

            # Process confirmed: reset strength, increment count
            seen = set()
            for t in confirmed:
                if t in seen:
                    continue
                seen.add(t)
                if t not in trait_map:
                    trait_map[t] = {"s": 5, "c": 1}
                else:
                    trait_map[t]["s"] = 5
                    trait_map[t]["c"] += 1

            # Decay unconfirmed: strength -= 1, floor = min(c, 3)
            for t in list(trait_map.keys()):
                if t not in seen:
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

        # 4. Shared jokes/context
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

        # 5. Boundary hits
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

        # 6. Dangling threads
        for dt in data.get("dangling_threads", []):
            self.conn.execute(
                "INSERT INTO event_memory (user_id, content, relationship_delta, emotional_weight, novelty, compressed_hint, created_at) VALUES (?, ?, 0.0, 0.3, 0.5, ?, ?)",
                (user_id, dt, dt[:40] + "..." if len(dt) > 40 else dt, now_str),
            )
        if session_id and data.get("dangling_threads"):
            row = self.conn.execute("SELECT turn_count FROM session_state WHERE session_id = ?", (session_id,)).fetchone()
            current_turn = row[0] if row else 0
            self.conn.execute(
                "UPDATE session_state SET dangling_threads = ? WHERE session_id = ?",
                (json.dumps({"threads": data["dangling_threads"], "turn": current_turn}, ensure_ascii=False), session_id),
            )

        # 7. Entities
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

        # 8. Event memory (if important enough)
        importance = data.get("importance", 0.0)
        if importance >= 0.5:
            summary = user_input[:80] + "..." if len(user_input) > 80 else user_input
            existing = self.conn.execute(
                "SELECT event_id FROM event_memory WHERE user_id = ? AND content = ?", (user_id, summary)
            ).fetchone()
            if not existing:
                self.conn.execute(
                    "INSERT INTO event_memory (user_id, content, relationship_delta, emotional_weight, novelty, compressed_hint, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (user_id, summary, rd, mood_intensity := data.get("mood_intensity", 0.0), importance, summary[:40] + "...", now_str),
                )

        log_parts = []
        if data.get("traits_updates"): log_parts.append(f"traits+{len(data['traits_updates'])}")
        if data.get("shared_jokes"): log_parts.append(f"jokes+{len(data['shared_jokes'])}")
        if data.get("boundary_hits"): log_parts.append(f"bounds+{len(data['boundary_hits'])}")
        if data.get("entities"): log_parts.append(f"ents+{len(data['entities'])}")
        if data.get("importance", 0) >= 0.5: log_parts.append("event")
        logger.info(f"ANALYSIS: write user={user_id} {' | '.join(log_parts) if log_parts else 'no-updates'}")

        self.conn.commit()

    # ── Graph Builder ──

    def _build_graph_from_memory(self, user_id: str, content: str):
        kws = self._extract_keywords(content, max_kw=8)
        if len(kws) < 2:
            return
        now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        node_ids = []
        for kw in kws:
            row = self.conn.execute("SELECT node_id FROM memory_graph_nodes WHERE user_id = ? AND label = ?", (user_id, kw)).fetchone()
            if row:
                self.conn.execute("UPDATE memory_graph_nodes SET freq = freq + 1, last_seen = ? WHERE node_id = ?", (now_str, row[0]))
                node_ids.append(row[0])
            else:
                cur = self.conn.execute("INSERT INTO memory_graph_nodes (user_id, label, freq, last_seen) VALUES (?, ?, 1, ?)", (user_id, kw, now_str))
                node_ids.append(cur.lastrowid)
        for i in range(len(node_ids)):
            for j in range(i + 1, len(node_ids)):
                a, b = sorted((node_ids[i], node_ids[j]))
                edge = self.conn.execute("SELECT weight, encounter_count FROM memory_graph_edges WHERE from_node_id = ? AND to_node_id = ?", (a, b)).fetchone()
                if edge:
                    self.conn.execute("UPDATE memory_graph_edges SET weight = weight + 0.5, encounter_count = encounter_count + 1, last_seen = ? WHERE from_node_id = ? AND to_node_id = ?", (now_str, a, b))
                else:
                    self.conn.execute("INSERT INTO memory_graph_edges (from_node_id, to_node_id, weight, encounter_count, last_seen) VALUES (?, ?, 1.0, 1, ?)", (a, b, now_str))
        self.conn.commit()

    # ── Save ──

    def save_turn(self, session_id: str, user_input: str, agent_reply: str, stance: str):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.conn.execute("INSERT INTO chat_history (session_id, role, content, created_at) VALUES (?, ?, ?, ?)", (session_id, 'user', user_input, timestamp))
        self.conn.execute("INSERT INTO chat_history (session_id, role, content, created_at) VALUES (?, ?, ?, ?)", (session_id, 'assistant', agent_reply, timestamp))
        # 首次会话插入初始行，后续递增
        self.conn.execute("INSERT OR IGNORE INTO session_state (session_id, stance, turn_count, last_active) VALUES (?, 'open', 0, ?)", (session_id, timestamp))
        self.conn.execute("UPDATE session_state SET turn_count = turn_count + 1, stance = ?, last_active = ? WHERE session_id = ?", (stance, timestamp, session_id))
        self.conn.commit()

    # ── Core Veto ──

    def _core_veto(self, prompt: str) -> str:
        for s in ['你应该', '你必须', '我教你', '听我说', '你这样不对']:
            if s in prompt:
                prompt = prompt.replace(s, '或许可以试试')
                break
        return prompt

    # ── Chat ──

    async def chat(self, user_id: str, session_id: str, user_input: str, backend: LLMBackend) -> str:
        memories = self.retrieve_memories(user_id, user_input, 'engaged')
        long_term = self._load_long_term_context(user_id)
        prompt = self.prompt_compressor(user_id, session_id, user_input, memories, long_term)
        prompt = self._core_veto(prompt)
        try:
            reply = await backend.generate(prompt)
        except Exception:
            try:
                reply = await backend.generate("简短回复，一句话以内。\n\n你:")
            except Exception:
                reply = "嗯。"
        self.save_turn(session_id, user_input, reply, 'open')
        self._post_update(user_id, session_id, user_input, 'open', reply)
        return reply

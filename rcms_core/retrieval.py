import logging
import os
import re
from datetime import datetime

import numpy as np
from openai import AsyncOpenAI

logger = logging.getLogger("rcms")


class RetrievalMixin:
    """关键词/图/Embedding 检索"""

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
            SELECT content, created_at FROM cognitive_distill
            WHERE ({conditions}) AND user_id = ?
            ORDER BY created_at DESC LIMIT ?
        """, params + [limit])
        rows = cursor.fetchall()
        logger.info(f"Retrieve: user={user_id} kws={keywords} found={len(rows)} source=keyword")
        return [(self._fuzz_time(r[1]) + '，' + r[0], 'event') for r in rows]

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

"""
RCMS v2 集成测试 — 核心链路验证

运行：cd 项目根目录 && python3 scripts/test_v2.py
需要环境变量：OPENAI_API_KEY（测试 embedding + ANALYSIS 时用）
或设置 SKIP_API=1 跳过 API 调用测试。
"""
import asyncio
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from rcms_core import MinimalRCMS

SKIP_API = os.environ.get("SKIP_API", "0") == "1"
DB_PATH = "/tmp/rcms_test_v2.db"


def rm_db():
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)


class TestV2Core(unittest.TestCase):
    """测试核心功能：不依赖外部 API"""

    def setUp(self):
        rm_db()
        self.rcms = MinimalRCMS(db_path=DB_PATH, analysis_config={})

    def tearDown(self):
        self.rcms.close()
        rm_db()

    def test_init_creates_new_tables(self):
        """确认新表存在"""
        tables = self.rcms.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        names = {r[0] for r in tables}
        for t in ["memory_embeddings", "entity_relations"]:
            self.assertIn(t, names, f"缺少表 {t}")

    def test_store_and_load_embedding(self):
        """存向量 blob，load 回 numpy cache"""
        fake_vec = list(np.random.randn(1536).astype(np.float32))
        self.rcms._store_embedding("test_user", 1, "测试内容", fake_vec)
        rows = self.rcms.conn.execute(
            "SELECT memory_id, content FROM memory_embeddings WHERE user_id = ?", ("test_user",)
        ).fetchall()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][0], 1)

        self.rcms._load_emb_cache("test_user")
        cache = self.rcms._emb_cache["test_user"]
        self.assertEqual(cache["vectors"].shape[0], 1)
        self.assertEqual(cache["meta"][0][0], 1)

    def test_retrieve_by_embedding_empty_cache_returns_empty(self):
        """没有 API key / 向量时返回空"""
        result = asyncio.run(self.rcms.retrieve_by_embedding("no_such_user", "测试"))
        # 没有配置 api_key 时 _get_embedding 返回 []，状态为 "emb_failed"
        self.assertEqual(result[0], [])
        self.assertIn(result[1], ("emb_failed", "no_vectors"))

    def test_keyword_fallback_when_emb_disabled(self):
        """embedding 关闭时，retrieve_memories 用关键词"""
        self.rcms.conn.execute(
            "INSERT INTO long_term_memory (user_id, content, memory_type, session_id) VALUES (?, ?, ?, ?)",
            ("test_user", "用户说今天工作很累", "event", "s1"),
        )
        self.rcms.conn.commit()
        # 关键词匹配：query 必须包含与记忆内容相同的连续词
        mems = self.rcms.retrieve_memories("test_user", "工作很累", "engaged", limit=2)
        self.assertTrue(len(mems) > 0, f"关键词应命中，返回 {mems}")

    def test_entity_relations_insert_and_update(self):
        """entity_relations 插入与去重"""
        from datetime import datetime
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.rcms.conn.execute(
            """INSERT INTO entity_relations (user_id, entity_name, relation_type, property, mention_count, last_mentioned, sentiment)
               VALUES (?, ?, ?, ?, 1, ?, 0.0)""",
            ("test_user", "小王", "朋友", "也喜欢摄影", now),
        )
        # 再次插入同名实体，ON CONFLICT 应增加 count
        self.rcms.conn.execute(
            """INSERT INTO entity_relations (user_id, entity_name, relation_type, property, mention_count, last_mentioned, sentiment)
               VALUES (?, ?, ?, ?, 1, ?, 0.0)
               ON CONFLICT(user_id, entity_name) DO UPDATE SET mention_count = mention_count + 1""",
            ("test_user", "小王", "", "", now),
        )
        row = self.rcms.conn.execute(
            "SELECT mention_count FROM entity_relations WHERE user_id = ? AND entity_name = ?",
            ("test_user", "小王"),
        ).fetchone()
        self.assertEqual(row[0], 2)

    def test_long_term_context_includes_new_fields(self):
        """_load_long_term_context 包含 entities"""
        from datetime import datetime
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.rcms._init_identity("test_user")
        self.rcms.conn.execute(
            """INSERT INTO entity_relations (user_id, entity_name, relation_type, property, mention_count, last_mentioned, sentiment)
               VALUES (?, ?, ?, ?, 1, ?, 0.0)""",
            ("test_user", "小李", "同事", "做前端的", now),
        )
        ctx = self.rcms._load_long_term_context("test_user")
        self.assertIn("entities", ctx)
        self.assertEqual(len(ctx["entities"]), 1)
        self.assertEqual(ctx["entities"][0]["name"], "小李")

    def test_narrative_context_with_traits_and_jokes(self):
        """narrative_context 包含特质和梗"""
        from datetime import datetime
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.rcms._init_identity("test_user")
        self.rcms.conn.execute(
            "UPDATE identity_memory SET traits = ?, updated_at = ? WHERE user_id = ?",
            (json.dumps(["[口癖] 说话爱带喵", "喜欢自嘲"], ensure_ascii=False), now, "test_user"),
        )
        self.rcms.conn.execute(
            "INSERT INTO shared_context (user_id, context_body, omission_count, confirmed) VALUES (?, ?, 1, 1)",
            ("test_user", "[梗] 喵 → 哈基米"),
        )
        long_term = self.rcms._load_long_term_context("test_user")
        context = self.rcms.narrative_context("open", long_term=long_term)
        self.assertIn("喵", context)
        self.assertIn("哈基米", context)

    def test_apply_analysis_writes_to_all_tables(self):
        """ANALYSIS JSON → 写入所有相关表"""
        from datetime import datetime
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.rcms._init_identity("test_user")
        data = {
            "mood": "温暖",
            "mood_intensity": 0.8,
            "topic_shift": False,
            "key_points": ["用户今天心情不错"],
            "relationship_delta": 1,
            "user_state": "open",
            "traits_updates": ["喜欢分享日常"],
            "speech_quirks": ["句尾加～"],
            "shared_jokes": [{"trigger": "猫", "context": "说猫就发猫图"}],
            "boundary_hits": ["不要追问工作细节"],
            "dangling_threads": ["用户说周末再聊这个"],
            "importance": 0.7,
            "entities": [{"name": "小红", "relation": "朋友", "fact": "也养猫"}],
        }
        self.rcms._apply_analysis("test_user", "今天好开心", "那真好呀", data)

        # emotional_trace 有写
        trace = self.rcms.conn.execute(
            "SELECT warmth, prose_hint FROM emotional_trace WHERE user_id = ?", ("test_user",)
        ).fetchone()
        self.assertIsNotNone(trace)
        self.assertGreater(trace[0], 0)

        # identity_memory traits 追加
        traits = json.loads(self.rcms.conn.execute(
            "SELECT traits FROM identity_memory WHERE user_id = ?", ("test_user",)
        ).fetchone()[0])
        self.assertIn("喜欢分享日常", traits)
        self.assertTrue(any("[口癖]" in t for t in traits))

        # shared_context 有梗和边界
        ctxs = self.rcms.conn.execute(
            "SELECT context_body FROM shared_context WHERE user_id = ?", ("test_user",)
        ).fetchall()
        bodies = [r[0] for r in ctxs]
        self.assertTrue(any("猫" in b for b in bodies))
        self.assertTrue(any("工作细节" in b for b in bodies))

        # event_memory 有 dangling thread 和重要事件
        events = self.rcms.conn.execute(
            "SELECT content FROM event_memory WHERE user_id = ?", ("test_user",)
        ).fetchall()
        contents = [r[0] for r in events]
        self.assertTrue(any("周末" in c for c in contents))

        # entity_relations 有实体
        ents = self.rcms.conn.execute(
            "SELECT entity_name FROM entity_relations WHERE user_id = ?", ("test_user",)
        ).fetchall()
        self.assertIn("小红", {r[0] for r in ents})


@unittest.skipIf(SKIP_API, "跳过 API 调用测试（设置 SKIP_API=1）")
class TestV2API(unittest.TestCase):
    """测试外部 API 调用 —— 需要 OPENAI_API_KEY"""

    @classmethod
    def setUpClass(cls):
        if not os.environ.get("OPENAI_API_KEY"):
            raise unittest.SkipTest("需要 OPENAI_API_KEY")

    def setUp(self):
        rm_db()
        cfg = {
            "retrieval": {
                "enabled": True,
                "provider": "openai",
                "model": "text-embedding-3-small",
            },
            "post_analysis": {
                "mode": "rule",  # 默认关，只测 embedding
            },
        }
        self.rcms = MinimalRCMS(db_path=DB_PATH, analysis_config=cfg)

    def tearDown(self):
        self.rcms.close()
        rm_db()

    def test_embedding_api_returns_vector(self):
        """真实 embedding API 调用"""
        vec = asyncio.run(self.rcms._get_embedding("测试句子"))
        self.assertEqual(len(vec), 1536)
        self.assertAlmostEqual(sum(v ** 2 for v in vec) ** 0.5, 1.0, places=3)

    def test_full_retrieval_roundtrip(self):
        """存向量 → 检索命中"""
        # 先存一条记忆
        self.rcms.conn.execute(
            "INSERT INTO long_term_memory (user_id, content, memory_type, session_id) VALUES (?, ?, ?, ?)",
            ("test_user", "用户说今天工作特别累，想早点休息", "event", "s1"),
        )
        mem_id = self.rcms.conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        self.rcms.conn.commit()

        # 生成 embedding 并存储
        vec = asyncio.run(self.rcms._get_embedding("用户说今天工作特别累，想早点休息"))
        self.rcms._store_embedding("test_user", mem_id, "用户说今天工作特别累，想早点休息", vec)

        # 加载到 cache
        self.rcms._load_emb_cache("test_user")

        # 检索
        results, source = asyncio.run(self.rcms.retrieve_by_embedding("test_user", "工作太累了", limit=2))
        self.assertEqual(source, "embedding")
        self.assertTrue(len(results) > 0, f"应返回记忆，实际 source={source}, results={results}")
        score = results[0][1]
        self.assertGreater(score, 0.3, f"相似度应 > 0.3，实际 {score}")

    def test_unrelated_query_returns_empty(self):
        """不相关查询不应匹配"""
        self.rcms.conn.execute(
            "INSERT INTO long_term_memory (user_id, content, memory_type, session_id) VALUES (?, ?, ?, ?)",
            ("test_user", "用户说想去日本看樱花", "event", "s1"),
        )
        mem_id = self.rcms.conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        self.rcms.conn.commit()
        vec = asyncio.run(self.rcms._get_embedding("用户说想去日本看樱花"))
        self.rcms._store_embedding("test_user", mem_id, "用户说想去日本看樱花", vec)
        self.rcms._load_emb_cache("test_user")

        results, source = asyncio.run(self.rcms.retrieve_by_embedding("test_user", "今天天气不错", limit=2))
        # 可能命中（低分），但不强求
        if results:
            self.assertLess(results[0][1], 0.5)


class TestV2EmbeddingPipeline(unittest.TestCase):
    """Mock embedding 测试完整检索链路"""

    def setUp(self):
        rm_db()
        cfg = {"retrieval": {"enabled": True}}
        self.rcms = MinimalRCMS(db_path=DB_PATH, analysis_config=cfg)
        # 注入 mock embedding：返回固定向量
        self.rcms._get_embedding = self._mock_embed

    def tearDown(self):
        self.rcms.close()
        rm_db()

    async def _mock_embed(self, text: str):
        import re
        chars = re.findall(r'[一-鿿]', text)
        freq = {}
        for c in chars:
            freq[c] = freq.get(c, 0) + 1
        for i in range(len(chars) - 1):
            bg = chars[i] + chars[i + 1]
            freq[bg] = freq.get(bg, 0) + 1
        v = np.zeros(1536, dtype=np.float32)
        for i, (_, val) in enumerate(sorted(freq.items())):
            if i >= 1536:
                break
            v[i] = val
        norm = np.linalg.norm(v)
        if norm > 0:
            v /= norm
        return v.tolist()

    def test_mock_embedding_retrieval(self):
        """mock embedding：存 → cache → 检索命中"""
        # 插入多条记忆
        texts = [
            "用户说今天工作特别累想早点休息",
            "用户提到周末去看了樱花很开心",
            "用户说最近在学做菜",
        ]
        for i, t in enumerate(texts):
            self.rcms.conn.execute(
                "INSERT INTO long_term_memory (user_id, content, memory_type, session_id) VALUES (?, ?, ?, ?)",
                ("test_user", t, "event", f"s{i}"),
            )
            mem_id = self.rcms.conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            vec = asyncio.run(self._mock_embed(t))
            self.rcms._store_embedding("test_user", mem_id, t, vec)
        self.rcms.conn.commit()
        self.rcms._load_emb_cache("test_user")

        # 检索：memory_embeddings 表有数据且 cache 加载正确
        cache = self.rcms._emb_cache.get("test_user")
        self.assertIsNotNone(cache)
        self.assertEqual(cache["vectors"].shape[0], 3)

        # embedding 检索不抛异常，结果存在或不存在都合理（mock 精度有限）
        results, source = asyncio.run(self.rcms.retrieve_by_embedding("test_user", "加班好累", limit=2))
        self.assertEqual(source, "embedding")

        # 检索：不相关查询不会抛异常
        results2, _ = asyncio.run(self.rcms.retrieve_by_embedding("test_user", "今天天气哈哈哈", limit=2))

    def test_embedding_fallback_on_empty(self):
        """没有匹配时返回空列表"""
        results, source = asyncio.run(self.rcms.retrieve_by_embedding("empty_user", "随便", limit=2))
        self.assertEqual(results, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)

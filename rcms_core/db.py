import logging

logger = logging.getLogger("rcms")


class DBMixin:
    """数据库初始化、迁移"""

    def _init_db(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS cognitive_distill (
                id INTEGER PRIMARY KEY,
                user_id TEXT,
                session_id TEXT,
                content TEXT NOT NULL,
                summary TEXT,
                mood TEXT DEFAULT '',
                mood_intensity REAL DEFAULT 0.0,
                importance REAL DEFAULT 0.3,
                entities TEXT DEFAULT '[]',
                embedding BLOB,
                turn_num INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS session_state (
                session_id TEXT PRIMARY KEY, user_id TEXT, stance TEXT DEFAULT 'open',
                mood REAL DEFAULT 0, focus_topic TEXT, turn_count INTEGER DEFAULT 0,
                stance_turns INTEGER DEFAULT 0, engagement_level TEXT DEFAULT 'coasting',
                momentum_depth REAL DEFAULT 0.0, momentum_energy REAL DEFAULT 0.0,
                last_active TIMESTAMP, residue_warmth REAL DEFAULT 0.0,
                residue_tension REAL DEFAULT 0.0, dangling_threads TEXT DEFAULT '[]',
                embedding_updated INTEGER DEFAULT 0,
                last_distill_turn INTEGER DEFAULT 0,
                last_distill_at TIMESTAMP
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
            CREATE TABLE IF NOT EXISTS entity_relations (
                id INTEGER PRIMARY KEY, user_id TEXT, entity_name TEXT,
                relation_type TEXT DEFAULT '', property TEXT DEFAULT '',
                mention_count INTEGER DEFAULT 1, last_mentioned TIMESTAMP,
                sentiment REAL DEFAULT 0.0,
                UNIQUE(user_id, entity_name)
            );
            CREATE INDEX IF NOT EXISTS idx_er_user ON entity_relations(user_id, entity_name);
            CREATE INDEX IF NOT EXISTS idx_cd_user ON cognitive_distill(user_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_cd_embed ON cognitive_distill(user_id) WHERE embedding IS NOT NULL;
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
            "ADD COLUMN last_distill_turn INTEGER DEFAULT 0",
            "ADD COLUMN last_distill_at TIMESTAMP",
        ]:
            try:
                self.conn.execute(f"ALTER TABLE session_state {col}")
            except Exception:
                pass
        # Migration: relationship_arc 去重 + 加 UNIQUE 约束
        dup_count = self.conn.execute(
            "SELECT COUNT(*) - COUNT(DISTINCT user_id) FROM relationship_arc"
        ).fetchone()[0]
        if dup_count > 0:
            self.conn.execute("""
                DELETE FROM relationship_arc WHERE arc_id NOT IN (
                    SELECT arc_id FROM (
                        SELECT arc_id, ROW_NUMBER() OVER (
                            PARTITION BY user_id ORDER BY updated_at DESC
                        ) AS rn FROM relationship_arc
                    ) WHERE rn = 1
                )
            """)
            logger.info(f"RCMS: relationship_arc 已去重 {dup_count} 行")
        self.conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_ra_user ON relationship_arc(user_id)"
        )
        # Migration: 旧表 → cognitive_distill
        self._migrate_to_cognitive_distill()
        self.conn.commit()

    def _migrate_to_cognitive_distill(self):
        """从 long_term_memory / event_memory / memory_embeddings 迁移到 cognitive_distill"""
        c = self.conn
        old = c.execute(
            "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='long_term_memory'"
        ).fetchone()[0]
        if not old:
            return

        existing = c.execute("SELECT count(*) FROM cognitive_distill").fetchone()[0]
        if existing > 0:
            # 已迁移过，只清理旧表
            for t in ("long_term_memory", "event_memory", "memory_embeddings"):
                c.execute(f"DROP TABLE IF EXISTS {t}")
            c.commit()
            logger.info("RCMS: 旧表已清理")
            return

        # 1. long_term_memory → cognitive_distill
        c.execute("""
            INSERT INTO cognitive_distill (user_id, session_id, content, importance, created_at)
            SELECT user_id, session_id, content,
                   CASE WHEN memory_type='event' THEN 0.5 ELSE 0.3 END,
                   created_at
            FROM long_term_memory
        """)
        ltm_count = c.execute("SELECT count(*) FROM cognitive_distill").fetchone()[0]

        # 2. 关联 embedding（按 memory_id 匹配）
        c.execute("""
            UPDATE cognitive_distill SET embedding = (
                SELECT me.embedding FROM memory_embeddings me
                WHERE me.user_id = cognitive_distill.user_id
                AND me.content = cognitive_distill.content
                LIMIT 1
            )
        """)

        # 3. event_memory → cognitive_distill（按事件追加）
        ev = c.execute("SELECT count(*) FROM event_memory").fetchone()[0]
        if ev:
            c.execute("""
                INSERT INTO cognitive_distill (user_id, content, summary, importance, created_at)
                SELECT user_id, content, compressed_hint, COALESCE(novelty, 0.5), created_at
                FROM event_memory
            """)
        total = c.execute("SELECT count(*) FROM cognitive_distill").fetchone()[0]
        embedded = c.execute(
            "SELECT count(*) FROM cognitive_distill WHERE embedding IS NOT NULL"
        ).fetchone()[0]
        logger.info(f"RCMS: 已迁移 {ltm_count} 行 long_term_memory + {ev} 行 event_memory → cognitive_distill (含 {embedded} 个向量)")

        # 4. 清理旧表
        for t in ("long_term_memory", "event_memory", "memory_embeddings"):
            c.execute(f"DROP TABLE IF EXISTS {t}")
        c.commit()

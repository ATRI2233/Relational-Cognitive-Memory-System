import logging
import sqlite3

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
                last_active TIMESTAMP, dangling_threads TEXT DEFAULT '[]',
                embedding_updated INTEGER DEFAULT 0,
                last_distill_turn INTEGER DEFAULT 0,
                last_distill_at TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY, session_id TEXT, role TEXT, content TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                turn_num INTEGER DEFAULT 0,
                importance REAL DEFAULT 0.3,
                mood TEXT DEFAULT ''
            );
            CREATE TABLE IF NOT EXISTS identity_memory (
                user_id TEXT PRIMARY KEY, traits TEXT DEFAULT '[]',
                preferences TEXT DEFAULT '{}',
                communication_style TEXT DEFAULT '',
                self_identity TEXT DEFAULT '[]',
                boundaries TEXT DEFAULT '[]',
                core_identity TEXT DEFAULT '{}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
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
                relation TEXT DEFAULT '',
                PRIMARY KEY (from_node_id, to_node_id)
            );
            CREATE INDEX IF NOT EXISTS idx_mgn_user_label ON memory_graph_nodes(user_id, label);
            CREATE INDEX IF NOT EXISTS idx_mge_from ON memory_graph_edges(from_node_id);
            CREATE INDEX IF NOT EXISTS idx_mge_to ON memory_graph_edges(to_node_id);
CREATE TABLE IF NOT EXISTS shared_context (
                context_id INTEGER PRIMARY KEY, user_id TEXT, context_body TEXT,
                omission_count INTEGER DEFAULT 0, confirmed INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_sc_user ON shared_context(user_id);
            CREATE INDEX IF NOT EXISTS idx_cd_user ON cognitive_distill(user_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_cd_embed ON cognitive_distill(user_id) WHERE embedding IS NOT NULL;
        """)
        # 新表：保存原始 LLM 蒸馏响应以便审计与回滚
        try:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS analysis_raw (
                    id INTEGER PRIMARY KEY,
                    user_id TEXT,
                    session_id TEXT,
                    content TEXT,
                    parsed INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
        except sqlite3.OperationalError:
            pass
        # Embedding 维度元数据列（用于检测模型变更）
        try:
            self.conn.execute("ALTER TABLE cognitive_distill ADD COLUMN embedding_dim INTEGER DEFAULT NULL")
        except sqlite3.OperationalError:
            pass
        for col in [
            "ADD COLUMN stance_turns INTEGER DEFAULT 0",
            "ADD COLUMN engagement_level TEXT DEFAULT 'coasting'",
            "ADD COLUMN momentum_depth REAL DEFAULT 0.0",
            "ADD COLUMN momentum_energy REAL DEFAULT 0.0",
            "ADD COLUMN last_active TIMESTAMP",
            "ADD COLUMN dangling_threads TEXT DEFAULT ''",
            "ADD COLUMN last_distill_turn INTEGER DEFAULT 0",
            "ADD COLUMN last_distill_at TIMESTAMP",
        ]:
            try:
                self.conn.execute(f"ALTER TABLE session_state {col}")
            except sqlite3.OperationalError:
                pass
        try:
            self.conn.execute("ALTER TABLE memory_graph_edges ADD COLUMN relation TEXT DEFAULT ''")
        except sqlite3.OperationalError:
            pass
        try:
            self.conn.execute("ALTER TABLE chat_history ADD COLUMN turn_num INTEGER DEFAULT 0")
        except sqlite3.OperationalError:
            pass
        for col in [
            "ADD COLUMN importance REAL DEFAULT 0.3",
            "ADD COLUMN mood TEXT DEFAULT ''",
        ]:
            try:
                self.conn.execute(f"ALTER TABLE chat_history {col}")
            except sqlite3.OperationalError:
                pass
        try:
            self.conn.execute("ALTER TABLE chat_history ADD COLUMN user_id TEXT DEFAULT ''")
        except sqlite3.OperationalError:
            pass
        try:
            self.conn.execute("ALTER TABLE chat_history ADD COLUMN sender_name TEXT DEFAULT ''")
        except sqlite3.OperationalError:
            pass
        try:
            self.conn.execute("ALTER TABLE memory_graph_nodes ADD COLUMN entity_type TEXT DEFAULT 'auto'")
        except sqlite3.OperationalError:
            pass
        try:
            self.conn.execute("ALTER TABLE memory_graph_edges ADD COLUMN created_at TEXT DEFAULT ''")
        except sqlite3.OperationalError:
            pass
        try:
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_ch_session ON chat_history(session_id)")
        except sqlite3.OperationalError:
            pass
        for col in [
            "ADD COLUMN preferences TEXT DEFAULT '{}'",
            "ADD COLUMN communication_style TEXT DEFAULT ''",
            "ADD COLUMN self_identity TEXT DEFAULT '[]'",
            "ADD COLUMN boundaries TEXT DEFAULT '[]'",
            "ADD COLUMN core_identity TEXT DEFAULT '{}'",
        ]:
            try:
                self.conn.execute(f"ALTER TABLE identity_memory {col}")
            except sqlite3.OperationalError:
                pass
        # 联合索引（提升认知蒸馏检索性能）
        try:
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_cd_user_imp ON cognitive_distill(user_id, importance DESC, created_at DESC)")
        except sqlite3.OperationalError:
            pass
        try:
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_cd_mood ON cognitive_distill(user_id, mood) WHERE mood IS NOT NULL")
        except sqlite3.OperationalError:
            pass
        try:
            self.conn.execute("ALTER TABLE cognitive_distill ADD COLUMN expires_at TIMESTAMP DEFAULT NULL")
        except sqlite3.OperationalError:
            pass
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

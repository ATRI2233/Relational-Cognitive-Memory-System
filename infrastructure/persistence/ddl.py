"""
ddl.py — 数据库 DDL 定义与 Schema 迁移

集中管理所有 CREATE TABLE / CREATE INDEX / ALTER TABLE 语句。
对应旧代码 DBMixin._init_db() 的全部逻辑，外加废弃表清理。

用法:
    from infrastructure.persistence.ddl import ensure_schema
    ensure_schema(conn, config.storage)
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3

from infrastructure.config.settings import StorageSettings

logger = logging.getLogger("rcms")

# ============================================================
# 全部 DDL 语句
# 与 tests/test_integration.py _DDL_STATEMENTS 以及
# scripts/schema_snapshot.sql 保持一致
# ============================================================

_DDL = """
CREATE TABLE IF NOT EXISTS analysis_raw (
    id INTEGER PRIMARY KEY,
    user_id TEXT,
    session_id TEXT,
    content TEXT,
    parsed INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS chat_history (
    id INTEGER PRIMARY KEY,
    session_id TEXT,
    role TEXT,
    content TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    turn_num INTEGER DEFAULT 0,
    importance REAL DEFAULT 0.3,
    mood TEXT DEFAULT '',
    user_id TEXT DEFAULT '',
    sender_name TEXT DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_ch_session ON chat_history(session_id);

CREATE TABLE IF NOT EXISTS cognitive_distill (
    id INTEGER PRIMARY KEY,
    user_id TEXT,
    session_id TEXT,
    content TEXT NOT NULL,
    keylabel TEXT,
    summary TEXT DEFAULT '',
    mood TEXT DEFAULT '',
    mood_intensity REAL DEFAULT 0.0,
    importance REAL DEFAULT 0.3,
    entities TEXT DEFAULT '[]',
    embedding BLOB,
    turn_num INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    embedding_dim INTEGER DEFAULT NULL,
    expires_at TIMESTAMP DEFAULT NULL
);

CREATE INDEX IF NOT EXISTS idx_cd_user ON cognitive_distill(user_id, created_at);
CREATE INDEX IF NOT EXISTS idx_cd_embed ON cognitive_distill(user_id) WHERE embedding IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_cd_user_imp ON cognitive_distill(user_id, importance DESC, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_cd_mood ON cognitive_distill(user_id, mood) WHERE mood IS NOT NULL;

CREATE TABLE IF NOT EXISTS identity_memory (
    user_id TEXT PRIMARY KEY,
    traits TEXT DEFAULT '[]',
    preferences TEXT DEFAULT '{}',
    self_identity TEXT DEFAULT '[]',
    boundaries TEXT DEFAULT '[]',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS memory_graph_edges (
    from_node_id INTEGER,
    to_node_id INTEGER,
    weight REAL DEFAULT 1.0,
    encounter_count INTEGER DEFAULT 1,
    last_seen TIMESTAMP,
    relation TEXT DEFAULT '',
    created_at TEXT DEFAULT '',
    PRIMARY KEY (from_node_id, to_node_id)
);

CREATE INDEX IF NOT EXISTS idx_mge_from ON memory_graph_edges(from_node_id);
CREATE INDEX IF NOT EXISTS idx_mge_to ON memory_graph_edges(to_node_id);

CREATE TABLE IF NOT EXISTS memory_graph_nodes (
    node_id INTEGER PRIMARY KEY,
    user_id TEXT,
    label TEXT,
    node_type TEXT DEFAULT 'keyword',
    freq INTEGER DEFAULT 1,
    last_seen TIMESTAMP,
    entity_type TEXT DEFAULT 'auto',
    UNIQUE(user_id, label)
);

CREATE INDEX IF NOT EXISTS idx_mgn_user_label ON memory_graph_nodes(user_id, label);

CREATE TABLE IF NOT EXISTS session_state (
    session_id TEXT PRIMARY KEY,
    user_id TEXT,
    stance TEXT DEFAULT 'open',
    mood REAL DEFAULT 0,
    turn_count INTEGER DEFAULT 0,
    stance_turns INTEGER DEFAULT 0,
    engagement_level TEXT DEFAULT 'coasting',
    momentum_depth REAL DEFAULT 0.0,
    momentum_energy REAL DEFAULT 0.0,
    last_active TIMESTAMP,
    dangling_threads TEXT DEFAULT '[]',
    embedding_updated INTEGER DEFAULT 0,
    last_distill_turn INTEGER DEFAULT 0,
    last_distill_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS user_mappings (
    id INTEGER PRIMARY KEY,
    session_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    label TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'nickname',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(session_id, user_id, label)
);

CREATE INDEX IF NOT EXISTS idx_um_session ON user_mappings(session_id);
CREATE INDEX IF NOT EXISTS idx_um_label ON user_mappings(session_id, label);

CREATE TABLE IF NOT EXISTS retrieval_trace (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    user_id TEXT,
    display_name TEXT,
    role TEXT,
    user_input TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    elapsed_ms INTEGER DEFAULT 0,
    embedding_ok INTEGER DEFAULT 0,
    embedding_dim INTEGER DEFAULT 0,
    keywords TEXT DEFAULT '[]',
    time_filter TEXT,
    channels TEXT DEFAULT '{}',
    fusion TEXT DEFAULT '{}',
    final_tags TEXT DEFAULT '[]',
    graph_paths TEXT DEFAULT '[]',
    extra TEXT DEFAULT '{}',
    injected TEXT DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_rt_created ON retrieval_trace(created_at);
CREATE INDEX IF NOT EXISTS idx_rt_user ON retrieval_trace(user_id);
"""


# ============================================================
# v2 已废弃的表
#
# 这些表在 v1 承担过职责，v2 里既没有写入方也没有读取方，
# 留着只会让人误以为数据在那里（面板/观测站都会列出来）。
# - shared_context：梗库，蒸馏从未产出 shared_jokes
# - memory_links：记忆间关联，v2 用图谱边取代
# - entity_relations：v1 三元组关系，v2 用 memory_graph_edges 取代
# - emotional_trace：情绪轨迹，v2 情绪写在 cognitive_distill.mood
# - relationship_arc：关系弧，v2 关系写在 session_state.stance
# - embedding_rebuild_queue：重建队列只进不出（无消费者）
# ============================================================

_OBSOLETE_TABLES = (
    "shared_context",
    "memory_links",
    "entity_relations",
    "emotional_trace",
    "relationship_arc",
    "embedding_rebuild_queue",
)


# ============================================================
# 迁移 — ALTER TABLE ADD COLUMN
# 所有迁移均幂等：已存在的列静默跳过
# ============================================================


def _run_migrations(conn: sqlite3.Connection) -> None:
    """增量 Schema 迁移：所有 ALTER TABLE 均在 try/except 中静默跳过重复列。

    对应旧代码 DBMixin._init_db() 行 102-199 的全部迁移逻辑。
    """
    # ── cognitive_distill 扩展列 ──
    for col in [
        "ADD COLUMN embedding_dim INTEGER DEFAULT NULL",
        "ADD COLUMN expires_at TIMESTAMP DEFAULT NULL",
    ]:
        try:
            conn.execute(f"ALTER TABLE cognitive_distill {col}")
        except sqlite3.OperationalError:
            pass

    # 列改名历史：summary → keylabel
    try:
        conn.execute("ALTER TABLE cognitive_distill RENAME COLUMN summary TO keylabel")
    except sqlite3.OperationalError:
        pass

    # 改名后重新添加 summary 列
    try:
        conn.execute("ALTER TABLE cognitive_distill ADD COLUMN summary TEXT DEFAULT ''")
    except sqlite3.OperationalError:
        pass

    # ── session_state 扩展列 ──
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
            conn.execute(f"ALTER TABLE session_state {col}")
        except sqlite3.OperationalError:
            pass

    # ── memory_graph_edges 扩展列 ──
    for col in [
        "ADD COLUMN relation TEXT DEFAULT ''",
        "ADD COLUMN created_at TEXT DEFAULT ''",
    ]:
        try:
            conn.execute(f"ALTER TABLE memory_graph_edges {col}")
        except sqlite3.OperationalError:
            pass

    # ── chat_history 扩展列 ──
    for col in [
        "ADD COLUMN turn_num INTEGER DEFAULT 0",
        "ADD COLUMN importance REAL DEFAULT 0.3",
        "ADD COLUMN mood TEXT DEFAULT ''",
        "ADD COLUMN user_id TEXT DEFAULT ''",
        "ADD COLUMN sender_name TEXT DEFAULT ''",
    ]:
        try:
            conn.execute(f"ALTER TABLE chat_history {col}")
        except sqlite3.OperationalError:
            pass

    # ── memory_graph_nodes 扩展列 ──
    try:
        conn.execute("ALTER TABLE memory_graph_nodes ADD COLUMN entity_type TEXT DEFAULT 'auto'")
    except sqlite3.OperationalError:
        pass

    # ── identity_memory 扩展列 ──
    for col in [
        "ADD COLUMN preferences TEXT DEFAULT '{}'",
        "ADD COLUMN self_identity TEXT DEFAULT '[]'",
        "ADD COLUMN boundaries TEXT DEFAULT '[]'",
    ]:
        try:
            conn.execute(f"ALTER TABLE identity_memory {col}")
        except sqlite3.OperationalError:
            pass

    # ── memory_graph_nodes UNIQUE 索引（upsert_node 原子化需要） ──
    try:
        # 先去重：对重复的 (user_id, label) 只保留 node_id 最小的行
        conn.execute("""
            DELETE FROM memory_graph_nodes WHERE node_id NOT IN (
                SELECT MIN(node_id) FROM memory_graph_nodes GROUP BY user_id, label
            )
        """)
    except sqlite3.OperationalError:
        pass
    try:
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_mgn_unique_user_label "
            "ON memory_graph_nodes(user_id, label)"
        )
    except sqlite3.OperationalError:
        pass

    # ── 额外的 CREATE INDEX（旧代码的补充索引） ──
    for idx in [
        "CREATE INDEX IF NOT EXISTS idx_cd_user_imp ON cognitive_distill(user_id, importance DESC, created_at DESC)",
        "CREATE INDEX IF NOT EXISTS idx_cd_mood ON cognitive_distill(user_id, mood) WHERE mood IS NOT NULL",
    ]:
        try:
            conn.execute(idx)
        except sqlite3.OperationalError:
            pass


# ============================================================
# 旧表迁移
# ============================================================


def _migrate_to_cognitive_distill(conn: sqlite3.Connection) -> None:
    """从 long_term_memory / event_memory / memory_embeddings 迁移到 cognitive_distill。

    对应旧代码 DBMixin._migrate_to_cognitive_distill() 行 235-290。
    仅在旧表存在且 cognitive_distill 为空时执行。
    """
    old = conn.execute(
        "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='long_term_memory'"
    ).fetchone()[0]
    if not old:
        return

    existing = conn.execute("SELECT count(*) FROM cognitive_distill").fetchone()[0]
    if existing > 0:
        # 已迁移过，只清理旧表
        for t in ("long_term_memory", "event_memory", "memory_embeddings"):
            try:
                conn.execute(f"DROP TABLE IF EXISTS {t}")
            except sqlite3.OperationalError:
                pass
        conn.commit()
        logger.info("旧表已清理")
        return

    logger.info("检测到旧表 long_term_memory，开始迁移至 cognitive_distill ...")

    # 1. long_term_memory → cognitive_distill
    conn.execute("""
        INSERT INTO cognitive_distill (user_id, session_id, content, summary, importance, created_at)
        SELECT user_id, session_id, content, content,
               CASE WHEN memory_type='event' THEN 0.5 ELSE 0.3 END,
               created_at
        FROM long_term_memory
    """)
    ltm_count = conn.execute("SELECT count(*) FROM cognitive_distill").fetchone()[0]

    # 2. 关联 embedding（按 content 匹配）
    conn.execute("""
        UPDATE cognitive_distill SET embedding = (
            SELECT me.embedding FROM memory_embeddings me
            WHERE me.user_id = cognitive_distill.user_id
            AND me.content = cognitive_distill.content
            LIMIT 1
        )
    """)

    # 3. event_memory → cognitive_distill
    ev = conn.execute("SELECT count(*) FROM event_memory").fetchone()[0]
    if ev:
        conn.execute("""
            INSERT INTO cognitive_distill (user_id, content, keylabel, summary, importance, created_at)
            SELECT user_id, content, compressed_hint, content, COALESCE(novelty, 0.5), created_at
            FROM event_memory
        """)

    total = conn.execute("SELECT count(*) FROM cognitive_distill").fetchone()[0]
    embedded = conn.execute(
        "SELECT count(*) FROM cognitive_distill WHERE embedding IS NOT NULL"
    ).fetchone()[0]
    logger.info(
        "已迁移 %d 行 long_term_memory + %d 行 event_memory -> cognitive_distill (含 %d 个向量)",
        ltm_count,
        ev or 0,
        embedded,
    )

    # 4. 清理旧表
    for t in ("long_term_memory", "event_memory", "memory_embeddings"):
        try:
            conn.execute(f"DROP TABLE IF EXISTS {t}")
        except sqlite3.OperationalError:
            pass

    conn.commit()


def _migrate_user_mappings(conn: sqlite3.Connection) -> None:
    """从 chat_history 回填 user_mappings（初次部署时历史数据）。

    对应旧代码 DBMixin._migrate_user_mappings() 行 207-233。
    仅当 user_mappings 为空且 chat_history 有数据时执行。
    """
    try:
        has_history = conn.execute(
            "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='chat_history'"
        ).fetchone()[0]
        if not has_history:
            return

        existing = conn.execute("SELECT count(*) FROM user_mappings").fetchone()[0]
        if existing > 0:
            return  # 已有数据，跳过回填

        conn.execute("""
            INSERT OR IGNORE INTO user_mappings (session_id, user_id, label, source)
            SELECT DISTINCT session_id, user_id, sender_name, 'nickname'
            FROM chat_history
            WHERE sender_name != '' AND user_id != ''
        """)
        backfilled = conn.execute("SELECT count(*) FROM user_mappings").fetchone()[0]
        if backfilled:
            logger.info("从 chat_history 回填 user_mappings %d 条", backfilled)
    except Exception as e:
        logger.warning("user_mappings 回填跳过 (%s)", e)


# ============================================================
# WAL Checkpoint
# ============================================================


def wal_checkpoint(
    conn: sqlite3.Connection,
    db_path: str = "",
    force_truncate: bool = False,
    max_wal_size_kb: int = 200,
) -> None:
    """执行 WAL checkpoint 以控制 WAL 文件大小。

    对应旧代码 SessionMixin.save_turn() 行 41-49 的 WAL 维护逻辑。

    Args:
        conn: SQLite 数据库连接。
        db_path: 数据库文件路径（用于检查 WAL 文件大小）。
        force_truncate: 强制 TRUNCATE checkpoint。
        max_wal_size_kb: WAL 文件超过此大小时使用 TRUNCATE（默认 200KB）。
    """
    try:
        if force_truncate:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            logger.debug("WAL TRUNCATE checkpoint 已执行")
        elif db_path:
            wal_path = f"{db_path}-wal"
            if os.path.isfile(wal_path) and os.path.getsize(wal_path) > max_wal_size_kb * 1024:
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                logger.debug("WAL TRUNCATE checkpoint 已执行 (文件 %.1f KB > %d KB)", os.path.getsize(wal_path) / 1024, max_wal_size_kb)
            else:
                conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
        else:
            conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
    except Exception as e:
        logger.debug("WAL checkpoint 跳过: %s", e)


# ============================================================
# 废弃表清理
# ============================================================


def _resolve_db_dir(conn: sqlite3.Connection) -> str:
    """从连接本身取数据库所在目录（避免依赖调用方传参）。"""
    try:
        for row in conn.execute("PRAGMA database_list").fetchall():
            if row[1] == "main" and row[2]:
                return os.path.dirname(os.path.abspath(row[2]))
    except sqlite3.Error:
        pass
    return ""


def _dump_table(conn: sqlite3.Connection, table: str) -> list[dict]:
    """把一张表读成可 JSON 序列化的记录列表。"""
    cols = [r[1] for r in conn.execute(f'PRAGMA table_info("{table}")').fetchall()]
    rows = conn.execute(f'SELECT * FROM "{table}"').fetchall()
    out: list[dict] = []
    for row in rows:
        record: dict = {}
        for name, value in zip(cols, row):
            if isinstance(value, bytes):
                record[name] = f"<BLOB {len(value)} bytes>"
            else:
                record[name] = value
        out.append(record)
    return out


def _drop_obsolete_tables(conn: sqlite3.Connection, data_dir: str = "") -> None:
    """删除 v2 已废弃的表。

    有数据先落到 ``obsolete_tables_<时间戳>.json`` 再删 —— 记忆库可以重建，
    但用户的历史数据不该被一个升级动作静默带走。空表直接删，不产生噪音文件。

    Args:
        conn: SQLite 连接。
        data_dir: 备份落盘目录；为空时用数据库文件所在目录。
    """
    try:
        existing = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        targets = [t for t in _OBSOLETE_TABLES if t in existing]
        if not targets:
            return

        backup: dict[str, list[dict]] = {}
        for table in targets:
            try:
                rows = _dump_table(conn, table)
            except sqlite3.Error as e:
                logger.warning("废弃表 %s 读取失败，跳过删除: %s", table, e)
                continue
            if rows:
                backup[table] = rows

        if backup:
            target_dir = data_dir or _resolve_db_dir(conn)
            try:
                os.makedirs(target_dir, exist_ok=True)
                stamp = conn.execute(
                    "SELECT strftime('%Y%m%d-%H%M%S','now')"
                ).fetchone()[0]
                path = os.path.join(
                    target_dir, f"obsolete_tables_{stamp}.json"
                )
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(backup, f, ensure_ascii=False, indent=2)
                logger.info(
                    "废弃表数据已备份到 %s（%s）",
                    path,
                    ", ".join(f"{k}:{len(v)}" for k, v in backup.items()),
                )
            except (OSError, sqlite3.Error) as e:
                logger.warning("废弃表备份失败，本次不删除非空表: %s", e)
                targets = [t for t in targets if t not in backup]

        for table in targets:
            try:
                conn.execute(f'DROP TABLE IF EXISTS "{table}"')
                logger.info("已删除废弃表 %s", table)
            except sqlite3.Error as e:
                logger.warning("删除废弃表 %s 失败: %s", table, e)
        conn.commit()
    except sqlite3.Error as e:
        logger.warning("废弃表清理跳过: %s", e)


# ============================================================
# 主入口 — 确保 Schema 完整
# ============================================================


def ensure_schema(conn: sqlite3.Connection, config: StorageSettings) -> None:
    """确保数据库 Schema 完整：建表 -> 建索引 -> 迁移 -> 旧表迁移。

    幂等安全，可重复调用。在 rcms_factory.create_core() 中，
    连接数据库后立即调用。

    Args:
        conn: SQLite 数据库连接。
        config: 存储配置（用于 busy_timeout、WAL autocheckpoint 等参数）。

    Raises:
        sqlite3.Error: Schema 初始化失败（建表/迁移失败）。
    """
    try:
        # 1. 创建所有表与索引
        conn.executescript(_DDL)
        logger.debug("DDL 建表/建索引完成")

        # 2. 增量 Schema 迁移（ALTER TABLE ADD COLUMN 等）
        _run_migrations(conn)

        # 3. 旧表迁移（long_term_memory / event_memory -> cognitive_distill）
        _migrate_to_cognitive_distill(conn)

        # 4. 从 chat_history 回填 user_mappings
        _migrate_user_mappings(conn)

        # 5. 删除 v2 已废弃的表（有数据先备份成 JSON）
        _drop_obsolete_tables(conn, getattr(config, "data_dir", "") or "")

        # 6. 设置 WAL autocheckpoint（从配置中读取，默认 50 页约 200KB）
        conn.execute(f"PRAGMA wal_autocheckpoint={config.wal_autocheckpoint_pages}")

        conn.commit()
        logger.info(
            "Schema 初始化/迁移完成 (wal_autocheckpoint=%d)",
            config.wal_autocheckpoint_pages,
        )
    except sqlite3.Error as e:
        conn.rollback()
        logger.error("Schema 初始化/迁移失败: %s", e)
        raise

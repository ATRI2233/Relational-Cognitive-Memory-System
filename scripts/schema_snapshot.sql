-- Schema Snapshot -- 生成日期: 2026-06-30
-- 来源: rcms_core/db.py + rcms_core/memory_link.py
-- 说明: 由 DDL 提取脚本自动生成，可直接于 sqlite3 中执行

-- ============================================================
-- 表: analysis_raw
-- 用途: 保存原始 LLM 蒸馏响应，以便审计与回滚
-- ============================================================
CREATE TABLE IF NOT EXISTS analysis_raw (
    id INTEGER PRIMARY KEY,
    user_id TEXT,
    session_id TEXT,
    content TEXT,
    parsed INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================
-- 表: chat_history
-- 用途: 聊天历史记录，存储用户与 AI 之间的对话轮次
-- ============================================================
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

-- ============================================================
-- 表: cognitive_distill
-- 用途: 核心认知蒸馏表，存储记忆摘要、情绪、重要性、实体、
--      向量嵌入、嵌入维度元数据和过期时间
-- ============================================================
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

-- ============================================================
-- 表: embedding_rebuild_queue
-- 用途: Embedding 重建队列，记录因模型更换或维度变化
--      需要重新向量化的条目
-- ============================================================
CREATE TABLE IF NOT EXISTS embedding_rebuild_queue (
    id INTEGER PRIMARY KEY,
    user_id TEXT NOT NULL,
    record_id INTEGER NOT NULL,
    reason TEXT DEFAULT '',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================
-- 表: identity_memory
-- 用途: 身份记忆表，存储用户的特质、偏好、自我认知和边界信息
-- ============================================================
CREATE TABLE IF NOT EXISTS identity_memory (
    user_id TEXT PRIMARY KEY,
    traits TEXT DEFAULT '[]',
    preferences TEXT DEFAULT '{}',
    self_identity TEXT DEFAULT '[]',
    boundaries TEXT DEFAULT '[]',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP
);

-- ============================================================
-- 表: memory_graph_edges
-- 用途: 记忆图谱边表，记录关键词/实体之间的关联权重和关系类型
-- ============================================================
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

-- ============================================================
-- 表: memory_graph_nodes
-- 用途: 记忆图谱节点表，存储关键词/实体节点及其出现频率
-- ============================================================
CREATE TABLE IF NOT EXISTS memory_graph_nodes (
    node_id INTEGER PRIMARY KEY,
    user_id TEXT,
    label TEXT,
    node_type TEXT DEFAULT 'keyword',
    freq INTEGER DEFAULT 1,
    last_seen TIMESTAMP,
    entity_type TEXT DEFAULT 'auto'
);

CREATE INDEX IF NOT EXISTS idx_mgn_user_label ON memory_graph_nodes(user_id, label);

-- ============================================================
-- 表: memory_links
-- 用途: 记忆关联表，让 cognitive_distill 中的任意两条记忆
--      (content / summary) 建立显式关联
-- ============================================================
CREATE TABLE IF NOT EXISTS memory_links (
    id INTEGER PRIMARY KEY,
    user_id TEXT NOT NULL,
    from_memory_id INTEGER NOT NULL,
    to_memory_id INTEGER NOT NULL,
    link_type TEXT DEFAULT 'related',
    reason TEXT DEFAULT '',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(from_memory_id, to_memory_id)
);

-- ============================================================
-- 表: session_state
-- 用途: 会话状态表，追踪每次对话的立场、情绪、轮次、
--      参与度、动量等会话级元数据
-- ============================================================
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

-- ============================================================
-- 表: shared_context
-- 用途: 共享上下文表，记录与用户共同确认的背景信息
-- ============================================================
CREATE TABLE IF NOT EXISTS shared_context (
    context_id INTEGER PRIMARY KEY,
    user_id TEXT,
    context_body TEXT,
    omission_count INTEGER DEFAULT 0,
    confirmed INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_sc_user ON shared_context(user_id);

-- ============================================================
-- 表: user_mappings
-- 用途: 用户映射表，将 session_id 映射到 user_id 及昵称标签
-- ============================================================
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

"""
memory_link.py — 最小记忆关联

让 cognitive_distill 中的任意两条记忆（content / summary）建立显式关联。
目前只提供建表，读写方式待定。
"""

CREATE_MEMORY_LINKS = """
CREATE TABLE IF NOT EXISTS memory_links (
    id          INTEGER PRIMARY KEY,
    user_id     TEXT NOT NULL,
    from_memory_id INTEGER NOT NULL,
    to_memory_id   INTEGER NOT NULL,
    link_type   TEXT DEFAULT 'related',
    reason      TEXT DEFAULT '',
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(from_memory_id, to_memory_id)
);
"""


def ensure_memory_links(conn):
    conn.execute(CREATE_MEMORY_LINKS)
    conn.commit()

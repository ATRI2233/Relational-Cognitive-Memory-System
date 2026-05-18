import logging
from datetime import datetime

logger = logging.getLogger("rcms")


class SessionMixin:
    """会话状态、save_turn"""

    def save_turn(self, session_id: str, user_input: str, agent_reply: str):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        row = self.conn.execute(
            "SELECT turn_count FROM session_state WHERE session_id = ?", (session_id,)
        ).fetchone()
        turn_num = (row[0] or 0) + 1 if row else 1
        # 计算基本 importance（基于情绪词规则）
        importance = 0.3
        emotional_words = getattr(self, '_EMOTIONAL_WORDS', [])
        hits = sum(1 for w in emotional_words if w in user_input)
        if hits:
            importance = min(0.3 + hits * 0.1, 0.8)
        if len(user_input) > 50:
            importance = min(importance + 0.1, 0.8)
        self.conn.execute("INSERT INTO chat_history (session_id, role, content, turn_num, created_at, importance) VALUES (?, ?, ?, ?, ?, ?)", (session_id, 'user', user_input, turn_num, timestamp, importance))
        self.conn.execute("INSERT INTO chat_history (session_id, role, content, turn_num, created_at) VALUES (?, ?, ?, ?, ?)", (session_id, 'assistant', agent_reply, turn_num, timestamp))
        self.conn.execute("INSERT OR IGNORE INTO session_state (session_id, stance, turn_count, last_active) VALUES (?, 'open', 0, ?)", (session_id, timestamp))
        self.conn.execute("UPDATE session_state SET turn_count = turn_count + 1, last_active = ? WHERE session_id = ?", (timestamp, session_id))
        self.conn.commit()

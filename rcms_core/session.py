import logging
import os
from datetime import datetime

logger = logging.getLogger("rcms")


class SessionMixin:
    """会话状态、save_turn"""

    def save_turn(self, session_id: str, user_input: str, agent_reply: str, user_id: str = "", sender_name: str = ""):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        # 使用实例级 DB 锁保证并发写入的原子性
        lock = getattr(self, '_db_lock', None)
        if lock:
            lock.acquire()
        try:
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
            self.conn.execute("INSERT INTO chat_history (session_id, role, content, turn_num, created_at, importance, user_id, sender_name) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (session_id, 'user', user_input, turn_num, timestamp, importance, user_id, sender_name))
            self.conn.execute("INSERT INTO chat_history (session_id, role, content, turn_num, created_at, user_id, sender_name) VALUES (?, ?, ?, ?, ?, ?, ?)", (session_id, 'assistant', agent_reply, turn_num, timestamp, user_id, sender_name))
            # 自动注册 sender_name 到 session 用户映射
            if sender_name:
                self.conn.execute(
                    "INSERT OR IGNORE INTO user_mappings (session_id, user_id, label, source) VALUES (?, ?, ?, 'nickname')",
                    (session_id, user_id, sender_name),
                )
            self.conn.execute("INSERT OR IGNORE INTO session_state (session_id, stance, turn_count, last_active) VALUES (?, 'open', 0, ?)", (session_id, timestamp))
            self.conn.execute("UPDATE session_state SET turn_count = turn_count + 1, last_active = ? WHERE session_id = ?", (timestamp, session_id))
            self.conn.commit()
            try:
                # WAL 超过 200KB 时强制 TRUNCATE，否则 PASSIVE
                wal_path = getattr(self, 'db_path', '') + '-wal'
                if wal_path and os.path.isfile(wal_path) and os.path.getsize(wal_path) > 200 * 1024:
                    self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                else:
                    self.conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
            except Exception:
                pass
        finally:
            if lock:
                try:
                    lock.release()
                except Exception:
                    pass

    def find_mentioned_users(self, session_id: str, text: str, speaker_id: str = "") -> list[tuple[str, str]]:
        """扫描消息文本，返回 (user_id, label)
        优先匹配当前 session，无结果时回退到发言者参与过的其他 session"""
        rows = self.conn.execute(
            "SELECT user_id, label FROM user_mappings WHERE session_id = ?",
            (session_id,),
        ).fetchall()

        result = []
        seen = set()
        for uid, label in rows:
            if label and label in text and uid not in seen:
                seen.add(uid)
                result.append((uid, label))

        # 当前 session 无匹配 → 查发言者参与过的其他 session
        if not result and speaker_id:
            rows = self.conn.execute("""
                SELECT DISTINCT um.user_id, um.label
                FROM user_mappings um
                WHERE um.session_id IN (
                    SELECT session_id FROM user_mappings WHERE user_id = ?
                )
                AND um.user_id != ?
            """, (speaker_id, speaker_id)).fetchall()
            for uid, label in rows:
                if label and label in text and uid not in seen:
                    seen.add(uid)
                    result.append((uid, label))

        return result

    def bind_user_label(self, session_id: str, user_id: str, label: str, source: str = 'custom'):
        """手动绑定用户自定义标识（别名、工号等），覆盖已有同源标签"""
        now_str = __import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.conn.execute(
            "INSERT OR REPLACE INTO user_mappings (session_id, user_id, label, source, created_at) VALUES (?, ?, ?, ?, ?)",
            (session_id, user_id, label, source, now_str),
        )
        self.conn.commit()

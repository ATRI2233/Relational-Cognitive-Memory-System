"""MinimalRCMS - 最小化关系认知记忆系统 (MVP)

使用方式:
    from backends import LLMBackend, MockBackend
    from minimal_rcms import MinimalRCMS

    rcms = MinimalRCMS()
    backend = MockBackend()
    reply = await rcms.chat("user_1", "session_1", "你好", backend)
"""
import re
import sqlite3
from datetime import datetime

from backends import LLMBackend


class MinimalRCMS:
    def __init__(self, db_path="memory.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self._init_db()

    def _init_db(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS long_term_memory (
                id INTEGER PRIMARY KEY,
                user_id TEXT,
                content TEXT,
                memory_type TEXT CHECK(memory_type IN ('event', 'impression')),
                session_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS session_state (
                session_id TEXT PRIMARY KEY,
                user_id TEXT,
                stance TEXT DEFAULT 'casual',
                mood REAL DEFAULT 0,
                focus_topic TEXT,
                turn_count INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY,
                session_id TEXT,
                role TEXT,
                content TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        self.conn.commit()

    def close(self):
        self.conn.close()

    # ========== Step 1: 判氛围 ==========
    def detect_stance(self, user_input: str) -> str:
        emotional_words = ['累', '烦', '难过', '开心', '怕', '想', '为什么', '怎么办',
                           '焦虑', '迷茫', '失望', '生气', '感动', '孤独', '压力']
        has_emotion = any(w in user_input for w in emotional_words)
        is_question = '?' in user_input or '？' in user_input
        is_long = len(user_input) > 20
        return 'engaged' if (has_emotion or is_question or is_long) else 'casual'

    # ========== Step 2: 查记忆 ==========
    def retrieve_memories(self, user_id: str, user_input: str, stance: str, limit: int = 2):
        if stance == 'casual':
            return []

        tokens = re.split(r'[\s,，。！？、；：""''（）()—\n]+', user_input)
        keywords = [w for w in tokens if len(w) > 1][:3]
        if not keywords:
            return []

        conditions = ' OR '.join(['content LIKE ?'] * len(keywords))
        params = [f'%{k}%' for k in keywords] + [user_id]

        cursor = self.conn.execute(f"""
            SELECT content, memory_type, created_at
            FROM long_term_memory
            WHERE ({conditions}) AND user_id = ?
            ORDER BY created_at DESC LIMIT ?
        """, params + [limit])

        return [(self._fuzz_time(r[2]) + '，' + r[0], r[1]) for r in cursor.fetchall()]

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

    def _get_history(self, session_id: str, limit: int = 3):
        rows = self.conn.execute("""
            SELECT role, content FROM chat_history
            WHERE session_id = ? ORDER BY created_at DESC LIMIT ?
        """, (session_id, limit)).fetchall()
        rows.reverse()
        return rows

    # ========== Step 3: 构建 Prompt ==========
    def build_prompt(self, user_id: str, session_id: str, user_input: str):
        stance = self.detect_stance(user_input)
        memories = self.retrieve_memories(user_id, user_input, stance)

        state = self.conn.execute(
            "SELECT stance, mood, turn_count FROM session_state WHERE session_id = ?",
            (session_id,)
        ).fetchone()

        if not state:
            mood = 0.0
            turn_count = 0
            self.conn.execute(
                "INSERT INTO session_state (session_id, user_id, stance, mood, turn_count) VALUES (?, ?, ?, ?, ?)",
                (session_id, user_id, stance, mood, turn_count)
            )
        else:
            mood = state[1]
            turn_count = state[2]

        history = self._get_history(session_id)
        history_text = "\n".join([f"{h[0]}: {h[1]}" for h in history]) if history else "（新对话）"

        memory_text = ""
        if memories:
            memory_text = "\n".join([f"- 你记得{m[0]}" for m in memories])
        else:
            memory_text = "- 没什么特别的联想"

        atmosphere = (
            "你现在随手回消息，不太走心，像朋友边刷手机边打字。"
            if stance == 'casual'
            else "你现在认真听他说话，可以想起以前的事，可以共情。"
        )
        mood_desc = '比较松' if mood > -0.3 else '有点沉'

        prompt = f"""【你是谁】
你是一个在网上认识很久的朋友。说话偏短，有留白，不堆术语。不主动说教，不假装完全理解。疲惫时会简短，但不会冷漠。

【当前气氛】
{atmosphere}
你们已经聊了{turn_count}轮。你现在整体感觉{mood_desc}。

【相关记忆】
{memory_text}

【最近对话】
{history_text}

用户: {user_input}
你:"""

        return prompt, stance

    # ========== Step 4: 保存与更新 ==========
    def save_turn(self, session_id: str, user_input: str, agent_reply: str, stance: str):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        self.conn.execute(
            "INSERT INTO chat_history (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (session_id, 'user', user_input, timestamp)
        )
        self.conn.execute(
            "INSERT INTO chat_history (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (session_id, 'assistant', agent_reply, timestamp)
        )

        self.conn.execute("""
            UPDATE session_state
            SET turn_count = turn_count + 1, stance = ?
            WHERE session_id = ?
        """, (stance, session_id))

        if stance == 'engaged' and len(user_input) > 10:
            recent = self.conn.execute("""
                SELECT content FROM long_term_memory
                WHERE session_id = ? AND created_at > datetime('now', '-1 hour')
            """, (session_id,)).fetchone()

            if not recent:
                summary = user_input[:50] + "..." if len(user_input) > 50 else user_input
                user_id_row = self.conn.execute(
                    "SELECT user_id FROM session_state WHERE session_id = ?", (session_id,)
                ).fetchone()
                user_id = user_id_row[0] if user_id_row else 'default'
                self.conn.execute(
                    "INSERT INTO long_term_memory (user_id, content, memory_type, session_id, created_at) VALUES (?, ?, ?, ?, ?)",
                    (user_id, summary, 'event', session_id, timestamp)
                )

        self.conn.commit()

    # ========== 主入口（异步） ==========
    async def chat(self, user_id: str, session_id: str, user_input: str, backend: LLMBackend) -> str:
        """主入口: 构建 prompt → 调用 LLM → 保存本轮"""
        prompt, stance = self.build_prompt(user_id, session_id, user_input)
        reply = await backend.generate(prompt)
        self.save_turn(session_id, user_input, reply, stance)
        return reply

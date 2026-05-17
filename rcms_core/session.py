import logging
from datetime import datetime

logger = logging.getLogger("rcms")


class SessionMixin:
    """会话状态、残量衰减、save_turn"""

    def _load_residue(self, session_id: str) -> tuple:
        row = self.conn.execute(
            "SELECT residue_warmth, residue_tension FROM session_state WHERE session_id = ?",
            (session_id,)
        ).fetchone()
        return (row[0] or 0.0, row[1] or 0.0) if row else (0.0, 0.0)

    def _decay_residue(self, session_id: str):
        warmth, tension = self._load_residue(session_id)
        warmth *= self._RESIDUE_DECAY
        tension *= self._RESIDUE_DECAY
        if abs(warmth) < 0.01:
            warmth = 0.0
        if abs(tension) < 0.01:
            tension = 0.0
        self.conn.execute(
            "UPDATE session_state SET residue_warmth = ?, residue_tension = ? WHERE session_id = ?",
            (warmth, tension, session_id)
        )

    def _write_residue(self, session_id: str, warmth_delta: float, tension_delta: float):
        cw, ct = self._load_residue(session_id)
        self.conn.execute(
            "UPDATE session_state SET residue_warmth = ?, residue_tension = ? WHERE session_id = ?",
            (max(-1.0, min(1.0, cw + warmth_delta)), max(-1.0, min(1.0, ct + tension_delta)), session_id)
        )

    def _apply_residue(self, depth: float, energy: float, session_id: str) -> tuple:
        rw, rt = self._load_residue(session_id)
        if abs(rw) > 0.01:
            energy += rw * 0.15
        if abs(rt) > 0.01:
            depth += rt * 0.10
        return (max(0.0, min(1.0, depth)), max(-1.0, min(1.0, energy)))

    def save_turn(self, session_id: str, user_input: str, agent_reply: str, stance: str):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.conn.execute("INSERT INTO chat_history (session_id, role, content, created_at) VALUES (?, ?, ?, ?)", (session_id, 'user', user_input, timestamp))
        self.conn.execute("INSERT INTO chat_history (session_id, role, content, created_at) VALUES (?, ?, ?, ?)", (session_id, 'assistant', agent_reply, timestamp))
        self.conn.execute("INSERT OR IGNORE INTO session_state (session_id, stance, turn_count, last_active) VALUES (?, 'open', 0, ?)", (session_id, timestamp))
        self.conn.execute("UPDATE session_state SET turn_count = turn_count + 1, stance = ?, last_active = ? WHERE session_id = ?", (stance, timestamp, session_id))
        self.conn.commit()

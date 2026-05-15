"""MemoryMixin — 长期 5 层记忆 / Residue / Post-Update"""
import json
import re
from datetime import datetime


class MemoryMixin:

    _ARC_STAGES = ['stranger', 'familiar', 'rapport', 'history', 'drift', 'reconnect']
    _RESIDUE_DECAY = 0.6

    # ── Silent Recall Residue ──

    def _load_residue(self, session_id: str) -> tuple:
        row = self.conn.execute(
            "SELECT residue_warmth, residue_tension FROM session_state WHERE session_id = ?",
            (session_id,)
        ).fetchone()
        if row:
            return (row[0] or 0.0, row[1] or 0.0)
        return (0.0, 0.0)

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
        current_warmth, current_tension = self._load_residue(session_id)
        new_warmth = max(-1.0, min(1.0, current_warmth + warmth_delta))
        new_tension = max(-1.0, min(1.0, current_tension + tension_delta))
        self.conn.execute(
            "UPDATE session_state SET residue_warmth = ?, residue_tension = ? WHERE session_id = ?",
            (new_warmth, new_tension, session_id)
        )

    def _apply_residue(self, momentum: tuple, session_id: str) -> tuple:
        depth, energy = momentum
        rw, rt = self._load_residue(session_id)
        if abs(rw) > 0.01:
            energy += rw * 0.15
        if abs(rt) > 0.01:
            depth += rt * 0.10
        depth = max(0.0, min(1.0, depth))
        energy = max(-1.0, min(1.0, energy))
        return (depth, energy)

    # ── Long-term 5 Layers ──

    def _init_identity(self, user_id: str):
        now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.conn.execute(
            "INSERT OR IGNORE INTO identity_memory (user_id, traits, voice_hint, updated_at) VALUES (?, '[]', '', ?)",
            (user_id, now_str)
        )
        self.conn.execute(
            "INSERT OR IGNORE INTO relationship_arc (user_id, stage, stage_score, updated_at) VALUES (?, 'stranger', 0.0, ?)",
            (user_id, now_str)
        )
        self.conn.commit()

    def _write_event_memory(self, user_id: str, session_id: str, content: str,
                             relationship_delta: float, emotional_weight: float):
        now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        compressed = content[:40] + '...' if len(content) > 40 else content
        self.conn.execute(
            "INSERT INTO event_memory (user_id, content, relationship_delta, emotional_weight, novelty, compressed_hint, created_at) "
            "VALUES (?, ?, ?, ?, 0.0, ?, ?)",
            (user_id, content, relationship_delta, emotional_weight, compressed, now_str)
        )
        self.conn.commit()

    def _write_emotional_trace(self, user_id: str, engagement: dict, momentum: tuple):
        depth, energy = momentum
        scores = engagement.get('scores', {})
        now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        warmth = 1.0 - abs(energy)
        tension = max(0.0, energy) if energy > 0 else 0.0
        uncertainty = scores.get('shift', 0.0) if scores.get('shift', 0) > 0.5 else 0.0
        distance = -depth if depth > 0.3 else depth
        self.conn.execute(
            "INSERT INTO emotional_trace (user_id, warmth, tension, uncertainty, distance, prose_hint, created_at) "
            "VALUES (?, ?, ?, ?, '', ?, ?)",
            (user_id, warmth, tension, uncertainty, distance, now_str)
        )
        self.conn.commit()

    def _update_relationship_arc(self, user_id: str, engagement: dict, depth: float):
        stage_row = self.conn.execute(
            "SELECT stage, stage_score FROM relationship_arc WHERE user_id = ?",
            (user_id,)
        ).fetchone()
        if not stage_row:
            return
        stage, score = stage_row
        now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        delta = 0.0
        if engagement['level'] == 'engaged_candidate':
            delta = 0.1 + depth * 0.1
        elif engagement['level'] == 'attentive':
            delta = 0.05
        new_score = score + delta
        new_stage = stage
        thresholds = {'stranger': 4.0, 'familiar': 10.0, 'rapport': 20.0, 'history': 35.0, 'drift': 0.0}
        if stage == 'stranger' and new_score >= thresholds['stranger']:
            new_stage = 'familiar'
        elif stage == 'familiar' and new_score >= thresholds['familiar']:
            new_stage = 'rapport'
        elif stage == 'rapport' and new_score >= thresholds['rapport']:
            new_stage = 'history'
        self.conn.execute(
            "UPDATE relationship_arc SET stage = ?, stage_score = ?, updated_at = ? WHERE user_id = ?",
            (new_stage, new_score, now_str, user_id)
        )
        self.conn.commit()

    def _build_shared_context(self, user_id: str, user_input: str, reply: str):
        tokens = re.split(r'[\s,，。！？、；：""''（）()—\n]+', user_input)
        kws = [w for w in tokens if len(w) > 2 and w not in self._TRIVIAL_MARKERS]
        if not kws:
            return
        kw = kws[0]
        existing = self.conn.execute(
            "SELECT context_id, omission_count FROM shared_context WHERE user_id = ? AND context_body LIKE ?",
            (user_id, f'%{kw}%')
        ).fetchone()
        if existing:
            self.conn.execute(
                "UPDATE shared_context SET omission_count = omission_count + 1 WHERE context_id = ?",
                (existing[0],)
            )
        else:
            self.conn.execute(
                "INSERT INTO shared_context (user_id, context_body, omission_count, confirmed) VALUES (?, ?, 1, 0)",
                (user_id, kw)
            )
        self.conn.commit()

    def _load_long_term_context(self, user_id: str) -> dict:
        identity = self.conn.execute(
            "SELECT traits, voice_hint FROM identity_memory WHERE user_id = ?",
            (user_id,)
        ).fetchone()
        recent_events = self.conn.execute(
            "SELECT compressed_hint, relationship_delta FROM event_memory WHERE user_id = ? ORDER BY created_at DESC LIMIT 2",
            (user_id,)
        ).fetchall()
        recent_trace = self.conn.execute(
            "SELECT prose_hint, warmth, tension FROM emotional_trace WHERE user_id = ? ORDER BY created_at DESC LIMIT 1",
            (user_id,)
        ).fetchone()
        arc = self.conn.execute(
            "SELECT stage, stage_score FROM relationship_arc WHERE user_id = ?",
            (user_id,)
        ).fetchone()
        shared = self.conn.execute(
            "SELECT context_body FROM shared_context WHERE user_id = ? AND confirmed = 1 ORDER BY omission_count DESC LIMIT 2",
            (user_id,)
        ).fetchall()
        return {
            'identity_traits': json.loads(identity[0]) if identity and identity[0] else [],
            'voice_hint': identity[1] if identity else '',
            'events': [{'hint': r[0], 'delta': r[1]} for r in recent_events],
            'trace': {'prose': recent_trace[0] if recent_trace else '',
                      'warmth': recent_trace[1] if recent_trace else 0.0,
                      'tension': recent_trace[2] if recent_trace else 0.0},
            'arc_stage': arc[0] if arc else 'stranger',
            'arc_score': arc[1] if arc else 0.0,
            'shared_contexts': [r[0] for r in shared],
        }

    # ── Post-Update ──

    def _post_update(self, user_id: str, session_id: str, user_input: str,
                     stance: str, engagement: dict, momentum: tuple,
                     reply: str = ""):
        depth, energy = momentum
        now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        self._init_identity(user_id)

        self.conn.execute(
            "UPDATE session_state SET last_active = ? WHERE session_id = ?",
            (now_str, session_id)
        )

        self._decay_residue(session_id)
        if engagement['level'] == 'engaged_candidate':
            warmth_delta = (1.0 - abs(energy)) * (1.0 if energy >= 0 else -0.5)
            tension_delta = min(abs(energy) * 0.6, 0.5)
            self._write_residue(session_id, warmth_delta, tension_delta)

        if engagement['level'] in ('engaged_candidate', 'attentive'):
            self._write_emotional_trace(user_id, engagement, momentum)

        self._update_relationship_arc(user_id, engagement, depth)

        if reply:
            self._build_shared_context(user_id, user_input, reply)

        should_write = (
            (engagement['level'] == 'engaged_candidate')
            or (engagement['level'] == 'attentive' and len(user_input) > 15)
        )
        if should_write:
            recent = self.conn.execute("""
                SELECT content FROM long_term_memory
                WHERE session_id = ? ORDER BY created_at DESC LIMIT 1
            """, (session_id,)).fetchone()
            summary = user_input[:50] + "..." if len(user_input) > 50 else user_input
            if not recent or recent[0] != summary:
                self.conn.execute(
                    "INSERT INTO long_term_memory (user_id, content, memory_type, session_id, created_at) VALUES (?, ?, ?, ?, ?)",
                    (user_id, summary, 'event', session_id, now_str)
                )
                self._build_graph_from_memory(user_id, summary)

                if depth > 0.5 and engagement['level'] == 'engaged_candidate':
                    self._write_event_memory(
                        user_id, session_id, summary,
                        relationship_delta=depth * 0.5,
                        emotional_weight=abs(energy)
                    )

        self.conn.commit()

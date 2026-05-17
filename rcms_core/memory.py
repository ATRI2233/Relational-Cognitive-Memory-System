import json
import logging
import re
from datetime import datetime

logger = logging.getLogger("rcms")


class MemoryMixin:
    """长期记忆：identity / events / relationship / shared_context / graph builder"""

    def _init_identity(self, user_id: str):
        now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.conn.execute("INSERT OR IGNORE INTO identity_memory (user_id, traits, voice_hint, updated_at) VALUES (?, '[]', '', ?)", (user_id, now_str))
        self.conn.execute("INSERT OR IGNORE INTO relationship_arc (user_id, stage, stage_score, updated_at) VALUES (?, 'stranger', 0.0, ?)", (user_id, now_str))
        self.conn.commit()

    def _update_relationship_arc(self, user_id: str, level: str):
        row = self.conn.execute("SELECT stage, stage_score FROM relationship_arc WHERE user_id = ?", (user_id,)).fetchone()
        if not row:
            return
        stage, score = row
        new_score = score + (0.05 if level == 'engaged_candidate' else 0.02)
        new_stage = stage
        thresholds = {'stranger': 4.0, 'familiar': 10.0, 'rapport': 20.0, 'history': 35.0}
        if stage == 'stranger' and new_score >= thresholds['stranger']:
            new_stage = 'familiar'
        elif stage == 'familiar' and new_score >= thresholds['familiar']:
            new_stage = 'rapport'
        elif stage == 'rapport' and new_score >= thresholds['rapport']:
            new_stage = 'history'
        self.conn.execute("UPDATE relationship_arc SET stage = ?, stage_score = ?, updated_at = ? WHERE user_id = ?",
                          (new_stage, new_score, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), user_id))
        self.conn.commit()

    def _build_shared_context(self, user_id: str, user_input: str, reply: str):
        tokens = re.split(r'[\s,，。！？、；：""''（）()—\n]+', user_input)
        kws = [w for w in tokens if len(w) > 2 and w not in self._TRIVIAL_MARKERS]
        if not kws:
            return
        kw = kws[0]
        existing = self.conn.execute("SELECT context_id, omission_count FROM shared_context WHERE user_id = ? AND context_body LIKE ?",
                                      (user_id, f'%{kw}%')).fetchone()
        if existing:
            self.conn.execute("UPDATE shared_context SET omission_count = omission_count + 1 WHERE context_id = ?", (existing[0],))
        else:
            self.conn.execute("INSERT INTO shared_context (user_id, context_body, omission_count, confirmed) VALUES (?, ?, 1, 0)", (user_id, kw))
        self.conn.commit()

    def _load_long_term_context(self, user_id: str) -> dict:
        identity = self.conn.execute("SELECT traits, voice_hint FROM identity_memory WHERE user_id = ?", (user_id,)).fetchone()
        recent_events = self.conn.execute("SELECT summary, importance FROM cognitive_distill WHERE user_id = ? AND summary IS NOT NULL ORDER BY created_at DESC LIMIT 2", (user_id,)).fetchall()
        recent_trace = self.conn.execute("SELECT prose_hint, warmth, tension FROM emotional_trace WHERE user_id = ? ORDER BY created_at DESC LIMIT 1", (user_id,)).fetchone()
        arc = self.conn.execute("SELECT stage, stage_score FROM relationship_arc WHERE user_id = ?", (user_id,)).fetchone()
        shared = self.conn.execute("SELECT context_body FROM shared_context WHERE user_id = ? AND confirmed = 1 ORDER BY omission_count DESC LIMIT 4", (user_id,)).fetchall()
        entities = self.conn.execute("SELECT entity_name, relation_type, property FROM entity_relations WHERE user_id = ? ORDER BY mention_count DESC LIMIT 5", (user_id,)).fetchall()
        raw_traits = json.loads(identity[0]) if identity and identity[0] else []
        trait_details = []
        for item in raw_traits:
            if isinstance(item, str):
                trait_details.append({"text": item, "strength": 3})
            elif isinstance(item, dict):
                trait_details.append({"text": item.get("t", ""), "strength": item.get("s", 0), "count": item.get("c", 0)})
        trait_details = [p for p in trait_details if p["text"] and p["strength"] > 0]
        return {
            'identity_traits': [p["text"] for p in trait_details],
            'trait_details': trait_details,
            'voice_hint': identity[1] if identity else '',
            'events': [{'hint': r[0], 'delta': 1 if r[1] and r[1] > 0.5 else 0} for r in recent_events],
            'trace': {'prose': recent_trace[0] if recent_trace else '', 'warmth': recent_trace[1] if recent_trace else 0.0, 'tension': recent_trace[2] if recent_trace else 0.0},
            'arc_stage': arc[0] if arc else 'stranger', 'arc_score': arc[1] if arc else 0.0,
            'shared_contexts': [r[0] for r in shared],
            'entities': [{'name': r[0], 'relation': r[1], 'fact': r[2]} for r in entities],
        }

    def _post_update(self, user_id: str, session_id: str, user_input: str, stance: str, reply: str = ""):
        now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self._init_identity(user_id)
        self.conn.execute("UPDATE session_state SET last_active = ? WHERE session_id = ?", (now_str, session_id))
        self._decay_residue(session_id)
        self._update_relationship_arc(user_id, 'attentive')
        if reply:
            self._build_shared_context(user_id, user_input, reply)
        if reply and len(user_input) > 15:
            summary = f"{user_input[:80]} → {reply[:80]}"
            recent = self.conn.execute("SELECT content FROM cognitive_distill WHERE session_id = ? ORDER BY created_at DESC LIMIT 1", (session_id,)).fetchone()
            if not recent or recent[0] != summary:
                self.conn.execute(
                    "INSERT INTO cognitive_distill (user_id, session_id, content, importance, created_at) VALUES (?, ?, ?, ?, ?)",
                    (user_id, session_id, summary, 0.3, now_str),
                )
                self._build_graph_from_memory(user_id, summary)
        # 双条件蒸馏触发：轮数或时间，先到先触发
        self._maybe_distill(user_id, session_id)
        self.conn.commit()

    def _maybe_distill(self, user_id: str, session_id: str):
        """双条件蒸馏触发：轮数或时间，先到先触发"""
        row = self.conn.execute(
            "SELECT turn_count, last_distill_turn, last_distill_at FROM session_state WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if not row:
            return
        turn_count, last_turn, last_at = row[0] or 0, row[1] or 0, row[2]
        max_turns = getattr(self, '_DISTILL_MAX_TURNS', 50)
        max_minutes = getattr(self, '_DISTILL_MAX_MINUTES', 120)
        triggered = False
        if turn_count - last_turn >= max_turns:
            triggered = True
        if last_at:
            elapsed = (datetime.now() - datetime.fromisoformat(str(last_at))).total_seconds() / 60
            if elapsed >= max_minutes:
                triggered = True
        if not triggered:
            return
        # 蒸馏：合并上次蒸馏以来的条目
        since_turn = self.conn.execute(
            "SELECT MIN(turn_num) FROM cognitive_distill WHERE session_id = ? AND id > COALESCE((SELECT MAX(id) FROM cognitive_distill WHERE session_id = ? AND summary IS NOT NULL AND importance >= 0.5), 0)",
            (session_id, session_id),
        ).fetchone()[0]
        if since_turn:
            snapshot = self.conn.execute(
                "SELECT content, mood FROM cognitive_distill WHERE session_id = ? AND id >= ? ORDER BY id",
                (session_id, since_turn),
            ).fetchall()
        else:
            snapshot = []
        if len(snapshot) < 3:
            return  # 条目太少，等下次
        # 合并为一条蒸馏摘要
        lines = [s[0] for s in snapshot if s[0]]
        if not lines:
            return
        body = " | ".join(lines[:10])
        now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.conn.execute(
            "INSERT INTO cognitive_distill (user_id, session_id, content, summary, importance, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (user_id, session_id, f"[蒸馏] {turn_count}轮对话摘要", body, 0.7, now_str),
        )
        self.conn.execute(
            "UPDATE session_state SET last_distill_turn = ?, last_distill_at = ? WHERE session_id = ?",
            (turn_count, now_str, session_id),
        )
        logger.info(f"RCMS: 蒸馏触发 user={user_id} session={session_id} turn={turn_count} entries={len(lines)}")

    def _build_graph_from_memory(self, user_id: str, content: str):
        kws = self._extract_keywords(content, max_kw=8)
        if len(kws) < 2:
            return
        now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        node_ids = []
        for kw in kws:
            row = self.conn.execute("SELECT node_id FROM memory_graph_nodes WHERE user_id = ? AND label = ?", (user_id, kw)).fetchone()
            if row:
                self.conn.execute("UPDATE memory_graph_nodes SET freq = freq + 1, last_seen = ? WHERE node_id = ?", (now_str, row[0]))
                node_ids.append(row[0])
            else:
                cur = self.conn.execute("INSERT INTO memory_graph_nodes (user_id, label, freq, last_seen) VALUES (?, ?, 1, ?)", (user_id, kw, now_str))
                node_ids.append(cur.lastrowid)
        for i in range(len(node_ids)):
            for j in range(i + 1, len(node_ids)):
                a, b = sorted((node_ids[i], node_ids[j]))
                edge = self.conn.execute("SELECT weight, encounter_count FROM memory_graph_edges WHERE from_node_id = ? AND to_node_id = ?", (a, b)).fetchone()
                if edge:
                    self.conn.execute("UPDATE memory_graph_edges SET weight = weight + 0.5, encounter_count = encounter_count + 1, last_seen = ? WHERE from_node_id = ? AND to_node_id = ?", (now_str, a, b))
                else:
                    self.conn.execute("INSERT INTO memory_graph_edges (from_node_id, to_node_id, weight, encounter_count, last_seen) VALUES (?, ?, 1.0, 1, ?)", (a, b, now_str))
        self.conn.commit()

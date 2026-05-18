import json
import logging
from datetime import datetime

logger = logging.getLogger("rcms")


class MemoryMixin:
    """长期记忆：identity / events / distill / graph builder"""

    def _init_identity(self, user_id: str):
        now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.conn.execute("INSERT OR IGNORE INTO identity_memory (user_id, traits, updated_at) VALUES (?, '[]', ?)", (user_id, now_str))
        self.conn.commit()

    def _load_long_term_context(self, user_id: str) -> dict:
        identity = self.conn.execute(
            "SELECT traits, preferences, communication_style, self_identity, boundaries, core_identity FROM identity_memory WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        recent_events = self.conn.execute("SELECT summary, importance FROM cognitive_distill WHERE user_id = ? AND summary IS NOT NULL ORDER BY created_at DESC LIMIT 2", (user_id,)).fetchall()
        entities = self.conn.execute("""
            SELECT n1.label, e.relation, n2.label
            FROM memory_graph_edges e
            JOIN memory_graph_nodes n1 ON e.from_node_id = n1.node_id
            JOIN memory_graph_nodes n2 ON e.to_node_id = n2.node_id
            WHERE n1.user_id = ? AND e.relation != ''
            ORDER BY e.weight DESC LIMIT 5
        """, (user_id,)).fetchall()
        shared_rows = self.conn.execute(
            "SELECT context_body FROM shared_context WHERE user_id = ? ORDER BY context_id DESC LIMIT 4",
            (user_id,),
        ).fetchall()
        raw_traits = json.loads(identity[0]) if identity and identity[0] else []
        trait_details = []
        for item in raw_traits:
            if isinstance(item, str):
                trait_details.append({"text": item, "strength": 3})
            elif isinstance(item, dict):
                trait_details.append({"text": item.get("t", ""), "strength": item.get("s", 0), "count": item.get("c", 0)})
        trait_details = [p for p in trait_details if p["text"] and p["strength"] > 0]
        # 结构化身份字段
        def _safe_json(val, default):
            if not val:
                return default
            try:
                return json.loads(val)
            except Exception:
                return default

        return {
            'identity_traits': [p["text"] for p in trait_details],
            'trait_details': trait_details,
            'preferences': _safe_json(identity[1], {}) if identity else {},
            'communication_style': identity[2] if identity and identity[2] else '',
            'self_identity': _safe_json(identity[3], []) if identity else [],
            'boundaries': _safe_json(identity[4], []) if identity else [],
            'core_identity': _safe_json(identity[5], {}) if identity else {},
            'entities': [{'name': r[0], 'relation': r[1], 'fact': r[2]} for r in entities],
            'shared_contexts': [r[0] for r in shared_rows],
        }

    async def post_update_rules(self, user_id: str, session_id: str, user_input: str, stance: str, reply: str = ""):
        """纯规则事后更新（不含 LLM 蒸馏触发）"""
        now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self._init_identity(user_id)
        self.conn.execute("UPDATE session_state SET last_active = ? WHERE session_id = ?", (now_str, session_id))
        # 悬案自动过期：超过 _DANGLING_EXPIRE_TURNS 轮无人提起则归档
        dt_row = self.conn.execute(
            "SELECT dangling_threads, turn_count FROM session_state WHERE session_id = ?", (session_id,)
        ).fetchone()
        if dt_row and dt_row[0]:
            try:
                dt_data = json.loads(dt_row[0])
                if isinstance(dt_data, dict) and dt_data.get("threads"):
                    since_turn = dt_data.get("turn", 0)
                    current_turn = dt_row[1] or 0
                    expire = getattr(self, '_DANGLING_EXPIRE_TURNS', 15)
                    if current_turn - since_turn >= expire:
                        self._archive_dangling(user_id, session_id, now_str, reason="过期")
            except Exception:
                pass
        new_rule_id = None
        if reply and len(user_input) > 15:
            summary = f"{user_input[:80]} → {reply[:80]}"
            recent = self.conn.execute("SELECT content FROM cognitive_distill WHERE session_id = ? ORDER BY created_at DESC LIMIT 1", (session_id,)).fetchone()
            if not recent or recent[0] != summary:
                self.conn.execute(
                    "INSERT INTO cognitive_distill (user_id, session_id, content, importance, created_at) VALUES (?, ?, ?, ?, ?)",
                    (user_id, session_id, summary, 0.3, now_str),
                )
                new_rule_id = self.conn.execute("SELECT last_insert_rowid()").fetchone()[0]
                self._build_graph_from_memory(user_id, summary)
        self.conn.commit()
        # 规则摘要也需要向量，否则向量检索通道永远找不到它们
        if new_rule_id and summary:
            vec = await self._get_embedding(summary[:512])
            if vec:
                self._store_embedding(user_id, new_rule_id, vec)

    def check_distill_needed(self, session_id: str) -> tuple:
        """检查是否需要触发蒸馏。返回 (triggered, last_turn, turn_count, snapshot_text)"""
        row = self.conn.execute(
            "SELECT turn_count, last_distill_turn, last_distill_at FROM session_state WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if not row:
            return (False, 0, 0, "")
        turn_count, last_turn, last_at = row[0] or 0, row[1] or 0, row[2]
        if turn_count == 0:
            return (False, 0, 0, "")
        max_turns = getattr(self, '_DISTILL_MAX_TURNS', 30)
        max_minutes = getattr(self, '_DISTILL_MAX_MINUTES', 60)
        triggered = False
        if turn_count - last_turn >= max_turns:
            triggered = True
        if not triggered and last_at:
            elapsed = (datetime.now() - datetime.fromisoformat(str(last_at))).total_seconds() / 60
            if elapsed >= max_minutes:
                triggered = True
        if not triggered:
            return (False, last_turn, turn_count, "")
        # 读取本轮次以来的 chat_history 作为快照（至少 3 轮对话 = 6 行）
        rows = self.conn.execute(
            "SELECT role, content FROM chat_history WHERE session_id = ? AND turn_num > ? AND turn_num <= ? ORDER BY turn_num, id",
            (session_id, last_turn, turn_count),
        ).fetchall()
        if len(rows) < 6:
            return (False, last_turn, turn_count, "")
        lines = [f"{r[0]}: {r[1][:200]}" for r in rows]
        snapshot_text = "\n".join(lines[:30])
        return (True, last_turn, turn_count, snapshot_text)

    async def _apply_distill(self, user_id: str, session_id: str, last_turn: int, turn_count: int, summary: str):
        """写入 LLM 蒸馏摘要 + 过期清理 + 低重要性碎片清理 + 图谱维护"""
        now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        # 1. 写入蒸馏摘要
        self.conn.execute(
            "INSERT INTO cognitive_distill (user_id, session_id, content, summary, importance, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (user_id, session_id, summary, summary[:80], 0.8, now_str),
        )
        distill_id = self.conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        # 2. 更新 last_distill_turn/at
        self.conn.execute(
            "UPDATE session_state SET last_distill_turn = ?, last_distill_at = ? WHERE session_id = ?",
            (turn_count, now_str, session_id),
        )
        # 3. 悬案归档
        self._archive_dangling(user_id, session_id, now_str, reason="蒸馏")
        # 3b. 清理已过期的 transient 记忆（不受 importance 限制）
        expired = self.conn.execute(
            "DELETE FROM cognitive_distill WHERE user_id = ? AND expires_at IS NOT NULL AND expires_at <= ?",
            (user_id, now_str),
        ).rowcount
        if expired:
            logger.info(f"RCMS: 已清理 {expired} 条过期记忆 user={user_id}")
        # 4. 规则摘要归并：保留最新 KEEP_RULE_SUMMARY 条 importance=0.3 的碎片
        KEEP_RULE_SUMMARY = 10
        self.conn.execute("""
            DELETE FROM cognitive_distill WHERE user_id = ? AND importance = 0.3
            AND id NOT IN (
                SELECT id FROM cognitive_distill
                WHERE user_id = ? AND importance = 0.3
                ORDER BY created_at DESC LIMIT ?
            )
        """, (user_id, user_id, KEEP_RULE_SUMMARY))
        # 5. 图谱维护：共现边衰减 + 孤立节点清理
        self._maintain_graph(user_id)
        self.conn.commit()
        # 蒸馏摘要也需要向量
        vec = await self._get_embedding(summary[:512])
        if vec:
            self._store_embedding(user_id, distill_id, vec)
        logger.info(f"RCMS: 蒸馏完成 user={user_id} session={session_id} turn={turn_count} summary={summary[:60]}")

    def _maintain_graph(self, user_id: str):
        """图衰减与清理：共现边降权删除 + 孤立节点清理"""
        # 共现边 weight 衰减 0.8，低于 0.3 的删除
        self.conn.execute("""
            UPDATE memory_graph_edges SET weight = ROUND(weight * 0.8, 2)
            WHERE from_node_id IN (SELECT node_id FROM memory_graph_nodes WHERE user_id = ?)
        """, (user_id,))
        dead_edges = self.conn.execute("""
            DELETE FROM memory_graph_edges WHERE relation = '' AND weight < 0.3
        """).rowcount
        # 语义边（relation != ''）不做自动删除，留待 LLM 蒸馏决策
        # 孤立节点清理（无任何边连接的节点）
        orphan_nodes = self.conn.execute("""
            DELETE FROM memory_graph_nodes WHERE user_id = ? AND node_id NOT IN (
                SELECT from_node_id FROM memory_graph_edges
                UNION
                SELECT to_node_id FROM memory_graph_edges
            )
        """, (user_id,)).rowcount
        if dead_edges or orphan_nodes:
            logger.info(f"RCMS: 图维护 user={user_id} deleted_edges={dead_edges} orphan_nodes={orphan_nodes}")

    def _archive_dangling(self, user_id: str, session_id: str, now_str: str, reason: str = ""):
        """将未结悬案归档到 cognitive_distill 并清空 session_state"""
        row = self.conn.execute(
            "SELECT dangling_threads FROM session_state WHERE session_id = ?", (session_id,)
        ).fetchone()
        if not row or not row[0]:
            return
        try:
            data = json.loads(row[0])
            if not isinstance(data, dict) or not data.get("threads"):
                return
            threads = data["threads"]
            tag = f"[悬案归档·{reason}]" if reason else "[悬案归档]"
            content = f"{tag} " + "、".join(threads[:3])
            self.conn.execute(
                "INSERT INTO cognitive_distill (user_id, session_id, content, summary, importance, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (user_id, session_id, content, content[:60], 0.7, now_str),
            )
            self.conn.execute(
                "UPDATE session_state SET dangling_threads = ? WHERE session_id = ?",
                ('[]', session_id),
            )
            logger.info(f"RCMS: dangling_threads archived ({reason}) user={user_id}")
        except Exception:
            pass

    def _build_graph_from_memory(self, user_id: str, content: str):
        kws = list(dict.fromkeys(self._extract_keywords(content, max_kw=8)))
        if len(kws) < 2:
            return
        now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        node_ids = [self._upsert_graph_node(user_id, kw, now_str) for kw in kws]
        for i in range(len(node_ids)):
            for j in range(i + 1, len(node_ids)):
                a, b = sorted((node_ids[i], node_ids[j]))
                if a != b:
                    self._upsert_graph_edge(a, b, now_str)
        self.conn.commit()

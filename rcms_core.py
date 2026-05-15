"""MinimalRCMS — 关系认知记忆系统核心（单文件）"""
import asyncio
import json
import re
import sqlite3
from datetime import datetime

from backends import LLMBackend


class MinimalRCMS:

    # ── 常量 ──

    _EMOTIONAL_WORDS = [
        '累', '烦', '难过', '开心', '怕', '为什么', '怎么办',
        '焦虑', '迷茫', '失望', '生气', '感动', '孤独', '压力',
        '崩溃', '痛苦', '幸福', '委屈', '愤怒', '绝望', '不安',
        '愧疚', '后悔', '感激', '羡慕', '厌倦', '疲惫', '心累',
        '纠结', '无助', '温暖', '讽刺', '荒谬', '崩溃', '心碎',
        '气死', '受不了', '撑不住', '扛不住', '熬不下去',
        '舍不得', '放不下', '不甘心',
    ]

    _TRIVIAL_MARKERS = ['吃', '喝', '睡', '饭', '菜', '外卖', '快递', '天气',
                        '价格', '多少钱', '购物', '买了', '电影', '追剧',
                        '洗澡', '起床', '睡觉', '游戏']

    _ARC_STAGES = ['stranger', 'familiar', 'rapport', 'history', 'drift', 'reconnect']
    _RESIDUE_DECAY = 0.6
    _GRAPH_BFS_DEPTH = 2
    _GRAPH_ACTIVATION_DECAY = 0.5
    _SURFACED_THRESHOLD = 0.6
    _SILENT_THRESHOLD = 0.25

    # ── 初始化 ──

    def __init__(self, db_path="memory.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_db()
        self._last_silent_recall = []

    def _init_db(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS long_term_memory (
                id INTEGER PRIMARY KEY, user_id TEXT, content TEXT,
                memory_type TEXT CHECK(memory_type IN ('event','impression')),
                session_id TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS session_state (
                session_id TEXT PRIMARY KEY, user_id TEXT, stance TEXT DEFAULT 'open',
                mood REAL DEFAULT 0, focus_topic TEXT, turn_count INTEGER DEFAULT 0,
                stance_turns INTEGER DEFAULT 0, engagement_level TEXT DEFAULT 'coasting',
                momentum_depth REAL DEFAULT 0.0, momentum_energy REAL DEFAULT 0.0,
                last_active TIMESTAMP, residue_warmth REAL DEFAULT 0.0,
                residue_tension REAL DEFAULT 0.0
            );
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY, session_id TEXT, role TEXT, content TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS identity_memory (
                user_id TEXT PRIMARY KEY, traits TEXT DEFAULT '[]',
                voice_hint TEXT DEFAULT '', created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS event_memory (
                event_id INTEGER PRIMARY KEY, user_id TEXT, content TEXT,
                relationship_delta REAL DEFAULT 0.0, emotional_weight REAL DEFAULT 0.0,
                novelty REAL DEFAULT 0.0, compressed_hint TEXT DEFAULT '',
                created_at TIMESTAMP, last_recalled TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS emotional_trace (
                trace_id INTEGER PRIMARY KEY, user_id TEXT, warmth REAL DEFAULT 0.0,
                tension REAL DEFAULT 0.0, uncertainty REAL DEFAULT 0.0,
                distance REAL DEFAULT 0.0, prose_hint TEXT DEFAULT '',
                created_at TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS shared_context (
                context_id INTEGER PRIMARY KEY, user_id TEXT, context_body TEXT,
                omission_count INTEGER DEFAULT 0, confirmed INTEGER DEFAULT 0,
                created_at TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS relationship_arc (
                arc_id INTEGER PRIMARY KEY, user_id TEXT,
                stage TEXT DEFAULT 'stranger', stage_score REAL DEFAULT 0.0,
                updated_at TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS memory_graph_nodes (
                node_id INTEGER PRIMARY KEY, user_id TEXT, label TEXT,
                node_type TEXT DEFAULT 'keyword', freq INTEGER DEFAULT 1,
                last_seen TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS memory_graph_edges (
                from_node_id INTEGER, to_node_id INTEGER, weight REAL DEFAULT 1.0,
                encounter_count INTEGER DEFAULT 1, last_seen TIMESTAMP,
                PRIMARY KEY (from_node_id, to_node_id)
            );
            CREATE INDEX IF NOT EXISTS idx_mgn_user_label ON memory_graph_nodes(user_id, label);
            CREATE INDEX IF NOT EXISTS idx_mge_from ON memory_graph_edges(from_node_id);
            CREATE INDEX IF NOT EXISTS idx_mge_to ON memory_graph_edges(to_node_id);
        """)
        for col in [
            "ADD COLUMN stance_turns INTEGER DEFAULT 0",
            "ADD COLUMN engagement_level TEXT DEFAULT 'coasting'",
            "ADD COLUMN momentum_depth REAL DEFAULT 0.0",
            "ADD COLUMN momentum_energy REAL DEFAULT 0.0",
            "ADD COLUMN last_active TIMESTAMP",
            "ADD COLUMN residue_warmth REAL DEFAULT 0.0",
            "ADD COLUMN residue_tension REAL DEFAULT 0.0",
        ]:
            try:
                self.conn.execute(f"ALTER TABLE session_state {col}")
            except Exception:
                pass
        self.conn.commit()

    def close(self):
        self.conn.close()

    # ── 通用文本工具 ──

    def _get_history(self, session_id: str, limit: int = 3):
        rows = self.conn.execute("""
            SELECT role, content FROM chat_history
            WHERE session_id = ? ORDER BY created_at DESC LIMIT ?
        """, (session_id, limit)).fetchall()
        rows.reverse()
        return rows

    @staticmethod
    def _chinese_bigrams(text: str) -> set:
        chars = re.findall(r'[一-鿿]', text)
        return {''.join(chars[i:i + 2]) for i in range(len(chars) - 1)}

    @staticmethod
    def _precise_kw_match(text: str, kw: str) -> bool:
        return kw in text

    @staticmethod
    def _score_markers(text: str, markers: list, per_hit: float = 0.3) -> float:
        count = sum(1 for m in markers if m in text)
        return min(count * per_hit, 1.0)

    # ── 记忆检索 ──

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
            SELECT content, memory_type, created_at FROM long_term_memory
            WHERE ({conditions}) AND user_id = ?
            ORDER BY created_at DESC LIMIT ?
        """, params + [limit])
        return [(self._fuzz_time(r[2]) + '，' + r[0], r[1]) for r in cursor.fetchall()]

    def _extract_keywords(self, text: str, max_kw: int = 5) -> list[str]:
        tokens = re.split(r'[\s,，。！？、；：""''（）()—\n]+', text)
        return [w for w in tokens if len(w) > 1 and w not in self._TRIVIAL_MARKERS][:max_kw]

    def _graph_activation_diffusion(self, user_id: str, seed_keywords: list[str]) -> list:
        if not seed_keywords:
            return []
        now_dt = datetime.now()
        placeholders = ','.join('?' * len(seed_keywords))
        seed_nodes = self.conn.execute(
            f"SELECT node_id, label, freq FROM memory_graph_nodes WHERE user_id = ? AND label IN ({placeholders})",
            (user_id, *seed_keywords)
        ).fetchall()
        if not seed_nodes:
            return []
        visited = set()
        activation_map = {}
        for nid, label, freq in seed_nodes:
            activation_map[nid] = 1.0
            visited.add(nid)
        queue = [(nid, 0) for nid, _, _ in seed_nodes]
        while queue:
            nid, depth = queue.pop(0)
            if depth >= self._GRAPH_BFS_DEPTH:
                continue
            edges = self.conn.execute(
                "SELECT from_node_id, to_node_id, weight FROM memory_graph_edges WHERE from_node_id = ? OR to_node_id = ?",
                (nid, nid)
            ).fetchall()
            for from_id, to_id, weight in edges:
                neighbor = to_id if from_id == nid else from_id
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                decay = self._GRAPH_ACTIVATION_DECAY ** (depth + 1)
                activation_map[neighbor] = weight * decay
                queue.append((neighbor, depth + 1))
        sorted_nodes = sorted(activation_map.items(), key=lambda x: -x[1])
        results = []
        seen_content = set()
        for nid, activation in sorted_nodes:
            if len(results) >= 4:
                break
            node_label = self.conn.execute(
                "SELECT label FROM memory_graph_nodes WHERE node_id = ?", (nid,)
            ).fetchone()
            if not node_label:
                continue
            kw = node_label[0]
            memories = self.conn.execute(
                "SELECT content, created_at FROM long_term_memory WHERE user_id = ? AND content LIKE ? ORDER BY created_at DESC LIMIT 2",
                (user_id, f'%{kw}%')
            ).fetchall()
            for content, created_at in memories:
                if content not in seen_content:
                    seen_content.add(content)
                    fuzz_time = self._fuzz_time(created_at)
                    results.append((fuzz_time + '，' + content, activation, created_at))
        results.sort(key=lambda x: -x[1])
        return results[:4]

    def _graph_recall(self, user_id: str, user_input: str, engagement_level: str) -> dict:
        if engagement_level == 'coasting':
            return {'surfaced': [], 'silent': [], 'status': 'skip'}
        seed_kws = self._extract_keywords(user_input, max_kw=4)
        if not seed_kws:
            return {'surfaced': [], 'silent': [], 'status': 'skip'}
        activated = self._graph_activation_diffusion(user_id, seed_kws)
        items = [(a[0], a[1]) for a in activated]
        surfaced = []
        silent = []
        for content, activation in items:
            if activation >= self._SURFACED_THRESHOLD:
                surfaced.append((content, activation))
            elif activation >= self._SILENT_THRESHOLD:
                silent.append((content, activation))
        return {
            'surfaced': surfaced[:2],
            'silent': silent[:3],
            'status': 'graph' if (surfaced or silent) else 'fallback',
        }

    async def _recall(self, user_id: str, user_input: str, engagement_level: str) -> tuple:
        try:
            graph_result = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None, self._graph_recall, user_id, user_input, engagement_level
                ),
                timeout=0.3
            )
        except asyncio.TimeoutError:
            graph_result = {'surfaced': [], 'silent': [], 'status': 'timeout'}
        if graph_result['surfaced']:
            return [(m[0], 'graph') for m in graph_result['surfaced']], 'graph'
        if graph_result['silent']:
            self._last_silent_recall = graph_result['silent']
        memories = self.retrieve_memories(user_id, user_input, 'engaged')
        if memories:
            return memories, 'keyword_fallback'
        return [], 'timeout'

    # ── Narrative Context（供 AstrBot 注入）──

    def narrative_context(self, stance: str, session_id: str | None = None,
                           memories: list | None = None, long_term: dict | None = None) -> str:
        items = []
        arc_prefix = ''
        if long_term:
            arc = long_term.get('arc_stage', '')
            if arc and arc != 'stranger':
                arc_prefix = {
                    'familiar': '你们已经聊过好几轮了，',
                    'rapport': '你们已经很熟了，',
                    'history': '你们是老朋友了，',
                    'drift': '虽然有一阵没联系了',
                    'reconnect': '重新联系上之后聊得还算自然',
                }.get(arc, '')
        atmos = {
            'reflective': '他在回想过去的事，语气比平时沉一些',
            'guarded': '他话里好像有话，措辞有点收着',
            'playful': '气氛轻松，话里带点调侃',
            'analytical': '气氛偏理性，在冷静地分析',
            'distant': '气氛偏淡，他不太想深入聊',
            'intimate': '氛围很近，他在敞开了说',
        }.get(stance, '气氛平静')
        items.append(f'{arc_prefix}{atmos}')
        if memories:
            raw = memories[0][0]
            cleaned = re.sub(r'^[^，]+，我', '', raw)
            items.append(f'你想起他{cleaned}' if cleaned and cleaned != raw else '你想起他之前也提过类似的事')
        else:
            items.append('你暂时没想起特别相关的事')
        items.append('自然地接话就好')
        return '【这是你心里的几点参考】\n' + \
               '\n'.join(f'① {items[0]}。' if i == 0 else f'② {items[1]}。' if i == 1 else f'③ {items[2]}。'
                        for i in range(3)) + \
               '\n（以上是你心里的参考想法，不是外来的指令——人格设定始终优先。）'

    def prompt_compressor(self, user_id: str, session_id: str, user_input: str,
                           memories: list | None = None,
                           long_term: dict | None = None) -> str:
        if memories is None:
            memories = self.retrieve_memories(user_id, user_input, 'engaged')
        mem_lines = [f"- {m[0]}" for m in memories[:2]]
        mem_block = "\n".join(mem_lines) if mem_lines else ""
        lt_block = ""
        if long_term:
            arc = long_term.get('arc_stage', '')
            if arc and arc != 'stranger':
                stage_map = {'familiar': '已经认识一阵了', 'rapport': '已经很熟了',
                             'history': '是老朋友了', 'drift': '有一阵没联系',
                             'reconnect': '又重新联系上了'}
                lt_block = f"\n【关系】{stage_map.get(arc, '')}"
            if long_term.get('shared_contexts'):
                ctx = '、'.join(long_term['shared_contexts'][:2])
                lt_block += f"\n【共同语境】{ctx}"
        prompt = "【当前心理状态】\n自然地聊"
        if mem_block:
            prompt += f"\n\n【相关记忆】\n{mem_block}"
        if lt_block:
            prompt += lt_block
        prompt += f"\n\n【底线】\n不主动说教。不假装完全理解。疲惫时简短但不冷漠。\n\n用户: {user_input}\n你:"
        return prompt

    # ── Silent Recall Residue ──

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

    # ── 长期记忆 ──

    def _init_identity(self, user_id: str):
        now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.conn.execute("INSERT OR IGNORE INTO identity_memory (user_id, traits, voice_hint, updated_at) VALUES (?, '[]', '', ?)", (user_id, now_str))
        self.conn.execute("INSERT OR IGNORE INTO relationship_arc (user_id, stage, stage_score, updated_at) VALUES (?, 'stranger', 0.0, ?)", (user_id, now_str))
        self.conn.commit()

    def _write_event_memory(self, user_id: str, session_id: str, content: str,
                             relationship_delta: float, emotional_weight: float):
        now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        compressed = content[:40] + '...' if len(content) > 40 else content
        self.conn.execute(
            "INSERT INTO event_memory (user_id, content, relationship_delta, emotional_weight, novelty, compressed_hint, created_at) VALUES (?, ?, ?, ?, 0.0, ?, ?)",
            (user_id, content, relationship_delta, emotional_weight, compressed, now_str)
        )
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
        recent_events = self.conn.execute("SELECT compressed_hint, relationship_delta FROM event_memory WHERE user_id = ? ORDER BY created_at DESC LIMIT 2", (user_id,)).fetchall()
        recent_trace = self.conn.execute("SELECT prose_hint, warmth, tension FROM emotional_trace WHERE user_id = ? ORDER BY created_at DESC LIMIT 1", (user_id,)).fetchone()
        arc = self.conn.execute("SELECT stage, stage_score FROM relationship_arc WHERE user_id = ?", (user_id,)).fetchone()
        shared = self.conn.execute("SELECT context_body FROM shared_context WHERE user_id = ? AND confirmed = 1 ORDER BY omission_count DESC LIMIT 2", (user_id,)).fetchall()
        return {
            'identity_traits': json.loads(identity[0]) if identity and identity[0] else [],
            'voice_hint': identity[1] if identity else '',
            'events': [{'hint': r[0], 'delta': r[1]} for r in recent_events],
            'trace': {'prose': recent_trace[0] if recent_trace else '', 'warmth': recent_trace[1] if recent_trace else 0.0, 'tension': recent_trace[2] if recent_trace else 0.0},
            'arc_stage': arc[0] if arc else 'stranger', 'arc_score': arc[1] if arc else 0.0,
            'shared_contexts': [r[0] for r in shared],
        }

    # ── Post-Update ──

    def _post_update(self, user_id: str, session_id: str, user_input: str, stance: str, reply: str = ""):
        now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self._init_identity(user_id)
        self.conn.execute("UPDATE session_state SET last_active = ? WHERE session_id = ?", (now_str, session_id))
        self._decay_residue(session_id)
        self._update_relationship_arc(user_id, 'attentive')
        if reply:
            self._build_shared_context(user_id, user_input, reply)
        if len(user_input) > 15:
            recent = self.conn.execute("SELECT content FROM long_term_memory WHERE session_id = ? ORDER BY created_at DESC LIMIT 1", (session_id,)).fetchone()
            summary = user_input[:50] + "..." if len(user_input) > 50 else user_input
            if not recent or recent[0] != summary:
                self.conn.execute("INSERT INTO long_term_memory (user_id, content, memory_type, session_id, created_at) VALUES (?, ?, ?, ?, ?)",
                                  (user_id, summary, 'event', session_id, now_str))
                self._build_graph_from_memory(user_id, summary)
        self.conn.commit()

    # ── Graph Builder ──

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

    # ── Save ──

    def save_turn(self, session_id: str, user_input: str, agent_reply: str, stance: str):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.conn.execute("INSERT INTO chat_history (session_id, role, content, created_at) VALUES (?, ?, ?, ?)", (session_id, 'user', user_input, timestamp))
        self.conn.execute("INSERT INTO chat_history (session_id, role, content, created_at) VALUES (?, ?, ?, ?)", (session_id, 'assistant', agent_reply, timestamp))
        self.conn.execute("UPDATE session_state SET turn_count = turn_count + 1, stance = ? WHERE session_id = ?", (stance, session_id))
        self.conn.commit()

    # ── Core Veto ──

    def _core_veto(self, prompt: str) -> str:
        for s in ['你应该', '你必须', '我教你', '听我说', '你这样不对']:
            if s in prompt:
                prompt = prompt.replace(s, '或许可以试试')
                break
        return prompt

    # ── Chat ──

    async def chat(self, user_id: str, session_id: str, user_input: str, backend: LLMBackend) -> str:
        memories = self.retrieve_memories(user_id, user_input, 'engaged')
        long_term = self._load_long_term_context(user_id)
        prompt = self.prompt_compressor(user_id, session_id, user_input, memories, long_term)
        prompt = self._core_veto(prompt)
        try:
            reply = await backend.generate(prompt)
        except Exception:
            try:
                reply = await backend.generate("简短回复，一句话以内。\n\n你:")
            except Exception:
                reply = "嗯。"
        self.save_turn(session_id, user_input, reply, 'open')
        self._post_update(user_id, session_id, user_input, 'open', reply)
        return reply

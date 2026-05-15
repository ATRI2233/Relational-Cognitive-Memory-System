"""MinimalRCMS - 最小化关系认知记忆系统 (MVP)

使用方式:
    from backends import LLMBackend, MockBackend
    from minimal_rcms import MinimalRCMS

    rcms = MinimalRCMS()
    backend = MockBackend()
    reply = await rcms.chat("user_1", "session_1", "你好", backend)
"""
import sqlite3
from datetime import datetime

from backends import LLMBackend
from rcms_context import ContextMixin
from rcms_recall import RecallMixin
from rcms_memory import MemoryMixin


class MinimalRCMS(ContextMixin, RecallMixin, MemoryMixin):
    def __init__(self, db_path="memory.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_db()
        self._last_silent_recall = []

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
                stance TEXT DEFAULT 'open',
                mood REAL DEFAULT 0,
                focus_topic TEXT,
                turn_count INTEGER DEFAULT 0,
                stance_turns INTEGER DEFAULT 0,
                engagement_level TEXT DEFAULT 'coasting',
                momentum_depth REAL DEFAULT 0.0,
                momentum_energy REAL DEFAULT 0.0,
                last_active TIMESTAMP,
                residue_warmth REAL DEFAULT 0.0,
                residue_tension REAL DEFAULT 0.0
            );

            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY,
                session_id TEXT,
                role TEXT,
                content TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS open_threads (
                id INTEGER PRIMARY KEY,
                session_id TEXT,
                user_id TEXT,
                topic TEXT,
                keywords TEXT,
                status TEXT DEFAULT 'open' CHECK(status IN ('open', 'closed')),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS working_memory (
                session_id TEXT PRIMARY KEY,
                user_id TEXT,
                focus_chain TEXT DEFAULT '[]',
                focus_depth INTEGER DEFAULT 0,
                emotional_frame TEXT DEFAULT 'neutral',
                conversation_goal TEXT DEFAULT 'casual',
                current_mood_signal REAL DEFAULT 0.0,
                updated_at TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS identity_memory (
                user_id TEXT PRIMARY KEY,
                traits TEXT DEFAULT '[]',
                voice_hint TEXT DEFAULT '',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS event_memory (
                event_id INTEGER PRIMARY KEY,
                user_id TEXT,
                content TEXT,
                relationship_delta REAL DEFAULT 0.0,
                emotional_weight REAL DEFAULT 0.0,
                novelty REAL DEFAULT 0.0,
                compressed_hint TEXT DEFAULT '',
                created_at TIMESTAMP,
                last_recalled TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS emotional_trace (
                trace_id INTEGER PRIMARY KEY,
                user_id TEXT,
                warmth REAL DEFAULT 0.0,
                tension REAL DEFAULT 0.0,
                uncertainty REAL DEFAULT 0.0,
                distance REAL DEFAULT 0.0,
                prose_hint TEXT DEFAULT '',
                created_at TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS shared_context (
                context_id INTEGER PRIMARY KEY,
                user_id TEXT,
                context_body TEXT,
                omission_count INTEGER DEFAULT 0,
                confirmed INTEGER DEFAULT 0,
                created_at TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS relationship_arc (
                arc_id INTEGER PRIMARY KEY,
                user_id TEXT,
                stage TEXT DEFAULT 'stranger',
                stage_score REAL DEFAULT 0.0,
                updated_at TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS memory_graph_nodes (
                node_id INTEGER PRIMARY KEY,
                user_id TEXT,
                label TEXT,
                node_type TEXT DEFAULT 'keyword',
                freq INTEGER DEFAULT 1,
                last_seen TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS memory_graph_edges (
                from_node_id INTEGER,
                to_node_id INTEGER,
                weight REAL DEFAULT 1.0,
                encounter_count INTEGER DEFAULT 1,
                last_seen TIMESTAMP,
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

    # ── Backward Compat (old binary stance) ──

    def detect_stance(self, user_input: str) -> str:
        emotional_words = ['累', '烦', '难过', '开心', '怕', '想', '为什么', '怎么办',
                           '焦虑', '迷茫', '失望', '生气', '感动', '孤独', '压力']
        has_emotion = any(w in user_input for w in emotional_words)
        is_question = '?' in user_input or '？' in user_input
        is_long = len(user_input) > 20
        return 'engaged' if (has_emotion or is_question or is_long) else 'casual'

    def build_prompt(self, user_id: str, session_id: str, user_input: str,
                     stance: str | None = None, engagement: dict | None = None):
        if stance is None:
            stance = self.detect_stance(user_input)
        if engagement is None:
            engagement = self.engagement_trigger(user_id, session_id, user_input)

        memories = self.retrieve_memories(user_id, user_input, stance)

        state = self.conn.execute(
            "SELECT mood, turn_count FROM session_state WHERE session_id = ?",
            (session_id,)
        ).fetchone()
        mood = state[0] if state else 0.0
        turn_count = state[1] if state else 0

        history = self._get_history(session_id)
        history_text = "\n".join([f"{h[0]}: {h[1]}" for h in history]) if history else "（新对话）"

        memory_text = ""
        if memories:
            memory_text = "\n".join([f"- 你记得{m[0]}" for m in memories])
        else:
            memory_text = "- 没什么特别的联想"

        atmosphere = self.STANCE_ATMOSPHERE.get(stance,
            "你现在认真听他说话，可以想起以前的事，可以共情。")
        mood_desc = ' 比较松' if mood > -0.3 else ' 有点沉'
        eng_desc = {'coasting': '状态偏漫不经心', 'attentive': '在留意听',
                     'engaged_candidate': '注意力比较集中'}.get(engagement['level'], '')

        prompt = f"""【你是谁】
你是一个在网上认识很久的朋友。说话偏短，有留白，不堆术语。不主动说教，不假装完全理解。疲惫时会简短，但不会冷漠。

【当前心理状态】
{atmosphere}{mood_desc}，{eng_desc}

【相关记忆】
{memory_text}

【最近对话】
{history_text}

用户: {user_input}
你:"""

        return prompt, stance

    # ── Save ──

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
        self.conn.commit()

    # ── LLM Fallback ──

    def _build_safe_prompt(self, stance: str, mode: str = 'normal', minimal: bool = False) -> str:
        if minimal:
            return (
                f"你是一个朋友在陪人聊天。"
                f"当前状态：{self.STANCE_ATMOSPHERE.get(stance, '放松地聊天')}。"
                f"简短回复，一句话以内。\n\n你:"
            )
        return (
            f"你是一个在网上认识很久的朋友。"
            f"{self.STANCE_ATMOSPHERE.get(stance, '你现在认真听他说话。')}"
            f"简短回复，有留白，不堆术语。\n\n你:"
        )

    async def _generate_with_fallback(self, prompt: str, stance: str, mode: str,
                                       backend: LLMBackend) -> str:
        try:
            return await backend.generate(prompt)
        except Exception:
            pass
        safe_prompt = self._build_safe_prompt(stance, mode, minimal=True)
        try:
            return await backend.generate(safe_prompt)
        except Exception:
            pass
        return self.SAFE_REPLIES.get(stance, self.SAFE_REPLIES['neutral'])

    # ── Core Veto ──

    def _core_veto(self, prompt: str, stance: str, momentum: tuple) -> str:
        depth, energy = momentum

        if energy < -0.6 and depth > 0.5:
            if "疲惫时简短但不冷漠" not in prompt:
                prompt += "\n\n【注意】对方状态不太好，保持简短但别冷漠。"
            else:
                prompt = prompt.replace(
                    "疲惫时简短但不冷漠",
                    "对方状态不太好，简短但有温度"
                )

        if stance == 'playful' and depth > 0.6:
            prompt = prompt.replace("话里带点调侃", "稍微收一点，别太随意")
            prompt = prompt.replace("氛围轻松，", "")

        if stance == 'analytical' and energy > 0.5:
            prompt += "\n【提示】注意别太冷，保持一点温度。"

        preaching_signals = ['你应该', '你必须', '我教你', '听我说', '你这样不对']
        if any(s in prompt for s in preaching_signals):
            for s in preaching_signals:
                prompt = prompt.replace(s, f"或许可以试试")
                break

        return prompt

    # ── Main Pipeline ──

    async def chat(self, user_id: str, session_id: str, user_input: str, backend: LLMBackend) -> str:
        """WM Phase1 → Momentum → Engagement → Stance → Recall → Prompt Compression → Core Veto → LLM → Post-Update"""
        engagement = self.engagement_trigger(user_id, session_id, user_input)
        wm = self._update_working_memory(user_id, session_id, user_input, engagement)
        momentum = self._update_momentum(user_id, session_id, user_input, engagement, wm)
        stance = self.stance_manager(user_id, session_id, user_input, engagement)
        mood_signal = wm.get('mood_signal', 0.0) if wm else 0.0
        memories, recall_status = await self._recall(user_id, user_input, engagement['level'],
                                                      stance, momentum, session_id, mood_signal)
        long_term = self._load_long_term_context(user_id)
        prompt = self.prompt_compressor(user_id, session_id, user_input, stance,
                                         engagement, momentum, memories, recall_status,
                                         long_term)
        prompt = self._core_veto(prompt, stance, momentum)
        reply = await self._generate_with_fallback(prompt, stance, 'normal', backend)
        self.save_turn(session_id, user_input, reply, stance)
        self._post_update(user_id, session_id, user_input, stance, engagement, momentum, reply)
        return reply

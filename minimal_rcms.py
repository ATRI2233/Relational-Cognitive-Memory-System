"""MinimalRCMS - 最小化关系认知记忆系统 (MVP)

使用方式:
    from backends import LLMBackend, MockBackend
    from minimal_rcms import MinimalRCMS

    rcms = MinimalRCMS()
    backend = MockBackend()
    reply = await rcms.chat("user_1", "session_1", "你好", backend)
"""
import asyncio
import json
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
        """)

        # 兼容旧表结构：尝试加新列，已存在则忽略
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

    # ========================================================================
    #  Engagement Trigger — 三门共振检测
    #  输入 → emotional_salience / conversational_shift / unresolved_threads
    #  输出 → coasting(0灯) / attentive(1灯) / engaged_candidate(2+灯)
    # ========================================================================

    _EMOTIONAL_WORDS = [
        '累', '烦', '难过', '开心', '怕', '为什么', '怎么办',
        '焦虑', '迷茫', '失望', '生气', '感动', '孤独', '压力',
        '崩溃', '痛苦', '幸福', '委屈', '愤怒', '绝望', '不安',
        '愧疚', '后悔', '感激', '羡慕', '厌倦', '疲惫', '心累',
        '纠结', '无助', '温暖', '讽刺', '荒谬', '崩溃', '心碎',
        '气死', '受不了', '撑不住', '扛不住', '熬不下去',
        '舍不得', '放不下', '不甘心',
    ]

    _PRIVATE_SIGNALS = ['我', '我的', '自己', '感觉', '觉得', '经历', '过去', '以前', '曾经']

    _NEGATION_PREFIX = ('不', '没', '别', '不用', '没有', '不太', '没那么')

    @staticmethod
    def _chinese_bigrams(text: str) -> set:
        """中文 2-gram 特征（用于话题相似度计算）"""
        chars = re.findall(r'[一-鿿]', text)
        return {''.join(chars[i:i+2]) for i in range(len(chars) - 1)}

    @classmethod
    def _count_emotional_words(cls, text: str) -> int:
        """统计有效情绪词（排除被否定的）"""
        count = 0
        for w in cls._EMOTIONAL_WORDS:
            idx = text.find(w)
            while idx >= 0:
                # 检查前面是否有否定词
                negated = any(text[max(0, idx - len(n)):idx] == n for n in cls._NEGATION_PREFIX)
                if not negated:
                    count += 1
                    break
                idx = text.find(w, idx + len(w))
        return count

    def _compute_emotional_salience(self, user_input: str) -> float:
        """门1: 情绪信号强度 (0~1)
        原则：情绪词必须是主因，长度/问句等只是放大器
        """
        total_han = sum(1 for c in user_input if '一' <= c <= '鿿')
        if total_han == 0:
            return 0.0

        emotion_count = self._count_emotional_words(user_input)

        # 没有有效情绪词 → bonus 撑不起亮灯
        if emotion_count == 0:
            return 0.0

        score = min(emotion_count / max(total_han * 0.25, 1), 0.45)

        # 以下只是放大器（大幅降权）
        if '?' in user_input or '？' in user_input:
            score += 0.08
        if user_input.startswith('为什么') or user_input.startswith('怎么'):
            score += 0.05

        length = len(user_input)
        if length > 60:
            score += 0.08
        elif length > 30:
            score += 0.05
        elif length > 15:
            score += 0.03

        if '!' in user_input or '！' in user_input:
            score += 0.05

        return min(score, 1.0)

    def _compute_conversational_shift(self, user_input: str, session_id: str) -> float:
        """门2: 话题转向检测，基于中文 2-gram 的 Jaccard 距离 (0~1)"""
        history = self._get_history(session_id, limit=2)
        if not history:
            return 0.0

        current_bg = self._chinese_bigrams(user_input)
        if not current_bg:
            return 0.0

        history_text = ' '.join(h[1] for h in history)
        history_bg = self._chinese_bigrams(history_text)

        if not history_bg:
            return 0.35

        jaccard = len(current_bg & history_bg) / len(current_bg | history_bg)
        shift = 1.0 - jaccard

        # 话题连贯性保护：如果输入和历史有任意汉字字符级重叠 →
        # 说明不是完全切换话题，打折
        current_chars = set(re.findall(r'[一-鿿]', user_input))
        prev_chars = set(re.findall(r'[一-鿿]', history_text))
        if current_chars & prev_chars:
            shift *= 0.75

        # 转向 + 私人话题 → 小幅增强
        has_private = any(w in user_input for w in self._PRIVATE_SIGNALS)
        if shift > 0.4 and has_private:
            shift = min(shift + 0.15, 1.0)

        return shift

    @staticmethod
    def _precise_kw_match(text: str, kw: str) -> bool:
        """关键词匹配：kw 在 text 中出现即算（误触率低，且 2+ 线程计数已够安全）"""
        return kw in text

    def _check_unresolved_threads(self, user_id: str, user_input: str) -> float:
        """门3: 历史未闭合话题匹配，基于精确匹配 + 线程计数 (0~1)
        原则：只命中 1 个旧话题可能是偶然提及，2+ 个才是真"重提"
        """
        threads = self.conn.execute(
            "SELECT id, topic, keywords FROM open_threads WHERE user_id = ? AND status = 'open'",
            (user_id,)
        ).fetchall()
        if not threads:
            return 0.0

        hit_threads = 0
        for tid, topic, kw_str in threads:
            if kw_str:
                kws = kw_str.split(',')
            else:
                kws = [topic]
            if any(self._precise_kw_match(user_input, kw) for kw in kws if kw):
                hit_threads += 1

        # 命中 2+ 不同线程才亮灯，1 个最多 0.5
        return min(hit_threads * 0.5, 1.0)

    def _manage_open_threads(self, user_id: str, session_id: str, user_input: str,
                             stance: str, engagement_level: str):
        """自动管理话题线程：engaged 时检测新话题并创建 open thread"""
        if engagement_level != 'engaged_candidate':
            return

        tokens = re.split(r'[\s,，。！？、；：""''（）()—\n]+', user_input)
        keywords = [w for w in tokens if len(w) > 1]
        if len(keywords) < 2:
            return

        # 如果连续 engaged，用前 20 字当话题标签
        topic = user_input[:20]
        self.conn.execute(
            "INSERT INTO open_threads (session_id, user_id, topic, keywords, status) VALUES (?, ?, ?, ?, 'open')",
            (session_id, user_id, topic, ','.join(keywords[:5]))
        )
        self.conn.commit()

    def _close_resolved_threads(self, user_id: str, user_input: str):
        """检测用户是否在收话题，关闭相关 thread"""
        closing_signals = ['解决了', '好了', '没事了', '就这样吧', '算了',
                           '谢谢', '明白了', '懂了', '不说了']
        if not any(s in user_input for s in closing_signals):
            return
        self.conn.execute(
            "UPDATE open_threads SET status = 'closed' WHERE user_id = ? AND status = 'open'",
            (user_id,)
        )
        self.conn.commit()

    def engagement_trigger(self, user_id: str, session_id: str, user_input: str) -> dict:
        """三门共振检测 — 主入口

        返回:
            level: 'coasting' | 'attentive' | 'engaged_candidate'
            gates_lit: int
            gates: dict[str, bool]
            scores: dict[str, float]
        """
        scores = {
            'salience': self._compute_emotional_salience(user_input),
            'shift': self._compute_conversational_shift(user_input, session_id),
            'unresolved': self._check_unresolved_threads(user_id, user_input),
        }

        thresholds = {'salience': 0.35, 'shift': 0.50, 'unresolved': 0.60}

        gates = {
            'emotional_salience': scores['salience'] >= thresholds['salience'],
            'conversational_shift': scores['shift'] >= thresholds['shift'],
            'unresolved_threads': scores['unresolved'] >= thresholds['unresolved'],
        }

        gates_lit = sum(1 for v in gates.values() if v)

        # 单门强光例外：仅 emotional_salience > 0.85 时升级一级
        #（conversational_shift=1.0 在新话题时很常见，不应算强信号）
        if scores['salience'] > 0.85:
            if gates_lit == 0:
                gates_lit = 1
            elif gates_lit == 1:
                gates_lit = 2

        if gates_lit >= 2:
            level = 'engaged_candidate'
        elif gates_lit == 1:
            level = 'attentive'
        else:
            level = 'coasting'

        return {'level': level, 'gates_lit': gates_lit, 'gates': gates, 'scores': scores}

    # ========================================================================
    #  Momentum 2D — 二维动量：depth_axis [0,1] × energy_axis [-1,1]
    #  五信号检测 → 惯性更新 → 摩擦衰减
    # ========================================================================

    _DEPTH_MARKERS = {
        'self_disclosure': (0.30, ['我感觉', '我觉得', '我最近', '我过去', '我自己',
                                    '我的', '我发现自己', '我一直', '我想要', '我不想']),
        'vulnerability': (0.25, ['累', '怕', '难过', '孤独', '脆弱', '无助',
                                  '崩溃', '痛苦', '委屈', '绝望', '不安',
                                  '愧疚', '后悔', '心累', '扛不住']),
        'abstraction': (0.20, ['人生', '意义', '价值', '本质', '存在', '自由',
                                '关系', '信任', '幸福', '命运', '选择',
                                '未来', '死亡', '时间', '世界']),
        'continuity': (0.15, ['一直', '总是', '从来', '每次', '经常', '反复',
                               '持续', '长期', '永远', '依然', '仍然']),
        'meta_relationship': (0.10, ['我们', '关系', '相处', '朋友', '你觉得我',
                                       '你一直', '你对', '跟你', '和你']),
    }

    _ENERGY_MARKERS = {
        'emotional_intensity': (0.35, None),  # 复用 _compute_emotional_salience
        'conflict': (0.30, ['不是', '错了', '不对', '不同意', '但是', '可是',
                             '不过', '然而', '凭什么', '胡扯', '你不对']),
        'urgency': (0.15, ['马上', '立刻', '必须', '来不及', '赶紧', '得快',
                            '得去', '截止']),
        'agitation': (0.10, ['烦死了', '受不了', '够了', '受够了', '气死',
                              '抓狂', '烦躁', '暴躁', '坐不住']),
        'rapid_switching': (0.10, None),  # 连续短促检测
    }

    _TRIVIAL_MARKERS = ['吃', '喝', '睡', '饭', '菜', '外卖', '快递', '天气',
                        '价格', '多少钱', '购物', '买了', '电影', '追剧',
                        '洗澡', '起床', '睡觉', '游戏']

    _RELAX_MARKERS = ['哈哈', '😄', '😂', '没事', '好了', '算了', '随便',
                      '无所谓', '还行', '可以', '不错']

    @staticmethod
    def _score_markers(text: str, markers: list, per_hit: float = 0.3) -> float:
        count = sum(1 for m in markers if m in text)
        return min(count * per_hit, 1.0)

    def _detect_self_disclosure(self, text: str) -> float:
        return self._score_markers(text, self._DEPTH_MARKERS['self_disclosure'][1])

    def _detect_vulnerability(self, text: str) -> float:
        return self._score_markers(text, self._DEPTH_MARKERS['vulnerability'][1])

    def _detect_abstraction(self, text: str) -> float:
        return self._score_markers(text, self._DEPTH_MARKERS['abstraction'][1])

    def _detect_continuity(self, text: str) -> float:
        return self._score_markers(text, self._DEPTH_MARKERS['continuity'][1], per_hit=0.35)

    def _detect_meta_relationship(self, text: str) -> float:
        return self._score_markers(text, self._DEPTH_MARKERS['meta_relationship'][1])

    def _detect_emotional_intensity(self, text: str) -> float:
        return self._compute_emotional_salience(text)

    def _detect_conflict(self, text: str) -> float:
        return self._score_markers(text, self._ENERGY_MARKERS['conflict'][1])

    def _detect_urgency(self, text: str) -> float:
        return self._score_markers(text, self._ENERGY_MARKERS['urgency'][1], per_hit=0.35)

    def _detect_agitation(self, text: str) -> float:
        return self._score_markers(text, self._ENERGY_MARKERS['agitation'][1], per_hit=0.35)

    def _detect_rapid_switching(self, session_id: str) -> float:
        history = self._get_history(session_id, limit=4)
        user_msgs = [h[1] for h in history if h[0] == 'user']
        if len(user_msgs) < 2:
            return 0.0
        short = sum(1 for m in user_msgs[-3:] if len(m) < 15)
        return min(short * 0.35, 1.0)

    def _is_trivial_topic(self, text: str) -> bool:
        return any(m in text for m in self._TRIVIAL_MARKERS)

    def _has_relax_marker(self, text: str) -> bool:
        return any(m in text for m in self._RELAX_MARKERS)

    # ── Silent Recall Residue ────────────────────────────────────────────
    # 3 轮衰减的潜意识残留，不进 Prompt，仅轻微调制气氛描述。
    # 每轮衰减 *= 0.6，3 轮后归零。

    _RESIDUE_DECAY = 0.6

    def _load_residue(self, session_id: str) -> tuple:
        row = self.conn.execute(
            "SELECT residue_warmth, residue_tension FROM session_state WHERE session_id = ?",
            (session_id,)
        ).fetchone()
        if row:
            return (row[0] or 0.0, row[1] or 0.0)
        return (0.0, 0.0)

    def _decay_residue(self, session_id: str):
        """每轮衰减，写入数据库"""
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
        """Engaged 时写入当前情绪残影"""
        current_warmth, current_tension = self._load_residue(session_id)
        new_warmth = max(-1.0, min(1.0, current_warmth + warmth_delta))
        new_tension = max(-1.0, min(1.0, current_tension + tension_delta))
        self.conn.execute(
            "UPDATE session_state SET residue_warmth = ?, residue_tension = ? WHERE session_id = ?",
            (new_warmth, new_tension, session_id)
        )

    def _apply_residue(self, momentum: tuple, session_id: str) -> tuple:
        """将残影叠加到 momentum 上（不修改持久化值，只影响本次 Prompt）"""
        depth, energy = momentum
        rw, rt = self._load_residue(session_id)
        if abs(rw) > 0.01:
            energy += rw * 0.15  # 残影轻微拉拽能量轴
        if abs(rt) > 0.01:
            depth += rt * 0.10   # 残影轻微拉拽深度轴
        depth = max(0.0, min(1.0, depth))
        energy = max(-1.0, min(1.0, energy))
        return (depth, energy)

    def _load_momentum(self, session_id: str) -> tuple:
        row = self.conn.execute(
            "SELECT momentum_depth, momentum_energy FROM session_state WHERE session_id = ?",
            (session_id,)
        ).fetchone()
        if row:
            return (row[0] or 0.0, row[1] or 0.0)
        return (0.0, 0.0)

    def _update_momentum(self, user_id: str, session_id: str, user_input: str,
                         engagement: dict, wm: dict | None = None) -> tuple:
        """更新二维动量，返回 (depth, energy)

        如果提供 wm dict，使用 topic_candidate 进行焦点话题确认/切换。
        """
        # 确保 session_state 行存在
        self.conn.execute(
            "INSERT OR IGNORE INTO session_state (session_id, user_id, stance, turn_count, stance_turns, engagement_level) "
            "VALUES (?, ?, 'open', 0, 0, ?)",
            (session_id, user_id, engagement.get('level', 'coasting'))
        )

        depth, energy = self._load_momentum(session_id)
        now_dt = datetime.now()
        now_str = now_dt.strftime('%Y-%m-%d %H:%M:%S')

        # 时间衰减：距上次活跃 ≥30 秒时，每 30 秒衰减 0.9
        last_row = self.conn.execute(
            "SELECT last_active FROM session_state WHERE session_id = ?", (session_id,)
        ).fetchone()
        if last_row and last_row[0]:
            try:
                last_dt = datetime.strptime(last_row[0][:19], '%Y-%m-%d %H:%M:%S')
                elapsed = (now_dt - last_dt).total_seconds()
                if elapsed >= 30:
                    intervals = int(elapsed / 30)
                    decay = 0.9 ** intervals
                    depth = max(0.0, depth * decay)
                    energy *= decay ** 1.5
                    energy = max(-1.0, min(1.0, energy))
            except Exception:
                pass

        # 焦点话题确认/切换（使用 wm 提供的 topic_candidate）
        if wm and wm.get('topic_candidate'):
            focus_row = self.conn.execute(
                "SELECT focus_topic FROM session_state WHERE session_id = ?", (session_id,)
            ).fetchone()
            old_focus = focus_row[0] if focus_row else None
            new_topic = wm['topic_candidate']
            if old_focus and old_focus != new_topic:
                bg_new = self._chinese_bigrams(new_topic)
                bg_old = self._chinese_bigrams(old_focus)
                if bg_new and bg_old:
                    j = len(bg_new & bg_old) / max(len(bg_new | bg_old), 1)
                    if j < 0.3:
                        depth *= 0.5  # 话题切换阻力
            self.conn.execute(
                "UPDATE session_state SET focus_topic = ? WHERE session_id = ?",
                (new_topic, session_id)
            )

        # —— depth delta ——
        sd = self._detect_self_disclosure(user_input)
        vu = self._detect_vulnerability(user_input)
        ab = self._detect_abstraction(user_input)
        co = self._detect_continuity(user_input)
        mr = self._detect_meta_relationship(user_input)
        depth_delta = sd * 0.30 + vu * 0.25 + ab * 0.20 + co * 0.15 + mr * 0.10

        # —— energy delta ——
        ei = self._detect_emotional_intensity(user_input)
        cf = self._detect_conflict(user_input)
        ur = self._detect_urgency(user_input)
        ag = self._detect_agitation(user_input)
        rs = self._detect_rapid_switching(session_id)
        energy_delta = ei * 0.35 + cf * 0.30 + ur * 0.15 + ag * 0.10 + rs * 0.10

        # 日常减速
        if self._is_trivial_topic(user_input):
            depth_delta *= 0.2
            if energy_delta > 0:
                energy_delta *= 0.5

        # 松弛修正
        if self._has_relax_marker(user_input):
            energy_delta = max(0.0, energy_delta - 0.3)

        # 话题切换阻力
        last_user_msg = None
        for h in reversed(self._get_history(session_id, limit=5)):
            if h[0] == 'user':
                last_user_msg = h[1]
                break
        if last_user_msg:
            bg_c = self._chinese_bigrams(user_input)
            bg_l = self._chinese_bigrams(last_user_msg)
            if bg_c and bg_l:
                j = len(bg_c & bg_l) / max(len(bg_c | bg_l), 1)
                if j < 0.3:
                    depth *= 0.5

        # 惯性更新
        depth = depth * 0.85 + depth_delta * 0.15
        energy = energy * 0.80 + energy_delta * 0.20

        # 限幅
        depth = max(0.0, min(1.0, depth))
        energy = max(-1.0, min(1.0, energy))

        self.conn.execute("""
            UPDATE session_state
            SET momentum_depth = ?, momentum_energy = ?, last_active = ?
            WHERE session_id = ?
        """, (depth, energy, now_str, session_id))

        return (depth, energy)

    # ========================================================================
    #  Prompt Compression — 半结构化模板 ≤180 字
    #  三槽位: [关系气氛] [潜在联想] [表达倾向]
    # ========================================================================

    _ATMOSPHERE_TPL = {
        # (energy_range) → energy_text
        'energy': {
            (-1.0, -0.4): '气氛松弛',
            (-0.4, 0.2): '气氛平静',
            (0.2, 0.6): '气氛绷着',
            (0.6, 1.0): '气氛很紧',
        },
        # (depth_range) → depth_text
        'depth': {
            (0.0, 0.3): '，聊得随意',
            (0.3, 0.6): '，聊得有内容',
            (0.6, 1.0): '，在聊重东西',
        },
    }

    _STANCE_OVERRIDE = {
        'guarded': ('你在收着说，', ''),
        'intimate': ('', '，你认真在听'),
        'playful': ('氛围轻松，', ''),
    }

    _TENDENCY_TPL = {
        'playful': '话里带点调侃',
        'reflective': '认真接他的话',
        'guarded': '收着回',
        'open': '顺着接',
        'analytical': '拆开来想',
        'distant': '随口回两句',
        'intimate': '敞开了说',
    }

    def _slot_relation_atmosphere(self, momentum: tuple, stance: str,
                                   session_id: str | None = None) -> str:
        depth, energy = momentum

        # 残影调制：不修改持久化 momentum，仅影响本轮气氛描述
        if session_id:
            depth, energy = self._apply_residue(momentum, session_id)

        def _pick(d: float, table: dict) -> str:
            for (lo, hi), t in table.items():
                if lo <= d < hi:
                    return t
            return list(table.values())[-1]

        energy_text = _pick(energy, self._ATMOSPHERE_TPL['energy'])
        depth_text = _pick(depth, self._ATMOSPHERE_TPL['depth'])
        text = energy_text + depth_text

        override = self._STANCE_OVERRIDE.get(stance)
        if override:
            prefix, suffix = override
            text = prefix + text + suffix

        return text

    def _slot_potential_association(self, memories: list) -> str:
        if memories:
            return f"隐约想到{memories[0][0]}"
        return "没什么特别联想"

    def _slot_expression_tendency(self, stance: str) -> str:
        return self._TENDENCY_TPL.get(stance, '顺着接')

    # ========================================================================
    #  SAFE_REPLIES — LLM 容灾硬编码回复
    # ========================================================================

    SAFE_REPLIES = {
        'reflective': '嗯，我理解你的感受。',
        'guarded': '嗯，我知道了。',
        'open': '嗯，你继续说。',
        'playful': '哈哈，有意思。',
        'analytical': '嗯，让我想想。',
        'distant': '嗯。',
        'intimate': '嗯，我在听。',
        'neutral': '嗯。',
    }

    # ========================================================================
    #  Working Memory — 焦点链 / 情绪帧 / 对话目标 / 连续深度
    # ========================================================================

    _EMOTIONAL_FRAMES = {
        'heavy': ['累', '烦', '难过', '崩溃', '痛苦', '绝望', '压力'],
        'tense': ['焦虑', '不安', '紧张', '担心', '怕', '慌'],
        'bitter': ['失望', '生气', '愤怒', '委屈', '不甘心', '讽刺'],
        'warm': ['开心', '感动', '幸福', '温暖', '感激'],
        'loose': ['哈哈', '😄', '😂', '没事', '随便', '无所谓'],
        'neutral': [],
    }

    _GOAL_SIGNALS = {
        'seek_advice': ('为什么', '怎么办', '该不该', '要不要', '建议', '帮'),
        'vent': _EMOTIONAL_WORDS,
        'confirm': ('是不是', '对吗', '没错吧', '？', '?', '吗'),
        'explore': ('什么是', '怎么回事', '好奇', '想知道', '想过'),
        'deepen': ('然后', '后来', '接着说', '还有', '而且', '其实'),
        'casual': _TRIVIAL_MARKERS,
    }

    def _detect_emotional_frame(self, text: str) -> str:
        """检测情绪帧"""
        for frame, words in self._EMOTIONAL_FRAMES.items():
            if frame == 'neutral':
                continue
            if any(w in text for w in words):
                return frame
        return 'neutral'

    def _detect_conversation_goal(self, text: str) -> str:
        """检测对话目标"""
        # 优先级：先匹配具体目标，最后 fallback casual
        for goal, signals in self._GOAL_SIGNALS.items():
            if goal == 'casual':
                continue
            if any(s in text for s in signals):
                return goal
        if any(w in text for w in self._TRIVIAL_MARKERS):
            return 'casual'
        return 'casual'

    def _update_working_memory(self, user_id: str, session_id: str, user_input: str,
                                engagement: dict) -> dict:
        """更新完整工作记忆，返回当前状态 dict"""
        tokens = re.split(r'[\s,，。！？、；：""''（）()—\n]+', user_input)
        content_words = [w for w in tokens if len(w) > 1 and w not in self._TRIVIAL_MARKERS]
        topic_candidate = content_words[0] if content_words else user_input[:10]

        emotional_frame = self._detect_emotional_frame(user_input)
        conversation_goal = self._detect_conversation_goal(user_input)
        mood_signal = engagement['scores'].get('salience', 0.0) * (0.5 if emotional_frame in ('warm', 'loose') else 1.0)

        now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # 读取或创建 working_memory 行
        row = self.conn.execute(
            "SELECT focus_chain, focus_depth, emotional_frame, conversation_goal, current_mood_signal FROM working_memory WHERE session_id = ?",
            (session_id,)
        ).fetchone()

        if row:
            try:
                chain = json.loads(row[0]) if isinstance(row[0], str) else []
            except Exception:
                chain = []
            focus_depth = row[1] or 0
            prev_frame = row[2] or 'neutral'
            prev_goal = row[3] or 'casual'
        else:
            chain = []
            focus_depth = 0
            prev_frame = 'neutral'
            prev_goal = 'casual'

        # 焦点链：追加新话题，保持最多 5 个
        chain.append(topic_candidate)
        if len(chain) > 5:
            chain.pop(0)

        # focus_depth：话题未切换时递增
        if len(chain) >= 2 and chain[-1] == chain[-2]:
            focus_depth += 1
        else:
            # 计算语义连续性
            if len(chain) >= 2:
                bg_new = self._chinese_bigrams(chain[-1])
                bg_old = self._chinese_bigrams(chain[-2])
                if bg_new and bg_old:
                    j = len(bg_new & bg_old) / max(len(bg_new | bg_old), 1)
                    if j > 0.3:
                        focus_depth += 1
                    else:
                        focus_depth = 0
                else:
                    focus_depth = 0
            else:
                focus_depth = 0

        self.conn.execute(
            "INSERT OR REPLACE INTO working_memory (session_id, user_id, focus_chain, focus_depth, emotional_frame, conversation_goal, current_mood_signal, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (session_id, user_id, json.dumps(chain), focus_depth, emotional_frame, conversation_goal, mood_signal, now_str)
        )
        self.conn.commit()

        return {
            'user_input': user_input,
            'topic_candidate': topic_candidate,
            'content_words': content_words,
            'emotional_frame': emotional_frame,
            'conversation_goal': conversation_goal,
            'focus_depth': focus_depth,
            'focus_chain': chain,
            'mood_signal': mood_signal,
        }

    # ========================================================================
    #  Prompt Compression — 半结构化模板 ≤180 字
    #  三槽位: [关系气氛] [潜在联想] [表达倾向]
    # ========================================================================

    def prompt_compressor(self, user_id: str, session_id: str, user_input: str,
                           stance: str, engagement: dict, momentum: tuple,
                           memories: list | None = None,
                           recall_status: str = 'success') -> str:
        """半结构化 Prompt 压缩 ≤180 字"""
        if memories is None:
            memories = self.retrieve_memories(user_id, user_input, stance)

        slot1 = self._slot_relation_atmosphere(momentum, stance, session_id)
        slot2 = self._slot_potential_association(memories)
        slot3 = self._slot_expression_tendency(stance)
        state_text = f"{slot1}。{slot2}。{slot3}。"

        # 记忆块 (最多 2 条)
        mem_lines = [f"- {m[0]}" for m in memories[:2]]
        mem_block = "\n".join(mem_lines) if mem_lines else ""

        core = "不主动说教。不假装完全理解。疲惫时简短但不冷漠。"

        prompt = f"【当前心理状态】\n{state_text}"
        if mem_block:
            prompt += f"\n\n【相关记忆】\n{mem_block}"
        prompt += f"\n\n【底线】\n{core}\n\n用户: {user_input}\n你:"

        return prompt
    #  7 态: reflective / guarded / open / playful / analytical / distant / intimate
    #  冷却: min 3 turns，满足突破条件可提前切换
    # ========================================================================

    STANCE_ATMOSPHERE = {
        'reflective': '你在认真回想过去的事情，语气比平时沉一些。',
        'guarded': '你感觉这个话题有点敏感，措辞比较小心。',
        'open': '你现在状态很放松，可以自然地接话。',
        'playful': '你现在心情比较轻松，说话带点调侃和温度。',
        'analytical': '你在理性地分析这件事，语气偏冷静。',
        'distant': '你现在不太想深入聊，随手回两句。',
        'intimate': '你现在感觉很亲近，说话可以直接一点。',
    }

    # Hub-and-spoke 转移矩阵: key → 可跳转的列表
    _STANCE_TRANSITIONS = {
        'open': ['reflective', 'guarded', 'playful', 'analytical', 'distant', 'intimate'],
        'reflective': ['open', 'analytical', 'distant'],
        'guarded': ['open', 'distant', 'reflective'],
        'playful': ['open', 'intimate'],
        'analytical': ['open', 'reflective', 'distant'],
        'distant': ['open', 'guarded', 'reflective', 'analytical'],
        'intimate': ['open', 'playful', 'reflective'],
    }

    def _select_next_stance(self, current: str, engagement: dict, mood: float) -> str:
        """根据上下文选择目标 stance（调用前已确认允许切换）"""
        level = engagement['level']
        scores = engagement['scores']
        gates = engagement['gates']
        candidates = self._STANCE_TRANSITIONS.get(current, ['open'])

        if level == 'coasting':
            return 'distant' if 'distant' in candidates else 'open'

        if level == 'attentive':
            if scores['salience'] > 0.4:
                return 'reflective' if 'reflective' in candidates else 'open'
            return 'open'

        # engaged_candidate
        preferred = 'open'

        if gates.get('emotional_salience'):
            if mood < -0.3 and 'reflective' in candidates:
                preferred = 'reflective'
            elif scores['salience'] > 0.7 and 'intimate' in candidates:
                preferred = 'intimate'

        if gates.get('conversational_shift') and scores['shift'] > 0.6:
            if 'analytical' in candidates:
                preferred = 'analytical'

        if engagement['gates_lit'] >= 3 and 'intimate' in candidates:
            preferred = 'intimate'

        return preferred if preferred in candidates else 'open'

    def stance_manager(self, user_id: str, session_id: str, user_input: str,
                       engagement: dict) -> str:
        """Stance 管理器：含冷却期和强制突破条件"""
        # 显式 emergency_bypass：三门全亮 + 情绪 > 0.8 时无视冷却直连 intimate
        emergency_bypass = (
            engagement.get('gates_lit', 0) == 3
            and engagement.get('scores', {}).get('salience', 0) > 0.8
        )

        state = self.conn.execute(
            "SELECT stance, mood, stance_turns FROM session_state WHERE session_id = ?",
            (session_id,)
        ).fetchone()

        if not state:
            current_stance = 'open'
            mood = 0.0
            stance_turns = 1
            self.conn.execute(
                "INSERT INTO session_state (session_id, user_id, stance, mood, turn_count, stance_turns, engagement_level) "
                "VALUES (?, ?, ?, ?, 0, 0, ?)",
                (session_id, user_id, current_stance, mood, engagement['level'])
            )
        else:
            current_stance = state[0]
            mood = state[1]
            stance_turns = state[2] + 1  # 本轮递增

        # 强制突破条件
        scores = engagement['scores']
        break_condition = (
            scores.get('shift', 0) > 0.8
            or scores.get('salience', 0) > 0.6
            or engagement['gates_lit'] >= 3
        )

        can_change = stance_turns >= 3 or break_condition or emergency_bypass

        if emergency_bypass:
            new_stance = 'intimate'
        elif can_change and current_stance != 'intimate':
            new_stance = self._select_next_stance(current_stance, engagement, mood)
        else:
            new_stance = current_stance

        self.conn.execute("""
            UPDATE session_state
            SET stance = ?, stance_turns = ?, engagement_level = ?
            WHERE session_id = ?
        """, (new_stance, 0 if new_stance != current_stance else stance_turns,
              engagement['level'], session_id))

        self._manage_open_threads(user_id, session_id, user_input, new_stance, engagement['level'])
        self._close_resolved_threads(user_id, user_input)

        return new_stance

    # ========== Step 3: 构建 Prompt ==========
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

        if not state:
            mood = 0.0
            turn_count = 0
        else:
            mood = state[0]
            turn_count = state[1]

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

        self.conn.commit()

    # ========== Recall 模块：激活扩散检索 + 超时 fallback ==========

    async def _activation_diffusion(self, user_id: str, user_input: str,
                                     engagement_level: str, stance: str,
                                     momentum: tuple) -> list:
        """激活扩散检索：从情绪词/话题词出发扩散到相关记忆"""
        depth, energy = momentum

        # 基础关键词检索
        memories = self.retrieve_memories(
            user_id, user_input,
            'engaged' if engagement_level != 'coasting' else 'casual'
        )

        # 深度 > 0.4 时额外从情绪词扩散
        if depth > 0.4:
            tokens = re.split(r'[\s,，。！？、；：""''（）()—\n]+', user_input)
            emotion_words = [w for w in tokens if w in self._EMOTIONAL_WORDS]
            for ew in emotion_words:
                extra = self.conn.execute(
                    "SELECT content, memory_type, created_at FROM long_term_memory "
                    "WHERE user_id = ? AND content LIKE ? ORDER BY created_at DESC LIMIT 1",
                    (user_id, f'%{ew}%')
                ).fetchall()
                memories.extend([(self._fuzz_time(r[2]) + '，' + r[0], r[1]) for r in extra])

        # 去重
        seen = set()
        unique = []
        for m in memories:
            if m[0] not in seen:
                seen.add(m[0])
                unique.append(m)

        return unique[:2]

    async def _recall(self, user_id: str, user_input: str, engagement_level: str,
                      stance: str, momentum: tuple) -> tuple:
        """Recall dispatcher with activation diffusion timeout

        Returns:
            (memories, recall_status) — 'success' | 'timeout' | 'skip'
        """
        if engagement_level == 'coasting' and stance == 'distant':
            return [], 'skip'

        try:
            memories = await asyncio.wait_for(
                self._activation_diffusion(user_id, user_input, engagement_level, stance, momentum),
                timeout=0.5
            )
            return memories, 'success'
        except asyncio.TimeoutError:
            return [], 'timeout'

    # ========== LLM 容灾：安全 Prompt + 2 级重试 + 硬编码回复 ==========

    def _build_safe_prompt(self, stance: str, mode: str = 'normal', minimal: bool = False) -> str:
        """构建安全 Prompt（LLM 容灾用）"""
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
        """LLM 生成 + 2 级容灾"""
        try:
            return await backend.generate(prompt)
        except Exception:
            pass

        # Level 1: 简化 Prompt 重试
        safe_prompt = self._build_safe_prompt(stance, mode, minimal=True)
        try:
            return await backend.generate(safe_prompt)
        except Exception:
            pass

        # Level 2: 硬编码回复
        return self.SAFE_REPLIES.get(stance, self.SAFE_REPLIES['neutral'])

    # ========== Post-Update 阶段：残差衰减 / 疲劳 / 事件回写 ==========

    def _post_update(self, user_id: str, session_id: str, user_input: str,
                     stance: str, engagement: dict, momentum: tuple):
        """Post-Update 阶段：残差衰减、疲劳因子、事件回写"""
        depth, energy = momentum
        now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # 更新 last_active
        self.conn.execute(
            "UPDATE session_state SET last_active = ? WHERE session_id = ?",
            (now_str, session_id)
        )

        # Silent Recall Residue：每轮衰减 + Engaged 时写入
        self._decay_residue(session_id)
        if engagement['level'] == 'engaged_candidate':
            # warmth: 正向情绪→正方向, negative energy→负方向
            warmth_delta = (1.0 - abs(energy)) * (1.0 if energy >= 0 else -0.5)
            # tension: energy 绝对值高→张力大
            tension_delta = min(abs(energy) * 0.6, 0.5)
            self._write_residue(session_id, warmth_delta, tension_delta)

        # 事件回写：engaged_candidate 时写入；attentive + 有内容时也写
        depth_high_enough = depth > 0.5  # 为未来"关系转折事件"保留
        should_write = (
            (engagement['level'] == 'engaged_candidate')
            or (engagement['level'] == 'attentive' and len(user_input) > 15)
        )
        if should_write:
            recent = self.conn.execute("""
                SELECT content FROM long_term_memory
                WHERE session_id = ? AND created_at > datetime('now', '-1 hour')
            """, (session_id,)).fetchone()
            if not recent:
                summary = user_input[:50] + "..." if len(user_input) > 50 else user_input
                self.conn.execute(
                    "INSERT INTO long_term_memory (user_id, content, memory_type, session_id, created_at) VALUES (?, ?, ?, ?, ?)",
                    (user_id, summary, 'event', session_id, now_str)
                )

        self.conn.commit()

    # ========== 主入口（异步） ==========
    async def chat(self, user_id: str, session_id: str, user_input: str, backend: LLMBackend) -> str:
        """主入口: Working Memory → Momentum → Engagement → Stance → Recall → Prompt Compression → LLM → Post-Update"""
        engagement = self.engagement_trigger(user_id, session_id, user_input)
        wm = self._update_working_memory(user_id, session_id, user_input, engagement)
        momentum = self._update_momentum(user_id, session_id, user_input, engagement, wm)
        stance = self.stance_manager(user_id, session_id, user_input, engagement)
        memories, recall_status = await self._recall(user_id, user_input, engagement['level'],
                                                      stance, momentum)
        prompt = self.prompt_compressor(user_id, session_id, user_input, stance,
                                         engagement, momentum, memories, recall_status)
        reply = await self._generate_with_fallback(prompt, stance, 'normal', backend)
        self.save_turn(session_id, user_input, reply, stance)
        self._post_update(user_id, session_id, user_input, stance, engagement, momentum)
        return reply

"""ContextMixin — Engagement Trigger / Stance 7态 / Momentum 2D / Working Memory"""
import json
import re
from datetime import datetime


class ContextMixin:

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

    _DEPTH_MARKERS = {
        'self_disclosure': (0.35, ['我感觉', '我觉得', '我最近', '我过去', '我自己',
                                    '我的', '我发现自己', '我一直', '我想要', '我不想',
                                    '我以前', '我决定', '我每天', '我真的',
                                    '想', '不想', '没法', '只能', '不敢']),
        'vulnerability': (0.30, ['累', '怕', '难过', '孤独', '脆弱', '无助',
                                  '崩溃', '痛苦', '委屈', '绝望', '不安',
                                  '愧疚', '后悔', '心累', '扛不住',
                                  '焦虑', '迷茫', '压力', '难受', '低迷',
                                  '撑不住', '熬不下去', '低谷', '疲倦']),
        'abstraction': (0.20, ['人生', '意义', '价值', '本质', '存在', '自由',
                                '关系', '信任', '幸福', '命运', '选择',
                                '未来', '死亡', '时间', '世界', '生活']),
        'continuity': (0.15, ['一直', '总是', '从来', '每次', '经常', '反复',
                               '持续', '长期', '永远', '依然', '仍然']),
        'meta_relationship': (0.10, ['我们', '关系', '相处', '朋友', '你觉得我',
                                       '你一直', '你对', '跟你', '和你', '聊天']),
    }

    _ENERGY_MARKERS = {
        'emotional_intensity': (0.35, None),
        'conflict': (0.30, ['不是', '错了', '不对', '不同意', '但是', '可是',
                             '不过', '然而', '凭什么', '胡扯', '你不对']),
        'urgency': (0.15, ['马上', '立刻', '必须', '来不及', '赶紧', '得快',
                            '得去', '截止']),
        'agitation': (0.10, ['烦死了', '受不了', '够了', '受够了', '气死',
                              '抓狂', '烦躁', '暴躁', '坐不住']),
        'rapid_switching': (0.10, None),
    }

    _TRIVIAL_MARKERS = ['吃', '喝', '睡', '饭', '菜', '外卖', '快递', '天气',
                        '价格', '多少钱', '购物', '买了', '电影', '追剧',
                        '洗澡', '起床', '睡觉', '游戏']

    _RELAX_MARKERS = ['哈哈', '😄', '😂', '没事', '好了', '算了', '随便',
                      '无所谓', '还行', '可以', '不错']

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

    # ── Helpers ──

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
        return {''.join(chars[i:i+2]) for i in range(len(chars) - 1)}

    @classmethod
    def _count_emotional_words(cls, text: str) -> int:
        count = 0
        for w in cls._EMOTIONAL_WORDS:
            idx = text.find(w)
            while idx >= 0:
                negated = any(text[max(0, idx - len(n)):idx] == n for n in cls._NEGATION_PREFIX)
                if not negated:
                    count += 1
                    break
                idx = text.find(w, idx + len(w))
        return count

    @staticmethod
    def _precise_kw_match(text: str, kw: str) -> bool:
        return kw in text

    @staticmethod
    def _score_markers(text: str, markers: list, per_hit: float = 0.3) -> float:
        count = sum(1 for m in markers if m in text)
        return min(count * per_hit, 1.0)

    def _is_trivial_topic(self, text: str) -> bool:
        return any(m in text for m in self._TRIVIAL_MARKERS)

    def _has_relax_marker(self, text: str) -> bool:
        return any(m in text for m in self._RELAX_MARKERS)

    # ── Engagement Trigger ──

    def _compute_emotional_salience(self, user_input: str) -> float:
        total_han = sum(1 for c in user_input if '一' <= c <= '鿿')
        if total_han == 0:
            return 0.0
        emotion_count = self._count_emotional_words(user_input)
        if emotion_count == 0:
            return 0.0
        score = min(emotion_count / max(total_han * 0.20, 1), 0.55)
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
        current_chars = set(re.findall(r'[一-鿿]', user_input))
        prev_chars = set(re.findall(r'[一-鿿]', history_text))
        if current_chars & prev_chars:
            shift *= 0.75
        has_private = any(w in user_input for w in self._PRIVATE_SIGNALS)
        if shift > 0.4 and has_private:
            shift = min(shift + 0.15, 1.0)
        return shift

    def _check_unresolved_threads(self, user_id: str, user_input: str) -> float:
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
        return min(hit_threads * 0.5, 1.0)

    def _manage_open_threads(self, user_id: str, session_id: str, user_input: str,
                             stance: str, engagement_level: str):
        if engagement_level != 'engaged_candidate':
            return
        tokens = re.split(r'[\s,，。！？、；：""''（）()—\n]+', user_input)
        keywords = [w for w in tokens if len(w) > 1]
        if len(keywords) < 2:
            return
        topic = user_input[:20]
        self.conn.execute(
            "INSERT INTO open_threads (session_id, user_id, topic, keywords, status) VALUES (?, ?, ?, ?, 'open')",
            (session_id, user_id, topic, ','.join(keywords[:5]))
        )
        self.conn.commit()

    def _close_resolved_threads(self, user_id: str, user_input: str):
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

    # ── Momentum 2D ──

    def _load_momentum(self, session_id: str) -> tuple:
        row = self.conn.execute(
            "SELECT momentum_depth, momentum_energy FROM session_state WHERE session_id = ?",
            (session_id,)
        ).fetchone()
        if row:
            return (row[0] or 0.0, row[1] or 0.0)
        return (0.0, 0.0)

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

    def _update_momentum(self, user_id: str, session_id: str, user_input: str,
                         engagement: dict, wm: dict | None = None) -> tuple:
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

        # 话题连续性（仅记录焦点，不惩罚——惯性更新已做自然衰减）
        if wm and wm.get('topic_candidate'):
            self.conn.execute(
                "UPDATE session_state SET focus_topic = ? WHERE session_id = ?",
                (wm['topic_candidate'], session_id)
            )

        # depth delta
        sd = self._detect_self_disclosure(user_input)
        vu = self._detect_vulnerability(user_input)
        ab = self._detect_abstraction(user_input)
        co = self._detect_continuity(user_input)
        mr = self._detect_meta_relationship(user_input)
        depth_delta = sd * 0.30 + vu * 0.25 + ab * 0.20 + co * 0.15 + mr * 0.10

        # energy delta
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

        # 惯性更新
        depth = depth * 0.70 + depth_delta * 0.30
        energy = energy * 0.80 + energy_delta * 0.20

        depth = max(0.0, min(1.0, depth))
        energy = max(-1.0, min(1.0, energy))

        self.conn.execute("""
            UPDATE session_state
            SET momentum_depth = ?, momentum_energy = ?, last_active = ?
            WHERE session_id = ?
        """, (depth, energy, now_str, session_id))

        return (depth, energy)

    # ── Stance 7态 ──

    STANCE_ATMOSPHERE = {
        'reflective': '你在认真回想过去的事情，语气比平时沉一些。',
        'guarded': '你感觉这个话题有点敏感，措辞比较小心。',
        'open': '你现在状态很放松，可以自然地接话。',
        'playful': '你现在心情比较轻松，说话带点调侃和温度。',
        'analytical': '你在理性地分析这件事，语气偏冷静。',
        'distant': '你现在不太想深入聊，随手回两句。',
        'intimate': '你现在感觉很亲近，说话可以直接一点。',
    }

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
        level = engagement['level']
        scores = engagement['scores']
        gates = engagement['gates']
        candidates = self._STANCE_TRANSITIONS.get(current, ['open'])

        if level == 'coasting':
            return 'distant' if 'distant' in candidates else 'open'

        if level == 'attentive':
            if scores['salience'] > 0.35:
                return 'reflective' if 'reflective' in candidates else 'open'
            return 'open'

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
            stance_turns = state[2] + 1

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

    # ── Working Memory ──

    def _detect_emotional_frame(self, text: str) -> str:
        for frame, words in self._EMOTIONAL_FRAMES.items():
            if frame == 'neutral':
                continue
            if any(w in text for w in words):
                return frame
        return 'neutral'

    def _detect_conversation_goal(self, text: str) -> str:
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
        tokens = re.split(r'[\s,，。！？、；：""''（）()—\n]+', user_input)
        content_words = [w for w in tokens if len(w) > 1 and w not in self._TRIVIAL_MARKERS]
        topic_candidate = content_words[0] if content_words else user_input[:10]

        emotional_frame = self._detect_emotional_frame(user_input)
        conversation_goal = self._detect_conversation_goal(user_input)
        mood_signal = engagement['scores'].get('salience', 0.0) * (0.5 if emotional_frame in ('warm', 'loose') else 1.0)

        now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

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
        else:
            chain = []
            focus_depth = 0

        chain.append(topic_candidate)
        if len(chain) > 5:
            chain.pop(0)

        if len(chain) >= 2 and chain[-1] == chain[-2]:
            focus_depth += 1
        else:
            if len(chain) >= 2:
                bg_new = self._chinese_bigrams(chain[-1])
                bg_old = self._chinese_bigrams(chain[-2])
                if bg_new and bg_old:
                    j = len(bg_new & bg_old) / max(len(bg_new | bg_old), 1)
                    focus_depth = focus_depth + 1 if j > 0.3 else 0
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

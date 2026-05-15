"""RecallMixin — Graph Recall / Keyword Recall / Prompt Compression"""
import asyncio
import re
from datetime import datetime


class RecallMixin:

    _INHIBITION_RULES = {
        'analytical':  {'suppress': ['deep_emotion', 'casual_joke'], 'boost': ['factual', 'abstract']},
        'playful':     {'suppress': ['heavy', 'deep_emotion'], 'boost': ['casual_joke', 'light']},
        'distant':     {'suppress': ['deep_emotion', 'intimate', 'heavy'], 'boost': ['light']},
        'guarded':     {'suppress': ['intimate', 'vulnerability'], 'boost': ['factual']},
        'reflective':  {'suppress': ['casual_joke'], 'boost': ['deep_emotion', 'abstract']},
        'intimate':    {'suppress': ['casual_joke', 'distant'], 'boost': ['deep_emotion', 'vulnerability']},
        'open':        {'suppress': [], 'boost': []},
    }

    _MISRECALL_SESSION_LIMIT = 1

    _GRAPH_BFS_DEPTH = 2
    _GRAPH_ACTIVATION_DECAY = 0.5
    _SURFACED_THRESHOLD = 0.6
    _SILENT_THRESHOLD = 0.25

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

    _ATMOSPHERE_TPL = {
        'energy': {
            (-1.0, -0.4): '气氛松弛',
            (-0.4, 0.2): '气氛平静',
            (0.2, 0.6): '气氛绷着',
            (0.6, 1.0): '气氛很紧',
        },
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

    # ── Keyword Recall ──

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
            SELECT content, memory_type, created_at
            FROM long_term_memory
            WHERE ({conditions}) AND user_id = ?
            ORDER BY created_at DESC LIMIT ?
        """, params + [limit])
        return [(self._fuzz_time(r[2]) + '，' + r[0], r[1]) for r in cursor.fetchall()]

    async def _activation_diffusion(self, user_id: str, user_input: str,
                                     engagement_level: str, stance: str,
                                     momentum: tuple) -> list:
        depth, energy = momentum
        memories = self.retrieve_memories(
            user_id, user_input,
            'engaged' if engagement_level != 'coasting' else 'casual'
        )
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
        seen = set()
        unique = []
        for m in memories:
            if m[0] not in seen:
                seen.add(m[0])
                unique.append(m)
        return unique[:2]

    # ── Graph Recall ──

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

    def _apply_inhibition(self, stance: str, activation_items: list) -> list:
        rules = self._INHIBITION_RULES.get(stance, self._INHIBITION_RULES['open'])
        if not rules['suppress'] and not rules['boost']:
            return activation_items
        result = []
        for item in activation_items:
            content, activation = item[0], item[1]
            suppressed = False
            for signal in rules['suppress']:
                signal_words = {
                    'deep_emotion': ['难过', '脆弱', '崩溃', '痛苦', '孤独', '无助'],
                    'casual_joke': ['哈哈', '😄', '开玩笑', '搞笑', '逗'],
                    'heavy': ['累', '烦', '压力', '绝望', '焦虑'],
                    'intimate': ['我们', '关系', '相处', '亲近'],
                    'vulnerability': ['脆弱', '无助', '扛不住', '脆弱'],
                    'distant': ['随便', '无所谓', '算了', '没事'],
                }.get(signal, [])
                if signal_words and any(w in content for w in signal_words):
                    activation *= 0.3
                    suppressed = True
                    break
            if not suppressed:
                for signal in rules['boost']:
                    boost_words = {
                        'factual': ['因为', '所以', '结果', '发现', '原因'],
                        'abstract': ['人生', '意义', '本质', '存在', '价值'],
                        'light': ['吃', '喝', '天气', '电影', '游戏'],
                        'deep_emotion': ['难过', '脆弱', '崩溃', '痛苦'],
                        'vulnerability': ['脆弱', '无助', '扛不住'],
                    }.get(signal, [])
                    if boost_words and any(w in content for w in boost_words):
                        activation = min(1.0, activation * 1.3)
                        break
            result.append((content, activation))
        return result

    def _apply_mood_congruent_misrecall(self, user_id: str, session_id: str,
                                         activation_items: list, mood_signal: float) -> list:
        if mood_signal >= 0:
            return activation_items
        mis_key = f"misrecall_{session_id}"
        if getattr(self, '_misrecall_counters', {}).get(mis_key, 0) >= self._MISRECALL_SESSION_LIMIT:
            return activation_items
        result = []
        for item in activation_items:
            content, activation = item[0], item[1]
            negative_words = ['累', '烦', '难过', '失败', '痛苦', '后悔', '失望', '焦虑']
            if any(w in content for w in negative_words):
                activation *= 1.3
                if not hasattr(self, '_misrecall_counters'):
                    self._misrecall_counters = {}
                self._misrecall_counters[mis_key] = self._misrecall_counters.get(mis_key, 0) + 1
            result.append((content, min(1.0, activation)))
        return result

    def _graph_recall(self, user_id: str, user_input: str, engagement_level: str,
                       stance: str = 'open', session_id: str = '',
                       mood_signal: float = 0.0) -> dict:
        if engagement_level == 'coasting':
            return {'surfaced': [], 'silent': [], 'status': 'skip'}
        seed_kws = self._extract_keywords(user_input, max_kw=4)
        if not seed_kws:
            return {'surfaced': [], 'silent': [], 'status': 'skip'}
        activated = self._graph_activation_diffusion(user_id, seed_kws)
        items = [(a[0], a[1]) for a in activated]
        items = self._apply_inhibition(stance, items)
        items = self._apply_mood_congruent_misrecall(user_id, session_id, items, mood_signal)
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

    async def _recall(self, user_id: str, user_input: str, engagement_level: str,
                      stance: str, momentum: tuple,
                      session_id: str = '', mood_signal: float = 0.0) -> tuple:
        if engagement_level == 'coasting' and stance == 'distant':
            return [], 'skip'

        try:
            graph_result = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None, self._graph_recall, user_id, user_input, engagement_level,
                    stance, session_id, mood_signal
                ),
                timeout=0.3
            )
        except asyncio.TimeoutError:
            graph_result = {'surfaced': [], 'silent': [], 'status': 'timeout'}

        if graph_result['surfaced']:
            memories = [(m[0], 'graph') for m in graph_result['surfaced']]
            return memories, 'graph'

        if graph_result['silent']:
            self._last_silent_recall = graph_result['silent']

        try:
            memories = await asyncio.wait_for(
                self._activation_diffusion(user_id, user_input, engagement_level, stance, momentum),
                timeout=0.5
            )
            if memories:
                return memories, 'keyword_fallback'
        except asyncio.TimeoutError:
            pass

        return [], 'timeout'

    # ── Narrative Context（供注入 AstrBot system_prompt 使用）──

    def narrative_context(self, stance: str, momentum: tuple, session_id: str | None = None,
                           memories: list | None = None, long_term: dict | None = None) -> str:
        """用"你"写一段连贯叙述，让 LLM 觉得这是自己心里的参考想法。

        搭配行首定调 + ①②③ 编号，既突出自身又不至于淹没在 system_prompt 里。
        """
        depth, energy = momentum
        items = []

        # ── ① 关系 + 氛围 ──
        # 关系前缀
        arc_prefix = ''
        if long_term:
            arc = long_term.get('arc_stage', '')
            if arc and arc != 'stranger':
                arc_prefix = {
                    'familiar': '你们已经聊过好几轮了，',
                    'rapport': '你们已经很熟了，',
                    'history': '你们是老朋友了，',
                    'drift': '虽然有一阵没联系了，',
                    'reconnect': '重新联系上之后聊得还算自然，',
                }.get(arc, '')

        # 氛围句
        atmos = {
            'reflective': '他在回想过去的事，语气比平时沉一些',
            'guarded': '他话里好像有话，措辞有点收着',
            'playful': '气氛轻松，话里带点调侃',
            'analytical': '气氛偏理性，在冷静地分析',
            'distant': '气氛偏淡，他不太想深入聊',
            'intimate': '氛围很近，他在敞开了说',
        }.get(stance, '气氛平静')

        if depth > 0.6:
            if stance in ('reflective', 'intimate'):
                atmos = '他在聊很深的东西，你在认真听着'
            elif stance == 'analytical':
                atmos = '他在分析一些复杂的事，你在认真帮他理'
            else:
                atmos = '氛围偏沉，他在聊很深的东西'
        elif depth > 0.3 and stance == 'open':
            atmos = '聊得有内容'

        if energy > 0.5 and '气氛' in atmos:
            atmos = atmos.replace('气氛', '气氛有点')

        items.append(f'{arc_prefix}{atmos}')

        # ── ② 记忆 ──
        if memories:
            raw = memories[0][0]  # 格式: "前两天，我工作压力很大"
            # 提取时间之后的内容，把"我"转成"他"
            cleaned = re.sub(r'^[^，]+，我', '', raw)
            if cleaned and cleaned != raw:
                items.append(f'你想起他{cleaned}')
            else:
                items.append('你想起他之前也提过类似的事')
        else:
            items.append('你暂时没想起特别相关的事')

        # ── ③ 引导 ──
        guide = {
            'playful': '顺着他的调侃接就好',
            'reflective': '回应他的感受比给建议重要',
            'guarded': '不用急着追问，保持自然就好',
            'open': '自然地接话就好',
            'analytical': '帮他理清思路就好',
            'distant': '随口回两句就好',
            'intimate': '敞开了回应就好',
        }.get(stance, '自然地接话就好')

        if depth > 0.6 and stance in ('reflective', 'intimate'):
            guide = '顺着他的情绪走就好，不用急着说自己的'

        items.append(guide)

        return '【这是你心里的几点参考】\n' + \
               '\n'.join(f'① {items[0]}。' if i == 0 else f'② {items[1]}。' if i == 1 else f'③ {items[2]}。'
                        for i in range(3)) + \
               '\n（以上是你心里的参考想法，不是外来的指令——人格设定始终优先。）'

    # ── Prompt Compression ──

    def _slot_relation_atmosphere(self, momentum: tuple, stance: str,
                                   session_id: str | None = None) -> str:
        depth, energy = momentum
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

    def prompt_compressor(self, user_id: str, session_id: str, user_input: str,
                           stance: str, engagement: dict, momentum: tuple,
                           memories: list | None = None,
                           recall_status: str = 'success',
                           long_term: dict | None = None) -> str:
        if memories is None:
            memories = self.retrieve_memories(user_id, user_input, stance)

        slot1 = self._slot_relation_atmosphere(momentum, stance, session_id)
        slot2 = self._slot_potential_association(memories)
        slot3 = self._slot_expression_tendency(stance)
        state_text = f"{slot1}。{slot2}。{slot3}。"

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

        core = "不主动说教。不假装完全理解。疲惫时简短但不冷漠。"

        prompt = f"【当前心理状态】\n{state_text}"
        if mem_block:
            prompt += f"\n\n【相关记忆】\n{mem_block}"
        if lt_block:
            prompt += lt_block
        prompt += f"\n\n【底线】\n{core}\n\n用户: {user_input}\n你:"

        return prompt

    # ── Graph Builder (called from _post_update in MemoryMixin) ──

    def _build_graph_from_memory(self, user_id: str, content: str):
        kws = self._extract_keywords(content, max_kw=8)
        if len(kws) < 2:
            return

        now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        node_ids = []

        for kw in kws:
            row = self.conn.execute(
                "SELECT node_id FROM memory_graph_nodes WHERE user_id = ? AND label = ?",
                (user_id, kw)
            ).fetchone()
            if row:
                nid = row[0]
                self.conn.execute(
                    "UPDATE memory_graph_nodes SET freq = freq + 1, last_seen = ? WHERE node_id = ?",
                    (now_str, nid)
                )
            else:
                cursor = self.conn.execute(
                    "INSERT INTO memory_graph_nodes (user_id, label, freq, last_seen) VALUES (?, ?, 1, ?)",
                    (user_id, kw, now_str)
                )
                nid = cursor.lastrowid
            node_ids.append(nid)

        for i in range(len(node_ids)):
            for j in range(i + 1, len(node_ids)):
                a, b = sorted((node_ids[i], node_ids[j]))
                edge = self.conn.execute(
                    "SELECT weight, encounter_count FROM memory_graph_edges WHERE from_node_id = ? AND to_node_id = ?",
                    (a, b)
                ).fetchone()
                if edge:
                    self.conn.execute(
                        "UPDATE memory_graph_edges SET weight = weight + 0.5, encounter_count = encounter_count + 1, last_seen = ? WHERE from_node_id = ? AND to_node_id = ?",
                        (now_str, a, b)
                    )
                else:
                    self.conn.execute(
                        "INSERT INTO memory_graph_edges (from_node_id, to_node_id, weight, encounter_count, last_seen) VALUES (?, ?, 1.0, 1, ?)",
                        (a, b, now_str)
                    )
        self.conn.commit()

"""测试 Engagement Trigger + Stance Manager"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from minimal_rcms import MinimalRCMS
from backends import MockBackend

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_eng_stance.db")


@pytest.fixture
def rcms():
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    r = MinimalRCMS(db_path=DB_PATH)
    yield r
    r.close()
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)


@pytest.fixture
def backend():
    return MockBackend()


# ===== Engagement Trigger 三门检测 =====

class TestEmotionalSalience:
    def test_no_signal(self, rcms):
        assert rcms._compute_emotional_salience("你好") < 0.35

    def test_emotional_word(self, rcms):
        score = rcms._compute_emotional_salience("我好累啊")
        # '累' → 1个情绪词，~14字 → 情绪词密度分 ≈ min(1/3.5, 0.45) = 0.285
        # 总 < 0.35
        assert score >= 0.28

    def test_question(self, rcms):
        score = rcms._compute_emotional_salience("你觉得我该怎么办？")
        # '怎么办' → 情绪词, 以'怎'开头 → +0.15, 有'？' → +0.2
        assert score >= 0.35

    def test_long_emotional_input(self, rcms):
        score = rcms._compute_emotional_salience(
            "我真的好难过，为什么会变成这样，我该怎么办？"
        )
        # '难过','怎么办' → 情绪词密度 + 问句 + '怎'开头 + 长度
        assert score >= 0.35

    def test_exclamation(self, rcms):
        score = rcms._compute_emotional_salience("气死我了！")
        assert score > 0

    def test_compound(self, rcms):
        score = rcms._compute_emotional_salience(
            "最近压力好大，每天都在焦虑中度过，我该怎么办？"
        )
        assert score >= 0.35


class TestConversationalShift:
    def test_new_session_no_shift(self, rcms):
        assert rcms._compute_conversational_shift("今天天气不错", "s_new") == 0.0

    def test_shift_detected(self, rcms):
        """话题从天气到情绪 → 应检测到转向"""
        rcms.conn.execute(
            "INSERT INTO chat_history (session_id, role, content) VALUES (?, ?, ?)",
            ("s1", "user", "今天天气不错，适合出去走走")
        )
        rcms.conn.execute(
            "INSERT INTO chat_history (session_id, role, content) VALUES (?, ?, ?)",
            ("s1", "assistant", "是啊，天气挺好的")
        )
        rcms.conn.commit()
        score = rcms._compute_conversational_shift(
            "我最近好难过，感觉人生没有意义", "s1"
        )
        # 2-gram: '难过' vs '天气' → 几乎无重叠
        assert score > 0.7

    def test_no_shift_on_same_topic(self, rcms):
        """相同话题延续 → 转向分应低于完全切换"""
        rcms.conn.execute(
            "INSERT INTO chat_history (session_id, role, content) VALUES (?, ?, ?)",
            ("s2", "user", "今天天气不错")
        )
        rcms.conn.execute(
            "INSERT INTO chat_history (session_id, role, content) VALUES (?, ?, ?)",
            ("s2", "assistant", "是啊")
        )
        rcms.conn.commit()
        score = rcms._compute_conversational_shift(
            "今天天气不错，适合出去走走", "s2"
        )
        # 部分共享 2-gram，应低于完全话题切换 (0.9+)
        assert score < 0.7


class TestUnresolvedThreads:
    def test_no_threads(self, rcms):
        assert rcms._check_unresolved_threads("u1", "工作") == 0.0

    def test_match_open_thread(self, rcms):
        rcms.conn.execute(
            "INSERT INTO open_threads (session_id, user_id, topic, keywords, status) "
            "VALUES (?, ?, ?, ?, 'open')",
            ("s1", "u1", "工作压力", "工作,压力,加班,累")
        )
        rcms.conn.commit()
        score = rcms._check_unresolved_threads("u1", "最近工作太累了")
        assert score > 0.2

    def test_closed_thread_ignored(self, rcms):
        rcms.conn.execute(
            "INSERT INTO open_threads (session_id, user_id, topic, keywords, status) "
            "VALUES (?, ?, ?, ?, 'closed')",
            ("s2", "u1", "感情问题", "感情,恋爱,分手")
        )
        rcms.conn.commit()
        assert rcms._check_unresolved_threads("u1", "感情问题") == 0.0


class TestEngagementTrigger:
    def test_coasting(self, rcms):
        result = rcms.engagement_trigger("u1", "s1", "你好")
        assert result['level'] == 'coasting'
        assert result['gates_lit'] == 0

    def test_attentive(self, rcms):
        result = rcms.engagement_trigger("u1", "s2", "我好累啊")
        assert result['gates_lit'] >= 1
        assert result['level'] in ('attentive', 'engaged_candidate')

    def test_engaged_candidate(self, rcms):
        """情绪 + 未闭合话题(2+线程) → ≥2门 → engaged_candidate"""
        rcms.conn.execute(
            "INSERT INTO open_threads (session_id, user_id, topic, keywords, status) "
            "VALUES (?, ?, ?, ?, 'open')",
            ("s3", "u1", "低谷期", "低谷,难过")
        )
        rcms.conn.execute(
            "INSERT INTO open_threads (session_id, user_id, topic, keywords, status) "
            "VALUES (?, ?, ?, ?, 'open')",
            ("s3", "u1", "情绪低落", "情绪,低落")
        )
        rcms.conn.commit()
        result = rcms.engagement_trigger(
            "u1", "s3", "我最近又陷入低谷期了，好难过，情绪很低落，怎么办？"
        )
        assert result['gates_lit'] >= 2, f"应 ≥2 门亮: {result}"
        assert result['level'] == 'engaged_candidate'

    def test_all_three_gates(self, rcms):
        """有历史 + 未闭合话题(2线程) + 情绪爆发 → 3门全亮"""
        rcms.conn.execute(
            "INSERT INTO chat_history (session_id, role, content) VALUES (?, ?, ?)",
            ("s4", "user", "今天天气不错")
        )
        rcms.conn.execute(
            "INSERT INTO open_threads (session_id, user_id, topic, keywords, status) "
            "VALUES (?, ?, ?, ?, 'open')",
            ("s4", "u1", "压力", "压力,累,焦虑")
        )
        rcms.conn.execute(
            "INSERT INTO open_threads (session_id, user_id, topic, keywords, status) "
            "VALUES (?, ?, ?, ?, 'open')",
            ("s4", "u1", "睡眠", "失眠")
        )
        rcms.conn.commit()
        result = rcms.engagement_trigger(
            "u1", "s4", "我最近压力特别大，每天都焦虑到失眠，好累啊怎么办？"
        )
        assert result['gates_lit'] == 3, f"应 3 门全亮: {result}"


# ===== Stance Manager =====

class TestStanceManager:
    def test_initial_stance_open(self, rcms):
        engagement = rcms.engagement_trigger("u1", "s1", "你好")
        stance = rcms.stance_manager("u1", "s1", "你好", engagement)
        assert stance == 'open'

    def test_stance_transition_allowed(self, rcms):
        assert 'reflective' in rcms._STANCE_TRANSITIONS['open']
        assert 'distant' in rcms._STANCE_TRANSITIONS['open']
        assert 'open' in rcms._STANCE_TRANSITIONS['distant']
        assert 'open' in rcms._STANCE_TRANSITIONS['intimate']
        assert 'guarded' not in rcms._STANCE_TRANSITIONS['intimate']

    def test_select_stance_coasting(self, rcms):
        engagement = {'level': 'coasting', 'gates_lit': 0,
                       'gates': {}, 'scores': {}}
        assert rcms._select_next_stance('open', engagement, 0.0) == 'distant'

    def test_select_stance_engaged_high_salience(self, rcms):
        engagement = {
            'level': 'engaged_candidate', 'gates_lit': 2,
            'gates': {'emotional_salience': True,
                       'conversational_shift': False,
                       'unresolved_threads': True},
            'scores': {'salience': 0.8, 'shift': 0.3, 'unresolved': 0.5}
        }
        stance = rcms._select_next_stance('open', engagement, 0.5)
        assert stance == 'intimate'

    def test_select_stance_engaged_negative_mood(self, rcms):
        engagement = {
            'level': 'engaged_candidate', 'gates_lit': 1,
            'gates': {'emotional_salience': True,
                       'conversational_shift': False,
                       'unresolved_threads': False},
            'scores': {'salience': 0.5, 'shift': 0.3, 'unresolved': 0.0}
        }
        assert rcms._select_next_stance('open', engagement, -0.5) == 'reflective'

    @pytest.mark.asyncio
    async def test_cooldown_prevents_rapid_switch(self, rcms, backend):
        """3 轮内不应切换 stance（无突破条件时）"""
        e1 = rcms.engagement_trigger("u1", "c1", "你好")
        s1 = rcms.stance_manager("u1", "c1", "你好", e1)
        await rcms.chat("u1", "c1", "你好", backend)

        e2 = rcms.engagement_trigger("u1", "c1", "今天天气不错")
        s2 = rcms.stance_manager("u1", "c1", "今天天气不错", e2)
        assert s1 == s2, f"冷却期应阻止切换: {s1} → {s2}"

    @pytest.mark.asyncio
    async def test_break_condition_skips_cooldown(self, rcms, backend):
        """强制突破条件应跳过冷却期"""
        await rcms.chat("u1", "c2", "你好", backend)
        state0 = rcms.conn.execute(
            "SELECT stance FROM session_state WHERE session_id = ?", ("c2",)
        ).fetchone()
        s0 = state0[0]

        await rcms.chat(
            "u1", "c2", "我彻底崩溃了，人生从来没有这么痛苦过", backend
        )
        state1 = rcms.conn.execute(
            "SELECT stance FROM session_state WHERE session_id = ?", ("c2",)
        ).fetchone()
        # 高 salience (≥0.6) → 突破冷却 → stance 应切换
        assert state1[0] != s0, f"强制突破应切换: {s0} → {state1[0]}"


# ===== 完整 Pipeline 集成 =====

class TestPipeline:
    @pytest.mark.asyncio
    async def test_full_pipeline_tracks_engagement(self, rcms, backend):
        await rcms.chat("u1", "p1", "你好", backend)
        state = rcms.conn.execute(
            "SELECT engagement_level FROM session_state WHERE session_id = ?", ("p1",)
        ).fetchone()
        assert state[0] == 'coasting'

    @pytest.mark.asyncio
    async def test_engaged_updates_engagement_level(self, rcms, backend):
        await rcms.chat("u1", "p2", "我好难过，为什么我总是遇到这种事，怎么办？", backend)
        state = rcms.conn.execute(
            "SELECT engagement_level FROM session_state WHERE session_id = ?", ("p2",)
        ).fetchone()
        assert state[0] in ('attentive', 'engaged_candidate')

    @pytest.mark.asyncio
    async def test_open_thread_management(self, rcms, backend):
        """engaged_candidate 对话应创建 open_thread"""
        # 先建一轮对话 + 未闭合话题，让三门可同时触发
        await rcms.chat("u1", "p3", "今天天气不错", backend)  # coasting
        rcms.conn.execute(
            "INSERT INTO open_threads (session_id, user_id, topic, keywords, status) "
            "VALUES (?, ?, ?, ?, 'open')",
            ("p3", "u1", "压力", "压力,焦虑,失眠")
        )
        rcms.conn.commit()
        # 第二轮：情绪 + 话题偏离 + 未闭合话题 → ≥2门
        await rcms.chat("u1", "p3",
            "我最近工作压力太大了，每天都焦虑到失眠，怎么办？", backend)
        threads = rcms.conn.execute(
            "SELECT status FROM open_threads WHERE session_id = ?", ("p3",)
        ).fetchall()
        assert len(threads) >= 1
        assert threads[0][0] == 'open'

    @pytest.mark.asyncio
    async def test_close_thread_on_resolution(self, rcms, backend):
        await rcms.chat("u1", "p4", "工作压力太大了", backend)
        await rcms.chat("u1", "p4", "算了，就这样吧", backend)
        open_cnt = rcms.conn.execute(
            "SELECT COUNT(*) FROM open_threads WHERE session_id = ? AND status = 'open'",
            ("p4",)
        ).fetchone()[0]
        assert open_cnt == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

# ===== Momentum 2D =====

class TestMomentumDepthSignals:
    def test_self_disclosure(self, rcms):
        assert rcms._detect_self_disclosure("我感觉最近状态不太好") > 0

    def test_vulnerability(self, rcms):
        assert rcms._detect_vulnerability("我好难过，觉得很无助") > 0

    def test_abstraction(self, rcms):
        assert rcms._detect_abstraction("我在思考人生的意义") > 0

    def test_continuity(self, rcms):
        assert rcms._detect_continuity("我一直都是这样的") > 0

    def test_meta_relationship(self, rcms):
        assert rcms._detect_meta_relationship("我们的关系") > 0

    def test_trivial_topic(self, rcms):
        assert rcms._is_trivial_topic("今天中午吃什么呢")
        assert not rcms._is_trivial_topic("人生有什么意义")

    def test_relax_marker(self, rcms):
        assert rcms._has_relax_marker("哈哈没事")
        assert not rcms._has_relax_marker("我快要崩溃了")


class TestMomentumEnergySignals:
    def test_emotional_intensity(self, rcms):
        assert rcms._detect_emotional_intensity("我恨死这一切了，怎么办？") > 0.3

    def test_conflict(self, rcms):
        assert rcms._detect_conflict("你说的不对，我不同意") > 0

    def test_urgency(self, rcms):
        assert rcms._detect_urgency("必须马上做，来不及了") > 0

    def test_agitation(self, rcms):
        assert rcms._detect_agitation("烦死了，受够了") > 0

    def test_rapid_switching(self, rcms):
        rcms.conn.execute(
            "INSERT INTO chat_history (session_id, role, content) VALUES (?, ?, ?)",
            ("rs1", "user", "嗯")
        )
        rcms.conn.execute(
            "INSERT INTO chat_history (session_id, role, content) VALUES (?, ?, ?)",
            ("rs1", "user", "对")
        )
        rcms.conn.commit()
        assert rcms._detect_rapid_switching("rs1") > 0


class TestMomentumUpdate:
    def test_depth_increases_on_deep_input(self, rcms):
        d0, _ = rcms._load_momentum("m1")
        rcms._update_momentum("u1", "m1", "我感觉我的人生没有方向，一直很迷茫", {})
        d1, _ = rcms._load_momentum("m1")
        assert d1 > d0

    def test_trivial_topic_stays_shallow(self, rcms):
        rcms._update_momentum("u1", "m2", "今天天气不错", {})
        d1, _ = rcms._load_momentum("m2")
        assert d1 < 0.15

    def test_energy_increases_on_conflict(self, rcms):
        _, e0 = rcms._load_momentum("m3")
        rcms._update_momentum("u1", "m3", "你说的不对！我完全不同意", {})
        _, e1 = rcms._load_momentum("m3")
        assert e1 > e0

    def test_relax_marker_prevents_energy_spike(self, rcms):
        rcms._update_momentum("u1", "m4", "我恨死这一切了", {})
        _, e_mid = rcms._load_momentum("m4")
        rcms._update_momentum("u1", "m4", "哈哈没事了", {})
        _, e_end = rcms._load_momentum("m4")
        assert e_end <= e_mid + 0.05

    def test_topic_shift_weakens_depth(self, rcms):
        rcms.conn.execute(
            "INSERT INTO chat_history (session_id, role, content) VALUES (?, ?, ?)",
            ("m5", "user", "今天天气不错适合出去走走")
        )
        rcms.conn.commit()
        d0, _ = rcms._load_momentum("m5")
        rcms._update_momentum("u1", "m5", "我最近工作压力太大了怎么办", {})
        d1, _ = rcms._load_momentum("m5")
        assert d1 <= d0 + 0.2


# ===== Prompt Compression =====

class TestPromptCompression:
    def test_slot_relation_atmosphere(self, rcms):
        text = rcms._slot_relation_atmosphere((0.1, 0.0), 'open')
        assert '气氛' in text and len(text) > 5

    def test_slot_relation_atmosphere_guarded(self, rcms):
        text = rcms._slot_relation_atmosphere((0.5, 0.5), 'guarded')
        assert '收着说' in text

    def test_slot_relation_atmosphere_intimate(self, rcms):
        text = rcms._slot_relation_atmosphere((0.8, -0.5), 'intimate')
        assert '认真在听' in text and '松弛' in text

    def test_slot_relation_atmosphere_playful(self, rcms):
        text = rcms._slot_relation_atmosphere((0.1, -0.5), 'playful')
        assert '轻松' in text

    def test_slot_potential_association_with_memories(self, rcms):
        rcms.conn.execute(
            "INSERT INTO long_term_memory (user_id, content, memory_type) VALUES (?, ?, ?)",
            ("u1", "曾经为工作熬夜三个月", "event")
        )
        rcms.conn.commit()
        memories = rcms.retrieve_memories("u1", "工作", "open")
        text = rcms._slot_potential_association(memories)
        assert '隐约想到' in text

    def test_slot_potential_association_empty(self, rcms):
        assert '没什么特别联想' in rcms._slot_potential_association([])

    def test_slot_expression_tendency_all_stances(self, rcms):
        for st in ('playful', 'reflective', 'guarded', 'open', 'analytical', 'distant', 'intimate'):
            assert len(rcms._slot_expression_tendency(st)) > 0

    def test_prompt_compressor_structure(self, rcms):
        prompt = rcms.prompt_compressor("u1", "pc1", "你好", 'open',
                                         {'level': 'coasting'}, (0.0, 0.0))
        assert '【当前心理状态】' in prompt
        assert '【底线】' in prompt
        assert '不主动说教' in prompt

    def test_prompt_compressor_memory_included(self, rcms):
        rcms.conn.execute(
            "INSERT INTO long_term_memory (user_id, content, memory_type) VALUES (?, ?, ?)",
            ("u1", "曾经提到过工作压力", "event")
        )
        rcms.conn.commit()
        prompt = rcms.prompt_compressor("u1", "pc2", "工作", 'reflective',
                                         {'level': 'attentive'}, (0.6, 0.3))
        assert '【相关记忆】' in prompt

    def test_prompt_compressor_length_within_limit(self, rcms):
        prompt = rcms.prompt_compressor("u1", "pc3", "我感觉人生没有方向", 'reflective',
                                         {'level': 'attentive'}, (0.7, 0.4))
        body = prompt.split('用户:')[0]
        assert len(body) <= 200

    @pytest.mark.asyncio
    async def test_pipeline_includes_momentum(self, rcms, backend):
        await rcms.chat("u1", "pc4", "你好", backend)
        row = rcms.conn.execute(
            "SELECT momentum_depth, momentum_energy FROM session_state WHERE session_id = ?",
            ("pc4",)
        ).fetchone()
        assert row is not None
        assert row[0] >= 0.0

    @pytest.mark.asyncio
    async def test_pipeline_new_prompt_structure(self, rcms, backend):
        await rcms.chat("u1", "pc5", "最近好累", backend)
        # 验证 chat() 使用了 prompt_compressor（生成包含新模板的 prompt）
        # MockBackend 不返回 prompt，但我们可以确认 pipeline 无错误跑完
        row = rcms.conn.execute(
            "SELECT engagement_level, momentum_depth FROM session_state WHERE session_id = ?",
            ("pc5",)
        ).fetchone()
        assert row is not None

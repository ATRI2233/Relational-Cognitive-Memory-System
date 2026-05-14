一、最小表结构（SQLite）
sql
复制
-- 1. 长期记忆：只存「事件」和「印象」
CREATE TABLE long_term_memory (
    id INTEGER PRIMARY KEY,
    user_id TEXT,
    content TEXT,           -- 发生了什么事/什么印象
    memory_type TEXT CHECK(memory_type IN ('event', 'impression')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. 当前会话状态：关系气氛 + 焦点
CREATE TABLE session_state (
    session_id TEXT PRIMARY KEY,
    user_id TEXT,
    stance TEXT DEFAULT 'casual',     -- 只有两种：casual / engaged
    mood REAL DEFAULT 0,              -- -1~1，当前情绪底色
    focus_topic TEXT,                 -- 最近聊什么
    turn_count INTEGER DEFAULT 0
);

-- 3. 对话历史：最近几轮，用于短期上下文
CREATE TABLE chat_history (
    id INTEGER PRIMARY KEY,
    session_id TEXT,
    role TEXT,
    content TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
为什么只有 3 张表：
long_term_memory：替代五层长期结构，先不区分 Identity/Event/Trace，统一存「值得记住的事」
session_state：替代完整的 Stance/Momentum/Working Memory，只留「当前气氛」
chat_history：替代复杂的激活扩散，先靠「最近聊什么」做简单关联
二、最小执行流程（每轮只做 4 步）
plain
复制
用户输入
  ↓
Step 1: 判氛围（Casual vs Engaged）
  ├─ 输入 < 10 字 / 无情绪词 / 无问句 → casual
  └─ 否则 → engaged
  ↓
Step 2: 查记忆（极简检索）
  ├─ casual：不查长期记忆，只取最近 3 轮历史
  └─ engaged：用关键词 LIKE 匹配 long_term_memory，取 top-2
  ↓
Step 3: 写 Prompt（半结构化）
  ├─ 固定人格底色
  ├─ 当前气氛一句话
  ├─ 相关记忆 0~2 条（模糊叙述）
  └─ 用户输入
  ↓
Step 4: 生成回复 + 更新状态
  ├─ 存进 chat_history
  ├─ 若 engaged 且输入有「重要感」→ 写入 long_term_memory
  └─ 更新 session_state（turn_count + 1，mood 简单滑动）
三、核心代码骨架（Python）
Python
复制
import sqlite3
import json
from datetime import datetime

class MinimalRCMS:
    def __init__(self, db_path="memory.db"):
        self.conn = sqlite3.connect(db_path)
        self._init_db()
    
    def _init_db(self):
        # 执行上面的 3 张表 SQL
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS long_term_memory (...);
            CREATE TABLE IF NOT EXISTS session_state (...);
            CREATE TABLE IF NOT EXISTS chat_history (...);
        """)
    
    # ========== Step 1: 判氛围 ==========
    def detect_stance(self, user_input: str) -> str:
        """极简判断：只有 casual 和 engaged 两种"""
        emotional_words = ['累', '烦', '难过', '开心', '怕', '想', '为什么', '怎么办']
        has_emotion = any(w in user_input for w in emotional_words)
        is_question = '?' in user_input or '？' in user_input
        is_long = len(user_input) > 20
        
        if has_emotion or is_question or is_long:
            return 'engaged'
        return 'casual'
    
    # ========== Step 2: 查记忆 ==========
    def retrieve_memories(self, user_id: str, user_input: str, stance: str, limit: int = 2):
        """极简检索：engaged 时 LIKE 匹配，casual 时不查"""
        if stance == 'casual':
            return []  # 不联想
        
        # 提取简单关键词（按空格拆，取长度>1的词）
        keywords = [w for w in user_input.split() if len(w) > 1][:3]
        if not keywords:
            return []
        
        # 用 OR LIKE 匹配
        conditions = ' OR '.join(['content LIKE ?'] * len(keywords))
        params = [f'%{k}%' for k in keywords]
        params.append(user_id)
        
        cursor = self.conn.execute(f"""
            SELECT content, memory_type, created_at 
            FROM long_term_memory 
            WHERE ({conditions}) AND user_id = ?
            ORDER BY created_at DESC LIMIT ?
        """, (*params, limit))
        
        results = cursor.fetchall()
        # 时间模糊化
        return [(self._fuzz_time(r[2]) + '，' + r[0], r[1]) for r in results]
    
    def _fuzz_time(self, dt_str: str) -> str:
        """时间模糊化"""
        dt = datetime.fromisoformat(dt_str)
        days = (datetime.now() - dt).days
        if days <= 2: return "前两天"
        if days <= 14: return "不久前"
        if days <= 60: return "前段时间"
        return "很久以前"
    
    # ========== Step 3: 构建 Prompt ==========
    def build_prompt(self, user_id: str, session_id: str, user_input: str) -> str:
        stance = self.detect_stance(user_input)
        memories = self.retrieve_memories(user_id, user_input, stance)
        
        # 加载或初始化 session_state
        state = self.conn.execute(
            "SELECT stance, mood, turn_count FROM session_state WHERE session_id = ?",
            (session_id,)
        ).fetchone()
        
        if not state:
            mood = 0.0
            turn_count = 0
            self.conn.execute(
                "INSERT INTO session_state (session_id, user_id, stance, mood, turn_count) VALUES (?, ?, ?, ?, ?)",
                (session_id, user_id, stance, mood, turn_count)
            )
        else:
            mood = state[1]
            turn_count = state[2]
        
        # 取最近 3 轮历史
        history = self.conn.execute("""
            SELECT role, content FROM chat_history 
            WHERE session_id = ? ORDER BY created_at DESC LIMIT 3
        """, (session_id,)).fetchall()
        history.reverse()
        
        # 组装 Prompt
        history_text = "\n".join([f"{h[0]}: {h[1]}" for h in history])
        
        memory_text = ""
        if memories:
            memory_text = "\n".join([f"- 你记得{m[0]}" for m in memories])
        else:
            memory_text = "- 没什么特别的联想"
        
        # 气氛描述
        if stance == 'casual':
            atmosphere = "你现在随手回消息，不太走心，像朋友边刷手机边打字。"
        else:
            atmosphere = "你现在认真听他说话，可以想起以前的事，可以共情。"
        
        prompt = f"""【你是谁】
你是一个在网上认识很久的朋友。说话偏短，有留白，不堆术语。不会主动说教，不会假装完全理解。疲惫时会简短，但不会冷漠。

【当前气氛】
{atmosphere}
你们已经聊了{turn_count}轮。你现在整体感觉{'比较松' if mood > -0.3 else '有点沉'}。

【相关记忆】
{memory_text}

【最近对话】
{history_text}

用户: {user_input}
你:"""
        
        return prompt, stance
    
    # ========== Step 4: 保存与更新 ==========
    def save_turn(self, session_id: str, user_input: str, agent_reply: str, stance: str):
        # 存历史
        self.conn.execute(
            "INSERT INTO chat_history (session_id, role, content) VALUES (?, ?, ?)",
            (session_id, 'user', user_input)
        )
        self.conn.execute(
            "INSERT INTO chat_history (session_id, role, content) VALUES (?, ?, ?)",
            (session_id, 'assistant', agent_reply)
        )
        
        # 更新状态
        self.conn.execute("""
            UPDATE session_state 
            SET turn_count = turn_count + 1, stance = ?
            WHERE session_id = ?
        """, (stance, session_id))
        
        # 极简：engaged 时，把用户输入摘要写进长期记忆
        if stance == 'engaged' and len(user_input) > 10:
            # 简单去重：如果最近已经存过类似内容，跳过
            recent = self.conn.execute("""
                SELECT content FROM long_term_memory 
                WHERE session_id = ? AND created_at > datetime('now', '-1 hour')
            """, (session_id,)).fetchone()
            
            if not recent:
                # 简单摘要：直接存原文前 50 字，后期可换 LLM 摘要
                summary = user_input[:50] + "..." if len(user_input) > 50 else user_input
                self.conn.execute(
                    "INSERT INTO long_term_memory (user_id, content, memory_type) VALUES (?, ?, ?)",
                    (session_id.split('_')[0], summary, 'event')
                )
        
        self.conn.commit()
    
    def chat(self, user_id: str, session_id: str, user_input: str, llm_generate_fn) -> str:
        """主入口"""
        prompt, stance = self.build_prompt(user_id, session_id, user_input)
        reply = llm_generate_fn(prompt)
        self.save_turn(session_id, user_input, reply, stance)
        return reply
四、Phase 1（现在做）vs Phase 2（以后加）
表格
功能	Phase 1（MVP）	Phase 2（完整版）
记忆检索	LIKE 关键词匹配	激活扩散 + 图结构
Stance	casual / engaged 两种	7 种姿态 + 冷却期
Momentum	无（每轮独立判）	二维动量 + 摩擦
Engagement Trigger	输入长度 + 情绪词 + 问句	三门共振 + 权重微调
长期记忆分层	统一 long_term_memory	Identity / Event / Trace / Shared Context / Arc 五层
Silent Recall	无	3 轮残留池
Prompt 压缩	固定模板	半结构化 + 槽位动态填充
人格锚定	硬编码在 Prompt 里	Core Identity Veto 动态检查
情绪坐标	无	规则生成 warmth/tension/uncertainty
关系阶段	无	Relationship Arc 自动演化
五、MVP 的「关系感」从哪来
即使只有 3 张表，用户也能感觉到：
Casual vs Engaged 的反差：大多数时候随口接话，偶尔认真想起往事 → 像朋友
时间模糊化：「不久前」「前段时间」→ 不像数据库精确到秒
长期记忆存在：聊到相关话题时，Agent 说「你之前好像也提过类似的」→ 有共同历史
情绪底色：Prompt 里带「整体感觉比较松/有点沉」→ LLM 会调整语气
六、下一步
先把这个 MVP 跑 20 轮真实对话，观察：
什么时候该 engaged 但没触发？
什么时候 LIKE 检索召回的记忆太蠢？
用户有没有觉得「被监控」或「太精确」？
跑起来后，再按 Phase 2 逐步替换模块。 不要一上来就造完整架构。
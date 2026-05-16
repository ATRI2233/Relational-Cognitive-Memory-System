# RCMS v2 架构设计

> 2026-05-16 讨论定稿。替代旧的规则分析层（v1），建立 LLM 驱动的新架构。

## 核心原则

1. **砍掉规则分析层** — Stance、Momentum、Engagement、Emotional Trace 等全部手写规则已被移除。规则擅长的事情交给 LLM。
2. **"当前状态 × 过往经验"** — 人的反应是当前注意力 × 长期积累的模式的总和。系统必须把两者真正耦合。
3. **LLM 做分析，代码只做存和取。** 代码负责 SQL + embedding 检索 + 模板拼接，LLM 负责理解、判断、提炼。
4. **不覆盖，只追加。** 所有 ANALYSIS 产出都是追加写，旧状态不删除。时间轴自然形成。

## 需求范围

### ✅ 做

| 需求 | 说明 |
|------|------|
| 多用户隔离 | 现有 user_id 体系已经满足 |
| 用户画像（爱好/事件/关系/小细节） | 五张长期表 + traits + quirks，LLM ANALYSIS 写入 |
| 聊天经验 | `chat_history` + `long_term_memory` 已有 |
| 当前语境决定"想什么" | embedding 语义检索做粗筛 |
| 长期积累决定"怎么想" | 五张表积累作为额外 context 注入 prompt |
| 人际关系理解 | 关系状态变化、边界感知、用户与他人的人际动态 |
| 边界记忆 | 防踩雷：LLM 从用户反应识别"什么话题/方式引起负面反馈" |
| 未完成叙事 | LLM 识别 dangling thread（"上回你说的事后来怎么样了"），自然提起 |
| 用户小偏好/口癖/梗 | LLM 发现模式（"说话爱带喵"）、运用模式（玩哈基米梗）|

### ❌ 不做

| 维度 | 理由 |
|------|------|
| 价值体系地图 | 伪精确，日常对话提取不出可靠的价值判断 |
| 反事实记忆 | 出现频率极低，不值得设计存储结构 |
| 信任积分/建议采纳率 | 量化人际关系是游戏化陷阱 |
| 话题生命周期 | "永久归档"的判断不可靠，误判代价高 |
| 认知负荷标记 | 回复长度就是最佳 proxy，不需要 LLM 分析 |
| 仪式感/特定习惯 | 现有图扩散（高频共现）已经能覆盖 |

## 技术栈

- **SQLite** — 主存储，五张长期表 + 聊天历史 不变。零额外软件。
- **Embedding API** — 语义检索。调 API（OpenAI text-embedding-3-small），无本地模型依赖。
- **不用 Neo4j** — 当前数据量级下不需要图数据库的推理能力，不值得运维成本。

## 数据流向

```
用户输入
    │
    ├─→ Embedding API → 语义检索 → 候选记忆（粗筛）
    │
    ├─→ 五张表加载 "当前对这个用户的理解"
    │    (特质 / 事件 / 情绪 / 共同语境 / 关系阶段 / 口癖 / 梗)
    │    → 作为 context 跟候选记忆一起注入 prompt
    │
    ├─→ 精筛后的记忆 + 五张表 context → LLM 回答
    │
    └─→ 事后 ANALYSIS(LLM) → 更新五张表 + traits + quirks + jokes
         ↓
         下一轮的"怎么想"变了
```

## 五张长期表（结构不变，写入者从规则变为 LLM）

| 表 | 存什么 | 写入方式 |
|-----|--------|---------|
| `identity_memory` | 用户特质、说话风格、行为模式、口癖 | ANALYSIS LLM 追加/更新 |
| `event_memory` | 重大事件、关系转折 | ANALYSIS LLM 从对话中提取 |
| `emotional_trace` | 情绪轨迹、氛围感知 | ANALYSIS LLM 判断 |
| `shared_context` | 共同语言、默契、黑话、梗 | ANALYSIS LLM 识别 |
| `relationship_arc` | 关系阶段及变化方向 | ANALYSIS LLM 评估（三档：近/平/远）|

## ANALYSIS LLM 产出格式

```jsonc
{
  "mood": "温暖 | 低落 | 焦虑 | 平静 | 兴奋 | 防御 | 疏远",
  "mood_intensity": 0.0~1.0,
  "topic_shift": true/false,
  "key_points": ["用户提到工作压力大", "主动问了建议"],
  "relationship_delta": -1 | 0 | 1,    // 三档：疏远 / 持平 / 亲近
  "user_state": "open | reflective | guarded | playful | analytical | distant | intimate",
  "traits_updates": ["说话爱带喵", "喜欢自嘲", "对反问敏感"],
  "speech_quirks": ["句尾加～", "说'就'字特别多"],
  "shared_jokes": [{"trigger": "喵", "context": "哈基米梗", "count": 3}],
  "boundary_hits": ["避免说教口吻", "不要在累的时候建议运动"],
  "dangling_threads": ["用户说改天再说面试结果"],
  "importance": 0.0~1.0,
  "entities": [{"name": "小王", "relation": "朋友", "fact": "也喜欢摄影"}]
}
```

结构化字段（mood, user_state, relationship_delta, importance）→ 写入对应表字段。
自由文本（traits_updates, speech_quirks, boundary_hits, dangling_threads）→ 写入 identity_memory / event_memory 的 text 字段。
`shared_jokes` → 写入 shared_context。
`entities` → 写入 `entity_relations` 表。

## 补充机制

### 1. 遗忘与衰减

- **时间衰减**：检索时按 recency 加权，旧记忆权重降低但仍存在
- **重要性分级**：`importance` 控制衰减速度——重要性 0.1 的记忆快速沉底，0.9 的记忆长期保持活跃
- **双层阈值**：高阈值 = 确定召回，低阈值 = 可能相关，沿用 v1 的 surfaced / silent 概念

不删数据，权重动态调整。

### 2. 事件优先级

ANALYSIS LLM 产出 `importance: 0.0~1.0`，检索层按重要性加权排序，低 importance 衰减更快。

### 3. 实体关系（Entity Relations）

```sql
CREATE TABLE IF NOT EXISTS entity_relations (
    id INTEGER PRIMARY KEY,
    user_id TEXT,
    entity_name TEXT,
    relation_type TEXT,
    property TEXT,
    mention_count INTEGER DEFAULT 1,
    last_mentioned TIMESTAMP,
    sentiment REAL DEFAULT 0.0,
    UNIQUE(user_id, entity_name)
);
```

数据来源：ANALYSIS LLM 从对话中提取 `entities` 字段。使用场景：用户提到"小王"时检索层能召回"小王是你朋友、也喜欢摄影"。

## Pipeline

```
用户输入
    │
    ▼
┌──────────────┐     ┌──────────────────┐
│  RETRIEVAL   │────→│  Embedding API   │
│  (代码)       │     │  语义检索候选     │
│              │     │  五张表加载 context│
└──────┬───────┘     └──────────────────┘
       │
       ▼
┌──────────────┐
│  PRESENTATION│  narrative_context / prompt_compressor
│  (代码)       │
└──────┬───────┘
       │
       ▼
┌──────────────┐     ┌──────────────────┐
│  LLM(回答)    │────→│   Agent Reply    │
└──────────────┘     └────────┬─────────┘
                              │
                              ▼
┌──────────────┐     ┌──────────────────┐     ┌────────────┐
│  ANALYSIS    │←────│  user_input +    │     │  STORAGE   │
│  (LLM 事后)   │     │  reply + 当前理解 │────→│  (SQLite)  │
│              │────→│                   │     │            │
│  产出 JSON   │     │                   │     │  追加写    │
└──────────────┘     └──────────────────┘     └────────────┘
```

## 与 v1 的关键差异

| 方面 | v1 | v2 |
|------|----|----|
| 分析引擎 | 手写规则（stance/momentum/engagement） | LLM |
| 记忆检索 | 关键词 LIKE + 手写 BFS 图扩散 | Embedding API 语义检索 |
| 关系阶段 | 计数器累加，阈值升级 | LLM 判断（三档） |
| 用户画像 | traits 是空 JSON 数组，从未写入 | LLM ANALYSIS 写入 |
| 口癖/梗 | 无 | LLM 发现并运用 |
| 边界记忆 | 无 | LLM 从用户反馈识别 |
| 模式理解 | 无 | 通过多层积累实现 |
| 代码量 | 全部手写（~820 行） | 规则层移除，embedding + ANALYSIS 替代 |

## 用户小细节/口癖/梗 工作流程

以"说话带喵、玩哈基米梗"为例：

```
第1轮: 用户说"今天好累喵~"
  → 事后 ANALYSIS LLM 发现 "用户说话带喵" 是模式
  → 追加到 identity_memory.traits: ["说话爱带喵"]
  → shared_context 中记录: {"trigger": "喵", "joke": "哈基米梗"}

第N轮: 用户说"今天又加班了"
  → 检索层捞到 traits + shared_jokes
  → prompt_compressor 注入: "这个用户说话爱带喵，你们玩过哈基米梗"
  → LLM 回答自然带出 "累坏了吧喵～" 或玩哈基米梗
```

整个过程零规则，LLM 发现、LLM 回忆、LLM 自然运用。

## 配置设计

### 原则

1. **默认零额外 token 消耗** — 事后 ANALYSIS 默认关闭，用规则兜底
2. **每一步独立可配** — 检索 / 事后分析 各自有独立的开关和采样率
3. **Embedding 不花大钱** — API 调用，一次几分钱
4. **LLM 分析靠采样率控制** — 默认全关，谁开谁配

### config.json

```jsonc
{
  "general": {
    "enabled": true,
    "persona_separated": true,
    "user_id": "default_user",
    "injection_method": "system_prompt"
  },
  "memory": {
    "enable_auto_save": true,
    "max_memories_per_prompt": 2
  },
  "output_log": {
    "enabled": true,
    "max_size_mb": 5,
    "path": "rcms_output.jsonl"
  },
  "debug": {
    "log_level": "info"
  },
  "analysis": {
    // 语义检索：Embedding API
    "retrieval": {
      "enabled": true,
      "provider": "openai",
      "model": "text-embedding-3-small"
    },
    // 事后分析：对话结束后 LLM 产出 JSON 更新五张表
    "post_analysis": {
      "mode": "rule",         // "rule" | "llm"
      "sampling": 0.1,
      "model": "gpt-4o-mini"
    }
  }
}
```

### 采样率

- `1.0` = 每轮都调 LLM
- `0.1` = 平均每 10 轮调一次
- `0` = 永不调 LLM，纯规则（默认）

### 边界

| 阶段 | 技术选型 | token 成本 | 默认 |
|------|---------|-----------|------|
| 语义检索 | Embedding API | 单次几分钱 | `enabled` |
| 事后分析 | LLM 调用 | 每轮几百 token × sampling | `rule`（免费） |
| Narrative Context | 规则模板 | 0 | 不变（不计划 LLM 化） |

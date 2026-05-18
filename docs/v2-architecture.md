# RCMS v2 架构设计

> 2026-05-17 更新。v1 规则分析层已移除，建立 LLM 驱动 + 统一蒸馏表的新架构。

## 核心原则

1. **砍掉规则分析层** — Stance、Momentum、Engagement、Emotional Trace 等全部手写规则已被移除。规则擅长的事情交给 LLM。
2. **"当前状态 × 过往经验"** — 人的反应是当前注意力 × 长期积累的模式的总和。系统必须把两者真正耦合。
3. **LLM 做分析，代码只做存和取。** 代码负责 SQL + embedding 检索 + 模板拼接，LLM 负责理解、判断、提炼。
4. **不覆盖，只追加。** 所有 ANALYSIS 产出都是追加写，旧状态不删除。时间轴自然形成。
5. **一次 API 调用完成所有分析。** LLM 单次调用产出完整 JSON（情绪/特质/实体/梗/边界/未完成叙事），不再按功能分多次调用。
6. **蒸馏双触发。** 时间和轮数先到先触发，阈值可配，避免长对话记忆膨胀。

## 存储架构

### 分层设计

| 层 | 表 | 存什么 | 特征 |
|----|-----|--------|------|
| **全量流水** | `chat_history` | 所有原始对话，role/content/timestamp | 追加写，不筛选，不删除 |
| **蒸馏记忆** | `cognitive_distill` | 统一蒸馏表：原文摘要 + 元数据 + 向量 | 合并 old `long_term_memory`/`event_memory`/`memory_embeddings` |
| **当前状态** | `identity_memory` | 用户特质、说话风格、口癖 | 当前画像，LLM 更新 |
| **当前状态** | `relationship_arc` | 关系阶段 + 分数 | 当前关系值，LLM 评估 |
| **当前状态** | `shared_context` | 共同语言、梗、边界规则 | 规则/默契，不参入蒸馏 |
| **时序信号** | `emotional_trace` | 情绪轨迹、氛围感知 | 时间轴，追加写 |
| **实体索引** | `entity_relations` | 用户提到的人物/事物及其属性 | 去重合并，mention_count 累积 |

> **"当前状态"三张表不可揉入蒸馏表！** 它们代表的是"当前对用户的认知/关系/默契"，是规则而非流水账，必须独立存在。

### cognitive_distill 表结构

```sql
CREATE TABLE IF NOT EXISTS cognitive_distill (
    id INTEGER PRIMARY KEY,
    user_id TEXT,
    session_id TEXT,
    content TEXT NOT NULL,        -- 原文或摘要
    summary TEXT,                  -- LLM 提炼摘要（可选）
    mood TEXT DEFAULT '',          -- 情绪标签
    mood_intensity REAL DEFAULT 0.0,
    importance REAL DEFAULT 0.3,  -- 重要性 0.0~1.0
    entities TEXT DEFAULT '[]',    -- JSON 实体列表
    embedding BLOB,                -- 向量（用于语义检索）
    turn_num INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

特点：
- 替代旧的三张表：`long_term_memory`（关键词模糊匹配）、`event_memory`（事件）、`memory_embeddings`（向量单独存放）
- 每条记录自带完整元数据：时间、情绪、重要性、实体标签、向量
- 来源多样：`_post_update` 写入对话摘要、蒸馏触发写入浓缩摘要、LLM ANALYSIS 写入事件/dangling_threads
- 向量列 `embedding` 为独立存储，`idx_cd_embed` 索引过滤 `NOT NULL` 行加速检索

### 蒸馏触发机制

双条件触发，先到先执行（`_DISTILL_MAX_TURNS` / `_DISTILL_MAX_MINUTES`，默认 30 轮 / 60 分钟）：

```
每轮对话结束 → _post_update → _maybe_distill
    ├─ turn_count - last_distill_turn >= 30?  → 触发
    └─ elapsed >= 60 分钟?                     → 触发
         ↓
    合并上次蒸馏以来未总结的条目 → 写入 cognitive_distill (importance=0.7)
    → 更新 last_distill_turn / last_distill_at
```

不足 3 条时不蒸馏，避免碎片。

### 迁移路径

已有数据库会自动执行迁移（`_migrate_to_cognitive_distill`）：
1. 检查是否存在旧表 `long_term_memory`
2. 若无旧表或已迁移过 → 跳过
3. old `long_term_memory` → INSERT 到 `cognitive_distill`
4. 按 content 匹配 `memory_embeddings.embedding` → UPDATE cognitive_distill.embedding
5. old `event_memory` → 追加到 `cognitive_distill`
6. DROP 旧三表

## 技术栈

- **SQLite** — 主存储，零额外软件。
- **Embedding API** — 语义检索。调 API（OpenAI text-embedding-3-small），无本地模型依赖。
- **不用 Neo4j** — 当前数据量级下不需要图数据库的推理能力，不值得运维成本。

## 数据流向

```
用户输入
    │
    ├─→ 三通道融合召回（retrieve_memories）
    │    ├─ 通道 1：importance × 时间衰减 → 近期大事 1-2 条
    │    ├─ 通道 2：时间过滤 → 图扩散扩词 → 向量余弦 → 情绪共振 → 2-3 条
    │    └─ 通道 3：图谱高权重边（relation 语义优先）→ 1-2 条自然语言陈述
    │
    ├─→ 当前状态加载
    │    (identity_memory / relationship_arc / shared_context / emotional_trace / entity_relations)
    │    → 注入 prompt 作为 "当前对这个用户的理解"
    │
    ├─→ 融合记忆 + 当前状态 → LLM 回答
    │
    ├─→ save_turn (chat_history 追加写)
    │
    └─→ _post_update
         ├─→ cognitive_distill 写入对话摘要
         ├─→ _build_graph_from_memory (memory_graph_nodes/edges)
         ├─→ _maybe_distill (双触发检查)
         └─→ ANALYSIS (LLM / 规则)
              ├─→ 情绪 → emotional_trace
              ├─→ 话题转移 → focus_topic（原 topic_shift + key_points）
              ├─→ 关系 → relationship_arc
              ├─→ 特质/口癖 → identity_memory
              ├─→ 梗/边界 → shared_context
              ├─→ 实体 → 图边（memory_graph_edges.relation 语义字段）
              └─→ 事件/dangling → cognitive_distill
```

## 三通道融合召回

`retrieve_memories` 是三通道异步融合引擎，替代旧版关键词 LIKE 检索：

| 通道 | 算法 | 得分公式 | 名额 |
|------|------|---------|------|
| **时间重要性** | `importance × exp(-λ·days)` 衰减排序 | 时间半衰衰减 | 保底 1，配额 ch_min[0]+1 |
| **多维共振** | 时间词硬过滤 → 图扩散扩关键词 → 原查询+扩散词拼装 embed → 余弦相似度 → 情绪同频加成 | `cos_sim×0.6 + imp_decay×0.4`，情绪匹配再 ×1.15 | 保底 1，配额 ch_min[1]+2 |
| **图谱骨架** | `memory_graph_edges` 高权重边，relation 优先级排序 | 有 relation 的边额外 +2 分 | 保底 1，配额 ch_min[2]+1 |

融合策略：每通道保底 → 前 25 字去重 → 按分排序 → 截断到 `total_cap`。

Configurable 参数（`config.json → analysis.retrieval`）：
- `total_cap`：总上限，默认 5
- `channel_min`：每通道保底，默认 `[1, 1, 1]`
- `time_decay_halflife`：重要性半衰期（天），默认 30
- `emotional_resonance_bonus`：情绪同频加成系数，默认 0.15

### 图谱 relation 字段

`memory_graph_edges` 新增 `relation TEXT` 字段，来源：
- **LLM ANALYSIS**：`entities` 中 `name` → from_node、`fact` → to_node、`relation` → edge.relation
- **共现构建**：`_build_graph_from_memory` 自动建立的关键词共现边 `relation` 为空

输出格式：有 relation 时 `「老板」--[当众骂]--> 「用户」`，无 relation 时 `话题「X」与「Y」常被提及`。

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
自由文本（traits_updates, speech_quirks, boundary_hits）→ 写入 identity_memory。
`dangling_threads` → 写入 cognitive_distill。
`shared_jokes` → 写入 shared_context。
`entities` → 写入 `memory_graph_edges` 图边。

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
│              │     │  认知蒸馏 + 当前状态加载│
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
| 存储架构 | 功能分表（event/embedding/long-term 各自独立），数据不交叉 | 统一蒸馏表 `cognitive_distill`，每条记录带齐元数据 |
| API 调用 | 按功能分多次调用（embedding + analysis 分开） | 单次 LLM ANALYSIS 产出完整 JSON，零碎 API 调用消除 |
| 遗忘机制 | 无 | `importance` 分级 + 时间加权衰减 |
| 蒸馏 | 无 | 双条件触发（轮数/时间），先到先执行 |

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
    "enable_auto_save": true
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
    // source=astrbot → 读 AstrBot cmd_config.json
    // source=custom   → 用 custom_api_key/base_url/model
    "retrieval": {
      "enabled": true,
      "source": "astrbot",            // "astrbot" | "custom"
      "astrbot_source_id": "",        // AstrBot provider source ID，留空自动匹配
      "custom_api_key": "",           // source=custom 时使用
      "custom_base_url": "https://api.openai.com/v1",
      "custom_model": "text-embedding-3-small"
    },
    // 事后分析：对话结束后 LLM 产出 JSON 更新五张表
    "post_analysis": {
      "mode": "rule",                 // "rule" | "llm"
      "sampling": 0.1,
      "source": "astrbot",            // "astrbot" | "custom"
      "astrbot_source_id": "",
      "custom_api_key": "",
      "custom_base_url": "https://api.openai.com/v1",
      "custom_model": "gpt-4o-mini"
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

# RCMS 参考手册

> 架构速览 + 全模块详解 + 检修 + 升级。

---

## 一、流水线全流程

```
用户输入
  │
  ├─ 1. 三通道融合召回（retrieve_memories）
  │
  ├─ 2. _load_long_term_context
  │      ├─ identity_memory → 用户画像
  │      ├─ shared_context  → 梗/共同语境
  │      ├─ cognitive_distill → 最近事件
  │      └─ memory_graph_edges → 实体关系
  │
  ├─ 3. narrative_context / prompt_compressor → 拼入 system_prompt
  │
  ├─ 4. LLM 回答
  │
  ├─ 5. save_turn → chat_history（规则 importance 计算）
  │
  ├─ 6. post_update_rules（纯规则）
  │      ├─ 写规则摘要（importance=0.3）到 cognitive_distill
  │      ├─ 关键词共现建图
  │      └─ 悬案自动过期检查
  │
  └─ 7. check_distill_needed（每 30 轮 / 60 分钟）
         └─ _run_distill_analysis
                ├─ LLM 一次调用 → 叙事摘要 + 9 维 JSON
                ├─ _apply_distill
                │     ├─ 写入精炼摘要（importance=0.8）
                │     ├─ 归档悬案
                │     ├─ 清理低重要性碎片（< 0.5，跨窗口）
                │     └─ 图衰减 + 孤立节点清理
                └─ _apply_analysis
                      ├─ 用户状态 → session_state.stance
                      ├─ 用户画像（traits/结构化字段/边界）
                      ├─ 实体 → 图谱语义边
                      ├─ 事件记忆 + 悬案
                      └─ key_facts → cognitive_distill
```

---

## 二、存储表详解

### 全量流水：chat_history

每轮对话的原始记录，**不参与召回**，只在蒸馏时做快照。

| 列 | 来源 | 说明 |
|----|------|------|
| content | `save_turn` | 用户/助手原始消息 |
| turn_num | `save_turn` | 递增轮号 |
| importance | `save_turn` 规则计算 | 0.3 基础 + 情绪词 +0.1 + 长文本 +0.1，上限 0.8 |

### 蒸馏记忆：cognitive_distill

三通道中通道 1 和通道 2 的数据源。混合存储两种条目：

| 条目类型 | importance | 写入时机 | 用途 |
|----------|-----------|----------|------|
| 规则摘要 | 0.3 | `post_update_rules` 每轮 | 短期对话锚点 |
| LLM 精炼摘要 | 0.8 | `_apply_distill` 蒸馏触发 | 长期记忆核心 |
| 事件记忆 | ≥ 0.5 | `_apply_analysis` | 重要事件存档 |
| 悬案归档 | 0.5~0.7 | 过期归档 / 蒸馏归档 | 未完成话题存档 |

每行带完整元数据：`created_at`、`mood`、`mood_intensity`、`importance`、`embedding`。

### 用户画像：identity_memory

每人一条。LLM 蒸馏时 `_apply_analysis` 全量更新。详见"用户画像"章节。

### 会话状态：session_state

每 session 一条，跟踪：
- `turn_count` — 对话轮数
- `focus_topic` — 当前焦点话题（LLM `topic_shift + key_points` 更新）
- `stance` — 用户状态（`user_state`，open/reflective/guarded/...）
- `dangling_threads` — 悬案列表 JSON `{"threads": [...], "turn": N}`
- `last_distill_turn` / `last_distill_at` — 蒸馏水位线

遗留列（表定义中保留，代码不再写入）：`mood`、`stance_turns`、`engagement_level`、`momentum_depth`、`momentum_energy`、`embedding_updated`。

### 图谱：memory_graph_nodes + memory_graph_edges

建图双路径：

| 路径 | 数据来源 | 边类型 |
|------|----------|--------|
| 规则共现 | `_build_graph_from_memory` 关键词提取 | 纯共现边（relation = ''） |
| LLM 语义 | `_apply_analysis` entities 字段 | 语义边（relation = '朋友'/'同事'/...） |

通道 3 展示优先语义边，无 relation 的共现边降权。

### 共同语境：shared_context

| 列 | 说明 |
|----|------|
| context_body | `[梗] trigger → context` 格式 |
| omission_count | 提及次数，用于排序展示 |

写入自 `_apply_analysis` 的 `shared_jokes` 字段。展示在 `narrative_context` 共同语境区块。

---

## 三、用户画像（identity_memory）

### 结构

```
identity_memory（每人一条）
├── traits          JSON: [{"t": "理性派", "s": 5, "c": 3}, ...]
├── preferences     JSON: {"likes": ["咖啡","猫"], "dislikes": ["说教"]}
├── communication_style   字符串
├── self_identity   JSON: ["理性派", "INFJ"]
├── core_identity   JSON: {"职业": "程序员", "角色": "全栈"}
└── boundaries      JSON: ["别说教", "别拿抑郁症开玩笑"]
```

### traits 强度衰减算法

每次 `_apply_analysis` 时执行。**count 永久记录历史确认次数，不受衰减影响；只有 strength 变化。**

```
LLM 产出的 traits_updates（确认列表）:
  - 新特质 → 加入 map，strength=5，count=1
  - 已有特质 → strength 重置为 5，count += 1（count 只增不减）

LLM 未确认的已有特质:
  - strength -= 1                     ← 每次蒸馏衰减 1
  - 下限 floor = min(count, 3)         ← count=1 下限 1, count≥3 下限 3
  - strength = max(strength, floor)    ← 不低于下限
  - strength ≤ 0 → 删除                ← count=1 时 5 轮不提即删

speech_quirks → 以 "[口癖] 内容" 格式加入同一 traits 池，同等强度管理
```

示例：一条 count=3 的特质被 LLM 连续 10 轮不提，strength 从 5 衰减到 3 后不再下降，永远不会自动消失。

### 结构化字段覆盖写策略

`preferences`、`communication_style`、`self_identity`、`core_identity` **全量覆盖**而非增量合并。

理由：LLM 已参考全部历史对话，产出即综合快照。增量合并会把过时的碎片和当前快照混在一起，反而产生矛盾。

### boundaries 覆盖写

**全量覆盖**而非增量合并，与其它结构化字段一致。LLM 已参考现有雷区（通过 lt_hint），产出即完整快照。

理由：避免雷区单向累积导致矛盾或膨胀。每次蒸馏 LLM 重新评估当前仍需保留的边界，而非只增不减。

---

## 四、共同语境（shared_context + 图谱实体）

### 梗/上下文（shared_context）

写入自 LLM `shared_jokes`：
```json
{"trigger": "喵", "context": "哈基米梗"}
→ "[梗] 喵 → 哈基米梗"
```

展示在 `narrative_context` 共同语境区块：
```
共同语境:
  · 梗: 喵 → 哈基米梗
```

### 实体关系（图谱边）

写入自 LLM `entities`：
```json
{"name": "小王", "relation": "朋友", "fact": "也喜欢摄影"}
→ 图节点「小王」--[朋友]-->「也喜欢摄影」
```

通道 3 召回时输出：
```
[图谱] 「小王」--[朋友]--> 「也喜欢摄影」
```

`narrative_context` 展示：
```
他提过的人/事: 小王 (朋友·也喜欢摄影)
```

---

## 五、三通道融合召回（retrieve_memories）

### 通道 1：时间 × importance

数据源：`cognitive_distill`（WHERE importance > 0.1，最新 50 条）

```
score = t × (0.5 + importance)
t = 2^(-days / half_life)
```

半衰期默认 30 天。importance 在所有时间尺度有恒定比例影响。
高重要性旧条目可压过低重要性新条目：`今天 imp=0.3 = 0.80 < 一周前 imp=0.8 = 1.10`

### 通道 2：多维共振

数据源：`cognitive_distill`

```
① 时间词硬过滤（今天/昨天/最近/上周…）
② 关键词提取 + 图扩散扩词（BFS depth=2, 每层激活衰减 ×0.5, 共现边额外 ×0.1）
③ 向量余弦检索（原词 + 扩散词拼装 query）
④ 关键词 SQL 候选 + 时间过滤
⑤ 评分:

   有向量命中: score = cos_sim × 0.6 + imp_decay × 0.4
   无向量:     score = imp_decay × 0.5
   无向量无关键词: importance ≥ 0.5 兜底

   情绪共振: 当前 mood == 条目 mood → score × (1 + resonance_bonus)

   扩散词分类（激活值）:
     surfaced (≥ 0.6): 强相关词，正常参与检索
     silent (0.25~0.6): 弱相关词，仅做候选补充
```

当前情绪读取自最近一条 `cognitive_distill.mood`（蒸馏 LLM 写入），不走单独情绪表。

### 通道 3：图谱骨架

数据源：`memory_graph_edges`

```
输入关键词 → 图节点 → BFS 取关联边
排序: relation != '' 优先（额外 +2），其次 weight DESC
输出: 「A」--[关系]-->「B」 或 「A」与「B」常被一起提及（权重 x.x）
```

### 融合器 fusion

```
Phase 1: 每通道保底 ch_min[i] 条
Phase 2: 剩余名额按分排序填充
全程: 前 25 字符去重
截断: total_cap 条
返回: [(content, tag), ...]  tag ∈ {recent, resonance, skeleton}
```

---

## 六、蒸馏流程详解

### 触发条件

`check_distill_needed` 在每轮 `post_update_rules` 之后检查：

```
触发 = (turn_count - last_distill_turn >= max_turns)
       OR (elapsed_minutes >= max_minutes)
       快照行数 ≥ 6（至少 3 轮对话）
```

### LLM 调用

`_run_distill_analysis` → `_build_distill_prompt` 两阶段 prompt：

```
第一阶段：理解对话脉络（事件顺序、情绪基调、人物关系）
第二阶段：精确提取 → JSON

产出 JSON:
{
  "summary": "连贯的叙事摘要（非要点罗列）",
  "analysis": {
    "key_facts": ["完整可独立理解的事实"],
    "mood": "温暖|低落|焦虑|...",
    "mood_intensity": 0.0~1.0,
    "topic_shift": true/false,
    "key_points": ["事件脉络摘要"],
    "user_state": "open|reflective|guarded|...",
    "traits_updates": ["新发现的特质"],
    "speech_quirks": ["说话特点"],
    "preferences": {"likes":[], "dislikes":[]},
    "communication_style": "...",
    "self_identity": [...],
    "core_identity": {...},
    "boundaries": ["雷区"],
    "dangling_threads": ["未完成话题"],
    "importance": 0.0~1.0,
    "entities": [{"name":"","relation":"","fact":""}]
  }
}
```

### 写入

**_apply_distill：**
1. 写精炼摘要到 `cognitive_distill`（importance=0.8）
2. 更新 `session_state.last_distill_turn/at`
3. 归档当前悬案到 `cognitive_distill`
4. 跨窗口低重要性碎片清理：删除该用户所有 importance < 0.5 的条目（保留最新一条），不限 session
5. 图维护：共现边 weight × 0.8，< 0.3 删除；孤立节点删除

**_apply_analysis：**
1. 焦点话题更新（topic_shift + key_points → focus_topic）
2. 提取 mood/intensity（供下游使用）
3. 用户状态写入（user_state → session_state.stance）
4. traits + quirks 强度衰减更新
5. 结构化字段覆盖写（preferences/style/identity/core/boundaries）
6. shared_jokes 写入 shared_context
7. dangling_threads 写入 cognitive_distill + session_state
8. entities 写入图谱语义边
9. 高重要性事件（importance ≥ 0.5）写入 cognitive_distill
10. key_facts 写入 cognitive_distill（importance 保底 0.5，不被碎片清理误删）

---

## 七、narrative_context 输出结构

```
[RCMS 关系上下文]
聊了 {turn_count} 轮

他是什么样的:
  · 理性派
  · 程序员
  · 口癖: 怎么说呢、其实吧

结构化画像:
  · 喜好: 咖啡、独立游戏
  · 沟通风格: 喜欢举例子
  · 身份: 程序员·后端

共同语境:
  · 梗: 喵 → 哈基米梗
  · 他提过的人/事: 小王 (朋友·也喜欢摄影)
  · 最近总聊: 工作压力

相关记忆:
  · {记忆条目}
  · {记忆条目}

未完成: ↘ 面试结果

→ 以上是你通过长期对话积累的对他的了解
```

---

## 八、悬案生命周期

```
话题提出（LLM dangling_threads）
  → 写入 session_state: {"threads": ["面试结果"], "turn": N}
  → 5 轮内未续：展示前缀 "↘"
  → 10 轮内未续：不展示
  → 15 轮内未续（dangling_expire_turns）：自动归档到 cognitive_distill [悬案归档·过期]
  → 蒸馏触发：同时归档 [悬案归档·蒸馏]
```

---

## 九、全文对照：distill prompt → 写入目标

| JSON 字段 | 写入目标 |
|-----------|----------|
| summary | `cognitive_distill.content` |
| mood + mood_intensity | `cognitive_distill.mood/mood_intensity` |
| user_state | `session_state.stance` |
| topic_shift + key_points | `session_state.focus_topic` |
| traits_updates + speech_quirks | `identity_memory.traits`（强度更新） |
| preferences / communication_style / self_identity / core_identity | `identity_memory`（覆盖写） |
| shared_jokes | `shared_context`（追加/计数） |
| boundaries | `identity_memory.boundaries`（覆盖写） |
| dangling_threads | `cognitive_distill` + `session_state` |
| entities | 图谱边 `memory_graph_edges.relation` |
| key_facts | `cognitive_distill`（importance 保底 0.5，不被碎片清理） |
| importance ≥ 0.5 | `cognitive_distill`（事件存档） |

---

## 十、检修检查项

### 启动检查

| 日志关键字 | 说明 |
|-----------|------|
| `RCMS init: db=...` | 核心初始化成功 |
| `RCMS: 插件已加载` | AstrBot 插件加载 |
| `RCMS: 创建人格记忆库` | 数据库文件创建 |
| `AstrBot provider loaded` | LLM/Embedding 回调就绪 |

### 运行检查

| 日志关键字 | 说明 |
|-----------|------|
| `RCMS: retrieve_memories` | 三通道召回完成 |
| `RCMS: [{persona}] done` | 当轮全流程结束 |
| `DISTILL: start` | 蒸馏触发 |
| `DISTILL: ok summary=...` | 蒸馏 LLM 调用成功 |
| `ANALYSIS: write user=...` | 分析结果写入成功 |
| `RCMS: 图维护 user=...` | 图衰减 + 孤立节点清理 |
| `Embedding: ok dim=...` | 向量生成成功（首次需等待） |

### 数据库检查

```bash
# 蒸馏记忆概览
sqlite3 data/rcms_memory_*.db "
  SELECT importance, count(*) FROM cognitive_distill GROUP BY importance;
"

# 蒸馏水位线
sqlite3 data/rcms_memory_*.db "
  SELECT turn_count, last_distill_turn, last_distill_at FROM session_state;
"

# 未向量化记录（需 embedding 才会生成）
sqlite3 data/rcms_memory_*.db "
  SELECT count(*) FROM cognitive_distill WHERE embedding IS NULL;
"

# 图谱规模（含孤立节点和死边）
sqlite3 data/rcms_memory_*.db "
  SELECT 'nodes', count(*) FROM memory_graph_nodes
  UNION ALL SELECT 'edges', count(*) FROM memory_graph_edges
  UNION ALL SELECT 'semantic_edges', count(*) FROM memory_graph_edges WHERE relation != ''
  UNION ALL SELECT 'orphan_nodes', count(*) FROM memory_graph_nodes n WHERE n.node_id NOT IN (SELECT from_node_id FROM memory_graph_edges UNION SELECT to_node_id FROM memory_graph_edges);
"

# 用户画像
sqlite3 data/rcms_memory_*.db "
  SELECT user_id, length(traits), preferences, communication_style,
         length(self_identity), length(boundaries), length(core_identity)
  FROM identity_memory;
"

# 共同语境
sqlite3 data/rcms_memory_*.db "
  SELECT count(*), group_concat(substr(context_body, 1, 30), ' || ') FROM shared_context;
"
```

### 常见问题

| 症状 | 排查 |
|------|------|
| 无记忆召回 | `cognitive_distill` 是否有数据；蒸馏是否触发过 |
| 孤立节点过多 | 检查蒸馏是否正常触发；`_maintain_graph` 随蒸馏执行 |
| 蒸馏不触发 | `max_turns`/`max_minutes` 配置；快照 ≥ 6 行 |
| Embedding 为空 | API key/url 配置；`embedding_enabled` |
| 情绪共振无效 | `cognitive_distill.mood` 是否有值 |
| 图谱边无 relation | LLM 蒸馏 `entities` 是否填了 relation |
| 用户画像无数据 | 蒸馏是否成功执行过 |
| 梗未展示 | `shared_context` 是否有数据 |

---

## 十一、config.json 完整参考

```jsonc
{
  "analysis": {
    "retrieval": {
      "embedding_enabled": true,        // 向量检索开关
      "source": "astrbot",              // astrbot | custom
      "astrbot_source_id": "",          // AstrBot 提供商 ID，留空自动
      "custom_api_key": "",
      "custom_base_url": "https://api.openai.com/v1",
      "custom_model": "text-embedding-3-small",
      "total_cap": 5,                   // 三通道总召回上限
      "channel_min": [1, 1, 1],         // 每通道保底
      "time_decay_halflife": 30,        // 半衰期（天）
      "emotional_resonance_bonus": 0.15 // 情绪共振倍率加成
    },
    "post_analysis": {
      "source": "astrbot",
      "astrbot_source_id": "",          // 留空自动匹配
      "custom_api_key": "",
      "custom_base_url": "https://api.openai.com/v1",
      "custom_model": "gpt-4o-mini",     // 蒸馏用模型
      "max_turns": 30,                   // 蒸馏轮数间隔
      "max_minutes": 60,                 // 蒸馏分钟间隔
      "dangling_expire_turns": 15        // 悬案过期轮数
    }
  }
}
```

旧 key 名（`enabled`/`custom_token`/`custom_url`）仍兼容，建议迁移到新名。

---

## 十二、升级注意事项

### 表结构变更

`_init_db()` 使用 `CREATE TABLE IF NOT EXISTS` + 逐列 `ALTER TABLE ADD COLUMN` 迁移。
旧表（`long_term_memory`/`event_memory`/`memory_embeddings`）自动迁移到 `cognitive_distill`。

### 配置变更

| 版本 | 变更 |
|------|------|
| 当前 | `retrieval.embedding_enabled`（旧名 `enabled` 兼容） |
| 当前 | `retrieval+post_analysis.custom_api_key/base_url`（旧 `custom_token/url` 兼容） |
| 当前 | `post_analysis.max_turns/max_minutes/dangling_expire_turns` 新增可配置 |
| 当前 | `post_analysis.mode/sampling` 已删除 |

### 已移除完整清单

- `emotional_trace` 表 + 全部代码
- `relationship_arc` 表 + 阶段晋升 + 里程碑
- `entity_relations` 表（僵尸表，已由图谱边替代）
- `voice_hint` 字段（被 `[口癖]` traits 池替代）
- `session_state.residue_warmth` / `residue_tension` 字段
- `chat_history.mood` 回灌（无消费方）
- `_load_residue` / `_decay_residue` / `_write_residue` / `_apply_residue` 方法
- `_run_analysis` / `_build_analysis_prompt`（旧每轮 LLM 分析）
- `_maybe_distill` SQL 拼接方式
- 规则分析层（stance/momentum/engagement）
- 常量 `_ARC_STAGES` / `_RESIDUE_DECAY` / `_last_silent_recall`
- 配置项 `mode` / `sampling`（旧每轮分析开关）

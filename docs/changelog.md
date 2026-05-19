# 变更记录

> 当前版本: v0.1.0-pre — 未发行阶段，所有版本号 ≤ v1.0
> 贡献者: [仓库主](https://github.com/ATRI)

## 2026-05-19

### feat: 3 种人格风格切换（default / cute / professional）
- `analysis_config.post_analysis.personality_type` 控制蒸馏 prompt 风格
- cute：第一人称，语气词「呀~呢~哦」，轻松温暖讲故事
- professional：第三人称，客观结构化，精炼分析
- default：原版第一人称日记叙事
- 风格分支影响：system_preamble、summary_instruction、first_stage、extra_rules

### feat: 4 项蒸馏 + 人格改进
#### snapshot 格式支持 [昵称] 前缀
- `check_distill_needed` 返回 5 元组（含 senders 列表）
- 快照格式从「用户: 内容」改为「[昵称] 内容」，昵称来自 chat_history.sender_name
- Bot 消息统一使用 `persona_name` 作为昵称

#### 私聊/群聊双模板 + Bot 第一人称叙事
- `_build_distill_prompt` 根据 is_group 输出不同版本：
  - 私聊版：关注互动节奏、情绪变化，无 participants 字段
  - 群聊版：参与者识别、人物关系、JSON 含 participants 字段
- summary 从第二人称「你」改为第一人称「我」日记式叙事

#### 人格风格注入蒸馏 prompt
- `_run_distill_analysis` 自动从 long_term 构建 personality_style（communication_style + 前 3 个 traits）
- 注入到 distill prompt 的「角色风格」区

#### user_id ↔ nickname 深度绑定
- 插件 `on_llm_request` 通过 `event.get_sender_name()` 提取显示名
- `on_llm_response` 将 sender_name 传入 `save_turn`
- chat_history 存储 sender_name → 蒸馏快照自动使用正确昵称

### fix: 图谱节点 label 清洗 — 去除前导/末尾非文字字符
- 避免 LLM 列表格式带出的 `-`、`·` 等符号导致同一实体分裂成两个节点

### feat: chat_history 添加 user_id 列 — 标记每条消息的发言者
- 新增 `ALTER TABLE` 迁移，`save_turn` 写入 user_id
- 群聊场景下可区分同一 session 内各消息的发送者

### fix: 蒸馏 summary 改为第二人称 + fuzz_time 自然时段 + 融合排序防垄断
- distill prompt 要求以「你」而非「用户/助手」叙述
- `_fuzz_time` 改为自然时段：刚刚/上午的时候/下午的时候/晚上的时候/昨天/前天/前几天/上周
- 融合 Phase 2 加入每通道上限 ceil(total_cap/3)，防止单通道垄断剩余名额

### refactor: 蒸馏 prompt 优化 + traits 衰减修复 + key_facts 写入优化
- traits_updates 加负例约束（禁止泛化特质）和15字上限
- mood 从枚举改为自由文本（如「轻松好奇」）
- speech_quirks/traits_updates 与 lt_hint 去重，只输出新发现
- summary 优先捕捉新变化，不重复已知事实
- key_facts 上限改为5条，按 importance 降序
- traits 衰减 floor 从 min(c,3) 改为 min(c//2,2)，新增30条容量上限
- key_facts 写入分 permanent（最多3条，保底0.5）和 transient（最多5条，无保底）

## 2026-05-18

### 23:11 feat: 图路径序列化 + 边衰减 + 矛盾检测
- `95ce0aa` — 图检索增加路径序列化输出，边权重支持时间衰减，新增记忆矛盾检测机制

### 23:02 feat: 图检索双向模糊匹配 + 反向边去"相关于" + distill prompt 实体关系优化
- `0928d32` — 图检索支持双向模糊匹配，消除反向边中的"相关于"冗余，优化 distill prompt 的实体关系表达

### 17:40 feat: 融合排序加通道权重 + 展示层去掉每通道[:2]截断
- `4b38e51` — 多通道融合排序引入权重系数，展示层移除通道级别的数量截断

### 16:10 feat: 实体多关系格式 + mood 写入 + post_update_rules 纯管理 + 图单来源
- `f9b16ac` — 实体支持多关系格式，mood 写入链路，post_update_rules 改为纯管理逻辑，图谱数据限定单来源

### 14:53 fix: 插件数据库存储路径改为插件目录
- `d73b063` — 修复 AstrBot 插件模式下数据库路径问题，改为存储在插件目录下

### 14:51 feat: 三通道记忆展示 + jieba 中文分词 + 停用词过滤 + embedding 全链路写入
- `b55dca5` — 新增三通道记忆展示界面，集成 jieba 分词与停用词过滤，embedding 全链路写入完成

### 13:40 fix: 图谱自环边防护
- `1a1de3d` — 修复图谱中可能出现自环边的问题，增加关键词去重和 from!=to 校验

### 13:34 feat: 记忆时效标签 + Session 预热
- `92245ee` — 记忆增加时效性标签，新增 Session 预热机制

### 13:29 feat: 5 项检索/存储优化 + 文档同步
- `0dfce58` — 多项检索与存储性能优化，同步更新相关文档

### 13:13 fix: save_turn 不再覆盖 stance，避免覆盖蒸馏 user_state
- `e973edf` — 修复 save_turn 覆盖 stance 的问题，防止蒸馏 user_state 被意外覆盖

### 13:09 docs: 全面同步参考手册与代码逻辑
- `524557f` — 参考手册与代码逻辑全面对齐同步

### 13:05 fix: 三个遗忘机制问题 + key_facts importance 保底
- `79a0c0e` — 修复三个遗忘机制的边缘情况，key_facts importance 增加保底值

### 12:59 refactor: 清理死代码 + 架构整理
- `7a968d8` — 清理废弃代码，整理项目架构

### 10:48 chore: 移除无用设置 max_memories_per_prompt
- `a80a111` — 移除从未生效的 max_memories_per_prompt 设置项

### 10:43 fix: Windows GBK 编码兼容
- `07fb6cf` — 修复 Windows 下 GBK 编码兼容问题，特殊字符替换 + stdout 重配置

### 10:43 refactor: 合并 api 段入 analysis + 性能优化异步化
- `a7d5419` — 将 api 段合并入 analysis，异步化性能优化

## 2026-05-17

### 23:24 fix: 文档蒸馏阈值/实体存储描述 + 安装脚本路径 bug
- `bc396d6` — 蒸馏流程图阈值 50/120 → 30/60（匹配代码），实体写入位置修正为图谱边，安装脚本路径修复

### 23:07 feat: 关系里程碑 + 统一 entity_relations 到图谱边
- `662a9bb` — 新增关系里程碑功能，将 entity_relations 统一存储到图谱边

### 23:04 chore: 清理已废弃的脚本和文档
- `de49a61` — 清理废弃的脚本和文档文件

### 22:50 feat: identity_memory 结构化字段
- `2685ae2` — identity_memory 新增结构化字段：喜好、沟通风格、自我认同、雷区、核心身份

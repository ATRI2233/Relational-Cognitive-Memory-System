# 变更记录

> 未发行阶段，所有版本 ≤ v1.0。
> 贡献者: [仓库主](https://github.com/ATRI)

## v0.1.2 — 2026-05-20

### refactor: cognitive_distill 向量嵌入源切换为 summary 短标签
- Prompt JSON 模板：`summary` 拆分为 `content`（完整叙事）+ `summary`（10-20 字语义标签），新增规则 9 约束标签格式（含具体实体、禁止叙事连接词）
- `_run_distill_analysis`：分别提取 `content`/`summary`，旧格式自动回退兼容（summary 超 30 字则用 key_facts 拼接兜底）
- `_apply_distill` 签名改为 `(content, summary)`，DB 写入 `content` 列存完整文本、`summary` 列存短标签、embedding 用短标签
- 关键事实/未结话题/悬案归档三处去除 `[:N]` 机械截断：关键事实和话题 `summary = content`；悬案归档去掉前缀取话题名
- 清理 `_load_long_term_context` 中从未使用的 `recent_events` 死代码查询
- 向后兼容：key_facts 回退路径加 `isinstance(f, str)` 守卫，防 LLM 混入 dict 崩溃

## v0.1.1 — 2026-05-19

### fix: 替换 silent `except Exception: pass` 为异常日志记录
- 扫描 `error_issues_and_fixes.md` 列举的 9 项问题，逐文件修正
- 后台异步任务（`_delayed_embed`、`_async_post_update`）失败时记录异常上下文
- DB 迁移 ALTER TABLE 的 broad-except 收窄为 `sqlite3.OperationalError`
- config 加载失败、LLM fallback 失败增加日志输出
- JSON 解析/时间解析失败从 silent-pass 改为捕获具体异常

### refactor: 异常处理精细化
- `rcms_core/db.py`: 所有 ALTER TABLE 迁移改用 `except sqlite3.OperationalError`
- `rcms_core/memory.py`: `post_update_rules` / `_archive_dangling` 的 JSON 解析用 `(json.JSONDecodeError, ValueError)`
- `rcms_core/context.py`: 4 处 broad-except 改为具体异常捕获或 `logger.exception`
- `rcms_core/core.py`: `chat()` 二次 fallback 记录异常后返回降级回复
- `plugins/rcms-astrbot/main.py`: 3 处 silent-pass 改为 `logger.exception` / `logger.warning`
- `scripts/install_astrbot.py`: 2 处 silent-pass 改为 `print` 提示

## v0.1.0 — 2026-05-19

### feat: 3 种人格风格切换（default / cute / professional）
- `analysis_config.post_analysis.personality_type` 控制蒸馏 prompt 风格
- cute：第一人称，语气词「呀~呢~哦」，轻松温暖讲故事
- professional：第三人称，客观结构化，精炼分析
- default：原版第一人称日记叙事
- 配置项: `config.json → analysis.post_analysis.personality_type`
- → 详见 [六、蒸馏流程详解 · LLM 调用](rcms-reference.md#六蒸馏流程详解) / [十一、config.json 参考](rcms-reference.md#十一configjson-完整参考)

## v0.0.9 — 2026-05-19

### feat: 4 项蒸馏 + 人格改进
#### snapshot 格式支持 [昵称] 前缀
- `check_distill_needed` 返回 5 元组（含 senders 列表）
- 快照格式从「用户: 内容」改为「[昵称] 内容」，昵称来自 chat_history.sender_name
- Bot 消息统一使用 `persona_name` 作为昵称
- → 详见 [二、存储表详解 · chat_history](rcms-reference.md#二存储表详解) / [六、蒸馏流程详解 · 触发条件](rcms-reference.md#六蒸馏流程详解)

#### 私聊/群聊双模板 + Bot 第一人称叙事
- `_build_distill_prompt` 根据 is_group 输出不同版本：
  - 私聊版：关注互动节奏、情绪变化，无 participants 字段
  - 群聊版：参与者识别、人物关系、JSON 含 participants 字段
- summary 从第二人称「你」改为第一人称「我」日记式叙事
- → 详见 [六、蒸馏流程详解 · LLM 调用](rcms-reference.md#六蒸馏流程详解)

#### 人格风格注入蒸馏 prompt
- `_run_distill_analysis` 自动从 long_term 构建 personality_style（communication_style + 前 3 个 traits）
- 注入到 distill prompt 的「角色风格」区
- → 详见 [六、蒸馏流程详解 · LLM 调用](rcms-reference.md#六蒸馏流程详解)

#### user_id ↔ nickname 深度绑定
- 插件 `on_llm_request` 通过 `event.get_sender_name()` 提取显示名
- `on_llm_response` 将 sender_name 传入 `save_turn`
- chat_history 存储 sender_name → 蒸馏快照自动使用正确昵称
- → 详见 [二、存储表详解 · chat_history](rcms-reference.md#二存储表详解)

### fix: 图谱节点 label 清洗
- 去除 LLM 列表格式带出的 `-`、`·` 等符号导致同一实体分裂
- → 详见 [二、存储表详解 · 图谱](rcms-reference.md#二存储表详解)

### feat: chat_history 添加 user_id 列
- 新增 `ALTER TABLE` 迁移，`save_turn` 写入 user_id
- → 详见 [二、存储表详解 · chat_history](rcms-reference.md#二存储表详解)

### fix: 蒸馏 summary 改为第二人称 + fuzz_time 自然时段 + 融合排序防垄断
- distill prompt 要求以「你」而非「用户/助手」叙述
- `_fuzz_time` 改为自然时段：刚刚/上午的时候/下午的时候/晚上的时候/昨天/前天/前几天/上周
- 融合 Phase 2 加入每通道上限 ceil(total_cap/3)
- → 详见 [六、蒸馏流程详解 · LLM 调用](rcms-reference.md#六蒸馏流程详解) / [五、融合器 fusion](rcms-reference.md#五三通道融合召回)

### refactor: 蒸馏 prompt 优化 + traits 衰减修复 + key_facts 写入优化
- traits_updates 加负例约束和 15 字上限
- mood 从枚举改为自由文本
- speech_quirks/traits_updates 与 lt_hint 去重
- key_facts 上限 5 条，按 importance 降序
- traits 衰减 floor: min(c//2,2)，30 条容量上限
- key_facts 分 permanent（≤3，保底 0.5）和 transient（≤5，无保底）
- → 详见 [三、用户画像](rcms-reference.md#三用户画像identity_memory) / [九、全文对照](rcms-reference.md#九全文对照distill-prompt--写入目标)

## v0.0.8 — 2026-05-18

### feat: 图路径序列化 + 边衰减 + 矛盾检测
- `95ce0aa`

### feat: 图检索双向模糊匹配 + 反向边去"相关于" + distill prompt 实体关系优化
- `0928d32`

### feat: 融合排序加通道权重 + 展示层去掉每通道截断
- `4b38e51`

### feat: 实体多关系格式 + mood 写入 + post_update_rules 纯管理 + 图单来源
- `f9b16ac`

### fix: 插件数据库存储路径改为插件目录
- `d73b063`

### feat: 三通道记忆展示 + jieba 分词 + 停用词过滤 + embedding 全链路
- `b55dca5`

### fix: 图谱自环边防护
- `1a1de3d`

### feat: 记忆时效标签 + Session 预热
- `92245ee`

### feat: 5 项检索/存储优化 + 文档同步
- `0dfce58`

### fix: save_turn 不再覆盖 stance
- `e973edf`

### docs: 全面同步参考手册与代码
- `524557f`

### fix: 三个遗忘机制 + key_facts importance 保底
- `79a0c0e`

### refactor: 清理死代码 + 架构整理
- `7a968d8`

### chore: 移除无用设置 max_memories_per_prompt
- `a80a111`

### fix: Windows GBK 编码兼容
- `07fb6cf`

### refactor: 合并 api 段入 analysis + 异步化
- `a7d5419`

### fix: 文档蒸馏阈值/实体存储描述 + 安装脚本路径
- `bc396d6`

### feat: 关系里程碑 + 统一 entity_relations 到图谱边
- `662a9bb`

### chore: 清理废弃脚本和文档
- `de49a61`

### feat: identity_memory 结构化字段
- `2685ae2`

## v0.0.7 — 2026-05-17

### feat: AstrBot 解耦 + 回调接口 + install.sh
- `f5bf6b0`, `7975bd2`

### fix: post_analysis 模型名覆盖导致静默失败 + traits/dangling 退化
- `59958ea`

### fix: embedding 维度硬编码 1536 导致向量跳过
- `37f94ea`

### feat: trait 相似合并 + 确认次数保底衰减 + 展示压缩
- `1f60e1d`

### feat: 统一蒸馏表 + 双触发蒸馏 + 单次 API 分析 + 包结构重构
- `a643cb0`

### feat: topic_shift 写入 focus_topic，蒸馏阈值下调
- `d9af6e1`

### feat: 三通道融合召回架构
- `2cc242f`

### fix: 通道 2 补向量 + 通道 3 relation 语义格式 + 图边加 relation
- `f4c01d8`

### feat: 三通道参数可配置化
- `5c86c57`

### chore: 提取图操作 helper + 文档更新
- `fc3329d`

### fix: 图扩散语义边优先 + dangling_threads 生命周期管理
- `7f7d7b7`

## v0.0.6 — 2026-05-16

### chore: 大清理 — 砍掉规则分析层，合并为单文件
- `8fff833`

### feat: v2 架构 — Embedding 检索 + ANALYSIS LLM + 配置化
- `2ce1838`

### test: v2 集成测试
- `fc60494`

### chore: config.json 添加 analysis 配置段
- `de44e82`

### chore: 全链路日志 + 诊断脚本
- `03df379`

## v0.0.5 — 2026-05-15

### feat: 长期 5 层记忆表
- `61fb5c4`

### feat: Inhibition / Core Veto / Misrecall
- `7cbe95f`

### chore: SQLite 线程安全修复 + 输出日志配置化
- `3b53e9e`

### docs: 补全 CHANGELOG
- `ddaf38a`

### feat: narrative 注入 + 人格解析 + 三种注入方式
- `9ceb08a`

### chore: 补全文档和规划记录
- `71a2bdf`

## v0.0.4 — 2026-05-14

### feat: Silent Recall Residue + 时间衰减
- `9203996`

### feat: Working Memory 完整版
- `9e7047e`

### feat: 激活扩散图结构（BFS 替代关键词 LIKE）
- `792cef1`

## v0.0.3 — 2026-05-13

### Initial commit: RCMS MVP FirstStep
- `e7afa54`

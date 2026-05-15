# Changelog

## [0.6.0] - 2026-05-15

### Added

- **WM Phase 1 — Working Memory 暂存** `_wm_phase1()`
  - 记录本轮原始输入 + 提取话题候选（去琐碎词，取前 3 个内容词）
  - 输出 dict 供 Momentum/后续模块消费

- **Recall 激活扩散检索** `_recall()` / `_activation_diffusion()`
  - 基础关键词检索 + 深度 > 0.4 时从情绪词额外扩散
  - 500ms 超时 fallback → 降级 attentive + "没什么特别联想"
  - coasting+distant 时跳过检索

- **LLM 容灾 2 级重试** `_generate_with_fallback()`
  - Level 0: 原始 Prompt 直接生成
  - Level 1: 简化安全 Prompt 重试
  - Level 2: `SAFE_REPLIES` 硬编码回复（7 态各自独立）
  - `_build_safe_prompt()` 支持 normal/minimal 两种模板

- **Post-Update 阶段** `_post_update()`
  - 事件回写：engaged_candidate + depth>0.5 或 attentive + 长内容时写入长期记忆
  - last_active 刷新

### Changed

- **Pipeline 顺序修正**：`WM Phase1 → Momentum → Engagement Trigger → Stance → Recall → Prompt Compression → LLM → Post-Update`
  - `_update_momentum()` 接受 wm dict，使用 topic_candidate 进行焦点话题确认/切换
  - `prompt_compressor()` 接受可选 memories/recall_status 参数，支持外部预检索
  - `save_turn()` 精简为纯保存（记忆回写移至 _post_update）

- **Emergency bypass 显式化**：`stance_manager()` 顶部声明独立 boolean
  - 条件：三门全亮 + emotional_salience > 0.8 → 无视冷却直连 intimate

- **`_check_emergency_bypass()` 移除**：逻辑内联到 `stance_manager()`

- **AstrBot 插件适配器更新** `plugins/rcms-astrbot/`
  - 旧版使用已被新版 Pipeline 替代

  旧 (v0.3.0): `detect_stance`(binary) + `retrieve_memories` + 手动拼接文本
  新 (v0.6.0): `engagement_trigger` + `stance_manager`(7态) + `update_momentum`(二维) + `prompt_compressor`(三槽位)
  - `on_llm_response` 新增 `_post_update()` 调用，支持 Post-Update 记忆回写

- **`_post_update()` 记忆回写条件修正** — 移除 `depth > 0.5` 限制导致首轮情绪无法写入的问题
  - 现在 engaged_candidate 时无条件写入，attentive + 长内容时也写入
  - `depth_high_enough` 变量保留为未来"关系转折事件"专用条件

### Testing

- 所有 64 项存量测试通过，无回归

### Added

- **Momentum 2D 二维动量** `_update_momentum()`
  - `depth_axis [0,1]`：五信号检测（self_disclosure/vulnerability/abstraction/continuity/meta_relationship），加权 delta + 惯性更新
  - `energy_axis [-1,1]`：五信号检测（emotional_intensity/conflict/urgency/agitation/rapid_switching），加权 delta + 惯性更新
  - 日常减速：琐碎话题标记时 depth_delta × 0.2
  - 松弛修正：松弛标记时 energy_delta 减 0.3
  - 话题切换阻力：2-gram Jaccard < 0.3 时 depth × 0.5
  - 存储在 `session_state.momentum_depth` / `momentum_energy`

- **Prompt Compression 半结构化模板** `prompt_compressor()`
  - 三槽位：`[关系气氛]`(energy+depth+stance覆盖) + `[潜在联想]`(记忆→弱投影) + `[表达倾向]`(stance→句型)
  - 【相关记忆】块（最多 2 条）+ 【底线】固定文本
  - 总 Prompt ≤ 180 字

### Changed

- `chat()` pipeline 更新为：`engagement_trigger → stance_manager → update_momentum → prompt_compressor → LLM → save_turn`
- `session_state` 新增 `momentum_depth REAL`, `momentum_energy REAL`, `last_active TIMESTAMP`

### Testing

- 新增 29 项测试覆盖 Momentum 信号/更新/集成 + Prompt Compression 槽位/长度/集成

## [0.4.0] - 2026-05-14

### Added

- **Engagement Trigger 三门共振检测** `engagement_trigger()`
  - 三门：`emotional_salience`(规则打分) / `conversational_shift`(2-gram Jaccard) / `unresolved_threads`(线程计数)
  - 输出三级：`coasting`(0灯) / `attentive`(1灯) / `engaged_candidate`(2+灯)
  - 单门强光例外：`salience > 0.85` 时升级一级
  - 阈值：情绪 0.35 / 转向 0.50 / 未闭合话题 0.60

- **Stance Manager 7 态 + 冷却期** `stance_manager()`
  - 7 态：`reflective / guarded / open / playful / analytical / distant / intimate`
  - Hub-and-spoke 转移矩阵，`open` 为中心
  - 冷却期：min 3 turns，满足强制突破可提前切换
  - 紧急通道：`salience > 0.8 + shift > 0.6 + unresolved 亮灯` 跳过 hub 直连

- **open_threads 表 + 自动管理** `_manage_open_threads()` / `_close_resolved_threads()`

### Changed

- **emotional_salience 阈值重平衡**：情绪词密度为主因，长度/问句全面降权为放大器，否定前缀过滤
- **conversational_shift 算法**：从关键词 Jaccard 改为中文 2-gram + 字符级连续性折扣
- **unresolved_threads 算法**：从关键词密度改为线程计数制（2+ 线程才亮灯）
- 旧 binary stance（casual/engaged）被 7 态系统取代
- `chat()` pipeline 更新为：`engagement_trigger → stance_manager → build_prompt → LLM → save_turn`

### Testing

- 新增 `tests/test_engagement_stance.py` — 27 项测试覆盖全部三门 + stance 逻辑
- 更新 `tests/test_firststep.py` — 适配新 stance 系统

## [0.3.0] - 2026-05-14

### Added

- **AstrBot Star 插件适配器** `plugins/rcms-astrbot/`
  - 无需修改 AstrBot 任何代码，直接作为插件加载
  - 通过 `on_llm_request` 注入 RCMS 关系氛围 + 记忆到 system_prompt
  - 通过 `on_llm_response` 记录对话到 RCMS 长期记忆
  - 部署位置: `~/.astrbot/data/plugins/rcms/`

### Architecture

```
RCMS 项目根 (独立)
  ├── minimal_rcms.py     ← 核心，零外部依赖
  ├── backends/            ← LLM 后端抽象
  └── plugins/
      └── rcms-astrbot/   ← AstrBot 适配器（薄层桥接）
                              ↓ 复制/symlink
        AstrBot plugins/   ← 在此集成
```

## [0.2.0] - 2026-05-14

### Added

- **LLM 后端抽象层** `backends/` — 支持接入任意 Agent
  - `LLMBackend` 协议: `generate(prompt) -> str`
  - `MockBackend`: 测试用固定回复
  - `AstrBotBackend`: 读取 AstrBot 配置，调用 OpenAI 兼容 API（自动发现 `~/.astrbot/data/cmd_config.json`）
  - `OpenAIBackend`: 直接配置 API key / base_url / model

- **对话演示脚本** `run_demo.py`
  - `--backend astrbot` 自动接入 AstrBot 配置的 LLM
  - `--backend openai` 直接指定 API
  - `--interactive` 交互模式 / 默认自动 7 轮演示
  - 统计输出（轮数、stance、记忆条数）

### Changed

- `minimal_rcms.py` 改为异步接口: `chat()` 接受 `LLMBackend` 而非 `llm_generate_fn`
- `test_firststep.py` 迁移至 pytest-asyncio

### Verified

- AstrBot 后端连接 DeepSeek V4 Flash 成功
- 7 轮自动对话演示通过：casual/engaged 切换、记忆检索、长期记忆持久化全部正常

## [0.1.0] - 2026-05-14

### Added

- **FirstStep MVP 实现** — 最小化关系认知记忆系统跑通 4 步 Pipeline
  - `minimal_rcms.py`: 核心实现，3 张 SQLite 表（long_term_memory / session_state / chat_history）
  - `test_firststep.py`: 8 项测试覆盖全部流程
  - Pipeline: 判氛围 → 查记忆 → 写 Prompt → 存状态
  - 时间模糊化、关键词 LIKE 检索、半结构化 Prompt 模板

### 架构

- **技术文档** `技术文档.md`: RCMS 完整三层架构设计
- **MVP 设计** `FirstStep.md`: 最小可行版本设计文档，定义 Phase 1/Phase 2 路线图

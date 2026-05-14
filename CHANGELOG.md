# Changelog

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

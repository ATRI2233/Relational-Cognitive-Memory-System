# RCMS — Relational Cognitive Memory System v2

关系认知记忆系统。让 AI Agent 在长期对话中有人际关系感知。

## 项目结构

```
domain/                  # 领域层 — 纯业务，零外部依赖
  ports/                 # 6个Protocol接口契约
  entities/              # 实体 + 值对象（自验证）
  services/              # 纯算法服务（无DB）
  events/                # 6个领域事件（不可变）
application/             # 应用层 — 编排
  use_cases/             # 3个Use Case（Chat / Distill / Retrieve）
  handlers/              # 4个Event Handler
  event_bus.py           # 事件总线
  analysis_writer.py     # 9维分析写入
  prompt_builder.py      # Prompt模板构建
infrastructure/          # 基础设施层
  clock.py               # SystemClock / FrozenClock
  config/settings.py     # pydantic-settings，60+字段全收口
  persistence/           # 4个SQLite Repository
adapter/                 # 适配器层
  rcms_factory.py        # DI工厂
  astrbot_plugin.py      # 精简AstrBot插件（~80行）
config.json              # 全局配置
scripts/                 # 工具脚本
tests/                   # 34项集成测试
```

## Pipeline

```
输入 → RetrieveContextUseCase(三通道) → PromptBuilder → ChatUseCase
                                                             ↓
                                                    LLM → save_turn
                                                             ↓
                                          EventBus → Handlers → DistillUseCase
```

## 架构原则

- 契约先行：跨模块调用必须定义 Protocol
- 依赖注入：构造函数显式传入所有依赖
- 无状态业务：Use Case 不持有 DB Session
- 配置收口：零硬编码字面量，全部从 pydantic-settings 读取

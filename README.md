# Relational Cognitive Memory System (RCMS)

关系认知记忆系统 — 让 AI Agent 在长期对话中拥有人际关系感知。

RCMS 不是简单的对话历史存储。它在每次对话后对 LLM 回复做 **事后蒸馏分析**，抽取用户画像、情绪氛围、实体关系、关键事实等多维信息，并在下次对话前通过 **三通道融合召回** 把最相关的记忆注入到 LLM 的上下文中。

## 核心能力

- **三通道记忆召回** — 时间重要性 × 语义向量 × 图谱关系，加权融合
- **LLM 事后蒸馏** — 每次对话后异步分析，抽取 9 维结构化记忆（用户画像、情绪、实体关系、关键事实等）
- **情绪共振** — 识别当前对话的情绪氛围，召回忆中情绪匹配的条目
- **图扩散检索** — 关键词通过知识图谱 BFS 扩散，找到语义关联的间接记忆
- **多用户隔离** — 每人独立 SQLite 数据库，支持多租户
- **AstrBot 插件** — 无缝集成 AstrBot 框架，支持三种注入方式（system_prompt / prompt_prefix / faketool）

## 流水线概览

```
用户输入
  ├─ 三通道融合召回（时间 + 语义 + 图谱）
  ├─ 加载长期语境（画像 + 梗 + 实体关系）
  ├─ 拼入 system_prompt → LLM 回答
  ├─ 写入对话历史（save_turn）
  ├─ 事后规则更新（悬案过期检查）
  └─ 蒸馏分析（每 N 轮 / N 分钟触发）
       └─ LLM 一次调用 → 叙事摘要 + 9 维 JSON → 写库
```

## 快速开始

### 作为 AstrBot 插件

1. 将 `plugins/rcms-astrbot` 放到 AstrBot 的 `plugins/` 目录
2. 配置 `config.json`（可参考项目根目录的示例）
3. 启动 AstrBot，RCMS 自动加载

### 单文件集成

```python
from rcms_core import MinimalRCMS

rcms = MinimalRCMS(db_path="memory.db")
# 配置 LLM 和 Embedding 回调后即可使用
reply = await rcms.chat(user_id, session_id, user_input, backend)
```

## 配置

配置通过 `config.json` 的 `analysis.retrieval` 和 `analysis.post_analysis` 段控制，支持自定义 LLM/Embedding 端点、召回参数、蒸馏频率等。

详见 [docs/rcms-reference.md](docs/rcms-reference.md)。

## 项目结构

```
rcms_core/
  core.py       — 核心 + DB 连接 + 初始化
  db.py         — 建表 + 迁移
  session.py    — 对话历史写入
  retrieval.py  — 三通道召回 + embedding
  memory.py     — 长期记忆 + 蒸馏写入 + 图谱维护
  analysis.py   — LLM 蒸馏分析 + 9 维结构化写入
  context.py    — prompt 拼装
  utils.py      — 工具函数
plugins/rcms-astrbot/  — AstrBot 插件适配器
docs/                   — 详细文档
```

## 文档

- [参考手册](docs/rcms-reference.md) — 架构、表结构、配置详解
- [变更记录](docs/changelog.md)
- [P0 操作手册](docs/rcms_p0_operations.md) — 并发/迁移/embedding 等安全加固

## 许可证

MIT

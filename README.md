# Relational Cognitive Memory System (RCMS)

![License](https://img.shields.io/badge/license-Apache%202.0-blue?style=flat-square)
![Python](https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square&logo=python)
![SQLite](https://img.shields.io/badge/database-SQLite-brightgreen?style=flat-square&logo=sqlite)
![Status](https://img.shields.io/badge/status-v2.0-green?style=flat-square)
![AstrBot](https://img.shields.io/badge/AstrBot-plugin-orange?style=flat-square)

关系认知记忆系统 — 让 AI Agent 在长期对话中拥有人际关系感知。

RCMS 不是简单的对话历史存储。它在每次对话后对 LLM 回复做 **事后蒸馏分析**，抽取用户画像、情绪氛围、实体关系、关键事实等多维信息，并在下次对话前通过 **三通道融合召回** 把最相关的记忆注入到 LLM 的上下文中。

## 核心能力

- **三通道记忆召回** — 时间重要性 × 语义向量 × 图谱关系，加权融合
- **LLM 事后蒸馏** — 每次对话后异步分析，抽取 9 维结构化记忆（用户画像、情绪、实体关系、关键事实等）
- **情绪共振** — 识别当前对话的情绪氛围，召回忆中情绪匹配的条目
- **图扩散检索** — 关键词通过知识图谱 BFS 扩散，找到语义关联的间接记忆
- **多用户隔离** — 按 `user_id` / `session_id` 隔离数据，单库多租户
- **AstrBot 插件** — 仓库根目录即插件本体，可直接用 WebUI 安装、重装与更新

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

## 安装

### 方式一：AstrBot WebUI 安装（推荐）

仓库根目录就是插件本体（`metadata.yaml` / `main.py` / `_conf_schema.json` 均在根），
所以可以直接用 AstrBot 面板安装、重新安装和检查更新：

1. 打开 AstrBot 面板 → 「插件管理」→「安装插件」
2. 仓库地址填 `https://github.com/ATRI2233/Relational-Cognitive-Memory-System`
3. 安装后进入插件设置，配置 Embedding / LLM 的 API 地址、Key 与模型名
4. 之后在插件管理里点「检查更新」→「更新」即可升级，配置与记忆库都会保留

> 插件依赖：`jieba`、`pydantic`、`pydantic-settings`、`openai`（见 `requirements.txt`，AstrBot 可一键安装）。
> Git 直连失败时可在面板的代理栏填 `https://ghproxy.net`。

### 方式二：手动放置

把本仓库克隆或解压到 `AstrBot/data/plugins/astrbot_plugin_rcms/`（根目录内容即为插件内容），
重启 AstrBot 即可加载。

### 方式三：代码集成

```python
from adapter.rcms_factory import create_core

core = create_core("memory.db", llm_call=my_llm, embed_call=my_embed)
await core.chat_use_case.save_turn_only(request, reply=reply)
ctx = await core.retrieve_context_use_case.build_multi_user_context(
    session_id=sid, user_input=text, speaker_id=uid, speaker_name=name,
)
```

## 配置

配置全部集中在 `analysis` 段，面板「插件设置」里改或直接改 `data/config/astrbot_plugin_rcms_config.json`：

| 段 | 作用 |
| --- | --- |
| `analysis.retrieval` | 语义检索：Embedding 凭证、召回条数、通道保底、时间衰减、情绪加成 |
| `analysis.post_analysis` | 蒸馏分析：LLM 凭证、触发轮数/分钟、悬案过期轮数 |
| `output_log` | JSONL 对话摘要日志（默认 `<插件目录>/rcms_output.jsonl`，超限自动裁剪） |
| `storage.data_dir` | 记忆库目录，留空则用 AstrBot 数据区的 `plugin_data/astrbot_plugin_rcms` |

`source` 支持两种取值：

- `astrbot` — 由宿主注入的提供商（当前适配器只读取下面的 custom 字段，如需自动取用请在面板里填 custom）
- `custom` — 手动填 `custom_base_url` / `custom_api_key` / `custom_model`

## 数据与升级

- 记忆库默认位于 `<AstrBot 数据目录>/plugin_data/astrbot_plugin_rcms/rcms_memory.db`
- **不要把 `data_dir` 设在插件目录内**：AstrBot 升级插件时会删除并重建插件目录，写在里面的数据库会一起没掉
- 升级前可用 SQLite 直接备份 `rcms_memory.db`（WAL 模式下请连 `-wal`/`-shm` 一起备份，或先停插件）

## 项目结构

```
main.py                    — AstrBot 插件入口（协议转换，只做 hook 与配置装配）
adapter/rcms_factory.py    — 依赖注入工厂，组装对象图
application/               — 用例层：chat / distill / retrieve + 事件总线与 handler
domain/                    — 领域层：实体、事件、端口、服务
infrastructure/            — 基础设施：SQLite 仓储、DDL 迁移、配置与提示词
tests/                     — pytest 用例
scripts/                   — 数据分布分析、完整性检查等运维脚本
```

## 许可证

Apache 2.0

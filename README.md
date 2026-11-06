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

> **什么时候才会蒸馏**：攒够 `max_turns` 轮（默认 30）**或**距离上次蒸馏 `max_minutes` 分钟（默认 60）就触发一次，
> 两条任意一条满足即可。从没蒸过的会话，时间从「会话第一条消息」算起 —— 所以一个聊了几轮就放下、
> 两小时后回来接着说的会话，也会在下一句话时补蒸一次。蒸馏失败会按 `max_minutes` 冷却后再试，
> 不会每条消息都打一次 LLM。
>
> **中途关掉程序也不会丢**：蒸馏的计时锚点落在库里（`first_seen_at`），关程序的那段时间照样算。
> 更关键的是插件启动时会补蒸一次 —— 聊了几轮、没到门槛就关掉、之后再也不回那个会话的对话，
> 也会在下次启动时补成记忆（`catch_up_on_start`，默认开；一次最多补 `catch_up_max_sessions` 个会话，
> 默认 3 个，避免开机连打一串模型请求）。刚聊完 10 分钟的会话不会被抢跑。
- **情绪共振** — 识别当前对话的情绪氛围，召回忆中情绪匹配的条目
- **图扩散检索** — 关键词通过知识图谱 BFS 扩散，找到语义关联的间接记忆
- **多用户隔离** — 按 `user_id` / `session_id` 隔离数据，单库多租户
- **本地观测站** — 插件自带 `http://127.0.0.1:8765`，总览/记忆/对话/关系/召回五个视图，能看它记住了什么、为什么没记住，也能逐条回放提示词的三路合成过程
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
3. 装完即生效：插件默认跟随 AstrBot 面板里**当前生效的对话模型与 Embedding 提供商**，不需要在插件里再填任何 Key
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
| `analysis.retrieval` | 语义检索：模型来源、召回条数、通道保底、时间衰减、情绪加成 |
| `analysis.post_analysis` | 蒸馏分析：模型来源、触发轮数/分钟、悬案过期轮数 |
| `output_log` | JSONL 对话摘要日志（相对路径落在记忆库目录，超限自动裁剪） |
| `storage.data_dir` | 记忆库目录，留空则用 AstrBot 数据区的 `plugin_data/astrbot_plugin_rcms` |
| `webui` | 本地观测站：开关、地址、端口、是否记录三路召回追踪 |

`analysis.retrieval.source` 与 `analysis.post_analysis.source` 各有两种取值：

- **`astrbot`（默认，推荐）** — 自动使用 AstrBot 面板里当前生效的模型：
  对话模型走 `get_using_provider_async()`，Embedding 走已加载的 Embedding 提供商。
  在面板换模型，插件立刻跟着换，插件里不需要出现任何 Key。
  `astrbot_source_id` 留空即自动匹配，填了则固定绑定某个提供商 ID。
- **`custom`** — 完全自定义，使用 `custom_base_url` / `custom_api_key` / `custom_model`，
  适合想让记忆系统单独走另一个模型/额度的情况。

> Embedding 是可降级能力：宿主没有配置 Embedding 提供商、或上游模型报错时，
> 插件会暂停语义通道（默认冷却 300 秒）并在日志里说明，
> 时间通道与图谱通道照常工作，不会拖慢对话或中断蒸馏。
> 换 Embedding 模型导致向量维度变化时，旧向量会自动清空并按新模型重建。

> 对话落库与蒸馏在**后台任务**里执行：每轮回复不会被蒸馏的 LLM 调用拖慢，
> 插件重载时后台任务会被取消并等待收敛，不会留下半截事务。

## 本地观测站

插件自带一个**只读**的 localhost 网页，用来回答三个问题：它到底记住了什么、为什么这次没记住、这次提示词是怎么拼出来的。

插件加载后就会启动，不需要额外安装任何东西（后端只用 Python 标准库）：

```
http://127.0.0.1:8765/
```

端口被占用时会自动往后顺延（8765–8774），实际地址会打在 AstrBot 日志里：

```
RCMS 观测站已就绪，浏览器打开 http://127.0.0.1:8765/ 可查看记忆、对话与三路召回
```

### 视图

左侧按「记忆 / 过程 / 调试」分组，不再是把所有表平铺出来。

**① 总览** — 一眼看清链路通没通

| 区块 | 内容 |
| --- | --- |
| 状态条 | 记忆链路是否跑通；为空时直接说明「还差几轮才蒸馏」，而不是让你自己猜 |
| 指标卡 | 记忆条目、对话消息、实体节点、召回记录、身份画像、库体积 |
| 蒸馏进度 | 轮次与时间两条门槛各自一条进度条：`已攒轮次 / max_turns`、`已过分钟 / max_minutes` |
| 近 14 天活动 | 每天新增的对话消息数与记忆数 |
| 情绪分布 | `cognitive_distill.mood` 的分布，看这段时间的基调 |

**② 记忆条目** — 主视图，卡片流而不是表格

每条是「内容 + 情绪 + 重要性 + 实体标签 + 时间」，左侧色带按情绪着色，超过 3 行可展开全文。
支持全文搜索、按时间/重要性排序、按时间范围与情绪筛选；标签里的会话点一下直接跳到对应对话。

**③ 对话流** — 左边选会话，右边按聊天气泡逐条回放，用户/助手左右分色。

**④ 实体关系** — 顶部是环形关系网（点越大出现越多、线越粗关联越强），下面是实体频次与关系边清单。

**⑤ 三路召回** — 每次构建历史提示词都会留一条记录，点进去能看到完整链路：

| 区块 | 内容 |
| --- | --- |
| 本次摘要 | 用户输入、关键词、进了几条、耗时、Embedding 是否可用 |
| 三条通道 | 每路各自召回了哪些候选，逐条给出原始分、进度条与打分明细（余弦、重要性、时间衰减、同会话加成、边权、衰减后天数…），候选可点击展开 |
| 融合结果 | 谁走了「保底」、谁走了「补位」、加权分多少、哪几条进了提示词、哪几条被判重丢弃 |
| 图谱关系链 | 从命中节点出发的连通路径 |
| 最终注入文本 | 真正拼进 system prompt 的那段文字 |

三条通道全空时只给一条提示并说明原因，不再渲染三块空卡片。

**⑥ 原始数据表** — 调试用。逐表浏览、分页、文本搜索、点行看单行详情、导出当前页 CSV。
`embedding` 这类二进制列默认隐藏，长 ID 列省略显示，其余短列不换行——想看全字段点开右边抽屉。

地址栏记的是当前视图，可以收藏或分享：`#/memories`、`#/dialogue/<会话ID>`、`#/traces/17`、`#/raw/cognitive_distill`。

### 隐私与开销

- 只监听回环地址（默认 `127.0.0.1`），不做鉴权，所以**不要把它改到公网地址**
- 数据访问一律走 SQLite 只读连接（`mode=ro`），不会写入或修改记忆库
- 追踪记录单独存在 `retrieval_trace` 表里，默认只保留最近 200 条，超出自动清理
- 不想要就关掉：面板里把 `webui.enabled` 设为 false，或只设 `trace_enabled: false` 保留看板但不记录召回细节

## 数据与升级

- 记忆库默认位于 `<AstrBot 数据目录>/plugin_data/astrbot_plugin_rcms/rcms_memory.db`
- **不要把 `data_dir` 设在插件目录内**：AstrBot 升级插件时会删除并重建插件目录，写在里面的数据库会一起没掉
- 升级前可用 SQLite 直接备份 `rcms_memory.db`（WAL 模式下请连 `-wal`/`-shm` 一起备份，或先停插件）

### 废弃表自动清理（v2.2+）

v1 留下的一批表在 v2 里既没有写入方也没有读取方，插件加载时会自动删除，删除前把非空表导出成
`obsolete_tables_<时间戳>.json`（与数据库同目录）：

| 表 | 为什么删 |
| --- | --- |
| `shared_context` | 梗库，蒸馏从未产出 `shared_jokes`，从未写入过 |
| `memory_links` | 记忆间关联，v2 用图谱边取代 |
| `entity_relations` | v1 三元组关系，v2 用 `memory_graph_edges` |
| `emotional_trace` | 情绪轨迹，v2 情绪存在 `cognitive_distill.mood` |
| `relationship_arc` | 关系弧，v2 关系阶段存在 `session_state.stance` |
| `embedding_rebuild_queue` | 重建队列只进不出，没有消费者 |

空表直接删、不留备份文件；非空表（例如 v1 遗留的 `relationship_arc`）先备份再删。
迁移是幂等的，重复加载不会再产生文件。

## 项目结构

```
main.py                    — AstrBot 插件入口（协议转换，只做 hook 与配置装配）
observatory/               — 本地观测站：只读 HTTP 服务 + 单页前端（纯标准库）
adapter/rcms_factory.py    — 依赖注入工厂，组装对象图
application/               — 用例层：chat / distill / retrieve + 事件总线与 handler
domain/                    — 领域层：实体、事件、端口、服务
infrastructure/            — 基础设施：SQLite 仓储、DDL 迁移、配置与提示词
tests/                     — pytest 用例
scripts/                   — 数据分布分析、完整性检查等运维脚本
```

## 许可证

Apache 2.0

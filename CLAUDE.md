# RCMS — Relational Cognitive Memory System

关系认知记忆系统。让 AI Agent 在长期对话中有人际关系感知：
有共同记忆、有状态变化、有情绪氛围，但不走"完全模拟人类"路线。

## 项目结构

```
rcms_core.py              # 单文件核心：记忆存储 / 检索 / 图扩散 / prompt 格式化
backends/                  # LLM 后端（OpenAI / Mock）
plugins/rcms-astrbot/     # AstrBot 插件适配器
config.json                # 全局配置
data/                      # SQLite 数据库存放目录
scripts/                   # 工具脚本
```

## Pipeline

```
输入 → 关键词/图检索 → 记忆 → prompt_compressor → LLM → save_turn → _post_update
```

AstrBot 插件中分两段执行：
- `on_llm_request`: 检索记忆 → **narrative_context** → 注入 system_prompt
- `on_llm_response`: save_turn → _record_output → _post_update

## 数据库表

session_state / chat_history — 会话层
memory_graph_nodes / memory_graph_edges — 图检索
long_term_memory — 旧版关键词 LIKE 记忆
identity_memory / event_memory / emotional_trace / shared_context / relationship_arc — 长期5层表

## 注入方式（AstrBot 插件）

config.json → `general.injection_method`:
- `system_prompt`: 追加到 `req.system_prompt` 尾部（默认）
- `prompt_prefix`: 放在用户消息前
- `faketool`: 以工具调用结果注入

## 当前状态

v2.0 重构中：已砍掉规则分析层（stance/momentum/engagement），
待设计 LLM 直接产出分析的方案。

## 开发约定

- 语言：中文
- 数据库：SQLite，每人格一个独立文件
- 架构：单文件核心 + 后端隔离 + 插件适配

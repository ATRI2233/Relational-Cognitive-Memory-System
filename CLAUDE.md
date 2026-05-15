# RCMS — Relational Cognitive Memory System

关系认知记忆系统。让 AI Agent 在长期对话中有人际关系感知：
有共同记忆、有状态变化、有情绪氛围，但不走"完全模拟人类"路线。

## 项目结构

```
minimal_rcms.py          # 入口类 MinimalRCMS，继承三个 Mixin
rcms_context.py           # ContextMixin — Engagement / Stance 7态 / Momentum 2D / Working Memory
rcms_recall.py             # RecallMixin — 关键词检索 / 激活扩散图 / 抑制 / 误联想 / Prompt Compression / Narrative Context
rcms_memory.py            # MemoryMixin — 长期5层记忆 / Silent Recall Residue / 时间衰减 / Post-Update
backends/                  # LLM 后端（OpenAI / Mock）
plugins/rcms-astrbot/     # AstrBot 插件适配器
config.json                # 全局配置
data/                      # SQLite 数据库存放目录
tests/                     # 测试
scripts/                   # 工具脚本
```

## Pipeline（完整请求流）

```
chat() → engagement_trigger → _update_working_memory → _update_momentum
       → stance_manager → _recall (graph+BFS+inhibition+misrecall)
       → prompt_compressor → _core_veto → LLM → save_turn → _post_update
```

AstrBot 插件中分两段执行：
- `on_llm_request`: engagement → wm → momentum → stance → **narrative_context** → 注入 system_prompt
- `on_llm_response`: save_turn → _record_output → _post_update

## Stance 7 态

| Stance | 含义 | 冷却期 |
|---|---|---|
| open | 默认，放松状态 | 3 turns |
| reflective | 回想，语气沉 | 3 turns |
| guarded | 敏感话题，收着说 | 3 turns |
| playful | 轻松调侃 | 3 turns |
| analytical | 理性分析 | 3 turns |
| distant | 不想深入 | 3 turns |
| intimate | 亲近，敞开说 | 冷却后需条件触发 |

## 数据库表

session_state / chat_history / open_threads — 会话层
working_memory / memory_graph_nodes / memory_graph_edges — 运行态层
long_term_memory — 旧版单表记忆（关键词 LIKE）
identity_memory / event_memory / emotional_trace / shared_context / relationship_arc — 长期5层表

## 注入方式（AstrBot 插件）

config.json → `general.injection_method`:
- `system_prompt`: 追加到 `req.system_prompt` 尾部（默认）
- `prompt_prefix`: 放在用户消息前
- `faketool`: 以工具调用结果注入，与人格完全隔离

## 当前状态

- v1.0.0: 全部模块完成
- v1.0.1: 参数调优完成（30轮实测）
- v2.0（计划）: embedding 语义匹配（并行通道，不替换 keyword）

## 已知坑 / 注意事项

- Chinese tokenization 未处理，纯关键词 LIKE 匹配
- session_state 行可能不存在（首轮 init_db 处理）
- detect_stance（旧版 binary stance）仍暴露为公开方法，用于向后兼容
- prompt 可能随历史增长膨胀
- 重构进度：minimal_rcms.py 中的逻辑正逐步拆分到三个 Mixin 中

## 开发约定

- 语言：中文（代码注释、变量名、Prompt 内容）
- 模式：Mixin 多继承（ContextMixin / RecallMixin / MemoryMixin）
- 数据库：SQLite，每人格一个独立文件
- 测试：pytest，64 项

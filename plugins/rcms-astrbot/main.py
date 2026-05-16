"""
RCMS × AstrBot 适配器

从项目 config.json 加载配置，AstrBot 的设置项会覆盖 config.json 中的值。
"""
import asyncio
import json
import os
import re
import sys
import time

_self = os.path.dirname(os.path.abspath(__file__))
if _self not in sys.path:
    sys.path.insert(0, _self)

_INJECTION_METHODS = ("system_prompt", "prompt_prefix", "faketool")


def _load_project_config() -> dict:
    """加载项目 config.json，不存在时返回空 dict"""
    for base in (_self, os.path.dirname(os.path.abspath(__file__))):
        p = os.path.join(base, "config.json")
        if os.path.exists(p):
            try:
                with open(p, encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                break
    return {}

from astrbot.api import star
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.provider import LLMResponse, ProviderRequest
from astrbot.api.star import register
from astrbot.core import logger
from astrbot.core.provider.entities import ToolCallsResult
from astrbot.core.agent.message import AssistantMessageSegment, ToolCall, ToolCallMessageSegment

from rcms_core import MinimalRCMS

# 输出日志（JSONL 格式，自动轮换）— 默认值，init 时会被配置覆盖
_OUTPUT_LOG = os.path.join(_self, "rcms_output.jsonl")
_MAX_LOG_SIZE = 5 * 1024 * 1024


@register(
    "astrbot_plugin_rcms",
    "RCMS",
    "Relational Cognitive Memory System — 为 AI Agent 添加长期关系记忆与情感氛围感知",
    "0.2.0",
    "https://github.com/your/rcms",
)
class RcmsPlugin(star.Star):
    def __init__(self, context: star.Context, config: dict = None) -> None:
        super().__init__(context)
        # 合并：项目 config.json（底座）+ AstrBot 设置（覆盖）
        project_cfg = _load_project_config()
        astrbot_cfg = config or {}
        self.config = self._merge_cfg(project_cfg, astrbot_cfg)

        # {persona_name: MinimalRCMS} — 每个人格一个独立数据库
        self._rcms_instances: dict[str, MinimalRCMS] = {}
        self._lock = asyncio.Lock()

        self.user_id = self._get_cfg("user_id", "default_user")
        self.enabled = self._get_cfg("enabled", True)
        self.persona_separated = self._get_cfg("persona_separated", True)
        self.max_memories = self._get_cfg("max_memories_per_prompt", 2)
        self.injection_method = self._get_cfg("injection_method", "system_prompt")
        if self.injection_method not in _INJECTION_METHODS:
            logger.warning(f"RCMS: 未知注入方式 {self.injection_method}，使用 system_prompt")
            self.injection_method = "system_prompt"
        self._persona_cache: dict[str, str] = {}  # session_id → persona_name
        self._write_count = 0

        # 数据库存放目录：项目根目录下的 data/
        _project_root = os.path.dirname(os.path.dirname(_self))
        self._data_dir = os.path.join(_project_root, "data")
        os.makedirs(self._data_dir, exist_ok=True)

        # 输出日志配置
        global _OUTPUT_LOG, _MAX_LOG_SIZE
        ol = self._get_cfg("output_log", {})
        if isinstance(ol, dict):
            path = ol.get("path") or "rcms_output.jsonl"
            _OUTPUT_LOG = os.path.join(_self, path) if not os.path.isabs(path) else path
            mb = ol.get("max_size_mb", 5)
            _MAX_LOG_SIZE = int(mb) * 1024 * 1024

        log_level = self._get_cfg("log_level", "info")
        if log_level == "debug":
            logger.setLevel("DEBUG")
        logger.info("RCMS: 插件已加载")

    @staticmethod
    def _merge_cfg(base: dict, override: dict) -> dict:
        """递归合并两个配置字典（override 优先）"""
        merged = dict(base)
        for k, v in override.items():
            if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
                merged[k] = RcmsPlugin._merge_cfg(merged[k], v)
            else:
                merged[k] = v
        return merged

    def _get_cfg(self, key: str, default):
        for cat in ["general", "memory", "debug", "analysis"]:
            cat_obj = self.config.get(cat)
            if isinstance(cat_obj, dict) and key in cat_obj:
                return cat_obj[key]
        return self.config.get(key, default)

    def _get_analysis_config(self) -> dict:
        return self.config.get("analysis", {})

    def _get_rcms(self, persona_name: str) -> MinimalRCMS:
        if persona_name not in self._rcms_instances:
            safe_name = re.sub(r'[^a-zA-Z0-9_一-鿿]', '_', persona_name)
            db_path = os.path.join(self._data_dir, f"rcms_memory_{safe_name}.db")
            self._rcms_instances[persona_name] = MinimalRCMS(
                db_path=db_path,
                analysis_config=self._get_analysis_config(),
            )
            logger.info(f"RCMS: 创建人格记忆库 [{persona_name}] -> {db_path}")
        return self._rcms_instances[persona_name]

    async def _resolve_persona(self, event: AstrMessageEvent) -> str:
        """解析当前会话使用哪个人格

        按会话缓存人格名，避免每轮重复查询。
        返回人格名称，作为数据库隔离的 key。
        关闭人格分离时统一返回 'default'。
        """
        if not self.persona_separated:
            return "default"

        session_id = event.unified_msg_origin
        cached = self._persona_cache.get(session_id)
        if cached:
            return cached

        mgr = getattr(self.context, 'persona_manager', None)
        if mgr is None:
            logger.warning("RCMS: context 无 persona_manager，使用 default")
            self._persona_cache[session_id] = "default"
            return "default"

        name = None
        errors = []

        # 优先：传入 session 标识按会话解析
        try:
            persona = await mgr.get_default_persona_v3(umo=session_id)
            name = persona.get("name") if isinstance(persona, dict) else getattr(persona, 'name', None)
        except Exception as e:
            errors.append(f"get_default_persona_v3(umo): {e}")

        # 回退 1：不传 umo 拿全局默认
        if not name:
            try:
                persona = await mgr.get_default_persona_v3()
                name = persona.get("name") if isinstance(persona, dict) else getattr(persona, 'name', None)
            except Exception as e:
                errors.append(f"get_default_persona_v3(): {e}")

        # 回退 2：直接从缓存拿已选人格
        if not name:
            try:
                selected = getattr(mgr, 'selected_default_persona_v3', None)
                if selected:
                    name = selected.get("name") if isinstance(selected, dict) else getattr(selected, 'name', None)
            except Exception as e:
                errors.append(f"selected_default_persona_v3: {e}")

        # 回退 3：遍历 personas_v3 取第一个
        if not name:
            try:
                pv3 = getattr(mgr, 'personas_v3', None)
                if pv3 and len(pv3) > 0:
                    p = pv3[0]
                    name = p.get("name") if isinstance(p, dict) else getattr(p, 'name', None)
            except Exception as e:
                errors.append(f"personas_v3: {e}")

        if not name:
            name = "default"
            logger.warning(f"RCMS: 所有人格解析方式均失败 ({'; '.join(errors)})，使用 default")

        self._persona_cache[session_id] = name
        return name

    # ── 输出日志（JSONL 自动轮换） ──────────────────────────

    def _record_output(self, persona: str, stance: str, user_input: str, reply: str,
                       context_prompt: str = "", system_prompt: str = ""):
        """写入一条输出日志，超出 5MB 时删除最旧条目。"""
        entry = {
            "t": time.time(),
            "p": persona,
            "s": stance,
            "u": user_input,
            "r": reply,
        }
        if context_prompt:
            entry["cp"] = context_prompt
        if system_prompt:
            entry["sp"] = system_prompt
        line = json.dumps(entry, ensure_ascii=False) + "\n"
        path = _OUTPUT_LOG

        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line)
        except Exception as e:
            logger.warning(f"RCMS: 写入输出日志失败 ({e})")
            return

        # 惰性检查：每写 5 条检查一次文件大小
        self._write_count += 1
        if self._write_count % 5 == 0:
            self._trim_log()

    def _trim_log(self):
        """超出 5MB 时从头部删除最旧条目直到体积合规。"""
        try:
            size = os.path.getsize(_OUTPUT_LOG)
            if size <= _MAX_LOG_SIZE:
                return
            with open(_OUTPUT_LOG, "r", encoding="utf-8") as f:
                lines = f.readlines()
            # 从尾部往前保留，直到体积 ≤ 5MB
            keep = []
            total = 0
            for line in reversed(lines):
                total += len(line.encode("utf-8"))
                if total > _MAX_LOG_SIZE:
                    break
                keep.append(line)
            keep.reverse()
            if len(keep) < len(lines):
                with open(_OUTPUT_LOG, "w", encoding="utf-8") as f:
                    f.writelines(keep)
                dropped = len(lines) - len(keep)
                logger.info(f"RCMS: 输出日志轮换，已删除 {dropped} 条旧记录")
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.warning(f"RCMS: 日志轮换失败 ({e})")

    async def initialize(self) -> None:
        logger.info("RCMS: 插件初始化完成")

    async def terminate(self) -> None:
        for name, rcms in self._rcms_instances.items():
            try:
                rcms.close()
            except Exception:
                pass
        logger.info(f"RCMS: 已关闭 {len(self._rcms_instances)} 个人格记忆库")

    @filter.on_llm_request()
    async def on_llm_request(
        self, event: AstrMessageEvent, req: ProviderRequest
    ) -> None:
        """在 LLM 请求前注入 RCMS 关系上下文到 system_prompt"""
        if not self.enabled:
            return
        user_input = event.message_str or req.prompt or ""
        if not user_input.strip():
            return

        async with self._lock:
            persona_name = await self._resolve_persona(event)
            rcms = self._get_rcms(persona_name)

        session_id = event.unified_msg_origin
        sender_id = event.get_sender_id()
        user_id = sender_id or self.user_id

        # 检索记忆 + 长期上下文
        analysis_cfg = self._get_analysis_config()
        retrieval_cfg = analysis_cfg.get("retrieval", {})
        use_emb = retrieval_cfg.get("enabled", False)
        if use_emb:
            memories, emb_source = await rcms.retrieve_by_embedding(user_id, user_input, limit=2)
            logger.info(f"RCMS: [{persona_name}] emb_retrieve source={emb_source} hits={len(memories)}")
        else:
            memories = rcms.retrieve_memories(user_id, user_input, 'engaged', limit=2)
            logger.info(f"RCMS: [{persona_name}] kw_retrieve hits={len(memories)}")
        long_term = rcms._load_long_term_context(user_id)
        arc = long_term.get("arc_stage", "stranger")
        traits_count = len(long_term.get("identity_traits", []))
        logger.debug(f"RCMS: [{persona_name}] context arc={arc} traits={traits_count} shared={len(long_term.get('shared_contexts',[]))} entities={len(long_term.get('entities',[]))}")
        context_part = rcms.narrative_context('open', session_id,
                                               memories=memories, long_term=long_term)

        # 按配置的注入方式插入
        if self.injection_method == "system_prompt":
            req.system_prompt += f"\n\n{context_part}"
        elif self.injection_method == "prompt_prefix":
            req.prompt = f"{context_part}\n\n{req.prompt}" if req.prompt else context_part
        elif self.injection_method == "faketool":
            tool_id = f"rcms_{int(time.time())}"
            info = AssistantMessageSegment(
                content=None,
                tool_calls=[ToolCall(
                    id=tool_id,
                    function=ToolCall.FunctionBody(
                        name="get_rcms_context",
                        arguments="{}"
                    ),
                )],
            )
            result = ToolCallMessageSegment(
                role="tool",
                tool_call_id=tool_id,
                content=context_part,
            )
            req.tool_calls_result = ToolCallsResult(
                tool_calls_info=info,
                tool_calls_result=[result],
            )

        # 持久化中间状态供 response hook 使用
        event.set_extra("rcms_persona", persona_name)
        event.set_extra("rcms_user_input", user_input)
        event.set_extra("rcms_stance", "open")
        event.set_extra("rcms_session_id", session_id)
        event.set_extra("rcms_user_id", user_id)
        event.set_extra("rcms_context_prompt", context_part)
        event.set_extra("rcms_system_prompt", req.system_prompt)

    @filter.on_llm_response()
    async def on_llm_response(
        self, event: AstrMessageEvent, resp: LLMResponse
    ) -> None:
        persona_name = event.get_extra("rcms_persona", "default")
        user_input = event.get_extra("rcms_user_input", "")
        stance = event.get_extra("rcms_stance", "open")
        session_id = event.get_extra("rcms_session_id", "")
        user_id = event.get_extra("rcms_user_id", self.user_id)
        reply = resp.completion_text or ""

        if not user_input or not reply or not session_id:
            return
        if not self._get_cfg("enable_auto_save", True):
            return

        async with self._lock:
            rcms = self._get_rcms(persona_name)

        rcms.save_turn(session_id, user_input, reply, stance)

        # 记录输出日志（JSONL，自动轮换）
        context_prompt = event.get_extra("rcms_context_prompt", "")
        system_prompt = event.get_extra("rcms_system_prompt", "")
        self._record_output(persona_name, stance, user_input, reply,
                            context_prompt=context_prompt,
                            system_prompt=system_prompt)

        rcms._post_update(user_id, session_id, user_input, stance, reply)

        # 异步触发事后 ANALYSIS + 记忆向量化（fire-and-forget，不阻塞回复）
        analysis_cfg = self._get_analysis_config()
        retrieval_cfg = analysis_cfg.get("retrieval", {})
        post_cfg = analysis_cfg.get("post_analysis", {})

        # Embedding：新记忆入库后异步向量化
        if retrieval_cfg.get("enabled", False) and len(user_input) > 15:
            logger.debug(f"RCMS: [{persona_name}] schedule_embed")
            asyncio.create_task(self._delayed_embed(rcms, user_id, session_id, user_input))

        # ANALYSIS LLM
        if post_cfg.get("mode") == "llm":
            logger.info(f"RCMS: [{persona_name}] schedule_analysis mode=llm sampling={post_cfg.get('sampling',0)}")
            long_term = rcms._load_long_term_context(user_id)
            asyncio.create_task(rcms._run_analysis(user_id, user_input, reply, long_term))
        else:
            logger.debug(f"RCMS: [{persona_name}] post_analysis=rule (skip LLM)")

        logger.info(f"RCMS: [{persona_name}] done turn_len={len(user_input)+len(reply)}")

    async def _delayed_embed(self, rcms: MinimalRCMS, user_id: str, session_id: str, content: str):
        """延迟嵌入：获取最近一条未向量化的记忆并生成 embedding"""
        try:
            row = rcms.conn.execute(
                "SELECT id, content FROM long_term_memory WHERE user_id = ? AND session_id = ? ORDER BY created_at DESC LIMIT 1",
                (user_id, session_id),
            ).fetchone()
            if not row:
                return
            mem_id, text = row
            existing = rcms.conn.execute(
                "SELECT id FROM memory_embeddings WHERE user_id = ? AND memory_id = ?", (user_id, mem_id)
            ).fetchone()
            if existing:
                return
            vec = await rcms._get_embedding(text[:512])
            if vec:
                rcms._store_embedding(user_id, mem_id, text[:512], vec)
                rcms._load_emb_cache(user_id)
        except Exception:
            pass

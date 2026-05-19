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
            except Exception as e:
                logger.warning(f"RCMS: 加载 config.json 失败 ({e}) path={p}")
                break
    return {}

from astrbot.api import star
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.provider import LLMResponse, ProviderRequest
from astrbot.api.star import register
import logging

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
        self.injection_method = self._get_cfg("injection_method", "system_prompt")
        if self.injection_method not in _INJECTION_METHODS:
            logger.warning(f"RCMS: 未知注入方式 {self.injection_method}，使用 system_prompt")
            self.injection_method = "system_prompt"
        self._persona_cache: dict[str, str] = {}  # session_id → persona_name
        self._write_count = 0
        self._provider_callbacks = None  # 首次构建后缓存

        # 数据库存放目录：插件目录下
        self._data_dir = os.path.join(_self, "data")
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

        # 配置 rcms_core.py 的 logger 输出到文件
        _rcms_log_path = os.path.join(_self, "rcms.log")
        _rcms_logger = logging.getLogger("rcms")
        _rcms_logger.setLevel(logging.DEBUG)
        if not _rcms_logger.handlers:
            _fh = logging.FileHandler(_rcms_log_path, encoding="utf-8", mode="a")
            _fh.setLevel(logging.DEBUG)
            _fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
            _rcms_logger.addHandler(_fh)

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
        for cat in ["general", "memory", "analysis", "output_log", "debug"]:
            cat_obj = self.config.get(cat)
            if isinstance(cat_obj, dict) and key in cat_obj:
                return cat_obj[key]
        return self.config.get(key, default)

    def _get_analysis_config(self) -> dict:
        """返回分析配置（含 API 字段，供 rcms_core 使用）"""
        cfg = self.config.get("analysis", {}).copy()
        for key in ("retrieval", "post_analysis"):
            sub = cfg.get(key, {})
            if isinstance(sub, dict):
                merged = dict(sub)
                # 新 key 优先，旧 key 回退兼容
                if not merged.get("custom_api_key"):
                    merged["custom_api_key"] = sub.get("custom_token", "")
                if not merged.get("custom_base_url"):
                    merged["custom_base_url"] = sub.get("custom_url", "https://api.openai.com/v1")
                cfg[key] = merged
        return cfg

    def _build_provider_callbacks(self):
        """从 AstrBot 配置或自定义设置构造 LLM/Embedding 回调

        从 config.analysis.{retrieval,post_analysis} 读取来源配置：
          - source=astrbot: 外部读取 AstrBot cmd_config.json
            astrbot_source_id 指定 provider source ID，留空自动匹配
          - source=custom: 使用 custom_url / custom_token / custom_model

        AstrBot 配置运行时不变，首次构建后缓存复用。
        """
        if self._provider_callbacks is not None:
            return self._provider_callbacks
        analysis_cfg = self.config.get("analysis", {})

        def _read_api(key: str) -> tuple:
            """读取某功能的 API 配置，返回 (token, url, model, source, src_id)"""
            sub = analysis_cfg.get(key, {})
            source = sub.get("source", "astrbot")
            token = (sub.get("custom_api_key", "") or sub.get("custom_token", "")) or None if source == "custom" else None
            url = sub.get("custom_base_url", "") or sub.get("custom_url", "https://api.openai.com/v1")
            model = sub.get("custom_model", "")
            return token, url, model, source, sub.get("astrbot_source_id", "")

        emb_key, emb_url, emb_model, emb_src, emb_src_id = _read_api("retrieval")
        llm_key, llm_url, llm_model, llm_src, llm_src_id = _read_api("post_analysis")

        # ── 2. 从 AstrBot cmd_config.json 补充（source=astrbot） ──
        need_astrbot_emb = emb_key is None and emb_src == "astrbot"
        need_astrbot_llm = llm_key is None and llm_src == "astrbot"

        if need_astrbot_emb or need_astrbot_llm:
            try:
                candidates = [
                    os.path.expanduser("~/.astrbot/data/cmd_config.json"),
                    os.path.join(os.getcwd(), "data", "cmd_config.json"),
                ]
                cfg_path = next((p for p in candidates if os.path.exists(p)), None)

                if cfg_path:
                    with open(cfg_path, encoding="utf-8-sig") as f:
                        astrbot_cfg = json.load(f)

                    sources = {s["id"]: s for s in astrbot_cfg.get("provider_sources", [])}

                    # Embedding provider
                    if need_astrbot_emb:
                        # AstrBot v3: embedding 存在 provider[] 中 type=openai_embedding
                        emb_providers = [p for p in astrbot_cfg.get("provider", [])
                                         if p.get("enable", False) and (
                                             p.get("type") == "openai_embedding"
                                             or p.get("provider_type") == "embedding")]
                        if emb_providers:
                            ep = emb_providers[0]
                            emb_key = ep.get("embedding_api_key", "") or None
                            emb_url = ep.get("embedding_api_base", "https://api.openai.com/v1")
                            emb_model = ep.get("embedding_model", "text-embedding-3-small")
                        else:
                            # 旧版 AstrBot: 顶层 embedding_provider 字段
                            ec = astrbot_cfg.get("embedding_provider", {}) or {}
                            src_id = ec.get("provider_source_id", "")
                            if src_id:
                                src = sources.get(src_id)
                                if src:
                                    emb_key = (src.get("key", [""])[0] if isinstance(src.get("key"), list) else src.get("key", "")) or None
                                    emb_url = src.get("api_base", "https://api.openai.com/v1")
                                emb_model = ec.get("model", "text-embedding-3-small")

                    # LLM provider
                    if need_astrbot_llm:
                        src_id = llm_src_id
                        if not src_id:
                            providers = [p for p in astrbot_cfg.get("provider", []) if p.get("enable", False)]
                            default_id = astrbot_cfg.get("provider_settings", {}).get("default_provider_id", "")
                            target = next((p for p in providers if p["id"] == default_id), providers[0] if providers else None)
                            if target:
                                src_id = target.get("provider_source_id", "")
                                llm_model = target.get("model", "gpt-4o")
                        src = sources.get(src_id)
                        if src:
                            llm_key = (src.get("key", [""])[0] if isinstance(src.get("key"), list) else src.get("key", "")) or None
                            llm_url = src.get("api_base", "https://api.openai.com/v1")

                    logger.info(f"RCMS: AstrBot provider loaded — LLM={llm_model} Embed={emb_model}")
            except Exception as e:
                logger.warning(f"RCMS: AstrBot provider 加载失败 ({e})")

        # ── 3. Fallback: embedding 未找到时复用 LLM 凭据 ──
        if emb_key is None and llm_key is not None:
            emb_key = llm_key
            emb_url = llm_url
            if not emb_model:
                emb_model = "text-embedding-3-small"
            logger.info(f"RCMS: Embedding fallback to LLM provider ({emb_model})")

        # ── 4. 构造回调 ──
        from openai import AsyncOpenAI

        emb_callable = None
        if emb_key:
            _ec = AsyncOpenAI(api_key=emb_key, base_url=emb_url)
            async def _embed(text):
                nonlocal _ec
                resp = await _ec.embeddings.create(model=emb_model, input=text.replace("\n", " "))
                return resp.data[0].embedding
            emb_callable = _embed

        llm_callable = None
        if llm_key:
            _lc = AsyncOpenAI(api_key=llm_key, base_url=llm_url)
            async def _llm(prompt, model=""):
                nonlocal _lc
                m = model or llm_model
                resp = await _lc.chat.completions.create(
                    model=m, messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                )
                return resp.choices[0].message.content or "{}"
            llm_callable = _llm

        self._provider_callbacks = (llm_callable, emb_callable)
        return llm_callable, emb_callable

    def _get_rcms(self, persona_name: str) -> MinimalRCMS:
        if persona_name not in self._rcms_instances:
            safe_name = re.sub(r'[^a-zA-Z0-9_一-鿿]', '_', persona_name)
            db_path = os.path.join(self._data_dir, f"rcms_memory_{safe_name}.db")
            llm_cb, emb_cb = self._build_provider_callbacks()
            self._rcms_instances[persona_name] = MinimalRCMS(
                db_path=db_path,
                analysis_config=self._get_analysis_config(),
                llm_call=llm_cb,
                embed_call=emb_cb,
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
        # 预热 default 人格：第一条消息不用等建库
        try:
            self._get_rcms("default")
        except Exception:
            logger.exception("RCMS: failed to prewarm default persona")
        logger.info("RCMS: 插件初始化完成")

    async def terminate(self) -> None:
        for name, rcms in self._rcms_instances.items():
            try:
                rcms.close()
            except Exception:
                logger.exception(f"RCMS: failed to close rcms instance {name}")
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
        sender_name = event.get_sender_name() or sender_id

        # 三通道融合召回（通道 1：原始消息 × 时间，通道 2：蒸馏语义 + 情绪，通道 3：图谱骨架）
        memories = await rcms.retrieve_memories(user_id, user_input, 'engaged', session_id=session_id)
        logger.info(f"RCMS: [{persona_name}] retrieve_memories hits={len(memories)}")
        long_term = rcms._load_long_term_context(user_id)
        traits_count = len(long_term.get("identity_traits", []))
        logger.info(f"RCMS: [{persona_name}] context traits={traits_count} shared={len(long_term.get('shared_contexts',[]))} entities={len(long_term.get('entities',[]))}")
        context_part = rcms.narrative_context('open', session_id,
                                               memories=memories, long_term=long_term,
                                               user_id=user_id)

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
        event.set_extra("rcms_sender_name", sender_name)
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

        sender_name = event.get_extra("rcms_sender_name", "")
        rcms.save_turn(session_id, user_input, reply, user_id=user_id, sender_name=sender_name)

        # 记录输出日志（JSONL，自动轮换）
        context_prompt = event.get_extra("rcms_context_prompt", "")
        system_prompt = event.get_extra("rcms_system_prompt", "")
        self._record_output(persona_name, stance, user_input, reply,
                            context_prompt=context_prompt,
                            system_prompt=system_prompt)

        # 事后处理异步化（不阻塞回复）
        asyncio.create_task(self._async_post_update(
            rcms, user_id, session_id, user_input, stance, reply
        ))

        # 蒸馏分析 + 记忆向量化（fire-and-forget）
        analysis_cfg = self._get_analysis_config()
        retrieval_cfg = analysis_cfg.get("retrieval", {})

        # Embedding：新记忆入库后异步向量化
        if retrieval_cfg.get("embedding_enabled", retrieval_cfg.get("enabled", False)) and len(user_input) > 15:
            logger.debug(f"RCMS: [{persona_name}] schedule_embed")
            asyncio.create_task(self._delayed_embed(rcms, user_id, session_id, user_input))

        # 蒸馏检查：post_update_rules 后触发 LLM 蒸馏分析
        asyncio.create_task(self._check_and_distill(rcms, user_id, session_id, persona_name))

        logger.info(f"RCMS: [{persona_name}] done turn_len={len(user_input)+len(reply)}")

    async def _delayed_embed(self, rcms: MinimalRCMS, user_id: str, session_id: str, content: str):
        """延迟嵌入：获取最近一条未向量化的记忆并生成 embedding"""
        try:
            row = rcms.conn.execute(
                "SELECT id, content FROM cognitive_distill WHERE user_id = ? AND session_id = ? AND embedding IS NULL ORDER BY created_at DESC LIMIT 1",
                (user_id, session_id),
            ).fetchone()
            if not row:
                return
            rec_id, text = row
            vec = await rcms._get_embedding(text[:512])
            if vec:
                rcms._store_embedding(user_id, rec_id, vec)
                rcms._load_emb_cache(user_id)
        except Exception:
            logger.exception(f"RCMS: delayed_embed failed user={user_id} session={session_id}")

    async def _async_post_update(self, rcms: MinimalRCMS, user_id: str, session_id: str,
                                  user_input: str, stance: str, reply: str):
        """异步执行事后更新（纯规则），不阻塞 LLM 回复返回"""
        try:
            await rcms.post_update_rules(user_id, session_id, user_input, stance, reply)
        except Exception:
            logger.exception(f"RCMS: async_post_update failed user={user_id} session={session_id}")

    async def _check_and_distill(self, rcms: MinimalRCMS, user_id: str, session_id: str,
                                  persona_name: str):
        """检查蒸馏条件，触发 LLM 蒸馏分析"""
        try:
            triggered, last_turn, turn_count, snapshot, senders = rcms.check_distill_needed(session_id, persona_name=persona_name)
            if triggered:
                logger.info(f"RCMS: [{persona_name}] distill triggered turn={last_turn}→{turn_count} senders={senders}")
                long_term = rcms._load_long_term_context(user_id)
                await rcms._run_distill_analysis(user_id, session_id, snapshot, long_term, last_turn, turn_count, persona_name=persona_name, senders=senders)
            else:
                logger.debug(f"RCMS: [{persona_name}] distill not needed")
        except Exception:
            logger.exception(f"RCMS: [{persona_name}] distill check failed")

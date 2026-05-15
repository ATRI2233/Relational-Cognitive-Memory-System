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

from minimal_rcms import MinimalRCMS

# 输出日志（JSONL 格式，自动轮换）
_OUTPUT_LOG = os.path.join(_self, "rcms_output.jsonl")
_MAX_LOG_SIZE = 5 * 1024 * 1024  # 5 MB


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
        self._default_persona_cache: tuple[str, str] | None = None
        self._write_count = 0

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
        for cat in ["general", "memory", "debug"]:
            cat_obj = self.config.get(cat)
            if isinstance(cat_obj, dict) and key in cat_obj:
                return cat_obj[key]
        return self.config.get(key, default)

    def _get_rcms(self, persona_name: str) -> MinimalRCMS:
        """获取/创建对应人格的 RCMS 实例（每人格独立数据库）"""
        if persona_name not in self._rcms_instances:
            safe_name = re.sub(r'[^a-zA-Z0-9_一-鿿]', '_', persona_name)
            db_path = os.path.join(_self, f"rcms_memory_{safe_name}.db")
            self._rcms_instances[persona_name] = MinimalRCMS(db_path=db_path)
            logger.info(f"RCMS: 创建人格记忆库 [{persona_name}] -> {db_path}")
        return self._rcms_instances[persona_name]

    async def _resolve_persona(self, event: AstrMessageEvent) -> str:
        """解析当前会话使用哪个人格

        返回人格名称，作为数据库隔离的 key。
        关闭人格分离时统一返回 'default'。
        """
        if not self.persona_separated:
            return "default"
        try:
            mgr = getattr(self.context, 'persona_manager', None)
            if mgr is None:
                return "default"
            persona = await mgr.get_default_persona_v3()
            name = persona.get("name") if isinstance(persona, dict) else getattr(persona, 'name', None)
            return name or "default"
        except Exception as e:
            logger.debug(f"RCMS: 获取人格失败 ({e})，使用 default")
            return "default"

    # ── 输出日志（JSONL 自动轮换） ──────────────────────────

    def _record_output(self, persona: str, stance: str, user_input: str, reply: str):
        """写入一条输出日志，超出 5MB 时删除最旧条目。"""
        entry = {
            "t": time.time(),
            "p": persona,
            "s": stance,
            "u": user_input,
            "r": reply,
        }
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
        """在 LLM 请求前注入 RCMS 关系上下文到 system_prompt
        完整 Pipeline: Engagement → Working Memory → Momentum → Stance → Prompt Compression
        """
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

        # 全 Pipeline：Engagement → Working Memory → Momentum → Stance
        engagement = rcms.engagement_trigger(user_id, session_id, user_input)
        wm = rcms._update_working_memory(user_id, session_id, user_input, engagement)
        momentum = rcms._update_momentum(user_id, session_id, user_input, engagement, wm)
        stance = rcms.stance_manager(user_id, session_id, user_input, engagement)

        # 生成关系上下文字段
        prompt = rcms.prompt_compressor(user_id, session_id, user_input,
                                         stance, engagement, momentum)
        context_part = prompt.rsplit("\n用户: ", 1)[0]

        req.system_prompt += f"\n\n【RCMS 关系上下文】\n{context_part}"

        # 持久化中间状态供 response hook 使用
        event.set_extra("rcms_persona", persona_name)
        event.set_extra("rcms_user_input", user_input)
        event.set_extra("rcms_stance", stance)
        event.set_extra("rcms_session_id", session_id)
        event.set_extra("rcms_engagement", json.dumps(engagement))
        event.set_extra("rcms_momentum", json.dumps(momentum))
        event.set_extra("rcms_user_id", user_id)

    @filter.on_llm_response()
    async def on_llm_response(
        self, event: AstrMessageEvent, resp: LLMResponse
    ) -> None:
        persona_name = event.get_extra("rcms_persona", "default")
        user_input = event.get_extra("rcms_user_input", "")
        stance = event.get_extra("rcms_stance", "open")
        session_id = event.get_extra("rcms_session_id", "")
        user_id = event.get_extra("rcms_user_id", self.user_id)
        engagement_json = event.get_extra("rcms_engagement", "{}")
        momentum_json = event.get_extra("rcms_momentum", "[0.0, 0.0]")
        reply = resp.completion_text or ""

        if not user_input or not reply or not session_id:
            return
        if not self._get_cfg("enable_auto_save", True):
            return

        async with self._lock:
            rcms = self._get_rcms(persona_name)

        rcms.save_turn(session_id, user_input, reply, stance)

        # 记录输出日志（JSONL，自动轮换）
        self._record_output(persona_name, stance, user_input, reply)

        engagement = json.loads(engagement_json)
        momentum = tuple(json.loads(momentum_json))
        rcms._post_update(user_id, session_id, user_input, stance, engagement, momentum, reply)

        logger.debug(f"RCMS: [{persona_name}] 已记录 [{stance}]")

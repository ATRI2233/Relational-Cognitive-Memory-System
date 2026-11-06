"""
RCMS × AstrBot 插件入口。

本文件即插件主类：仓库根目录可直接作为 AstrBot 插件安装（含 metadata.yaml /
_conf_schema.json），因此 WebUI 的「安装」「重新安装」「检查更新」按钮都能正常工作。

职责边界：只做协议转换（AstrMessageEvent → ChatRequest → 注入 system_prompt），
不在插件层执行任何业务逻辑或原始 SQL，实现都在 adapter/ application/ domain/
infrastructure/ 里。

数据目录策略：默认放在 AstrBot 数据区（<astrbot_data>/plugin_data/astrbot_plugin_rcms），
不放插件目录内 —— AstrBot 升级插件时会整目录删除重建，放在插件目录里的数据库会被清掉。
需要自定义时在配置面板的「存储」里填 data_dir。
"""
from __future__ import annotations

import json
import os
import sys

_self = os.path.dirname(os.path.abspath(__file__))
if _self not in sys.path:
    sys.path.insert(0, _self)

from datetime import datetime

from astrbot.api import star
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.provider import LLMResponse, ProviderRequest
from astrbot.api.star import register
from astrbot.core import logger

from adapter.rcms_factory import create_core, CoreContext

DEFAULT_DATA_DIRNAME = "astrbot_plugin_rcms"


def _resolve_data_dir(config: dict) -> str:
    """解析记忆库所在目录。

    优先级：配置项 storage.data_dir > 顶层 data_dir（旧版兼容） >
    AstrBot 数据区的 plugin_data/<插件名> > 插件目录下的 data。

    Args:
        config: 合并后的插件配置。

    Returns:
        绝对路径（目录可能尚不存在，调用方负责创建）。
    """
    configured = ""
    storage = config.get("storage")
    if isinstance(storage, dict):
        configured = str(storage.get("data_dir") or "").strip()
    if not configured:
        configured = str(config.get("data_dir") or "").strip()
    if configured:
        return os.path.abspath(os.path.expanduser(configured))

    try:
        from astrbot.core.utils.astrbot_path import get_astrbot_data_path

        return os.path.join(
            get_astrbot_data_path(), "plugin_data", DEFAULT_DATA_DIRNAME
        )
    except Exception:  # pragma: no cover - 兼容旧版 AstrBot
        return os.path.join(_self, "data")


@register(
    "astrbot_plugin_rcms",
    "RCMS",
    "Relational Cognitive Memory System v2",
    "2.0.1",
    "https://github.com/ATRI2233/Relational-Cognitive-Memory-System",
)
class RcmsPlugin(star.Star):
    """RCMS AstrBot 插件 — 只做协议转换。"""

    def __init__(self, context: star.Context, config: dict = None) -> None:
        super().__init__(context)
        config = config or {}
        general = config.get("general") if isinstance(config.get("general"), dict) else {}
        self.enabled = general.get("enabled", config.get("enabled", True))
        self.user_id = general.get("user_id", config.get("user_id", "default_user"))
        self._core: CoreContext | None = None
        self._config = config

        # 仓库内附带的 config.json 作为默认值，AstrBot 面板配置优先覆盖
        _cfg_path = os.path.join(_self, "config.json")
        if os.path.exists(_cfg_path):
            try:
                with open(_cfg_path, encoding="utf-8") as _f:
                    project_cfg = json.load(_f)
                self._config = {**project_cfg, **config}
            except (json.JSONDecodeError, IOError):
                pass

    def _get_core(self) -> CoreContext:
        if self._core is None:
            data_dir = _resolve_data_dir(self._config)
            os.makedirs(data_dir, exist_ok=True)
            db_path = os.path.join(data_dir, "rcms_memory.db")
            self._db_path = db_path
            llm_callable, embed_callable = self._build_callbacks()
            self._core = create_core(
                db_path=db_path,
                llm_call=llm_callable,
                embed_call=embed_callable,
            )
        return self._core

    async def terminate(self) -> None:
        """关闭 RCMS 核心，执行 WAL TRUNCATE 后释放资源。

        AstrBot 在卸载/重载插件时会 await 本方法，因此必须是协程。
        """
        if self._core is not None:
            db_path = getattr(self, "_db_path", None)
            if db_path:
                try:
                    import sqlite3

                    tmp = sqlite3.connect(db_path)
                    tmp.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                    tmp.close()
                except Exception as e:
                    logger.debug("terminate WAL checkpoint 跳过: %s", e)
            # 必须显式关闭核心连接：仓储 / 事件总线之间存在引用环，
            # 只把 self._core 置空并不会立刻触发回收，句柄会一直留在
            # 进程里，导致卸载或更新插件时数据库文件被占用（WinError 32）。
            conn = getattr(self._core, "conn", None)
            if conn is not None:
                try:
                    conn.close()
                except Exception as e:
                    logger.debug("terminate 关闭数据库连接失败: %s", e)
            self._core = None
            logger.info("RCMS 核心已关闭")

    def _build_callbacks(self):
        analysis_cfg = self._config.get("analysis", {})
        if not analysis_cfg:
            return None, None
        llm_cfg = analysis_cfg.get("post_analysis", {})
        llm_callable = None
        if llm_cfg.get("api_key") or llm_cfg.get("custom_api_key"):
            from openai import AsyncOpenAI

            client = AsyncOpenAI(
                api_key=llm_cfg.get("api_key") or llm_cfg["custom_api_key"],
                base_url=llm_cfg.get("base_url")
                or llm_cfg.get("custom_base_url", "https://api.openai.com/v1"),
            )

            async def _llm(prompt: str) -> str:
                resp = await client.chat.completions.create(
                    model=llm_cfg.get("model")
                    or llm_cfg.get("custom_model", "gpt-4o-mini"),
                    messages=[{"role": "user", "content": prompt}],
                )
                return resp.choices[0].message.content or ""

            llm_callable = _llm
        emb_cfg = analysis_cfg.get("retrieval", {})
        embed_callable = None
        if emb_cfg.get("api_key") or emb_cfg.get("custom_api_key"):
            from openai import AsyncOpenAI

            ecl = AsyncOpenAI(
                api_key=emb_cfg.get("api_key") or emb_cfg["custom_api_key"],
                base_url=emb_cfg.get("base_url")
                or emb_cfg.get("custom_base_url", "https://api.openai.com/v1"),
            )

            async def _embed(text: str) -> list[float]:
                resp = await ecl.embeddings.create(
                    model=emb_cfg.get("model")
                    or emb_cfg.get("custom_model", "text-embedding-3-small"),
                    input=text.replace("\n", " "),
                )
                return resp.data[0].embedding

            embed_callable = _embed
        return llm_callable, embed_callable

    @filter.on_llm_request()
    async def on_llm_request(self, event: AstrMessageEvent, req: ProviderRequest) -> None:
        if not self.enabled:
            return
        user_input = event.message_str or req.prompt or ""
        if not user_input.strip():
            return
        event.set_extra("rcms_user_input", user_input)
        core = self._get_core()
        uid = event.get_sender_id() or self.user_id
        sid = uid
        sender_name = event.get_sender_name() or uid
        ctx = await core.retrieve_context_use_case.build_multi_user_context(
            session_id=sid,
            user_input=user_input,
            speaker_id=uid,
            speaker_name=sender_name,
        )
        if ctx:
            if req.system_prompt is None:
                req.system_prompt = ""
            req.system_prompt += f"\n\n{ctx}"

    @filter.on_llm_response()
    async def on_llm_response(self, event: AstrMessageEvent, resp: LLMResponse) -> None:
        if not self.enabled:
            return
        user_input = event.get_extra("rcms_user_input", "")
        reply = resp.completion_text or ""
        if not user_input or not reply:
            return
        core = self._get_core()
        uid = event.get_sender_id() or self.user_id
        sid = uid
        from application.use_cases.chat_use_case import ChatRequest

        request = ChatRequest(
            user_id=uid,
            session_id=sid,
            user_input=user_input,
            sender_name=event.get_sender_name() or uid,
        )
        await core.chat_use_case.save_turn_only(request, reply=reply)
        logger.info(f"RCMS: [{sid}] saved turn")

        self._write_output_log(sid, uid, user_input, reply)

    def _write_output_log(
        self, session_id: str, user_id: str, user_input: str, reply: str
    ) -> None:
        """按配置写 JSONL 对话摘要日志（超出上限时从头裁剪）。"""
        log_cfg = self._config.get("output_log")
        log_cfg = log_cfg if isinstance(log_cfg, dict) else {}
        if not log_cfg.get("enabled", True):
            return
        try:
            log_path = str(log_cfg.get("path") or "rcms_output.jsonl")
            if not os.path.isabs(log_path):
                log_path = os.path.join(_self, log_path)
            max_bytes = int(log_cfg.get("max_size_mb", 5)) * 1024 * 1024
            entry = json.dumps(
                {
                    "timestamp": datetime.now().isoformat(),
                    "session": session_id,
                    "user_id": user_id,
                    "user_input": user_input[:200],
                    "reply": reply[:200],
                },
                ensure_ascii=False,
            )
            with open(log_path, "a", encoding="utf-8") as lf:
                lf.write(entry + "\n")
            if os.path.getsize(log_path) > max_bytes:
                lines = open(log_path, encoding="utf-8").readlines()
                with open(log_path, "w", encoding="utf-8") as lf:
                    lf.writelines(lines[100:])
        except Exception:
            pass

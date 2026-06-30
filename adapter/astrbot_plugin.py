"""
RCMS × AstrBot 适配器 — 精简版。

只做协议转换：AstrMessageEvent → ChatRequest → 注入 system_prompt。
不在插件层执行任何业务逻辑或原始 SQL。
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


@register(
    "astrbot_plugin_rcms",
    "RCMS",
    "Relational Cognitive Memory System v2",
    "2.0.0",
    "https://github.com/your/rcms",
)
class RcmsPlugin(star.Star):
    """RCMS AstrBot 插件 — 只做协议转换。"""

    def __init__(self, context: star.Context, config: dict = None) -> None:
        super().__init__(context)
        config = config or {}
        self.enabled = config.get("enabled", True)
        self.user_id = config.get("user_id", "default_user")
        self._core: CoreContext | None = None
        self._config = config

        # Load project config.json if available
        _cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config.json")
        if os.path.exists(_cfg_path):
            try:
                with open(_cfg_path, encoding="utf-8") as _f:
                    project_cfg = json.load(_f)
                # Merge project config as defaults, plugin config overrides
                merged = {**project_cfg, **config}
                self._config = merged
            except (json.JSONDecodeError, IOError):
                pass

    def _get_core(self) -> CoreContext:
        if self._core is None:
            data_dir = self._config.get("data_dir", os.path.join(_self, "data"))
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

    def terminate(self) -> None:
        """关闭 RCMS 核心，执行 WAL TRUNCATE 后释放资源。

        对应旧版插件的 terminate() 生命周期方法。
        """
        if self._core is not None:
            db_path = getattr(self, '_db_path', None)
            if db_path:
                try:
                    import sqlite3
                    tmp = sqlite3.connect(db_path)
                    tmp.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                    tmp.close()
                except Exception as e:
                    logger.debug("terminate WAL checkpoint 跳过: %s", e)
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
                base_url=llm_cfg.get("base_url") or llm_cfg.get("custom_base_url", "https://api.openai.com/v1"),
            )
            async def _llm(prompt: str) -> str:
                resp = await client.chat.completions.create(
                    model=llm_cfg.get("model") or llm_cfg.get("custom_model", "gpt-4o-mini"),
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
                base_url=emb_cfg.get("base_url") or emb_cfg.get("custom_base_url", "https://api.openai.com/v1"),
            )
            async def _embed(text: str) -> list[float]:
                resp = await ecl.embeddings.create(
                    model=emb_cfg.get("model") or emb_cfg.get("custom_model", "text-embedding-3-small"),
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
            session_id=sid, user_input=user_input,
            speaker_id=uid, speaker_name=sender_name,
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
            user_id=uid, session_id=sid,
            user_input=user_input, sender_name=event.get_sender_name() or uid,
        )
        await core.chat_use_case.save_turn_only(request, reply=reply)
        logger.info(f"RCMS: [{sid}] saved turn")

        # Output log: JSONL 记录对话摘要，含 5MB 自动轮换
        try:
            log_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "..", "rcms_output.jsonl"
            )
            entry = json.dumps({
                "timestamp": datetime.now().isoformat(),
                "session": sid,
                "user_id": uid,
                "user_input": user_input[:200],
                "reply": reply[:200],
            }, ensure_ascii=False)
            with open(log_path, "a", encoding="utf-8") as lf:
                lf.write(entry + "\n")
            if os.path.getsize(log_path) > 5 * 1024 * 1024:
                lines = open(log_path, encoding="utf-8").readlines()
                with open(log_path, "w", encoding="utf-8") as lf:
                    lf.writelines(lines[100:])
        except Exception:
            pass

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

import asyncio
import json
import os
import sys
import time

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

# Embedding provider 连续失败后的冷却时间（秒）。
# 语义通道是可降级的：上游模型不可用时，时间/图谱通道仍然完整工作，
# 但不能每轮对话都去等一次超时，否则会明显拖慢回复。
_EMBED_FAIL_COOLDOWN_SECONDS = 300

# 单次调用超时。provider 自带 timeout 往往很宽松（默认 20s），
# 上游异常时会把 on_llm_request 卡住，直接体现为用户等首字。
_EMBED_TIMEOUT_SECONDS = 4
_LLM_TIMEOUT_SECONDS = 120


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
    "2.5.0",
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
        # 模型解析是运行期惰性的：这里只维护状态，不在这里抓 provider
        self._current_umo: str | None = None
        self._noticed: set[str] = set()
        self._embed_fail_until = 0.0
        # 后台落库任务，terminate 时统一收敛
        self._bg_tasks: set[asyncio.Task] = set()
        # 本地观测站（localhost 只读面板）
        self._webui = None
        self._webui_cfg: dict = {}
        # 启动补蒸：关程序前没来得及蒸的会话，加载后补一次
        self._catch_up_task: asyncio.Task | None = None
        self._catch_up_done = False

        # 仓库内附带的 config.json 作为默认值，AstrBot 面板配置优先覆盖
        _cfg_path = os.path.join(_self, "config.json")
        if os.path.exists(_cfg_path):
            try:
                with open(_cfg_path, encoding="utf-8") as _f:
                    project_cfg = json.load(_f)
                self._config = {**project_cfg, **config}
            except (json.JSONDecodeError, IOError):
                pass

        # 观测站要在插件一加载就能打开，不能等到第一条消息
        self._ensure_schema_ready()
        self._start_webui()
        # 把上次关程序前没蒸完的会话补上（要等宿主把模型 provider 备好）
        self._ensure_catch_up(delay=20)

    # ── 本地观测站 ──────────────────────────────────────────────────

    def _ensure_schema_ready(self) -> None:
        """插件加载时就把 Schema 迁移做完。

        核心对象是懒创建的（第一条消息才建），但废弃表清理、索引补齐这类
        迁移如果拖到那时，观测站会先显示一堆早该删掉的表。这里单独开一次
        连接跑完 ensure_schema 再关掉，代价只有几毫秒。

        失败不影响插件工作：真正建库时 create_core 还会再跑一遍。
        """
        try:
            import sqlite3

            from infrastructure.persistence.ddl import ensure_schema

            data_dir = _resolve_data_dir(self._config)
            os.makedirs(data_dir, exist_ok=True)
            conn = sqlite3.connect(
                os.path.join(data_dir, "rcms_memory.db"),
                check_same_thread=False,
                timeout=5.0,
            )
            try:
                ensure_schema(conn, self._build_settings(data_dir).storage)
            finally:
                conn.close()
        except Exception as exc:  # noqa: BLE001 - 迁移失败不该拦住插件
            logger.warning("RCMS: Schema 预检失败（会在首次对话时重试）: %s", exc)

    def _start_webui(self) -> None:
        """启动插件自带的 localhost 只读面板。

        只读、只监听回环地址，失败不影响插件正常工作。
        """
        cfg = self._config.get("webui")
        cfg = cfg if isinstance(cfg, dict) else {}
        self._webui_cfg = cfg
        if not cfg.get("enabled", True):
            logger.info("RCMS: 观测站已关闭（webui.enabled = false）")
            return
        try:
            from observatory.server import RcmsWebServer

            data_dir = _resolve_data_dir(self._config)
            db_path = os.path.join(data_dir, "rcms_memory.db")
            # 面板本身读不到运行期参数，把这些门槛传进去，
            # 它才能在总览里解释「为什么记忆还是空的」。
            hints: dict = {}
            try:
                settings = self._build_settings(data_dir)
                hints = {
                    "distill_max_turns": settings.analysis.max_turns,
                    "distill_max_minutes": settings.analysis.max_minutes,
                    "total_cap": settings.retrieval.total_cap,
                    "embedding_enabled": settings.retrieval.embedding_enabled,
                }
            except Exception as exc:  # noqa: BLE001 - 提示信息缺失不影响面板
                logger.debug("RCMS: 观测站参数读取失败: %s", exc)
            server = RcmsWebServer(
                db_path=db_path,
                host=str(cfg.get("host") or "127.0.0.1"),
                port=int(cfg.get("port") or 8765),
                hints=hints,
            )
            if server.start():
                self._webui = server
                logger.info(
                    "RCMS 观测站已就绪，浏览器打开 %s 可查看记忆、对话与三路召回",
                    server.url,
                )
        except Exception as exc:  # noqa: BLE001 - 观测站不能拖垮插件
            logger.warning("RCMS: 观测站启动失败（不影响记忆功能）: %s", exc)

    # ── 启动补蒸 ────────────────────────────────────────────────────

    def _ensure_catch_up(
        self, delay: float = 0.0, exclude_session_id: str = "",
    ) -> None:
        """排一个后台补蒸任务（幂等）。

        蒸馏只在新一轮对话存库（TurnSaved）时检查，所以「聊了几轮、没到
        门槛就关掉程序、之后再也不回这个会话」的对话，原本永远不会变成
        记忆。这里在插件加载后补一次。

        两个时序细节：
          - 加载时拿不到运行中的事件循环就先放弃，改由第一条消息补排；
          - 加载时给 20 秒延迟，零配置下模型 provider 是运行期惰性解析的，
            插件刚加载完就去要模型，很可能拿到空。
        """
        if not self.enabled or self._catch_up_done:
            return
        if self._catch_up_task is not None and not self._catch_up_task.done():
            return  # 已经排上了，不重复排
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.debug("RCMS: 当前没有事件循环，补蒸推迟到第一条消息")
            return
        task = loop.create_task(self._run_catch_up(delay, exclude_session_id))
        self._catch_up_task = task
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)

    async def _run_catch_up(self, delay: float, exclude_session_id: str) -> None:
        """后台任务：补蒸关程序前没来得及蒸的会话。只跑一次。"""
        if self._catch_up_done:
            return
        self._catch_up_done = True
        try:
            if delay > 0:
                await asyncio.sleep(delay)
            core = self._get_core()
            results = await core.distill_use_case.catch_up(exclude_session_id)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - 补蒸失败不该影响正常对话
            logger.warning("RCMS: 启动补蒸失败（不影响正常对话蒸馏）: %s", exc)
            return
        if results:
            logger.info(
                "RCMS 启动补蒸：补出 %d 条记忆 —— %s",
                len(results), "、".join(r.keylabel for r in results if r.keylabel),
            )
        else:
            logger.debug("RCMS 启动补蒸：没有欠账的会话")

    def _get_core(self) -> CoreContext:
        if self._core is None:
            data_dir = _resolve_data_dir(self._config)
            os.makedirs(data_dir, exist_ok=True)
            db_path = os.path.join(data_dir, "rcms_memory.db")
            self._db_path = db_path
            llm_callable, embed_callable = self._build_callbacks()
            trace_on = bool(self._webui_cfg.get("enabled", True)) and bool(
                self._webui_cfg.get("trace_enabled", True)
            )
            try:
                trace_max = int(self._webui_cfg.get("trace_max") or 200)
            except (TypeError, ValueError):
                trace_max = 200
            self._core = create_core(
                db_path=db_path,
                settings=self._build_settings(data_dir),
                llm_call=llm_callable,
                embed_call=embed_callable,
                trace_enabled=trace_on,
                trace_max=trace_max,
            )
        return self._core

    def _build_settings(self, data_dir: str):
        """把 AstrBot 面板里的插件配置映射成 RCMS 的 Settings。

        不这样做的话 create_core 会走 get_settings() 单例：它读不到面板配置
        （面板配置是内存里的 dict，磁盘上没有 config.json），于是 max_turns、
        召回上限、权重等全部退回内置默认值 —— 用户在面板改什么都不生效。

        Args:
            data_dir: 已解析出的记忆库目录。

        Returns:
            与面板配置一致的 Settings；解析失败时退回内置默认值。
        """
        from infrastructure.config.settings import Settings

        try:
            settings = Settings.load_from_dict(self._config)
        except Exception as exc:
            logger.warning("RCMS: 插件配置解析失败，使用内置默认值: %s", exc)
            return Settings()
        # 记忆库目录以 _resolve_data_dir 为准（面板里可能留空或写了相对路径）
        try:
            settings.storage.data_dir = data_dir
        except Exception:
            pass
        return settings

    async def terminate(self) -> None:
        """关闭 RCMS 核心，执行 WAL TRUNCATE 后释放资源。

        AstrBot 在卸载/重载插件时会 await 本方法，因此必须是协程。
        """
        # 观测站与后台任务都不依赖 core，先停掉再关库
        if self._webui is not None:
            try:
                self._webui.stop()
            except Exception as exc:  # noqa: BLE001
                logger.debug("关闭观测站失败: %s", exc)
            self._webui = None

        # 先收敛后台任务，否则它们会继续使用马上要被关闭的数据库连接
        if self._bg_tasks:
            pending = list(self._bg_tasks)
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            self._bg_tasks.clear()

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
        """构造 LLM / Embedding 回调。

        设计前提是零配置：默认 ``analysis.*.source = "astrbot"``，运行时向 AstrBot
        要「当前生效的 provider」，用户在面板里换模型插件就跟着换，不需要在插件
        里再填一遍 API Key。只有显式设置 ``source = "custom"`` 且填了 Key 时，才
        走插件内自建的 OpenAI 客户端。

        Returns:
            (llm_callable, embed_callable)，两者都是 async 回调；
            嵌入被关闭或不可用时 embed_callable 为 None。
        """
        analysis_cfg = self._config.get("analysis")
        analysis_cfg = analysis_cfg if isinstance(analysis_cfg, dict) else {}
        llm_cfg = analysis_cfg.get("post_analysis")
        llm_cfg = llm_cfg if isinstance(llm_cfg, dict) else {}
        emb_cfg = analysis_cfg.get("retrieval")
        emb_cfg = emb_cfg if isinstance(emb_cfg, dict) else {}

        # source=custom 但没填 Key 时静默回退即可，但要让用户知道为什么没生效
        for label, cfg in (("蒸馏 LLM", llm_cfg), ("Embedding", emb_cfg)):
            if str(cfg.get("source") or "").strip().lower() != "custom":
                continue
            if not str(
                cfg.get("api_key") or cfg.get("custom_api_key") or ""
            ).strip():
                logger.warning(
                    "RCMS: %s 的 source 设为 custom 但未填 API Key，"
                    "已自动回退到 AstrBot 当前模型",
                    label,
                )

        return self._make_llm_callable(llm_cfg), self._make_embed_callable(emb_cfg)

    # ── 模型自动解析 ────────────────────────────────────────────────

    @staticmethod
    def _clean(value: object) -> str:
        return str(value or "").strip()

    @staticmethod
    def _should_use_custom(cfg: dict) -> bool:
        """是否走「插件内自定义 API」分支。

        兼容老配置：source 缺省但填了 Key 时，仍按 custom 处理，避免升级后
        原本能用的配置突然失效。
        """
        has_key = bool(
            str(cfg.get("api_key") or cfg.get("custom_api_key") or "").strip()
        )
        if not has_key:
            return False
        source = str(cfg.get("source") or "").strip().lower()
        return source == "custom" or not source

    @staticmethod
    def _provider_label(provider: object) -> str:
        """拼一个可读的 provider 标识，用于日志。"""
        cfg = getattr(provider, "provider_config", None)
        cfg = cfg if isinstance(cfg, dict) else {}
        provider_id = str(cfg.get("id") or "").strip()
        model = ""
        get_model = getattr(provider, "get_model", None)
        if callable(get_model):
            try:
                model = str(get_model() or "").strip()
            except Exception:
                model = ""
        if not model:
            model = str(
                cfg.get("embedding_model") or cfg.get("model") or ""
            ).strip()
        if provider_id and model:
            return f"{provider_id} / {model}"
        return provider_id or model or type(provider).__name__

    def _notice_once(self, key: str, message: str, *args: object) -> None:
        """同一条模型信息只提示一次，避免每轮对话刷屏。"""
        if key in self._noticed:
            return
        self._noticed.add(key)
        logger.info(message, *args)

    async def _resolve_chat_provider(self, provider_id: str = ""):
        """取当前用于对话生成的 provider。

        优先使用配置指定的 provider_id；没指定或找不到时，回退到 AstrBot 的
        「当前会话生效 provider」——也就是面板里显示正在用的那个模型。
        """
        ctx = getattr(self, "context", None)
        if ctx is None:
            return None
        pid = self._clean(provider_id)
        if pid:
            provider = ctx.get_provider_by_id(pid)
            if provider is not None:
                return provider
            logger.warning(
                "RCMS: 配置的对话提供商「%s」不存在，回退到 AstrBot 当前默认提供商", pid
            )
        try:
            return await ctx.get_using_provider_async(self._current_umo)
        except Exception as exc:
            logger.warning("RCMS: 获取 AstrBot 当前对话提供商失败: %s", exc)
            return None

    def _resolve_embedding_provider(self, provider_id: str = ""):
        """取一个可用的 Embedding provider。

        顺序：配置指定 → AstrBot 已加载的 Embedding 提供商列表。
        AstrBot 没有独立的 Embedding 提供商时会返回 None，调用方降级处理。
        """
        ctx = getattr(self, "context", None)
        if ctx is None:
            return None
        pid = self._clean(provider_id)
        if pid:
            provider = ctx.get_provider_by_id(pid)
            if provider is not None and hasattr(provider, "get_embedding"):
                return provider
            logger.warning(
                "RCMS: 配置的 Embedding 提供商「%s」不存在或不是 Embedding 类型", pid
            )
        try:
            providers = ctx.get_all_embedding_providers() or []
        except Exception as exc:
            logger.warning("RCMS: 读取 AstrBot Embedding 提供商列表失败: %s", exc)
            return None
        return providers[0] if providers else None

    # ── 回调构造 ────────────────────────────────────────────────────

    def _make_llm_callable(self, cfg: dict):
        if self._should_use_custom(cfg):
            from openai import AsyncOpenAI

            client = AsyncOpenAI(
                api_key=cfg.get("api_key") or cfg.get("custom_api_key"),
                base_url=self._clean(cfg.get("base_url"))
                or self._clean(cfg.get("custom_base_url"))
                or "https://api.openai.com/v1",
            )
            model_name = (
                self._clean(cfg.get("model"))
                or self._clean(cfg.get("custom_model"))
                or "gpt-4o-mini"
            )

            async def _llm_custom(prompt: str) -> str:
                resp = await client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                )
                return resp.choices[0].message.content or ""

            logger.info("RCMS: 蒸馏模型使用插件内自定义配置（%s）", model_name)
            return _llm_custom

        async def _llm_from_astrbot(prompt: str) -> str:
            provider = await self._resolve_chat_provider(
                cfg.get("astrbot_source_id")
            )
            if provider is None:
                raise RuntimeError(
                    "AstrBot 当前没有可用的对话模型，请先在面板的「服务提供商」里添加"
                )
            self._notice_once(
                "llm",
                "RCMS: 自动使用 AstrBot 当前对话模型 %s",
                self._provider_label(provider),
            )
            resp = await asyncio.wait_for(
                provider.text_chat(prompt=prompt), timeout=_LLM_TIMEOUT_SECONDS
            )
            return (resp.completion_text or "") if resp is not None else ""

        return _llm_from_astrbot

    def _make_embed_callable(self, cfg: dict):
        if not cfg.get("embedding_enabled", True):
            logger.info("RCMS: 已关闭向量检索，语义通道停用（时间/图谱通道正常）")
            return None

        if self._should_use_custom(cfg):
            from openai import AsyncOpenAI

            client = AsyncOpenAI(
                api_key=cfg.get("api_key") or cfg.get("custom_api_key"),
                base_url=self._clean(cfg.get("base_url"))
                or self._clean(cfg.get("custom_base_url"))
                or "https://api.openai.com/v1",
            )
            model_name = (
                self._clean(cfg.get("model"))
                or self._clean(cfg.get("custom_model"))
                or "text-embedding-3-small"
            )

            async def _embed_custom(text: str) -> list[float]:
                resp = await client.embeddings.create(
                    model=model_name,
                    input=text.replace("\n", " "),
                )
                return resp.data[0].embedding

            logger.info("RCMS: Embedding 使用插件内自定义配置（%s）", model_name)
            return _embed_custom

        async def _embed_from_astrbot(text: str) -> list[float]:
            # 冷却期内直接返回空向量：语义通道跳过，其余通道照常工作，
            # 避免上游模型不可用时每轮都白等一次超时。
            if time.monotonic() < self._embed_fail_until:
                return []
            provider = self._resolve_embedding_provider(
                cfg.get("astrbot_source_id")
            )
            if provider is None:
                self._embed_fail_until = (
                    time.monotonic() + _EMBED_FAIL_COOLDOWN_SECONDS
                )
                self._notice_once(
                    "embed_missing",
                    "RCMS: AstrBot 未配置 Embedding 提供商，语义通道停用"
                    "（时间/图谱通道正常工作）。需要向量检索时请到面板添加 "
                    "Embedding 提供商。",
                )
                return []
            try:
                vec = await asyncio.wait_for(
                    provider.get_embedding(text.replace("\n", " ")),
                    timeout=_EMBED_TIMEOUT_SECONDS,
                )
            except asyncio.TimeoutError:
                self._embed_fail_until = (
                    time.monotonic() + _EMBED_FAIL_COOLDOWN_SECONDS
                )
                logger.warning(
                    "RCMS: Embedding 提供商 %s 超过 %ss 未返回，%d 秒内只走时间/图谱通道",
                    self._provider_label(provider),
                    _EMBED_TIMEOUT_SECONDS,
                    _EMBED_FAIL_COOLDOWN_SECONDS,
                )
                return []
            except Exception as exc:
                self._embed_fail_until = (
                    time.monotonic() + _EMBED_FAIL_COOLDOWN_SECONDS
                )
                logger.warning(
                    "RCMS: Embedding 提供商 %s 调用失败，%d 秒内只走时间/图谱通道: %s",
                    self._provider_label(provider),
                    _EMBED_FAIL_COOLDOWN_SECONDS,
                    exc,
                )
                return []
            self._embed_fail_until = 0.0
            self._notice_once(
                "embed",
                "RCMS: 自动使用 AstrBot 当前 Embedding 模型 %s（dim=%d）",
                self._provider_label(provider),
                len(vec),
            )
            return vec

        return _embed_from_astrbot

    @filter.on_llm_request()
    async def on_llm_request(self, event: AstrMessageEvent, req: ProviderRequest) -> None:
        if not self.enabled:
            return
        # 记录当前会话来源：模型解析用它去取「本会话生效」的 provider，
        # 这样面板里开启会话级模型隔离时也能自动跟随。
        self._current_umo = (
            getattr(event, "unified_msg_origin", None) or self._current_umo
        )
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
        uid = event.get_sender_id() or self.user_id
        sid = uid
        sender_name = event.get_sender_name() or uid

        # 落库与蒸馏放到后台执行。TurnSaved 事件会串起「后处理 + 蒸馏」，
        # 而蒸馏本身要调一次 LLM（秒级）；同步 await 会把每轮回复硬生生拖慢
        # 好几秒，用户感知就是「插件装了以后机器人变卡」。
        task = asyncio.create_task(
            self._persist_turn(sid, uid, sender_name, user_input, reply)
        )
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)

    async def _persist_turn(
        self,
        sid: str,
        uid: str,
        sender_name: str,
        user_input: str,
        reply: str,
    ) -> None:
        """后台任务：写入本轮对话并触发蒸馏链路。"""
        try:
            core = self._get_core()
            from application.use_cases.chat_use_case import ChatRequest

            request = ChatRequest(
                user_id=uid,
                session_id=sid,
                user_input=user_input,
                sender_name=sender_name,
            )
            await core.chat_use_case.save_turn_only(request, reply=reply)
            logger.info(f"RCMS: [{sid}] saved turn")
            self._write_output_log(sid, uid, user_input, reply)
            # 加载时没排上补蒸（拿不到事件循环）就用第一条消息补排；
            # 当前会话排除掉，它走正常的每轮蒸馏链路。
            self._ensure_catch_up(exclude_session_id=sid)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error("RCMS: 保存本轮对话失败: %s", exc, exc_info=True)

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
                # 相对路径一律落到记忆库目录。插件目录在 WebUI 更新/重装时会被整体
                # 删除重建，写在里面的文件必然丢失；v1 时代写在插件目录的 rcms.log
                # 还曾长期占用目录句柄，导致卸载直接报 WinError 32。
                log_dir = _resolve_data_dir(self._config)
                os.makedirs(log_dir, exist_ok=True)
                log_path = os.path.join(log_dir, log_path)
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

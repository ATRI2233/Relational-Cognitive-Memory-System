"""
RcmsFactory — 依赖注入工厂。

组装整个 RCMS 新架构的对象图。
只有一个入口函数 create_core() 返回 CoreContext 包含所有 Use Case。

用法::

    from adapter.rcms_factory import create_core

    core = create_core("memory.db", llm_call=my_llm, embed_call=my_embed)
    response = await core.chat_use_case.execute(
        ChatRequest(user_id="u1", session_id="s1", user_input="你好")
    )
"""
from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
from collections.abc import Awaitable, Callable
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import jieba

from domain.entities.memory import SessionId, UserId
from domain.entities.session import TurnRecord as _TurnRec
from domain.events.memory_events import MemoryDistilled, TurnSaved
from domain.ports.clock import IClock
from domain.ports.graph_repo import IGraphRepository
from domain.ports.identity_repo import IIdentityRepository
from domain.ports.repositories import IMemoryRepository, ISessionRepository, IUserMappingRepository

from domain.services.fusion_service import FusionService
from domain.services.time_service import TimeService
from domain.services.keyword_service import KeywordService

from application.event_bus import EventBus
from application.post_update_service import PostUpdateService
from application.analysis_writer import AnalysisWriter
from application.prompt_builder import PromptBuilder

from application.use_cases.chat_use_case import ChatUseCase
from application.use_cases.distill_use_case import DistillUseCase
from application.use_cases.retrieve_context_use_case import (
    RetrieveContextUseCase,
)

from application.handlers.post_update_handler import PostUpdateHandler
from application.handlers.delayed_embed_handler import DelayedEmbedHandler
from application.handlers.distill_checker_handler import DistillCheckerHandler
from application.handlers.graph_maintenance_handler import (
    GraphMaintenanceHandler,
)

from infrastructure.clock import SystemClock
from infrastructure.config.settings import Settings, get_settings
from infrastructure.persistence.ddl import ensure_schema
from infrastructure.persistence.sqlite_graph_repo import SQLiteGraphRepository
from infrastructure.persistence.sqlite_identity_repo import (
    SQLiteIdentityRepository,
)
from infrastructure.persistence.sqlite_memory_repo import SQLiteMemoryRepository
from infrastructure.persistence.sqlite_session_repo import SQLiteSessionRepository
from infrastructure.persistence.sqlite_shared_context_repo import (
    SQLiteSharedContextRepository,
)
from infrastructure.persistence.sqlite_user_mapping_repo import (
    SQLiteUserMappingRepository,
)

logger = logging.getLogger("rcms")

# ═══════════════════════════════════════════════════════════════════════
# 分词器回调
# ═══════════════════════════════════════════════════════════════════════


def _tokenize(text: str) -> list[str]:
    """使用 jieba 的中文分词回调。"""
    return jieba.lcut(text)


# ═══════════════════════════════════════════════════════════════════════
# 内部适配器（接口桥接）
# ═══════════════════════════════════════════════════════════════════════


class _LLMServiceAdapter:
    """将 llm_call 回调适配为 ILLMService Protocol。

    ChatUseCase.ILLMService 和 DistillUseCase.ILLMService 均接受此包装。
    """

    def __init__(self, llm_call: Callable[[str], Awaitable[str]]):
        self._llm_call = llm_call

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        return await self._llm_call(prompt)


class _EmbeddingServiceAdapter:
    """将 embed_call 回调适配为 IEmbeddingService Protocol。

    RetrieveContextUseCase.IEmbeddingService 和 DistillUseCase.IEmbeddingService
    均接受此包装。
    """

    def __init__(self, embed_call: Callable[[str], Awaitable[list[float]]]):
        self._embed_call = embed_call

    async def embed(self, text: str) -> list[float]:
        return await self._embed_call(text)


class _RetrievalStrategyAdapter:
    """适配 RetrieveContextUseCase.retrieve_memories → ChatUseCase.IRetrievalStrategy。

    ChatUseCase 的 IRetrievalStrategy.retrieve() 返回
    (memories, graph_paths) 二元组。
    RetrieveContextUseCase.retrieve_memories 现已直接返回
    (memories, graph_paths) 二元组。
    """

    def __init__(self, retrieve_uc: RetrieveContextUseCase):
        self._uc = retrieve_uc

    async def retrieve(
        self,
        user_id: str,
        user_input: str,
        session_id: str,
    ) -> tuple[list[tuple[str, str]], list[str]]:
        return await self._uc.retrieve_memories(
            user_id, user_input, session_id=session_id
        )


class _PromptBuilderAdapter:
    """适配 PromptBuilder → ChatUseCase.IPromptBuilder。

    将 PromptBuilder.build_compressed_prompt（同步）包装为
    IPromptBuilder.build（异步）。
    同时包装 PromptBuilder 为 RetrieveContextUseCase 的 ITextTemplateProvider。
    """

    def __init__(self, pb: PromptBuilder):
        self._pb = pb

    # ── IPromptBuilder ──

    async def build(
        self,
        user_id: str,
        session_id: str,
        user_input: str,
        memories: list[tuple[str, str]],
        long_term: dict[str, Any],
        graph_paths: list[str],
    ) -> str:
        return self._pb.build_compressed_prompt(
            user_input=user_input,
            memories=memories,
            long_term=long_term,
            graph_paths=graph_paths,
        )

    # ── ITextTemplateProvider (for RetrieveContextUseCase) ──

    def narrative_templates(self) -> dict:
        return self._pb._templates.get("narrative_context", {})

    def channel_labels(self) -> dict:
        return self._pb._templates.get("channel_labels", self._pb._channel_labels)

    def memories_display_order(self) -> list:
        return self._pb._templates.get(
            "memories_display_order", ["resonance", "skeleton", "recent"]
        )

    def prompt_compressor_templates(self) -> dict:
        return self._pb._templates.get("prompt_compressor", {})


class _DistillRunnerAdapter:
    """适配 DistillUseCase.check_and_run → ChatUseCase.IDistillRunner。

    IDistillRunner.check_and_run(user_id, session_id, turn_count)
    忽略 turn_count 参数（DistillUseCase 不再接收此参数）。
    """

    def __init__(self, distill_uc: DistillUseCase):
        self._uc = distill_uc

    async def check_and_run(
        self,
        user_id: str,
        session_id: str,
        turn_count: int,
    ) -> None:
        await self._uc.check_and_run(user_id, session_id)


class _KeywordConfigAdapter:
    """组合 EmotionalWordsSettings + TimeWordSettings → IWordListConfig。

    KeywordService 需要的 IWordListConfig 包含 trivial_markers、stop_words
    和 time_words 三个属性，分属 Settings 的两个子配置。
    """

    def __init__(
        self,
        trivial_markers: list[str],
        stop_words: list[str],
        time_words: dict[str, list[int]],
    ):
        self.trivial_markers = trivial_markers
        self.stop_words = stop_words
        self.time_words = time_words


class _ChatHistoryProvider:
    """默认 chat_history_provider，直接查询 sqlite3 chat_history 表。

    供 DistillUseCase 构建对话快照使用。
    签名匹配 Callable[[SessionId, int, int], Awaitable[list[TurnRecord]]]。
    """

    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    async def __call__(
        self,
        session_id: SessionId,
        last_turn: int,
        turn_count: int,
    ) -> list:
        rows = self._conn.execute(
            "SELECT session_id, role, content, turn_num, user_id, "
            "sender_name, importance "
            "FROM chat_history "
            "WHERE session_id = ? AND turn_num > ? AND turn_num <= ? "
            "ORDER BY turn_num ASC, role ASC",
            (session_id.value, last_turn, turn_count),
        ).fetchall()

        records: list = []
        for r in rows:
            records.append(
                _TurnRec(
                    session_id=SessionId(r[0]),
                    role=r[1],
                    content=r[2] or "",
                    turn_num=r[3],
                    user_id=str(r[4] or ""),
                    sender_name=str(r[5] or ""),
                    importance=float(r[6] or 0.3),
                )
            )
        return records


class _StubSharedContextRepo:
    """ISharedContextRepository 的桩实现。

    未提供真正实现时返回空列表，避免 RetrieveContextUseCase 构造失败。
    """

    async def get_recent(self, user_id: str, limit: int = 4) -> list[str]:
        return []


class _StubUserMappingRepo:
    """IUserMappingRepository 的桩实现。

    未提供真正实现时返回空结果。
    """

    async def find_mentioned(
        self, session_id: str, text: str, speaker_id: str = ""
    ) -> list[tuple[str, str]]:
        return []

    async def get_labels(self, session_id: str, user_id: str) -> list[str]:
        return []


# ═══════════════════════════════════════════════════════════════════════
# CoreContext — 组装好的全部 Use Case 和辅助对象
# ═══════════════════════════════════════════════════════════════════════


class CoreContext:
    """核心上下文 — 组装好的全部 Use Case 和辅助对象。

    Attributes:
        chat_use_case: 对话编排 Use Case
        distill_use_case: 蒸馏分析 Use Case
        retrieve_context_use_case: 上下文检索 Use Case
        event_bus: 事件总线（已注册所有 Handler）
        prompt_builder: Prompt 模板构建器
        settings: 全局配置
        conn: SQLite 数据库连接（由 create_core 创建，调用者负责关闭）
    """

    def __init__(
        self,
        chat_uc: ChatUseCase,
        distill_uc: DistillUseCase,
        retrieve_uc: RetrieveContextUseCase,
        event_bus: EventBus,
        prompt_builder: PromptBuilder,
        settings: Settings,
        conn: sqlite3.Connection,
    ) -> None:
        self.chat_use_case = chat_uc
        self.distill_use_case = distill_uc
        self.retrieve_context_use_case = retrieve_uc
        self.event_bus = event_bus
        self.prompt_builder = prompt_builder
        self.settings = settings
        self.conn = conn

    def close(self) -> None:
        """关闭数据库连接和所有资源。

        调用此方法后不应再使用本 CoreContext 或其包含的任何 Use Case。
        可安全重复调用。
        """
        if self.conn is not None:
            try:
                self.conn.close()
            except Exception:
                pass
            self.conn = None


# ═══════════════════════════════════════════════════════════════════════
# 工厂入口
# ═══════════════════════════════════════════════════════════════════════


def create_core(
    db_path: str,
    settings: Optional[Settings] = None,
    llm_call: Optional[Callable[[str], Awaitable[str]]] = None,
    embed_call: Optional[Callable[[str], Awaitable[list[float]]]] = None,
    shared_context_repo: Any = None,
    user_mapping_repo: Any = None,
    session_query_repo: Any = None,
    persona_name: str = "Bot",
    save_shared_context_fn: Optional[
        Callable[[str, str, str], Awaitable[None]]
    ] = None,
    trim_low_importance_fn: Optional[
        Callable[..., Awaitable[int]]
    ] = None,
) -> CoreContext:
    """创建完整 RCMS 核心对象图。

    Args:
        db_path: SQLite 数据库文件路径。
        settings: 全局配置对象（可选，默认使用 get_settings() 单例）。
        llm_call: LLM 生成回调，接受 prompt 返回回复文本。
                  可选，不提供时 Use Case 运行时会失败。
        embed_call: Embedding 回调，接受文本返回浮点向量。
                    可选，不提供时向量检索功能降级。
        shared_context_repo: ISharedContextRepository 实现。
                              可选，不提供时使用桩实现（返回空列表）。
        user_mapping_repo: IUserMappingRepository 实现。
                           可选，不提供时使用桩实现。
        session_query_repo: ISessionQueryRepository 实现。
                            可选，不提供时使用桩实现（跳过 session_warmup）。
        persona_name: AI 角色的名称（默认 "Bot"）。
        save_shared_context_fn: 可选回调，用于 shared_context 的 upsert。
        trim_low_importance_fn: 可选回调，用于低重要性记忆截断。

    Returns:
        CoreContext 包含所有 Use Case 和辅助对象。

    Raises:
        sqlite3.Error: 数据库连接失败。
        FileNotFoundError: prompts.json 文件未找到。
        ValueError: prompts.json 内容为空或缺少必需模板段。
    """
    # 向后兼容: 自动探测 config.json 加载
    _cfg_json_path = Path(db_path).parent / "config.json"
    if not _cfg_json_path.exists():
        _cfg_json_path = Path("config.json")
    config = settings or get_settings(
        config_path=str(_cfg_json_path) if _cfg_json_path.exists() else None
    )
    clock: IClock = SystemClock()

    # ── 数据库连接 ──────────────────────────────────────────────────
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(f"PRAGMA busy_timeout={config.storage.busy_timeout_ms}")
        conn.execute(
            f"PRAGMA wal_autocheckpoint={config.storage.wal_autocheckpoint_pages}"
        )
        # 确保所有表、索引、迁移已执行
        ensure_schema(conn, config.storage)
    except sqlite3.Error as exc:
        logger.error("数据库连接失败: %s", exc)
        raise

    # ── Repositories ───────────────────────────────────────────────
    memory_repo: IMemoryRepository = SQLiteMemoryRepository(conn, clock)
    session_repo: ISessionRepository = SQLiteSessionRepository(conn, clock)
    identity_repo: IIdentityRepository = SQLiteIdentityRepository(conn, clock)
    graph_repo: IGraphRepository = SQLiteGraphRepository(conn, clock)

    # ── Domain Services ────────────────────────────────────────────
    fusion_svc = FusionService(config.retrieval)
    time_svc = TimeService(clock, config.retrieval.time_decay_halflife)

    kw_config = _KeywordConfigAdapter(
        trivial_markers=config.emotional_words.trivial_markers,
        stop_words=config.emotional_words.stop_words,
        time_words=config.time_words.time_words,
    )
    kw_svc = KeywordService(tokenizer=_tokenize, config=kw_config)

    # ── Event Bus ──────────────────────────────────────────────────
    event_bus = EventBus()

    # ── PromptBuilder（加载 prompts.json）───────────────────────────
    prompts_path = (
        Path(__file__).resolve().parent.parent / "infrastructure" / "config" / "prompts.json"
    )
    try:
        with open(prompts_path, encoding="utf-8") as f:
            prompts_templates = json.load(f)
        if not prompts_templates:
            raise ValueError("prompts.json 内容为空")
    except FileNotFoundError:
        logger.error("prompts.json 未找到: %s", prompts_path)
        raise
    except json.JSONDecodeError as exc:
        logger.error("prompts.json 解析失败: %s", exc)
        raise

    prompt_builder = PromptBuilder(prompts_templates)

    # ── 模板提供者适配器（包装 PromptBuilder）───────────────────────
    template_provider = _PromptBuilderAdapter(prompt_builder)

    # ── LLM / Embedding 适配器 ────────────────────────────────────
    llm_adapter = _LLMServiceAdapter(llm_call) if llm_call else None
    embed_adapter = (
        _EmbeddingServiceAdapter(embed_call) if embed_call else None
    )

    # ── PostUpdateService ─────────────────────────────────────────
    post_update_svc = PostUpdateService(
        session_repo=session_repo,
        identity_repo=identity_repo,
        memory_repo=memory_repo,
        clock=clock,
        dangling_expire_turns=config.analysis.dangling_expire_turns,
    )

    # ── AnalysisWriter ────────────────────────────────────────────
    analysis_writer = AnalysisWriter(
        memory_repo=memory_repo,
        session_repo=session_repo,
        identity_repo=identity_repo,
        graph_repo=graph_repo,
        clock=clock,
        settings=config,
        upsert_shared_context=save_shared_context_fn,
    )

    # ── 可选仓储的桩 / 注入 ───────────────────────────────────────
    _shared_context_repo = shared_context_repo or SQLiteSharedContextRepository(conn, clock)
    _user_mapping_repo = user_mapping_repo or SQLiteUserMappingRepository(conn, clock)
    # SQLiteSessionRepository 同时提供 ISessionQueryRepository 的 session_warmup 方法
    _session_query_repo = session_query_repo or session_repo

    # ── chat_history_provider（用于 DistillUseCase 快照构建）───────
    _chat_history_provider = _ChatHistoryProvider(conn)

    # ── analysis_raw 审计日志回调（Task 2）─────────────────────────
    async def _save_raw_analysis_fn(user_id: str, session_id: str, content: str) -> int:
        cur = await asyncio.to_thread(
            conn.execute,
            "INSERT INTO analysis_raw (user_id, session_id, content) VALUES (?, ?, ?)",
            (user_id, session_id, content),
        )
        raw_id = cur.lastrowid
        await asyncio.to_thread(conn.commit)
        return raw_id

    async def _mark_raw_parsed_fn(raw_id: int) -> None:
        await asyncio.to_thread(
            conn.execute,
            "UPDATE analysis_raw SET parsed = 1 WHERE id = ?",
            (raw_id,),
        )
        await asyncio.to_thread(conn.commit)

    # ── 默认低重要性记忆清理 ──
    if trim_low_importance_fn is None:
        _low_imp_threshold = config.maintenance.cleanup_importance_threshold

        async def _default_trim_low_importance(user_id: UserId, keep_count: int) -> int:
            cur = conn.execute(
                "DELETE FROM cognitive_distill "
                "WHERE user_id = ? AND importance = ? "
                "AND id NOT IN ("
                "  SELECT id FROM cognitive_distill "
                "  WHERE user_id = ? AND importance = ? "
                "  ORDER BY created_at DESC LIMIT ?"
                ")",
                (user_id.value, _low_imp_threshold, user_id.value, _low_imp_threshold, keep_count),
            )
            conn.commit()
            deleted = cur.rowcount
            if deleted > 0:
                logger.info("低重要性记忆已清理 user=%s count=%d", user_id.value, deleted)
            return deleted
        trim_low_importance_fn = _default_trim_low_importance

    # ── DistillUseCase ────────────────────────────────────────────
    distill_uc = DistillUseCase(
        session_repo=session_repo,
        memory_repo=memory_repo,
        identity_repo=identity_repo,
        graph_repo=graph_repo,
        llm=llm_adapter,
        embedding=embed_adapter,
        clock=clock,
        event_bus=event_bus,
        settings=config.analysis,
        prompt_builder=None,  # 使用 DefaultDistillPromptBuilder（内置）
        persona_name=persona_name,
        chat_history_provider=_chat_history_provider,
        save_shared_context_fn=save_shared_context_fn,
        trim_low_importance_fn=trim_low_importance_fn,
        # Task 2: analysis_raw 审计日志
        save_raw_analysis_fn=_save_raw_analysis_fn,
        mark_raw_parsed_fn=_mark_raw_parsed_fn,
        # Task 3: 注入 AnalysisWriter
        analysis_writer=analysis_writer,
    )

    # ── RetrieveContextUseCase ────────────────────────────────────
    retrieve_uc = RetrieveContextUseCase(
        memory_repo=memory_repo,
        identity_repo=identity_repo,
        graph_repo=graph_repo,
        session_repo=session_repo,
        shared_context_repo=_shared_context_repo,
        user_mapping_repo=_user_mapping_repo,
        fusion_service=fusion_svc,
        time_service=time_svc,
        keyword_service=kw_svc,
        embed_service=embed_adapter,
        clock=clock,
        config=config.retrieval,
        session_query_repo=_session_query_repo,
        template_provider=template_provider,
    )

    # ── 适配器: RetrieveContextUseCase → IRetrievalStrategy ──────
    retrieval_adapter = _RetrievalStrategyAdapter(retrieve_uc)

    # ── 适配器: DistillUseCase → IDistillRunner ──────────────────
    distill_runner_adapter = _DistillRunnerAdapter(distill_uc)

    # ── ChatUseCase ───────────────────────────────────────────────
    chat_uc = ChatUseCase(
        memory_repo=memory_repo,
        session_repo=session_repo,
        identity_repo=identity_repo,
        graph_repo=graph_repo,
        llm=llm_adapter,
        clock=clock,
        event_bus=event_bus,
        settings=config,
        retrieval_strategy=retrieval_adapter,
        prompt_builder=template_provider,
        distill_runner=distill_runner_adapter,
        shared_context_repo=_shared_context_repo,
        user_mapping_repo=_user_mapping_repo,
    )

    # ── 注册 Event Handler ────────────────────────────────────────
    post_update_handler = PostUpdateHandler(post_update_svc)
    event_bus.register(TurnSaved, post_update_handler.handle)

    distill_checker = DistillCheckerHandler(distill_uc)
    event_bus.register(TurnSaved, distill_checker.handle)

    graph_maintenance = GraphMaintenanceHandler(graph_repo)
    event_bus.register(MemoryDistilled, graph_maintenance.handle)

    # 当提供了 embed_call 时注册 DelayedEmbedHandler
    if embed_adapter is not None:
        delayed_embed_handler = DelayedEmbedHandler(
            memory_repo=memory_repo,
            embedding=embed_adapter,
            event_bus=event_bus,
        )
        event_bus.register(TurnSaved, delayed_embed_handler.handle)

    # ── 返回 CoreContext ──────────────────────────────────────────
    return CoreContext(
        chat_uc=chat_uc,
        distill_uc=distill_uc,
        retrieve_uc=retrieve_uc,
        event_bus=event_bus,
        prompt_builder=prompt_builder,
        settings=config,
        conn=conn,
    )

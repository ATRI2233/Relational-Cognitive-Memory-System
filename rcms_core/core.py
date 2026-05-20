import logging
import os
import sqlite3
import threading
from datetime import datetime
from typing import Optional

import numpy as np
from openai import AsyncOpenAI

from backends import LLMBackend
from .utils import UtilsMixin
from .db import DBMixin
from .retrieval import RetrievalMixin
from .context import ContextMixin
from .session import SessionMixin
from .memory import MemoryMixin
from .analysis import AnalysisMixin

logger = logging.getLogger("rcms")


class MinimalRCMS(
    DBMixin,
    UtilsMixin,
    RetrievalMixin,
    ContextMixin,
    SessionMixin,
    MemoryMixin,
    AnalysisMixin,
):
    """MinimalRCMS — 关系认知记忆系统核心"""

    # ── 常量（可被 post_analysis 配置覆盖） ──
    _EMOTIONAL_WORDS = [
        '累', '烦', '难过', '开心', '怕', '为什么', '怎么办',
        '焦虑', '迷茫', '失望', '生气', '感动', '孤独', '压力',
        '崩溃', '痛苦', '幸福', '委屈', '愤怒', '绝望', '不安',
        '愧疚', '后悔', '感激', '羡慕', '厌倦', '疲惫', '心累',
        '纠结', '无助', '温暖', '讽刺', '荒谬', '崩溃', '心碎',
        '气死', '受不了', '撑不住', '扛不住', '熬不下去',
        '舍不得', '放不下', '不甘心',
    ]

    _TRIVIAL_MARKERS = ['吃', '喝', '睡', '饭', '菜', '外卖', '快递', '天气',
                        '价格', '多少钱', '购物', '买了', '电影', '追剧',
                        '洗澡', '起床', '睡觉', '游戏']

    _STOP_WORDS = [
        # 时间副词（不具备关键词区分度）
        '最近', '今天', '明天', '昨天', '前天', '刚才', '已经', '正在',
        '将要', '即将', '马上', '立刻', '刚刚', '忽然', '曾经', '往往',
        # 代词/指示词
        '什么', '怎么', '为什么', '哪个', '哪些', '谁', '这个', '那个',
        '这些', '那些', '哪里', '这里', '那里', '如何', '何时',
        # 虚词
        '一个', '没有', '不是', '可以', '就是', '还是', '但是', '而且',
        '因为', '所以', '虽然', '如果', '然后', '不过', '一定',
        '一些', '有点', '一下', '非常',
        '是否', '能够', '应该', '必须', '好像', '真是',
        # 语气词
        '好吧', '好了', '是的', '没错', '对了',
        # 纯情绪感知词
        '觉得', '感觉', '认为',
    ]

    _DISTILL_MAX_TURNS = 30
    _DISTILL_MAX_MINUTES = 60
    _DANGLING_EXPIRE_TURNS = 15
    _GRAPH_BFS_DEPTH = 2
    _GRAPH_ACTIVATION_DECAY = 0.5
    _SURFACED_THRESHOLD = 0.6
    _SILENT_THRESHOLD = 0.25

    # ── 初始化 ──

    def __init__(self, db_path="memory.db", analysis_config: dict | None = None,
                 llm_call=None, embed_call=None):
        self.db_path = db_path
        self.analysis_config = analysis_config or {}
        self._llm_call = llm_call
        self._embed_call = embed_call
        # 单一连接，但启用 WAL 与 busy timeout 以降低锁冲突概率
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        try:
            self.conn.execute("PRAGMA journal_mode=WAL;")
            self.conn.execute("PRAGMA busy_timeout = 5000")
        except Exception:
            # 在极少数 sqlite 构建中 PRAGMA 可能不可用，忽略不阻塞初始化
            pass
        # 进程内用于序列化写操作的简单互斥锁（短期阻塞）
        self._db_lock = threading.RLock()
        self._init_db()
        # Embedding 缓存
        self._emb_cache: dict[str, dict] = {}
        self._emb_client: Optional[AsyncOpenAI] = None
        rc = self.analysis_config.get("retrieval", {})
        self._emb_model = rc.get("custom_model", "") or rc.get("model", "text-embedding-3-small")
        # 从 post_analysis 配置覆盖蒸馏常量
        pa = self.analysis_config.get("post_analysis", {})
        if pa.get("max_turns"): self._DISTILL_MAX_TURNS = pa["max_turns"]
        if pa.get("max_minutes"): self._DISTILL_MAX_MINUTES = pa["max_minutes"]
        if pa.get("dangling_expire_turns"): self._DANGLING_EXPIRE_TURNS = pa["dangling_expire_turns"]
        logger.info(f"RCMS init: db={db_path} distill_turns={self._DISTILL_MAX_TURNS}min={self._DISTILL_MAX_MINUTES}")

    def close(self):
        self.conn.close()

    # ── Chat（standalone） ──

    async def chat(self, user_id: str, session_id: str, user_input: str, backend: LLMBackend) -> str:
        memories = await self.retrieve_memories(user_id, user_input, 'engaged', session_id=session_id)
        long_term = self._load_long_term_context(user_id)
        prompt = await self.prompt_compressor(user_id, session_id, user_input, memories, long_term)
        prompt = self._core_veto(prompt)
        try:
            reply = await backend.generate(prompt)
        except Exception:
            try:
                reply = await backend.generate("简短回复，一句话以内。\n\n你:")
            except Exception:
                logger.exception(f"RCMS: chat() backend 两次调用均失败 user={user_id}")
                reply = "嗯。"
        self.save_turn(session_id, user_input, reply, user_id=user_id)
        await self.post_update_rules(user_id, session_id, user_input, 'open', reply)
        # 蒸馏检查（standalone 模式下同步等待）
        triggered, last_turn, turn_count, snapshot, senders = self.check_distill_needed(session_id)
        if triggered:
            long_term = self._load_long_term_context(user_id)
            await self._run_distill_analysis(user_id, session_id, snapshot, long_term, last_turn, turn_count, senders=senders)
        return reply

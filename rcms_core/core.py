import logging
import os
import sqlite3
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

    # ── 常量 ──
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

    _ARC_STAGES = ['stranger', 'familiar', 'rapport', 'history', 'drift', 'reconnect']
    _RESIDUE_DECAY = 0.6
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
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_db()
        self._last_silent_recall = []
        # Embedding 缓存
        self._emb_cache: dict[str, dict] = {}
        self._emb_client: Optional[AsyncOpenAI] = None
        rc = self.analysis_config.get("retrieval", {})
        self._emb_model = rc.get("custom_model", "") or rc.get("model", "text-embedding-3-small")
        logger.info(f"RCMS init: db={db_path}, retrieval={self.analysis_config.get('retrieval', {}).get('enabled', False)}, post_analysis={self.analysis_config.get('post_analysis', {}).get('mode', 'rule')}")

    def close(self):
        self.conn.close()

    # ── Chat（standalone） ──

    async def chat(self, user_id: str, session_id: str, user_input: str, backend: LLMBackend) -> str:
        memories = await self.retrieve_memories(user_id, user_input, 'engaged')
        long_term = self._load_long_term_context(user_id)
        prompt = await self.prompt_compressor(user_id, session_id, user_input, memories, long_term)
        prompt = self._core_veto(prompt)
        try:
            reply = await backend.generate(prompt)
        except Exception:
            try:
                reply = await backend.generate("简短回复，一句话以内。\n\n你:")
            except Exception:
                reply = "嗯。"
        self.save_turn(session_id, user_input, reply, 'open')
        self._post_update(user_id, session_id, user_input, 'open', reply)
        return reply

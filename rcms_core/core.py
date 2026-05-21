import json
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
from .analysis import AnalysisMixin

logger = logging.getLogger("rcms")


class MinimalRCMS(
    DBMixin,
    UtilsMixin,
    RetrievalMixin,
    ContextMixin,
    SessionMixin,
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
            self.conn.execute("PRAGMA wal_autocheckpoint = 50")  # ~200KB 即触发 checkpoint，避免 WAL 过度膨胀
        except Exception as e:
            logger.warning(f"RCMS: PRAGMA 设置失败 — WAL/busy_timeout 不可用 ({e})，DB 并发保护降级")
        # 进程内用于序列化写操作的简单互斥锁（短期阻塞）
        self._db_lock = threading.RLock()
        self._init_db()
        # Embedding 缓存
        self._emb_cache: dict[str, dict] = {}
        # 维度不匹配待重建队列（record_id 集合）— 模型变更后自动积累
        self._emb_rebuild_queue: set[int] = set()
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
        try:
            self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except Exception:
            pass
        self.conn.close()

    async def build_multi_user_context(
        self, session_id: str, user_input: str,
        speaker_id: str, speaker_name: str,
    ) -> str:
        """为发言者 + 被提及用户分别构建标注了姓名的 narrative context 块"""
        user_entries = [(speaker_id, speaker_name, "当前发言")]
        for mid, label in self.find_mentioned_users(session_id, user_input, speaker_id):
            if mid != speaker_id:
                user_entries.append((mid, label, "被提及"))

        blocks = []
        for uid, display_name, role in user_entries:
            # 查出该用户在当前 session 的所有名字
            rows = self.conn.execute(
                "SELECT label FROM user_mappings WHERE session_id = ? AND user_id = ?",
                (session_id, uid),
            ).fetchall()
            name_parts = [r[0] for r in rows]
            main_name = display_name if display_name in name_parts else (name_parts[0] if name_parts else display_name)
            aliases = [n for n in name_parts if n != main_name]
            alias_str = f"，也被叫做：{'、'.join(aliases)}" if aliases else ""
            header = f"[RCMS 关系上下文: {main_name}（{role}{alias_str}）]"

            mems = await self.retrieve_memories(uid, user_input, 'engaged', session_id=session_id)
            lt = self._load_long_term_context(uid)
            ctx = self.narrative_context('open', session_id, memories=mems, long_term=lt, user_id=uid)
            blocks.append(f"{header}\n" + ctx)
            logger.info(f"RCMS: multi_user_context user={display_name} role={role} memories={len(mems)}")

        return "\n\n".join(blocks)

    def _init_identity(self, user_id: str):
        now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.conn.execute("INSERT OR IGNORE INTO identity_memory (user_id, traits, updated_at) VALUES (?, '[]', ?)", (user_id, now_str))
        self.conn.commit()

    def _load_long_term_context(self, user_id: str) -> dict:
        identity = self.conn.execute(
            "SELECT traits, preferences, self_identity, boundaries FROM identity_memory WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        entities = self.conn.execute("""
            SELECT n1.label, n1.entity_type, e.relation, n2.label
            FROM memory_graph_edges e
            JOIN memory_graph_nodes n1 ON e.from_node_id = n1.node_id
            JOIN memory_graph_nodes n2 ON e.to_node_id = n2.node_id
            WHERE n1.user_id = ? AND e.relation != ''
            ORDER BY e.weight DESC LIMIT 10
        """, (user_id,)).fetchall()
        shared_rows = self.conn.execute(
            "SELECT context_body FROM shared_context WHERE user_id = ? ORDER BY context_id DESC LIMIT 4",
            (user_id,),
        ).fetchall()
        raw_traits = json.loads(identity[0]) if identity and identity[0] else []
        trait_details = []
        for item in raw_traits:
            if isinstance(item, str):
                trait_details.append({"text": item, "strength": 3})
            elif isinstance(item, dict):
                trait_details.append({"text": item.get("t", ""), "strength": item.get("s", 0), "count": item.get("c", 0)})
        trait_details = [p for p in trait_details if p["text"] and p["strength"] > 0]
        def _safe_json(val, default):
            if not val:
                return default
            try:
                return json.loads(val)
            except Exception:
                return default
        return {
            'identity_traits': [p["text"] for p in trait_details],
            'trait_details': trait_details,
            'preferences': _safe_json(identity[1], {}) if identity else {},
            'self_identity': _safe_json(identity[2], []) if identity else [],
            'boundaries': _safe_json(identity[3], []) if identity else [],
            'entities': [{'name': r[0], 'type': r[1] or 'auto', 'relation': r[2], 'fact': r[3]} for r in entities],
            'shared_contexts': [r[0] for r in shared_rows],
        }

    async def post_update_rules(self, user_id: str, session_id: str, user_input: str, stance: str, reply: str = ""):
        """纯管理操作（不做任何 LLM 替代的写入）"""
        now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        lock = getattr(self, '_db_lock', None)
        if lock:
            lock.acquire()
        try:
            self._init_identity(user_id)
            self.conn.execute("UPDATE session_state SET last_active = ? WHERE session_id = ?", (now_str, session_id))
            dt_row = self.conn.execute(
                "SELECT dangling_threads, turn_count FROM session_state WHERE session_id = ?", (session_id,)
            ).fetchone()
            if dt_row and dt_row[0]:
                try:
                    dt_data = json.loads(dt_row[0])
                    if isinstance(dt_data, dict) and dt_data.get("threads"):
                        since_turn = dt_data.get("turn", 0)
                        current_turn = dt_row[1] or 0
                        expire = getattr(self, '_DANGLING_EXPIRE_TURNS', 15)
                        if current_turn - since_turn >= expire:
                            self._archive_dangling(user_id, session_id, now_str, reason="过期")
                except (json.JSONDecodeError, ValueError):
                    logger.debug(f"RCMS: 解析 dangling_threads JSON 失败 user={user_id}")
                self.conn.commit()
        finally:
            if lock:
                try:
                    lock.release()
                except Exception:
                    pass

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
                reply = await backend.generate(self._load_prompts().get("fallback_prompt", "简短回复，一句话以内。\n\n你:"))
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

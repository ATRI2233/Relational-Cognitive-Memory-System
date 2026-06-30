"""
SQLite 实现的 Identity 仓储
===========================

提供 IIdentityRepository 协议的 SQLite 具体实现。
处理 identity_memory 表的 CRUD 操作，以及 Trait 的确认/衰减逻辑。

identity_memory 表结构：
  user_id       TEXT PRIMARY KEY
  traits        TEXT DEFAULT '[]'       — JSON: [{"t": str, "s": int, "c": int}]
  preferences   TEXT DEFAULT '{}'       — JSON: {"likes": [...], "dislikes": [...]}
  self_identity TEXT DEFAULT '[]'       — JSON: ["..."]
  boundaries    TEXT DEFAULT '[]'       — JSON: [{"description": "..."}]
  created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
  updated_at    TIMESTAMP
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime
from typing import List, Optional

from domain.entities.identity import Boundary, Identity, Preferences, Trait
from domain.ports.clock import IClock
from domain.ports.identity_repo import IIdentityRepository

logger = logging.getLogger("rcms")


class SQLiteIdentityRepository(IIdentityRepository):
    """基于 SQLite 的身份仓储实现。

    将 Identity 聚合根持久化到 identity_memory 表。
    JSON 字段映射:
      - traits:       Trait 列表，JSON 数组格式 [{"t": str, "s": int, "c": int}]
      - preferences:  Preferences 值对象，JSON 对象格式 {"likes": [...], "dislikes": [...]}
      - self_identity: 字符串列表，JSON 数组格式 ["..."]
      - boundaries:   Boundary 列表，JSON 数组格式 [{"description": "..."}]

    Args:
        conn: SQLite 数据库连接
        clock: 时间源，用于生成 updated_at 时间戳
    """

    def __init__(self, conn: sqlite3.Connection, clock: IClock) -> None:
        self._conn = conn
        self._clock = clock

    # ── 内部工具方法 ──────────────────────────────────────────────────

    def _now_str(self) -> str:
        """返回格式化的当前时间字符串。"""
        return self._clock.strftime("%Y-%m-%d %H:%M:%S")

    def _ensure_row_exists(self, user_id: str) -> None:
        """确保 identity_memory 中存在该用户的行记录。

        如果不存在则插入一行空记录（仅设 user_id 和 traits 默认值）。
        """
        try:
            self._conn.execute(
                "INSERT OR IGNORE INTO identity_memory (user_id, traits, updated_at) "
                "VALUES (?, '[]', ?)",
                (user_id, self._now_str()),
            )
            self._conn.commit()
        except sqlite3.Error as e:
            logger.error("_ensure_row_exists 失败: %s", e)
            self._conn.rollback()
            raise

    # ── JSON 序列化 / 反序列化 ────────────────────────────────────────

    @staticmethod
    def _traits_to_json(traits: List[Trait]) -> str:
        """将 Trait 列表序列化为 JSON 字符串。

        格式: [{"t": text, "s": strength, "c": count}, ...]
        """
        return json.dumps(
            [{"t": t.text, "s": t.strength, "c": t.count} for t in traits],
            ensure_ascii=False,
        )

    @staticmethod
    def _traits_from_json(raw: Optional[str]) -> List[Trait]:
        """从 JSON 字符串反序列化 Trait 列表。

        兼容旧格式（字符串或旧 dict 结构），统一返回 Trait 值对象。
        旧格式字符串 -> Trait(text=item, strength=3, count=0)
        旧格式 dict   -> Trait(text=item["t"], strength=item["s"], count=item["c"])
        """
        if not raw:
            return []
        try:
            items = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return []
        traits: List[Trait] = []
        for item in items:
            try:
                if isinstance(item, str):
                    # 兼容旧格式: 纯字符串特质
                    traits.append(Trait(text=item, strength=3, count=0))
                elif isinstance(item, dict):
                    # 兼容新旧 dict 格式: key 名 t/text, s/strength, c/count
                    text = item.get("t", item.get("text", ""))
                    strength = item.get("s", item.get("strength", 0))
                    count = item.get("c", item.get("count", 0))
                    traits.append(Trait(text=text, strength=strength, count=count))
            except (ValueError, TypeError):
                continue
        return traits

    @staticmethod
    def _preferences_to_json(prefs: Preferences) -> str:
        """将 Preferences 序列化为 JSON 字符串。

        格式: {"likes": [...], "dislikes": [...]}
        """
        return json.dumps(
            {"likes": prefs.likes, "dislikes": prefs.dislikes},
            ensure_ascii=False,
        )

    @staticmethod
    def _preferences_from_json(raw: Optional[str]) -> Preferences:
        """从 JSON 字符串反序列化 Preferences。"""
        if not raw:
            return Preferences()
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return Preferences()
        if not isinstance(data, dict):
            return Preferences()
        likes = data.get("likes", []) if isinstance(data.get("likes"), list) else []
        dislikes = (
            data.get("dislikes", []) if isinstance(data.get("dislikes"), list) else []
        )
        return Preferences(likes=likes, dislikes=dislikes)

    @staticmethod
    def _boundaries_to_json(boundaries: List[Boundary]) -> str:
        """将 Boundary 列表序列化为 JSON 字符串。

        格式: [{"description": "..."}, ...]
        """
        return json.dumps(
            [{"description": b.description} for b in boundaries],
            ensure_ascii=False,
        )

    @staticmethod
    def _boundaries_from_json(raw: Optional[str]) -> List[Boundary]:
        """从 JSON 字符串反序列化 Boundary 列表。"""
        if not raw:
            return []
        try:
            items = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return []
        boundaries: List[Boundary] = []
        for item in items:
            try:
                if isinstance(item, dict):
                    desc = item.get("description", "")
                else:
                    desc = str(item)
                boundaries.append(Boundary(description=desc))
            except (ValueError, TypeError):
                continue
        return boundaries

    @staticmethod
    def _parse_timestamp(val: object) -> datetime:
        """将数据库返回的时间戳值解析为 datetime。

        SQLite 无原生 datetime 类型，值可能为字符串或已由
        connection 工厂转换为 datetime 对象。
        """
        if isinstance(val, datetime):
            return val
        if isinstance(val, str):
            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
                try:
                    return datetime.strptime(val, fmt)
                except ValueError:
                    continue
        return datetime.now()

    # ── IIdentityRepository 接口方法 ─────────────────────────────────

    async def get(self, user_id: str) -> Optional[Identity]:
        """获取用户的完整身份信息。

        读取 identity_memory 表的一行，将 JSON 字段反序列化为
        Trait / Preferences / Boundary 领域对象。

        Args:
            user_id: 用户标识

        Returns:
            如果用户存在则返回 Identity 实体，否则返回 None
        """
        try:
            row = self._conn.execute(
                "SELECT user_id, traits, preferences, self_identity, boundaries, "
                "       created_at, updated_at "
                "FROM identity_memory WHERE user_id = ?",
                (user_id,),
            ).fetchone()
        except sqlite3.Error as e:
            logger.error("Failed to get identity: %s", e)
            raise

        if row is None:
            return None

        traits = self._traits_from_json(row[1])

        prefs = self._preferences_from_json(row[2])

        self_identity: List[str] = []
        if row[3]:
            try:
                parsed = json.loads(row[3])
                if isinstance(parsed, list):
                    self_identity = [str(s) for s in parsed]
            except (json.JSONDecodeError, TypeError):
                self_identity = []

        boundaries = self._boundaries_from_json(row[4])

        created_at = self._parse_timestamp(row[5]) if row[5] else datetime.now()
        updated_at = self._parse_timestamp(row[6]) if row[6] else datetime.now()

        return Identity(
            user_id=str(row[0]),
            traits=traits,
            preferences=prefs,
            self_identity=self_identity,
            boundaries=boundaries,
            created_at=created_at,
            updated_at=updated_at,
        )

    async def save_traits(self, user_id: str, traits: List[Trait]) -> None:
        """保存用户特质列表（含确认/衰减合并逻辑）。

        处理逻辑与 rcms_core/analysis.py 中的 traits_updates 一致:
          1. 加载已有特质
          2. 传入的 traits 视为"本轮被确认的特质":
              新特质: strength=5, count=1
              已有特质: strength 置 5, count +1
          3. 未被确认的特质: strength 衰减 -1, 下限为 min(count//2, 2)
             衰减后 strength <= 0 则移除该特质
          4. 超过 30 条时按 s*2 + c 排序保留前 30

        Args:
            user_id: 用户标识
            traits: 本轮被确认的特质列表
        """
        self._ensure_row_exists(user_id)

        try:
            raw = self._conn.execute(
                "SELECT traits FROM identity_memory WHERE user_id = ?",
                (user_id,),
            ).fetchone()
        except sqlite3.Error as e:
            logger.error("save_traits SELECT 失败: %s", e)
            raise

        # 1. 加载已有特质
        existing_traits = self._traits_from_json(raw[0] if raw else None)

        # text -> {"strength": int, "count": int}
        trait_map: dict[str, dict[str, int]] = {}
        for t in existing_traits:
            trait_map[t.text] = {"strength": t.strength, "count": t.count}

        # 2. 确认传入的特质: 新特质 strength=5, count=1; 已有置 5, count+1
        confirmed_texts = {t.text for t in traits}
        for t in traits:
            if t.text not in trait_map:
                trait_map[t.text] = {"strength": 5, "count": 1}
            else:
                trait_map[t.text]["strength"] = 5
                trait_map[t.text]["count"] += 1

        # 3. 衰减未被确认的特质
        for text in list(trait_map.keys()):
            if text not in confirmed_texts:
                floor = min(trait_map[text]["count"] // 2, 2)
                trait_map[text]["strength"] = max(
                    trait_map[text]["strength"] - 1, floor
                )
                if trait_map[text]["strength"] <= 0:
                    del trait_map[text]

        # 4. 容量限制: 超过 30 条时按 s*2 + c 排序保留前 30
        if len(trait_map) > 30:
            sorted_items = sorted(
                trait_map.items(),
                key=lambda x: x[1]["strength"] * 2 + x[1]["count"],
                reverse=True,
            )[:30]
            trait_map = dict(sorted_items)

        # 5. 写回数据库
        new_traits = [
            Trait(text=text, strength=v["strength"], count=v["count"])
            for text, v in trait_map.items()
        ]
        new_json = self._traits_to_json(new_traits)

        try:
            self._conn.execute(
                "UPDATE identity_memory SET traits = ?, updated_at = ? WHERE user_id = ?",
                (new_json, self._now_str(), user_id),
            )
            self._conn.commit()
        except sqlite3.Error as e:
            logger.error("save_traits UPDATE 失败: %s", e)
            self._conn.rollback()
            raise

    async def save_preferences(self, user_id: str, prefs: Preferences) -> None:
        """保存用户偏好信息（覆盖写入）。

        对应 identity_memory.preferences 字段的 JSON 序列化写入。
        覆盖写入，不合并。

        Args:
            user_id: 用户标识
            prefs: 偏好值对象
        """
        self._ensure_row_exists(user_id)
        prefs_json = self._preferences_to_json(prefs)
        try:
            self._conn.execute(
                "UPDATE identity_memory SET preferences = ?, updated_at = ? WHERE user_id = ?",
                (prefs_json, self._now_str(), user_id),
            )
            self._conn.commit()
        except sqlite3.Error as e:
            logger.error("save_preferences 失败: %s", e)
            self._conn.rollback()
            raise

    async def save_self_identity(self, user_id: str, identities: List[str]) -> None:
        """保存用户自我认知（覆盖写入）。

        对应 identity_memory.self_identity 字段的 JSON 序列化写入。
        覆盖写入，不合并。

        Args:
            user_id: 用户标识
            identities: 自我认知描述字符串列表
        """
        self._ensure_row_exists(user_id)
        identities_json = json.dumps(identities, ensure_ascii=False)
        try:
            self._conn.execute(
                "UPDATE identity_memory SET self_identity = ?, updated_at = ? WHERE user_id = ?",
                (identities_json, self._now_str(), user_id),
            )
            self._conn.commit()
        except sqlite3.Error as e:
            logger.error("save_self_identity 失败: %s", e)
            self._conn.rollback()
            raise

    async def save_boundaries(self, user_id: str, boundaries: List[Boundary]) -> None:
        """保存用户边界/雷区（覆盖写入）。

        对应 identity_memory.boundaries 字段的 JSON 序列化写入。
        覆盖写入，不合并。

        Args:
            user_id: 用户标识
            boundaries: 边界值对象列表
        """
        self._ensure_row_exists(user_id)
        boundaries_json = self._boundaries_to_json(boundaries)
        try:
            self._conn.execute(
                "UPDATE identity_memory SET boundaries = ?, updated_at = ? WHERE user_id = ?",
                (boundaries_json, self._now_str(), user_id),
            )
            self._conn.commit()
        except sqlite3.Error as e:
            logger.error("save_boundaries 失败: %s", e)
            self._conn.rollback()
            raise

    async def update_identity(self, user_id: str, identity: Identity) -> None:
        """更新完整的用户身份信息（全字段覆盖写入）。

        使用 INSERT OR REPLACE 一次性写入 Identity 实体的所有字段。
        当需要原子更新整个聚合根时使用此方法。

        Args:
            user_id: 用户标识
            identity: 完整的 Identity 实体
        """
        traits_json = self._traits_to_json(identity.traits)
        prefs_json = self._preferences_to_json(identity.preferences)
        self_id_json = json.dumps(identity.self_identity, ensure_ascii=False)
        boundaries_json = self._boundaries_to_json(identity.boundaries)
        now_str = self._now_str()
        try:
            self._conn.execute(
                "INSERT OR REPLACE INTO identity_memory "
                "(user_id, traits, preferences, self_identity, boundaries, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (user_id, traits_json, prefs_json, self_id_json, boundaries_json, now_str),
            )
            self._conn.commit()
        except sqlite3.Error as e:
            logger.error("update_identity 失败: %s", e)
            self._conn.rollback()
            raise

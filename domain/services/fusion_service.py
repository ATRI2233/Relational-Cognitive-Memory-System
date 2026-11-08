"""
FusionService — 三通道融合算法。

从 retrieval.py 的 _fusion() 方法提取，职责单一：在固定的注入额度内
挑出「最该被提起的那几条」，而不是「凑满额度」。

选择准则（按优先级）：
1. 通道内保底（ch_min），但保底项同样要过全局相对质量门槛
2. 双去重：优先按业务身份（memory_id 等 dedup_key），其次按内容 hash，
   再对措辞不同但语义重复的条目做字符 2-gram 近似去重
3. 剩余名额按加权分补位，低于质量门槛的一律不补
4. total_cap 是硬上限：宁肯少给，不凑数

关于「质量门槛」的设计理由：
    channel_min 原本是无条件保底，导致某个通道即使完全不相关也要占一个
    名额（实测中「活泼一点喵」这类输入会被硬塞一条三天前的技术讨论）。
    这里改成以全局最高加权分为基准的相对门槛，弱势通道自动让位。

关于「近似去重」的设计理由：
    同一条记忆的时间通道输出是「模糊时间 + summary」，语义通道输出是
    原始 content，两者 hash 不同，会被当成两条。而图谱关系链存在
    「用户[讨论过]A → A[可自动注入]B」和「B[包含]C → C[由用户维护]用户」
    这类同源不同起点的表述，实测 2-gram jaccard 达 0.58~0.69。
    业务身份去重解决前者，2-gram 近似去重解决后者。
"""
from __future__ import annotations

import hashlib
import re
from typing import Any, Protocol, runtime_checkable


# 保底项质量下限：低于全局最高加权分的这个比例，视为该通道本轮不相关
_DEFAULT_FLOOR_RATIO = 0.35
# 字符 2-gram jaccard 相似度达到该值即视为同一条的重复表述
_DEFAULT_NEAR_DUP_RATIO = 0.5
# 短文本的 2-gram 统计不可靠，token 数低于此值不做近似去重
_NEAR_DUP_MIN_TOKENS = 8


@runtime_checkable
class IFusionConfig(Protocol):
    """融合配置接口 — 可从 Settings 或测试配置传入

    fusion_floor_ratio / fusion_near_dup_ratio 为可选字段，
    未提供时使用模块默认值，保证旧配置对象仍可工作。
    """
    total_cap: int
    channel_min: list[int]
    channel_weights: list[float]


class ChannelTag:
    """通道标签常量"""
    RECENT = "recent"
    RESONANCE = "resonance"
    SKELETON = "skeleton"


# 通道标签到索引的映射
TAG_TO_INDEX = {
    ChannelTag.RECENT: 0,
    ChannelTag.RESONANCE: 1,
    ChannelTag.SKELETON: 2,
}

# 固定处理顺序：保底与补位都按此顺序遍历，保证结果可复现
_CHANNEL_ORDER = (ChannelTag.RECENT, ChannelTag.RESONANCE, ChannelTag.SKELETON)


class FusionService:
    """三通道融合服务 — 质量门槛 / 双去重 / 加权排序 / 硬截断"""

    def __init__(self, config: IFusionConfig):
        self._config = config

    def fuse(
        self, channels: dict[str, list[tuple]]
    ) -> list[tuple[str, str]]:
        """融合三通道结果

        Args:
            channels: {tag: [(content, score), ...] 或 [(content, score, dedup_key), ...]},
                      tag 为 'recent'/'resonance'/'skeleton'

        Returns:
            [(content, tag), ...] 按权重降序，总数不超过 total_cap
        """
        return self.fuse_with_detail(channels)[0]

    def fuse_with_detail(
        self, channels: dict[str, list[tuple]]
    ) -> tuple[list[tuple[str, str]], dict]:
        """融合三通道结果，同时返回可观测的过程明细。

        与 :meth:`fuse` 的唯一区别是额外产出一份 detail 字典，供 WebUI 展示
        「保底哪几条、补位哪几条、哪些被门槛或去重丢掉」，算法完全相同。

        Args:
            channels: {tag: [(content, score[, dedup_key]), ...]}。
                      dedup_key 用于按业务身份去重（通常是 memory_id），
                      缺省时退化为按内容 hash 去重。

        Returns:
            (results, detail)：
              results: [(content, tag), ...] 按加权分降序，不超过 total_cap
              detail:  融合参数、门槛值、保底阶段、补位候选、各类丢弃、最终明细
        """
        ch_min = self._config.channel_min
        ch_weights = self._config.channel_weights
        total_cap = self._config.total_cap
        if sum(ch_min) > total_cap:
            raise ValueError(
                f"sum(ch_min)={sum(ch_min)} exceeds total_cap={total_cap}"
            )
        if len(ch_min) < 3 or len(ch_weights) < 3:
            raise ValueError(
                f"ch_min ({len(ch_min)}) and ch_weights ({len(ch_weights)}) "
                f"must each have at least 3 elements"
            )

        floor_ratio = float(
            getattr(self._config, "fusion_floor_ratio", _DEFAULT_FLOOR_RATIO)
        )
        near_dup_ratio = float(
            getattr(self._config, "fusion_near_dup_ratio", _DEFAULT_NEAR_DUP_RATIO)
        )

        detail: dict = {
            "channel_min": list(ch_min),
            "channel_weights": list(ch_weights),
            "total_cap": total_cap,
            "floor_ratio": round(floor_ratio, 6),
            "best_weighted": 0.0,
            "gate": 0.0,
            "near_dup_ratio": round(near_dup_ratio, 6),
            "floor": [],            # 保底阶段实际取走的条目
            "gate_skipped": [],     # 因低于质量门槛被跳过的通道/条目
            "fill_pool": [],        # 补位阶段的候选池（已排序）
            "dedup_dropped": [],    # 完全重复被丢弃（stage 为 pre 表示同源预筛）
            "near_dup_dropped": [], # 近似重复被丢弃
            "final": [],
        }

        # ── 规范化每个通道的候选，并做通道内归一化 ──
        # 三个通道的分数量纲完全不同：骨架通道 score = 边权 + 2.0（有关联就白送
        # 两分，典型 1.9~3.0），时间通道 (decay+boost)×(0.5+imp)（典型 0.8~2.0），
        # 语义通道 cos×0.6 + imp×0.4（典型 0.4~0.9）。直接加权比较的话，
        # channel_weights=[0.5,1.0,0.6] 的设计意图（语义最重要）永远不成立，
        # 骨架边永远霸占前几位。先除以各通道本轮最高分映射到 0~1，权重才生效。
        normalized: dict[str, list[tuple[str, float, Any]]] = {}
        raw_buckets: dict[str, list[tuple[str, float, float, Any]]] = {}
        for tag_key in _CHANNEL_ORDER:
            idx = TAG_TO_INDEX[tag_key]
            weight = ch_weights[idx]
            bucket: list[tuple[str, float, float, Any]] = []
            for item in channels.get(tag_key, []) or []:
                if not item:
                    continue
                content = item[0]
                raw_score = float(item[1]) if len(item) > 1 else 0.0
                dedup_key = item[2] if len(item) > 2 else None
                bucket.append((content, raw_score, 0.0, dedup_key))
            raw_buckets[tag_key] = bucket
            top = max((b[1] for b in bucket), default=0.0)
            if top > 0:
                normalized[tag_key] = [
                    (c, (s / top) * weight, k) for c, s, _, k in bucket
                ]
            else:
                normalized[tag_key] = []

        def _identity(content: str, dedup_key: Any) -> tuple:
            """业务身份优先，缺省退化为内容 hash"""
            if dedup_key is not None and dedup_key != "":
                return ("id", str(dedup_key))
            return ("txt", self._hash_key(content))

        # ── 同一业务身份只保留信息量最大的版本 ──
        # 时间通道输出「模糊时间 + summary」，语义通道输出原始 content，
        # 二者是同一件事的长短两版。保底顺序是 recent 在前，若不预筛，
        # 压缩摘要会先占位，把更详细的原文挤成「重复」丢弃，信息量反而下降。
        best_len: dict[tuple, int] = {}
        for tag_key in _CHANNEL_ORDER:
            for content, _, dedup_key in normalized[tag_key]:
                ident = _identity(content, dedup_key)
                if len(content) > best_len.get(ident, -1):
                    best_len[ident] = len(content)

        best_weighted = 0.0
        for tag_key in _CHANNEL_ORDER:
            kept: list[tuple[str, float, Any]] = []
            for content, weighted, dedup_key in normalized[tag_key]:
                if len(content) < best_len[_identity(content, dedup_key)]:
                    detail["dedup_dropped"].append(
                        {"tag": tag_key, "stage": "pre", "content": content}
                    )
                    continue
                kept.append((content, weighted, dedup_key))
                if weighted > best_weighted:
                    best_weighted = weighted
            normalized[tag_key] = kept

        # 相对质量门槛：以本轮归一化后的全局最高分为基准
        gate = best_weighted * floor_ratio if best_weighted > 0 else 0.0
        detail["best_weighted"] = round(best_weighted, 6)
        detail["gate"] = round(gate, 6)

        seen_keys: set = set()
        seen_tokens: list[frozenset] = []
        merged: list[tuple[str, str, float, str]] = []  # (content, tag, weighted, stage)

        def _near_dup(content: str) -> bool:
            tokens = self._tokens(content)
            if len(tokens) < _NEAR_DUP_MIN_TOKENS:
                return False
            for prev in seen_tokens:
                inter = len(tokens & prev)
                if not inter:
                    continue
                union = len(tokens | prev)
                if union and inter / union >= near_dup_ratio:
                    return True
            return False

        def _accept(content: str, dedup_key: Any) -> str:
            """尝试收纳一条候选，返回动作：'taken' / 'dup' / 'near_dup'"""
            key = _identity(content, dedup_key)
            if key in seen_keys:
                return "dup"
            if _near_dup(content):
                return "near_dup"
            seen_keys.add(key)
            seen_tokens.append(frozenset(self._tokens(content)))
            return "taken"

        # ── Phase 1: 每通道保底，但保底项同样要过质量门槛 ──
        # 记录每个通道在保底阶段考察到的位置，补位阶段从这里接着走，
        # 避免同一条候选被两个阶段重复评估。
        consumed_upto = [0, 0, 0]
        for tag_key in _CHANNEL_ORDER:
            idx = TAG_TO_INDEX[tag_key]
            min_count = ch_min[idx]
            bucket = normalized[tag_key]
            taken = 0
            pos = 0
            for i, (content, weighted, dedup_key) in enumerate(bucket):
                if taken >= min_count:
                    pos = i
                    break
                pos = i + 1
                # 通道内已按分数降序，低于门槛说明整个通道本轮不相关，
                # 后续条目只会更低，直接整体让位
                if weighted < gate:
                    detail["gate_skipped"].append(
                        {
                            "tag": tag_key,
                            "stage": "floor",
                            "reason": "below_gate",
                            "weighted": round(weighted, 6),
                            "content": content,
                        }
                    )
                    break
                action = _accept(content, dedup_key)
                if action == "taken":
                    merged.append((content, tag_key, weighted, "floor"))
                    taken += 1
                    detail["floor"].append(
                        {
                            "tag": tag_key,
                            "weight": ch_weights[idx],
                            "weighted": round(weighted, 6),
                            "content": content,
                        }
                    )
                elif action == "near_dup":
                    detail["near_dup_dropped"].append(
                        {"tag": tag_key, "stage": "floor", "content": content}
                    )
                else:
                    detail["dedup_dropped"].append(
                        {"tag": tag_key, "stage": "floor", "content": content}
                    )
            consumed_upto[idx] = pos

        # ── Phase 2: 剩余名额按加权分补位；不足门槛的一律不补 ──
        remaining = total_cap - len(merged)
        if remaining > 0:
            pool: list[tuple[str, str, float, Any]] = []
            for tag_key in _CHANNEL_ORDER:
                idx = TAG_TO_INDEX[tag_key]
                for content, weighted, dedup_key in normalized[tag_key][consumed_upto[idx]:]:
                    pool.append((content, tag_key, weighted, dedup_key))

            pool.sort(key=lambda x: -x[2])
            detail["fill_pool"] = [
                {"tag": tag, "weighted": round(score, 6), "content": content}
                for content, tag, score, _ in pool
            ]
            for content, tag_key, weighted, dedup_key in pool:
                if len(merged) >= total_cap:
                    break
                if weighted < gate:
                    detail["gate_skipped"].append(
                        {
                            "tag": tag_key,
                            "stage": "fill",
                            "reason": "below_gate",
                            "weighted": round(weighted, 6),
                            "content": content,
                        }
                    )
                    continue
                action = _accept(content, dedup_key)
                if action == "taken":
                    merged.append((content, tag_key, weighted, "fill"))
                elif action == "near_dup":
                    detail["near_dup_dropped"].append(
                        {"tag": tag_key, "stage": "fill", "content": content}
                    )
                else:
                    detail["dedup_dropped"].append(
                        {"tag": tag_key, "stage": "fill", "content": content}
                    )

        # 返回 (content, tag) 结构，按加权分降序；total_cap 是硬上限
        merged.sort(key=lambda x: -x[2])
        ordered = merged[:total_cap]
        detail["final"] = [
            {
                "tag": tag,
                "stage": stage,
                "weighted": round(score, 6),
                "content": content,
            }
            for content, tag, score, stage in ordered
        ]
        return [(content, tag) for content, tag, _, _ in ordered], detail

    @staticmethod
    def _hash_key(content: str) -> str:
        return hashlib.md5(content.strip().encode("utf-8")).hexdigest()

    @staticmethod
    def _tokens(content: str) -> set[str]:
        """字符 2-gram 集合 — 对中文无分词依赖，够用且零成本"""
        text = re.sub(r"\s+", "", content or "")
        if not text:
            return set()
        if len(text) < 2:
            return {text}
        return {text[i:i + 2] for i in range(len(text) - 1)}

"""
FusionService — 三通道融合算法。

从 retrieval.py 的 _fusion() 方法提取，职责单一：
1. 每通道保底取 ch_min[i] 条
2. 内容 hash 去重
3. 加权排序后截取 total_cap 条
"""
from __future__ import annotations

import hashlib
from typing import Protocol, runtime_checkable


@runtime_checkable
class IFusionConfig(Protocol):
    """融合配置接口 — 可从 Settings 或测试配置传入"""
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


class FusionService:
    """三通道融合服务 — 保底/去重/加权排序/截断"""

    def __init__(self, config: IFusionConfig):
        self._config = config

    def fuse(
        self, channels: dict[str, list[tuple[str, float]]]
    ) -> list[tuple[str, str]]:
        """融合三通道结果

        Args:
            channels: {tag: [(content_str, score), ...]},
                      tag 为 'recent'/'resonance'/'skeleton'

        Returns:
            [(content, tag), ...] 按权重降序，总数不超过 total_cap
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

        seen: set[str] = set()
        merged: list[tuple[str, str, float]] = []  # (content, tag, weighted_score)
        actual_taken = [0, 0, 0]  # actual items taken per channel in Phase 1
        attempted_taken = [0, 0, 0]  # items consumed from iterator per channel

        # Phase 1: 每通道保底（using item_score * channel_weight）
        for tag_key in [ChannelTag.RECENT, ChannelTag.RESONANCE, ChannelTag.SKELETON]:
            items = channels.get(tag_key, [])
            min_count = ch_min[TAG_TO_INDEX[tag_key]]
            taken = 0
            attempted = 0
            for content, item_score in items:
                if taken >= min_count:
                    break
                attempted += 1
                key = self._hash_key(content)
                if key not in seen:
                    seen.add(key)
                    weighted = item_score * ch_weights[TAG_TO_INDEX[tag_key]]
                    merged.append((content, tag_key, weighted))
                    taken += 1
            actual_taken[TAG_TO_INDEX[tag_key]] = taken
            attempted_taken[TAG_TO_INDEX[tag_key]] = attempted

        # Phase 2: 剩余名额，按加权分取前 N
        n = total_cap - sum(ch_min)
        if n > 0:
            pool: list[tuple[str, str, float]] = []
            for tag_key in [ChannelTag.RECENT, ChannelTag.RESONANCE, ChannelTag.SKELETON]:
                items = channels.get(tag_key, [])
                # 跳过已经保底取走的部分
                taken_so_far = attempted_taken[TAG_TO_INDEX[tag_key]]
                for content, item_score in items[taken_so_far:taken_so_far + n]:
                    key = self._hash_key(content)
                    if key not in seen:
                        weighted = item_score * ch_weights[TAG_TO_INDEX[tag_key]]
                        pool.append((content, tag_key, weighted))

            pool.sort(key=lambda x: -x[2])
            for content, tag_key, score in pool:
                if len(merged) >= total_cap:
                    break
                key = self._hash_key(content)
                if key not in seen:
                    seen.add(key)
                    merged.append((content, tag_key, score))

        # 返回 (content, tag) 结构，按加权分降序
        merged.sort(key=lambda x: -x[2])
        return [(content, tag) for content, tag, _ in merged[:total_cap]]

    @staticmethod
    def _hash_key(content: str) -> str:
        return hashlib.md5(content.strip().encode("utf-8")).hexdigest()

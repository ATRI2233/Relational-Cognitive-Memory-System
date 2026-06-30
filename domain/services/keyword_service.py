"""
KeywordService — 关键词提取和中文文本分析。

提取自 retrieval.py / utils.py，职责：
1. 中英文关键词提取（依赖通过回调注入的分词器）
2. 中文二元组提取
3. 时间词范围过滤

分词器接口 ITokenizer 定义在此文件，具体实现（jieba）在 infrastructure 层。
"""
from __future__ import annotations

import logging
import re
from typing import Callable, Protocol, runtime_checkable


@runtime_checkable
class ITokenizer(Protocol):
    """分词器接口 — 接受文本返回词列表"""
    def cut(self, text: str) -> list[str]: ...


@runtime_checkable
class IWordListConfig(Protocol):
    """词表配置接口 — 可从 Settings 传入"""
    trivial_markers: list[str]
    stop_words: list[str]
    time_words: dict[str, tuple[int, int] | list[int]]


class KeywordService:
    """关键词提取和文本分析服务"""

    def __init__(self, tokenizer: ITokenizer | Callable[[str], list[str]],
                 config: IWordListConfig):
        self._tokenizer = tokenizer
        self._config = config

    def extract_keywords(self, text: str, max_kw: int = 5) -> list[str]:
        """从文本中提取关键词

        对含中文的 segment 用分词器处理，过滤停用词和琐事标记词；
        纯英文/数字 segment 直接保留。

        Args:
            text: 输入文本
            max_kw: 最大关键词数，默认 5

        Returns:
            关键词列表，按文本中出现顺序
        """
        if not text or not text.strip():
            return []

        tokens = re.split(r'[\s,，。！？、；：""''（）()—\n]+', text)
        result: list[str] = []
        stop = set(self._config.stop_words)
        trivial = set(self._config.trivial_markers)

        for t in tokens:
            if not t:
                continue
            if re.search(r'[一-鿿]', t):
                segs = self._tokenizer.cut(t) if hasattr(self._tokenizer, 'cut') else self._tokenizer(t)
                for s in segs:
                    s = s.strip()
                    if (len(s) > 1 and s not in trivial and s not in stop
                            and s not in result):
                        result.append(s)
                        if len(result) >= max_kw:
                            return result[:max_kw]
            else:
                if (len(t) > 1 and t not in trivial and t not in stop
                        and t not in result):
                    result.append(t)
                    if len(result) >= max_kw:
                        return result[:max_kw]

        return result[:max_kw]

    def parse_time_filter(self, user_input: str) -> tuple[int, int] | None:
        """解析输入文本中的时间词 → (min_days_ago, max_days_ago)

        Args:
            user_input: 用户输入文本

        Returns:
            (最小天数, 最大天数) 或 None（无匹配时间词）
        """
        for word, value in self._config.time_words.items():
            if len(value) != 2:
                logger = logging.getLogger("rcms")
                logger.warning("time_words[%r] has %d elements, expected 2", word, len(value))
                continue
            min_d, max_d = value  # type: ignore
            if word in user_input:
                return (min_d, max_d)
        return None

    @staticmethod
    def chinese_bigrams(text: str) -> set[str]:
        """提取文本中的所有中文二元组

        Args:
            text: 输入文本

        Returns:
            中文二元组集合（相邻两个中文字符）
        """
        chars = re.findall(r'[一-鿿]', text)
        return {''.join(chars[i:i + 2]) for i in range(len(chars) - 1)}

    @staticmethod
    def precise_kw_match(text: str, kw: str) -> bool:
        """精确关键词匹配

        Args:
            text: 被搜索文本
            kw: 关键词

        Returns:
            kw 是否在 text 中
        """
        return kw in text

    @staticmethod
    def score_markers(text: str, markers: list[str], per_hit: float = 0.3) -> float:
        """按标记词命中次数计分，上限 1.0

        Args:
            text: 输入文本
            markers: 标记词列表
            per_hit: 每次命中的加分，默认 0.3

        Returns:
            [0.0, 1.0] 范围的分数
        """
        count = sum(1 for m in markers if m in text)
        return min(count * per_hit, 1.0)

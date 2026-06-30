"""
ImportanceService — 重要性相关工具服务

提取自 analysis.py / retrieval.py 中的重要性处理逻辑：
1. clamp / scale — 值域钳制与线性映射
2. decay_floor — analysis.py 中特质衰减最低保留值计算
3. distill_importance — 蒸馏条目重要性（不低于配置下限）
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class IImportanceConfig(Protocol):
    """重要性配置接口

    与 infrastructure.config.settings.StorageMaintenanceSettings 的
    重要性相关字段形状匹配，实现零耦合适配。
    """

    cognitive_distill_importance_floor: float  # 蒸馏重要性下限


class ImportanceService:
    """重要性工具服务

    职责：
      - clamp:          将原始值钳制到合法范围
      - scale:          线性映射（两范围间归一化）
      - decay_floor:    计算特质衰减最低保留值（来自 analysis.py）
      - distill_importance: 计算蒸馏条目的重要性（不低于下限）

    使用示例::

        from domain.services.importance_service import ImportanceService
        from infrastructure.config.settings import get_settings

        config = get_settings().maintenance          # 实现 IImportanceConfig
        svc = ImportanceService(config)
        imp = svc.distill_importance(0.5)
        # => 0.5 (高于 floor 0.1)
    """

    def __init__(self, config: IImportanceConfig) -> None:
        """初始化

        Args:
            config: 实现 IImportanceConfig 协议的配置对象
        """
        self._config = config

    # ── 工具方法 ──────────────────────────────────────────────

    def clamp(
        self,
        value: float,
        floor: float | None = None,
        ceil: float | None = None,
    ) -> float:
        """将值钳制在 [floor, ceil] 范围内

        Args:
            value: 原始值
            floor: 下限（默认 0.0）
            ceil:  上限（默认 1.0）

        Returns:
            钳制后的值
        """
        low = floor if floor is not None else 0.0
        high = ceil if ceil is not None else 1.0
        return max(low, min(high, value))

    @staticmethod
    def scale(
        value: float,
        min_in: float,
        max_in: float,
        min_out: float,
        max_out: float,
    ) -> float:
        """线性映射

        将 [min_in, max_in] 范围的 value 映射到 [min_out, max_out]。

        当 max_in == min_in 时（零宽区间），返回输出范围的中点。

        Args:
            value:  输入值
            min_in: 输入范围下限
            max_in: 输入范围上限
            min_out: 输出范围下限
            max_out: 输出范围上限

        Returns:
            映射后的值
        """
        if max_in == min_in:
            return (min_out + max_out) / 2.0
        ratio = (value - min_in) / (max_in - min_in)
        return min_out + ratio * (max_out - min_out)

    @staticmethod
    def decay_floor(strength: int, min_strength: int = 2) -> int:
        """计算特质衰减的最低强度值

        对应 analysis.py 中特质衰减的 floor 计算：
          floor = max(strength - 1, min_strength)

        Args:
            strength:     当前强度值
            min_strength: 最低保留强度，默认 2

        Returns:
            衰减后的最低强度值
        """
        return max(strength - 1, min_strength)

    def distill_importance(self, raw_importance: float | None = None) -> float:
        """计算蒸馏条目的重要性

        蒸馏条目的重要性至少为 config.cognitive_distill_importance_floor。
        如果指定了 raw_importance 则取两者较大值，否则直接使用 floor。

        Args:
            raw_importance: 原始重要性（可选）

        Returns:
            最终重要性值（不低于 cognitive_distill_importance_floor）
        """
        base = (
            raw_importance
            if raw_importance is not None
            else self._config.cognitive_distill_importance_floor
        )
        return max(base, self._config.cognitive_distill_importance_floor)

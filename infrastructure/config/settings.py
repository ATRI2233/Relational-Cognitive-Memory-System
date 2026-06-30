"""
RCMS 集中配置层

用 pydantic-settings 收口所有硬编码常量，支持:
  - 环境变量覆盖 (RCMS_ 前缀)
  - 数值范围校验 (ge/le)
  - 类型注解完整检查

用法:
    from infrastructure.config.settings import Settings, RetrievalSettings

    settings = Settings()
    settings.retrieval.total_cap          # 5
    settings.emotional_words.emotional_words  # ['累', '烦', ...]

    # 环境变量覆盖示例:
    #   RCMS_STORAGE_DB_PATH=/data/rcms/memory.db
    #   RCMS_RETRIEVAL_TOTAL_CAP=10
"""

import json
import logging
import os
from typing import Any, Optional

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


# ==============================================================
# 子模型 (每层独立校验)
# ==============================================================


class StorageSettings(BaseModel):
    """数据库存储配置 (对应 core.py db_path / PRAGMA)"""

    db_path: str = Field(
        default="memory.db",
        description="SQLite 数据库文件路径",
    )
    data_dir: str = Field(
        default="data",
        description="数据库存放目录 (多用户隔离时每个文件放此处)",
    )
    busy_timeout_ms: int = Field(
        default=5000,
        ge=100,
        le=60000,
        description="SQLite busy_timeout (ms)",
    )
    wal_autocheckpoint_pages: int = Field(
        default=50,
        ge=10,
        le=1000,
        description="WAL 自动 checkpoint 页数 (每页 ~4KB, 50 页约 200KB)",
    )


class RetrievalSettings(BaseModel):
    """三通道召回配置 (对应 config.json analysis.retrieval + core.py 图参数)"""

    embedding_enabled: bool = Field(
        default=True,
        description="启用 Embedding 向量检索",
    )
    total_cap: int = Field(
        default=5,
        ge=1,
        le=50,
        description="三通道融合召回总上限 (去重后)",
    )
    channel_min: list[int] = Field(
        default=[1, 1, 1],
        description="每通道最低保底条数 [ch1, ch2, ch3]",
    )
    channel_weights: list[float] = Field(
        default=[0.5, 1.0, 0.6],
        description="每通道加权系数 [ch1, ch2, ch3]",
    )
    time_decay_halflife: int = Field(
        default=15,
        ge=1,
        le=365,
        description="重要性时间衰减半衰期 (天)",
    )
    emotional_resonance_bonus: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="情绪共振加成系数 (匹配时 score 增幅倍数)",
    )
    graph_bfs_depth: int = Field(
        default=2,
        ge=1,
        le=5,
        description="图激活扩散 BFS 最大深度",
    )
    graph_activation_decay: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="图激活扩散逐层衰减系数",
    )
    graph_edge_decay_factor: float = Field(
        default=0.95,
        ge=0.5,
        le=1.0,
        description="图边权重每日衰减因子 (weight * factor**days)",
    )
    graph_edge_weight_floor: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="图边权重衰减最低值",
    )
    graph_maintenance_decay_min: float = Field(
        default=0.80,
        ge=0.0,
        le=1.0,
        description="图维护衰减最低系数 (encounter_count 低时)",
    )
    graph_maintenance_decay_max: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="图维护衰减最高系数 (encounter_count 高时)",
    )
    graph_maintenance_relation_bonus: float = Field(
        default=0.05,
        ge=0.0,
        le=0.5,
        description="语义边在图维护中的额外保护系数",
    )
    graph_dead_edge_threshold: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="图边存活阈值 (低于此值被清理)",
    )
    graph_cooccurrence_noise_penalty: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="无 relation 的共现边扩散权重惩罚因子",
    )
    graph_diffusion_max_results: int = Field(
        default=4,
        ge=1,
        le=20,
        description="图扩散最大返回结果数",
    )
    surfaced_threshold: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="图召回「显性」激活阈值",
    )
    silent_threshold: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="图召回「隐性」激活阈值",
    )
    embedding_cosine_threshold: float = Field(
        default=0.3,
        ge=-1.0,
        le=1.0,
        description="向量余弦相似度最低阈值",
    )
    session_importance_boost: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="当前 session 条目重要性额外加成",
    )
    fusion_fallback_channel_weights: list[float] = Field(
        default=[1.0, 1.0, 0.4],
        description="融合兜底权重 (当 channel_weights 未配置时)",
    )


class LLMProviderSettings(BaseModel):
    """LLM 服务商配置 (对应 config.json analysis.post_analysis)"""

    api_key: str = Field(
        default="",
        description="API 密钥",
    )
    base_url: str = Field(
        default="https://api.openai.com/v1",
        description="API 基础地址",
    )
    model: str = Field(
        default="gpt-4o-mini",
        description="模型名称",
    )
    source: str = Field(
        default="astrbot",
        description="配置来源: astrbot | custom",
    )
    astrbot_source_id: str = Field(
        default="",
        description="AstrBot 提供商 ID (留空自动匹配)",
    )


class EmbeddingProviderSettings(BaseModel):
    """Embedding 服务商配置 (对应 config.json analysis.retrieval embedding 部分)"""

    api_key: str = Field(
        default="",
        description="API 密钥 (留空则复用 LLM 凭据或 OPENAI_API_KEY)",
    )
    base_url: str = Field(
        default="https://api.openai.com/v1",
        description="API 基础地址",
    )
    model: str = Field(
        default="text-embedding-3-small",
        description="Embedding 模型名",
    )
    source: str = Field(
        default="astrbot",
        description="配置来源: astrbot | custom",
    )
    astrbot_source_id: str = Field(
        default="",
        description="AstrBot 提供商 ID (留空自动匹配)",
    )


class AnalysisSettings(BaseModel):
    """事后分析/蒸馏配置 (对应 config.json analysis.post_analysis + core.py 蒸馏常量)"""

    llm: LLMProviderSettings = Field(
        default_factory=LLMProviderSettings,
        description="LLM 服务商配置",
    )
    max_turns: int = Field(
        default=30,
        ge=5,
        le=200,
        description="每 N 轮触发蒸馏",
    )
    max_minutes: int = Field(
        default=60,
        ge=5,
        le=1440,
        description="每 N 分钟触发蒸馏",
    )
    dangling_expire_turns: int = Field(
        default=15,
        ge=3,
        le=100,
        description="悬案过期轮数 (超过此轮无人提起则归档)",
    )
    personality_type: str = Field(
        default="default",
        description="人格风格: default | cute | professional",
    )
    distill_min_turns: int = Field(
        default=6,
        ge=2,
        le=50,
        description="蒸馏最少对话轮数 (不足则跳过)",
    )
    permanent_fact_max: int = Field(
        default=3,
        ge=1,
        le=20,
        description="永久事实每轮写入上限",
    )
    transient_fact_max: int = Field(
        default=5,
        ge=1,
        le=20,
        description="临时事实每轮写入上限",
    )
    max_snapshot_lines: int = Field(
        default=30,
        ge=5,
        le=200,
        description="蒸馏快照最大行数",
    )
    dangling_fallback_importance: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="悬案归档时默认重要性",
    )
    archived_dangling_importance: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="归档悬案写入 importance",
    )
    distill_entry_importance: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="蒸馏摘要写入 importance",
    )


class EmotionalWordsSettings(BaseModel):
    """情绪/停用/琐事词表 (替换 core.py 中的硬编码列表)"""

    emotional_words: list[str] = Field(
        default=[
            "累", "烦", "难过", "开心", "怕", "为什么", "怎么办",
            "焦虑", "迷茫", "失望", "生气", "感动", "孤独", "压力",
            "崩溃", "痛苦", "幸福", "委屈", "愤怒", "绝望", "不安",
            "愧疚", "后悔", "感激", "羡慕", "厌倦", "疲惫", "心累",
            "纠结", "无助", "温暖", "讽刺", "荒谬", "心碎",
            "气死", "受不了", "撑不住", "扛不住", "熬不下去",
            "舍不得", "放不下", "不甘心",
        ],
        description="情绪感知词表 (用于 save_turn 重要性计算与蒸馏分析)",
    )
    trivial_markers: list[str] = Field(
        default=[
            "吃", "喝", "睡", "饭", "菜", "外卖", "快递", "天气",
            "价格", "多少钱", "购物", "买了", "电影", "追剧",
            "洗澡", "起床", "睡觉", "游戏",
        ],
        description="琐事标记词表 (用于关键词提取时过滤)",
    )
    stop_words: list[str] = Field(
        default=[
            # 时间副词
            "最近", "今天", "明天", "昨天", "前天", "刚才", "已经", "正在",
            "将要", "即将", "马上", "立刻", "刚刚", "忽然", "曾经", "往往",
            # 代词/指示词
            "什么", "怎么", "为什么", "哪个", "哪些", "谁", "这个", "那个",
            "这些", "那些", "哪里", "这里", "那里", "如何", "何时",
            # 虚词
            "一个", "没有", "不是", "可以", "就是", "还是", "但是", "而且",
            "因为", "所以", "虽然", "如果", "然后", "不过", "一定",
            "一些", "有点", "一下", "非常",
            "是否", "能够", "应该", "必须", "好像", "真是",
            # 语气词
            "好吧", "好了", "是的", "没错", "对了",
            # 纯情绪感知词
            "觉得", "感觉", "认为",
        ],
        description="停用词表 (用于关键词提取时过滤掉无区分度的词)",
    )


class TimeWordSettings(BaseModel):
    """时间词映射 (替换 retrieval.py 中的 _TIME_WORDS 硬编码字典)"""

    time_words: dict[str, list[int]] = Field(
        default={
            "今天": [0, 0],
            "今日": [0, 0],
            "昨天": [1, 1],
            "昨日": [1, 1],
            "前天": [2, 2],
            "最近": [0, 7],
            "近来": [0, 7],
            "近期": [0, 7],
            "上周": [7, 13],
            "上星期": [7, 13],
        },
        description="时间词 → [最小天数, 最大天数], 用于时间范围过滤",
    )


class OppositeRelationSettings(BaseModel):
    """矛盾关系映射 (替换 retrieval.py 中的 _OPPOSITE_RELATIONS 硬编码字典)"""

    opposite_relations: dict[str, str] = Field(
        default={
            "喜欢": "讨厌",
            "讨厌": "喜欢",
            "使用": "放弃",
            "放弃": "使用",
            "朋友": "敌人",
            "敌人": "朋友",
        },
        description="关系 → 对立关系, 图谱矛盾检测用",
    )


class InverseRelationSettings(BaseModel):
    """反向关系映射 (替换 analysis.py 中的 _INVERSE_RELATIONS 硬编码字典)"""

    inverse_relations: dict[str, str] = Field(
        default={
            "朋友": "朋友",
            "同事": "同事",
            "喜欢": "被喜欢",
            "喜欢玩": "被喜欢玩",
            "讨论过": "被讨论过",
            "讨厌": "被讨厌",
            "居住": "居住地于",
            "属于": "包含",
            "对立": "对立",
            "同类": "同类",
            "使用": "被使用",
            "提及": "被提及",
            "养了": "主人是",
        },
        description="关系 → 反向关系, 图谱双向边插入用",
    )


class StorageMaintenanceSettings(BaseModel):
    """存储维护配置 (对应 session.py WAL + analysis.py 清理)"""

    wal_checkpoint_size_kb: int = Field(
        default=200,
        ge=50,
        le=10240,
        description="WAL 文件超过此大小时强制执行 TRUNCATE checkpoint (KB)",
    )
    keep_rule_summary: int = Field(
        default=10,
        ge=1,
        le=100,
        description="保留的低重要性记忆条数 (importance=0.3 的规则摘要)",
    )
    max_trait_count: int = Field(
        default=30,
        ge=10,
        le=200,
        description="身份特质上限 (超出后按 s*2+c 排序截断)",
    )
    cognitive_distill_importance_floor: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="cognitive_distill 时间通道查询 importance 下限",
    )
    cleanup_importance_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="低重要性清理阈值 (cognitive_distill 中 <= 此值的条目视为低重要性)",
    )


class LoggingSettings(BaseModel):
    """日志配置 (对应 config.json output_log + debug)"""

    log_level: str = Field(
        default="info",
        description="日志级别: debug | info | warning",
    )
    output_log_enabled: bool = Field(
        default=True,
        description="启用输出日志 (JSONL 格式)",
    )
    output_log_max_size_mb: int = Field(
        default=5,
        ge=1,
        le=100,
        description="日志文件最大体积 (MB)",
    )
    output_log_path: str = Field(
        default="rcms_output.jsonl",
        description="日志文件路径",
    )


# ==============================================================
# config.json 加载工具
# ==============================================================

_config_logger = logging.getLogger("rcms.config")

_config_json_mapping: list[tuple[tuple[str, ...], tuple[str, ...]]] = [
    # analysis.retrieval → retrieval
    (("analysis", "retrieval", "embedding_enabled"), ("retrieval", "embedding_enabled")),
    (("analysis", "retrieval", "total_cap"), ("retrieval", "total_cap")),
    (("analysis", "retrieval", "channel_min"), ("retrieval", "channel_min")),
    (("analysis", "retrieval", "channel_weights"), ("retrieval", "channel_weights")),
    (("analysis", "retrieval", "time_decay_halflife"), ("retrieval", "time_decay_halflife")),
    (("analysis", "retrieval", "emotional_resonance_bonus"), ("retrieval", "emotional_resonance_bonus")),
    # analysis.retrieval → embedding
    (("analysis", "retrieval", "source"), ("embedding", "source")),
    (("analysis", "retrieval", "astrbot_source_id"), ("embedding", "astrbot_source_id")),
    (("analysis", "retrieval", "custom_api_key"), ("embedding", "api_key")),
    (("analysis", "retrieval", "custom_base_url"), ("embedding", "base_url")),
    (("analysis", "retrieval", "custom_model"), ("embedding", "model")),
    # analysis.post_analysis → analysis / analysis.llm
    (("analysis", "post_analysis", "source"), ("analysis", "llm", "source")),
    (("analysis", "post_analysis", "astrbot_source_id"), ("analysis", "llm", "astrbot_source_id")),
    (("analysis", "post_analysis", "custom_api_key"), ("analysis", "llm", "api_key")),
    (("analysis", "post_analysis", "custom_base_url"), ("analysis", "llm", "base_url")),
    (("analysis", "post_analysis", "custom_model"), ("analysis", "llm", "model")),
    (("analysis", "post_analysis", "max_turns"), ("analysis", "max_turns")),
    (("analysis", "post_analysis", "max_minutes"), ("analysis", "max_minutes")),
    (("analysis", "post_analysis", "dangling_expire_turns"), ("analysis", "dangling_expire_turns")),
    (("analysis", "post_analysis", "personality_type"), ("analysis", "personality_type")),
    # output_log → logging
    (("output_log", "enabled"), ("logging", "output_log_enabled")),
    (("output_log", "max_size_mb"), ("logging", "output_log_max_size_mb")),
    (("output_log", "path"), ("logging", "output_log_path")),
    # debug → logging
    (("debug", "log_level"), ("logging", "log_level")),
]


def _deep_get(obj: Any, path: tuple[str, ...]) -> Any:
    """从嵌套字典/对象按路径安全取值。"""
    current: Any = obj
    for key in path:
        if isinstance(current, dict):
            current = current.get(key)
        else:
            current = getattr(current, key, None)
        if current is None:
            return None
    return current


def _deep_set(obj: Any, path: tuple[str, ...], value: Any) -> None:
    """在嵌套对象上按路径设值（路径除叶节点外均为 BaseModel 属性）。"""
    current: Any = obj
    for key in path[:-1]:
        current = getattr(current, key)
    setattr(current, path[-1], value)


# ==============================================================
# 顶层 Settings — 单入口
# ==============================================================


class Settings(BaseSettings):
    """RCMS 全局配置 — 单入口, 所有硬编码集中于此

    环境变量覆盖规则:
      - 前缀 RCMS_ (自动大写)
      - 多级字段用 __ 连接 (注意: 字段名自动全大写)
      例如:
        RCMS_STORAGE__DB_PATH=/data/rcms/memory.db
        RCMS_RETRIEVAL__TOTAL_CAP=10
        RCMS_EMOTIONAL_WORDS__EMOTIONAL_WORDS='["开心","难过"]'
    """

    storage: StorageSettings = Field(
        default_factory=StorageSettings,
        description="数据库存储配置",
    )
    retrieval: RetrievalSettings = Field(
        default_factory=RetrievalSettings,
        description="三通道召回配置 (含图参数)",
    )
    analysis: AnalysisSettings = Field(
        default_factory=AnalysisSettings,
        description="事后分析/蒸馏配置",
    )
    embedding: EmbeddingProviderSettings = Field(
        default_factory=EmbeddingProviderSettings,
        description="Embedding 服务商配置",
    )
    emotional_words: EmotionalWordsSettings = Field(
        default_factory=EmotionalWordsSettings,
        description="情绪词/琐事标记/停用词表",
    )
    time_words: TimeWordSettings = Field(
        default_factory=TimeWordSettings,
        description="时间词映射字典",
    )
    opposite_relations: OppositeRelationSettings = Field(
        default_factory=OppositeRelationSettings,
        description="矛盾关系映射",
    )
    inverse_relations: InverseRelationSettings = Field(
        default_factory=InverseRelationSettings,
        description="反向关系映射",
    )
    maintenance: StorageMaintenanceSettings = Field(
        default_factory=StorageMaintenanceSettings,
        description="存储维护配置 (WAL/重要性/容量上限)",
    )
    logging: LoggingSettings = Field(
        default_factory=LoggingSettings,
        description="日志配置",
    )

    model_config = {
        "env_prefix": "RCMS_",
        "env_nested_delimiter": "__",
    }

    @classmethod
    def load_from_config_json(cls, config_path: str = "config.json") -> "Settings":
        """从 config.json 加载配置，环境变量仍优先于 JSON 值。

        优先级: 代码默认值 < config.json < 环境变量 (RCMS_ 前缀)

        Args:
            config_path: config.json 文件路径，默认 "config.json"。

        Returns:
            Settings 实例，已合并 config.json 和 RCMS_ 环境变量。
        """
        # 纯默认值（无环境变量，无 config.json）
        defaults: Settings = cls.model_construct()

        # 带环境变量覆盖的 Settings
        settings: Settings = cls()

        # 尝试读取 config.json
        if not os.path.exists(config_path):
            _config_logger.info(
                "config.json('%s') 不存在，纯环境变量模式", config_path
            )
            return settings

        try:
            with open(config_path, encoding="utf-8") as f:
                raw: dict[str, Any] = json.load(f)
        except json.JSONDecodeError as e:
            _config_logger.warning(
                "config.json('%s') 解析失败: %s", config_path, e
            )
            return settings
        except PermissionError as e:
            _config_logger.warning(
                "config.json('%s') 无权限读取: %s", config_path, e
            )
            return settings

        # 根据映射表覆盖字段（仅当未被环境变量覆盖时）
        count: int = 0
        for json_path, settings_path in _config_json_mapping:
            json_val: Any = _deep_get(raw, json_path)
            if json_val is None:
                continue

            current_val: Any = _deep_get(settings, settings_path)
            default_val: Any = _deep_get(defaults, settings_path)

            if current_val == default_val:
                _deep_set(settings, settings_path, json_val)
                count += 1

        _config_logger.info(
            "config.json('%s') 加载完成: 覆盖 %d/%d 个字段",
            config_path,
            count,
            len(_config_json_mapping),
        )
        return settings


# ==============================================================
# 便捷访问 — 可调用 settings() 获取单例
# ==============================================================

_settings: Settings | None = None


def get_settings(config_path: Optional[str] = None) -> Settings:
    """返回 Settings 单例 (懒加载)，可选从 config.json 加载。

    Args:
        config_path: config.json 文件路径。为 None 时仅使用环境变量。

    Returns:
        Settings 单例。
    """
    global _settings
    if _settings is None:
        _settings = (
            Settings.load_from_config_json(config_path)
            if config_path
            else Settings()
        )
    return _settings

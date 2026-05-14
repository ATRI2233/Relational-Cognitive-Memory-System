"""AstrBot LLM 后端 — 读取 AstrBot 配置，调用 OpenAI 兼容 API"""
import json
import os

from openai import AsyncOpenAI

from backends import LLMBackend


class AstrBotBackend(LLMBackend):
    """从 AstrBot 配置文件读取 provider 信息，创建 OpenAI 兼容客户端"""

    def __init__(self, config_path: str = None):
        """
        Args:
            config_path: AstrBot cmd_config.json 路径，默认自动查找
        """
        config_path = config_path or self._find_config()
        self.config = self._load_config(config_path)
        self.provider_source, self.provider_cfg = self._resolve_active_provider()

        self.client = AsyncOpenAI(
            api_key=self.provider_source["key"][0],
            base_url=self.provider_source["api_base"],
            timeout=self.provider_source.get("timeout", 120),
        )
        self.model = self.provider_cfg["model"]

    @staticmethod
    def _find_config() -> str:
        candidates = [
            os.path.expanduser("~/.astrbot/data/cmd_config.json"),
            os.path.join(os.getcwd(), "data", "cmd_config.json"),
        ]
        for p in candidates:
            if os.path.exists(p):
                return p
        raise FileNotFoundError(f"未找到 AstrBot 配置，尝试过: {candidates}")

    @staticmethod
    def _load_config(path: str) -> dict:
        with open(path, encoding="utf-8-sig") as f:
            return json.load(f)

    def _resolve_active_provider(self):
        """解析默认使用的 provider（source + model 配置）"""
        settings = self.config.get("provider_settings", {})
        default_id = settings.get("default_provider_id", "")

        providers = self.config.get("provider", [])
        sources = {s["id"]: s for s in self.config.get("provider_sources", [])}

        # 找默认 provider，没有则用第一个启用的
        target = None
        for p in providers:
            if p.get("enable", False):
                if p["id"] == default_id or not target:
                    target = p

        if not target:
            raise ValueError("AstrBot 配置中没有启用的 provider")

        source_id = target.get("provider_source_id", "")
        source = sources.get(source_id)
        if not source:
            raise ValueError(f"未找到 provider_source: {source_id}")

        return source, target

    async def generate(self, prompt: str, **kwargs) -> str:
        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content or ""

    async def close(self):
        await self.client.close()

    def get_model_name(self) -> str:
        return self.model

"""直接 OpenAI 兼容 API 后端（不依赖 AstrBot）"""
from openai import AsyncOpenAI

from backends import LLMBackend


class OpenAIBackend(LLMBackend):
    """直接配置 API key 和 base_url 使用任意 OpenAI 兼容接口"""

    def __init__(self, api_key: str, base_url: str = "https://api.openai.com/v1", model: str = "gpt-4o"):
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    async def generate(self, prompt: str, **kwargs) -> str:
        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content or ""

    async def close(self):
        await self.client.close()

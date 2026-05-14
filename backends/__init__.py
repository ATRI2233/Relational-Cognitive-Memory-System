"""LLM 后端抽象层 — 支持接入任意 Agent"""
from abc import ABC, abstractmethod


class LLMBackend(ABC):
    """LLM 后端接口，RCMS 通过此接口调用任意 LLM"""

    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        """根据 prompt 生成回复文本"""
        ...

    @abstractmethod
    async def close(self):
        """释放资源"""
        ...


class MockBackend(LLMBackend):
    """测试用 Mock 后端"""

    def __init__(self, reply: str = "嗯，我在听，你继续说。"):
        self.reply = reply

    async def generate(self, prompt: str, **kwargs) -> str:
        print("\n" + "="*60)
        print("【Generated Prompt】")
        print(prompt)
        print("="*60)
        return self.reply

    async def close(self):
        pass


__all__ = ["LLMBackend", "MockBackend"]

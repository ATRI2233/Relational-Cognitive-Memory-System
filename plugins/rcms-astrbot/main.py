"""
RCMS × AstrBot 适配器（分发源）

将此目录（或 symlink/junction）放到 AstrBot 的 plugins/ 目录即可。
"""
import os
import sys

_self = os.path.dirname(os.path.abspath(__file__))
if _self not in sys.path:
    sys.path.insert(0, _self)

from astrbot.api import star
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.provider import LLMResponse, ProviderRequest
from astrbot.core import logger

from minimal_rcms import MinimalRCMS


class RcmsPlugin(star.Star):
    def __init__(self, context: star.Context, config: dict = None) -> None:
        super().__init__(context)
        db_path = os.path.join(_self, "rcms_memory.db")
        self.rcms = MinimalRCMS(db_path=db_path)
        logger.info(f"RCMS: 数据库已初始化 ({db_path})")
        self.user_id = "astrbot_user"

    async def initialize(self) -> None:
        logger.info("RCMS: 插件已加载")

    async def terminate(self) -> None:
        self.rcms.close()
        logger.info("RCMS: 插件已卸载")

    @filter.on_llm_request()
    async def on_llm_request(
        self, event: AstrMessageEvent, req: ProviderRequest
    ) -> None:
        """在 LLM 请求前注入 RCMS 上下文到 system_prompt"""
        user_input = event.message_str or req.prompt or ""
        if not user_input.strip():
            return

        stance = self.rcms.detect_stance(user_input)
        memories = self.rcms.retrieve_memories(self.user_id, user_input, stance)

        memory_lines = []
        if memories:
            for m in memories:
                memory_lines.append(f"  - 你记得{m[0]}")
        else:
            memory_lines.append("  - 没什么特别的联想")

        atmosphere = (
            "你现在随手回消息，不太走心，像朋友边刷手机边打字。"
            if stance == "casual"
            else "你现在认真听他说话，可以想起以前的事，可以共情。"
        )

        rcms_context = f"""
【RCMS 关系氛围】
{atmosphere}

【RCMS 相关记忆】
{chr(10).join(memory_lines)}
"""
        req.system_prompt += rcms_context
        event.set_extra("rcms_stance", stance)
        event.set_extra("rcms_user_input", user_input)

    @filter.on_llm_response()
    async def on_llm_response(
        self, event: AstrMessageEvent, resp: LLMResponse
    ) -> None:
        """在 LLM 响应后记录到 RCMS"""
        user_input = event.get_extra("rcms_user_input", "")
        stance = event.get_extra("rcms_stance", "casual")
        reply = resp.completion_text or ""

        if not user_input or not reply:
            return

        self.rcms.save_turn(event.unified_msg_origin, user_input, reply, stance)
        logger.debug(f"RCMS: 已记录 [{stance}]")

import re
import logging
from datetime import datetime

logger = logging.getLogger("rcms")


class UtilsMixin:
    """文本工具、历史、core_veto"""

    def _get_history(self, session_id: str, limit: int = 3):
        rows = self.conn.execute("""
            SELECT role, content FROM chat_history
            WHERE session_id = ? ORDER BY created_at DESC LIMIT ?
        """, (session_id, limit)).fetchall()
        rows.reverse()
        return rows

    @staticmethod
    def _chinese_bigrams(text: str) -> set:
        chars = re.findall(r'[一-鿿]', text)
        return {''.join(chars[i:i + 2]) for i in range(len(chars) - 1)}

    @staticmethod
    def _precise_kw_match(text: str, kw: str) -> bool:
        return kw in text

    @staticmethod
    def _score_markers(text: str, markers: list, per_hit: float = 0.3) -> float:
        count = sum(1 for m in markers if m in text)
        return min(count * per_hit, 1.0)

    def _core_veto(self, prompt: str) -> str:
        for s in ['你应该', '你必须', '我教你', '听我说', '你这样不对']:
            if s in prompt:
                prompt = prompt.replace(s, '或许可以试试')
                break
        return prompt

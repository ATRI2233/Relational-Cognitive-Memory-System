"""PromptBuilder — prompt 模板构建服务。

从 analysis.py _build_distill_prompt 提取。
职责：使用 prompts.json 模板，注入变量，拼接完整的 LLM prompt。
"""
from __future__ import annotations

import copy
import json
import logging
from typing import Any, Optional


logger = logging.getLogger("rcms")


def _safe_format(template: str, **kwargs: str) -> str:
    """Format a template string, returning it unchanged on KeyError.

    Prevents crashes when user-provided or config-provided template strings
    contain unexpected ``{...}`` placeholders.

    Args:
        template: The template string with ``{name}`` placeholders.
        **kwargs: The keyword arguments to substitute.

    Returns:
        Formatted string on success, or the original template on KeyError.
    """
    try:
        return template.format(**kwargs)
    except KeyError:
        logger.warning(
            "Template has unexpected placeholder: %s (kwargs: %s)",
            template,
            set(kwargs),
        )
        return template


class PromptBuilder:
    """Prompt 构建服务

    处理三类 prompt 的构建：
    1. 蒸馏分析 prompt（_build_distill_prompt）
    2. 叙事上下文 prompt（narrative_context）
    3. 压缩 prompt（prompt_compressor）
    """

    def __init__(self, templates: dict[str, Any],
                 channel_labels: Optional[dict[str, str]] = None):
        """初始化

        Args:
            templates: prompts.json 的完整内容（由调用方加载后传入）
            channel_labels: 通道标签映射，覆盖默认值

        Raises:
            ValueError: templates 为空时抛出
        """
        if not templates:
            raise ValueError("templates 不能为空")
        self._templates = templates
        self._channel_labels = channel_labels or {
            "recent": "时间·重要性",
            "resonance": "语义检索",
            "skeleton": "图谱关联",
        }

    # ────────────────────────────────────────────────────────────────
    # 蒸馏分析 prompt
    # ────────────────────────────────────────────────────────────────

    @staticmethod
    def _build_lt_hint(long_term: dict) -> str:
        """从长期记忆字典构建已知信息提示

        Args:
            long_term: 长期记忆字典，可能包含 identity_traits / preferences /
                       self_identity / boundaries

        Returns:
            多行字符串，每行一个已知信息类别；无数据时返回空字符串
        """
        if not long_term:
            return ""
        lt_hint = ""

        if long_term.get("identity_traits"):
            lt_hint += (
                "\n已知特质: "
                + json.dumps(long_term["identity_traits"], ensure_ascii=False)
            )

        if long_term.get("preferences"):
            lt_hint += (
                "\n已知喜好: "
                + json.dumps(long_term["preferences"], ensure_ascii=False)
            )

        if long_term.get("self_identity"):
            lt_hint += (
                "\n自我认同: "
                + json.dumps(long_term["self_identity"], ensure_ascii=False)
            )

        if long_term.get("boundaries"):
            lt_hint += (
                "\n已知雷区: "
                + json.dumps(long_term["boundaries"], ensure_ascii=False)
            )

        return lt_hint

    def build_distill_prompt(
        self,
        snapshot_text: str,
        long_term: dict,
        persona_name: str = "Bot",
        personality_style: str = "",
        is_group: bool = False,
        personality_type: str = "default",
    ) -> str:
        """构建蒸馏分析 prompt

        对应 analysis.py _build_distill_prompt（行 424-489）。
        支持三种人格风格（default / cute / professional）和私聊/群聊双模板。

        Args:
            snapshot_text: 对话快照
            long_term: 长期语境字典
            persona_name: 人格名称
            personality_style: 人格风格描述
            is_group: 是否群聊
            personality_type: 人格类型（default / cute / professional）

        Returns:
            完整的 prompt 字符串

        Raises:
            ValueError: 模板中缺少 distill_prompt 段
        """
        distill = self._templates.get("distill_prompt")
        if not distill:
            raise ValueError("模板中缺少 distill_prompt 段")

        lt_hint = self._build_lt_hint(long_term)

        # ── 人格变体与模式选择 ──
        ptype_key = (
            personality_type
            if personality_type in ("cute", "professional")
            else "default"
        )
        mode_key = "group" if is_group else "private"
        tpl = (
            distill.get(ptype_key, {}).get(mode_key, {})
            or distill.get("default", {}).get("private", {})
        )

        preamble = _safe_format(tpl.get("preamble", ""), persona_name=persona_name)
        content_instruction = _safe_format(
            tpl.get("content_instruction", ""), persona_name=persona_name
        )
        first_stage = tpl.get("first_stage", "")
        extra_rules = tpl.get("extra_rules", "")

        # ── 输出格式模板 ──
        intro = distill.get("output_intro", "").replace(
            "{first_stage}", first_stage
        )
        schema = copy.deepcopy(distill.get("output_schema", {}))
        rules = list(distill.get("output_rules", []))
        footer = distill.get("output_footer", "只输出 JSON。")
        lt_hint_label = distill.get(
            "lt_hint_label", "已了解的信息（后续字段需与之去重）："
        )

        # 群聊才保留 participants 字段
        if not is_group:
            analysis_schema = schema.get("analysis", {})
            if isinstance(analysis_schema, dict):
                analysis_schema.pop("participants", None)

        # schema -> 格式化的 JSON 字符串
        schema_str = json.dumps(schema, ensure_ascii=False, indent=2)

        # 替换占位符
        schema_str = schema_str.replace(
            "{content_instruction}", content_instruction
        )
        schema_str = schema_str.replace("{persona_name}", persona_name)

        # 组装规则
        rules_str = "\n".join(
            r.replace("{persona_name}", persona_name) for r in rules
        )
        if extra_rules:
            rules_str += "\n" + extra_rules

        return f"""[SYSTEM]
{preamble}

角色风格：
{personality_style or '（无特殊设定）'}

[USER]
对话快照：
{snapshot_text}

{lt_hint_label}
{lt_hint}

{intro}{schema_str}

规则：
{rules_str}
{footer}"""

    # ────────────────────────────────────────────────────────────────
    # 叙事上下文 prompt
    # ────────────────────────────────────────────────────────────────

    def _build_memory_section(
        self,
        memories: list,
        nc: dict,
        display_order: list[str],
        channel_labels: dict[str, str],
    ) -> str:
        """构建三通道记忆段落

        Args:
            memories: [(content, tag), ...] 回忆列表
            nc: narrative_context 模板段
            display_order: 通道展示顺序
            channel_labels: 通道标签映射

        Returns:
            格式化后的记忆段落，无记忆时返回空字符串
        """
        grouped: dict[str, list[str]] = {}
        for content, tag in memories:
            grouped.setdefault(tag, []).append(content)

        ch_lines: list[str] = []
        for key in display_order:
            items = grouped.get(key)
            if items:
                label = channel_labels.get(key, key)
                ch_lines.append(
                    f"【{label}】\n"
                    + "\n".join(f"  · {c}" for c in items)
                )

        if not ch_lines:
            return ""

        return (
            nc.get("memories_title", "相关记忆")
            + ":\n"
            + "\n\n".join(ch_lines)
        )

    def _build_profile_section(
        self, long_term: dict, nc: dict
    ) -> str:
        """构建用户画像段落（特质 + 小细节）

        Args:
            long_term: 长期语境字典
            nc: narrative_context 模板段

        Returns:
            格式化后的用户画像段落
        """
        if not long_term:
            return ""

        remaining_traits_tpl = nc.get(
            "remaining_traits_template", "及其他 {count} 条特质"
        )
        quirk_label = nc.get("quirk_prefix", "小细节小习惯")

        trait_details = sorted(
            long_term.get("trait_details", []),
            key=lambda x: x.get("strength", 0),
            reverse=True,
        )

        all_traits = [
            td
            for td in trait_details
            if not td["text"].startswith("[小细节小习惯]")
        ]

        profile_lines: list[str] = []
        max_show = 5
        for td in all_traits[:max_show]:
            strength = td.get("strength", 0)
            prefix = (
                ""
                if strength >= 5
                else "↘ "
                if strength <= 2
                else "· "
            )
            profile_lines.append(f"{prefix}{td['text']}")

        remaining = len(all_traits) - max_show
        if remaining > 0:
            profile_lines.append(
                _safe_format(remaining_traits_tpl, count=remaining)
            )

        quirks = [
            (
                td.get("strength", 0),
                td["text"].replace("[小细节小习惯] ", ""),
            )
            for td in trait_details
            if td["text"].startswith("[小细节小习惯]")
        ]
        quirks.sort(key=lambda x: x[0], reverse=True)
        if quirks:
            q_mark = "↘ " if any(q[0] <= 2 for q in quirks) else ""
            profile_lines.append(
                f"{q_mark}{quirk_label}: "
                + "、".join(q[1] for q in quirks[:2])
            )

        if not profile_lines:
            return ""

        return (
            nc.get("profile_title", "他是什么样的")
            + ":\n"
            + "\n".join(f"  · {t}" for t in profile_lines)
        )

    def _build_struct_section(
        self, long_term: dict, nc: dict
    ) -> str:
        """构建结构化画像段落（偏好 / 自我认同 / 雷区）

        Args:
            long_term: 长期语境字典
            nc: narrative_context 模板段

        Returns:
            格式化后的结构化画像段落
        """
        if not long_term:
            return ""

        struct_lines: list[str] = []

        prefs = long_term.get("preferences", {})
        if prefs.get("likes"):
            struct_lines.append(
                nc.get("pref_likes", "喜好")
                + ": "
                + "、".join(prefs["likes"][:5])
            )
        if prefs.get("dislikes"):
            struct_lines.append(
                nc.get("pref_dislikes", "不喜欢")
                + ": "
                + "、".join(prefs["dislikes"][:3])
            )

        si = long_term.get("self_identity", [])
        if si:
            struct_lines.append(
                nc.get("self_identity", "自我认同")
                + ": "
                + "、".join(si[:3])
            )

        bounds = long_term.get("boundaries", [])
        if bounds:
            bound_texts = [
                b["description"] if isinstance(b, dict) else str(b)
                for b in bounds[:3]
            ]
            struct_lines.append(
                nc.get("boundaries", "雷区")
                + ": "
                + "、".join(bound_texts)
            )

        if not struct_lines:
            return ""

        return (
            nc.get("struct_title", "结构化画像")
            + ":\n"
            + "\n".join(f"  · {s}" for s in struct_lines)
        )

    def _build_context_section(
        self, long_term: dict, nc: dict
    ) -> str:
        """构建共同语境段落（梗 / 实体）

        Args:
            long_term: 长期语境字典
            nc: narrative_context 模板段

        Returns:
            格式化后的共同语境段落
        """
        if not long_term:
            return ""

        context_title = nc.get("context_title", "共同语境")
        joke_label = nc.get("joke_label", "梗")
        entity_label_tpl = nc.get(
            "entity_label_template", "他提过的{type}"
        )

        ctx_lines: list[str] = []

        shared = long_term.get("shared_contexts", [])
        jokes = [
            s.replace("[梗] ", "")
            for s in shared
            if s.startswith("[梗]")
        ][:2]
        other = [
            s for s in shared if not s.startswith("[梗]")
        ][:2]
        ctx_lines.extend(f"{joke_label}: {j}" for j in jokes)
        ctx_lines.extend(other)

        entities = long_term.get("entities", [])
        if entities:
            grouped_ents: dict[str, list[str]] = {}
            seen_names: set[str] = set()
            for e in entities:
                name = e.get("name", "")
                if not name or name in seen_names:
                    continue
                seen_names.add(name)
                etype = e.get("type", "auto")
                if etype not in grouped_ents:
                    grouped_ents[etype] = []
                tag = (
                    f" ({e.get('relation', '')})"
                    if e.get("relation")
                    else ""
                )
                grouped_ents[etype].append(f"{name}{tag}")

            type_labels = {
                "person": "人",
                "place": "地方",
                "concept": "概念",
                "activity": "活动",
                "auto": "相关",
            }
            for etype, items in grouped_ents.items():
                label = type_labels.get(etype, etype)
                ctx_lines.append(
                _safe_format(entity_label_tpl, type=label)
                    + ": "
                    + "、".join(items[:4])
                )

        if not ctx_lines:
            return ""

        return (
            context_title
            + ":\n"
            + "\n".join(f"  · {c}" for c in ctx_lines)
        )

    def _build_dangling_section(
        self,
        dangling: Any,
        turn_count: int,
        nc: dict,
    ) -> str:
        """构建未完成话题段落

        Args:
            dangling: 已解析的悬案数据（dict 含 threads/turn，或 list）
            turn_count: 当前对话轮次
            nc: narrative_context 模板段

        Returns:
            格式化后的未完成话题段落
        """
        if not dangling:
            return ""

        stale_prefix = nc.get("dangling_stale_prefix", "↘ ")
        dangling_label = nc.get("dangling_label", "未完成")

        if isinstance(dangling, dict):
            dt_list = dangling.get("threads", [])
            since_turn = dangling.get("turn", 0)
            if dt_list and turn_count - since_turn <= 10:
                stale = turn_count - since_turn > 5
                prefix = stale_prefix if stale else ""
                dangling_display = prefix + "、".join(dt_list[:3])
                return f"{dangling_label}: {dangling_display}"
        elif isinstance(dangling, list) and dangling:
            return (
                f"{dangling_label}: " + "、".join(dangling[:3])
            )

        return ""

    def build_context_prompt(
        self,
        stance: str,
        memories: list,
        long_term: dict,
        session_id: str = "",
        user_id: str = "",
        turn_count: int = 0,
        dangling: Any = "",
        graph_paths: Optional[list[str]] = None,
        warmup_text: str = "",
    ) -> str:
        """构建叙事上下文 prompt

        对应 context.py narrative_context（行 51-202）。
        session_warmup 由调用方预计算后通过 warmup_text 传入。

        Args:
            stance: 对话立场（当前未使用，保留接口兼容性）
            memories: [(content, tag), ...] 三通道回忆列表
            long_term: 长期语境字典
            session_id: 会话标识
            user_id: 用户标识
            turn_count: 当前对话轮次
            dangling: 已解析的悬案数据——dict（含 threads/turn）或 list（纯字符串列表）；
                      调用方应在传入前完成 json.loads 解析
            graph_paths: 图谱关系链描述列表
            warmup_text: 由调用方预计算的 session_warmup 文本（来自前一 session 的悬案）

        Returns:
            完整的 prompt 字符串

        Raises:
            ValueError: 模板中缺少 narrative_context 段
        """
        nc = self._templates.get("narrative_context")
        if not nc:
            raise ValueError("模板中缺少 narrative_context 段")

        channel_labels = self._templates.get(
            "channel_labels", self._channel_labels
        )
        display_order = self._templates.get(
            "memories_display_order", ["resonance", "skeleton", "recent"]
        )

        parts: list[str] = []

        # ── 三通道记忆 ──
        if memories:
            mem_block = self._build_memory_section(
                memories, nc, display_order, channel_labels
            )
            if mem_block:
                parts.append(mem_block)

        # ── 新 session 预热 ──
        if warmup_text:
            parts.append(
                nc.get("session_warmup_title", "[上次聊到]")
                + "\n"
                + warmup_text
            )

        # ── 轮数 ──
        if turn_count:
            parts.append(
                _safe_format(
                    nc.get("turn_count_template", "聊了 {count} 轮"),
                    count=turn_count,
                )
            )

        # ── 用户画像: traits + quirks ──
        profile = self._build_profile_section(long_term, nc)
        if profile:
            parts.append(profile)

        # ── 结构化画像 ──
        struct = self._build_struct_section(long_term, nc)
        if struct:
            parts.append(struct)

        # ── 共同语境 ──
        ctx = self._build_context_section(long_term, nc)
        if ctx:
            parts.append(ctx)

        # ── 图谱关系链 ──
        graph_chain_title = nc.get("graph_chain_title", "【图谱关系链】")
        if graph_paths:
            parts.append(
                graph_chain_title
                + "\n"
                + "\n".join(f"  · {p}" for p in graph_paths)
            )

        # ── 未完成话题 ──
        dangling_out = self._build_dangling_section(
            dangling, turn_count, nc
        )
        if dangling_out:
            parts.append(dangling_out)

        # ── 页脚 ──
        parts.append(
            nc.get(
                "footer",
                "→ 以上是你通过长期对话积累的对他的了解，用来更好地理解他的意图。",
            )
        )

        return (
            nc.get(
                "prefix",
                "[RCMS 关系上下文,这里面放置了你对他的了解,按需使用]",
            )
            + "\n"
            + "\n\n".join(parts)
        )

    # ────────────────────────────────────────────────────────────────
    # 压缩 prompt
    # ────────────────────────────────────────────────────────────────

    def _build_compressed_memory_block(
        self,
        memories: list,
        pc: dict,
        channel_labels: dict[str, str],
    ) -> str:
        """构建压缩 prompt 的记忆段落

        Args:
            memories: [(content, tag), ...] 回忆列表
            pc: prompt_compressor 模板段
            channel_labels: 通道标签映射

        Returns:
            格式化后的记忆段落
        """
        grouped: dict[str, list[str]] = {}
        order: list[str] = []
        for content, tag in memories:
            if tag not in grouped:
                grouped[tag] = []
                order.append(tag)
            grouped[tag].append(content)

        mem_lines: list[str] = []
        for key in order:
            if key not in grouped:
                continue
            label = channel_labels.get(key, key)
            items = grouped[key][:2]
            mem_lines.append(
                f"【{label}】\n"
                + "\n".join(f"  · {c}" for c in items)
            )

        return "\n\n".join(mem_lines) if mem_lines else ""

    def _build_compressed_lt_block(
        self, long_term: dict, pc: dict
    ) -> str:
        """构建压缩 prompt 的长期语境段落

        Args:
            long_term: 长期语境字典
            pc: prompt_compressor 模板段

        Returns:
            格式化后的长期语境段落
        """
        if not long_term:
            return ""

        lt_block = ""

        shared_ctx = long_term.get("shared_contexts", [])
        if shared_ctx:
            ctx = "、".join(shared_ctx[:3])
            lt_block += (
                "\n"
                + pc.get("shared_context_title", "【共同语境】")
                + ctx
            )

        traits = long_term.get("identity_traits", [])
        if traits:
            trait_strs = [
                t
                for t in traits
                if not t.startswith("[小细节小习惯]")
            ][:3]
            if trait_strs:
                lt_block += (
                    "\n"
                    + pc.get("trait_title", "【用户特质】")
                    + "；".join(trait_strs)
                )

            quirks = [
                t
                for t in traits
                if t.startswith("[小细节小习惯]")
            ][:2]
            if quirks:
                lt_block += (
                    "\n"
                    + pc.get("quirk_title", "【小细节小习惯】")
                    + "；".join(
                        q.replace("[小细节小习惯] ", "") for q in quirks
                    )
                )

        si = long_term.get("self_identity", [])
        if si:
            lt_block += (
                "\n"
                + pc.get("identity_title", "【自我认同】")
                + "、".join(si[:2])
            )

        bounds = long_term.get("boundaries", [])
        if bounds:
            bound_texts = [
                b["description"] if isinstance(b, dict) else str(b)
                for b in bounds[:2]
            ]
            lt_block += (
                "\n"
                + pc.get("boundary_title", "【雷区】")
                + "、".join(bound_texts)
            )

        return lt_block

    def build_compressed_prompt(
        self,
        user_input: str,
        memories: list,
        long_term: dict,
        graph_paths: Optional[list[str]] = None,
    ) -> str:
        """构建压缩 prompt

        对应 context.py prompt_compressor（行 204-267）。

        Args:
            user_input: 用户输入文本
            memories: [(content, tag), ...] 三通道召回的记忆列表
            long_term: 长期语境字典
            graph_paths: 图谱关系链描述列表

        Returns:
            完整的 prompt 字符串

        Raises:
            ValueError: 模板中缺少 prompt_compressor 段
        """
        pc = self._templates.get("prompt_compressor")
        if not pc:
            raise ValueError("模板中缺少 prompt_compressor 段")

        channel_labels = self._templates.get(
            "channel_labels", self._channel_labels
        )

        # ── 分组记忆 ──
        mem_block = self._build_compressed_memory_block(
            memories, pc, channel_labels
        )

        # ── 图谱关系链 ──
        gp_block = ""
        graph_chain_title = pc.get(
            "graph_chain_title", "【图谱关系链】"
        )
        if graph_paths:
            gp_block = (
                "\n"
                + graph_chain_title
                + "\n"
                + "\n".join(f"  · {pth}" for pth in graph_paths)
            )

        # ── 长期语境 ──
        lt_block = self._build_compressed_lt_block(long_term, pc)

        # ── 组装 ──
        prompt = (
            pc.get("mood_title", "【当前心理状态】")
            + "\n"
            + pc.get("mood_default", "自然地聊")
        )

        if mem_block:
            prompt += (
                "\n\n"
                + pc.get("memory_title", "【相关记忆】")
                + "\n"
                + mem_block
            )

        if gp_block:
            prompt += gp_block

        if lt_block:
            prompt += lt_block

        prompt += (
            "\n\n"
            + pc.get("bottom_line_label", "【底线】")
            + "\n"
            + pc.get(
                "bottom_line_text",
                "不主动说教。不假装完全理解。疲惫时简短但不冷漠。",
            )
            + "\n\n"
            + _safe_format(
                pc.get("user_template", "用户: {user_input}\n你:"),
                user_input=user_input,
            )
        )

        return prompt

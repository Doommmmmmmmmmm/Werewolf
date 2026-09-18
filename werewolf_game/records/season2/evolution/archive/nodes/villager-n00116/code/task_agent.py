"""最小可运行的角色 Task-Agent。

每次行动从一个新的模型会话开始。模型默认只收到当前状态；本轮对话只能通过唯一的
受限工具按需读取。这里不包含策略进化、长期记忆、外部检索或其他 Harness。
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
import hashlib
import inspect
import json
import re
from typing import Any

from ..core.errors import ModelClientError, RuleViolationError
from .llm.coordinator import ModelRequestCoordinator
from ..prompts import RoleProfile, render_prompt


_CHINESE_CHARACTER = re.compile(r"[\u3400-\u9fff]")

CURRENT_ROUND_DIALOGUE_TOOL_NAME = "read_current_round_dialogue"
DEFAULT_MAX_TOOL_CALLS_PER_DECISION = 5
DEFAULT_MAX_TOOL_RESULT_TOKENS = 1000
DEFAULT_MAX_PROMPT_CHARS = 12000
# 这是 Task-Agent 层的“纠错重试”次数：首次模型调用之外，最多再请求两次。
# 传输层的 HTTP/网络重试仍由 ModelClient 的 MODEL_MAX_RETRIES 单独控制。
DEFAULT_MAX_DECISION_RETRIES = 2
_SPEECH_ACTION_KINDS = frozenset({"speak", "last_words"})
_DAY_VOTE_PASS_KINDS = frozenset({"pass", "skip", "noop", "idle"})
_PUBLIC_ID_KEYS = ("player_id", "playerId", "id", "target_id", "targetId")


CURRENT_ROUND_DIALOGUE_TOOL: dict[str, Any] = {
    "name": CURRENT_ROUND_DIALOGUE_TOOL_NAME,
    "description": "读取当前昼夜轮次中、当前玩家依法可见的已发生发言。",
    "parameters": {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    },
}


def _shorten_text(text: str, limit: int) -> str:
    text = str(text).strip()
    if limit <= 0:
        return ""
    return text if len(text) <= limit else text[: max(0, limit - 1)] + "…"


def _unique_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        normalized = str(value).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


def _coerce_player_id(value: Any) -> str | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, Mapping):
        for key in _PUBLIC_ID_KEYS:
            nested = value.get(key)
            coerced = _coerce_player_id(nested)
            if coerced:
                return coerced
        return None
    if isinstance(value, (list, tuple, set)):
        for item in value:
            coerced = _coerce_player_id(item)
            if coerced:
                return coerced
        return None
    text = str(value).strip()
    return text or None


def _collect_values_by_keys(node: Any, key_names: set[str], results: list[Any], seen: set[int]) -> None:
    if node is None:
        return
    if isinstance(node, (str, bytes, int, float, bool)):
        return
    node_id = id(node)
    if node_id in seen:
        return
    seen.add(node_id)
    if isinstance(node, Mapping):
        for key, value in node.items():
            if str(key) in key_names:
                results.append(value)
            _collect_values_by_keys(value, key_names, results, seen)
    elif isinstance(node, (list, tuple, set)):
        for item in node:
            _collect_values_by_keys(item, key_names, results, seen)


def _extract_id_list(node: Mapping[str, Any], key_names: tuple[str, ...]) -> list[str]:
    candidates: list[Any] = []
    _collect_values_by_keys(node, set(key_names), candidates, set())
    ids: list[str] = []
    for candidate in candidates:
        if isinstance(candidate, (list, tuple, set)):
            for item in candidate:
                coerced = _coerce_player_id(item)
                if coerced:
                    ids.append(coerced)
        else:
            coerced = _coerce_player_id(candidate)
            if coerced:
                ids.append(coerced)
    return _unique_preserve_order(ids)


_DIALOGUE_TEXT_KEYS = ("text", "content", "message", "speech", "utterance", "dialogue")


def _extract_dialogue_texts(current_dialogue: Any) -> list[str]:
    texts: list[str] = []
    if isinstance(current_dialogue, str):
        text = current_dialogue.strip()
        if text:
            texts.append(text)
        return texts
    if isinstance(current_dialogue, Mapping):
        for key in _DIALOGUE_TEXT_KEYS:
            value = current_dialogue.get(key)
            if isinstance(value, str) and value.strip():
                texts.append(value.strip())
        return texts
    if isinstance(current_dialogue, (list, tuple, set)):
        for item in current_dialogue:
            texts.extend(_extract_dialogue_texts(item))
    return texts


def _extract_day_vote_targets(request: Mapping[str, Any]) -> list[str]:
    phase = str(request.get("phase") or request.get("channel") or request.get("kind") or "").lower()
    allowed_actions = request.get("allowed_actions")
    if not isinstance(allowed_actions, list):
        return []
    targets: list[str] = []
    for item in allowed_actions:
        if not isinstance(item, Mapping):
            continue
        kind = str(item.get("kind") or "").lower()
        if "vote" not in kind or "sheriff" in kind:
            continue
        if "day" not in kind and "day" not in phase and kind != "vote":
            continue
        for target_id in item.get("target_ids") or []:
            coerced = _coerce_player_id(target_id)
            if coerced:
                targets.append(coerced)
    return _unique_preserve_order(targets)


_PUBLIC_CLAIM_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("self_claim", re.compile(r"(?:我是|我就是|我才是|自称).{0,8}?(?:预言家|先知)")),
    ("wolf_check", re.compile(r"(?:查杀|验了|查了|昨晚验了|昨晚查了).{0,10}?(?P<target>p\d+)(?:.{0,12}?(?:狼|狼人|查杀))?")),
    ("wolf_label", re.compile(r"(?P<target>p\d+).{0,4}?(?:是狼|是狼人)")),
    ("sheriff_call", re.compile(r"(?:(?:警长|警徽).{0,6}(?:归票|归)|今天归票?|归票).{0,5}?(?P<target>p\d+)")),
    ("ordinary_vote", re.compile(r"(?:投票给|投|归票).{0,5}?(?P<target>p\d+)")),
    ("gold", re.compile(r"(?:金水).{0,4}?(?P<target>p\d+)")),
    ("opposition", re.compile(r"对跳|跳预言家")),
)


def _dialogue_entries(current_dialogue: Any) -> list[tuple[str | None, str | None, str]]:
    """保留发言的公开元数据；不把没有说话人信息的文本升级为硬事实。"""
    entries: list[tuple[str | None, str | None, str]] = []
    if isinstance(current_dialogue, Mapping):
        text = next((current_dialogue.get(key) for key in _DIALOGUE_TEXT_KEYS
                     if isinstance(current_dialogue.get(key), str)), None)
        if isinstance(text, str) and text.strip():
            speaker = _coerce_player_id(
                current_dialogue.get("player_id")
                or current_dialogue.get("speaker_id")
                or current_dialogue.get("speaker")
                or current_dialogue.get("author_id")
            )
            stage = current_dialogue.get("stage") or current_dialogue.get("phase")
            entries.append((speaker, str(stage).strip() if stage is not None else None, text.strip()))
        else:
            for value in current_dialogue.values():
                entries.extend(_dialogue_entries(value))
    elif isinstance(current_dialogue, (list, tuple, set)):
        for item in current_dialogue:
            entries.extend(_dialogue_entries(item))
    elif isinstance(current_dialogue, str) and current_dialogue.strip():
        entries.append((None, None, current_dialogue.strip()))
    return entries


def _extract_public_claim_hint(current_dialogue: Any, sheriff_id: str | None = None) -> dict[str, Any]:
    snippets: list[str] = []
    claim_targets: list[str] = []
    hard_targets: list[str] = []
    records: list[dict[str, str]] = []
    contested = False

    for speaker_id, stage, raw_text in _dialogue_entries(current_dialogue):
        text = " ".join(str(raw_text).split())
        if not text:
            continue
        self_claim = bool(re.search(r"(?:我是|我就是|我才是|自称).{0,8}?(?:预言家|先知)", text))
        authority = self_claim or (sheriff_id is not None and speaker_id == sheriff_id)
        matched_here = False
        for label, pattern in _PUBLIC_CLAIM_PATTERNS:
            if label == "opposition":
                if pattern.search(text):
                    contested = True
                    matched_here = True
                continue
            for match in pattern.finditer(text):
                matched_here = True
                target = match.groupdict().get("target")
                if not target:
                    continue
                if label != "ordinary_vote":
                    claim_targets.append(target)
                snippets.append(f"{label}:{target}")
                records.append({
                    "speaker_id": speaker_id or "unknown",
                    "stage": stage or "unknown",
                    "kind": label,
                    "target": target,
                })
                # 普通人的“投/归票/查杀”只是声明。只有明确查验或警长归票，
                # 且发言人有相应公开身份，才允许进入 clear_target。
                if authority and label in {"wolf_check", "wolf_label", "sheriff_call"}:
                    hard_targets.append(target)
        if matched_here and (self_claim or re.search(r"查杀|归票|验了|是狼|金水|对跳", text)):
            snippets.append(_shorten_text(text, 56))
        if len(snippets) >= 8:
            break

    claim_targets = _unique_preserve_order(claim_targets)
    hard_targets = _unique_preserve_order(hard_targets)
    if len(claim_targets) > 1 or len(hard_targets) > 1:
        contested = True

    hint: dict[str, Any] = {}
    if claim_targets:
        hint["claim_targets"] = claim_targets[:4]
    if snippets:
        hint["public_claims"] = _unique_preserve_order(snippets)[:4]
    if records:
        hint["claim_records"] = records[:6]
    if contested:
        hint["contested"] = True
    if len(hard_targets) == 1 and not contested:
        hint["clear_target"] = hard_targets[0]
    return hint


def _public_values(public_state: Mapping[str, Any], keys: tuple[str, ...]) -> list[Any]:
    values: list[Any] = []
    _collect_values_by_keys(public_state, set(keys), values, set())
    return values


def _compact_public_values(values: list[Any], limit: int = 6) -> list[str]:
    result: list[str] = []
    for value in values[-limit:]:
        try:
            text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        except (TypeError, ValueError):
            text = str(value)
        text = _shorten_text(text, 180)
        if text and text not in result:
            result.append(text)
    return result


def _bounded_hint(hint: dict[str, Any], limit: int = 1200) -> dict[str, Any]:
    # 提示不是行动依据；超限时优先丢弃声明细节，保留名单和系统字段。
    while len(json.dumps(hint, ensure_ascii=False, separators=(",", ":"))) > limit:
        for key in ("claim_records", "public_claims", "vote_targets", "vote_counts", "revealed_roles"):
            if key in hint:
                hint.pop(key)
                break
        else:
            break
    return hint


def build_villager_public_hint(turn_packet: Mapping[str, Any], current_dialogue: Any) -> dict[str, Any]:
    public_state = turn_packet.get("public_state")
    request = turn_packet.get("request") if isinstance(turn_packet.get("request"), Mapping) else {}
    hint: dict[str, Any] = {}

    if isinstance(public_state, Mapping):
        alive_ids = _extract_id_list(
            public_state,
            ("alive_ids", "alive_players", "living_ids", "survivor_ids", "living"),
        )
        dead_ids = _extract_id_list(
            public_state,
            ("dead_ids", "dead_players", "deceased_ids", "eliminated_ids", "dead"),
        )
        sheriff_ids = _extract_id_list(
            public_state,
            ("sheriff_id", "sheriff_player_id", "sheriff", "警长", "警徽"),
        )
        if alive_ids:
            hint["alive_ids"] = alive_ids[:16]
        if dead_ids:
            hint["dead_ids"] = dead_ids[:16]
        if sheriff_ids:
            hint["sheriff_id"] = sheriff_ids[0]
            if alive_ids:
                hint["sheriff_alive"] = sheriff_ids[0] in alive_ids
        reveal_values = _public_values(
            public_state,
            ("revealed_roles", "role_reveals", "reveals", "flipped_roles"),
        )
        if reveal_values:
            hint["revealed_roles"] = _compact_public_values(reveal_values, 3)
        vote_counts = _public_values(
            public_state,
            ("vote_counts", "vote_results", "day_vote_resolved", "DAY_VOTE_RESOLVED"),
        )
        vote_targets = _public_values(
            public_state,
            ("vote_targets", "votes", "vote_cast", "VOTE_CAST"),
        )
        if vote_counts:
            hint["vote_counts"] = _compact_public_values(vote_counts, 2)
        if vote_targets:
            hint["vote_targets"] = _compact_public_values(vote_targets, 2)

    day_vote_targets = _extract_day_vote_targets(request) if request else []
    if day_vote_targets:
        hint["day_vote_targets"] = day_vote_targets[:16]

    sheriff_id = hint.get("sheriff_id") if isinstance(hint.get("sheriff_id"), str) else None
    claim_hint = _extract_public_claim_hint(current_dialogue, sheriff_id)
    hint.update(claim_hint)
    return _bounded_hint(hint)


def soft_quality_warning(action: dict[str, Any], request: Mapping[str, Any], hint: Mapping[str, Any] | None) -> str | None:
    phase = str(request.get("phase") or request.get("channel") or request.get("kind") or "").lower()
    allowed_actions = request.get("allowed_actions")
    if not isinstance(allowed_actions, list):
        return None
    if "vote" not in phase and not any("vote" in str(item.get("kind") or "").lower() for item in allowed_actions if isinstance(item, Mapping)):
        return None
    vote_targets = _extract_day_vote_targets(request)
    if not vote_targets:
        return None
    action_kind = str(action.get("kind") or "").lower()
    if not isinstance(hint, Mapping):
        hint = {}
    if not hint.get("contested"):
        clear_target = hint.get("clear_target")
        if isinstance(clear_target, str) and clear_target in vote_targets:
            selected_target = str(action.get("target_id") or "")
            if action_kind in _DAY_VOTE_PASS_KINDS or selected_target != clear_target:
                return f"公开查杀/警长归票目标为{clear_target}；请说明硬反证。无硬反证时应投{clear_target}。"
    if action_kind in _DAY_VOTE_PASS_KINDS:
        return "当前有可投目标，平民除非会明显误伤强神或金水，否则不要弃票；请比较两个候选后投证据最强者。"

    return None


# 兼容旧测试/调用方名称；这是策略提示，不是协议错误。
villager_quality_error = soft_quality_warning


def render_villager_decision_checklist(request: Mapping[str, Any]) -> str:
    """给平民用的短决策清单，只作为提示文本，不引入任何隐藏状态。"""

    phase = str(request.get("phase") or request.get("channel") or request.get("kind") or "")
    phase_lower = phase.lower()
    allowed_actions = request.get("allowed_actions")
    allowed_kinds = {
        str(item.get("kind") or "").lower()
        for item in allowed_actions
        if isinstance(item, Mapping)
    } if isinstance(allowed_actions, list) else set()
    lines = ["【平民决策清单】先分系统事实 / 玩家声明 / 自己推测；投票前至少比较两个候选。"]
    has_sheriff_vote = any(key in phase_lower for key in ("sheriff_vote", "警长投票", "警徽投票")) or any(
        kind in {"sheriff_vote", "sheriff"} or "sheriff_vote" in kind for kind in allowed_kinds
    )
    has_sheriff_candidacy = any(key in phase_lower for key in ("sheriff_candidacy", "警长竞选", "竞选", "竞警")) or any(
        "candidate" in kind or "candid" in kind or "sheriff_candidacy" in kind for kind in allowed_kinds
    )
    has_day_vote = (
        (any(key in phase_lower for key in ("vote", "投票")) or any("vote" in kind for kind in allowed_kinds))
        and not has_sheriff_vote
        and not has_sheriff_candidacy
    )
    has_speak = (
        any(key in phase_lower for key in ("speak", "discussion", "发言", "白天"))
        or any(kind in {"speak", "discussion", "day_speak"} for kind in allowed_kinds)
    ) and not has_day_vote and not has_sheriff_vote and not has_sheriff_candidacy
    has_last_words = any(key in phase_lower for key in ("last", "遗言")) or "last_words" in allowed_kinds
    has_sheriff = any(key in phase_lower for key in ("sheriff", "警长", "警徽")) or any(
        key in allowed_kinds for key in ("sheriff_vote", "sheriff")
    )
    if has_day_vote:
        lines.append("投票时先排除 dead_ids，并提高已公开金水、警徽链和可信保护对象的保护权重；要投他们必须有硬矛盾。")
        lines.append("没有明确查杀或警长归票，不要声称存在查杀链；按公开事实错误、票型异常、发言前后矛盾排序，再比较两个候选。")
        lines.append("查杀/归票可信度只是策略权重，不是强制协议；普通玩家的投票倾向、复盘和遗言不能生成硬目标。")
        lines.append("若当前轮或此前公开链中只有一名持续报验的预言家/警长明确查杀或归票，且没有硬对跳、事实矛盾、被公开查杀等强反证，day_vote 可优先投该目标；但先核对其仍存活。")
        lines.append("优先找未被金水覆盖且存在票型异常、事实错误、被查杀或爆狼线索指向的人。")
    if has_sheriff_vote:
        lines.append("警长投票时优先看谁的公开报验链更清晰、更稳定；不要把空泛自证或低权重印象压过查杀和归票信息。")
    if has_sheriff_candidacy:
        lines.append("警长竞选时平民默认不上警；只有需要挡刀、保警徽传递或扰乱狼视角时才考虑上警。")
    if has_speak:
        lines.append("发言前先核对存活名单、已死名单和警长是否仍存活；本轮不要把 dead_ids 中玩家说成当前可放逐对象，只能复盘其遗言/票型。")
        lines.append("没有系统翻牌或公开 reveal 时，不得说‘系统确认某人是狼/神’或‘系统确认某人是刀口’；查杀、刀口、守护、女巫用药只能说某玩家声称或我推测。")
        lines.append("发言至少点出一个关注对象和一个暂不投对象，并说明依据哪条公开信息。")
    if has_last_words:
        lines.append("遗言只留公开事实、票型、查验链和怀疑对象，不把推测说成系统事实。")
    if has_sheriff:
        lines.append("警长/警徽相关回合优先看报验链是否清晰一致；接徽时继承公开遗产但不要把它当系统确认。")
    lines.append("残局或持警徽时，额外复盘存活名单、已死名单、可信查验链和灰区名单，再决定出谁。")
    return "\n".join(lines)


def render_action_contract(request: Mapping[str, Any]) -> str:
    """把本次 ``allowed_actions`` 转成模型容易遵守的最终 JSON 契约。

    行动包本身仍是唯一的规则来源；这个文本只是将其中当前阶段真正可用的
    schema 前置，并为每一种合法行动给出一个**当前就合法**的 JSON 示例。这样
    ``kind`` / ``target_id`` 等协议字段不再需要模型从很长的 packet 中自行猜测。
    """

    raw_actions = request.get("allowed_actions")
    allowed_actions = (
        [item for item in raw_actions if isinstance(item, Mapping)]
        if isinstance(raw_actions, list)
        else []
    )
    phase = str(request.get("phase") or "unknown")
    channel = str(request.get("channel") or "unknown")
    lines = [
        "【本次行动的最终 JSON 契约】",
        f"当前内部阶段：{phase}；频道：{channel}。",
        "最终回答只能是一个 JSON 对象，不能附加 Markdown、解释、代码块或其他字段。",
        "只能从以下当前合法模板中选一个；目标类行动只能把 target_id 改为该模板下列出的候选值。",
    ]
    if not allowed_actions:
        lines.append("当前没有可用行动；不要臆造行动。")
        return "\n".join(lines)

    for allowed in allowed_actions:
        kind = str(allowed.get("kind") or "")
        if not kind:
            continue
        example: dict[str, str] = {"kind": kind}
        target_ids = allowed.get("target_ids")
        valid_targets = (
            [str(target_id) for target_id in target_ids]
            if isinstance(target_ids, list) and target_ids
            else []
        )
        if kind in _SPEECH_ACTION_KINDS:
            # 单个汉字对任意合法的最小 max_chars 都有效，且示例本身通过“必须含中文”的
            # 固定规则。实际 text 可以更完整，也可以包含英文或玩家编号。
            example["text"] = "好"
        elif valid_targets:
            example["target_id"] = valid_targets[0]

        lines.append(
            "- `" + json.dumps(example, ensure_ascii=False, separators=(",", ":")) + "`"
        )
        if valid_targets:
            lines.append("  target_id 候选值：" + "、".join(valid_targets) + "。")
        if kind in _SPEECH_ACTION_KINDS:
            max_chars = allowed.get("max_chars")
            constraints: list[str] = []
            if max_chars is not None:
                constraints.append(f"text 最多 {max_chars} 个字符")
            if allowed.get("require_chinese"):
                constraints.append("text 至少包含一个中文字符")
            constraints.append("text 可以包含英文、数字和玩家编号")
            lines.append("  " + "；".join(constraints) + "。")

    lines.append(
        "kind 和 target_id 是协议字段，使用英文是合法且必须原样保留；不要把它们翻译成中文。"
    )
    lines.append(
        "最终 JSON 只能包含实际行动字段；比较、复盘和理由先在内部完成，除非是 speak / last_words 才把简短依据写进 text。"
    )
    return "\n".join(lines)


def normalize_decision(raw: dict[str, Any] | None, request: dict[str, Any]) -> dict[str, Any]:
    """把模型可能使用的 action / targetId 字段规范成引擎格式。"""

    source = raw.get("decision", raw) if isinstance(raw, dict) else {}
    action = {
        "request_id": request["request_id"],
        "player_id": request["player_id"],
        "kind": str(source.get("kind", source.get("action", ""))),
    }
    target_id = source.get("target_id", source.get("targetId"))
    if target_id:
        action["target_id"] = str(target_id)
    if isinstance(source.get("text"), str):
        action["text"] = source["text"]
    return action


def decision_error(action: dict[str, Any], request: dict[str, Any]) -> str | None:
    """在提交引擎前做一次本地校验，便于让模型重试。"""

    allowed = next(
        (item for item in request["allowed_actions"] if item["kind"] == action["kind"]),
        None,
    )
    if allowed is None:
        return "kind 必须是 allowed_actions 中的一项"
    if action["kind"] in _SPEECH_ACTION_KINDS:
        text = str(action.get("text", "")).strip()
        if not text:
            return f"{action['kind']} 必须提供非空 text"
        if len(text) > int(allowed["max_chars"]):
            return f"发言超过 max_chars={allowed['max_chars']}"
        if allowed.get("require_chinese") and not _CHINESE_CHARACTER.search(text):
            return "发言必须包含中文"
    target_ids = allowed.get("target_ids")
    if target_ids:
        if action.get("target_id") not in target_ids:
            return "target_id 必须是 target_ids 中的一项"
    elif action.get("target_id"):
        return "该行动不能提供 target_id"
    return None


class TaskAgent:
    """一个角色在一局中的最小任务求解器。"""

    def __init__(
        self,
        *,
        player_id: str,
        profile: RoleProfile,
        model_client: Any,
        persona: str = "",
        request_coordinator: ModelRequestCoordinator | None = None,
        max_tokens: int = 900,
        max_decision_retries: int = DEFAULT_MAX_DECISION_RETRIES,
        max_tool_calls_per_decision: int = DEFAULT_MAX_TOOL_CALLS_PER_DECISION,
        max_tool_result_tokens: int = DEFAULT_MAX_TOOL_RESULT_TOKENS,
        max_prompt_chars: int = DEFAULT_MAX_PROMPT_CHARS,
    ) -> None:
        if not hasattr(model_client, "complete_json"):
            raise ValueError("TaskAgent 需要具有 complete_json 的模型客户端")
        self.player_id = str(player_id)
        self.profile = profile
        self.model_client = model_client
        self.persona = persona
        self.request_coordinator = request_coordinator
        self.max_tokens = max_tokens
        self.max_decision_retries = max(0, int(max_decision_retries))
        self.max_tool_calls_per_decision = max(0, int(max_tool_calls_per_decision))
        self.max_tool_result_tokens = max(1, int(max_tool_result_tokens))
        self.max_prompt_chars = max(1, int(max_prompt_chars))
        # 只保存引擎公开同步中明确标注的事实；每类事件有界，且按 game_id 隔离。
        self._public_ledger: dict[str, Any] = {
            "game_id": None,
            "round": None,
            "alive_ids": [],
            "dead_ids": [],
            "sheriff_id": None,
            "deaths": [],
            "votes": [],
            "reveals": [],
        }
        self._model_token_usage: dict[str, int] = {
            "successful_response_count": 0,
            "api_attempt_count": 0,
            "reported_usage_response_count": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }

    async def observe(self, sync_packet: dict[str, Any]) -> None:
        """只压缩保存公开同步，绝不保存私有信息或完整历史。"""

        if not isinstance(sync_packet, Mapping):
            return
        public_state = sync_packet.get("public_state")
        if not isinstance(public_state, Mapping):
            # 某些同步包直接以 public_state 形状发送；仍只读取公开键名。
            public_state = sync_packet
        game_values = [
            sync_packet[key] for key in ("game_id", "gameId") if key in sync_packet
        ]
        if not game_values:
            game_values = _public_values(public_state, ("game_id", "gameId"))
        game_id = _coerce_player_id(game_values[0]) if game_values else None
        old_game_id = self._public_ledger.get("game_id")
        if game_id and old_game_id and game_id != old_game_id:
            self._public_ledger = {
                "game_id": game_id, "round": None, "alive_ids": [], "dead_ids": [],
                "sheriff_id": None, "deaths": [], "votes": [], "reveals": [],
            }
        elif game_id:
            self._public_ledger["game_id"] = game_id

        rounds = [
            sync_packet[key]
            for key in ("round", "round_id", "roundId")
            if key in sync_packet
        ]
        if not rounds:
            rounds = _public_values(public_state, ("round", "round_id", "roundId"))
        if rounds:
            self._public_ledger["round"] = _shorten_text(str(rounds[-1]), 32)
        alive = _extract_id_list(public_state, ("alive_ids", "alive_players", "living_ids", "survivor_ids", "living"))
        dead = _extract_id_list(public_state, ("dead_ids", "dead_players", "deceased_ids", "eliminated_ids", "dead"))
        sheriff = _extract_id_list(public_state, ("sheriff_id", "sheriff_player_id", "sheriff"))
        if alive:
            self._public_ledger["alive_ids"] = alive[:16]
        if dead:
            self._public_ledger["dead_ids"] = dead[:16]
        if sheriff:
            self._public_ledger["sheriff_id"] = sheriff[0]

        for field, keys in {
            "deaths": ("deaths", "death_events", "recent_deaths", "night_deaths", "eliminations", "last_dead"),
            "votes": ("vote_counts", "vote_results", "day_vote_resolved", "DAY_VOTE_RESOLVED", "votes", "vote_cast", "VOTE_CAST"),
            "reveals": ("revealed_roles", "role_reveals", "reveals", "flipped_roles"),
        }.items():
            values = _compact_public_values(_public_values(public_state, keys), 6)
            if values:
                self._public_ledger[field] = (self._public_ledger[field] + values)[-6:]

    def _public_ledger_summary(self) -> str:
        ledger = self._public_ledger
        parts = [
            "【公开事实核对】",
            "存活:" + ",".join(ledger.get("alive_ids") or []),
            "已死:" + ",".join(ledger.get("dead_ids") or []),
            "警徽:" + str(ledger.get("sheriff_id") or "未知"),
        ]
        for label, key in (("最近死亡", "deaths"), ("最近票型", "votes"), ("公开翻牌", "reveals")):
            values = ledger.get(key) or []
            if values:
                parts.append(label + ":" + "；".join(values[-6:]))
        return _shorten_text("；".join(parts), 1200)

    async def decide(self, turn_packet: dict[str, Any]) -> dict[str, Any]:
        if turn_packet["self"]["player_id"] != self.player_id:
            raise ValueError("行动包被发送给了错误玩家")
        private = turn_packet["private_information"]
        if str(private.get("role")) != self.profile.role:
            raise ValueError("Task-Agent 的角色档案与行动包身份不一致")

        system = self._system_prompt(private)
        prompt = {
            "game": turn_packet["game"],
            "public_rules": turn_packet["public_rules"],
            "self": turn_packet["self"],
            "private_information": private,
            "public_state": turn_packet["public_state"],
            "request": turn_packet["request"],
            "history_policy": {
                "default_context": "current_state_only",
                "current_round_dialogue_tool": CURRENT_ROUND_DIALOGUE_TOOL_NAME,
                "max_tool_calls": self.max_tool_calls_per_decision,
                "max_tool_result_tokens": self.max_tool_result_tokens,
            },
        }

        tool_context = turn_packet.get("tool_context") or {}
        current_dialogue = tool_context.get("current_round_dialogue") or []
        villager_public_hint = build_villager_public_hint(turn_packet, current_dialogue)
        ledger_summary = self._public_ledger_summary()

        def execute_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
            del arguments
            if name != CURRENT_ROUND_DIALOGUE_TOOL_NAME:
                return {"error": f"不支持的工具：{name}"}
            return {
                "round": turn_packet["game"].get("round"),
                "phase": turn_packet["game"].get("public_phase", turn_packet["game"].get("phase")),
                "dialogue": current_dialogue,
            }

        feedback = ""
        for _attempt in range(self.max_decision_retries + 1):
            # 一次玩家决策的工具预算不能因纠错重试而被放大。首个模型尝试可读取
            # 当前轮对话；所有后续纠错尝试都是没有工具的新会话。
            allow_tool = _attempt == 0 and self.max_tool_calls_per_decision > 0
            user_content: dict[str, Any] = {
                "instruction": self._turn_instruction(
                    turn_packet["request"], feedback, villager_public_hint, ledger_summary
                ),
                "packet": prompt,
            }
            try:
                raw = await self._complete_json(
                    system=system,
                    messages=[
                        {
                            "role": "user",
                            "content": json.dumps(user_content, ensure_ascii=False),
                        }
                    ],
                    tools=[CURRENT_ROUND_DIALOGUE_TOOL] if allow_tool else None,
                    tool_executor=execute_tool if allow_tool else None,
                    max_tool_calls=self.max_tool_calls_per_decision if allow_tool else 0,
                    max_tool_result_tokens=self.max_tool_result_tokens,
                    max_prompt_chars=self.max_prompt_chars,
                )
            except ModelClientError:
                if _attempt >= self.max_decision_retries:
                    raise
                # 不把上游错误正文放回模型上下文，既避免把服务端内容当指令，也避免
                # 将网关诊断混进公开可见的游戏文本。
                feedback = "上一轮模型调用没有产生可用的结构化行动，请直接重新作答。"
                continue
            self._record_token_usage(raw)
            action = normalize_decision(raw, turn_packet["request"])
            error = decision_error(action, turn_packet["request"])
            if error is not None:
                if _attempt >= self.max_decision_retries:
                    # 只保留规范化后的行动字段；完整原始响应和推理文本不会进入记录。
                    raise RuleViolationError(
                        f"模型返回非法行动：{error}",
                        attempted_action=action,
                        validation_error=error,
                    )
                feedback = f"上一次输出未通过协议校验：{error}"
                continue

            quality_warning = soft_quality_warning(action, turn_packet["request"], villager_public_hint)
            if quality_warning is None:
                return action
            if _attempt >= self.max_decision_retries:
                # 质量提示不是协议错误：最后一次 schema 合法行动照常提交。
                return action
            feedback = f"策略建议（不是协议错误）：{quality_warning} 请重新比较公开事实后作答。"

    def agent_manifest(self) -> dict[str, Any]:
        return {
            "agent_type": "task_agent",
            "player_id": self.player_id,
            "role": self.profile.role,
            "conversation_mode": "new_session_per_decision",
            "tool_name": CURRENT_ROUND_DIALOGUE_TOOL_NAME,
            "max_tool_calls_per_decision": self.max_tool_calls_per_decision,
            "max_tool_result_tokens": self.max_tool_result_tokens,
            "max_prompt_chars": self.max_prompt_chars,
            "max_decision_retries": self.max_decision_retries,
            "task_fingerprint": hashlib.sha256(
                self.profile.task.encode("utf-8")
            ).hexdigest(),
        }

    def model_token_usage_snapshot(self) -> dict[str, int]:
        return dict(self._model_token_usage)

    def _system_prompt(self, private: Mapping[str, Any]) -> str:
        return render_prompt(
            "player_system.txt",
            player_id=self.player_id,
            role=private["role"],
            team=private["team"],
            persona=self.persona,
            role_base=self.profile.base,
            role_task=self.profile.task,
            tool_name=CURRENT_ROUND_DIALOGUE_TOOL_NAME,
            max_tool_calls=self.max_tool_calls_per_decision,
            max_tool_result_tokens=self.max_tool_result_tokens,
            max_prompt_chars=self.max_prompt_chars,
        )

    @staticmethod
    def _turn_instruction(
        request: Mapping[str, Any],
        feedback: str,
        public_hint: Mapping[str, Any] | None,
        ledger_summary: str = "",
    ) -> str:
        checklist = render_villager_decision_checklist(request)
        feedback_text = f"上一次输出未通过校验：{feedback}" if feedback else ""
        hint_text = ""
        if isinstance(public_hint, Mapping) and public_hint:
            hint_text = "【公开信息提示】" + json.dumps(public_hint, ensure_ascii=False, separators=(",", ":"))
        dynamic_rules: list[str] = []
        if isinstance(public_hint, Mapping) and public_hint.get("dead_ids"):
            dynamic_rules.append("本轮发言不要把 dead_ids 中玩家作为当前可放逐对象，只能复盘其公开遗言或票型。")
        if isinstance(public_hint, Mapping) and not public_hint.get("revealed_roles"):
            dynamic_rules.append("当前没有公开翻牌依据时，不得说系统确认某人是狼、神或刀口；只能标明玩家声称或自己的推测。")
        validation_feedback = "\n".join(
            part for part in (feedback_text, ledger_summary, hint_text, "\n".join(dynamic_rules), checklist) if part
        )
        return render_prompt(
            "player_turn_instruction.txt",
            action_contract=render_action_contract(request),
            validation_feedback=validation_feedback,
        )

    async def _complete_json(self, **kwargs: Any) -> dict[str, Any]:
        if self.request_coordinator is not None:
            return await self.request_coordinator.complete_json(
                self.model_client, max_tokens=self.max_tokens, **kwargs
            )
        complete_json = self.model_client.complete_json
        if inspect.iscoroutinefunction(complete_json):
            return await complete_json(max_tokens=self.max_tokens, **kwargs)
        return await asyncio.to_thread(complete_json, max_tokens=self.max_tokens, **kwargs)

    def _record_token_usage(self, response: object) -> None:
        self._model_token_usage["successful_response_count"] += 1
        attempts = self._nonnegative_int(getattr(response, "api_attempts", 1), fallback=1)
        self._model_token_usage["api_attempt_count"] += max(1, attempts or 1)
        usage = getattr(response, "token_usage", None)
        if not isinstance(usage, dict):
            return
        values = {
            key: self._nonnegative_int(usage.get(key), fallback=None)
            for key in ("input_tokens", "output_tokens", "total_tokens")
        }
        if all(value is None for value in values.values()):
            return
        self._model_token_usage["reported_usage_response_count"] += 1
        for key, value in values.items():
            if value is not None:
                self._model_token_usage[key] += value

    @staticmethod
    def _nonnegative_int(value: object, *, fallback: int | None) -> int | None:
        if isinstance(value, bool):
            return fallback
        try:
            normalized = int(value)
        except (TypeError, ValueError):
            return fallback
        return normalized if normalized >= 0 else fallback

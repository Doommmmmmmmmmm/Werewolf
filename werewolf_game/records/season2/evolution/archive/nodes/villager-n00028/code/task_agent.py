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


CURRENT_ROUND_DIALOGUE_TOOL: dict[str, Any] = {
    "name": CURRENT_ROUND_DIALOGUE_TOOL_NAME,
    "description": "读取当前昼夜轮次中、当前玩家依法可见的已发生发言。",
    "parameters": {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    },
}


def _search_public_values(root: Any, target_keys: tuple[str, ...], *, max_results: int = 3, max_depth: int = 4) -> list[Any]:
    lowered_targets = tuple(key.lower() for key in target_keys)
    results: list[Any] = []
    queue: list[tuple[Any, int]] = [(root, 0)]
    seen: set[int] = set()

    while queue and len(results) < max_results:
        node, depth = queue.pop(0)
        if depth > max_depth:
            continue
        if isinstance(node, Mapping):
            node_id = id(node)
            if node_id in seen:
                continue
            seen.add(node_id)
            for key, value in node.items():
                key_text = str(key).lower()
                if any(
                    target == key_text or target in key_text or key_text in target
                    for target in lowered_targets
                ):
                    results.append(value)
                    if len(results) >= max_results:
                        break
                if isinstance(value, (Mapping, list, tuple, set)) and depth < max_depth:
                    queue.append((value, depth + 1))
        elif isinstance(node, (list, tuple, set)):
            for item in node:
                if isinstance(item, (Mapping, list, tuple, set)) and depth < max_depth:
                    queue.append((item, depth + 1))
    return results


def _player_label(value: Any) -> str | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (str, int)):
        text = str(value).strip()
        return text or None
    if isinstance(value, Mapping):
        for key in (
            "player_id",
            "playerId",
            "id",
            "name",
            "target_id",
            "targetId",
            "holder",
            "holder_id",
            "badge_holder",
            "badgeHolder",
            "sheriff_id",
        ):
            item = value.get(key)
            if isinstance(item, (str, int)):
                text = str(item).strip()
                if text:
                    return text
        role = value.get("role")
        player_id = value.get("player_id") or value.get("playerId") or value.get("id")
        if isinstance(role, str) and isinstance(player_id, (str, int)):
            pid = str(player_id).strip()
            if pid:
                return f"{pid}({role})"
    return None


def _collect_compact_labels(value: Any, *, limit: int = 6) -> list[str]:
    labels: list[str] = []
    seen: set[str] = set()

    def add(label: str | None) -> None:
        if not label:
            return
        cleaned = str(label).strip()
        if not cleaned or cleaned in seen:
            return
        seen.add(cleaned)
        labels.append(cleaned)

    def visit(node: Any) -> None:
        if len(labels) >= limit:
            return
        label = _player_label(node)
        if label is not None:
            add(label)
            return
        if isinstance(node, Mapping):
            direct_label = _player_label(node)
            if direct_label is not None:
                add(direct_label)
                return
            for key, item in node.items():
                if len(labels) >= limit:
                    break
                key_label = _player_label(key)
                if key_label is not None and not isinstance(item, (Mapping, list, tuple, set)):
                    if isinstance(item, bool):
                        if item:
                            add(key_label)
                    elif isinstance(item, (int, float)):
                        if item:
                            add(f"{key_label}:{item}")
                    elif isinstance(item, str):
                        text = item.strip()
                        if text:
                            add(f"{key_label}:{text}")
                    else:
                        add(key_label)
                else:
                    visit(item)
        elif isinstance(node, (list, tuple, set)):
            for item in node:
                if len(labels) >= limit:
                    break
                visit(item)

    visit(value)
    return labels


def _format_compact_value(value: Any, *, limit: int = 6) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, (str, int, float)):
        return str(value).strip()
    if isinstance(value, Mapping):
        parts: list[str] = []
        for key, item in value.items():
            if len(parts) >= limit:
                break
            key_text = _player_label(key) or str(key).strip()
            if not key_text:
                continue
            if isinstance(item, bool):
                if item:
                    parts.append(key_text)
                continue
            if isinstance(item, (str, int, float)):
                item_text = str(item).strip()
                if item_text:
                    parts.append(f"{key_text}:{item_text}")
                continue
            nested = _format_compact_value(item, limit=2)
            if nested:
                parts.append(f"{key_text}:{nested}")
        if not parts:
            parts = _collect_compact_labels(value, limit=limit)
        return "、".join(parts[:limit])
    if isinstance(value, (list, tuple, set)):
        return "、".join(_collect_compact_labels(value, limit=limit))
    return str(value).strip()


_FALSELIKE_COMPACT_TEXTS = {"", "否", "false", "False", "0", "无", "none", "None"}


def _is_meaningful_compact_text(text: str) -> bool:
    return str(text).strip() not in _FALSELIKE_COMPACT_TEXTS


def _format_vote_snapshot(value: Any) -> str:
    if isinstance(value, Mapping):
        parts: list[str] = []
        for key, item in value.items():
            if len(parts) >= 5:
                break
            key_text = _player_label(key) or str(key).strip()
            if not key_text:
                continue
            if isinstance(item, (int, float)):
                parts.append(f"{key_text}×{item}")
            elif isinstance(item, bool):
                if item:
                    parts.append(key_text)
            else:
                item_text = _format_compact_value(item, limit=2)
                if item_text:
                    parts.append(f"{key_text}:{item_text}")
        if parts:
            return "、".join(parts)
    if isinstance(value, (list, tuple, set)):
        recent_targets: list[str] = []
        counts: dict[str, int] = {}
        for item in list(value)[-10:]:
            if isinstance(item, Mapping):
                target = (
                    item.get("target_id")
                    or item.get("targetId")
                    or item.get("target")
                    or item.get("vote_target")
                )
                label = _player_label(target)
                if label is None and target is not None:
                    label = str(target).strip() or None
                if label:
                    recent_targets.append(label)
                    counts[label] = counts.get(label, 0) + 1
        if counts:
            ranked = sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))
            return "、".join(f"{name}×{count}" for name, count in ranked[:5])
        if recent_targets:
            return "、".join(recent_targets[-5:])
    return _format_compact_value(value, limit=6)


def _extract_first_public_count(public_state: Any, target_keys: tuple[str, ...]) -> int | None:
    for value in _search_public_values(public_state, target_keys, max_results=4):
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            normalized = int(value)
            if normalized >= 0:
                return normalized
        if isinstance(value, (list, tuple, set)):
            return len(value)
        if isinstance(value, Mapping):
            truthy_count = 0
            seen_simple_status = False
            for item in value.values():
                if isinstance(item, bool):
                    seen_simple_status = True
                    if item:
                        truthy_count += 1
                elif isinstance(item, (int, float)):
                    seen_simple_status = True
                    if item > 0:
                        truthy_count += 1
                elif isinstance(item, str):
                    seen_simple_status = True
                    if item.strip().lower() not in {"", "0", "false", "dead", "eliminated", "out", "none", "null"}:
                        truthy_count += 1
                else:
                    seen_simple_status = False
                    truthy_count = 0
                    break
            if seen_simple_status and truthy_count >= 0:
                return truthy_count
    return None


def _collect_allowed_targets(request: Mapping[str, Any]) -> list[str]:
    allowed_actions = request.get("allowed_actions")
    targets: list[str] = []
    if not isinstance(allowed_actions, list):
        return targets
    for action in allowed_actions:
        if not isinstance(action, Mapping):
            continue
        raw_targets = action.get("target_ids")
        if not isinstance(raw_targets, list):
            continue
        for target_id in raw_targets:
            label = _player_label(target_id)
            if label is None:
                label = str(target_id).strip()
            if label and label not in targets:
                targets.append(label)
    return targets


def render_public_state_brief(public_state: Any, request: Mapping[str, Any]) -> str:
    phase = str(request.get("phase") or request.get("channel") or request.get("kind") or "unknown")
    lines = [f"【当前公开态势】阶段：{phase}。"]

    alive_count = _extract_first_public_count(public_state, ("alive_players", "living_players", "alive", "survivors", "live_players", "living"))
    if alive_count is not None:
        lines.append(f"存活人数：{alive_count}。")

    alive_values = _search_public_values(public_state, ("alive_players", "living_players", "survivors", "live_players", "living"), max_results=1)
    if alive_values:
        text = _format_compact_value(alive_values[0], limit=8)
        if _is_meaningful_compact_text(text):
            lines.append(f"存活名单：{text}。")

    dead_values = _search_public_values(public_state, ("dead_players", "eliminated_players", "dead", "graveyard", "death_list"), max_results=1)
    if dead_values:
        text = _format_compact_value(dead_values[0], limit=8)
        if _is_meaningful_compact_text(text):
            lines.append(f"已死名单：{text}。")

    sheriff_values = _search_public_values(public_state, ("sheriff", "sheriff_id", "badge_holder", "badgeHolder", "badge", "police", "captain"), max_results=2)
    if sheriff_values:
        text = _format_compact_value(sheriff_values[0], limit=4)
        if _is_meaningful_compact_text(text):
            lines.append(f"警长/警徽：{text}。")

    claim_values = _search_public_values(public_state, ("public_claims", "claims", "claims", "revealed_roles", "role_claims", "check_log", "checks"), max_results=1)
    if claim_values:
        text = _format_compact_value(claim_values[0], limit=6)
        if _is_meaningful_compact_text(text):
            lines.append(f"公开查验/自报：{text}。")

    vote_values = _search_public_values(public_state, ("vote_history", "votes", "ballots", "vote_matrix", "voting"), max_results=1)
    if vote_values:
        text = _format_vote_snapshot(vote_values[0])
        if _is_meaningful_compact_text(text):
            lines.append(f"近期票型：{text}。")

    targets = _collect_allowed_targets(request)
    if targets:
        lines.append(f"本次可选目标：{'、'.join(targets[:6])}。")

    return "\n".join(lines[:7])


def render_villager_decision_checklist(request: Mapping[str, Any], public_state: Any | None = None) -> str:
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
    has_vote = any(key in phase_lower for key in ("vote", "投票")) or any("vote" in kind for kind in allowed_kinds)
    has_speak = any(key in phase_lower for key in ("speak", "discussion", "发言", "白天", "day")) or any(
        kind in {"speak", "discussion", "day_speak"} for kind in allowed_kinds
    )
    has_last_words = any(key in phase_lower for key in ("last", "遗言")) or "last_words" in allowed_kinds
    has_sheriff = any(key in phase_lower for key in ("sheriff", "警长", "警徽")) or any(
        key in allowed_kinds for key in ("sheriff_vote", "sheriff")
    )
    alive_count = _extract_first_public_count(public_state, ("alive_players", "living_players", "alive", "survivors", "live_players", "living")) if public_state is not None else None
    sheriff_values = _search_public_values(public_state, ("sheriff", "sheriff_id", "badge_holder", "badgeHolder", "badge", "police", "captain"), max_results=2) if public_state is not None else []
    sheriff_text = _format_compact_value(sheriff_values[0], limit=4) if sheriff_values else ""
    if not _is_meaningful_compact_text(sheriff_text):
        sheriff_text = ""
    claim_values = _search_public_values(public_state, ("public_claims", "claims", "revealed_roles", "role_claims", "check_log", "checks"), max_results=1) if public_state is not None else []
    vote_values = _search_public_values(public_state, ("vote_history", "votes", "ballots", "vote_matrix", "voting"), max_results=1) if public_state is not None else []

    if has_vote:
        lines.append("投票时优先保护可信金水、持续报验且查杀被票型或发言支持的预言家、强神声明者和警徽传递链；只有硬对跳、明显矛盾、查杀命中或事实错误时才推翻。")
        lines.append("优先找未被金水覆盖且存在票型异常、事实错误、被查杀或爆狼线索指向的人。")
    if has_speak:
        lines.append("发言至少点出一个关注对象和一个暂不投对象，并说明依据哪条公开信息。")
    if has_last_words:
        lines.append("遗言只留公开事实、票型、查验链和怀疑对象，不把推测说成系统事实。")
    if has_sheriff:
        lines.append("警长/警徽相关回合优先看报验链是否清晰一致；接徽时继承公开遗产但不要把它当系统确认。")
    if alive_count is not None and alive_count <= 6:
        lines.append("残局强制规则：先在当前存活者里排出两个嫌疑人，并明确说明谁更像狼、为什么；不能只复读旧链或只保守跟票。")
        lines.append("比较时优先用存活/死亡顺序、投票轨迹、公开查验链和当前矛盾来重排嫌疑，不要让已死或无徽的旧权威自动保留高权重。")
    if sheriff_text:
        lines.append("若警长已死、失徽、转移或其身份链被公开打断，就按当前存活证据重新排序，不再默认跟随旧警徽链。")
    if claim_values and not sheriff_text:
        lines.append("公开查验链若只剩单一路径，也必须拿它和一个基于票型/死亡/矛盾的替代嫌疑直接对比后再投。")
    if vote_values:
        lines.append("投票比较时优先看谁更像被跟票、谁的死亡顺序与公开声明冲突、谁在当前存活集里更难自洽。")
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
        self._model_token_usage: dict[str, int] = {
            "successful_response_count": 0,
            "api_attempt_count": 0,
            "reported_usage_response_count": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }

    async def observe(self, sync_packet: dict[str, Any]) -> None:
        """接收公开同步；最小基线不把它跨行动保存为模型记忆。"""

        # 保留 Participant 接口，但刻意不维护跨行动状态。
        del sync_packet

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
            "public_state_brief": render_public_state_brief(turn_packet["public_state"], turn_packet["request"]),
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
                "instruction": self._turn_instruction(turn_packet["request"], turn_packet["public_state"], feedback),
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
            if error is None:
                return action
            if _attempt >= self.max_decision_retries:
                # 只保留规范化后的行动字段；完整原始响应和推理文本不会进入记录。
                # Runner 可将该行动存入管理员审计记录，便于定位 schema / 长度等问题，
                # 而不会泄露给公开记录。
                raise RuleViolationError(
                    f"模型返回非法行动：{error}",
                    attempted_action=action,
                    validation_error=error,
                )
            feedback = f"上一次行动未通过校验：{error}"

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
    def _turn_instruction(request: Mapping[str, Any], public_state: Any, feedback: str) -> str:
        checklist = render_villager_decision_checklist(request, public_state)
        public_brief = render_public_state_brief(public_state, request)
        feedback_text = f"上一次输出未通过校验：{feedback}" if feedback else ""
        validation_feedback = "\n".join(
            part for part in (public_brief, feedback_text, checklist) if part
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

"""最小可运行的角色 Task-Agent。

每次行动从一个新的模型会话开始。模型默认只收到当前状态；本轮对话只能通过唯一的
受限工具按需读取。这里不包含策略进化、长期记忆、外部检索或其他 Harness。
"""

from __future__ import annotations

import asyncio
from collections import deque
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
_MEMORY_ROUND_LIMIT = 6
_MEMORY_NOTE_LIMIT = 5
_MEMORY_TEXT_LIMIT = 140

_STATE_PRIORITY_KEYS = (
    "round",
    "day",
    "phase",
    "public_phase",
    "status",
    "stage",
    "turn",
    "sheriff_id",
    "leader_id",
    "votes",
    "vote_result",
    "tied_target_ids",
    "target_id",
    "target_ids",
    "dead_player_ids",
    "death_player_ids",
    "deaths",
    "alive_player_ids",
    "revealed_role",
    "revealed_roles",
    "result",
    "reason",
    "no_kill",
    "claim",
    "claims",
    "public_events",
    "event",
    "kind",
    "type",
    "text",
    "message",
)


CURRENT_ROUND_DIALOGUE_TOOL: dict[str, Any] = {
    "name": CURRENT_ROUND_DIALOGUE_TOOL_NAME,
    "description": "读取当前昼夜轮次中、当前玩家依法可见的已发生发言。",
    "parameters": {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    },
}


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


def _truncate_text(text: object, limit: int) -> str:
    string = str(text).strip()
    if limit <= 0 or len(string) <= limit:
        return string
    return string[: max(0, limit - 1)] + "…"


def _has_content(value: object) -> bool:
    return value not in (None, "", [], {}, ())


def _compact_value(
    value: object,
    *,
    max_depth: int = 2,
    max_items: int = 8,
    max_chars: int = 120,
    exclude_keys: frozenset[str] = frozenset(),
) -> object:
    if isinstance(value, (bool, int, float)) or value is None:
        return value
    if isinstance(value, str):
        return _truncate_text(value, max_chars)
    if max_depth <= 0:
        return _truncate_text(
            json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str),
            max_chars,
        )
    if isinstance(value, Mapping):
        return _compact_mapping(
            value,
            max_depth=max_depth,
            max_items=max_items,
            max_chars=max_chars,
            exclude_keys=exclude_keys,
        )
    if isinstance(value, set):
        value = list(value)
    if isinstance(value, (list, tuple)):
        items: list[object] = []
        for item in list(value)[:max_items]:
            compacted = _compact_value(
                item,
                max_depth=max_depth - 1,
                max_items=max_items,
                max_chars=max_chars,
                exclude_keys=exclude_keys,
            )
            if _has_content(compacted):
                items.append(compacted)
        return items
    return _truncate_text(value, max_chars)


def _compact_mapping(
    data: Mapping[str, Any],
    *,
    max_depth: int = 2,
    max_items: int = 8,
    max_chars: int = 120,
    exclude_keys: frozenset[str] = frozenset(),
) -> dict[str, object]:
    ordered_keys: list[str] = []
    seen: set[str] = set()
    for key in _STATE_PRIORITY_KEYS:
        if key in data and key not in exclude_keys and key not in seen:
            ordered_keys.append(key)
            seen.add(key)
    for key in data.keys():
        key_str = str(key)
        if key_str in seen or key_str in exclude_keys:
            continue
        ordered_keys.append(key_str)
        seen.add(key_str)

    result: dict[str, object] = {}
    for key in ordered_keys[:max_items]:
        compacted = _compact_value(
            data[key],
            max_depth=max_depth - 1,
            max_items=max_items,
            max_chars=max_chars,
            exclude_keys=exclude_keys,
        )
        if _has_content(compacted):
            result[str(key)] = compacted
    return result


def _first_present(mapping: Mapping[str, Any], *keys: str) -> object | None:
    for key in keys:
        if key in mapping and _has_content(mapping[key]):
            return mapping[key]
    return None


def _short_id_list(value: object, *, max_items: int = 6) -> list[str] | None:
    if isinstance(value, (list, tuple, set)):
        items = [str(item) for item in list(value)[:max_items]]
        return items if items else None
    return None


def _signature_for_object(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode(
            "utf-8"
        )
    ).hexdigest()


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
        self._recent_rounds: deque[dict[str, Any]] = deque(maxlen=_MEMORY_ROUND_LIMIT)
        self._own_actions: deque[dict[str, Any]] = deque(maxlen=_MEMORY_ROUND_LIMIT)
        self._last_wolf_plan: dict[str, Any] | None = None
        self._last_sync_signature: str | None = None

    async def observe(self, sync_packet: dict[str, Any]) -> None:
        """接收公开同步，并维护一个严格限长的结构化记忆。"""

        if not isinstance(sync_packet, Mapping):
            return
        snapshot = self._build_sync_snapshot(sync_packet)
        if not snapshot:
            return
        signature = _signature_for_object(snapshot)
        if signature == self._last_sync_signature:
            return
        self._last_sync_signature = signature
        self._merge_sync_snapshot(snapshot)

    async def decide(self, turn_packet: dict[str, Any]) -> dict[str, Any]:
        if turn_packet["self"]["player_id"] != self.player_id:
            raise ValueError("行动包被发送给了错误玩家")
        private = turn_packet["private_information"]
        if str(private.get("role")) != self.profile.role:
            raise ValueError("Task-Agent 的角色档案与行动包身份不一致")

        wolf_plan = self._wolf_plan(turn_packet, private)
        if wolf_plan:
            self._last_wolf_plan = wolf_plan

        system = self._system_prompt(private)
        prompt = self._build_turn_packet(turn_packet, private, wolf_plan)

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
                "instruction": self._turn_instruction(
                    turn_packet["request"],
                    feedback,
                    role=str(private.get("role") or ""),
                    wolf_plan=wolf_plan,
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
            if error is None:
                self._remember_decision(turn_packet, action, wolf_plan)
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
            "memory_round_limit": _MEMORY_ROUND_LIMIT,
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

    def _turn_instruction(
        self,
        request: Mapping[str, Any],
        feedback: str,
        *,
        role: str,
        wolf_plan: dict[str, Any] | None,
    ) -> str:
        instruction = render_prompt(
            "player_turn_instruction.txt",
            action_contract=render_action_contract(request),
            validation_feedback=(f"上一次输出未通过校验：{feedback}" if feedback else ""),
        )
        extra_lines: list[str] = []
        if role == "wolf":
            if self._request_has_speech_action(request):
                extra_lines.append(
                    "狼人发言必须按‘事实-判断-下一步’组织，并至少回应票型、查验、警徽或遗言中的一项具体信息。"
                )
                extra_lines.append("避免连续复用同一句模板，优先点名当前最关键的一名玩家。")
            if wolf_plan:
                preferred = wolf_plan.get("preferred_target_id")
                backup = wolf_plan.get("backup_target_id")
                if preferred:
                    line = f"夜间优先按主刀={preferred} 收敛"
                    if backup:
                        line += f"，备刀={backup}"
                    line += "；队友分裂时按该顺序确定。"
                    extra_lines.append(line)
        if extra_lines:
            instruction = instruction + "\n" + "\n".join(extra_lines)
        return instruction

    def _build_turn_packet(
        self,
        turn_packet: Mapping[str, Any],
        private: Mapping[str, Any],
        wolf_plan: dict[str, Any] | None,
    ) -> dict[str, Any]:
        prompt: dict[str, Any] = {
            "turn": self._compact_turn_view(turn_packet),
            "memory": self._memory_snapshot(),
            "history_policy": {
                "default_context": "compressed_state_plus_memory",
                "current_round_dialogue_tool": CURRENT_ROUND_DIALOGUE_TOOL_NAME,
                "max_tool_calls": self.max_tool_calls_per_decision,
                "max_tool_result_tokens": self.max_tool_result_tokens,
                "memory_round_limit": _MEMORY_ROUND_LIMIT,
            },
        }
        if wolf_plan:
            prompt["wolf_plan"] = wolf_plan
        # 将角色信息保持在 turn 视图中，方便模型把当前判断与自己的身份对齐，但不把
        # 整包原样透传。
        prompt["self_context"] = {
            "player_id": str(turn_packet.get("self", {}).get("player_id", self.player_id)),
            "role": str(private.get("role") or ""),
            "team": str(private.get("team") or ""),
        }
        return prompt

    def _compact_turn_view(self, turn_packet: Mapping[str, Any]) -> dict[str, Any]:
        view: dict[str, Any] = {}
        for key in ("game", "public_rules", "self", "private_information", "public_state", "request"):
            value = turn_packet.get(key)
            if isinstance(value, Mapping):
                exclude = frozenset({"players", "player_states", "history", "dialogue", "logs"})
                max_depth = 2 if key in {"private_information", "request"} else 1
                max_items = 10 if key in {"request", "public_state"} else 8
                max_chars = 140 if key in {"request", "private_information"} else 110
                compacted = _compact_mapping(
                    value,
                    max_depth=max_depth,
                    max_items=max_items,
                    max_chars=max_chars,
                    exclude_keys=exclude,
                )
                if _has_content(compacted):
                    view[key] = compacted
        tool_context = turn_packet.get("tool_context")
        if isinstance(tool_context, Mapping):
            current_dialogue = tool_context.get("current_round_dialogue")
            if _has_content(current_dialogue):
                view["tool_context"] = {
                    "current_round_dialogue": self._summarize_dialogue(current_dialogue)
                }
        return view

    def _summarize_dialogue(self, current_dialogue: object) -> list[object] | None:
        if not isinstance(current_dialogue, (list, tuple)):
            return None
        summary: list[object] = []
        for item in list(current_dialogue)[:6]:
            if isinstance(item, Mapping):
                summary.append(
                    _compact_mapping(
                        item,
                        max_depth=1,
                        max_items=6,
                        max_chars=120,
                        exclude_keys=frozenset({"players", "player_states", "history", "dialogue", "logs"}),
                    )
                )
            else:
                summary.append(_truncate_text(item, 160))
        return summary or None

    def _memory_snapshot(self) -> dict[str, Any]:
        snapshot: dict[str, Any] = {
            "recent_rounds": list(self._recent_rounds),
        }
        if self._own_actions:
            snapshot["own_public_actions"] = list(self._own_actions)
        if self._last_wolf_plan:
            snapshot["last_wolf_plan"] = self._last_wolf_plan
        return snapshot

    def _build_sync_snapshot(self, sync_packet: Mapping[str, Any]) -> dict[str, Any] | None:
        round_key = self._round_key_from_packet(sync_packet)
        phase = self._phase_from_packet(sync_packet)
        event = _compact_mapping(
            sync_packet,
            max_depth=1,
            max_items=10,
            max_chars=_MEMORY_TEXT_LIMIT,
            exclude_keys=frozenset({"players", "player_states", "history", "dialogue", "logs"}),
        )
        public_state = sync_packet.get("public_state")
        if isinstance(public_state, Mapping):
            public_state_view = _compact_mapping(
                public_state,
                max_depth=1,
                max_items=10,
                max_chars=_MEMORY_TEXT_LIMIT,
                exclude_keys=frozenset({"players", "player_states", "history", "dialogue", "logs"}),
            )
        else:
            public_state_view = {}
        snapshot: dict[str, Any] = {
            "round_key": round_key,
            "phase": phase,
            "notes": self._sync_notes(sync_packet, event, public_state_view),
        }
        actor_id = _first_present(
            sync_packet,
            "actor_id",
            "player_id",
            "speaker_id",
            "source_id",
            "from_player_id",
        )
        if actor_id is not None:
            snapshot["actor_id"] = str(actor_id)
        if actor_id is not None and str(actor_id) == self.player_id:
            snapshot["self_event"] = True
        return snapshot if _has_content(snapshot.get("notes")) or snapshot.get("self_event") else None

    def _sync_notes(
        self,
        sync_packet: Mapping[str, Any],
        event: Mapping[str, Any],
        public_state: Mapping[str, Any],
    ) -> list[str]:
        notes: list[str] = []
        round_key = self._round_key_from_packet(sync_packet)
        phase = self._phase_from_packet(sync_packet)
        if round_key:
            notes.append(f"r={round_key}")
        if phase:
            notes.append(f"p={phase}")

        event_kind = _first_present(event, "kind", "type", "action")
        if event_kind is not None:
            notes.append(f"event={_truncate_text(event_kind, 40)}")
        for key in ("result", "status", "reason"):
            value = event.get(key)
            if _has_content(value):
                notes.append(f"{key}={_truncate_text(value, 70)}")
        actor_id = _first_present(event, "actor_id", "player_id", "speaker_id", "source_id")
        if actor_id is not None:
            notes.append(f"actor={_truncate_text(actor_id, 24)}")
        target_id = _first_present(event, "target_id", "target")
        if target_id is not None:
            notes.append(f"target={_truncate_text(target_id, 24)}")
        text = _first_present(event, "text", "message")
        if text is not None:
            notes.append(f"text={_truncate_text(text, 80)}")

        for key in (
            "sheriff_id",
            "leader_id",
            "votes",
            "vote_result",
            "tied_target_ids",
            "dead_player_ids",
            "death_player_ids",
            "deaths",
            "alive_player_ids",
            "revealed_role",
            "revealed_roles",
            "no_kill",
        ):
            value = public_state.get(key)
            if _has_content(value):
                notes.append(f"{key}={_truncate_text(value, 80)}")

        return notes[:_MEMORY_NOTE_LIMIT]

    def _merge_sync_snapshot(self, snapshot: Mapping[str, Any]) -> None:
        round_key = str(snapshot.get("round_key") or "unknown")
        phase = str(snapshot.get("phase") or "")
        bucket = self._recent_rounds[-1] if self._recent_rounds and self._recent_rounds[-1].get("round_key") == round_key else None
        if bucket is None:
            bucket = {
                "round_key": round_key,
                "phase": phase,
                "notes": [],
            }
            self._recent_rounds.append(bucket)
        elif phase and not bucket.get("phase"):
            bucket["phase"] = phase

        notes = bucket.setdefault("notes", [])
        for note in snapshot.get("notes", []):
            if note not in notes:
                notes.append(note)
        if len(notes) > _MEMORY_NOTE_LIMIT:
            del notes[:-_MEMORY_NOTE_LIMIT]

        if snapshot.get("self_event"):
            self._append_own_action({
                "round_key": round_key,
                "phase": phase,
                "notes": list(snapshot.get("notes", [])),
            })

    def _append_own_action(self, action_snapshot: dict[str, Any]) -> None:
        if not action_snapshot:
            return
        self._own_actions.append(action_snapshot)
        if len(self._own_actions) > _MEMORY_ROUND_LIMIT:
            while len(self._own_actions) > _MEMORY_ROUND_LIMIT:
                self._own_actions.popleft()

    def _remember_decision(
        self,
        turn_packet: Mapping[str, Any],
        action: Mapping[str, Any],
        wolf_plan: dict[str, Any] | None,
    ) -> None:
        round_key = self._round_key_from_packet(turn_packet)
        phase = self._phase_from_packet(turn_packet)
        record: dict[str, Any] = {
            "round_key": round_key,
            "phase": phase,
            "kind": str(action.get("kind") or ""),
        }
        if action.get("target_id") is not None:
            record["target_id"] = str(action["target_id"])
        if isinstance(action.get("text"), str):
            record["text"] = _truncate_text(action["text"], 120)
        self._append_own_action(record)
        if wolf_plan:
            self._last_wolf_plan = wolf_plan

    def _wolf_plan(self, turn_packet: Mapping[str, Any], private: Mapping[str, Any]) -> dict[str, Any] | None:
        if str(private.get("role") or "") != "wolf":
            return None
        request = turn_packet.get("request")
        if not isinstance(request, Mapping):
            return None
        allowed_actions = request.get("allowed_actions")
        if not isinstance(allowed_actions, list):
            return None
        candidate_ids: list[str] = []
        for item in allowed_actions:
            if not isinstance(item, Mapping):
                continue
            target_ids = item.get("target_ids")
            if not isinstance(target_ids, list):
                continue
            for target_id in target_ids:
                target = str(target_id)
                if target not in candidate_ids and target != self.player_id:
                    candidate_ids.append(target)
        if not candidate_ids:
            return None

        excluded_ids = set(self._known_allies(private))
        filtered = [candidate for candidate in candidate_ids if candidate not in excluded_ids]
        if not filtered:
            filtered = candidate_ids

        round_key = self._round_key_from_packet(turn_packet)
        public_state = turn_packet.get("public_state") if isinstance(turn_packet.get("public_state"), Mapping) else {}
        threat_scores = {
            candidate: self._wolf_threat_score(candidate, public_state, round_key)
            for candidate in filtered
        }
        ordered = sorted(
            filtered,
            key=lambda candidate: (-threat_scores[candidate], self._stable_candidate_hash(round_key, candidate)),
        )
        preferred = ordered[0]
        backup = ordered[1] if len(ordered) > 1 else None
        reason_bits = [f"score[{preferred}]={threat_scores[preferred]}"]
        if backup:
            reason_bits.append(f"backup={backup}")
        if len(ordered) > 2:
            reason_bits.append(f"pool={len(ordered)}")
        return {
            "role": "wolf",
            "round_key": round_key,
            "preferred_target_id": preferred,
            "backup_target_id": backup,
            "candidate_order": ordered[:5],
            "reason": ";".join(reason_bits),
        }

    def _known_allies(self, private: Mapping[str, Any]) -> list[str]:
        for key in ("teammates", "allies", "partners", "known_wolves", "wolf_team", "team_mates"):
            value = private.get(key)
            if isinstance(value, (list, tuple, set)):
                allies = [str(item) for item in value if str(item)]
                if allies:
                    return allies
        return []

    def _wolf_threat_score(
        self,
        candidate: str,
        public_state: Mapping[str, Any],
        round_key: str,
    ) -> int:
        score = 0
        for key in ("sheriff_id", "leader_id", "vote_result", "tied_target_ids", "votes", "claims", "revealed_roles"):
            value = public_state.get(key)
            text = _truncate_text(value, 240)
            if candidate and candidate in text:
                score += 2
        if candidate in _truncate_text(public_state.get("dead_player_ids"), 240):
            score -= 100
        if candidate in _truncate_text(public_state.get("death_player_ids"), 240):
            score -= 100
        if candidate in _truncate_text(public_state.get("alive_player_ids"), 240):
            score += 1
        for round_snapshot in self._recent_rounds:
            for note in round_snapshot.get("notes", []):
                note_text = str(note)
                if candidate not in note_text:
                    continue
                if any(token in note_text for token in ("sheriff", "leader", "claim", "查验", "警徽", "票", "vote")):
                    score += 3
                if any(token in note_text for token in ("dead", "死亡", "出局", "night", "night_kill", "刀")):
                    score += 1
                if any(token in note_text for token in ("rev", "false", "假", "冲", "打")):
                    score += 1
        if round_key:
            score += int(hashlib.sha256(f"{round_key}:{candidate}".encode("utf-8")).hexdigest(), 16) % 3
        return score

    @staticmethod
    def _stable_candidate_hash(round_key: str, candidate: str) -> str:
        return hashlib.sha256(f"{round_key}:{candidate}".encode("utf-8")).hexdigest()

    @staticmethod
    def _round_key_from_packet(packet: Mapping[str, Any]) -> str:
        game = packet.get("game")
        public_state = packet.get("public_state")
        request = packet.get("request")
        parts: list[str] = []
        for source in (game, public_state, request, packet):
            if not isinstance(source, Mapping):
                continue
            value = _first_present(source, "round", "day", "phase", "public_phase", "turn", "step")
            if value is not None:
                parts.append(_truncate_text(value, 24))
            if parts:
                break
        if not parts:
            return "unknown"
        return "|".join(parts)

    @staticmethod
    def _phase_from_packet(packet: Mapping[str, Any]) -> str:
        for source in (
            packet.get("request"),
            packet.get("game"),
            packet.get("public_state"),
            packet,
        ):
            if isinstance(source, Mapping):
                value = _first_present(source, "phase", "public_phase", "stage")
                if value is not None:
                    return str(value)
        return ""

    @staticmethod
    def _request_has_speech_action(request: Mapping[str, Any]) -> bool:
        raw_actions = request.get("allowed_actions")
        if not isinstance(raw_actions, list):
            return False
        return any(
            isinstance(item, Mapping) and str(item.get("kind") or "") in _SPEECH_ACTION_KINDS
            for item in raw_actions
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

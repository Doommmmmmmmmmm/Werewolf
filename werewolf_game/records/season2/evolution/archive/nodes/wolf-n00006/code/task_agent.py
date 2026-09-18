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
_TEXT_NORMALIZER = re.compile(r"[\s\u3000，。！？、,.!?;；:：\"'“”‘’`·…、/\\|()（）\[\]{}<>-]+")

CURRENT_ROUND_DIALOGUE_TOOL_NAME = "read_current_round_dialogue"
DEFAULT_MAX_TOOL_CALLS_PER_DECISION = 5
DEFAULT_MAX_TOOL_RESULT_TOKENS = 1000
DEFAULT_MAX_PROMPT_CHARS = 12000
# 这是 Task-Agent 层的“纠错重试”次数：首次模型调用之外，最多再请求两次。
# 传输层的 HTTP/网络重试仍由 ModelClient 的 MODEL_MAX_RETRIES 单独控制。
DEFAULT_MAX_DECISION_RETRIES = 2
_SPEECH_ACTION_KINDS = frozenset({"speak", "last_words"})
_PUBLIC_MEMORY_HISTORY_LIMIT = 4
_RECENT_SELF_ACTION_LIMIT = 4
_RECENT_NIGHT_TARGET_LIMIT = 3
_GENERIC_SPEECH_TEMPLATES = frozenset(
    {
        "先听一圈",
        "看票型",
        "我先过",
        "先过",
        "别乱踩",
        "暂时没想法",
        "先观望",
        "我过一下",
        "先不站边",
        "看后面发言",
        "别急",
        "先听发言",
        "等下再说",
    }
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
        self._public_memory_history: deque[dict[str, Any]] = deque(maxlen=_PUBLIC_MEMORY_HISTORY_LIMIT)
        self._recent_self_speeches: deque[str] = deque(maxlen=_RECENT_SELF_ACTION_LIMIT)
        self._recent_night_targets: deque[str] = deque(maxlen=_RECENT_NIGHT_TARGET_LIMIT)
        self._last_public_memory_signature = ""

    async def observe(self, sync_packet: dict[str, Any]) -> None:
        """接收公开同步，并压缩保存为跨回合轻量记忆。"""

        summary = self._summarize_sync_packet(sync_packet)
        signature = self._stable_signature(summary)
        if signature and signature == self._last_public_memory_signature:
            return
        self._last_public_memory_signature = signature
        self._public_memory_history.append(summary)

    async def decide(self, turn_packet: dict[str, Any]) -> dict[str, Any]:
        if turn_packet["self"]["player_id"] != self.player_id:
            raise ValueError("行动包被发送给了错误玩家")
        private = turn_packet["private_information"]
        if str(private.get("role")) != self.profile.role:
            raise ValueError("Task-Agent 的角色档案与行动包身份不一致")

        system = self._system_prompt(private)
        prompt = self._build_prompt(turn_packet)
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
                "instruction": self._turn_instruction(turn_packet, feedback),
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
            semantic_error = self._semantic_action_error(action, turn_packet)
            if semantic_error is None:
                error = decision_error(action, turn_packet["request"])
            else:
                error = semantic_error
            if error is None:
                self._record_successful_action(action)
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

    def _build_prompt(self, turn_packet: Mapping[str, Any]) -> dict[str, Any]:
        game = turn_packet.get("game", {})
        public_state = turn_packet.get("public_state", {})
        request = turn_packet.get("request", {})
        self_state = turn_packet.get("self", {})
        private = turn_packet.get("private_information", {})
        compact_request = self._compact_request(request)
        return {
            "game": self._compact_value(game, max_depth=1, max_items=8, max_string=120),
            "public_rules": self._compact_value(
                turn_packet.get("public_rules", {}), max_depth=1, max_items=8, max_string=120
            ),
            "self": self._compact_value(self_state, max_depth=1, max_items=8, max_string=120),
            "private_information": self._compact_value(
                private, max_depth=1, max_items=8, max_string=120
            ),
            "public_state": self._compact_value(
                public_state, max_depth=1, max_items=8, max_string=120
            ),
            "request": compact_request,
            "memory": {
                "public_history": list(self._public_memory_history),
                "recent_self_speeches": list(self._recent_self_speeches),
                "recent_night_targets": list(self._recent_night_targets),
            },
            "history_policy": {
                "default_context": "compact_state_plus_light_memory",
                "current_round_dialogue_tool": CURRENT_ROUND_DIALOGUE_TOOL_NAME,
                "max_tool_calls": self.max_tool_calls_per_decision,
                "max_tool_result_tokens": self.max_tool_result_tokens,
                "memory_slots": _PUBLIC_MEMORY_HISTORY_LIMIT,
                "self_action_slots": _RECENT_SELF_ACTION_LIMIT,
            },
        }

    @staticmethod
    def _turn_instruction(turn_packet: Mapping[str, Any], feedback: str) -> str:
        request = turn_packet["request"]
        phase = str(turn_packet.get("game", {}).get("public_phase", turn_packet.get("game", {}).get("phase", "unknown")))
        guardrails = [
            "按当前行动包作答，不要补充解释。",
            f"当前阶段：{phase}。",
            "夜谈先给一句短判断；夜刀给主目标 + 备选并说明原因。",
            "白天发言必须回应具体人名、票型或上一轮公开信息。",
            "投票时明确跟谁/反谁以及原因。",
            "尽量避免重复最近自己的开场和最近的夜刀目标。",
        ]
        return render_prompt(
            "player_turn_instruction.txt",
            action_contract=render_action_contract(request),
            validation_feedback=(f"上一次输出未通过校验：{feedback}" if feedback else ""),
            playbook="\n".join(f"- {item}" for item in guardrails),
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

    def _record_successful_action(self, action: Mapping[str, Any]) -> None:
        kind = str(action.get("kind") or "")
        if kind in _SPEECH_ACTION_KINDS:
            text = str(action.get("text") or "").strip()
            if text:
                normalized = self._normalize_text(text)
                if normalized:
                    self._recent_self_speeches.append(normalized)
        target_id = action.get("target_id")
        if target_id:
            self._recent_night_targets.append(str(target_id))

    def _semantic_action_error(self, action: Mapping[str, Any], turn_packet: Mapping[str, Any]) -> str | None:
        kind = str(action.get("kind") or "")
        if kind not in _SPEECH_ACTION_KINDS:
            return None
        text = str(action.get("text") or "").strip()
        normalized = self._normalize_text(text)
        if not normalized:
            return f"{kind} 必须提供非空 text"
        if self._is_repeated_speech(normalized):
            return "发言不要重复自己最近的说法"
        if self._looks_generic_opening(normalized):
            return "发言太泛，请点到当前人名、票型或公开信息"
        if not self._has_current_reference(text, turn_packet):
            return "发言需要提到当前局内的人名、票型或轮次信息"
        return None

    def _compact_request(self, request: Mapping[str, Any]) -> Any:
        compact = self._compact_value(request, max_depth=2, max_items=8, max_string=120)
        if not isinstance(compact, dict):
            return compact
        allowed_actions = compact.get("allowed_actions")
        if not isinstance(allowed_actions, list):
            return compact
        recent_targets = {self._normalize_text(target) for target in self._recent_night_targets}
        prioritized_actions: list[Any] = []
        for item in allowed_actions:
            if not isinstance(item, dict):
                prioritized_actions.append(item)
                continue
            target_ids = item.get("target_ids")
            if isinstance(target_ids, list) and target_ids:
                deduped: list[Any] = []
                seen: set[str] = set()
                repeated: list[Any] = []
                for target_id in target_ids:
                    normalized = self._normalize_text(str(target_id))
                    if normalized in seen:
                        continue
                    seen.add(normalized)
                    if normalized in recent_targets:
                        repeated.append(target_id)
                    else:
                        deduped.append(target_id)
                item = dict(item)
                item["target_ids"] = deduped + repeated
            prioritized_actions.append(item)
        compact["allowed_actions"] = prioritized_actions
        return compact

    def _summarize_sync_packet(self, sync_packet: Mapping[str, Any]) -> dict[str, Any]:
        summary: dict[str, Any] = {}
        fields: dict[str, tuple[Any, ...]] = {
            "round": ("round", ("game", "round"), ("state", "round")),
            "phase": ("phase", "public_phase", ("game", "phase"), ("game", "public_phase")),
            "alive_players": (
                "alive_players",
                "alive",
                "living_players",
                ("state", "alive_players"),
                ("game", "alive_players"),
            ),
            "dead_players": (
                "dead_players",
                "dead",
                "eliminated_players",
                ("state", "dead_players"),
                ("game", "dead_players"),
            ),
            "sheriff": (
                "sheriff",
                "police",
                "captain",
                "leader",
                ("state", "sheriff"),
                ("game", "sheriff"),
            ),
            "votes": (
                "votes",
                "vote",
                "vote_records",
                "voting",
                ("state", "votes"),
                ("game", "votes"),
            ),
            "dialogue": (
                "dialogue",
                "public_dialogue",
                "messages",
                "events",
                "announcements",
                ("state", "dialogue"),
                ("game", "dialogue"),
            ),
        }
        for field, aliases in fields.items():
            value = self._first_present(sync_packet, aliases)
            if value is not None:
                summary[field] = self._compact_value(value, max_depth=1, max_items=6, max_string=100)
        if not summary:
            summary["snapshot"] = self._compact_value(sync_packet, max_depth=2, max_items=6, max_string=100)
        else:
            summary["snapshot"] = self._compact_value(sync_packet, max_depth=1, max_items=6, max_string=100)
        return summary

    def _has_current_reference(self, text: str, turn_packet: Mapping[str, Any]) -> bool:
        normalized_text = self._normalize_text(text)
        reference_tokens = self._collect_reference_tokens(
            turn_packet.get("game"),
            turn_packet.get("public_state"),
            turn_packet.get("request"),
            self._public_memory_history[-1] if self._public_memory_history else None,
        )
        if any(token and token in normalized_text for token in reference_tokens):
            return True
        return False

    def _is_repeated_speech(self, normalized_text: str) -> bool:
        return normalized_text in set(self._recent_self_speeches)

    def _looks_generic_opening(self, normalized_text: str) -> bool:
        return normalized_text in self._normalize_text_set(_GENERIC_SPEECH_TEMPLATES)

    def _collect_reference_tokens(self, *values: Any) -> list[str]:
        tokens: list[str] = []
        seen: set[str] = set()

        def visit(value: Any) -> None:
            if len(tokens) >= 24:
                return
            if isinstance(value, Mapping):
                for key, item in value.items():
                    if isinstance(key, str):
                        normalized_key = self._normalize_text(key)
                        if normalized_key and normalized_key not in seen and len(normalized_key) <= 12:
                            seen.add(normalized_key)
                            tokens.append(normalized_key)
                            if len(tokens) >= 24:
                                return
                    visit(item)
            elif isinstance(value, (list, tuple, set)):
                for item in value:
                    visit(item)
                    if len(tokens) >= 24:
                        return
            elif isinstance(value, str):
                normalized = self._normalize_text(value)
                if (
                    normalized
                    and normalized not in seen
                    and len(normalized) <= 16
                    and (normalized.isdigit() or _CHINESE_CHARACTER.search(normalized) or re.search(r"[A-Za-z]", normalized))
                ):
                    seen.add(normalized)
                    tokens.append(normalized)
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                normalized = str(int(value) if isinstance(value, float) and value.is_integer() else value)
                if normalized not in seen:
                    seen.add(normalized)
                    tokens.append(normalized)

        for value in values:
            visit(value)
            if len(tokens) >= 24:
                break
        return tokens

    @staticmethod
    def _first_present(mapping: Mapping[str, Any], aliases: tuple[Any, ...]) -> Any:
        for alias in aliases:
            if isinstance(alias, tuple):
                current: Any = mapping
                found = True
                for part in alias:
                    if not isinstance(current, Mapping) or part not in current:
                        found = False
                        break
                    current = current[part]
                if found and current is not None:
                    return current
            elif alias in mapping and mapping[alias] is not None:
                return mapping[alias]
        return None

    def _compact_value(
        self,
        value: Any,
        *,
        max_depth: int,
        max_items: int,
        max_string: int,
    ) -> Any:
        if max_depth < 0:
            return self._stringify_value(value, max_string=max_string)
        if isinstance(value, Mapping):
            compact: dict[str, Any] = {}
            for index, (key, item) in enumerate(value.items()):
                if index >= max_items:
                    compact["__truncated__"] = True
                    break
                compact[str(key)] = self._compact_value(
                    item,
                    max_depth=max_depth - 1,
                    max_items=max_items,
                    max_string=max_string,
                )
            return compact
        if isinstance(value, list):
            compact_list = [
                self._compact_value(
                    item,
                    max_depth=max_depth - 1,
                    max_items=max_items,
                    max_string=max_string,
                )
                for item in value[:max_items]
            ]
            if len(value) > max_items:
                compact_list.append("__truncated__")
            return compact_list
        if isinstance(value, tuple):
            return self._compact_value(
                list(value),
                max_depth=max_depth,
                max_items=max_items,
                max_string=max_string,
            )
        if isinstance(value, str):
            return self._truncate_string(value, max_string)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return value
        if value is None or isinstance(value, bool):
            return value
        return self._truncate_string(str(value), max_string)

    @staticmethod
    def _stringify_value(value: Any, *, max_string: int) -> Any:
        if isinstance(value, str):
            return value[:max_string]
        return str(value)[:max_string]

    @staticmethod
    def _truncate_string(text: str, max_string: int) -> str:
        if len(text) <= max_string:
            return text
        if max_string <= 1:
            return text[:max_string]
        return text[: max_string - 1] + "…"

    @staticmethod
    def _normalize_text(text: str) -> str:
        return _TEXT_NORMALIZER.sub("", text).lower()

    @classmethod
    def _normalize_text_set(cls, texts: set[str] | frozenset[str]) -> set[str]:
        return {cls._normalize_text(text) for text in texts}

    @staticmethod
    def _stable_signature(value: Any) -> str:
        try:
            payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        except TypeError:
            payload = json.dumps(str(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _nonnegative_int(value: object, *, fallback: int | None) -> int | None:
        if isinstance(value, bool):
            return fallback
        try:
            normalized = int(value)
        except (TypeError, ValueError):
            return fallback
        return normalized if normalized >= 0 else fallback

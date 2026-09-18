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
        if kind == "guard_protect" and valid_targets:
            lines.append("  守卫必须只用当前 target_ids；优先避开上一晚保护对象，若原目标非法就改选合法候选。")
        if kind in _SPEECH_ACTION_KINDS:
            max_chars = allowed.get("max_chars")
            constraints: list[str] = []
            if max_chars is not None:
                constraints.append(f"text 最多 {max_chars} 个字符")
            if allowed.get("require_chinese"):
                constraints.append("text 至少包含一个中文字符")
            constraints.append("text 可以包含英文、数字和玩家编号")
            lines.append("  " + ";".join(constraints) + "。")

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


def decision_error(
    action: dict[str, Any],
    request: dict[str, Any],
    guard_state: Mapping[str, Any] | None = None,
) -> str | None:
    """在提交引擎前做一次本地校验，便于让模型重试。"""

    allowed = next(
        (
            item
            for item in request["allowed_actions"]
            if isinstance(item, Mapping) and item.get("kind") == action["kind"]
        ),
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
    raw_target_ids = allowed.get("target_ids")
    target_ids = raw_target_ids if isinstance(raw_target_ids, list) else None
    if target_ids:
        if action.get("target_id") not in target_ids:
            return "target_id 必须是 target_ids 中的一项"
        if action["kind"] == "guard_protect" and guard_state is not None:
            last_protected_id = guard_state.get("last_protected_id")
            if last_protected_id and action.get("target_id") == last_protected_id and len(target_ids) > 1:
                return f"guard_protect 不能连续守同一人：last_protected_id={last_protected_id}"
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
        self._guard_memory: dict[str, Any] = {
            "last_protected_id": None,
            "last_night_round": None,
            "last_public_confirmed_guard_target": None,
        }

    async def observe(self, sync_packet: dict[str, Any]) -> None:
        """接收公开同步；仅保留守卫决策所需的最小状态。"""

        self._update_guard_memory(sync_packet)

    async def decide(self, turn_packet: dict[str, Any]) -> dict[str, Any]:
        if turn_packet["self"]["player_id"] != self.player_id:
            raise ValueError("行动包被发送给了错误玩家")
        private = turn_packet["private_information"]
        if str(private.get("role")) != self.profile.role:
            raise ValueError("Task-Agent 的角色档案与行动包身份不一致")

        self._update_guard_memory(turn_packet)
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
                "instruction": self._turn_instruction(turn_packet["request"], feedback),
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
            action, repair_note = self._repair_guard_action(action, turn_packet)
            error = decision_error(action, turn_packet["request"], self._guard_memory)
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
            feedback = self._build_validation_feedback(action, turn_packet["request"], error, repair_note)

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

    def _update_guard_memory(self, source: Any) -> None:
        if not isinstance(source, Mapping):
            return
        candidates: list[Mapping[str, Any]] = [source]
        for key in ("public_state", "private_information", "sync_packet", "role_state"):
            nested = source.get(key)
            if isinstance(nested, Mapping):
                candidates.append(nested)
                nested_role_state = nested.get("role_state")
                if isinstance(nested_role_state, Mapping):
                    candidates.append(nested_role_state)
        for candidate in candidates:
            self._merge_guard_state(candidate)

    def _merge_guard_state(self, state: Mapping[str, Any]) -> None:
        last_protected_id = self._first_str(
            state.get("last_protected_id"),
            state.get("last_protected_target_id"),
            state.get("protected_target_id"),
            state.get("protected_id"),
        )
        if last_protected_id is not None:
            self._guard_memory["last_protected_id"] = last_protected_id

        confirmed_target = self._first_str(
            state.get("last_public_confirmed_guard_target"),
            state.get("confirmed_guard_target"),
            state.get("public_confirmed_guard_target"),
        )
        if confirmed_target is not None:
            self._guard_memory["last_public_confirmed_guard_target"] = confirmed_target

        last_round = self._first_int(
            state.get("last_night_round"),
            state.get("last_guard_round"),
            state.get("night_round"),
            state.get("round"),
        )
        if last_round is not None:
            self._guard_memory["last_night_round"] = last_round

    def _repair_guard_action(
        self, action: dict[str, Any], turn_packet: dict[str, Any]
    ) -> tuple[dict[str, Any], str | None]:
        if action.get("kind") != "guard_protect":
            return action, None
        allowed = next(
            (
                item
                for item in turn_packet["request"].get("allowed_actions", [])
                if isinstance(item, Mapping) and item.get("kind") == "guard_protect"
            ),
            None,
        )
        if not isinstance(allowed, Mapping):
            return action, None
        raw_target_ids = allowed.get("target_ids")
        if not isinstance(raw_target_ids, list):
            return action, None
        target_ids = [str(target_id) for target_id in raw_target_ids if target_id is not None]
        if not target_ids:
            return action, None

        chosen = action.get("target_id")
        chosen_id = str(chosen) if chosen is not None else None
        last_protected_id = self._guard_memory.get("last_protected_id")
        if chosen_id in target_ids and chosen_id != last_protected_id:
            self._guard_memory["last_protected_id"] = chosen_id
            self._guard_memory["last_night_round"] = self._current_round(turn_packet)
            return action, None

        replacement = self._choose_guard_target(target_ids, turn_packet)
        if replacement is None:
            return action, None
        repaired = dict(action)
        repaired["target_id"] = replacement
        self._guard_memory["last_protected_id"] = replacement
        self._guard_memory["last_night_round"] = self._current_round(turn_packet)

        if chosen_id is None:
            note = f"guard_protect 缺少 target_id，已自动改写为 {replacement}。"
        elif chosen_id not in target_ids:
            note = (
                f"guard_protect 的 target_id={chosen_id} 不在当前合法 target_ids 中，"
                f"已自动改写为 {replacement}。"
            )
        else:
            note = (
                f"guard_protect 不能连续守同一人：last_protected_id={last_protected_id}，"
                f"已自动改写为 {replacement}。"
            )
        return repaired, note

    def _choose_guard_target(self, target_ids: list[str], turn_packet: dict[str, Any]) -> str | None:
        if not target_ids:
            return None
        last_protected_id = str(self._guard_memory.get("last_protected_id") or "")
        confirmed = str(self._guard_memory.get("last_public_confirmed_guard_target") or "")
        alive_ids = self._extract_alive_player_ids(turn_packet.get("public_state"))
        candidates = [target_id for target_id in target_ids if target_id != last_protected_id]
        if not candidates:
            candidates = list(target_ids)
        if alive_ids:
            alive_candidates = [target_id for target_id in candidates if target_id in alive_ids]
            if alive_candidates:
                candidates = alive_candidates
        return sorted(
            candidates,
            key=lambda target_id: (
                0 if confirmed and target_id == confirmed else 1,
                self._seat_order_key(target_id),
                target_id,
            ),
        )[0]

    @staticmethod
    def _extract_alive_player_ids(public_state: Any) -> set[str]:
        alive_ids: set[str] = set()
        if not isinstance(public_state, Mapping):
            return alive_ids
        for key in ("alive_player_ids", "living_player_ids", "alive_ids"):
            value = public_state.get(key)
            if isinstance(value, list):
                alive_ids.update(str(item) for item in value if item is not None)
        players = public_state.get("players")
        if isinstance(players, list):
            for player in players:
                if not isinstance(player, Mapping):
                    continue
                if player.get("alive") is False or player.get("is_alive") is False:
                    continue
                if str(player.get("status") or "").lower() in {"dead", "eliminated"}:
                    continue
                player_id = player.get("player_id", player.get("id"))
                if player_id is not None:
                    alive_ids.add(str(player_id))
        return alive_ids

    @staticmethod
    def _seat_order_key(player_id: object) -> tuple[int, str]:
        text = str(player_id or "")
        match = re.search(r"(\d+)$", text)
        return (int(match.group(1)) if match else 10**9, text)

    @staticmethod
    def _first_str(*values: object) -> str | None:
        for value in values:
            if value is None:
                continue
            text = str(value).strip()
            if text:
                return text
        return None

    @staticmethod
    def _first_int(*values: object) -> int | None:
        for value in values:
            if value is None or isinstance(value, bool):
                continue
            try:
                normalized = int(value)
            except (TypeError, ValueError):
                continue
            return normalized
        return None

    def _current_round(self, turn_packet: Mapping[str, Any]) -> int | None:
        game = turn_packet.get("game")
        if not isinstance(game, Mapping):
            return None
        return self._first_int(game.get("round"))

    def _build_validation_feedback(
        self,
        action: Mapping[str, Any],
        request: Mapping[str, Any],
        error: str,
        repair_note: str | None,
    ) -> str:
        if action.get("kind") == "guard_protect":
            allowed = next(
                (
                    item
                    for item in request.get("allowed_actions", [])
                    if isinstance(item, Mapping) and item.get("kind") == "guard_protect"
                ),
                None,
            )
            target_ids = []
            if isinstance(allowed, Mapping):
                raw_target_ids = allowed.get("target_ids")
                if isinstance(raw_target_ids, list):
                    target_ids = [str(target_id) for target_id in raw_target_ids if target_id is not None]
            attempted_target = action.get("target_id")
            last_protected_id = self._guard_memory.get("last_protected_id")
            parts = [f"guard_protect 校验失败：{error}。"]
            if repair_note:
                parts.append(f"本地已尝试修复：{repair_note}")
            if target_ids:
                parts.append("当前合法 target_ids=" + "、".join(target_ids) + "。")
            if last_protected_id:
                parts.append(f"last_protected_id={last_protected_id}。")
            if attempted_target is not None:
                parts.append(f"请从合法候选中重选，不要再次给出 {attempted_target} 这类非法目标。")
            else:
                parts.append("请从合法候选中重选，不要再次输出缺少 target_id 的守护行动。")
            return " ".join(parts)
        return f"上一次行动未通过校验：{error}"

    def _system_prompt(self, private: Mapping[str, Any]) -> str:
        role_task = self.profile.task
        if str(private.get("role")) == "guard":
            guard_note = [
                "守卫约束：必须使用当前 allowed target_ids；必须记住上一晚保护对象，严禁连守同一人。",
                "若模型原目标非法或与 last_protected_id 冲突，先改成合法候选再输出；不要把守护成功当作确认事实。",
            ]
            if self._guard_memory.get("last_protected_id"):
                guard_note.append(f"已知上一晚保护对象：{self._guard_memory['last_protected_id']}。")
            role_task = role_task + "\n\n" + " ".join(guard_note)
        return render_prompt(
            "player_system.txt",
            player_id=self.player_id,
            role=private["role"],
            team=private["team"],
            persona=self.persona,
            role_base=self.profile.base,
            role_task=role_task,
            tool_name=CURRENT_ROUND_DIALOGUE_TOOL_NAME,
            max_tool_calls=self.max_tool_calls_per_decision,
            max_tool_result_tokens=self.max_tool_result_tokens,
            max_prompt_chars=self.max_prompt_chars,
        )

    @staticmethod
    def _turn_instruction(request: Mapping[str, Any], feedback: str) -> str:
        return render_prompt(
            "player_turn_instruction.txt",
            action_contract=render_action_contract(request),
            validation_feedback=(f"上一次输出未通过校验：{feedback}" if feedback else ""),
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

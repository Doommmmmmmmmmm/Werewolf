"""最小可运行的角色 Task-Agent。

每次行动从一个新的模型会话开始。模型默认只收到当前状态；本轮对话只能通过唯一的
受限工具按需读取。这里不包含策略进化、长期记忆、外部检索或其他 Harness。
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import re
from collections.abc import Mapping
from typing import Any

from ..core.errors import ModelClientError, RuleViolationError
from .llm.coordinator import ModelRequestCoordinator
from ..prompts import RoleProfile, render_prompt


_CHINESE_CHARACTER = re.compile(r"[\u3400-\u9fff]")
_PLAYER_ID_PATTERN = re.compile(r"p\d+", re.IGNORECASE)

CURRENT_ROUND_DIALOGUE_TOOL_NAME = "read_current_round_dialogue"
DEFAULT_MAX_TOOL_CALLS_PER_DECISION = 5
DEFAULT_MAX_TOOL_RESULT_TOKENS = 1000
DEFAULT_MAX_PROMPT_CHARS = 12000
# 这是 Task-Agent 层的“纠错重试”次数：首次模型调用之外，最多再请求两次。
# 传输层的 HTTP/网络重试仍由 ModelClient 的 MODEL_MAX_RETRIES 单独控制。
DEFAULT_MAX_DECISION_RETRIES = 2
_SPEECH_ACTION_KINDS = frozenset({"speak", "last_words"})
_PASS_ACTION_KINDS = frozenset({"pass", "skip", "noop", "do_nothing", "none"})
_WITCH_POISON_MIN_SCORE = 4
_WITCH_POISON_MIN_EVIDENCE_TYPES = 2
_WITCH_HEAL_MIN_SCORE = 2


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
        self._memory_game_id: str | None = None
        self._recent_public_summaries: list[str] = []
        self._last_public_round_phase: str = ""
        self._last_private_resource_summary: str = ""
        self._last_decision_audit: dict[str, Any] | None = None
        self._witch_evidence_ledger: dict[str, dict[str, Any]] = {}
        self._witch_recent_public_deaths: list[str] = []
        self._witch_recent_vote_pressure: list[str] = []
        self._witch_recent_counterclaims: list[str] = []
        self._witch_recent_role_chain: list[str] = []

    async def observe(self, sync_packet: dict[str, Any]) -> None:
        """接收公开同步，并压缩保留少量与后续决策有关的摘要。"""

        game_id = self._extract_first_scalar(
            sync_packet,
            ("game_id", "match_id", "session_id", "table_id", "room_id"),
        )
        if game_id and game_id != self._memory_game_id:
            self._reset_compact_memory(game_id)

        digest = self._build_public_digest(sync_packet)
        if digest:
            if not self._recent_public_summaries or self._recent_public_summaries[-1] != digest:
                self._recent_public_summaries.append(digest)
                self._recent_public_summaries = self._recent_public_summaries[-3:]
        if self.profile.role == "witch" or str(sync_packet.get("role")) == "witch":
            self._update_witch_memory(sync_packet)
        self._last_public_round_phase = self._build_round_phase_tag(sync_packet)

    async def decide(self, turn_packet: dict[str, Any]) -> dict[str, Any]:
        if turn_packet["self"]["player_id"] != self.player_id:
            raise ValueError("行动包被发送给了错误玩家")
        private = turn_packet["private_information"]
        if str(private.get("role")) != self.profile.role:
            raise ValueError("Task-Agent 的角色档案与行动包身份不一致")

        private_state = self._build_private_resource_summary(private)
        if private_state:
            self._last_private_resource_summary = private_state
        witch_context = self._build_witch_decision_context(private, turn_packet)
        compact_memory = self._compact_memory_snapshot(witch_context)
        safety_notes = self._build_safety_notes(private, witch_context)

        system = self._system_prompt(private)
        prompt = {
            "game": turn_packet["game"],
            "public_rules": turn_packet["public_rules"],
            "self": turn_packet["self"],
            "private_information": private,
            "public_state": turn_packet["public_state"],
            "request": turn_packet["request"],
            "compact_memory": compact_memory,
            "private_resource_state": private_state,
            "witch_decision_context": witch_context,
            "safety_notes": safety_notes,
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
            action = self._apply_role_safe_fallback(action, turn_packet, raw, witch_context)
            error = decision_error(action, turn_packet["request"])
            if error is None:
                self._last_decision_audit = {
                    "raw_response": raw,
                    "normalized_action": action,
                }
                return action
            if _attempt >= self.max_decision_retries:
                # 只保留规范化后的行动字段；完整原始响应和推理文本不会进入记录。
                # Runner 可将该行动存入管理员审计记录，便于定位 schema / 长度等问题，
                # 而不会泄露给公开记录。
                self._last_decision_audit = {
                    "raw_response": raw,
                    "normalized_action": action,
                    "validation_error": error,
                }
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

    def _compact_memory_snapshot(self, witch_context: Mapping[str, Any] | None = None) -> dict[str, Any]:
        snapshot: dict[str, Any] = {
            "recent_public": self._recent_public_summaries[-3:],
        }
        if self._last_public_round_phase:
            snapshot["last_public_round_phase"] = self._last_public_round_phase
        if self._last_private_resource_summary:
            snapshot["private_resource_state"] = self._last_private_resource_summary
        if witch_context:
            snapshot["witch_decision_summary"] = self._compact_witch_context(witch_context)
        if self.profile.role == "witch" and self._witch_evidence_ledger:
            snapshot["witch_top_suspects"] = self._build_witch_ledger_snapshot()
        return snapshot

    def _extract_private_wolf_target(self, private: Mapping[str, Any]) -> dict[str, Any]:
        """Read only the engine's explicit private wolf-target field.

        In particular, ``target_id`` by itself is deliberately not accepted: it is
        commonly the target of an unrelated action or a public event.  The three
        states are useful to the decision gate: a missing field is not the same as
        an explicitly empty night.
        """
        aliases = {
            "wolf_target", "night_kill_target", "night_attack_target",
            "attack_target", "attacked_player_id", "victim_id",
        }
        found = False
        targets: list[str] = []

        def scalar_ids(value: Any) -> list[str]:
            if value is None:
                return []
            if isinstance(value, Mapping):
                ids: list[str] = []
                for key in ("player_id", "victim_id", "attacked_player_id", "target_id"):
                    if key in value:
                        ids.extend(scalar_ids(value[key]))
                return list(dict.fromkeys(ids))
            if isinstance(value, list):
                ids: list[str] = []
                for item in value:
                    ids.extend(scalar_ids(item))
                return list(dict.fromkeys(ids))
            text = str(value).strip()
            if not text or text.lower() in {"none", "null", "no_target", "no target", "无", "没有", "nil"}:
                return []
            return self._extract_player_ids(text)

        def walk(node: Any) -> None:
            nonlocal found, targets
            if isinstance(node, Mapping):
                for key, value in node.items():
                    key_text = re.sub(r"(?<!^)([A-Z])", r"_\1", str(key)).lower()
                    if key_text in aliases:
                        found = True
                        targets.extend(scalar_ids(value))
                    elif isinstance(value, (Mapping, list)):
                        walk(value)
            elif isinstance(node, list):
                for item in node:
                    walk(item)

        walk(private)
        targets = list(dict.fromkeys(targets))
        return {
            "status": "concrete" if targets else ("none" if found else "unknown"),
            "target_ids": targets,
        }

    def _extract_private_resource_state(self, private: Mapping[str, Any]) -> dict[str, bool | None]:
        aliases = {
            "antidote_available": ("antidote_available", "save_available", "heal_available"),
            "poison_available": ("poison_available", "poison_left", "kill_available"),
            "antidote_used": ("antidote_used", "used_antidote", "save_used"),
            "poison_used": ("poison_used", "used_poison", "kill_used"),
        }
        return {
            name: self._coerce_bool_scalar(self._extract_first_scalar(private, hints))
            for name, hints in aliases.items()
        }

    def _build_private_resource_summary(self, private: Mapping[str, Any]) -> str:
        if self.profile.role != "witch" and str(private.get("role")) != "witch":
            return ""
        keys = self._extract_private_resource_state(private)
        parts: list[str] = []
        for key, value in keys.items():
            if value is not None:
                parts.append(f"{key}={self._compact_scalar(value)}")
        return "; ".join(parts)

    def _build_witch_decision_context(
        self,
        private: Mapping[str, Any],
        turn_packet: Mapping[str, Any],
    ) -> dict[str, Any]:
        if self.profile.role != "witch" and str(private.get("role")) != "witch":
            return {}

        request = turn_packet.get("request") if isinstance(turn_packet.get("request"), Mapping) else {}
        allowed_actions = request.get("allowed_actions") if isinstance(request, Mapping) else []
        action_kinds: list[str] = []
        action_targets: list[str] = []
        if isinstance(allowed_actions, list):
            for allowed in allowed_actions:
                if not isinstance(allowed, Mapping):
                    continue
                kind = str(allowed.get("kind") or "").strip()
                if kind and kind not in action_kinds:
                    action_kinds.append(kind)
                targets = allowed.get("target_ids")
                if isinstance(targets, list):
                    for target in targets:
                        target_id = str(target).strip()
                        if target_id and target_id not in action_targets:
                            action_targets.append(target_id)
                target_id = allowed.get("target_id")
                if target_id is not None:
                    target_text = str(target_id).strip()
                    if target_text and target_text not in action_targets:
                        action_targets.append(target_text)

        resource_state = self._extract_private_resource_state(private)
        target_info = self._extract_private_wolf_target(private)
        target_status = str(target_info["status"])
        current_night_death_targets = (
            list(target_info["target_ids"]) if target_status == "concrete" else []
        )[:5]
        game_state = turn_packet.get("game") if isinstance(turn_packet.get("game"), Mapping) else {}
        phase_text = str(request.get("phase") or request.get("channel") or game_state.get("phase") or game_state.get("public_phase") or "").lower()
        phase_is_witch_action = any(
            self._is_pass_action_kind(kind) or self._is_heal_action_kind(kind) or self._is_poison_action_kind(kind)
            for kind in action_kinds
        ) and ("witch" in phase_text or any(self._is_heal_action_kind(k) or self._is_poison_action_kind(k) for k in action_kinds))
        pass_kind = next((k for k in action_kinds if self._is_pass_action_kind(k)), None)
        heal_kind = next((k for k in action_kinds if self._is_heal_action_kind(k)), None)
        poison_kind = next((k for k in action_kinds if self._is_poison_action_kind(k)), None)

        ledger_snapshot = self._build_witch_ledger_snapshot()
        high_confidence_targets = self._rank_witch_targets(
            ledger_snapshot, min_score=_WITCH_POISON_MIN_SCORE,
            min_evidence_types=_WITCH_POISON_MIN_EVIDENCE_TYPES,
        )
        high_value_rescue_targets = self._rank_witch_rescue_targets(
            ledger_snapshot, current_night_death_targets=current_night_death_targets,
            resource_state=resource_state, action_kinds=action_kinds,
            target_status=target_status,
        )
        rescue_candidates = current_night_death_targets if target_status == "concrete" and heal_kind and resource_state.get("antidote_available") else []
        poison_candidates = high_confidence_targets if poison_kind and resource_state.get("poison_available") else []
        recommendation = "pass"
        if phase_is_witch_action and pass_kind is not None:
            if target_status == "concrete" and high_value_rescue_targets:
                recommendation = f"heal:{high_value_rescue_targets[0]}"
            elif poison_candidates:
                recommendation = f"poison:{poison_candidates[0]}"

        return {
            "resource_state": resource_state,
            "current_action_kinds": action_kinds[:4],
            "current_action_targets": action_targets[:6],
            "decision_gate": {
                "phase_is_witch_action": phase_is_witch_action,
                "pass_kind": pass_kind, "heal_kind": heal_kind, "poison_kind": poison_kind,
                "target_status": target_status,
                "antidote_available": resource_state.get("antidote_available"),
                "poison_available": resource_state.get("poison_available"),
                "rescue_candidates": rescue_candidates[:5],
                "poison_candidates": poison_candidates[:5],
            },
            "target_status": target_status,
            "current_night_death_targets": current_night_death_targets,
            "night_death_targets": current_night_death_targets,
            "rescue_target_candidates": rescue_candidates[:5],
            "high_value_rescue_targets": high_value_rescue_targets,
            "poison_candidates": poison_candidates[:5],
            "high_confidence_targets": high_confidence_targets,
            "recent_public_deaths": self._witch_recent_public_deaths[-3:],
            "recent_vote_pressure": self._witch_recent_vote_pressure[-3:],
            "counterclaim_summary": self._witch_recent_counterclaims[-3:],
            "role_chain_summary": self._witch_recent_role_chain[-3:],
            "ledger_top_suspects": ledger_snapshot,
            "evidence_gate": {"poison_min_score": _WITCH_POISON_MIN_SCORE, "poison_min_evidence_types": _WITCH_POISON_MIN_EVIDENCE_TYPES, "heal_min_score": _WITCH_HEAL_MIN_SCORE},
            "recommendation": recommendation,
        }

    def _update_witch_memory(self, sync_packet: Mapping[str, Any]) -> None:
        if not sync_packet:
            return
        round_value = self._coerce_round_number(self._extract_first_scalar(sync_packet, ("round", "day", "night", "turn")))
        for signal, weight, sink, keywords in self._witch_signal_specs():
            records = self._collect_structured_evidence(sync_packet, signal, keywords)
            for record in records:
                target = str(record.get("target_id") or "").strip()
                if not target:
                    continue
                note = f"{signal}:{target}"
                self._bump_witch_evidence([target], signal, weight, note, round_value, record=record)
                self._append_compact_entry(sink, note)

    @staticmethod
    def _witch_signal_specs() -> tuple[tuple[str, int, list[str], tuple[str, ...]], ...]:
        return (
            ("death_chain", 3, ["死亡", "出局", "淘汰", "night kill", "wolf kill", "被刀", "刀口"], ("死亡", "出局", "淘汰", "night kill", "wolf kill", "被刀", "刀口", "killed", "dead")),
            ("vote_pressure", 2, ["投票", "票型", "归票", "vote", "ballot", "pressure"], ("投票", "票型", "归票", "vote", "ballot", "pressure")),
            ("claim_conflict", 2, ["悍跳", "对跳", "查杀", "金水", "claim", "counterclaim"], ("悍跳", "对跳", "查杀", "金水", "claim", "counterclaim", "reveal")),
            ("role_chain", 1, ["警长", "警徽", "验出", "查验", "sheriff", "seer", "reveal", "确认"], ("警长", "警徽", "验出", "查验", "sheriff", "seer", "reveal", "确认", "investigate")),
        )

    def _collect_structured_targets(self, node: Any, keywords: tuple[str, ...]) -> list[str]:
        """Compatibility wrapper; never extracts IDs from arbitrary text/subtrees."""
        signal = "claim_conflict" if any("claim" in k.lower() or "对跳" in k for k in keywords) else "vote_pressure" if any("vote" in k.lower() or "投票" in k for k in keywords) else "role_chain" if any("role" in k.lower() or "警" in k for k in keywords) else "death_chain"
        return list(dict.fromkeys(str(r.get("target_id")) for r in self._collect_structured_evidence(node, signal, keywords) if r.get("target_id")))

    def _collect_structured_evidence(self, node: Any, signal: str, keywords: tuple[str, ...]) -> list[dict[str, Any]]:
        """Parse public evidence by explicit event fields, not keyword-hit ID lists."""
        records: list[dict[str, Any]] = []
        target_keys = {"target_id", "vote_target_id", "victim_id", "dead_player_id", "eliminated_player_id", "accused_id", "claimed_target_id"}
        speaker_keys = {"speaker_id", "claimant_id", "voter_id", "from_player_id", "source_player_id"}
        role_keys = {"claimed_role", "role", "claimed_team", "team"}
        def norm_key(key: Any) -> str:
            return re.sub(r"(?<!^)([A-Z])", r"_\1", str(key)).lower()
        def ids(value: Any) -> list[str]:
            if isinstance(value, str): return self._extract_player_ids(value)
            if isinstance(value, (int, float)): return [str(value)] if str(value).startswith("p") else []
            return []
        def walk(value: Any) -> None:
            if isinstance(value, Mapping):
                keys = {norm_key(k) for k in value}
                key_blob = " ".join(keys)
                event_values = [str(value[k]).lower() for k in value if norm_key(k) in {"type", "event_type", "kind", "event", "category", "signal"} and isinstance(value[k], (str, int))]
                explicit_event = any(k in key_blob for k in keywords) or any(any(keyword.lower() in event for keyword in keywords) for event in event_values)
                if signal == "role_chain" and any(norm_key(k) in role_keys for k in value):
                    explicit_event = True
                target: list[str] = []
                speaker: list[str] = []
                for key, child in value.items():
                    key_l = norm_key(key)
                    if key_l in target_keys or (signal == "role_chain" and key_l == "player_id"): target.extend(ids(child))
                    if key_l in speaker_keys: speaker.extend(ids(child))
                if target and (explicit_event or (signal == "vote_pressure" and speaker)):
                    role = next((str(value[k]) for k in value if norm_key(k) in role_keys and isinstance(value[k], (str, int))), "")
                    conflict = signal == "claim_conflict" and ("conflict" in key_blob or "counterclaim" in key_blob or "对跳" in key_blob or "悍跳" in key_blob or "查杀" in key_blob)
                    if signal != "claim_conflict" or (speaker and conflict):
                        for target_id in dict.fromkeys(target):
                            records.append({"signal": signal, "target_id": target_id, "speaker_id": speaker[0] if speaker else None, "claimed_role": role, "source": key_blob[:100]})
                for child in value.values(): walk(child)
            elif isinstance(value, list):
                for child in value: walk(child)
        walk(node)
        return records

    def _bump_witch_evidence(
        self,
        players: list[str],
        signal: str,
        weight: int,
        note: str,
        round_value: int | None,
        record: Mapping[str, Any] | None = None,
    ) -> None:
        if not players:
            return
        field_map = {
            "vote_pressure": "vote_pressure",
            "claim_conflict": "claim_conflict",
            "death_chain": "death_chain",
            "role_chain": "role_chain",
        }
        field_name = field_map.get(signal, "score")
        for player_id in players:
            entry = self._witch_evidence_ledger.setdefault(
                player_id,
                {
                    "score": 0,
                    "vote_pressure": 0,
                    "claim_conflict": 0,
                    "death_chain": 0,
                    "role_chain": 0,
                    "evidence_types": [],
                    "last_round": None,
                    "notes": [],
                    "evidence_records": [],
                },
            )
            entry["score"] = int(entry.get("score", 0)) + weight
            entry[field_name] = int(entry.get(field_name, 0)) + weight
            evidence_types = entry.setdefault("evidence_types", [])
            if signal not in evidence_types:
                evidence_types.append(signal)
            if round_value is not None:
                current_round = self._coerce_round_number(entry.get("last_round"))
                if current_round is None or round_value >= current_round:
                    entry["last_round"] = round_value
            records = entry.setdefault("evidence_records", [])
            if isinstance(record, Mapping):
                compact_record = {"signal": signal, "speaker_id": record.get("speaker_id"), "claimed_role": record.get("claimed_role"), "source": self._truncate_text(str(record.get("source") or ""), 80), "round": round_value}
                if compact_record not in records:
                    records.append(compact_record)
                    del records[:-6]
            notes = entry.setdefault("notes", [])
            if note and (not notes or notes[-1] != note):
                notes.append(note)
                del notes[:-3]

    def _append_compact_entry(self, sink: list[str], entry: str, limit: int = 3) -> None:
        compact_entry = self._truncate_text(self._squash_whitespace(entry), 120)
        if not compact_entry:
            return
        if sink and sink[-1] == compact_entry:
            return
        sink.append(compact_entry)
        if len(sink) > limit:
            del sink[:-limit]

    def _build_witch_ledger_snapshot(self) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for player_id, payload in self._witch_evidence_ledger.items():
            evidence_types = list(dict.fromkeys(str(item) for item in payload.get("evidence_types", []) if str(item).strip()))
            items.append(
                {
                    "player_id": player_id,
                    "score": int(payload.get("score", 0)),
                    "vote_pressure": int(payload.get("vote_pressure", 0)),
                    "claim_conflict": int(payload.get("claim_conflict", 0)),
                    "death_chain": int(payload.get("death_chain", 0)),
                    "role_chain": int(payload.get("role_chain", 0)),
                    "evidence_types": evidence_types,
                    "evidence_type_count": len(evidence_types),
                    "last_round": self._coerce_round_number(payload.get("last_round")),
                    "notes": list(payload.get("notes", []))[-2:],
                    "evidence_records": list(payload.get("evidence_records", []))[-4:],
                }
            )
        items.sort(
            key=lambda item: (
                -int(item.get("score", 0)),
                -int(item.get("evidence_type_count", 0)),
                -(item.get("last_round") if item.get("last_round") is not None else -1),
                str(item.get("player_id", "")),
            )
        )
        return items[:4]

    def _compact_witch_context(self, witch_context: Mapping[str, Any]) -> str:
        summary = {
            "resource_state": witch_context.get("resource_state"),
            "decision_gate": witch_context.get("decision_gate", {}),
            "target_status": witch_context.get("target_status", "unknown"),
            "current_night_death_targets": witch_context.get("current_night_death_targets", []),
            "high_value_rescue_targets": witch_context.get("high_value_rescue_targets", []),
            "poison_candidates": witch_context.get("poison_candidates", []),
            "recommendation": witch_context.get("recommendation", "pass"),
        }
        return self._truncate_text(self._safe_json_dump(summary), 420)

    def _build_safety_notes(self, private: Mapping[str, Any], witch_context: Mapping[str, Any] | None = None) -> str:
        if self.profile.role != "witch" and str(private.get("role")) != "witch":
            return ""
        notes = [
            "公开发言只允许基于公开事实；",
            "不得自报女巫、药水余量、夜刀目标、夜间选择、是否已用药、未公开身份链；",
            "任何内部状态都不能被说成已公开事实。",
        ]
        if witch_context:
            notes.append("决策顺序固定为 phase/allowed_actions -> target_status -> heal -> poison -> pass；")
            notes.append("target_status=unknown 或 none 必须 pass；救人只绑定私密当前刀口，毒药只在结构化多源候选集内使用。")
        return "".join(notes)

    def _apply_role_safe_fallback(
        self,
        action: dict[str, Any],
        turn_packet: dict[str, Any],
        raw_response: object,
        witch_context: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        kind = str(action.get("kind", ""))
        is_poison = self._is_poison_action_kind(kind)
        is_heal = self._is_heal_action_kind(kind)
        if not is_poison and not is_heal:
            self._last_decision_audit = {
                "raw_response": raw_response,
                "normalized_action": action,
            }
            return action

        pass_action = self._find_pass_action(turn_packet["request"].get("allowed_actions", []), turn_packet["request"])
        target_id = str(action.get("target_id") or "").strip()
        if not target_id:
            if pass_action is not None:
                self._last_decision_audit = {
                    "raw_response": raw_response,
                    "normalized_action": action,
                    "fallback_action": pass_action,
                    "fallback_reason": f"witch_{'poison' if is_poison else 'heal'}_missing_target",
                }
                return pass_action
            self._last_decision_audit = {
                "raw_response": raw_response,
                "normalized_action": action,
            }
            return action

        # Final safety gate is independent of the model's explanation.  A witch
        # action must also be present in this request's contract and resource state.
        if self.profile.role != "witch" and str(turn_packet.get("private_information", {}).get("role")) != "witch":
            return action
        ledger_snapshot = self._build_witch_ledger_snapshot()
        poison_targets = set(witch_context.get("poison_candidates", [])) if witch_context else set(self._rank_witch_targets(ledger_snapshot, min_score=_WITCH_POISON_MIN_SCORE, min_evidence_types=_WITCH_POISON_MIN_EVIDENCE_TYPES))
        private = turn_packet.get("private_information")
        target_info = self._extract_private_wolf_target(private) if isinstance(private, Mapping) else {"status": "unknown", "target_ids": []}
        current_night_death_targets = list(target_info.get("target_ids", [])) if target_info.get("status") == "concrete" else []
        resource_state = self._extract_private_resource_state(private) if isinstance(private, Mapping) else {}
        action_kinds = [str(a.get("kind")) for a in turn_packet["request"].get("allowed_actions", []) if isinstance(a, Mapping)]
        rescue_targets = set(witch_context.get("high_value_rescue_targets", [])) if witch_context else set(self._rank_witch_rescue_targets(ledger_snapshot, current_night_death_targets=current_night_death_targets, resource_state=resource_state, action_kinds=action_kinds, target_status=str(target_info.get("status"))))
        candidate = self._lookup_witch_candidate(target_id, ledger_snapshot)

        if is_poison:
            contract_ok = any(self._is_poison_action_kind(k) for k in action_kinds)
            if resource_state.get("poison_available") is True and target_info.get("status") == "concrete" and contract_ok and self._is_witch_poison_candidate(candidate) and target_id in poison_targets:
                self._last_decision_audit = {
                    "raw_response": raw_response,
                    "normalized_action": action,
                    "poison_targets": sorted(poison_targets),
                }
                return action
            if pass_action is not None:
                self._last_decision_audit = {
                    "raw_response": raw_response,
                    "normalized_action": action,
                    "fallback_action": pass_action,
                    "fallback_reason": "private_target_unknown" if target_info.get("status") == "unknown" else "no_current_attack" if target_info.get("status") == "none" else "low_confidence_poison",
                    "poison_targets": sorted(poison_targets),
                    "candidate": candidate,
                }
                return pass_action
            self._last_decision_audit = {
                "raw_response": raw_response,
                "normalized_action": action,
                "poison_targets": sorted(poison_targets),
                "candidate": candidate,
            }
            return action

        if is_heal:
            contract_ok = any(self._is_heal_action_kind(k) for k in action_kinds)
            if resource_state.get("antidote_available") is True and contract_ok and target_info.get("status") == "concrete" and target_id in current_night_death_targets and target_id in rescue_targets:
                self._last_decision_audit = {
                    "raw_response": raw_response,
                    "normalized_action": action,
                    "current_night_death_targets": current_night_death_targets,
                    "rescue_targets": sorted(rescue_targets),
                }
                return action
            if pass_action is not None:
                self._last_decision_audit = {
                    "raw_response": raw_response,
                    "normalized_action": action,
                    "fallback_action": pass_action,
                    "fallback_reason": "private_target_unknown" if target_info.get("status") == "unknown" else "no_current_attack" if target_info.get("status") == "none" else "low_value_rescue",
                    "current_night_death_targets": current_night_death_targets,
                    "rescue_targets": sorted(rescue_targets),
                    "candidate": candidate,
                }
                return pass_action
        self._last_decision_audit = {
            "raw_response": raw_response,
            "normalized_action": action,
        }
        return action

    def _extract_current_night_death_targets(self, turn_packet: Mapping[str, Any]) -> set[str]:
        """Compatibility entry point; the public packet is never a source of the knife."""
        private = turn_packet.get("private_information")
        if not isinstance(private, Mapping):
            return set()
        info = self._extract_private_wolf_target(private)
        return set(info["target_ids"]) if info["status"] == "concrete" else set()

    def _rank_witch_targets(
        self,
        ledger_snapshot: list[dict[str, Any]],
        *,
        min_score: int,
        min_evidence_types: int,
    ) -> list[str]:
        ranked: list[str] = []
        allowed_types = {"vote_pressure", "claim_conflict", "role_chain"}
        for item in ledger_snapshot:
            player_id = str(item.get("player_id") or "").strip()
            if not player_id:
                continue
            score = int(item.get("score", 0))
            types = set(str(x) for x in item.get("evidence_types", [])) & allowed_types
            records = [r for r in item.get("evidence_records", []) if isinstance(r, Mapping)]
            sources = {(str(r.get("signal")), str(r.get("source")), str(r.get("speaker_id"))) for r in records}
            if score < min_score or len(types) < min_evidence_types or len(sources) < min_evidence_types:
                continue
            ranked.append(player_id)
        return ranked

    def _rank_witch_rescue_targets(
        self,
        ledger_snapshot: list[dict[str, Any]],
        *,
        current_night_death_targets: list[str],
        resource_state: Mapping[str, Any] | None = None,
        action_kinds: list[str] | None = None,
        target_status: str = "concrete",
    ) -> list[str]:
        if target_status != "concrete":
            return []
        if resource_state is not None and resource_state.get("antidote_available") is not True:
            return []
        if action_kinds is not None and not any(self._is_heal_action_kind(k) for k in action_kinds):
            return []
        death_target_set = {str(item).strip() for item in current_night_death_targets if str(item).strip()}
        ranked: list[str] = []
        for player_id in death_target_set:
            item = next((x for x in ledger_snapshot if str(x.get("player_id")) == player_id), None)
            # Self-save is explicitly valuable even when the player has no public record.
            if player_id == self.player_id or self._is_witch_heal_candidate(item):
                ranked.append(player_id)
        return ranked

    def _lookup_witch_candidate(
        self,
        player_id: str,
        ledger_snapshot: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        player_text = str(player_id).strip()
        if not player_text:
            return None
        for item in ledger_snapshot:
            if str(item.get("player_id") or "").strip() == player_text:
                return item
        return None

    @staticmethod
    def _is_witch_poison_candidate(candidate: Mapping[str, Any] | None) -> bool:
        if not isinstance(candidate, Mapping):
            return False
        score = int(candidate.get("score", 0))
        types = set(str(x) for x in candidate.get("evidence_types", [])) & {"vote_pressure", "claim_conflict", "role_chain"}
        records = [r for r in candidate.get("evidence_records", []) if isinstance(r, Mapping)]
        sources = {(str(r.get("signal")), str(r.get("source")), str(r.get("speaker_id"))) for r in records}
        return score >= _WITCH_POISON_MIN_SCORE and len(types) >= _WITCH_POISON_MIN_EVIDENCE_TYPES and len(sources) >= _WITCH_POISON_MIN_EVIDENCE_TYPES

    @staticmethod
    def _is_witch_heal_candidate(candidate: Mapping[str, Any] | None) -> bool:
        if not isinstance(candidate, Mapping):
            return False
        score = int(candidate.get("score", 0))
        if score < _WITCH_HEAL_MIN_SCORE:
            return False
        evidence_types = {str(item) for item in candidate.get("evidence_types", []) if str(item).strip()}
        return bool(evidence_types & {"claim_conflict", "role_chain", "vote_pressure"})

    def _extract_targets_from_text(self, text: str, keywords: tuple[str, ...]) -> set[str]:
        snippet = self._extract_keyword_snippet(text, keywords)
        if not snippet:
            return set()
        return set(self._extract_player_ids(snippet))

    def _find_pass_action(
        self,
        allowed_actions: Any,
        request: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        if not isinstance(allowed_actions, list):
            return None
        for allowed in allowed_actions:
            if not isinstance(allowed, Mapping):
                continue
            kind = str(allowed.get("kind") or "")
            if not self._is_pass_action_kind(kind):
                continue
            action = {
                "request_id": request["request_id"],
                "player_id": request["player_id"],
                "kind": kind,
            }
            target_id = allowed.get("target_id")
            if target_id:
                action["target_id"] = str(target_id)
            text = allowed.get("text")
            if isinstance(text, str):
                action["text"] = text
            return action
        return None

    @staticmethod
    def _is_poison_action_kind(kind: str) -> bool:
        normalized = kind.lower().strip()
        return "poison" in normalized or normalized == "witch_poison"

    @staticmethod
    def _is_heal_action_kind(kind: str) -> bool:
        normalized = kind.lower().strip()
        return any(token in normalized for token in ("heal", "save", "antidote"))

    @staticmethod
    def _is_pass_action_kind(kind: str) -> bool:
        normalized = kind.lower().strip()
        return normalized in _PASS_ACTION_KINDS or normalized.endswith("_pass") or "pass" == normalized

    def _reset_compact_memory(self, game_id: str | None) -> None:
        self._memory_game_id = game_id
        self._recent_public_summaries = []
        self._last_public_round_phase = ""
        self._last_private_resource_summary = ""
        self._last_decision_audit = None
        self._witch_evidence_ledger = {}
        self._witch_recent_public_deaths = []
        self._witch_recent_vote_pressure = []
        self._witch_recent_counterclaims = []
        self._witch_recent_role_chain = []

    def _build_public_digest(self, sync_packet: Mapping[str, Any]) -> str:
        round_tag = self._build_round_phase_tag(sync_packet)
        serialized = self._safe_json_dump(sync_packet)
        snippets: list[str] = []
        for label, keywords in (
            ("death", ("死亡", "出局", "淘汰", "死亡原因", "night kill", "wolf kill")),
            ("vote", ("投票", "票型", "归票", "vote", "ballot")),
            ("sheriff", ("警长", "上警", "警徽", "sheriff", "police")),
            ("speech", ("发言", "自称", "查杀", "金水", "悍跳", "claim", "say")),
        ):
            snippet = self._extract_keyword_snippet(serialized, keywords)
            if snippet:
                snippets.append(f"{label}:{snippet}")
        if not snippets:
            ids = self._extract_player_ids(serialized)
            if ids:
                snippets.append("players:" + ",".join(ids[:5]))
        parts = [part for part in (round_tag, *snippets) if part]
        if not parts:
            return ""
        return self._truncate_text(" | ".join(parts), 260)

    def _build_round_phase_tag(self, packet: Mapping[str, Any]) -> str:
        round_value = self._extract_first_scalar(packet, ("round", "day", "night", "turn"))
        phase_value = self._extract_first_scalar(packet, ("phase", "public_phase", "stage", "channel"))
        parts: list[str] = []
        if round_value is not None:
            parts.append(f"r{self._compact_scalar(round_value)}")
        if phase_value is not None:
            parts.append(f"phase={self._compact_scalar(phase_value)}")
        return " ".join(parts)

    def _safe_json_dump(self, value: Any) -> str:
        try:
            return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
        except TypeError:
            return self._truncate_text(str(value), 1000)

    def _extract_keyword_snippet(self, text: str, keywords: tuple[str, ...]) -> str:
        lowered = text.lower()
        best_index = -1
        best_keyword = ""
        for keyword in keywords:
            idx = lowered.find(keyword.lower())
            if idx != -1 and (best_index == -1 or idx < best_index):
                best_index = idx
                best_keyword = keyword
        if best_index == -1:
            return ""
        start = max(0, best_index - 45)
        end = min(len(text), best_index + max(60, len(best_keyword) + 45))
        snippet = text[start:end]
        return self._truncate_text(self._squash_whitespace(snippet), 140)

    def _extract_player_ids(self, text: str) -> list[str]:
        return list(dict.fromkeys(match.group(0) for match in _PLAYER_ID_PATTERN.finditer(text)))

    def _extract_first_scalar(self, node: Any, key_hints: tuple[str, ...]) -> Any:
        if isinstance(node, Mapping):
            for key, value in node.items():
                key_lower = str(key).lower()
                if any(hint in key_lower for hint in key_hints):
                    scalar = self._compact_scalar(value)
                    if scalar is not None:
                        return scalar
                nested = self._extract_first_scalar(value, key_hints)
                if nested is not None:
                    return nested
        elif isinstance(node, list):
            for item in node:
                nested = self._extract_first_scalar(item, key_hints)
                if nested is not None:
                    return nested
        return None

    def _compact_scalar(self, value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            if isinstance(value, float) and value.is_integer():
                return str(int(value))
            return str(value)
        if isinstance(value, str):
            text = self._squash_whitespace(value)
            return self._truncate_text(text, 80)
        return self._truncate_text(self._squash_whitespace(str(value)), 80)

    @staticmethod
    def _coerce_bool_scalar(value: Any) -> bool | None:
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            normalized = re.sub(r"\s+", "", value).strip().lower()
            if normalized in {"true", "1", "yes", "y", "on"}:
                return True
            if normalized in {"false", "0", "no", "n", "off"}:
                return False
        return None

    @staticmethod
    def _squash_whitespace(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _truncate_text(text: str, limit: int) -> str:
        if limit <= 0:
            return ""
        if len(text) <= limit:
            return text
        return text[: max(0, limit - 1)] + "…"

    @staticmethod
    def _coerce_round_number(value: Any) -> int | None:
        if value is None or isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value if value >= 0 else None
        if isinstance(value, float):
            return int(value) if value.is_integer() and value >= 0 else None
        if isinstance(value, str):
            match = re.search(r"\d+", value)
            if match:
                try:
                    normalized = int(match.group(0))
                except ValueError:
                    return None
                return normalized if normalized >= 0 else None
        return None

    @staticmethod
    def _nonnegative_int(value: object, *, fallback: int | None) -> int | None:
        if isinstance(value, bool):
            return fallback
        try:
            normalized = int(value)
        except (TypeError, ValueError):
            return fallback
        return normalized if normalized >= 0 else fallback

"""最小可运行的角色 Task-Agent。

每次行动从一个新的模型会话开始；这里额外维护一小段跨回合公开记忆，用于压缩上下文、
稳定狼人夜刀收敛与白天叙事，但不访问任何外部状态。
"""

from __future__ import annotations

import asyncio
from collections import Counter
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
_MAX_HISTORY_ITEMS = 6
_MAX_SNIPPET_LENGTH = 80
_MAX_COMPACT_DEPTH = 2
_TEMPLATE_PHRASES = (
    "先听一圈",
    "看票型",
    "中间位",
    "别划水",
    "稳一点",
    "我先过",
    "先稳",
    "刀中间位",
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
        self._game_memory: dict[str, Any] = {
            "round": None,
            "phase": None,
            "alive_players": [],
            "dead_players": [],
            "sheriff_id": None,
            "death_order": [],
            "recent_votes": [],
            "wolf_plan": None,
            "resolved_night_memory": [],
            "_claim_state": {},
            "_vote_state": {},
            "_dead_state": [],
            "_known_sheriff_id": None,
            "_recent_vote_state": [],
            "_day_case_history": [],
            "_last_day_case_signature": None,
            "_last_observed_round": None,
            "_last_observed_phase": None,
            "_last_observed_dead_players": [],
        }
        self._public_ledger: dict[str, list[dict[str, Any]]] = {
            "claims": [],
            "votes": [],
            "deaths": [],
            "sheriff_changes": [],
        }
        self._public_history: list[dict[str, Any]] = []
        self._self_history: list[dict[str, Any]] = []
        self._recent_public_texts: list[str] = []
        self._recent_self_texts: list[str] = []

    async def observe(self, sync_packet: dict[str, Any]) -> None:
        """接收公开同步并压缩成轻量记忆。"""

        summary = self._summarize_sync_packet(sync_packet)
        if not summary:
            return

        self._update_public_ledger(sync_packet, summary)

        self._update_night_resolution(
            summary,
            previous_phase=self._game_memory.get("_last_observed_phase"),
            previous_round=self._game_memory.get("_last_observed_round"),
            previous_dead_players=list(self._game_memory.get("_last_observed_dead_players") or []),
        )

        for key in ("round", "phase", "sheriff_id"):
            value = summary.get(key)
            if value is not None:
                self._game_memory[key] = value
        for key in ("alive_players", "dead_players", "death_order"):
            value = summary.get(key)
            if value:
                self._game_memory[key] = value
        votes = summary.get("recent_votes")
        if votes:
            self._game_memory["recent_votes"] = votes[-_MAX_HISTORY_ITEMS:]

        public_texts = summary.get("public_texts") or []
        self._recent_public_texts.extend(public_texts)
        self._recent_public_texts = self._recent_public_texts[-_MAX_HISTORY_ITEMS:]

        self._game_memory["_last_observed_round"] = summary.get("round")
        self._game_memory["_last_observed_phase"] = summary.get("phase")
        self._game_memory["_last_observed_dead_players"] = list(summary.get("dead_players") or [])

        self._public_history.append(summary)
        self._public_history = self._public_history[-_MAX_HISTORY_ITEMS:]

    async def decide(self, turn_packet: dict[str, Any]) -> dict[str, Any]:
        if turn_packet["self"]["player_id"] != self.player_id:
            raise ValueError("行动包被发送给了错误玩家")
        private = turn_packet["private_information"]
        if str(private.get("role")) != self.profile.role:
            raise ValueError("Task-Agent 的角色档案与行动包身份不一致")

        memory_snapshot = self._memory_snapshot()
        current_dialogue = self._extract_current_round_dialogue(turn_packet)
        system = self._system_prompt(private)
        prompt = self._build_compact_turn_prompt(turn_packet, memory_snapshot, current_dialogue)

        tool_context = turn_packet.get("tool_context") or {}
        current_dialogue_tool = tool_context.get("current_round_dialogue") or []

        def execute_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
            del arguments
            if name != CURRENT_ROUND_DIALOGUE_TOOL_NAME:
                return {"error": f"不支持的工具：{name}"}
            return {
                "round": turn_packet["game"].get("round"),
                "phase": turn_packet["game"].get("public_phase", turn_packet["game"].get("phase")),
                "dialogue": current_dialogue_tool,
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
            error = decision_error(action, turn_packet["request"])
            if error is None:
                self._record_self_action(turn_packet, action)
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

    def _record_self_action(self, turn_packet: Mapping[str, Any], action: Mapping[str, Any]) -> None:
        summary = {
            "round": self._safe_get(turn_packet.get("game"), "round"),
            "phase": self._safe_get(turn_packet.get("request"), "phase"),
            "kind": action.get("kind"),
        }
        target_id = action.get("target_id")
        if target_id:
            summary["target_id"] = str(target_id)
        text = action.get("text")
        if isinstance(text, str) and text.strip():
            clipped = self._clip_text(text, _MAX_SNIPPET_LENGTH)
            summary["text"] = clipped
            self._recent_self_texts.append(clipped)
            self._recent_self_texts = self._recent_self_texts[-_MAX_HISTORY_ITEMS:]
        self._self_history.append(summary)
        self._self_history = self._self_history[-_MAX_HISTORY_ITEMS:]

    def _update_night_resolution(
        self,
        summary: Mapping[str, Any],
        *,
        previous_phase: Any,
        previous_round: Any,
        previous_dead_players: list[Any],
    ) -> None:
        if not self._is_night_phase(previous_phase):
            return
        current_phase = str(summary.get("phase") or "")
        if self._is_night_phase(current_phase):
            return
        plan = self._game_memory.get("wolf_plan")
        if not isinstance(plan, Mapping):
            return
        target_id = self._target_ref_text(plan.get("primary_target"))
        backup_id = self._target_ref_text(plan.get("backup_target"))
        if not target_id and not backup_id:
            return
        current_dead_players = [str(item) for item in summary.get("dead_players") or []]
        previous_dead_players = [str(item) for item in previous_dead_players]
        dawn_deaths = [item for item in current_dead_players if item not in set(previous_dead_players)]
        if target_id and target_id in dawn_deaths:
            outcome = "success"
            risk_tag = "主刀命中"
            reason = "夜刀命中，主目标不必继续降权"
        elif not dawn_deaths:
            outcome = "no_death"
            risk_tag = "疑似被守/被救"
            reason = "上一晚无死亡，主目标与备刀都应明显降权"
        else:
            outcome = "other_death"
            risk_tag = "主刀未兑现"
            reason = "本晚死亡未落在主刀目标上，需要重新排序"
        resolved = self._game_memory.setdefault("resolved_night_memory", [])
        resolved.append(
            {
                "round": summary.get("round", previous_round),
                "wolf_target": target_id,
                "backup_target": backup_id,
                "dawn_deaths": dawn_deaths,
                "outcome": outcome,
                "risk_tag": risk_tag,
                "reason": reason,
                "previous_phase": previous_phase,
                "current_phase": current_phase,
            }
        )
        if len(resolved) > _MAX_HISTORY_ITEMS:
            del resolved[:-_MAX_HISTORY_ITEMS]

    @staticmethod
    def _is_night_phase(phase: Any) -> bool:
        phase_text = str(phase or "").lower()
        return "night" in phase_text or "夜" in phase_text

    @staticmethod
    def _target_ref_text(value: Any) -> str | None:
        if isinstance(value, Mapping):
            for key in ("player_id", "target_id", "id", "seat", "seat_id"):
                if key in value and value[key] is not None:
                    return str(value[key])
            return None
        if value is None:
            return None
        return str(value)

    @staticmethod
    def _latest_night_resolution(memory_snapshot: Mapping[str, Any]) -> Mapping[str, Any] | None:
        entries = memory_snapshot.get("night_resolution") or []
        for item in reversed(entries):
            if isinstance(item, Mapping):
                return item
        return None

    def _night_target_adjustment(
        self,
        target_id: str,
        memory_snapshot: Mapping[str, Any],
    ) -> tuple[int, list[str]]:
        penalty = 0
        reasons: list[str] = []
        recent_failures = 0
        for idx, entry in enumerate(reversed(memory_snapshot.get("night_resolution") or []), start=1):
            if not isinstance(entry, Mapping):
                continue
            weight = max(2, 8 - idx * 2)
            outcome = str(entry.get("outcome") or "")
            primary = self._target_ref_text(entry.get("wolf_target"))
            backup = self._target_ref_text(entry.get("backup_target"))
            if target_id == primary:
                if outcome == "success":
                    reasons.append("近期刀口已兑现")
                    penalty -= 1
                elif outcome:
                    penalty += weight
                    recent_failures += 1
                    reasons.append(str(entry.get("risk_tag") or "上一晚刀口失败"))
            elif target_id == backup and outcome != "success":
                penalty += max(1, weight // 2)
                reasons.append("备刀曾被用于失败回合")
        if recent_failures >= 2:
            penalty += 3
            reasons.append("连续夜刀失败")
        return penalty, reasons

    def _memory_snapshot(self) -> dict[str, Any]:
        night_plan = self._game_memory.get("wolf_plan")
        if isinstance(night_plan, Mapping):
            signature = night_plan.get("signature") if isinstance(night_plan.get("signature"), Mapping) else {}
            phase = str(self._game_memory.get("phase") or "")
            if signature.get("round") != self._game_memory.get("round") or not self._is_night_phase(phase):
                night_plan = None
            else:
                night_plan = {
                    "primary_target": night_plan.get("primary_target"),
                    "backup_target": night_plan.get("backup_target"),
                    "tie_break_reason": night_plan.get("tie_break_reason"),
                    "switch_conditions": night_plan.get("switch_conditions"),
                    "last_failed_kill_reason": night_plan.get("last_failed_kill_reason"),
                    "candidate_order": night_plan.get("candidate_order", [])[:3],
                }
        snapshot = {
            "game": {
                key: self._game_memory.get(key)
                for key in ("round", "phase", "sheriff_id", "alive_players", "dead_players", "death_order")
            },
            "public_ledger": {
                "claims": self._public_ledger["claims"][-2:],
                "votes": self._public_ledger["votes"][-2:],
                "deaths": self._public_ledger["deaths"][-2:],
                "sheriff_changes": self._public_ledger["sheriff_changes"][-1:],
            },
            "recent_vote_summaries": self._game_memory.get("recent_votes", [])[-3:],
            "recent_public_claims": self._public_ledger["claims"][-3:],
            "recent_self_summary": self._self_history[-3:],
            "night_resolution": self._game_memory.get("resolved_night_memory", [])[-3:],
            "night_plan": night_plan,
            "anti_template_notes": self._anti_template_notes(),
        }
        return snapshot

    def _build_compact_turn_prompt(
        self,
        turn_packet: Mapping[str, Any],
        memory_snapshot: Mapping[str, Any],
        current_dialogue: list[dict[str, Any]],
    ) -> dict[str, Any]:
        request = self._compact_snapshot(
            turn_packet.get("request"),
            preferred_keys=(
                "request_id",
                "phase",
                "channel",
                "allowed_actions",
                "max_chars",
                "target_ids",
                "speaker_id",
            ),
        )
        self_state = self._compact_snapshot(
            turn_packet.get("self"),
            preferred_keys=(
                "player_id",
                "seat",
                "seat_id",
                "alive",
                "status",
                "vote_target",
                "speak_order",
            ),
        )
        private_information = self._compact_snapshot(
            turn_packet.get("private_information"),
            preferred_keys=("role", "team", "identity", "faction", "ability"),
        )
        wolf_strategy = self._build_wolf_strategy(turn_packet, memory_snapshot, current_dialogue)
        case_summary = wolf_strategy.get("case_summary") if isinstance(wolf_strategy, Mapping) else {}
        state_digest = self._build_state_digest(turn_packet, memory_snapshot, wolf_strategy)
        return {
            "decision_state": {
                "state_digest": state_digest,
                "recent_dialogue": current_dialogue[-3:],
                "case_summary": case_summary if isinstance(case_summary, Mapping) else {},
                "request": request,
                "self": self_state,
                "private_information": private_information,
            }
        }

    def _build_wolf_strategy(
        self,
        turn_packet: Mapping[str, Any],
        memory_snapshot: Mapping[str, Any],
        current_dialogue: list[dict[str, Any]],
    ) -> dict[str, Any]:
        request = turn_packet.get("request") or {}
        phase = str(request.get("phase") or turn_packet.get("game", {}).get("phase") or "unknown")
        channel = str(request.get("channel") or "unknown")
        allowed_actions = request.get("allowed_actions") if isinstance(request.get("allowed_actions"), list) else []
        target_actions = [item for item in allowed_actions if isinstance(item, Mapping) and item.get("target_ids")]
        speech_actions = [item for item in allowed_actions if isinstance(item, Mapping) and item.get("kind") in _SPEECH_ACTION_KINDS]
        phase_lower = phase.lower()
        channel_lower = channel.lower()
        is_night = any(token in phase_lower for token in ("night", "夜")) or any(
            token in channel_lower for token in ("night", "wolf")
        )
        if is_night and target_actions:
            signature = self._night_plan_signature(turn_packet, target_actions)
            target_ids = self._night_target_ids(target_actions)
            cached_plan = self._game_memory.get("wolf_plan")
            if isinstance(cached_plan, Mapping) and cached_plan.get("signature") == signature:
                locked_plan = cached_plan.get("locked_plan")
                if self._night_locked_plan_is_valid(locked_plan, target_ids):
                    plan = cached_plan.get("strategy")
                    if isinstance(plan, Mapping):
                        return dict(plan)
            ranked_targets = self._rank_targets(turn_packet, memory_snapshot, target_actions)
            locked_plan = self._build_locked_night_plan(ranked_targets, signature, memory_snapshot)
            plan = {
                "stage": "night",
                "objective": "锁定一个主刀和一个备刀，并在重试时复用同一计划",
                "locked_plan": locked_plan,
                "primary_target": locked_plan.get("primary_target"),
                "backup_target": locked_plan.get("backup_target"),
                "discussion_signal": locked_plan.get("discussion_signal"),
                "tie_break_reason": locked_plan.get("discussion_signal"),
                "candidate_order": locked_plan.get("candidate_order", []),
                "notes": self._night_notes(memory_snapshot, current_dialogue),
                "signature": signature,
                "last_failed_kill_reason": locked_plan.get("last_failed_kill_reason", ""),
                "switch_conditions": locked_plan.get("switch_conditions", []),
            }
            self._game_memory["wolf_plan"] = {
                "signature": signature,
                "primary_target": plan.get("primary_target"),
                "backup_target": plan.get("backup_target"),
                "tie_break_reason": plan.get("tie_break_reason"),
                "locked_plan": locked_plan,
                "strategy": plan,
            }
            return plan
        if speech_actions:
            day_focus = self._select_day_focus(turn_packet, memory_snapshot, current_dialogue, speech_actions)
            case_summary = day_focus.get("case_summary") if isinstance(day_focus, Mapping) else {}
            return {
                "stage": "day",
                "objective": "用对象、证据类型、下一步动作和目的维持可解释叙事",
                "case_summary": case_summary if isinstance(case_summary, Mapping) else {},
                "day_focus": day_focus,
                "pressure_points": self._day_pressure_points(memory_snapshot),
                "dialogue_focus": self._dialogue_focus(current_dialogue),
                "avoid_templates": self._anti_template_notes(),
                "anti_repeat": self._recent_self_texts[-1] if self._recent_self_texts else "",
            }
        return {
            "stage": "other",
            "objective": "只做当前合法行动，不额外发散",
            "avoid_templates": self._anti_template_notes(),
        }

    def _night_plan_signature(
        self,
        turn_packet: Mapping[str, Any],
        target_actions: list[Mapping[str, Any]],
    ) -> dict[str, Any]:
        request = turn_packet.get("request") or {}
        game = turn_packet.get("game") or {}
        return {
            "round": self._safe_get(game, "round"),
            "phase": str(request.get("phase") or self._safe_get(game, "phase") or "unknown"),
            "channel": str(request.get("channel") or "unknown"),
            "targets": self._night_target_ids(target_actions),
        }

    @staticmethod
    def _night_target_ids(target_actions: list[Mapping[str, Any]]) -> list[str]:
        target_ids: list[str] = []
        for action in target_actions:
            for target_id in action.get("target_ids") or []:
                target = str(target_id)
                if target not in target_ids:
                    target_ids.append(target)
        return sorted(target_ids)

    def _night_locked_plan_is_valid(self, locked_plan: Any, target_ids: list[str]) -> bool:
        if not isinstance(locked_plan, Mapping):
            return False
        available = {str(target) for target in target_ids if target is not None}
        primary = self._target_ref_text(locked_plan.get("primary_target"))
        backup = self._target_ref_text(locked_plan.get("backup_target"))
        if primary and primary not in available:
            return False
        if backup and backup not in available:
            return False
        return bool(primary or backup)

    def _build_locked_night_plan(
        self,
        ranked_targets: list[dict[str, Any]],
        signature: Mapping[str, Any],
        memory_snapshot: Mapping[str, Any],
    ) -> dict[str, Any]:
        primary = ranked_targets[0] if ranked_targets else None
        backup = ranked_targets[1] if len(ranked_targets) > 1 else None
        latest_resolution = self._latest_night_resolution(memory_snapshot)
        last_failed_kill_reason = ""
        if isinstance(latest_resolution, Mapping) and str(latest_resolution.get("outcome") or "") != "success":
            last_failed_kill_reason = str(latest_resolution.get("reason") or latest_resolution.get("risk_tag") or "")
        switch_conditions = [
            "若主刀合法性变化或出现新的公开身份链，则重新排序",
            "若上一晚空刀/被守目标仍无新证据变化，则优先切换备刀",
        ]
        if last_failed_kill_reason:
            switch_conditions.insert(0, "上一晚失败记忆优先降权同一目标")
        discussion_signal = (
            primary.get("reason") if isinstance(primary, Mapping) else "优先高威胁公共目标，再按座位/ID固定顺序"
        )
        if last_failed_kill_reason:
            discussion_signal = f"{last_failed_kill_reason}；{discussion_signal}"
        return {
            "primary_target": primary,
            "backup_target": backup,
            "discussion_signal": discussion_signal,
            "candidate_order": ranked_targets[:3],
            "switch_conditions": switch_conditions[:3],
            "last_failed_kill_reason": last_failed_kill_reason,
            "signature": dict(signature),
        }

    def _update_public_ledger(self, sync_packet: Mapping[str, Any], summary: Mapping[str, Any]) -> None:
        game = sync_packet.get("game") if isinstance(sync_packet.get("game"), Mapping) else {}
        public_state = (
            sync_packet.get("public_state") if isinstance(sync_packet.get("public_state"), Mapping) else {}
        )
        memory_game = {"game": self._game_memory}
        records = self._extract_player_records(public_state, game, memory_game)
        round_value = summary.get("round")
        if round_value is None:
            round_value = self._safe_get(game, "round") or self._safe_get(public_state, "round")
        claim_state = self._game_memory.setdefault("_claim_state", {})
        vote_state = self._game_memory.setdefault("_vote_state", {})
        dead_state = {str(item) for item in self._game_memory.setdefault("_dead_state", [])}
        known_sheriff_id = self._game_memory.get("_known_sheriff_id")
        recent_vote_state = self._game_memory.setdefault("_recent_vote_state", [])

        claim_events: list[dict[str, Any]] = []
        vote_events: list[dict[str, Any]] = []
        death_events: list[dict[str, Any]] = []
        sheriff_events: list[dict[str, Any]] = []

        sheriff_id = summary.get("sheriff_id")
        if sheriff_id is not None:
            sheriff_id_text = str(sheriff_id)
            if sheriff_id_text != str(known_sheriff_id):
                sheriff_events.append(
                    {
                        "round": round_value,
                        "from": known_sheriff_id,
                        "to": sheriff_id_text,
                    }
                )
                self._game_memory["_known_sheriff_id"] = sheriff_id_text

        for pid in summary.get("dead_players") or []:
            target = str(pid)
            if target not in dead_state:
                death_events.append({"round": round_value, "player_id": target})
                dead_state.add(target)
        self._game_memory["_dead_state"] = list(dead_state)

        for pid, record in records.items():
            claim_text = self._public_claim_text(record)
            if claim_text:
                previous_claim = claim_state.get(pid)
                if claim_text != previous_claim:
                    claim_state[pid] = claim_text
                    claim_events.append(
                        {
                            "round": round_value,
                            "player_id": pid,
                            "claim": claim_text,
                        }
                    )
            vote_target = record.get("vote_target")
            if vote_target is not None:
                vote_target_text = str(vote_target)
                previous_vote = vote_state.get(pid)
                if vote_target_text != previous_vote:
                    vote_state[pid] = vote_target_text
                    vote_events.append(
                        {
                            "round": round_value,
                            "voter": pid,
                            "target_id": vote_target_text,
                        }
                    )

        for vote_entry in summary.get("recent_votes") or []:
            vote_text = str(vote_entry)
            if vote_text not in recent_vote_state:
                recent_vote_state.append(vote_text)
                vote_events.append({"round": round_value, "vote": vote_text})
        self._game_memory["_recent_vote_state"] = recent_vote_state[-_MAX_HISTORY_ITEMS:]

        self._append_bounded(self._public_ledger["claims"], claim_events, _MAX_HISTORY_ITEMS)
        self._append_bounded(self._public_ledger["votes"], vote_events, _MAX_HISTORY_ITEMS)
        self._append_bounded(self._public_ledger["deaths"], death_events, _MAX_HISTORY_ITEMS)
        self._append_bounded(self._public_ledger["sheriff_changes"], sheriff_events, _MAX_HISTORY_ITEMS)

    @staticmethod
    def _append_bounded(bucket: list[dict[str, Any]], items: list[dict[str, Any]], limit: int) -> None:
        if not items:
            return
        bucket.extend(items)
        if len(bucket) > limit:
            del bucket[:-limit]

    def _public_claim_text(self, record: Mapping[str, Any]) -> str | None:
        for key in ("public_role", "claimed_role", "claim"):
            value = record.get(key)
            if value is None:
                continue
            if isinstance(value, (Mapping, list, tuple)):
                value = self._compact_snapshot(value, preferred_keys=(), depth=1, max_items=4)
            text = self._clip_text(str(value).strip(), 32)
            if text:
                return text
        return None

    def _rank_targets(
        self,
        turn_packet: Mapping[str, Any],
        memory_snapshot: Mapping[str, Any],
        target_actions: list[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        public_state = turn_packet.get("public_state")
        players = self._extract_player_records(public_state, turn_packet.get("game"), memory_snapshot)
        candidates: list[dict[str, Any]] = []
        seen: set[str] = set()
        for action in target_actions:
            for target_id in action.get("target_ids") or []:
                target = str(target_id)
                if target in seen:
                    continue
                seen.add(target)
                record = players.get(target, {})
                score, reason = self._score_target(target, record, memory_snapshot)
                candidates.append(
                    {
                        "player_id": target,
                        "score": score,
                        "reason": reason,
                        "seat": self._seat_sort_value(record, target),
                    }
                )
        candidates.sort(key=lambda item: (-int(item["score"]), item["seat"], item["player_id"]))
        return [
            {"player_id": item["player_id"], "reason": item["reason"]}
            for item in candidates[:3]
        ]

    def _score_target(
        self,
        target_id: str,
        record: Mapping[str, Any],
        memory_snapshot: Mapping[str, Any],
    ) -> tuple[int, str]:
        score = 0
        reasons: list[str] = []
        if target_id == self.player_id:
            return -100, "自己不能作为刀口"
        ledger = memory_snapshot.get("public_ledger") or {}
        if record:
            if record.get("alive") is False or record.get("status") in {"dead", "eliminated"}:
                score -= 50
                reasons.append("已死亡")
            if record.get("is_sheriff") or record.get("sheriff"):
                score += 8
                reasons.append("警长位")
            claim_text = self._public_claim_text(record)
            if claim_text:
                score += 5
                reasons.append("身份公开")
            vote_count = record.get("vote_count", record.get("votes"))
            if isinstance(vote_count, int):
                score += min(4, max(0, vote_count))
                if vote_count:
                    reasons.append("票型集中")
            speak_count = record.get("speak_count", record.get("speech_count"))
            if isinstance(speak_count, int):
                score += min(3, max(0, speak_count))
                if speak_count:
                    reasons.append("发言活跃")
        claim_entries = ledger.get("claims") or []
        if any(entry.get("player_id") == target_id for entry in claim_entries[-3:]):
            score += 3
            reasons.append("近期公开身份链")
        vote_entries = ledger.get("votes") or []
        if any(entry.get("player_id") == target_id or entry.get("voter") == target_id or entry.get("target_id") == target_id for entry in vote_entries[-4:]):
            score += 2
            reasons.append("近期票型关联")
        sheriff_entries = ledger.get("sheriff_changes") or []
        if any(entry.get("to") == target_id or entry.get("from") == target_id for entry in sheriff_entries[-2:]):
            score += 4
            reasons.append("警徽链相关")
        mentions = self._recent_mention_count(target_id)
        if mentions:
            score += min(3, mentions)
            reasons.append(f"近期被提及{mentions}次")
        night_penalty, night_reasons = self._night_target_adjustment(target_id, memory_snapshot)
        if night_penalty:
            score -= night_penalty
            reasons.extend(night_reasons[:2])
        elif night_reasons:
            reasons.extend(night_reasons[:1])
        if self._is_alive_record(record) is False:
            score -= 50
            if "已死亡" not in reasons:
                reasons.append("不在存活名单")
        death_order = memory_snapshot.get("game", {}).get("death_order") or []
        if isinstance(death_order, list) and target_id in {str(item) for item in death_order[-2:]}:
            score += 1
            reasons.append("延续已有压力")
        if not reasons:
            reasons.append("默认公共威胁")
        return score, ",".join(reasons[:3])

    def _select_day_focus(
        self,
        turn_packet: Mapping[str, Any],
        memory_snapshot: Mapping[str, Any],
        current_dialogue: list[dict[str, Any]],
        speech_actions: list[Mapping[str, Any]],
    ) -> dict[str, Any]:
        public_state = turn_packet.get("public_state")
        records = self._extract_player_records(public_state, turn_packet.get("game"), memory_snapshot)
        candidates: list[dict[str, Any]] = []
        for pid, record in records.items():
            if pid == self.player_id:
                continue
            if self._is_alive_record(record) is False:
                continue
            score = 0
            reasons: list[str] = []
            if record.get("is_sheriff") or record.get("sheriff"):
                score += 8
                reasons.append("警长位")
            claim_text = self._public_claim_text(record)
            if claim_text:
                score += 7
                reasons.append("公开身份声明")
            vote_count = record.get("vote_count", record.get("votes"))
            if isinstance(vote_count, int) and vote_count > 0:
                score += min(5, vote_count)
                reasons.append("票型集中")
            speak_count = record.get("speak_count", record.get("speech_count"))
            if isinstance(speak_count, int) and speak_count > 0:
                score += min(4, speak_count)
                reasons.append("发言活跃")
            mentions = self._recent_mention_count(pid)
            if mentions:
                score += min(4, mentions)
                reasons.append(f"近期被提及{mentions}次")
            if any(entry.get("player_id") == pid for entry in (memory_snapshot.get("public_ledger") or {}).get("claims", [])[-3:]):
                score += 2
                reasons.append("最新身份链")
            if any(entry.get("to") == pid or entry.get("from") == pid for entry in (memory_snapshot.get("public_ledger") or {}).get("sheriff_changes", [])[-2:]):
                score += 2
                reasons.append("警徽链")
            if any(str(entry.get("player_id") or entry.get("voter") or "") == pid for entry in (memory_snapshot.get("public_ledger") or {}).get("votes", [])[-3:]):
                score += 1
                reasons.append("近期票线")
            candidates.append(
                {
                    "player_id": pid,
                    "score": score,
                    "reason": ",".join(reasons[:4]) if reasons else "默认公共压力",
                    "claim": claim_text,
                    "signals": len(reasons),
                }
            )
        candidates.sort(
            key=lambda item: (
                -int(item["score"]),
                self._seat_sort_value(records.get(item["player_id"], {}), item["player_id"]),
                item["player_id"],
            )
        )
        chosen = candidates[0] if candidates else {}
        runner_up = candidates[1] if len(candidates) > 1 else {}
        pressure_gap = int(chosen.get("score", 0)) - int(runner_up.get("score", 0)) if chosen else 0
        chosen_reason = str(chosen.get("reason") or "")
        chosen_signals = int(chosen.get("signals") or 0)
        mode = "attack"
        mode_reason = "默认压最高公共压力点"
        if chosen:
            if int(chosen.get("score", 0)) >= 14 and chosen_signals >= 3 and ("票型" in chosen_reason or "警长" in chosen_reason or chosen.get("claim")):
                mode = "sacrifice"
                mode_reason = "高压对象已足够成势，优先用牺牲叙事换整体局面"
            elif pressure_gap <= 2 and (chosen.get("claim") or "警徽链" in chosen_reason or "最新身份链" in chosen_reason):
                mode = "deflect"
                mode_reason = "头部压力接近，改用分票和转移叙事"
            elif int(chosen.get("score", 0)) >= 8:
                mode_reason = "存在稳定高压点，继续攻击更划算"
            elif any("警徽链" in item.get("reason", "") or item.get("claim") for item in candidates[:3]):
                mode = "deflect"
                mode_reason = "前排证据更适合拿来分票"
        evidence = self._select_day_evidence(chosen.get("player_id"), records, memory_snapshot, current_dialogue)
        next_action_map = {
            "attack": "继续压票并逼出破绽",
            "deflect": "把压力转到更能分票的对象或链条",
            "sacrifice": "承认局部代价，切断与该对象的绑定",
        }
        day_focus = {
            "mode": mode,
            "mode_reason": mode_reason,
            "target": chosen,
            "evidence": evidence,
            "next_action": next_action_map.get(mode, "继续围绕具体对象和证据推进"),
            "anti_repeat": self._recent_self_texts[-1] if self._recent_self_texts else "",
            "candidates": candidates[:3],
            "pressure_gap": pressure_gap,
            "speech_budget": speech_actions[0].get("max_chars") if speech_actions else None,
        }
        case_summary = self._decision_case_summary(day_focus)
        day_focus["case_summary"] = case_summary
        day_focus["evidence_type"] = case_summary.get("evidence_type")
        self._record_day_case_summary(turn_packet, case_summary)
        return day_focus

    def _select_day_evidence(
        self,
        target_id: str | None,
        records: Mapping[str, Mapping[str, Any]],
        memory_snapshot: Mapping[str, Any],
        current_dialogue: list[dict[str, Any]],
    ) -> dict[str, Any]:
        ledger = memory_snapshot.get("public_ledger") or {}
        recent_case_history = self._game_memory.get("_day_case_history") or []
        recent_kinds = [
            str(entry.get("evidence_type") or "")
            for entry in recent_case_history
            if isinstance(entry, Mapping) and entry.get("evidence_type")
        ]
        last_kind = recent_kinds[-1] if recent_kinds else ""
        repeated_kind = last_kind if len(recent_kinds) >= 2 and recent_kinds[-1] == recent_kinds[-2] else ""
        kind_priority = {"claim": 0, "vote": 1, "sheriff": 2, "death": 3, "dialogue": 4, "fallback": 5}
        candidates: list[dict[str, Any]] = []

        def add_candidate(
            kind: str,
            text: str,
            *,
            source: str,
            rank: int = 0,
            target_hit: bool = False,
        ) -> None:
            text = self._clip_text(text, 70)
            if not text:
                return
            base_score = {"claim": 50, "vote": 46, "sheriff": 42, "death": 38, "dialogue": 28, "fallback": 10}.get(kind, 20)
            score = base_score + max(0, 6 - rank * 2)
            if target_hit:
                score += 6
            if kind not in recent_kinds[-2:]:
                score += 8
            else:
                score -= 4
            if repeated_kind and kind == repeated_kind:
                score -= 12
            elif kind == last_kind:
                score -= 3
            candidates.append(
                {
                    "kind": kind,
                    "text": text,
                    "source": source,
                    "score": score,
                    "rank": rank,
                }
            )

        def add_latest_matching(
            kind: str,
            entries: Any,
            *,
            target_match: Any | None = None,
            text_factory: Any | None = None,
            source: str,
            target_hit: bool = False,
            limit: int = 4,
        ) -> None:
            if not isinstance(entries, list):
                return
            for rank, entry in enumerate(reversed(entries[-limit:])):
                if not isinstance(entry, Mapping):
                    continue
                if target_match is not None and not target_match(entry):
                    continue
                text = text_factory(entry) if text_factory is not None else self._clip_text(json.dumps(entry, ensure_ascii=False), 56)
                add_candidate(kind, text, source=source, rank=rank, target_hit=target_hit)
                return

        if target_id and target_id in records:
            record = records[target_id]
            claim_text = self._public_claim_text(record)
            if claim_text:
                add_candidate("claim", f"{target_id}公开身份:{claim_text}", source="target_claim", target_hit=True)
            add_latest_matching(
                "vote",
                ledger.get("votes"),
                target_match=lambda entry: entry.get("player_id") == target_id or entry.get("voter") == target_id or entry.get("target_id") == target_id,
                source="target_vote",
                target_hit=True,
            )
            add_latest_matching(
                "sheriff",
                ledger.get("sheriff_changes"),
                target_match=lambda entry: entry.get("to") == target_id or entry.get("from") == target_id,
                source="target_sheriff",
                target_hit=True,
            )

        add_latest_matching(
            "claim",
            ledger.get("claims"),
            text_factory=lambda entry: self._clip_text(
                f"{entry.get('player_id')}公开身份:{entry.get('claim') or entry.get('claimed_role') or entry.get('public_role') or ''}",
                64,
            ),
            source="claim_chain",
        )
        add_latest_matching(
            "vote",
            ledger.get("votes"),
            text_factory=lambda entry: self._clip_text(json.dumps(entry, ensure_ascii=False), 64),
            source="vote_chain",
        )
        add_latest_matching(
            "sheriff",
            ledger.get("sheriff_changes"),
            text_factory=lambda entry: self._clip_text(json.dumps(entry, ensure_ascii=False), 64),
            source="sheriff_chain",
        )
        add_latest_matching(
            "death",
            ledger.get("deaths"),
            text_factory=lambda entry: self._clip_text(json.dumps(entry, ensure_ascii=False), 64),
            source="death_chain",
        )
        if current_dialogue:
            for rank, item in enumerate(reversed(current_dialogue[-3:])):
                speaker = item.get("speaker") or item.get("player_id") or item.get("from")
                text = item.get("text") or item.get("content") or item.get("message")
                if speaker is not None and text:
                    add_candidate(
                        "dialogue",
                        f"{speaker}:{self._clip_text(str(text), 40)}",
                        source="dialogue",
                        rank=rank,
                    )
                    break

        if not candidates:
            return {"kind": "fallback", "text": "用当前公开信息补一条最具体的链条"}

        candidates.sort(
            key=lambda item: (
                -int(item["score"]),
                int(item["rank"]),
                kind_priority.get(str(item["kind"]), 99),
                str(item["text"]),
            )
        )
        chosen = candidates[0]
        return {"kind": chosen["kind"], "text": chosen["text"], "source": chosen["source"]}

    def _decision_case_summary(self, day_focus: Mapping[str, Any]) -> dict[str, Any]:
        target = day_focus.get("target") if isinstance(day_focus.get("target"), Mapping) else {}
        evidence = day_focus.get("evidence") if isinstance(day_focus.get("evidence"), Mapping) else {}
        purpose_map = {
            "attack": "保票",
            "deflect": "分票",
            "sacrifice": "切叙事",
        }
        object_text = self._target_ref_text(target.get("player_id")) if isinstance(target, Mapping) else None
        if not object_text:
            object_text = "未知对象"
        evidence_type = str(evidence.get("kind") or "fallback")
        next_action = str(day_focus.get("next_action") or self._day_next_action(evidence_type))
        return {
            "object": object_text,
            "evidence_type": evidence_type,
            "next_action": next_action,
            "purpose": purpose_map.get(str(day_focus.get("mode") or ""), "维持叙事"),
        }

    def _record_day_case_summary(self, turn_packet: Mapping[str, Any], case_summary: Mapping[str, Any]) -> None:
        request = turn_packet.get("request") or {}
        game = turn_packet.get("game") or {}
        signature = str(
            self._safe_get(request, "request_id")
            or f"{self._safe_get(game, 'round')}:{self._safe_get(request, 'phase') or self._safe_get(game, 'phase')}:{self._safe_get(request, 'channel')}"
        )
        if self._game_memory.get("_last_day_case_signature") == signature:
            return
        history = self._game_memory.setdefault("_day_case_history", [])
        if not isinstance(history, list):
            history = []
        history.append({"signature": signature, **dict(case_summary)})
        if len(history) > _MAX_HISTORY_ITEMS:
            del history[:-_MAX_HISTORY_ITEMS]
        self._game_memory["_day_case_history"] = history
        self._game_memory["_last_day_case_signature"] = signature

    def _evidence_freshness_hint(self) -> dict[str, Any]:
        history = self._game_memory.get("_day_case_history") or []
        kinds = [
            str(entry.get("evidence_type") or "")
            for entry in history
            if isinstance(entry, Mapping) and entry.get("evidence_type")
        ]
        recent_types = kinds[-3:]
        if not recent_types:
            return {"recent_types": [], "switch_hint": "优先使用最新链条"}
        switch_hint = "优先切换证据类型" if len(recent_types) >= 2 and recent_types[-1] == recent_types[-2] else "优先使用最近未用过的链条"
        return {"recent_types": recent_types, "switch_hint": switch_hint}

    def _build_state_digest(
        self,
        turn_packet: Mapping[str, Any],
        memory_snapshot: Mapping[str, Any],
        wolf_strategy: Mapping[str, Any],
    ) -> dict[str, Any]:
        game = turn_packet.get("game") if isinstance(turn_packet.get("game"), Mapping) else {}
        request = turn_packet.get("request") if isinstance(turn_packet.get("request"), Mapping) else {}
        self_state = turn_packet.get("self") if isinstance(turn_packet.get("self"), Mapping) else {}
        alive_ids = self._extract_id_list(game, preferred_keys=("alive_players", "alive", "living_players"))
        dead_ids = self._extract_id_list(game, preferred_keys=("dead_players", "dead", "eliminated_players"))
        digest: dict[str, Any] = {
            "turn": {
                "round": self._safe_get(game, "round"),
                "phase": str(request.get("phase") or self._safe_get(game, "phase") or "unknown"),
                "channel": str(request.get("channel") or "unknown"),
                "request_id": self._safe_get(request, "request_id"),
            },
            "self": {
                "player_id": self.player_id,
                "alive": self._safe_get(self_state, "alive"),
                "seat": self._safe_get(self_state, "seat") or self._safe_get(self_state, "seat_id"),
            },
            "table": {
                "alive_count": len(alive_ids),
                "dead_count": len(dead_ids),
                "sheriff_id": memory_snapshot.get("game", {}).get("sheriff_id"),
                "death_tail": self._format_id_list((memory_snapshot.get("game", {}).get("death_order") or [])[-2:]),
            },
            "signals": {
                "pressure_points": self._day_pressure_points(memory_snapshot)[:2],
                "recent_votes": (memory_snapshot.get("recent_vote_summaries") or [])[-2:],
            },
        }
        latest_resolution = self._latest_night_resolution(memory_snapshot)
        if isinstance(latest_resolution, Mapping):
            digest["signals"]["latest_night"] = {
                "outcome": latest_resolution.get("outcome"),
                "risk_tag": latest_resolution.get("risk_tag"),
                "reason": self._clip_text(str(latest_resolution.get("reason") or ""), 40),
            }
        if isinstance(wolf_strategy, Mapping) and wolf_strategy.get("stage") == "night":
            locked_plan = wolf_strategy.get("locked_plan") if isinstance(wolf_strategy.get("locked_plan"), Mapping) else {}
            digest["night_lock"] = {
                "primary": self._target_ref_text(locked_plan.get("primary_target") or wolf_strategy.get("primary_target")),
                "backup": self._target_ref_text(locked_plan.get("backup_target") or wolf_strategy.get("backup_target")),
                "discussion_signal": self._clip_text(
                    str(locked_plan.get("discussion_signal") or wolf_strategy.get("discussion_signal") or ""),
                    60,
                ),
            }
            switch_conditions = locked_plan.get("switch_conditions") or wolf_strategy.get("switch_conditions") or []
            if isinstance(switch_conditions, list) and switch_conditions:
                digest["night_lock"]["switch"] = [
                    self._clip_text(str(item), 40) for item in switch_conditions[:2] if item is not None
                ]
        elif isinstance(wolf_strategy, Mapping) and wolf_strategy.get("stage") == "day":
            digest["day_freshness"] = self._evidence_freshness_hint()
            case_summary = wolf_strategy.get("case_summary")
            if isinstance(case_summary, Mapping):
                digest["case_summary"] = {
                    "object": case_summary.get("object"),
                    "evidence_type": case_summary.get("evidence_type"),
                    "purpose": case_summary.get("purpose"),
                }
        return digest

    @staticmethod
    def _day_next_action(evidence_kind: str | None) -> str:
        mapping = {
            "claim": "继续追问细节并逼其补足逻辑",
            "vote": "顺着票型继续压，并要求给出投票理由",
            "sheriff": "围绕警徽链转向，先解释传递再站边",
            "death": "用死亡顺序重排站边，避免空泛表态",
            "dialogue": "先围绕这句发言做定点追问",
        }
        return mapping.get(evidence_kind or "", "先把对象、证据和动作说完整，再决定站边")

    def _night_notes(
        self,
        memory_snapshot: Mapping[str, Any],
        current_dialogue: list[dict[str, Any]],
    ) -> list[str]:
        notes: list[str] = []
        game = memory_snapshot.get("game", {})
        sheriff_id = game.get("sheriff_id")
        if sheriff_id:
            notes.append(f"优先考虑警长{self._clip_text(str(sheriff_id), 18)}")
        public_ledger = memory_snapshot.get("public_ledger") or {}
        claims = public_ledger.get("claims") or []
        if claims:
            notes.append("优先处理公开身份链")
        latest_resolution = self._latest_night_resolution(memory_snapshot)
        if isinstance(latest_resolution, Mapping):
            reason = str(latest_resolution.get("reason") or "")
            risk_tag = str(latest_resolution.get("risk_tag") or "")
            if reason:
                notes.append(self._clip_text(reason, 40))
            if risk_tag:
                notes.append(self._clip_text(risk_tag, 18))
        recent_votes = memory_snapshot.get("recent_vote_summaries") or []
        if recent_votes:
            notes.append("结合最近票型收敛刀口")
        if current_dialogue:
            notes.append("优先处理夜间讨论中的高威胁公共目标")
        return notes[:3]

    def _day_pressure_points(self, memory_snapshot: Mapping[str, Any]) -> list[str]:
        points: list[str] = []
        game = memory_snapshot.get("game", {})
        dead_players = game.get("dead_players") or []
        if dead_players:
            points.append(f"围绕最近死亡：{self._format_id_list(dead_players[-2:])}")
        public_ledger = memory_snapshot.get("public_ledger") or {}
        sheriff_changes = public_ledger.get("sheriff_changes") or []
        if sheriff_changes:
            last = sheriff_changes[-1]
            if isinstance(last, Mapping):
                points.append(f"警徽变化：{last.get('from')}→{last.get('to')}")
        recent_votes = memory_snapshot.get("recent_vote_summaries") or []
        if recent_votes:
            points.append("根据上一轮票型调整站边")
        night_resolution = memory_snapshot.get("night_resolution") or []
        if night_resolution:
            latest = next((item for item in reversed(night_resolution) if isinstance(item, Mapping)), None)
            if latest:
                points.append(self._clip_text(str(latest.get("risk_tag") or latest.get("reason") or ""), 28))
        return points[:3]

    def _dialogue_focus(self, current_dialogue: list[dict[str, Any]]) -> list[str]:
        focus: list[str] = []
        for item in current_dialogue[-3:]:
            speaker = item.get("speaker") or item.get("player_id") or item.get("from")
            text = item.get("text") or item.get("content") or item.get("message")
            if speaker is not None and text:
                focus.append(f"{speaker}:{self._clip_text(str(text), 28)}")
        return focus

    def _anti_template_notes(self) -> list[str]:
        corpus = self._recent_public_texts[-4:] + self._recent_self_texts[-4:]
        repeated: list[str] = []
        for phrase in _TEMPLATE_PHRASES:
            hits = sum(1 for text in corpus if phrase in text)
            if hits >= 2:
                repeated.append(phrase)
        if repeated:
            return repeated[:3]
        if self._recent_self_texts:
            last = self._recent_self_texts[-1]
            if last:
                repeated.append(self._clip_text(last, 24))
        return repeated[:3]

    def _recent_mention_count(self, target_id: str) -> int:
        if not target_id:
            return 0
        corpus = " ".join(self._recent_public_texts[-6:] + self._recent_self_texts[-3:])
        if not corpus:
            return 0
        return corpus.count(str(target_id))

    def _extract_current_round_dialogue(self, turn_packet: Mapping[str, Any]) -> list[dict[str, Any]]:
        tool_context = turn_packet.get("tool_context") or {}
        raw_dialogue = tool_context.get("current_round_dialogue") or []
        return self._summarize_dialogue(raw_dialogue)

    def _summarize_sync_packet(self, sync_packet: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(sync_packet, Mapping):
            return {}
        game = sync_packet.get("game") if isinstance(sync_packet.get("game"), Mapping) else {}
        public_state = (
            sync_packet.get("public_state") if isinstance(sync_packet.get("public_state"), Mapping) else {}
        )
        combined: dict[str, Any] = {}
        combined.update(self._flatten_sync_fields(sync_packet))
        combined.update(self._flatten_sync_fields(game))
        combined.update(self._flatten_sync_fields(public_state))
        summary: dict[str, Any] = {}
        round_value = combined.get("round")
        if round_value is not None:
            summary["round"] = round_value
        phase_value = combined.get("phase", combined.get("public_phase"))
        if phase_value is not None:
            summary["phase"] = phase_value
        sheriff_value = combined.get("sheriff_id", combined.get("chairman_id"))
        if sheriff_value is None and isinstance(combined.get("sheriff"), Mapping):
            sheriff_value = self._first_value(combined["sheriff"], ("player_id", "id", "seat", "seat_id"))
        if sheriff_value is not None:
            summary["sheriff_id"] = sheriff_value
        alive_players = self._extract_id_list(
            combined,
            preferred_keys=("alive_players", "alive", "living_players", "survivors"),
        )
        if alive_players:
            summary["alive_players"] = alive_players
        dead_players = self._extract_id_list(
            combined,
            preferred_keys=("dead_players", "dead", "eliminated_players", "casualties"),
        )
        if dead_players:
            summary["dead_players"] = dead_players
            summary["death_order"] = dead_players
        votes = self._summarize_votes(combined)
        if votes:
            summary["recent_votes"] = votes
        texts = self._extract_text_snippets(sync_packet)
        if texts:
            summary["public_texts"] = texts
        if not summary:
            summary["round"] = self._safe_get(game, "round") or self._safe_get(public_state, "round")
            summary["phase"] = self._safe_get(game, "phase") or self._safe_get(public_state, "phase")
        return summary

    def _flatten_sync_fields(self, value: Any) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            return {}
        flattened: dict[str, Any] = {}
        for key in (
            "round",
            "phase",
            "public_phase",
            "sheriff_id",
            "sheriff",
            "chairman_id",
            "chairman",
            "alive_players",
            "dead_players",
            "alive",
            "dead",
            "players",
            "votes",
            "vote_history",
            "dialogue",
            "public_dialogue",
            "events",
            "history",
            "death_order",
        ):
            if key in value:
                flattened[key] = value[key]
        for key in ("sheriff", "chairman"):
            if key in flattened and isinstance(flattened[key], Mapping):
                flattened[f"{key}_id"] = self._first_value(flattened[key], ("player_id", "id", "seat", "seat_id"))
        return flattened

    def _summarize_dialogue(self, dialogue: Any) -> list[dict[str, Any]]:
        if not dialogue:
            return []
        items = dialogue if isinstance(dialogue, list) else [dialogue]
        snippets: list[dict[str, Any]] = []
        for item in items[-3:]:
            if isinstance(item, Mapping):
                speaker = self._first_value(item, ("speaker", "speaker_id", "player_id", "from", "by"))
                text = self._first_value(item, ("text", "content", "utterance", "speech", "message"))
                phase = self._first_value(item, ("phase", "public_phase"))
                snippet: dict[str, Any] = {}
                if speaker is not None:
                    snippet["speaker"] = str(speaker)
                if phase is not None:
                    snippet["phase"] = str(phase)
                if text is not None:
                    snippet["text"] = self._clip_text(str(text), _MAX_SNIPPET_LENGTH)
                if snippet:
                    snippets.append(snippet)
            elif isinstance(item, str):
                snippets.append({"text": self._clip_text(item, _MAX_SNIPPET_LENGTH)})
            else:
                snippets.append({"text": self._clip_text(str(item), _MAX_SNIPPET_LENGTH)})
        return snippets[-3:]

    def _extract_text_snippets(self, value: Any) -> list[str]:
        snippets: list[str] = []
        if isinstance(value, Mapping):
            for key in ("dialogue", "public_dialogue", "events", "history"):
                if key in value:
                    snippets.extend(self._extract_text_snippets(value[key]))
            for key in ("text", "content", "message", "utterance"):
                if key in value and isinstance(value[key], str):
                    snippets.append(self._clip_text(value[key], _MAX_SNIPPET_LENGTH))
        elif isinstance(value, list):
            for item in value[-_MAX_HISTORY_ITEMS:]:
                snippets.extend(self._extract_text_snippets(item))
        elif isinstance(value, str):
            snippets.append(self._clip_text(value, _MAX_SNIPPET_LENGTH))
        return snippets[-_MAX_HISTORY_ITEMS:]

    def _summarize_votes(self, value: Mapping[str, Any]) -> list[str]:
        votes = value.get("votes") or value.get("vote_history")
        if not votes:
            return []
        items: list[str] = []
        if isinstance(votes, Mapping):
            for voter, target in votes.items():
                items.append(f"{voter}->{target}")
        elif isinstance(votes, list):
            for entry in votes[-_MAX_HISTORY_ITEMS:]:
                if isinstance(entry, Mapping):
                    voter = self._first_value(entry, ("voter", "from", "player_id"))
                    target = self._first_value(entry, ("target", "target_id", "to"))
                    if voter is not None or target is not None:
                        items.append(f"{voter}->{target}")
                else:
                    items.append(self._clip_text(str(entry), 40))
        return items[-_MAX_HISTORY_ITEMS:]

    def _extract_player_records(
        self,
        public_state: Any,
        game: Any,
        memory_snapshot: Mapping[str, Any],
    ) -> dict[str, dict[str, Any]]:
        records: dict[str, dict[str, Any]] = {}

        def ensure_record(player_id: Any) -> dict[str, Any]:
            pid = str(player_id)
            record = records.setdefault(pid, {"player_id": pid})
            return record

        def ingest_players(players: Any, alive: bool | None = None) -> None:
            if isinstance(players, Mapping):
                for pid, info in players.items():
                    record = ensure_record(pid)
                    if isinstance(info, Mapping):
                        record.update(self._compact_snapshot(info, preferred_keys=("seat", "seat_id", "alive", "status", "role", "claim", "claimed_role", "public_role", "vote_count", "votes", "speak_count", "speech_count")))
                    if alive is not None and "alive" not in record:
                        record["alive"] = alive
            elif isinstance(players, list):
                for item in players:
                    if isinstance(item, Mapping):
                        pid = self._first_value(item, ("player_id", "id", "seat", "seat_id"))
                        if pid is None:
                            continue
                        record = ensure_record(pid)
                        record.update(self._compact_snapshot(item, preferred_keys=("player_id", "id", "seat", "seat_id", "alive", "status", "role", "claim", "claimed_role", "public_role", "vote_count", "votes", "speak_count", "speech_count")))
                        if alive is not None and "alive" not in record:
                            record["alive"] = alive
                    else:
                        record = ensure_record(item)
                        if alive is not None:
                            record["alive"] = alive

        if isinstance(public_state, Mapping):
            ingest_players(public_state.get("players"))
            ingest_players(public_state.get("alive_players"), alive=True)
            ingest_players(public_state.get("dead_players"), alive=False)
            ingest_players(public_state.get("living_players"), alive=True)
            ingest_players(public_state.get("eliminated_players"), alive=False)
            sheriff = self._first_value(public_state, ("sheriff_id", "chairman_id"))
            if sheriff is None and isinstance(public_state.get("sheriff"), Mapping):
                sheriff = self._first_value(public_state["sheriff"], ("player_id", "id", "seat", "seat_id"))
            if sheriff is None and isinstance(public_state.get("chairman"), Mapping):
                sheriff = self._first_value(public_state["chairman"], ("player_id", "id", "seat", "seat_id"))
            if sheriff is not None:
                ensure_record(sheriff)["is_sheriff"] = True
            votes = public_state.get("votes")
            if isinstance(votes, Mapping):
                for voter, target in votes.items():
                    if isinstance(target, Mapping):
                        target_id = self._first_value(target, ("player_id", "id", "seat", "seat_id"))
                    else:
                        target_id = target
                    ensure_record(voter).setdefault("vote_target", target_id)
                    if target_id is None:
                        continue
                    target_record = ensure_record(target_id)
                    current_votes = self._nonnegative_int(target_record.get("vote_count"), fallback=0) or 0
                    target_record["vote_count"] = current_votes + 1
        if isinstance(game, Mapping):
            ingest_players(game.get("alive_players"), alive=True)
            ingest_players(game.get("dead_players"), alive=False)
            ingest_players(game.get("players"))
            if "death_order" in game:
                for player_id in self._extract_id_list(game, preferred_keys=("death_order",)):
                    ensure_record(player_id).setdefault("death_order", True)

        for player_id in memory_snapshot.get("game", {}).get("alive_players") or []:
            ensure_record(player_id).setdefault("alive", True)
        for player_id in memory_snapshot.get("game", {}).get("dead_players") or []:
            ensure_record(player_id).setdefault("alive", False)
        if memory_snapshot.get("game", {}).get("sheriff_id") is not None:
            ensure_record(memory_snapshot["game"]["sheriff_id"])["is_sheriff"] = True
        return records

    def _compact_public_state(self, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return self._compact_snapshot(value)
        return self._compact_snapshot(
            value,
            preferred_keys=(
                "round",
                "phase",
                "public_phase",
                "alive_players",
                "dead_players",
                "players",
                "votes",
                "vote_history",
                "sheriff",
                "sheriff_id",
                "chairman",
                "chairman_id",
                "death_order",
                "dialogue",
                "public_dialogue",
            ),
        )

    def _compact_snapshot(
        self,
        value: Any,
        *,
        preferred_keys: tuple[str, ...] = (),
        depth: int = _MAX_COMPACT_DEPTH,
        max_items: int = 8,
    ) -> Any:
        if depth <= 0:
            return self._clip_primitive(value)
        if isinstance(value, Mapping):
            compact: dict[str, Any] = {}
            for key in preferred_keys:
                if key in value and key not in compact:
                    compact[str(key)] = self._compact_snapshot(
                        value[key], preferred_keys=(), depth=depth - 1, max_items=max_items
                    )
            for key, item in value.items():
                if len(compact) >= max_items:
                    break
                if key in compact:
                    continue
                compact[str(key)] = self._compact_snapshot(
                    item, preferred_keys=(), depth=depth - 1, max_items=max_items
                )
            return compact
        if isinstance(value, list):
            compact_list = [
                self._compact_snapshot(item, preferred_keys=(), depth=depth - 1, max_items=max_items)
                for item in value[:max_items]
            ]
            if len(value) > max_items:
                compact_list.append(f"...(+{len(value) - max_items})")
            return compact_list
        if isinstance(value, tuple):
            return self._compact_snapshot(list(value), preferred_keys=preferred_keys, depth=depth, max_items=max_items)
        return self._clip_primitive(value)

    def _clip_primitive(self, value: Any) -> Any:
        if isinstance(value, str):
            return self._clip_text(value, 120)
        return value

    @staticmethod
    def _clip_text(text: str, limit: int) -> str:
        text = str(text)
        if len(text) <= limit:
            return text
        return text[: max(0, limit - 1)] + "…"

    @staticmethod
    def _safe_get(value: Any, key: str) -> Any:
        if isinstance(value, Mapping):
            return value.get(key)
        return None

    def _first_value(self, value: Any, keys: tuple[str, ...]) -> Any:
        if not isinstance(value, Mapping):
            return None
        for key in keys:
            if key in value and value[key] is not None:
                return value[key]
        return None

    def _extract_id_list(self, value: Mapping[str, Any], *, preferred_keys: tuple[str, ...]) -> list[str]:
        for key in preferred_keys:
            if key not in value:
                continue
            raw = value[key]
            if raw is None:
                continue
            if isinstance(raw, list):
                return [str(item) for item in raw if item is not None]
            if isinstance(raw, Mapping):
                return [str(item) for item in raw.keys() if item is not None]
            return [str(raw)]
        return []

    @staticmethod
    def _is_alive_record(record: Mapping[str, Any]) -> bool | None:
        if not record:
            return None
        if record.get("alive") is True:
            return True
        status = record.get("status")
        if status in {"dead", "eliminated"}:
            return False
        if status in {"alive", "living"}:
            return True
        if record.get("is_dead") is True:
            return False
        if record.get("is_alive") is True:
            return True
        return None

    @staticmethod
    def _seat_sort_value(record: Mapping[str, Any], target_id: str) -> tuple[int, str]:
        seat = record.get("seat") if isinstance(record, Mapping) else None
        if seat is None and isinstance(record, Mapping):
            seat = record.get("seat_id")
        if seat is None:
            match = re.search(r"\d+", str(target_id))
            if match:
                return int(match.group()), str(target_id)
            return 10**9, str(target_id)
        try:
            return int(seat), str(target_id)
        except (TypeError, ValueError):
            match = re.search(r"\d+", str(seat))
            if match:
                return int(match.group()), str(target_id)
            return 10**9, str(target_id)

    @staticmethod
    def _format_id_list(values: list[Any]) -> str:
        return "、".join(str(item) for item in values if item is not None)



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
        if str(action.get("target_id")) not in {str(target_id) for target_id in target_ids}:
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
            "_claim_state": {},
            "_vote_state": {},
            "_dead_state": [],
            "_known_sheriff_id": None,
            "_recent_vote_state": [],
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
                # 最后一次只允许在本次 request 的 target_ids 内做一次安全修复；这不是
                # 新的决策来源，只是避免模型把过期编号带进引擎。
                repaired = self._repair_illegal_target(
                    action,
                    turn_packet["request"],
                    self._build_wolf_strategy(turn_packet, memory_snapshot, current_dialogue),
                )
                if repaired is not None and decision_error(repaired, turn_packet["request"]) is None:
                    self._record_self_action(turn_packet, repaired)
                    return repaired
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

    def _memory_snapshot(self) -> dict[str, Any]:
        night_plan = self._game_memory.get("wolf_plan")
        if isinstance(night_plan, Mapping):
            signature = night_plan.get("signature") if isinstance(night_plan.get("signature"), Mapping) else {}
            phase = str(self._game_memory.get("phase") or "")
            if signature.get("round") != self._game_memory.get("round") or ("night" not in phase.lower() and "夜" not in phase):
                night_plan = None
        snapshot = {
            "game": {
                key: self._game_memory.get(key)
                for key in ("round", "phase", "sheriff_id", "alive_players", "dead_players", "death_order")
            },
            "public_ledger": {
                "claims": self._public_ledger["claims"][-3:],
                "votes": self._public_ledger["votes"][-3:],
                "deaths": self._public_ledger["deaths"][-3:],
                "sheriff_changes": self._public_ledger["sheriff_changes"][-2:],
            },
            "recent_vote_summary": self._game_memory.get("recent_votes", [])[-2:],
            "recent_public_texts": self._recent_public_texts[-3:],
            "recent_self_texts": self._recent_self_texts[-2:],
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
        game = self._compact_snapshot(
            turn_packet.get("game"),
            preferred_keys=(
                "round",
                "phase",
                "public_phase",
                "day",
                "night",
                "alive_count",
                "dead_count",
            ),
        )
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
        current_digest = self._current_public_state_digest(turn_packet, memory_snapshot)
        wolf_strategy = self._build_wolf_strategy(
            turn_packet, memory_snapshot, current_dialogue, current_digest
        )
        state_digest = {
            **current_digest,
            "recent_votes": memory_snapshot.get("recent_vote_summary", [])[-2:],
            "sheriff_changes": (memory_snapshot.get("public_ledger") or {}).get("sheriff_changes", [])[-2:],
            "recent_public_texts": memory_snapshot.get("recent_public_texts", [])[-2:],
            "recent_self_texts": memory_snapshot.get("recent_self_texts", [])[-2:],
        }
        return {
            "decision_state": {
                "game": game,
                "request": request,
                "self": self_state,
                "private_information": private_information,
                "state_digest": state_digest,
                "recent_dialogue": current_dialogue[-2:],
                "case_summary": self._decision_case_summary(wolf_strategy, current_digest),
                "wolf_strategy": wolf_strategy,
            }
        }

    def _build_wolf_strategy(
        self,
        turn_packet: Mapping[str, Any],
        memory_snapshot: Mapping[str, Any],
        current_dialogue: list[dict[str, Any]],
        current_digest: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        request = turn_packet.get("request") or {}
        phase = str(request.get("phase") or turn_packet.get("game", {}).get("phase") or "unknown")
        channel = str(request.get("channel") or "unknown")
        allowed_actions = request.get("allowed_actions") if isinstance(request.get("allowed_actions"), list) else []
        target_actions = [item for item in allowed_actions if isinstance(item, Mapping) and item.get("target_ids")]
        speech_actions = [item for item in allowed_actions if isinstance(item, Mapping) and item.get("kind") in _SPEECH_ACTION_KINDS]
        current_digest = current_digest or self._current_public_state_digest(turn_packet, memory_snapshot)
        phase_lower = phase.lower()
        channel_lower = channel.lower()
        is_night = any(token in phase_lower for token in ("night", "夜")) or any(
            token in channel_lower for token in ("night", "wolf")
        )
        if is_night and target_actions:
            candidate_ids = self._collect_target_ids(target_actions)
            discussion_signal = self._parse_wolf_discussion(current_dialogue, candidate_ids)
            signature = self._night_plan_signature(turn_packet, target_actions, discussion_signal)
            cached_plan = self._game_memory.get("wolf_plan")
            if isinstance(cached_plan, Mapping) and cached_plan.get("signature") == signature:
                plan = cached_plan.get("strategy")
                if isinstance(plan, Mapping):
                    return dict(plan)
            ranked_targets = self._rank_targets(turn_packet, memory_snapshot, target_actions, current_dialogue)
            primary = ranked_targets[0] if ranked_targets else None
            backup = ranked_targets[1] if len(ranked_targets) > 1 else None
            locked_plan = {
                "primary_target_id": primary.get("player_id") if isinstance(primary, Mapping) else None,
                "backup_target_id": backup.get("player_id") if isinstance(backup, Mapping) else None,
                "consensus_target_id": discussion_signal.get("consensus_target"),
                "discussion_primary_target_id": discussion_signal.get("primary_target"),
                "discussion_backup_target_id": discussion_signal.get("backup_target"),
            }
            plan = {
                "stage": "night",
                "objective": "先锁主刀与备刀；重试时复用同一锁定计划，只在候选合法性变化时重排",
                "primary_target": primary,
                "backup_target": backup,
                "tie_break_reason": primary.get("reason") if isinstance(primary, Mapping) else "优先高威胁公共目标，再按座位/ID固定顺序",
                "candidate_order": ranked_targets,
                "discussion_signal": discussion_signal,
                "locked_plan": locked_plan,
                "notes": self._night_notes(memory_snapshot, current_dialogue, discussion_signal),
                "signature": signature,
            }
            self._game_memory["wolf_plan"] = {
                "signature": signature,
                "primary_target_id": locked_plan.get("primary_target_id"),
                "backup_target_id": locked_plan.get("backup_target_id"),
                "consensus_target_id": locked_plan.get("consensus_target_id"),
                "strategy": plan,
            }
            return plan
        day_target_kinds = [
            str(item.get("kind") or "")
            for item in allowed_actions
            if isinstance(item, Mapping)
            and any(token in str(item.get("kind") or "").lower() for token in ("day_vote", "sheriff_vote", "sheriff_candidate", "sheriff_candidacy", "vote", "candidate"))
        ]
        if day_target_kinds:
            return self._build_day_target_strategy(
                turn_packet,
                memory_snapshot,
                current_dialogue,
                target_actions,
                current_digest,
            )
        if speech_actions:
            day_focus = self._select_day_focus(turn_packet, memory_snapshot, current_dialogue, speech_actions, current_digest)
            return {
                "stage": "day",
                "objective": "先固定对象、证据类型、下一步动作，再用一句解释补齐本局新证据",
                "must_include": ["对象", "证据类型", "下一步动作", "一句解释"],
                "day_focus": day_focus,
                "pressure_points": self._day_pressure_points(memory_snapshot, current_digest),
                "dialogue_focus": self._dialogue_focus(current_dialogue),
                "avoid_templates": self._anti_template_notes(),
                "anti_repeat": self._recent_self_texts[-1] if self._recent_self_texts else "",
            }
        return {
            "stage": "other",
            "objective": "只做当前合法行动，不额外发散",
            "avoid_templates": self._anti_template_notes(),
        }

    def _current_sheriff_id(
        self,
        public_state: Any,
        game: Any,
        memory_snapshot: Mapping[str, Any] | None = None,
    ) -> str | None:
        for source in (public_state, game):
            if not isinstance(source, Mapping):
                continue
            value = self._first_value(source, ("sheriff_id", "chairman_id"))
            if value is None:
                for key in ("sheriff", "chairman"):
                    if isinstance(source.get(key), Mapping):
                        value = self._first_value(source[key], ("player_id", "id", "seat", "seat_id"))
                        if value is not None:
                            break
            if value is not None:
                return str(value)
        if isinstance(memory_snapshot, Mapping):
            value = (memory_snapshot.get("game") or {}).get("sheriff_id")
            if value is not None:
                return str(value)
        return None

    def _current_alive_dead(
        self,
        public_state: Any,
        game: Any,
        memory_snapshot: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """从本次行动包的最新公开状态提取存活/死亡事实。"""
        alive: list[str] = []
        dead: list[str] = []
        players_alive_map: dict[str, bool] = {}

        def add(values: Any, bucket: list[str], alive_value: bool) -> None:
            ids = self._extract_id_list(values, preferred_keys=("players",)) if isinstance(values, Mapping) else []
            if not ids:
                raw = values.keys() if isinstance(values, Mapping) else values
                if isinstance(raw, list):
                    ids = [str(item) for item in raw if item is not None]
                elif raw is not None and not isinstance(raw, (str, bytes)):
                    ids = [str(item) for item in raw if item is not None]
                elif raw is not None:
                    ids = [str(raw)]
            for pid in ids:
                if pid not in bucket:
                    bucket.append(pid)
                players_alive_map[pid] = alive_value

        def inspect_players(source: Any) -> None:
            if isinstance(source, Mapping):
                entries = source.items()
            elif isinstance(source, list):
                entries = []
                for item in source:
                    if isinstance(item, Mapping):
                        pid = self._first_value(item, ("player_id", "id", "seat", "seat_id"))
                        if pid is not None:
                            entries.append((pid, item))
                    elif item is not None:
                        entries.append((item, {}))
            else:
                entries = []
            for pid, info in entries:
                pid = str(pid)
                state = info if isinstance(info, Mapping) else {}
                is_alive = self._is_alive_record(state)
                if is_alive is not None:
                    players_alive_map[pid] = is_alive
                    bucket = alive if is_alive else dead
                    if pid not in bucket:
                        bucket.append(pid)

        public_map = public_state if isinstance(public_state, Mapping) else {}
        game_map = game if isinstance(game, Mapping) else {}
        public_has_players = public_map.get("players") is not None
        public_has_alive = public_map.get("alive_players") is not None or public_map.get("living_players") is not None
        public_has_dead = public_map.get("dead_players") is not None or public_map.get("eliminated_players") is not None
        inspect_players(public_map.get("players"))
        if public_map.get("alive_players") is not None:
            add(public_map.get("alive_players"), alive, True)
        if public_map.get("living_players") is not None:
            add(public_map.get("living_players"), alive, True)
        if public_map.get("dead_players") is not None:
            add(public_map.get("dead_players"), dead, False)
        if public_map.get("eliminated_players") is not None:
            add(public_map.get("eliminated_players"), dead, False)
        # public_state 是本次行动包的最新事实源；game 只补齐它没有提供的类别，
        # 防止夜间转徽/死亡后的旧 game 快照覆盖当前状态。
        if not public_has_players:
            inspect_players(game_map.get("players"))
        if not public_has_alive:
            add(game_map.get("alive_players"), alive, True)
        if not public_has_dead and not public_has_alive:
            add(game_map.get("dead_players"), dead, False)

        if isinstance(memory_snapshot, Mapping):
            old_game = memory_snapshot.get("game") or {}
            if not alive:
                add(old_game.get("alive_players"), alive, True)
            if not dead:
                add(old_game.get("dead_players"), dead, False)
            for pid in old_game.get("alive_players") or []:
                players_alive_map.setdefault(str(pid), True)
            for pid in old_game.get("dead_players") or []:
                players_alive_map[str(pid)] = False
        dead_set = set(dead)
        alive = [pid for pid in alive if pid not in dead_set]
        death_order: list[str] = []
        for source in (public_state, game):
            if isinstance(source, Mapping):
                raw = self._extract_id_list(source, preferred_keys=("death_order",))
                for pid in raw:
                    if pid not in death_order:
                        death_order.append(pid)
        for pid in dead:
            if pid not in death_order:
                death_order.append(pid)
        return {
            "alive_player_ids": alive,
            "dead_player_ids": dead,
            "recent_death_order": death_order[-4:],
            "players_alive_map": {pid: bool(players_alive_map[pid]) for pid in sorted(players_alive_map)},
        }

    def _current_public_state_digest(
        self,
        turn_packet: Mapping[str, Any],
        memory_snapshot: Mapping[str, Any],
    ) -> dict[str, Any]:
        public_state = turn_packet.get("public_state") if isinstance(turn_packet.get("public_state"), Mapping) else {}
        game = turn_packet.get("game") if isinstance(turn_packet.get("game"), Mapping) else {}
        status = self._current_alive_dead(public_state, game, memory_snapshot)
        round_value = self._safe_get(game, "round")
        if round_value is None:
            round_value = self._safe_get(public_state, "round")
        phase_value = self._safe_get(game, "public_phase") or self._safe_get(game, "phase")
        if phase_value is None:
            phase_value = self._safe_get(public_state, "public_phase") or self._safe_get(public_state, "phase")
        status.update({
            "round": round_value,
            "phase": phase_value,
            "current_sheriff_id": self._current_sheriff_id(public_state, game, memory_snapshot),
        })
        # 兼容旧字段名，但 current_sheriff_id 是唯一应优先相信的徽位事实。
        status["sheriff_id"] = status["current_sheriff_id"]
        return status

    def _wolf_team_ids(self, private_information: Any) -> list[str]:
        if not isinstance(private_information, Mapping):
            return []
        found: list[str] = []
        for key in ("wolf_ids", "wolves", "teammates", "known_wolves", "team_members"):
            raw = private_information.get(key)
            if isinstance(raw, Mapping):
                values = raw.keys()
            elif isinstance(raw, (list, tuple, set)):
                values = raw
            elif raw is None:
                values = []
            else:
                values = [raw]
            for value in values:
                if value is not None and str(value) not in found:
                    found.append(str(value))
        return found

    def _alive_wolf_ids(
        self,
        private_information: Any,
        current_public_state_digest: Mapping[str, Any],
    ) -> list[str]:
        alive = {str(pid) for pid in current_public_state_digest.get("alive_player_ids") or []}
        return [pid for pid in self._wolf_team_ids(private_information) if pid in alive]

    def _build_day_target_strategy(
        self,
        turn_packet: Mapping[str, Any],
        memory_snapshot: Mapping[str, Any],
        current_dialogue: list[dict[str, Any]],
        target_actions: list[Mapping[str, Any]],
        current_digest: Mapping[str, Any],
    ) -> dict[str, Any]:
        kinds = [str(item.get("kind") or "") for item in (turn_packet.get("request", {}).get("allowed_actions") or []) if isinstance(item, Mapping)]
        lower_kinds = [kind.lower() for kind in kinds]
        is_sheriff = any("sheriff" in kind and ("vote" in kind or "candidate" in kind) for kind in lower_kinds)
        if is_sheriff:
            return self._build_sheriff_vote_strategy(turn_packet, memory_snapshot, current_dialogue, target_actions, current_digest)
        return self._build_day_vote_strategy(turn_packet, memory_snapshot, current_dialogue, target_actions, current_digest)

    def _build_day_vote_strategy(
        self,
        turn_packet: Mapping[str, Any],
        memory_snapshot: Mapping[str, Any],
        current_dialogue: list[dict[str, Any]],
        target_actions: list[Mapping[str, Any]],
        current_digest: Mapping[str, Any],
    ) -> dict[str, Any]:
        request = turn_packet.get("request") or {}
        allowed = [item for item in request.get("allowed_actions", []) if isinstance(item, Mapping)]
        vote_action = next((item for item in allowed if "day_vote" in str(item.get("kind", "")).lower()), None)
        if vote_action is None:
            vote_action = next((item for item in target_actions if "vote" in str(item.get("kind", "")).lower()), None)
        candidates = [str(pid) for pid in (vote_action or {}).get("target_ids", []) if pid is not None]
        records = self._extract_player_records(turn_packet.get("public_state"), turn_packet.get("game"), memory_snapshot)
        alive = set(current_digest.get("alive_player_ids") or [])
        dead = set(current_digest.get("dead_player_ids") or [])
        alive_known = bool(alive)
        teammates = set(self._alive_wolf_ids(turn_packet.get("private_information"), current_digest))
        pressure: list[tuple[int, tuple[int, str], str, str]] = []
        for pid in candidates:
            if pid in dead or (alive_known and pid not in alive) or pid == self.player_id:
                continue
            record = records.get(pid, {})
            score = 0
            reasons: list[str] = []
            mentions = self._recent_mention_count(pid)
            votes = self._nonnegative_int(record.get("vote_count", record.get("votes")), fallback=0) or 0
            if mentions:
                score += min(8, mentions * 2); reasons.append("近期被多人提及")
            if votes:
                score += min(8, votes * 2); reasons.append("票型集中")
            if self._public_claim_text(record):
                score += 3; reasons.append("公开身份链")
            if record.get("is_sheriff") or record.get("sheriff") or pid == current_digest.get("current_sheriff_id"):
                score += 3; reasons.append("当前警徽位")
            if not reasons:
                reasons.append("公共压力较低")
            pressure.append((score, self._seat_sort_value(record, pid), pid, ",".join(reasons[:3])))
        pressure.sort(key=lambda item: (-item[0], item[1], item[2]))
        non_wolves = [item for item in pressure if item[2] not in teammates]
        wolf_top = pressure[0] if pressure and pressure[0][2] in teammates else None
        text_blob = " ".join(str(item.get("text") or "") for item in current_dialogue[-6:])
        hard_bus = bool(wolf_top and (
            wolf_top[0] >= 8 or any(word in text_blob and wolf_top[2] in text_blob for word in ("查杀", "定狼", "出局", "归票"))
        ))
        pass_kind = next((str(item.get("kind")) for item in allowed if "pass" in str(item.get("kind", "")).lower()), None)
        if wolf_top and not hard_bus and non_wolves:
            chosen = non_wolves[0]; mode = "deflect"
        elif wolf_top and hard_bus:
            chosen = wolf_top; mode = "bus_teammate"
        elif non_wolves:
            chosen = non_wolves[0]; mode = "pressure_non_wolf"
        elif pass_kind:
            return {"stage": "day_vote", "mode": "pass", "recommended_kind": pass_kind, "recommended_target_id": None, "backup_target_id": None, "avoid_target_ids": list(teammates), "reason": "当前合法目标不足，保留弃票", "teammate_risk": {"alive_wolf_ids": sorted(teammates)}}
        elif pressure:
            chosen = pressure[0]; mode = "forced_legal_target"
        else:
            chosen = None; mode = "no_candidate"
        backup = next((item for item in non_wolves if not chosen or item[2] != chosen[2]), None)
        return {
            "stage": "day_vote",
            "recommended_kind": vote_action.get("kind") if vote_action else None,
            "recommended_target_id": chosen[2] if chosen else None,
            "backup_target_id": backup[2] if backup else None,
            "avoid_target_ids": sorted(teammates),
            "mode": mode,
            "reason": chosen[3] if chosen else "没有可排序的合法目标",
            "teammate_risk": {"alive_wolf_ids": sorted(teammates), "top_is_teammate": bool(wolf_top), "hard_bus": hard_bus},
        }

    def _build_sheriff_vote_strategy(
        self,
        turn_packet: Mapping[str, Any],
        memory_snapshot: Mapping[str, Any],
        current_dialogue: list[dict[str, Any]],
        target_actions: list[Mapping[str, Any]],
        current_digest: Mapping[str, Any],
    ) -> dict[str, Any]:
        allowed = [item for item in (turn_packet.get("request", {}).get("allowed_actions") or []) if isinstance(item, Mapping)]
        action = next((item for item in target_actions if "sheriff" in str(item.get("kind", "")).lower()), None)
        candidates = [str(pid) for pid in (action or {}).get("target_ids", []) if pid is not None]
        alive_wolves = set(self._alive_wolf_ids(turn_packet.get("private_information"), current_digest))
        blob = " ".join(str(item.get("text") or "") for item in current_dialogue[-6:]) + " " + " ".join(self._recent_public_texts[-4:])
        avoid: list[str] = []
        claimed_speakers = {
            str(item.get("speaker") or item.get("player_id") or item.get("from"))
            for item in current_dialogue[-6:]
            if any(word in str(item.get("text") or "") for word in ("预言家", "查杀", "报杀", "验人"))
        }
        for pid in candidates:
            if pid in alive_wolves and pid in blob:
                avoid.append(pid)
            elif pid in claimed_speakers:
                avoid.append(pid)
            elif pid in blob and any(word in blob for word in ("预言家", "查杀", "报杀", "验人")):
                avoid.append(pid)
        alive = set(current_digest.get("alive_player_ids") or [])
        dead = set(current_digest.get("dead_player_ids") or [])
        safe = [
            pid for pid in candidates
            if pid not in avoid and pid != self.player_id and pid not in dead
            and (not alive or pid in alive)
        ]
        pass_kind = next((str(item.get("kind")) for item in allowed if "pass" in str(item.get("kind", "")).lower()), None)
        if safe:
            recommended = safe[0]
            mode = "avoid_claimed_seer"
        elif pass_kind:
            return {"stage": "sheriff_vote", "mode": "pass", "recommended_kind": pass_kind, "recommended_target_id": None, "backup_target_id": None, "avoid_target_ids": avoid, "reason": "候选均与高风险查杀/预言家发言绑定", "teammate_risk": {"alive_wolf_ids": sorted(alive_wolves)}}
        elif candidates:
            recommended = candidates[0]; mode = "forced_legal_candidate"
        else:
            recommended = None; mode = "no_candidate"
        backup = next((pid for pid in candidates if pid != recommended and pid not in avoid), None)
        return {"stage": "sheriff_vote", "recommended_kind": action.get("kind") if action else None, "recommended_target_id": recommended, "backup_target_id": backup, "avoid_target_ids": avoid, "mode": mode, "reason": "优先避开公开跳预言家或报查杀的候选", "teammate_risk": {"alive_wolf_ids": sorted(alive_wolves)}}

    def _repair_illegal_target(
        self,
        action: Mapping[str, Any],
        request: Mapping[str, Any],
        wolf_strategy: Mapping[str, Any] | None,
    ) -> dict[str, Any] | None:
        repaired = dict(action)
        allowed = [item for item in request.get("allowed_actions", []) if isinstance(item, Mapping)]
        current = next((item for item in allowed if item.get("kind") == action.get("kind")), None)
        target_ids = [str(pid) for pid in (current or {}).get("target_ids", []) if pid is not None]
        pass_kind = next((str(item.get("kind")) for item in allowed if "pass" in str(item.get("kind", "")).lower()), None)
        if current is not None and pass_kind and isinstance(wolf_strategy, Mapping) and wolf_strategy.get("mode") == "pass":
            repaired.pop("target_id", None)
            repaired["kind"] = pass_kind
            return repaired
        if not target_ids:
            if action.get("target_id") and "vote" in str(action.get("kind", "")).lower() and pass_kind:
                repaired.pop("target_id", None); repaired["kind"] = pass_kind
                return repaired
            return None
        recommendations = []
        if isinstance(wolf_strategy, Mapping):
            for key in ("recommended_target_id", "backup_target_id"):
                value = wolf_strategy.get(key)
                if value is not None:
                    recommendations.append(str(value))
        for value in recommendations + target_ids:
            if value in target_ids:
                repaired["target_id"] = value
                return repaired
        return None

    def _night_plan_signature(
        self,
        turn_packet: Mapping[str, Any],
        target_actions: list[Mapping[str, Any]],
        discussion_signal: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        request = turn_packet.get("request") or {}
        game = turn_packet.get("game") or {}
        target_ids = self._collect_target_ids(target_actions)
        signal = discussion_signal or self._parse_wolf_discussion([], target_ids)
        return {
            "round": self._safe_get(game, "round"),
            "phase": str(request.get("phase") or self._safe_get(game, "phase") or "unknown"),
            "channel": str(request.get("channel") or "unknown"),
            "targets": sorted(target_ids),
            "discussion": {
                "primary_target": signal.get("primary_target"),
                "backup_target": signal.get("backup_target"),
                "consensus_target": signal.get("consensus_target"),
                "locked_target": signal.get("locked_target"),
                "has_disagreement": bool(signal.get("has_disagreement")),
                "candidate_targets": list(signal.get("candidate_targets") or [])[:3],
            },
        }

    def _collect_target_ids(self, target_actions: list[Mapping[str, Any]]) -> list[str]:
        target_ids: list[str] = []
        for action in target_actions:
            for target_id in action.get("target_ids") or []:
                target = str(target_id)
                if target not in target_ids:
                    target_ids.append(target)
        return target_ids

    def _parse_wolf_discussion(
        self,
        current_dialogue: list[dict[str, Any]],
        candidate_ids: list[str],
    ) -> dict[str, Any]:
        candidate_ids = [str(item) for item in candidate_ids if str(item)]
        if not current_dialogue or not candidate_ids:
            return {
                "candidate_targets": candidate_ids[:3],
                "mention_counts": {},
                "primary_target": None,
                "backup_target": None,
                "consensus_target": None,
                "locked_target": None,
                "disagreement_targets": [],
                "discussion_lines": [],
                "has_disagreement": False,
            }
        target_mentions: Counter[str] = Counter()
        primary_mentions: Counter[str] = Counter()
        backup_mentions: Counter[str] = Counter()
        consensus_mentions: Counter[str] = Counter()
        discussion_lines: list[str] = []
        primary_target: str | None = None
        backup_target: str | None = None
        consensus_target: str | None = None
        for item in current_dialogue[-6:]:
            text = item.get("text") or item.get("content") or item.get("message")
            if not text:
                continue
            text_value = str(text)
            hits = [candidate for candidate in candidate_ids if candidate in text_value]
            if not hits:
                continue
            discussion_lines.append(self._clip_text(text_value, 48))
            for candidate in hits:
                target_mentions[candidate] += 1
            if any(keyword in text_value for keyword in ("主刀", "先刀", "第一刀", "主目标", "优先刀", "先处理")):
                primary_target = hits[0]
                primary_mentions[hits[0]] += 2
            if any(keyword in text_value for keyword in ("备刀", "次刀", "保底", "兜底", "备选", "替补")):
                backup_target = hits[0]
                backup_mentions[hits[0]] += 2
            if any(keyword in text_value for keyword in ("定", "锁", "一致", "统一", "拍板", "最后", "就刀", "就打")):
                consensus_target = hits[0]
                consensus_mentions[hits[0]] += 2
        if primary_target is None and primary_mentions:
            primary_target = primary_mentions.most_common(1)[0][0]
        if primary_target is None and target_mentions:
            primary_target = target_mentions.most_common(1)[0][0]
        if backup_target is None and backup_mentions:
            backup_target = backup_mentions.most_common(1)[0][0]
        if backup_target is None:
            backup_target = next((target for target, _ in target_mentions.most_common() if target != primary_target), None)
        if consensus_target is None and consensus_mentions:
            consensus_target = consensus_mentions.most_common(1)[0][0]
        if consensus_target is None:
            consensus_target = primary_target
        candidate_order = [target for target, _ in target_mentions.most_common()]
        for target in (primary_target, backup_target, consensus_target):
            if target and target not in candidate_order:
                candidate_order.insert(0, target)
        disagreement_targets = [target for target in candidate_order if target_mentions.get(target, 0) > 0 and target != consensus_target]
        return {
            "candidate_targets": candidate_order[:5],
            "mention_counts": dict(target_mentions),
            "primary_target": primary_target,
            "backup_target": backup_target,
            "consensus_target": consensus_target,
            "locked_target": consensus_target or primary_target,
            "disagreement_targets": disagreement_targets[:3],
            "discussion_lines": discussion_lines[-3:],
            "has_disagreement": len(target_mentions) > 1,
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
        current_dialogue: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        public_state = turn_packet.get("public_state")
        players = self._extract_player_records(public_state, turn_packet.get("game"), memory_snapshot)
        current_digest = self._current_public_state_digest(turn_packet, memory_snapshot)
        dead_ids = {str(pid) for pid in current_digest.get("dead_player_ids") or []}
        candidate_ids = self._collect_target_ids(target_actions)
        discussion_signal = self._parse_wolf_discussion(current_dialogue, candidate_ids)
        candidates: list[dict[str, Any]] = []
        seen: set[str] = set()
        for action in target_actions:
            for target_id in action.get("target_ids") or []:
                target = str(target_id)
                if target in seen:
                    continue
                seen.add(target)
                record = players.get(target, {})
                if target in dead_ids or self._is_alive_record(record) is False:
                    continue
                score, reason = self._score_target(target, record, memory_snapshot)
                candidates.append(
                    {
                        "player_id": target,
                        "score": score,
                        "reason": reason,
                        "seat": self._seat_sort_value(record, target),
                        "base_reason": reason,
                    }
                )
        candidates.sort(key=lambda item: (-int(item["score"]), item["seat"], item["player_id"]))
        base_pool = candidates[:5]
        base_lookup = {item["player_id"]: item for item in candidates}
        pool_ids = [item["player_id"] for item in base_pool]
        for target in discussion_signal.get("candidate_targets", [])[:3]:
            if target in base_lookup and target not in pool_ids:
                pool_ids.append(target)
        locked_plan = self._game_memory.get("wolf_plan") if isinstance(self._game_memory.get("wolf_plan"), Mapping) else {}
        adjusted: list[dict[str, Any]] = []
        for target in pool_ids:
            item = base_lookup[target]
            score = int(item["score"])
            reasons = [item["base_reason"]]
            if target == discussion_signal.get("consensus_target"):
                score += 12
                reasons.append("夜间最后一致目标")
            if target == discussion_signal.get("primary_target"):
                score += 8
                reasons.append("狼队主刀共识")
            if target == discussion_signal.get("backup_target"):
                score += 4
                reasons.append("狼队备刀共识")
            if target in set(discussion_signal.get("disagreement_targets") or []):
                score += 1
                reasons.append("讨论中反复出现")
            if target == locked_plan.get("primary_target_id"):
                score += 2
                reasons.append("沿用已锁主刀")
            if target == locked_plan.get("backup_target_id"):
                score += 1
                reasons.append("沿用已锁备刀")
            if any(tag in item["base_reason"] for tag in ("公开", "票型", "警徽", "身份")):
                score += 1
                reasons.append("容易公开解释")
            adjusted.append(
                {
                    "player_id": target,
                    "score": score,
                    "reason": ",".join(reasons[:4]),
                    "seat": item["seat"],
                }
            )
        adjusted.sort(key=lambda item: (-int(item["score"]), item["seat"], item["player_id"]))
        return [
            {"player_id": item["player_id"], "reason": item["reason"]}
            for item in adjusted[:3]
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
        current_digest: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        public_state = turn_packet.get("public_state")
        records = self._extract_player_records(public_state, turn_packet.get("game"), memory_snapshot)
        current_digest = current_digest or self._current_public_state_digest(turn_packet, memory_snapshot)
        current_sheriff_id = current_digest.get("current_sheriff_id")
        dead_ids = {str(pid) for pid in current_digest.get("dead_player_ids") or []}
        candidates: list[dict[str, Any]] = []
        for pid, record in records.items():
            if pid == self.player_id:
                continue
            score = 0
            reasons: list[str] = []
            if pid in dead_ids or self._is_alive_record(record) is False:
                continue
            if record.get("is_sheriff") or record.get("sheriff") or pid == current_sheriff_id:
                score += 8
                reasons.append("当前持徽人")
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
            candidates.append(
                {
                    "player_id": pid,
                    "score": score,
                    "reason": ",".join(reasons[:3]) if reasons else "默认公共压力",
                    "claim": claim_text,
                }
            )
        candidates.sort(key=lambda item: (-int(item["score"]), self._seat_sort_value(records.get(item["player_id"], {}), item["player_id"]), item["player_id"]))
        chosen = candidates[0] if candidates else {}
        target_id = chosen.get("player_id")
        record = records.get(target_id, {}) if target_id else {}
        evidence = self._select_day_evidence(target_id, records, memory_snapshot, current_dialogue, current_digest)
        next_action_text = self._day_next_action(evidence.get("kind"))
        opening_guard = self._day_opening_guard()
        one_sentence_hint = self._clip_text(
            f"围绕{target_id or '当前焦点'}的{evidence.get('kind', 'fallback')}，先点证据，再接下一步动作：{next_action_text}",
            120,
        )
        return {
            "object": {
                "player_id": target_id,
                "seat": record.get("seat", record.get("seat_id")),
                "claim": chosen.get("claim"),
                "reason": chosen.get("reason"),
            },
            "evidence": evidence,
            "next_action": {
                "kind": evidence.get("kind"),
                "text": next_action_text,
            },
            "one_sentence_hint": one_sentence_hint,
            "opening_guard": opening_guard,
            "candidate_order": candidates[:3],
            "speech_budget": speech_actions[0].get("max_chars") if speech_actions else None,
        }

    def _select_day_evidence(
        self,
        target_id: str | None,
        records: Mapping[str, Mapping[str, Any]],
        memory_snapshot: Mapping[str, Any],
        current_dialogue: list[dict[str, Any]],
        current_digest: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        ledger = memory_snapshot.get("public_ledger") or {}
        current_digest = current_digest or {}
        dead_ids = {str(pid) for pid in current_digest.get("dead_player_ids") or []}
        current_sheriff_id = current_digest.get("current_sheriff_id")
        if target_id and target_id in dead_ids:
            return {"kind": "death", "text": f"{target_id}已死亡，不要求其继续发言、报查验或解释"}
        if target_id and target_id in records:
            record = records[target_id]
            claim_text = self._public_claim_text(record)
            if claim_text:
                historical = any(
                    entry.get("from") == target_id or entry.get("to") == target_id
                    for entry in (ledger.get("sheriff_changes") or [])[-3:]
                    if isinstance(entry, Mapping)
                )
                label = "当前持徽人" if target_id == current_sheriff_id else ("曾持徽/警徽链相关" if historical else "公开身份")
                return {"kind": "claim", "text": f"{target_id}{label}:{claim_text}"}
            for entry in (ledger.get("votes") or [])[-4:]:
                if entry.get("player_id") == target_id or entry.get("voter") == target_id or entry.get("target_id") == target_id:
                    return {"kind": "vote", "text": self._clip_text(json.dumps(entry, ensure_ascii=False), 48)}
            for entry in (ledger.get("sheriff_changes") or [])[-2:]:
                if entry.get("to") == target_id or entry.get("from") == target_id:
                    label = "当前持徽人" if target_id == current_sheriff_id else "曾持徽/警徽链相关"
                    return {"kind": "sheriff", "text": f"{label}:{self._clip_text(json.dumps(entry, ensure_ascii=False), 40)}"}
        for entry in (ledger.get("deaths") or [])[-2:]:
            if isinstance(entry, Mapping):
                return {"kind": "death", "text": self._clip_text(json.dumps(entry, ensure_ascii=False), 48)}
        for item in reversed(current_dialogue[-3:]):
            speaker = item.get("speaker") or item.get("player_id") or item.get("from")
            text = item.get("text") or item.get("content") or item.get("message")
            if speaker is not None and text:
                return {
                    "kind": "dialogue",
                    "text": f"{speaker}:{self._clip_text(str(text), 40)}",
                }
        return {"kind": "fallback", "text": "用当前公开信息补一条最具体的链条"}

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
        discussion_signal: Mapping[str, Any] | None = None,
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
        recent_votes = memory_snapshot.get("recent_vote_summary") or []
        if recent_votes:
            notes.append("结合最近票型收敛刀口")
        if discussion_signal:
            primary = discussion_signal.get("primary_target")
            backup = discussion_signal.get("backup_target")
            locked = discussion_signal.get("consensus_target") or discussion_signal.get("locked_target")
            if locked:
                notes.append(f"狼队锁定{self._clip_text(str(locked), 18)}")
            if backup and backup != locked:
                notes.append(f"备刀{self._clip_text(str(backup), 18)}")
            if discussion_signal.get("has_disagreement"):
                notes.append("讨论有分歧时优先沿用最后一致目标")
            elif primary:
                notes.append(f"主刀{self._clip_text(str(primary), 18)}")
            lines = discussion_signal.get("discussion_lines") or []
            if lines:
                notes.append(f"讨论原文:{lines[-1]}")
        elif current_dialogue:
            notes.append("优先处理夜间讨论中的高威胁公共目标")
        return notes[:4]

    def _day_pressure_points(
        self,
        memory_snapshot: Mapping[str, Any],
        current_digest: Mapping[str, Any] | None = None,
    ) -> list[str]:
        points: list[str] = []
        current_digest = current_digest or {}
        game = memory_snapshot.get("game", {})
        dead_players = current_digest.get("dead_player_ids") or game.get("dead_players") or []
        if dead_players:
            points.append(f"围绕最近死亡：{self._format_id_list(dead_players[-2:])}")
        public_ledger = memory_snapshot.get("public_ledger") or {}
        sheriff_changes = public_ledger.get("sheriff_changes") or []
        current_sheriff = current_digest.get("current_sheriff_id")
        if current_sheriff:
            points.append(f"当前持徽人：{current_sheriff}")
        if sheriff_changes:
            last = sheriff_changes[-1]
            if isinstance(last, Mapping):
                points.append(f"最近警徽变化：{last.get('from')}→{last.get('to')}")
        recent_votes = memory_snapshot.get("recent_vote_summary") or []
        if recent_votes:
            points.append("根据上一轮票型调整站边")
        return points[:3]

    def _dialogue_focus(self, current_dialogue: list[dict[str, Any]]) -> list[str]:
        focus: list[str] = []
        for item in current_dialogue[-3:]:
            speaker = item.get("speaker") or item.get("player_id") or item.get("from")
            text = item.get("text") or item.get("content") or item.get("message")
            if speaker is not None and text:
                focus.append(f"{speaker}:{self._clip_text(str(text), 28)}")
        return focus

    def _day_opening_guard(self) -> dict[str, Any]:
        banned_openings: list[str] = []
        for text in self._recent_self_texts[-2:]:
            opening = self._leading_clause(text)
            if opening and opening not in banned_openings:
                banned_openings.append(opening)
        return {
            "banned_openings": banned_openings,
            "note": "不要复用这些开场或其同义句，必须换成本局新证据。",
        }

    @staticmethod
    def _leading_clause(text: str) -> str:
        value = str(text).strip()
        for separator in ("。", "，", "：", ":", "；", ";", "、", "\n"):
            if separator in value:
                value = value.split(separator, 1)[0].strip()
                break
        return value[:24]

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

    def _decision_case_summary(
        self,
        wolf_strategy: Mapping[str, Any],
        current_digest: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        stage = str(wolf_strategy.get("stage") or "other")
        if stage == "night":
            return {
                "mode": "night",
                "primary": wolf_strategy.get("primary_target"),
                "backup": wolf_strategy.get("backup_target"),
                "discussion_lock": (wolf_strategy.get("discussion_signal") or {}).get("consensus_target"),
                "reason": wolf_strategy.get("tie_break_reason"),
                "current_sheriff_id": (current_digest or {}).get("current_sheriff_id"),
            }
        if stage == "day":
            day_focus = wolf_strategy.get("day_focus")
            if isinstance(day_focus, Mapping):
                return {
                    "mode": "day",
                    "object": day_focus.get("object"),
                    "evidence": day_focus.get("evidence"),
                    "next_action": day_focus.get("next_action"),
                    "opening_guard": day_focus.get("opening_guard"),
                    "current_sheriff_id": (current_digest or {}).get("current_sheriff_id"),
                }
        if stage in {"day_vote", "sheriff_vote"}:
            return {
                "mode": stage,
                "recommended_kind": wolf_strategy.get("recommended_kind"),
                "recommended_target_id": wolf_strategy.get("recommended_target_id"),
                "backup_target_id": wolf_strategy.get("backup_target_id"),
                "avoid_target_ids": wolf_strategy.get("avoid_target_ids", []),
                "strategy_mode": wolf_strategy.get("mode"),
                "reason": wolf_strategy.get("reason"),
                "current_sheriff_id": (current_digest or {}).get("current_sheriff_id"),
            }
        return {"mode": stage}

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
        for item in items[-_MAX_HISTORY_ITEMS:]:
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
        return snippets[-_MAX_HISTORY_ITEMS:]

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
        if record.get("alive") is False:
            return False
        status = record.get("status")
        if status in {"dead", "eliminated"}:
            return False
        if status in {"alive", "living"}:
            return True
        if record.get("is_dead") is True:
            return False
        if record.get("is_alive") is True:
            return True
        if record.get("is_alive") is False:
            return False
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



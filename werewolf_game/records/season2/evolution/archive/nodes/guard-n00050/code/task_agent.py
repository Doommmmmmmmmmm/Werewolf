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
_PLAYER_ID_PATTERN = re.compile(r"\bp\d+\b")

CURRENT_ROUND_DIALOGUE_TOOL_NAME = "read_current_round_dialogue"
DEFAULT_MAX_TOOL_CALLS_PER_DECISION = 5
DEFAULT_MAX_TOOL_RESULT_TOKENS = 1000
DEFAULT_MAX_PROMPT_CHARS = 12000
# 这是 Task-Agent 层的“纠错重试”次数：首次模型调用之外，最多再请求两次。
# 传输层的 HTTP/网络重试仍由 ModelClient 的 MODEL_MAX_RETRIES 单独控制。
DEFAULT_MAX_DECISION_RETRIES = 2
_SPEECH_ACTION_KINDS = frozenset({"speak", "last_words"})
_GUARD_ROLE_NAME = "guard"
_POSITIVE_TARGET_CLAIM_TYPES = frozenset({"check_clear", "public_check_clear"})
_NEGATIVE_TARGET_CLAIM_TYPES = frozenset({"check_kill", "public_check_kill", "sheriff_from_suspect"})
_ROLE_SELF_CLAIM_TYPES = frozenset({"seer_claim", "guard_claim", "witch_claim", "hunter_claim"})


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
        normalized_target_ids = {
            str(target_id)
            for target_id in target_ids
            if target_id is not None
        }
        if str(action.get("target_id")) not in normalized_target_ids:
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
        self.last_protected_id: str | None = None
        self.last_protected_round: Any = None
        self.public_claims: dict[str, set[str]] = {}
        self.public_claim_records: list[dict[str, Any]] = []
        self._claim_event_seq = 0
        self.suspected_wolves: set[str] = set()
        self.trusted_info_ids: set[str] = set()
        self.self_publicly_claimed_guard = False
        self.current_public_sheriff_id: str | None = None
        self.public_alive_ids: list[str] = []
        self.public_dead_ids: set[str] = set()
        self._model_token_usage: dict[str, int] = {
            "successful_response_count": 0,
            "api_attempt_count": 0,
            "reported_usage_response_count": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }

    async def observe(self, sync_packet: dict[str, Any]) -> None:
        """接收公开同步，并只保存本局可见的最小摘要。"""

        self._ingest_public_packet(sync_packet)

    async def decide(self, turn_packet: dict[str, Any]) -> dict[str, Any]:
        if turn_packet["self"]["player_id"] != self.player_id:
            raise ValueError("行动包被发送给了错误玩家")
        private = turn_packet["private_information"]
        if str(private.get("role")) != self.profile.role:
            raise ValueError("Task-Agent 的角色档案与行动包身份不一致")

        self._ingest_public_packet(turn_packet)

        system = self._system_prompt(private)
        guard_summary = self._guard_memory_snapshot(turn_packet)
        prompt = {
            "request": turn_packet["request"],
            "guard_summary": guard_summary,
            "history_policy": {
                "default_context": "guard_summary_only",
                "current_round_dialogue_tool": CURRENT_ROUND_DIALOGUE_TOOL_NAME,
                "max_tool_calls": self.max_tool_calls_per_decision,
                "max_tool_result_tokens": self.max_tool_result_tokens,
            },
        }

        tool_context = turn_packet.get("tool_context") or {}
        current_dialogue = tool_context.get("current_round_dialogue") or []
        self._ingest_dialogue(current_dialogue)

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
                "instruction": self._turn_instruction(turn_packet["request"], feedback, guard_summary),
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
            action = self._repair_guard_decision(action, turn_packet["request"], turn_packet)
            error = decision_error(action, turn_packet["request"])
            if error is None:
                self._record_successful_guard_action(action, turn_packet)
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

    def _turn_instruction(
        self,
        request: Mapping[str, Any],
        feedback: str,
        guard_summary: Mapping[str, Any] | None = None,
    ) -> str:
        instruction = render_prompt(
            "player_turn_instruction.txt",
            action_contract=render_action_contract(request),
            validation_feedback=(f"上一次输出未通过校验：{feedback}" if feedback else ""),
        )
        extra_notes: list[str] = []
        if self.profile.role == _GUARD_ROLE_NAME:
            summary_bits: list[str] = []
            if isinstance(guard_summary, Mapping):
                top_targets = guard_summary.get("top_targets")
                if isinstance(top_targets, list) and top_targets:
                    candidate_bits: list[str] = []
                    for item in top_targets[:3]:
                        if not isinstance(item, Mapping):
                            continue
                        target_id = self._normalize_player_id(item.get("target_id")) or "?"
                        score = item.get("score")
                        candidate_bits.append(f"{target_id}:{score}")
                    if candidate_bits:
                        summary_bits.append("候选分数：" + "；".join(candidate_bits))
                contradictions = guard_summary.get("strongest_public_contradictions")
                if isinstance(contradictions, list) and contradictions:
                    contradiction_bits = [str(item) for item in contradictions[:3] if item]
                    if contradiction_bits:
                        summary_bits.append("公开矛盾：" + "；".join(contradiction_bits))
                if guard_summary.get("self_protect_hint"):
                    summary_bits.append("存在自守优先信号。")
            extra_notes.append(
                "守卫专用：只在本次 contract 的 target_ids 中选择。目标优先级应按“今晚更可能被狼刀 + 守住后对好人更有价值”来排，不要机械追警长。"
            )
            if summary_bits:
                extra_notes.append("守卫摘要：" + " ".join(summary_bits))
            extra_notes.append(
                "评分优先看近期公开压力、真假信息源、警长可靠度、上一夜守护历史、是否已暴露，以及残局是否需要自守。"
            )
            if self.self_publicly_claimed_guard:
                extra_notes.append(
                    "你已经公开跳过守卫；后续发言继续保持守卫视角，不要改口成平民或否认夜间身份。"
                )
        if extra_notes:
            instruction = instruction + "\n\n" + "\n".join(extra_notes)
        return instruction

    def _guard_memory_snapshot(self, turn_packet: Mapping[str, Any]) -> dict[str, Any]:
        request = turn_packet.get("request") if isinstance(turn_packet, Mapping) else None
        round_value = self._current_round_value(turn_packet)
        guard_targets = self._extract_allowed_target_ids(request, "guard_protect") if isinstance(request, Mapping) else []
        known_claims = self._compact_public_claims(max_players=6)
        if guard_targets and len(guard_targets) > 6:
            guard_targets = guard_targets[:6]
        return {
            "last_protected_id": self.last_protected_id,
            "last_protected_round": self.last_protected_round,
            "current_round": round_value,
            "self_public_claim": "guard" if self.self_publicly_claimed_guard else "hidden",
            "trusted_info_ids": self._compact_id_list(self.trusted_info_ids, limit=5),
            "suspected_wolves": self._compact_id_list(self.suspected_wolves, limit=5),
            "known_public_claims": known_claims,
            "current_sheriff_id": self.current_public_sheriff_id,
            "current_guard_targets": guard_targets,
        }

    def _repair_guard_decision(
        self,
        action: dict[str, Any],
        request: Mapping[str, Any],
        turn_packet: Mapping[str, Any],
    ) -> dict[str, Any]:
        if action.get("kind") != "guard_protect":
            return action
        allowed_target_ids = self._extract_allowed_target_ids(request, "guard_protect")
        if not allowed_target_ids:
            if self._pass_allowed(request):
                return self._make_pass_action(request, action)
            return action
        target_id = action.get("target_id")
        if target_id in allowed_target_ids:
            return action
        ranked = self._rank_guard_targets(allowed_target_ids, turn_packet)
        if ranked:
            repaired = dict(action)
            repaired["target_id"] = ranked[0]
            return repaired
        if self._pass_allowed(request):
            return self._make_pass_action(request, action)
        return action

    def _rank_guard_targets(
        self,
        candidates: list[str],
        turn_packet: Mapping[str, Any],
    ) -> list[str]:
        candidate_scores: list[tuple[int, int, str]] = []
        for index, candidate in enumerate(candidates):
            score, _ = self._score_guard_target(candidate, turn_packet)
            candidate_scores.append((score, index, candidate))
        candidate_scores.sort(key=lambda item: (-item[0], item[1]))
        return [candidate for _, _, candidate in candidate_scores]

    def _score_guard_target(self, candidate: str, turn_packet: Mapping[str, Any]) -> tuple[int, list[str]]:
        candidate_id = self._normalize_player_id(candidate)
        if not candidate_id:
            return -1000, ["空目标"]
        public_state = turn_packet.get("public_state") if isinstance(turn_packet, Mapping) else None
        alive_ids = self._alive_id_list(public_state) if isinstance(public_state, Mapping) else []
        if not alive_ids:
            alive_ids = list(self.public_alive_ids)
        alive_set = set(alive_ids)
        dead_ids = self._dead_id_set(public_state) if isinstance(public_state, Mapping) else set()
        if self.public_dead_ids:
            dead_ids.update(self.public_dead_ids)
        if candidate_id in dead_ids or (alive_set and candidate_id not in alive_set):
            return -1000, ["已死亡"]
        current_round = self._current_round_value(turn_packet)
        alive_count = len(alive_set) if alive_set else len(self.public_alive_ids)
        claim_records = self._claim_records_for_target(candidate_id)
        score = 0
        reasons: list[str] = []

        pressure_score, pressure_notes = self._claim_pressure_score(candidate_id)
        score += pressure_score
        reasons.extend(pressure_notes[:2])

        supported_claims = sum(1 for record in claim_records if str(record.get("status") or "") == "supported")
        contradicted_claims = sum(1 for record in claim_records if str(record.get("status") or "") == "contradicted")
        if supported_claims:
            support_bonus = min(20, supported_claims * 4)
            score += support_bonus
            reasons.append(f"已证{supported_claims}")
        if contradicted_claims:
            contradiction_penalty = min(24, contradicted_claims * 5)
            score -= contradiction_penalty
            reasons.append(f"反证{contradicted_claims}")

        if candidate_id in self.trusted_info_ids:
            info_bonus = 34 + min(12, max(0, pressure_score))
            score += info_bonus
            reasons.append("可信信息位")
        if candidate_id == self.current_public_sheriff_id:
            sheriff_reliability = self._speaker_reliability(candidate_id)
            if candidate_id in self.suspected_wolves or sheriff_reliability < 0:
                sheriff_bonus = -18
                reasons.append("不稳警长")
            elif sheriff_reliability >= 10:
                sheriff_bonus = 22 + min(14, sheriff_reliability // 2)
                reasons.append("可靠警长")
            else:
                sheriff_bonus = 10 + max(0, sheriff_reliability // 3)
                reasons.append("警长")
            score += sheriff_bonus
        if candidate_id in self.suspected_wolves:
            score -= 70
            reasons.append("狼嫌")

        reliability = self._speaker_reliability(candidate_id)
        if reliability > 0 and candidate_id != self.player_id:
            score += min(18, reliability // 2)
            reasons.append("来源可靠")
        elif reliability < 0:
            score += max(-16, reliability // 2)
            reasons.append("来源不稳")

        if candidate_id == self.last_protected_id:
            gap = self._round_gap(self.last_protected_round, current_round)
            penalty = 90 if gap is None or gap <= 1 else 58
            if alive_count and alive_count <= 4:
                penalty -= 18
            score -= penalty
            reasons.append("刚守过")

        if alive_count and alive_count <= 5 and candidate_id != self.player_id:
            score += 5
        if not claim_records and candidate_id != self.player_id:
            score -= 6
            reasons.append("信息稀薄")

        if candidate_id == self.player_id:
            self_bonus = 12
            if alive_count and alive_count <= 4:
                self_bonus += 50
                reasons.append("残局自守")
            elif alive_count and alive_count <= 5:
                self_bonus += 28
                reasons.append("低人数自守")
            if self.self_publicly_claimed_guard:
                self_bonus += 12
                reasons.append("已跳守卫")
            own_pressure = pressure_score + max(0, self._speaker_reliability(self.player_id) // 2)
            if own_pressure >= 12:
                self_bonus += min(16, own_pressure)
                reasons.append("自守压力高")
            elif own_pressure <= -8:
                self_bonus += 6
                reasons.append("低暴露")
            score += self_bonus
        elif alive_count and alive_count <= 4 and candidate_id in self.trusted_info_ids:
            score += 8
            reasons.append("残局信息位")
        elif alive_count and alive_count <= 4 and candidate_id == self.current_public_sheriff_id:
            score += 6
            reasons.append("残局警长")

        return score, reasons

    def _record_successful_guard_action(self, action: Mapping[str, Any], turn_packet: Mapping[str, Any]) -> None:
        if action.get("kind") != "guard_protect":
            return
        target_id = action.get("target_id")
        if not target_id:
            return
        self.last_protected_id = str(target_id)
        self.last_protected_round = self._current_round_value(turn_packet)

    def _ingest_public_packet(self, packet: Mapping[str, Any] | Any) -> None:
        if not isinstance(packet, Mapping):
            return
        public_state = packet.get("public_state")
        if isinstance(public_state, Mapping):
            self._ingest_public_state(public_state)
        elif "private_information" not in packet:
            self._ingest_public_state(packet)
        tool_context = packet.get("tool_context")
        if isinstance(tool_context, Mapping):
            self._ingest_dialogue(tool_context.get("current_round_dialogue"))
        self._ingest_dialogue(packet.get("current_round_dialogue"))

    def _ingest_public_state(self, public_state: Mapping[str, Any]) -> None:
        alive_ids = self._alive_id_list(public_state)
        if alive_ids:
            self.public_alive_ids = alive_ids
        dead_ids = self._dead_id_set(public_state)
        if dead_ids:
            self.public_dead_ids.update(dead_ids)
        sheriff_id = self._extract_first_player_id(
            self._first_present(
                public_state,
                "sheriff_id",
                "current_sheriff_id",
                "police_id",
                "badge_owner_id",
            )
        )
        if sheriff_id:
            self.current_public_sheriff_id = sheriff_id
        for key, value in public_state.items():
            if key in {"alive_players", "alive_player_ids", "living_players"}:
                continue
            if key in {"dead_players", "dead_player_ids", "executed_players", "graveyard"}:
                continue
            self._ingest_value(value)

    def _ingest_dialogue(self, dialogue: Any) -> None:
        if dialogue is None:
            return
        if isinstance(dialogue, Mapping):
            self._ingest_dialogue_item(dialogue)
            return
        if isinstance(dialogue, list):
            for item in dialogue:
                self._ingest_dialogue(item)
            return
        if isinstance(dialogue, str):
            self._ingest_text(None, dialogue)

    def _ingest_dialogue_item(self, item: Mapping[str, Any]) -> None:
        speaker = self._extract_first_player_id(
            self._first_present(item, "speaker_id", "player_id", "sender_id", "author_id")
        )
        text = self._first_text_present(item, "text", "content", "message", "utterance", "line")
        if text:
            self._ingest_text(speaker, text)
        for value in item.values():
            if value is item.get("text") or value is item.get("content") or value is item.get("message"):
                continue
            if isinstance(value, (Mapping, list, str)):
                self._ingest_value(value, speaker=speaker)

    def _ingest_value(self, value: Any, speaker: str | None = None) -> None:
        if isinstance(value, Mapping):
            text = self._first_text_present(value, "text", "content", "message", "utterance", "line")
            if text:
                nested_speaker = self._extract_first_player_id(
                    self._first_present(value, "speaker_id", "player_id", "sender_id", "author_id")
                ) or speaker
                self._ingest_text(nested_speaker, text)
            for nested in value.values():
                if isinstance(nested, (Mapping, list, str)):
                    self._ingest_value(nested, speaker=speaker)
            return
        if isinstance(value, list):
            for item in value:
                self._ingest_value(item, speaker=speaker)
            return
        if isinstance(value, str):
            self._ingest_text(speaker, value)

    def _ingest_text(self, speaker_id: str | None, text: str) -> None:
        cleaned = str(text).strip()
        if not cleaned:
            return
        if not self._should_track_text(cleaned):
            return
        speaker = self._normalize_player_id(speaker_id)
        lower = cleaned.lower()
        ids_in_text = self._extract_player_ids(cleaned)

        if speaker == self.player_id and ("守卫" in cleaned or "guard" in lower):
            self.self_publicly_claimed_guard = True
        if "我是守卫" in cleaned or ("守卫" in cleaned and ("我是" in cleaned or "跳守卫" in cleaned)):
            self._note_public_claim(speaker, "guard_claim")
            self._register_claim_record(speaker, "guard_claim", speaker, cleaned)
            if speaker == self.player_id:
                self.self_publicly_claimed_guard = True
        if "我是预言家" in cleaned or ("预言家" in cleaned and ("查杀" in cleaned or "金水" in cleaned or "验" in cleaned)):
            self._note_public_claim(speaker, "seer_claim")
            self._register_claim_record(speaker, "seer_claim", speaker, cleaned)
        if "我是女巫" in cleaned or ("女巫" in cleaned and "我是" in cleaned):
            self._note_public_claim(speaker, "witch_claim")
            self._register_claim_record(speaker, "witch_claim", speaker, cleaned)
        if "我是猎人" in cleaned or ("猎人" in cleaned and "我是" in cleaned):
            self._note_public_claim(speaker, "hunter_claim")
            self._register_claim_record(speaker, "hunter_claim", speaker, cleaned)
        if "警长" in cleaned or "警徽" in cleaned:
            self._note_public_claim(speaker, "sheriff_related")
        if "移交" in cleaned and "警徽" in cleaned:
            self._note_public_claim(speaker, "sheriff_transfer")
            for pid in ids_in_text:
                if pid != speaker:
                    self._register_claim_record(speaker, "sheriff_transfer", pid, cleaned)
                    if speaker in self.suspected_wolves:
                        self._note_public_claim(pid, "sheriff_from_suspect")
        if "查杀" in cleaned:
            self._note_public_claim(speaker, "check_kill")
            if self._looks_like_info_provider(cleaned, speaker):
                self.trusted_info_ids.add(speaker)
            for pid in ids_in_text:
                if pid != speaker:
                    self.suspected_wolves.add(pid)
                    self._note_public_claim(pid, "public_check_kill")
                    self._register_claim_record(speaker, "check_kill", pid, cleaned)
        if "金水" in cleaned:
            self._note_public_claim(speaker, "check_clear")
            if self._looks_like_info_provider(cleaned, speaker):
                self.trusted_info_ids.add(speaker)
            for pid in ids_in_text:
                if pid != speaker:
                    self._note_public_claim(pid, "public_check_clear")
                    self._register_claim_record(speaker, "check_clear", pid, cleaned)
        if "对跳" in cleaned and "守卫" in cleaned:
            self._note_public_claim(speaker, "guard_counterclaim")
        if speaker and speaker == self.player_id and "守卫" in cleaned:
            self.self_publicly_claimed_guard = True

    def _looks_like_info_provider(self, text: str, speaker: str | None) -> bool:
        if speaker is None:
            return False
        normalized = self._normalize_player_id(speaker)
        if normalized in self.trusted_info_ids:
            return True
        reliability = self._speaker_reliability(normalized)
        saw_structured_info = False
        for record in self.public_claim_records:
            if self._normalize_player_id(record.get("speaker_id")) != normalized:
                continue
            if record.get("claim_type") not in {"seer_claim", "check_kill", "check_clear"}:
                continue
            status = str(record.get("status") or "pending")
            if status == "contradicted":
                continue
            saw_structured_info = True
            if status == "supported":
                return True
        if saw_structured_info and reliability >= 6:
            return True
        if ("预言家" in text or "验" in text or "查杀" in text or "金水" in text) and reliability >= 8:
            return True
        labels = self.public_claims.get(normalized, set())
        return "seer_claim" in labels and reliability >= 10

    def _note_public_claim(self, player_id: str | None, label: str) -> None:
        normalized = self._normalize_player_id(player_id)
        if not normalized or not label:
            return
        labels = self.public_claims.setdefault(normalized, set())
        labels.add(label)

    def _register_claim_record(self, speaker_id: str | None, claim_type: str, target_id: str | None, text: str) -> None:
        speaker = self._normalize_player_id(speaker_id)
        claim_type = str(claim_type or "").strip()
        if not speaker or not claim_type:
            return
        target = self._normalize_player_id(target_id) or (speaker if claim_type in _ROLE_SELF_CLAIM_TYPES else None)
        record: dict[str, Any] = {
            "seq": self._claim_event_seq + 1,
            "speaker_id": speaker,
            "claim_type": claim_type,
            "target_id": target,
            "text": text,
            "status": "pending",
        }
        self._claim_event_seq += 1
        self._update_claim_record_status(record)
        self.public_claim_records.append(record)
        if len(self.public_claim_records) > 80:
            self.public_claim_records.pop(0)
        self._note_public_claim(speaker, claim_type)
        if target and target != speaker:
            self._note_public_claim(target, f"{claim_type}_target")
        if speaker == self.player_id and claim_type == "guard_claim":
            self.self_publicly_claimed_guard = True

    def _update_claim_record_status(self, record: Mapping[str, Any]) -> None:
        speaker = self._normalize_player_id(record.get("speaker_id"))
        target = self._normalize_player_id(record.get("target_id"))
        claim_type = str(record.get("claim_type") or "")
        if not speaker or not claim_type:
            return
        status = str(record.get("status") or "pending")
        if speaker in self.trusted_info_ids and claim_type in _POSITIVE_TARGET_CLAIM_TYPES.union(_NEGATIVE_TARGET_CLAIM_TYPES):
            status = "supported"
        if speaker in self.suspected_wolves and claim_type in _POSITIVE_TARGET_CLAIM_TYPES.union(_ROLE_SELF_CLAIM_TYPES):
            status = "contradicted"
        for existing in self.public_claim_records:
            existing_speaker = self._normalize_player_id(existing.get("speaker_id"))
            existing_target = self._normalize_player_id(existing.get("target_id"))
            existing_type = str(existing.get("claim_type") or "")
            existing_status = str(existing.get("status") or "pending")
            if existing_speaker == speaker:
                if existing_type == claim_type and existing_target != target:
                    existing["status"] = "contradicted"
                    status = "contradicted"
                if existing_type == "seer_claim" and claim_type in {"check_kill", "check_clear"}:
                    existing["status"] = "supported"
                    if status != "contradicted":
                        status = "supported"
                if claim_type == "seer_claim" and existing_type in {"check_kill", "check_clear"}:
                    existing["status"] = "supported"
                    if status != "contradicted":
                        status = "supported"
            if target and existing_target == target:
                if self._claims_conflict(existing_type, claim_type):
                    existing["status"] = "contradicted"
                    status = "contradicted"
                elif existing_type == claim_type and existing_speaker != speaker and existing_status != "contradicted":
                    existing["status"] = "supported"
                    if status != "contradicted":
                        status = "supported"
        if status != "contradicted" and claim_type in _POSITIVE_TARGET_CLAIM_TYPES and speaker in self.trusted_info_ids:
            status = "supported"
        record_status = status if status in {"pending", "supported", "contradicted"} else "pending"
        if isinstance(record, dict):
            record["status"] = record_status

    def _claims_conflict(self, first: Any, second: Any) -> bool:
        first_type = str(first or "")
        second_type = str(second or "")
        if not first_type or not second_type or first_type == second_type:
            return False
        if (first_type in _POSITIVE_TARGET_CLAIM_TYPES and second_type in _NEGATIVE_TARGET_CLAIM_TYPES) or (
            first_type in _NEGATIVE_TARGET_CLAIM_TYPES and second_type in _POSITIVE_TARGET_CLAIM_TYPES
        ):
            return True
        return first_type in _ROLE_SELF_CLAIM_TYPES and second_type in _ROLE_SELF_CLAIM_TYPES

    def _claim_records_for_target(self, target_id: str | None) -> list[dict[str, Any]]:
        normalized_target = self._normalize_player_id(target_id)
        if not normalized_target:
            return []
        return [
            record
            for record in self.public_claim_records
            if self._normalize_player_id(record.get("target_id")) == normalized_target
        ]

    def _claim_record_weight(self, record: Mapping[str, Any]) -> int:
        claim_type = str(record.get("claim_type") or "")
        status = str(record.get("status") or "pending")
        speaker = self._normalize_player_id(record.get("speaker_id"))
        base = 4
        if claim_type in _NEGATIVE_TARGET_CLAIM_TYPES:
            base = 14
        elif claim_type in _POSITIVE_TARGET_CLAIM_TYPES:
            base = 12
        elif claim_type in _ROLE_SELF_CLAIM_TYPES:
            base = 8
        elif claim_type == "sheriff_transfer":
            base = 6
        if status == "supported":
            base += 6
        elif status == "contradicted":
            base = max(2, base // 3)
        base += max(-6, min(8, self._speaker_reliability(speaker) // 3))
        seq = self._nonnegative_int(record.get("seq"), fallback=None)
        if seq is not None:
            age = max(0, self._claim_event_seq - seq)
            if age <= 1:
                base += 4
            elif age <= 3:
                base += 2
            elif age > 8:
                base = max(2, base - 2)
        return max(1, base)

    def _speaker_reliability(self, speaker_id: str | None) -> int:
        speaker = self._normalize_player_id(speaker_id)
        if not speaker:
            return 0
        score = 0
        if speaker in self.trusted_info_ids:
            score += 18
        if speaker in self.suspected_wolves:
            score -= 18
        for record in self.public_claim_records:
            if self._normalize_player_id(record.get("speaker_id")) != speaker:
                continue
            claim_type = str(record.get("claim_type") or "")
            status = str(record.get("status") or "pending")
            if claim_type in _ROLE_SELF_CLAIM_TYPES:
                if status == "supported":
                    score += 10
                elif status == "contradicted":
                    score -= 14
            elif claim_type in _POSITIVE_TARGET_CLAIM_TYPES.union(_NEGATIVE_TARGET_CLAIM_TYPES):
                if status == "supported":
                    score += 5
                elif status == "contradicted":
                    score -= 6
        return score

    def _claim_pressure_score(self, target_id: str | None) -> tuple[int, list[str]]:
        records = self._claim_records_for_target(target_id)
        if not records:
            return 0, []
        positive = 0
        negative = 0
        positive_sources: set[str] = set()
        negative_sources: set[str] = set()
        notes: list[str] = []
        for record in records:
            claim_type = str(record.get("claim_type") or "")
            weight = self._claim_record_weight(record)
            speaker = self._normalize_player_id(record.get("speaker_id"))
            status = str(record.get("status") or "pending")
            if claim_type in _POSITIVE_TARGET_CLAIM_TYPES:
                positive += weight
                positive_sources.add(speaker or "")
                if status == "supported":
                    positive += 2
                elif status == "contradicted":
                    positive = max(0, positive - 4)
            elif claim_type in _NEGATIVE_TARGET_CLAIM_TYPES:
                negative += weight
                negative_sources.add(speaker or "")
                if status == "supported":
                    negative += 2
                elif status == "contradicted":
                    negative = max(0, negative - 4)
            elif claim_type == "sheriff_transfer" and speaker in self.suspected_wolves:
                negative += max(2, weight // 2)
                notes.append("狼嫌转移")
        net = positive - negative
        if positive and negative:
            notes.append(f"正负{positive}/{negative}")
        elif negative:
            notes.append(f"压力-{negative}")
        elif positive:
            notes.append(f"支撑+{positive}")
        distinct_negative = len({speaker for speaker in negative_sources if speaker})
        if distinct_negative >= 2:
            net -= 3 * (distinct_negative - 1)
            notes.append(f"{distinct_negative}源指向")
        if self._normalize_player_id(target_id) in self.suspected_wolves:
            net -= 18
            notes.append("狼嫌")
        return net, notes

    def _strongest_public_contradictions(self, limit: int) -> list[str]:
        ranked: list[tuple[int, str]] = []
        seen: set[str] = set()
        for record in reversed(self.public_claim_records):
            status = str(record.get("status") or "pending")
            if status != "contradicted":
                continue
            speaker = self._normalize_player_id(record.get("speaker_id")) or "?"
            claim_type = str(record.get("claim_type") or "?")
            target = self._normalize_player_id(record.get("target_id")) or "?"
            seq = self._nonnegative_int(record.get("seq"), fallback=None)
            label = f"{speaker}:{claim_type}->{target}"
            if label in seen:
                continue
            seen.add(label)
            if seq is None:
                seq = 0
            ranked.append((seq, label))
        ranked.sort(key=lambda item: -item[0])
        return [label for _, label in ranked[: max(0, int(limit))]]

    def _guard_memory_snapshot(self, turn_packet: Mapping[str, Any]) -> dict[str, Any]:
        request = turn_packet.get("request") if isinstance(turn_packet, Mapping) else None
        public_state = turn_packet.get("public_state") if isinstance(turn_packet, Mapping) else None
        round_value = self._current_round_value(turn_packet)
        allowed_targets = (
            self._extract_allowed_target_ids(request, "guard_protect") if isinstance(request, Mapping) else []
        )
        ranked_targets = self._rank_guard_targets(allowed_targets, turn_packet) if allowed_targets else []
        top_targets: list[dict[str, Any]] = []
        for candidate in ranked_targets[:3]:
            score, reasons = self._score_guard_target(candidate, turn_packet)
            top_targets.append({"target_id": candidate, "score": score, "reasons": reasons[:3]})
        alive_ids = self._alive_id_list(public_state) if isinstance(public_state, Mapping) else []
        if not alive_ids:
            alive_ids = list(self.public_alive_ids)
        dead_ids = self._dead_id_set(public_state) if isinstance(public_state, Mapping) else set()
        if self.public_dead_ids:
            dead_ids.update(self.public_dead_ids)
        sheriff_id = self._extract_first_player_id(
            self._first_present(
                public_state if isinstance(public_state, Mapping) else {},
                "sheriff_id",
                "current_sheriff_id",
                "police_id",
                "badge_owner_id",
            )
        ) or self.current_public_sheriff_id
        return {
            "round": round_value,
            "alive_ids": self._compact_id_list(set(alive_ids), limit=8),
            "dead_ids": self._compact_id_list(dead_ids, limit=8),
            "sheriff_id": sheriff_id,
            "last_protected_id": self.last_protected_id,
            "last_protected_round": self.last_protected_round,
            "self_public_claim": "guard" if self.self_publicly_claimed_guard else "hidden",
            "self_protect_hint": self._should_self_protect(turn_packet, top_targets),
            "trusted_info_ids": self._compact_id_list(self.trusted_info_ids, limit=5),
            "suspected_wolves": self._compact_id_list(self.suspected_wolves, limit=5),
            "top_targets": top_targets,
            "strongest_public_contradictions": self._strongest_public_contradictions(limit=3),
        }

    def _should_self_protect(self, turn_packet: Mapping[str, Any], top_targets: list[dict[str, Any]]) -> bool:
        request = turn_packet.get("request") if isinstance(turn_packet, Mapping) else None
        allowed_targets = (
            self._extract_allowed_target_ids(request, "guard_protect") if isinstance(request, Mapping) else []
        )
        if self.player_id not in allowed_targets:
            return False
        public_state = turn_packet.get("public_state") if isinstance(turn_packet, Mapping) else None
        alive_ids = self._alive_id_list(public_state) if isinstance(public_state, Mapping) else []
        alive_count = len(alive_ids) if alive_ids else len(self.public_alive_ids)
        own_score, own_reasons = self._score_guard_target(self.player_id, turn_packet)
        best_other = max((item.get("score", own_score) for item in top_targets if item.get("target_id") != self.player_id), default=own_score)
        if alive_count and alive_count <= 4:
            return True
        if self.self_publicly_claimed_guard and alive_count and alive_count <= 5 and own_score >= best_other - 2:
            return True
        if alive_count and alive_count <= 5 and own_score >= best_other - 1:
            return True
        if alive_count and alive_count <= 6 and self.player_id in self.suspected_wolves:
            return own_score >= best_other - 1
        if any("自守" in reason for reason in own_reasons) and own_score >= best_other:
            return True
        return own_score >= best_other - 3

    def _compact_public_claims(self, *, max_players: int) -> dict[str, list[str]]:
        compact: dict[str, list[str]] = {}
        for record in reversed(self.public_claim_records):
            speaker = self._normalize_player_id(record.get("speaker_id"))
            if not speaker:
                continue
            claim_type = str(record.get("claim_type") or "")
            target = self._normalize_player_id(record.get("target_id"))
            status = str(record.get("status") or "pending")
            label = claim_type if not target else f"{claim_type}:{target}"
            if status != "pending":
                label = f"{label}[{status[0]}]"
            bucket = compact.setdefault(speaker, [])
            if label not in bucket:
                bucket.append(label)
            if len(bucket) > 4:
                bucket.pop(0)
            if len(compact) >= max_players and speaker not in compact:
                break
        return compact

    def _extract_allowed_target_ids(self, request: Mapping[str, Any] | None, kind: str) -> list[str]:
        if not isinstance(request, Mapping):
            return []
        allowed_actions = request.get("allowed_actions")
        if not isinstance(allowed_actions, list):
            return []
        allowed = next(
            (item for item in allowed_actions if isinstance(item, Mapping) and item.get("kind") == kind),
            None,
        )
        if not isinstance(allowed, Mapping):
            return []
        target_ids = allowed.get("target_ids")
        if not isinstance(target_ids, list):
            return []
        return [str(target_id) for target_id in target_ids if target_id is not None]

    def _pass_allowed(self, request: Mapping[str, Any] | None) -> bool:
        if not isinstance(request, Mapping):
            return False
        allowed_actions = request.get("allowed_actions")
        if not isinstance(allowed_actions, list):
            return False
        return any(isinstance(item, Mapping) and item.get("kind") == "pass" for item in allowed_actions)

    def _make_pass_action(self, request: Mapping[str, Any], action: Mapping[str, Any]) -> dict[str, Any]:
        pass_action = {"request_id": request["request_id"], "player_id": request["player_id"], "kind": "pass"}
        # 若引擎对 pass 也要求携带原字段，可保留其余字段；这里仅保留最小合法结构。
        del action
        return pass_action

    def _alive_id_list(self, public_state: Mapping[str, Any]) -> list[str]:
        candidates = [
            self._first_present(
                public_state,
                "alive_players",
                "alive_player_ids",
                "living_players",
                "players_alive",
            )
        ]
        for value in candidates:
            ids = self._collect_player_ids(value)
            if ids:
                return ids
        return []

    def _alive_id_set(self, public_state: Mapping[str, Any]) -> set[str]:
        return set(self._alive_id_list(public_state))

    def _dead_id_set(self, public_state: Mapping[str, Any]) -> set[str]:
        candidates = [
            self._first_present(
                public_state,
                "dead_players",
                "dead_player_ids",
                "executed_players",
                "graveyard",
            )
        ]
        result: set[str] = set()
        for value in candidates:
            result.update(self._collect_player_ids(value))
        return result

    def _collect_player_ids(self, value: Any) -> list[str]:
        ids: list[str] = []
        if value is None:
            return ids
        if isinstance(value, str):
            return self._extract_player_ids(value)
        if isinstance(value, Mapping):
            for key in (
                "player_id",
                "playerId",
                "id",
                "speaker_id",
                "speakerId",
                "target_id",
                "targetId",
                "owner_id",
                "ownerId",
                "from_player_id",
                "to_player_id",
            ):
                extracted = self._extract_first_player_id(value.get(key))
                if extracted:
                    ids.append(extracted)
            for nested in value.values():
                if isinstance(nested, (Mapping, list, str)):
                    ids.extend(self._collect_player_ids(nested))
            return self._dedupe_ids(ids)
        if isinstance(value, list):
            for item in value:
                ids.extend(self._collect_player_ids(item))
            return self._dedupe_ids(ids)
        return ids

    def _first_present(self, mapping: Mapping[str, Any], *keys: str) -> Any:
        for key in keys:
            if key in mapping and mapping.get(key) is not None:
                return mapping.get(key)
        return None

    def _first_text_present(self, mapping: Mapping[str, Any], *keys: str) -> str | None:
        value = self._first_present(mapping, *keys)
        return str(value) if isinstance(value, str) and value.strip() else None

    def _extract_first_player_id(self, value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            match = _PLAYER_ID_PATTERN.search(value)
            return match.group(0) if match else value.strip() or None
        if isinstance(value, list):
            for item in value:
                extracted = self._extract_first_player_id(item)
                if extracted:
                    return extracted
            return None
        if isinstance(value, Mapping):
            for nested_key in (
                "player_id",
                "playerId",
                "id",
                "target_id",
                "targetId",
                "speaker_id",
                "speakerId",
            ):
                extracted = self._extract_first_player_id(value.get(nested_key))
                if extracted:
                    return extracted
        return None

    def _extract_player_ids(self, text: str) -> list[str]:
        return self._dedupe_ids(_PLAYER_ID_PATTERN.findall(text))

    def _dedupe_ids(self, ids: list[str]) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for player_id in ids:
            normalized = self._normalize_player_id(player_id)
            if normalized and normalized not in seen:
                seen.add(normalized)
                ordered.append(normalized)
        return ordered

    def _compact_id_list(self, ids: set[str], *, limit: int) -> list[str]:
        ordered = sorted(self._normalize_player_id(player_id) for player_id in ids)
        compact: list[str] = []
        for player_id in ordered:
            if player_id and player_id not in compact:
                compact.append(player_id)
            if len(compact) >= limit:
                break
        return compact

    def _normalize_player_id(self, player_id: Any) -> str | None:
        if player_id is None:
            return None
        text = str(player_id).strip()
        return text or None

    def _current_round_value(self, turn_packet: Mapping[str, Any]) -> Any:
        game = turn_packet.get("game") if isinstance(turn_packet, Mapping) else None
        if isinstance(game, Mapping):
            return game.get("round")
        return None

    def _should_track_text(self, text: str) -> bool:
        if _PLAYER_ID_PATTERN.search(text):
            return True
        tracked_keywords = (
            "预言家",
            "查杀",
            "金水",
            "守卫",
            "女巫",
            "猎人",
            "警长",
            "警徽",
            "移交",
            "对跳",
            "狼人",
            "平民",
        )
        return any(keyword in text for keyword in tracked_keywords)

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

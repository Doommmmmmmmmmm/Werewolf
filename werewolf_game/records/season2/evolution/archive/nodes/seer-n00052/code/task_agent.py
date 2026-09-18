"""预言家 Task-Agent。

每次行动从一个新的模型会话开始；但 Task-Agent 实例会维护轻量的 seer 记忆，
用于跨回合保存查验、候选人、对跳与票型摘要。
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import inspect
import json
import re
from typing import Any

from ..core.errors import ModelClientError, RuleViolationError
from .llm.coordinator import ModelRequestCoordinator
from ..prompts import RoleProfile, render_prompt


_CHINESE_CHARACTER = re.compile(r"[\u3400-\u9fff]")
_PLAYER_ID_PATTERN = re.compile(r"\bp\d+\b", re.IGNORECASE)
_SEAT_REFERENCE_PATTERN = re.compile(r"(?:第\s*)?(?P<num>\d{1,2})\s*(?:号|位|席|號|名)\b")
_CURRENT_DIALOGUE_TEXT_LIMIT = 120
_CURRENT_DIALOGUE_ITEM_LIMIT = 6
_SUMMARY_TEXT_LIMIT = 160
_SUMMARY_LIST_LIMIT = 6

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


@dataclass
class SeerMemory:
    game_id: str = ""
    round_no: str = ""
    phase: str = ""
    alive: list[str] = field(default_factory=list)
    dead: list[str] = field(default_factory=list)
    sheriff_id: str = ""
    sheriff_candidates: list[str] = field(default_factory=list)
    inspected: list[dict[str, str]] = field(default_factory=list)
    declared_claims: list[dict[str, str]] = field(default_factory=list)
    challenged_players: list[str] = field(default_factory=list)
    seat_to_player_id: dict[str, str] = field(default_factory=dict)
    player_to_seat: dict[str, str] = field(default_factory=dict)
    recent_vote_summary: str = ""
    recent_vote_detail: dict[str, Any] = field(default_factory=dict)
    dialogue_digest: list[str] = field(default_factory=list)
    claim_timeline: list[dict[str, str]] = field(default_factory=list)
    vote_timeline: list[dict[str, Any]] = field(default_factory=list)
    pressure_targets: list[dict[str, str]] = field(default_factory=list)
    self_risk_mode: str = "normal"
    inspection_map: dict[str, str] = field(default_factory=dict)

    def reset(self, game_id: str = "") -> None:
        self.game_id = game_id
        self.round_no = ""
        self.phase = ""
        self.alive.clear()
        self.dead.clear()
        self.sheriff_id = ""
        self.sheriff_candidates.clear()
        self.inspected.clear()
        self.declared_claims.clear()
        self.challenged_players.clear()
        self.seat_to_player_id.clear()
        self.player_to_seat.clear()
        self.recent_vote_summary = ""
        self.recent_vote_detail.clear()
        self.dialogue_digest.clear()
        self.claim_timeline.clear()
        self.vote_timeline.clear()
        self.pressure_targets.clear()
        self.self_risk_mode = "normal"
        self.inspection_map.clear()


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
        self._seer_memory = SeerMemory()

    async def observe(self, sync_packet: dict[str, Any]) -> None:
        """接收公开同步并更新轻量 seer 记忆。"""

        self._ingest_packet(sync_packet, source="observe")

    async def decide(self, turn_packet: dict[str, Any]) -> dict[str, Any]:
        if turn_packet["self"]["player_id"] != self.player_id:
            raise ValueError("行动包被发送给了错误玩家")
        private = turn_packet["private_information"]
        if str(private.get("role")) != self.profile.role:
            raise ValueError("Task-Agent 的角色档案与行动包身份不一致")

        self._ingest_packet(turn_packet, source="decide")
        mode = self._decide_mode(turn_packet)
        seer_brief = self._build_seer_brief(turn_packet, mode)
        system = self._system_prompt(private)
        prompt = {
            "game": self._compact_value(turn_packet.get("game"), max_depth=1),
            "self": self._compact_value(turn_packet.get("self"), max_depth=1),
            "request": self._compact_request(turn_packet.get("request") or {}),
            "seer_memory": self._compact_seer_memory(),
            "seer_brief": seer_brief,
            "public_state": self._compact_value(turn_packet.get("public_state"), max_depth=2),
            "private_information": self._compact_private_information(private),
            "history_policy": {
                "default_context": "seer_memory_plus_current_round_summary",
                "current_round_dialogue_tool": CURRENT_ROUND_DIALOGUE_TOOL_NAME,
                "max_tool_calls": self.max_tool_calls_per_decision,
                "max_tool_result_tokens": self.max_tool_result_tokens,
            },
        }

        tool_context = turn_packet.get("tool_context") or {}
        current_dialogue = tool_context.get("current_round_dialogue") or []
        current_dialogue_excerpt = self._summarize_dialogue(current_dialogue)
        if current_dialogue_excerpt:
            prompt["current_round_dialogue_excerpt"] = current_dialogue_excerpt

        def execute_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
            del arguments
            if name != CURRENT_ROUND_DIALOGUE_TOOL_NAME:
                return {"error": f"不支持的工具：{name}"}
            return self._current_round_dialogue_tool_result(turn_packet, current_dialogue)

        feedback = ""
        for _attempt in range(self.max_decision_retries + 1):
            # 一次玩家决策的工具预算不能因纠错重试而被放大。首个模型尝试可读取
            # 当前轮对话；所有后续纠错尝试都是没有工具的新会话。
            allow_tool = _attempt == 0 and self.max_tool_calls_per_decision > 0
            user_content: dict[str, Any] = {
                "instruction": self._turn_instruction(
                    turn_packet["request"], feedback, mode=mode, seer_brief=seer_brief
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
    def _turn_instruction(
        request: Mapping[str, Any],
        feedback: str,
        *,
        mode: str,
        seer_brief: dict[str, Any],
    ) -> str:
        pieces = [render_action_contract(request)]
        facts = seer_brief.get("facts") or []
        pressure_points = seer_brief.get("pressure_points") or []
        pressure_signals = seer_brief.get("pressure_signals") or []
        self_risk = str(seer_brief.get("self_risk") or "")
        preferred_targets = seer_brief.get("preferred_targets") or []
        target_reasons = seer_brief.get("target_reasons") or []

        def fixed_speech_skeleton(label: str) -> str:
            return (
                f"{label}必须按固定骨架输出：1个事实→1个主要矛盾→1个明确票向/自保位；"
                "不要并列多个目标，不要解释整套推理链。"
            )

        if mode == "night_inspect":
            pieces.append(
                "夜验只选一个目标，优先未解决对跳链、公共压力位、票型关键位和警长链关键位；"
                "避开已验过、自己和信息增益已耗尽的人。"
            )
            if preferred_targets:
                pieces.append("建议目标：" + "、".join(preferred_targets[:_SUMMARY_LIST_LIMIT]))
            if target_reasons:
                pieces.append("目标理由：" + "；".join(target_reasons[:_SUMMARY_LIST_LIMIT]))
        elif mode == "sheriff_election_speech":
            pieces.append(fixed_speech_skeleton("竞选警长发言"))
            pieces.append("优先说明已知查验与对跳时间线，再给出带队票向。")
        elif mode == "day_speech":
            pieces.append(fixed_speech_skeleton("白天发言"))
            pieces.append("优先压住最危险的矛盾点，再落到单点投票方向。")
        elif mode == "vote":
            pieces.append("投票必须单点落票：查验事实→主要矛盾→最终票向，只输出一个最值得投的对象。")
        elif mode == "self_risk":
            pieces.append("自保短模板：1个事实→1个压迫点→1个自保位/跟票位，句子短、目标单一。")
        else:
            pieces.append("保持 seer 视角，围绕查验、对跳和票型输出，不要套通用狼人杀模板。")

        if self_risk and self_risk != "常规风险：优先保留信息优势，不要过度展开。":
            pieces.append("高压提示：" + self_risk)
        if facts:
            pieces.append("事实：" + "；".join(facts[:_SUMMARY_LIST_LIMIT]))
        if pressure_points:
            pieces.append("矛盾：" + "；".join(pressure_points[:_SUMMARY_LIST_LIMIT]))
        if pressure_signals:
            pieces.append("压力信号：" + "；".join(pressure_signals[:_SUMMARY_LIST_LIMIT]))
        if seer_brief.get("pressure_targets"):
            pieces.append("压力位：" + "；".join(seer_brief["pressure_targets"]))
        if feedback:
            pieces.append(f"上一次输出未通过校验：{feedback}")
        pieces.append("最终回答只能是符合契约的 JSON 对象。")
        return "\n".join(pieces)

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

    def _ingest_packet(self, packet: Mapping[str, Any], *, source: str) -> None:
        del source
        if not isinstance(packet, Mapping):
            return
        game_id = self._extract_game_id(packet)
        if game_id and game_id != self._seer_memory.game_id:
            self._seer_memory.reset(game_id)
        elif game_id and not self._seer_memory.game_id:
            self._seer_memory.game_id = game_id

        self._update_seat_aliases_from_packet(packet)
        self._normalize_seer_memory_aliases()

        self._seer_memory.round_no = self._stringify(
            self._first_present(
                packet,
                (
                    ("game", "round"),
                    ("game", "day"),
                    ("round",),
                    ("day",),
                    ("public_state", "round"),
                    ("public_state", "day"),
                ),
            )
        )
        self._seer_memory.phase = self._stringify(self._extract_phase(packet))

        alive = self._extract_player_ids(
            self._first_present(
                packet,
                (
                    ("public_state", "alive"),
                    ("public_state", "alive_players"),
                    ("public_state", "survivors"),
                    ("game", "alive"),
                    ("alive",),
                ),
            )
        )
        dead = self._extract_player_ids(
            self._first_present(
                packet,
                (
                    ("public_state", "dead"),
                    ("public_state", "dead_players"),
                    ("public_state", "eliminated"),
                    ("game", "dead"),
                    ("dead",),
                ),
            )
        )
        if alive:
            self._seer_memory.alive = alive
        if dead:
            self._seer_memory.dead = dead

        sheriff = self._extract_single_player_id(
            self._first_present(
                packet,
                (
                    ("public_state", "sheriff_id"),
                    ("public_state", "sheriff"),
                    ("public_state", "captain"),
                    ("game", "sheriff_id"),
                    ("game", "sheriff"),
                    ("sheriff_id",),
                    ("sheriff",),
                    ("captain",),
                ),
            )
        )
        if sheriff:
            self._seer_memory.sheriff_id = sheriff

        sheriff_candidates = self._extract_player_ids(
            self._first_present(
                packet,
                (
                    ("public_state", "sheriff_candidates"),
                    ("public_state", "candidates"),
                    ("public_state", "election_candidates"),
                    ("game", "sheriff_candidates"),
                    ("sheriff_candidates",),
                    ("candidates",),
                ),
            )
        )
        if sheriff_candidates:
            self._seer_memory.sheriff_candidates = sheriff_candidates

        vote_summary, vote_detail = self._summarize_vote_info(packet)
        if vote_summary:
            self._seer_memory.recent_vote_summary = vote_summary
        if vote_summary or vote_detail:
            self._append_vote_timeline(vote_summary, vote_detail)
        if vote_detail:
            self._seer_memory.recent_vote_detail = vote_detail

        dialogue_digest = self._extract_dialogue_digest(packet)
        if dialogue_digest:
            self._seer_memory.dialogue_digest = dialogue_digest
            self._update_claims_from_dialogue(dialogue_digest)

        self._update_claims_and_challenges_from_packet(packet)
        self._update_inspection_records(packet)
        self._refresh_pressure_state()

    def _extract_game_id(self, packet: Mapping[str, Any]) -> str:
        for path in (("game", "game_id"), ("game", "id"), ("game_id",), ("match_id",), ("session_id",)):
            value = self._first_present(packet, (path,))
            game_id = self._stringify(value)
            if game_id:
                return game_id
        return ""

    def _extract_phase(self, packet: Mapping[str, Any]) -> str:
        for path in (
            ("game", "public_phase"),
            ("game", "phase"),
            ("public_state", "phase"),
            ("public_state", "public_phase"),
            ("request", "phase"),
            ("phase",),
        ):
            value = self._first_present(packet, (path,))
            phase = self._stringify(value)
            if phase:
                return phase
        return ""

    def _first_present(self, packet: Mapping[str, Any], paths: Sequence[Sequence[str]]) -> Any:
        for path in paths:
            current: Any = packet
            found = True
            for key in path:
                if not isinstance(current, Mapping) or key not in current:
                    found = False
                    break
                current = current[key]
            if found and current not in (None, ""):
                return current
        return None

    def _extract_player_ids(self, value: Any) -> list[str]:
        result: list[str] = []
        if value is None:
            return result
        if isinstance(value, str):
            result.extend(_PLAYER_ID_PATTERN.findall(value))
            result.extend(self._resolved_player_ids_from_text(value))
            if result:
                return self._unique_stable(result)
            return [value] if value.strip() else []
        if isinstance(value, Mapping):
            for key in ("player_id", "id", "target_id", "speaker", "name", "player", "target", "seat_player_id"):
                nested = value.get(key)
                if isinstance(nested, str) and nested:
                    result.append(nested)
            for seat, player_id in self._seat_player_pairs_from_mapping(value):
                self._remember_seat_alias(seat, player_id)
                result.append(player_id)
            for nested in value.values():
                result.extend(self._extract_player_ids(nested))
            return self._unique_stable(result)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            for item in value:
                if isinstance(item, Mapping):
                    result.extend(self._extract_player_ids(item))
                elif isinstance(item, str):
                    result.extend(_PLAYER_ID_PATTERN.findall(item))
                    result.extend(self._resolved_player_ids_from_text(item))
                    if not _PLAYER_ID_PATTERN.search(item) and not self._resolved_player_ids_from_text(item):
                        result.append(item)
                else:
                    text = self._stringify(item)
                    if text:
                        ids = _PLAYER_ID_PATTERN.findall(text)
                        result.extend(ids or self._resolved_player_ids_from_text(text) or [text])
            return self._unique_stable(result)
        text = self._stringify(value)
        if text:
            ids = _PLAYER_ID_PATTERN.findall(text)
            resolved = self._resolved_player_ids_from_text(text)
            return self._unique_stable(ids or resolved or [text])
        return result

    def _extract_single_player_id(self, value: Any) -> str:
        ids = self._extract_player_ids(value)
        return ids[0] if ids else ""

    def _seat_player_pairs_from_mapping(self, value: Mapping[str, Any]) -> list[tuple[str, str]]:
        pairs: list[tuple[str, str]] = []
        direct_seat = self._extract_seat_label(
            value.get("seat")
            or value.get("position")
            or value.get("seat_no")
            or value.get("seat_id")
        )
        direct_player = self._extract_single_player_id(
            value.get("player_id")
            or value.get("id")
            or value.get("player")
            or value.get("target_id")
            or value.get("speaker")
            or value.get("name")
        )
        if direct_seat and direct_player:
            pairs.append((direct_seat, direct_player))
        for key, nested in value.items():
            key_seat = self._extract_seat_label(key)
            if key_seat:
                nested_player = self._extract_single_player_id(nested)
                if nested_player:
                    pairs.append((key_seat, nested_player))
        return self._unique_pair_list(pairs)

    def _extract_seat_label(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            match = _SEAT_REFERENCE_PATTERN.search(value)
            if match:
                return match.group("num")
            return ""
        if isinstance(value, Mapping):
            for key in ("seat", "position", "seat_no", "seat_id"):
                label = self._extract_seat_label(value.get(key))
                if label:
                    return label
            for nested in value.values():
                label = self._extract_seat_label(nested)
                if label:
                    return label
            return ""
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            for item in value:
                label = self._extract_seat_label(item)
                if label:
                    return label
            return ""
        text = self._stringify(value)
        match = _SEAT_REFERENCE_PATTERN.search(text)
        return match.group("num") if match else ""

    def _resolved_player_ids_from_text(self, text: str) -> list[str]:
        result: list[str] = []
        if not text:
            return result
        for seat in self._extract_seat_labels(text):
            player_id = self._seer_memory.seat_to_player_id.get(seat)
            if player_id:
                result.append(player_id)
                continue
            fallback = self._seat_label_to_player_id(seat)
            if fallback:
                result.append(fallback)
        return self._unique_stable(result)

    def _extract_seat_labels(self, value: Any) -> list[str]:
        labels: list[str] = []
        if value is None:
            return labels
        if isinstance(value, str):
            labels.extend(match.group("num") for match in _SEAT_REFERENCE_PATTERN.finditer(value))
            return self._unique_stable(labels)
        if isinstance(value, Mapping):
            for key in ("seat", "position", "seat_no", "seat_id"):
                labels.extend(self._extract_seat_labels(value.get(key)))
            for nested in value.values():
                labels.extend(self._extract_seat_labels(nested))
            return self._unique_stable(labels)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            for item in value:
                labels.extend(self._extract_seat_labels(item))
            return self._unique_stable(labels)
        text = self._stringify(value)
        labels.extend(match.group("num") for match in _SEAT_REFERENCE_PATTERN.finditer(text))
        return self._unique_stable(labels)

    def _seat_label_to_player_id(self, seat_label: str) -> str:
        seat_label = self._stringify(seat_label)
        if not seat_label:
            return ""
        if seat_label in self._seer_memory.seat_to_player_id:
            return self._seer_memory.seat_to_player_id[seat_label]
        if seat_label.isdigit():
            return f"p{seat_label}"
        return ""

    def _seat_reference_variants(self, seat_label: str) -> list[str]:
        seat_label = self._stringify(seat_label)
        if not seat_label:
            return []
        variants = [seat_label]
        if seat_label.isdigit():
            variants.extend(
                [
                    f"{seat_label}号",
                    f"{seat_label}位",
                    f"第{seat_label}号",
                    f"第{seat_label}位",
                    f"{seat_label}席",
                    f"{seat_label}名",
                    f"{seat_label}號",
                ]
            )
        return self._unique_stable(variants)

    def _reference_aliases(self, reference: str) -> list[str]:
        reference = self._stringify(reference)
        if not reference:
            return []
        aliases = [reference]
        if reference in self._seer_memory.player_to_seat:
            aliases.extend(self._seat_reference_variants(self._seer_memory.player_to_seat[reference]))
        match = _PLAYER_ID_PATTERN.fullmatch(reference)
        if match:
            seat_label = match.group(0)[1:]
            aliases.extend(self._seat_reference_variants(seat_label))
        elif reference.isdigit():
            aliases.extend(self._seat_reference_variants(reference))
            aliases.append(f"p{reference}")
        return self._unique_stable(aliases)

    def _text_mentions_reference(self, text: str, reference: str) -> bool:
        text = self._stringify(text)
        if not text or not reference:
            return False
        for alias in self._reference_aliases(reference):
            if alias.startswith("p") and alias[1:].isdigit():
                if re.search(rf"(?<![A-Za-z0-9_]){re.escape(alias)}(?![A-Za-z0-9_])", text):
                    return True
            elif alias in text:
                return True
        return False

    def _normalize_alias_text(self, text: str) -> str:
        text = self._stringify(text)
        if not text or not self._seer_memory.seat_to_player_id:
            return text
        normalized = text
        for seat_label, player_id in sorted(
            self._seer_memory.seat_to_player_id.items(), key=lambda item: -len(item[0])
        ):
            for variant in self._seat_reference_variants(seat_label):
                normalized = re.sub(re.escape(variant), player_id, normalized)
        return normalized

    def _remember_seat_alias(self, seat_label: str, player_id: str) -> None:
        seat_label = self._stringify(seat_label)
        player_id = self._stringify(player_id)
        if not seat_label or not player_id:
            return
        existing = self._seer_memory.seat_to_player_id.get(seat_label)
        if existing and existing != player_id:
            return
        self._seer_memory.seat_to_player_id[seat_label] = player_id
        self._seer_memory.player_to_seat[player_id] = seat_label

    def _update_seat_aliases_from_packet(self, packet: Mapping[str, Any]) -> None:
        if not isinstance(packet, Mapping):
            return
        visited: set[int] = set()
        stack: list[Any] = [packet]
        while stack:
            value = stack.pop()
            if isinstance(value, Mapping):
                value_id = id(value)
                if value_id in visited:
                    continue
                visited.add(value_id)
                for seat_label, player_id in self._seat_player_pairs_from_mapping(value):
                    self._remember_seat_alias(seat_label, player_id)
                for nested in value.values():
                    if isinstance(nested, (Mapping, Sequence)) and not isinstance(nested, (str, bytes, bytearray)):
                        stack.append(nested)
            elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
                for item in value:
                    if isinstance(item, (Mapping, Sequence)) and not isinstance(item, (str, bytes, bytearray)):
                        stack.append(item)

    def _normalize_seer_memory_aliases(self) -> None:
        if self._seer_memory.sheriff_id:
            self._seer_memory.sheriff_id = self._normalize_reference(self._seer_memory.sheriff_id)
        self._seer_memory.alive = [self._normalize_reference(item) for item in self._seer_memory.alive]
        self._seer_memory.dead = [self._normalize_reference(item) for item in self._seer_memory.dead]
        self._seer_memory.sheriff_candidates = [
            self._normalize_reference(item) for item in self._seer_memory.sheriff_candidates
        ]
        self._seer_memory.challenged_players = [
            self._normalize_reference(item) for item in self._seer_memory.challenged_players
        ]
        self._seer_memory.inspected = [
            {
                "round": self._stringify(item.get("round")),
                "target": self._normalize_reference(item.get("target")),
                "result": self._normalize_inspection_result(item.get("result")),
            }
            for item in self._seer_memory.inspected
            if isinstance(item, Mapping)
        ]
        self._seer_memory.inspection_map = {
            self._normalize_reference(key): self._normalize_inspection_result(value)
            for key, value in self._seer_memory.inspection_map.items()
            if self._normalize_reference(key)
        }
        self._seer_memory.declared_claims = [
            {
                "speaker": self._normalize_reference(item.get("speaker")),
                "claim": self._normalize_alias_text(self._stringify(item.get("claim"))),
            }
            for item in self._seer_memory.declared_claims
            if isinstance(item, Mapping)
        ]
        self._seer_memory.declared_claims = self._dedupe_dict_list(self._seer_memory.declared_claims)
        self._seer_memory.dialogue_digest = [self._normalize_alias_text(line) for line in self._seer_memory.dialogue_digest]
        self._seer_memory.recent_vote_summary = self._normalize_alias_text(self._seer_memory.recent_vote_summary)
        self._seer_memory.claim_timeline = [self._normalize_claim_timeline(item) for item in self._seer_memory.claim_timeline if isinstance(item, Mapping)]
        self._seer_memory.vote_timeline = [self._normalize_vote_timeline(item) for item in self._seer_memory.vote_timeline if isinstance(item, Mapping)]
        self._seer_memory.pressure_targets = [self._normalize_pressure_target(item) for item in self._seer_memory.pressure_targets if isinstance(item, Mapping)]
        self._seer_memory.self_risk_mode = self._normalize_self_risk_mode(self._seer_memory.self_risk_mode)
        if self._seer_memory.recent_vote_detail:
            self._seer_memory.recent_vote_detail = self._normalize_vote_detail(self._seer_memory.recent_vote_detail)

    def _normalize_vote_detail(self, value: Any) -> Any:
        if isinstance(value, Mapping):
            normalized: dict[str, Any] = {}
            for key, nested in value.items():
                normalized[self._stringify(key)] = self._normalize_vote_detail(nested)
            return normalized
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return [self._normalize_vote_detail(item) for item in value]
        if isinstance(value, str):
            return self._normalize_alias_text(value)
        return value

    def _normalize_inspection_result(self, value: Any) -> str:
        text = self._inspection_result_text(value)
        return text if text else ""

    def _normalize_claim_timeline(self, item: Mapping[str, Any]) -> dict[str, str]:
        return {
            "round": self._stringify(item.get("round")),
            "phase": self._stringify(item.get("phase")),
            "order": self._stringify(item.get("order")),
            "speaker": self._normalize_reference(item.get("speaker")),
            "claim": self._truncate_text(self._normalize_alias_text(self._stringify(item.get("claim"))), _SUMMARY_TEXT_LIMIT),
        }

    def _normalize_vote_timeline(self, item: Mapping[str, Any]) -> dict[str, Any]:
        normalized = {
            "round": self._stringify(item.get("round")),
            "phase": self._stringify(item.get("phase")),
            "summary": self._truncate_text(self._normalize_alias_text(self._stringify(item.get("summary"))), _SUMMARY_TEXT_LIMIT),
        }
        detail = item.get("detail")
        if detail is not None:
            normalized["detail"] = self._normalize_vote_detail(detail)
        return normalized

    def _normalize_pressure_target(self, item: Mapping[str, Any]) -> dict[str, str]:
        return {
            "round": self._stringify(item.get("round")),
            "target": self._normalize_reference(item.get("target")),
            "reason": self._truncate_text(self._normalize_alias_text(self._stringify(item.get("reason"))), _SUMMARY_TEXT_LIMIT),
        }

    def _normalize_self_risk_mode(self, value: Any) -> str:
        text = self._stringify(value).lower()
        if any(token in text for token in ("high", "risk", "self", "自保")):
            return "high"
        if any(token in text for token in ("mid", "medium", "中等")):
            return "medium"
        return "normal"

    def _normalize_reference(self, value: Any) -> str:
        text = self._extract_single_player_id(value)
        if text:
            return text
        raw = self._stringify(value)
        return self._normalize_alias_text(raw)

    def _unique_pair_list(self, values: Sequence[tuple[str, str]]) -> list[tuple[str, str]]:
        seen: set[tuple[str, str]] = set()
        ordered: list[tuple[str, str]] = []
        for seat_label, player_id in values:
            key = (self._stringify(seat_label), self._stringify(player_id))
            if not key[0] or not key[1] or key in seen:
                continue
            seen.add(key)
            ordered.append((key[0], key[1]))
        return ordered

    def _mentions_player_with_alias(self, text: str, target_id: str) -> bool:
        return self._text_mentions_reference(text, target_id)

    def _summarize_vote_info(self, packet: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
        vote_value = self._first_present(
            packet,
            (
                ("public_state", "votes"),
                ("public_state", "vote"),
                ("public_state", "vote_result"),
                ("public_state", "day_vote"),
                ("public_state", "election_vote"),
                ("game", "votes"),
                ("votes",),
                ("vote_result",),
            ),
        )
        if vote_value is None:
            return "", {}
        detail = self._compact_value(vote_value, max_depth=2)
        summary = self._vote_summary_text(vote_value)
        return summary, detail if isinstance(detail, dict) else {"vote": detail}

    def _append_claim_timeline(self, speaker: str, claim: str, source: str) -> None:
        if not speaker and not claim:
            return
        entry = {
            "round": self._seer_memory.round_no,
            "phase": self._seer_memory.phase,
            "order": str(len(self._seer_memory.claim_timeline) + 1),
            "speaker": speaker,
            "claim": claim,
            "source": source,
        }
        self._seer_memory.claim_timeline.append(entry)

    def _append_vote_timeline(self, summary: str, detail: dict[str, Any]) -> None:
        entry = {
            "round": self._seer_memory.round_no,
            "phase": self._seer_memory.phase,
            "summary": summary,
            "detail": detail,
        }
        self._seer_memory.vote_timeline.append(entry)

    def _append_pressure_target(self, target: str, reason: str) -> None:
        if not target and not reason:
            return
        entry = {
            "round": self._seer_memory.round_no,
            "target": target,
            "reason": reason,
        }
        self._seer_memory.pressure_targets.append(entry)

    def _refresh_pressure_state(self) -> None:
        current = "normal"
        if self.player_id in self._seer_memory.challenged_players:
            current = "high"
        elif len(self._recent_pressure_signals()) >= 3:
            current = "high"
        elif self._seer_memory.challenged_players or self._seer_memory.pressure_targets:
            current = "medium"
        self._seer_memory.self_risk_mode = current

    def _vote_summary_text(self, value: Any) -> str:
        if isinstance(value, Mapping):
            target = self._extract_single_player_id(
                value.get("target") or value.get("target_id") or value.get("lynch") or value.get("eliminate")
            )
            counts = value.get("counts") or value.get("votes") or value.get("detail")
            counts_text = self._vote_summary_text(counts)
            if target and counts_text:
                return f"{target} | {counts_text}"
            if target:
                return target
            if counts_text:
                return counts_text
            items = []
            for key, nested in value.items():
                if key in {"round", "day", "phase"}:
                    continue
                ids = self._extract_player_ids(nested)
                if ids:
                    items.append(f"{key}:{'、'.join(ids[:3])}")
                else:
                    text = self._truncate_text(self._stringify(nested), 60)
                    if text:
                        items.append(f"{key}:{text}")
            return "；".join(items[:_SUMMARY_LIST_LIMIT])
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            items = []
            for item in list(value)[:_SUMMARY_LIST_LIMIT]:
                ids = self._extract_player_ids(item)
                if ids:
                    items.append("、".join(ids[:3]))
                else:
                    items.append(self._truncate_text(self._stringify(item), 60))
            return "；".join(item for item in items if item)
        return self._truncate_text(self._stringify(value), 80)

    def _extract_dialogue_digest(self, packet: Mapping[str, Any]) -> list[str]:
        dialogue = self._first_present(
            packet,
            (
                ("tool_context", "current_round_dialogue"),
                ("public_state", "current_round_dialogue"),
                ("public_state", "dialogue"),
                ("dialogue",),
                ("current_round_dialogue",),
            ),
        )
        return self._summarize_dialogue(dialogue)

    def _summarize_dialogue(self, dialogue: Any) -> list[str]:
        if not dialogue:
            return []
        items: list[str] = []
        if isinstance(dialogue, Mapping):
            dialogue = [dialogue]
        if not isinstance(dialogue, Sequence) or isinstance(dialogue, (str, bytes, bytearray)):
            text = self._truncate_text(self._stringify(dialogue), _CURRENT_DIALOGUE_TEXT_LIMIT)
            return [text] if text else []
        for item in dialogue:
            if len(items) >= _CURRENT_DIALOGUE_ITEM_LIMIT:
                break
            if isinstance(item, Mapping):
                speaker = self._normalize_reference(
                    item.get("speaker") or item.get("player_id") or item.get("id") or item.get("from")
                )
                text = self._normalize_alias_text(
                    self._stringify(
                        item.get("text")
                        or item.get("content")
                        or item.get("message")
                        or item.get("utterance")
                        or item
                    )
                )
            else:
                speaker = ""
                text = self._normalize_alias_text(self._stringify(item))
            text = self._truncate_text(text.strip(), _CURRENT_DIALOGUE_TEXT_LIMIT)
            if not text:
                continue
            if speaker:
                items.append(f"{speaker}:{text}")
            else:
                items.append(text)
        return items

    def _current_round_dialogue_tool_result(
        self, turn_packet: Mapping[str, Any], current_dialogue: Any
    ) -> dict[str, Any]:
        return {
            "round": self._stringify(self._first_present(turn_packet, (("game", "round"), ("round",)))) ,
            "phase": self._stringify(self._extract_phase(turn_packet)),
            "count": len(current_dialogue) if isinstance(current_dialogue, Sequence) and not isinstance(current_dialogue, (str, bytes, bytearray)) else (1 if current_dialogue else 0),
            "summary": self._summarize_dialogue(current_dialogue),
        }

    def _update_claims_from_dialogue(self, dialogue_digest: list[str]) -> None:
        for line in dialogue_digest:
            if not line:
                continue
            speaker = ""
            text = line
            if ":" in line:
                speaker, text = line.split(":", 1)
                speaker = self._normalize_reference(speaker.strip())
                text = self._normalize_alias_text(text.strip())
            claim_keywords = ("我是", "我跳", "我真", "预言家", "对跳", "查验", "警长", "竞选", "验了", "验过")
            if any(keyword in text for keyword in claim_keywords):
                entry = {
                    "speaker": speaker or self._speaker_from_text(text),
                    "claim": self._truncate_text(self._normalize_alias_text(text), _SUMMARY_TEXT_LIMIT),
                }
                if entry["speaker"] or entry["claim"]:
                    self._seer_memory.declared_claims.append(entry)
                    self._append_claim_timeline(entry["speaker"], entry["claim"], "dialogue")
            if any(keyword in text for keyword in ("对跳", "质疑", "假", "狼", "冲票", "先出", "出他", "票他")):
                ids = self._extract_player_ids(text)
                if speaker:
                    ids.append(speaker)
                normalized_ids = [self._normalize_reference(pid) for pid in ids]
                self._seer_memory.challenged_players = self._unique_stable(
                    list(self._seer_memory.challenged_players) + normalized_ids
                )
                for pid in normalized_ids:
                    self._append_pressure_target(pid, self._truncate_text(self._normalize_alias_text(text), _SUMMARY_TEXT_LIMIT))
        self._seer_memory.declared_claims = self._dedupe_dict_list(self._seer_memory.declared_claims)
        self._seer_memory.challenged_players = [self._normalize_reference(item) for item in self._seer_memory.challenged_players]
        self._seer_memory.challenged_players = self._unique_stable(self._seer_memory.challenged_players)

    def _update_claims_and_challenges_from_packet(self, packet: Mapping[str, Any]) -> None:
        claims = self._first_present(
            packet,
            (
                ("public_state", "claims"),
                ("public_state", "roles_claimed"),
                ("claims",),
                ("roles_claimed",),
            ),
        )
        if isinstance(claims, Sequence) and not isinstance(claims, (str, bytes, bytearray)):
            for item in claims:
                if isinstance(item, Mapping):
                    speaker = self._normalize_reference(
                        item.get("speaker") or item.get("player_id") or item.get("id")
                    )
                    claim = self._truncate_text(
                        self._normalize_alias_text(
                            self._stringify(
                                item.get("claim")
                                or item.get("role")
                                or item.get("text")
                                or item.get("content")
                                or item
                            )
                        ),
                        _SUMMARY_TEXT_LIMIT,
                    )
                    if speaker or claim:
                        self._seer_memory.declared_claims.append({"speaker": speaker, "claim": claim})
                        self._append_claim_timeline(speaker, claim, "packet")
        challenged = self._first_present(
            packet,
            (
                ("public_state", "challenged_players"),
                ("public_state", "suspects"),
                ("public_state", "challenge_targets"),
                ("challenged_players",),
                ("suspects",),
            ),
        )
        if challenged is not None:
            ids = [self._normalize_reference(pid) for pid in self._extract_player_ids(challenged)]
            if ids:
                self._seer_memory.challenged_players = self._unique_stable(
                    list(self._seer_memory.challenged_players) + ids
                )
                for pid in ids:
                    self._append_pressure_target(pid, "公开被质疑/对跳")
        self._seer_memory.declared_claims = self._dedupe_dict_list(self._seer_memory.declared_claims)
        self._seer_memory.challenged_players = self._unique_stable(self._seer_memory.challenged_players)

    def _update_inspection_records(self, packet: Mapping[str, Any]) -> None:
        private = self._first_present(
            packet,
            (
                ("private_information",),
                ("private",),
                ("self", "private_information"),
            ),
        )
        if not isinstance(private, Mapping):
            return
        target = self._normalize_reference(
            self._first_present(
                private,
                (
                    ("inspect_target",),
                    ("target_id",),
                    ("target",),
                    ("checked_player",),
                    ("seer_target",),
                ),
            )
        )
        result_value = self._first_present(
            private,
            (
                ("inspect_result",),
                ("check_result",),
                ("seer_result",),
                ("night_result",),
                ("result",),
            ),
        )
        if not target and not result_value:
            return
        result_text = self._inspection_result_text(result_value)
        if not result_text:
            return
        round_no = self._seer_memory.round_no
        if target:
            entry = {"round": round_no, "target": target, "result": result_text}
            if entry not in self._seer_memory.inspected:
                self._seer_memory.inspected.append(entry)
            self._seer_memory.inspection_map[target] = result_text

    def _inspection_result_text(self, value: Any) -> str:
        text = self._truncate_text(self._stringify(value), _SUMMARY_TEXT_LIMIT)
        if not text:
            return ""
        normalized = text.lower()
        if any(token in normalized for token in ("wolf", "werewolf", "evil", "狼", "bad", "killer")):
            return "wolf"
        if any(token in normalized for token in ("villager", "good", "human", "好人", "村民", "平民", "village")):
            return "villager"
        if isinstance(value, Mapping):
            team = self._stringify(value.get("team") or value.get("camp") or value.get("alignment")).lower()
            if team in {"village", "villager", "good", "human", "好人", "村民", "平民"}:
                return "villager"
            if team in {"wolf", "werewolf", "evil", "狼", "bad"}:
                return "wolf"
        return text

    def _build_seer_brief(self, turn_packet: Mapping[str, Any], mode: str) -> dict[str, Any]:
        request = turn_packet.get("request") or {}
        allowed_actions = request.get("allowed_actions") if isinstance(request, Mapping) else None
        allowed_kinds = [
            str(item.get("kind"))
            for item in allowed_actions
            if isinstance(item, Mapping) and item.get("kind")
        ] if isinstance(allowed_actions, Sequence) and not isinstance(allowed_actions, (str, bytes, bytearray)) else []
        target_candidates = self._allowed_target_ids(allowed_actions)
        preferred_targets, reasons = self._rank_inspect_targets(turn_packet, target_candidates)
        facts = self._seer_facts_summary()
        pressure_points = self._pressure_points_summary(turn_packet)
        pressure_signals = self._recent_pressure_signals()
        self_risk = self._self_risk_summary(turn_packet)
        brief: dict[str, Any] = {
            "mode": mode,
            "allowed_kinds": allowed_kinds[:_SUMMARY_LIST_LIMIT],
            "facts": facts[:_SUMMARY_LIST_LIMIT],
            "pressure_points": pressure_points[:_SUMMARY_LIST_LIMIT],
            "pressure_signals": pressure_signals[:_SUMMARY_LIST_LIMIT],
            "self_risk": self_risk,
            "self_risk_mode": self._seer_memory.self_risk_mode,
        }
        if preferred_targets:
            brief["preferred_targets"] = preferred_targets[:_SUMMARY_LIST_LIMIT]
            brief["target_reasons"] = reasons[:_SUMMARY_LIST_LIMIT]
            brief["pressure_targets"] = preferred_targets[:_SUMMARY_LIST_LIMIT]
        if self._seer_memory.claim_timeline:
            brief["claim_timeline"] = self._seer_memory.claim_timeline[-_SUMMARY_LIST_LIMIT:]
        if self._seer_memory.vote_timeline:
            brief["vote_timeline"] = self._seer_memory.vote_timeline[-_SUMMARY_LIST_LIMIT:]
        if self._seer_memory.pressure_targets:
            brief["pressure_target_history"] = self._seer_memory.pressure_targets[-_SUMMARY_LIST_LIMIT:]
        return brief

    def _seer_facts_summary(self) -> list[str]:
        facts: list[str] = []
        for item in self._seer_memory.inspected[-_SUMMARY_LIST_LIMIT:]:
            if not isinstance(item, Mapping):
                continue
            round_no = self._stringify(item.get("round"))
            target = self._stringify(item.get("target"))
            result = self._normalize_inspection_result(item.get("result"))
            if target and result:
                facts.append(f"{round_no + '轮' if round_no else '查验'}{target}={result}")
        if self._seer_memory.inspection_map:
            recent_map = ", ".join(
                f"{target}->{result}" for target, result in list(self._seer_memory.inspection_map.items())[-3:]
            )
            if recent_map:
                facts.append(f"查验映射：{recent_map}")
        if self._seer_memory.sheriff_id:
            facts.append(f"警长/队长位：{self._seer_memory.sheriff_id}")
        if self._seer_memory.recent_vote_summary:
            facts.append(f"最近票型：{self._seer_memory.recent_vote_summary}")
        if self._seer_memory.claim_timeline:
            last_claim = self._seer_memory.claim_timeline[-1]
            speaker = self._stringify(last_claim.get("speaker"))
            claim = self._stringify(last_claim.get("claim"))
            if speaker or claim:
                facts.append(f"最新公开声明：{speaker + ' ' if speaker else ''}{claim}".strip())
        if self._seer_memory.pressure_targets:
            latest_pressure = self._seer_memory.pressure_targets[-1]
            target = self._stringify(latest_pressure.get("target"))
            reason = self._stringify(latest_pressure.get("reason"))
            if target or reason:
                facts.append(f"当前压力位：{target + ' ' if target else ''}{reason}".strip())
        if self._seer_memory.seat_to_player_id:
            seat_map = ", ".join(
                f"{seat}->{player}" for seat, player in list(self._seer_memory.seat_to_player_id.items())[:3]
            )
            if seat_map:
                facts.append(f"席位映射：{seat_map}")
        return facts[:_SUMMARY_LIST_LIMIT]

    def _pressure_points_summary(self, turn_packet: Mapping[str, Any]) -> list[str]:
        points: list[str] = []
        if self._seer_memory.sheriff_candidates:
            points.append("警长候选：" + "、".join(self._seer_memory.sheriff_candidates[:_SUMMARY_LIST_LIMIT]))
        if self._seer_memory.challenged_players:
            points.append("未解决对跳链：" + "、".join(self._seer_memory.challenged_players[:_SUMMARY_LIST_LIMIT]))
        if self._seer_memory.pressure_targets:
            latest_targets = [self._stringify(item.get("target")) for item in self._seer_memory.pressure_targets[-_SUMMARY_LIST_LIMIT:]]
            latest_targets = [item for item in latest_targets if item]
            if latest_targets:
                points.append("压力目标：" + "、".join(self._unique_stable(latest_targets)))
        vote_detail = self._seer_memory.recent_vote_detail
        if isinstance(vote_detail, Mapping) and vote_detail:
            points.append("票型详情已更新")
        pressure = self._recent_pressure_signals()
        if pressure:
            points.append("压力信号：" + "、".join(pressure[:_SUMMARY_LIST_LIMIT]))
        request = turn_packet.get("request") or {}
        allowed_targets = self._allowed_target_ids(request.get("allowed_actions") if isinstance(request, Mapping) else None)
        ranked, reasons = self._rank_inspect_targets(turn_packet, allowed_targets)
        if ranked:
            points.append("夜验优先：" + "、".join(ranked[:_SUMMARY_LIST_LIMIT]))
        if reasons:
            points.append("夜验理由：" + "；".join(reasons[:_SUMMARY_LIST_LIMIT]))
        return points[:_SUMMARY_LIST_LIMIT]

    def _self_risk_summary(self, turn_packet: Mapping[str, Any]) -> str:
        self_seen = self.player_id in self._seer_memory.challenged_players
        other_claims = len(self._seer_memory.declared_claims)
        phase = self._extract_phase(turn_packet).lower()
        pressure_count = len(self._recent_pressure_signals())
        if self_seen or self._seer_memory.self_risk_mode == "high":
            return "自保压力高：已被公开点名或竞争激烈，发言只保留事实、矛盾、票向。"
        if pressure_count >= 3 or self._seer_memory.self_risk_mode == "medium":
            return "自保压力中等：用短模板输出，先事实后矛盾再票向。"
        if "elect" in phase or "sheriff" in phase or "警长" in phase or "竞选" in phase:
            return "身份暴露风险中等：竞选期要先立可信事实，再给投票方向。"
        if other_claims >= 3:
            return "信息竞争激烈：要把查验事实尽快转成压票点。"
        return "常规风险：优先保留信息优势，不要过度展开。"

    def _recent_pressure_signals(self) -> list[str]:
        signals: list[str] = []
        if self._seer_memory.challenged_players:
            signals.append("未解决对跳链")
        if self._seer_memory.pressure_targets:
            signals.append("压力目标已形成")
        if self._seer_memory.recent_vote_summary:
            signals.append("票型已更新")
        if len(self._seer_memory.declared_claims) >= 2:
            signals.append("公开声明较多")
        if any(self._speaker_claim_variation_count(item.get("speaker", "")) >= 2 for item in self._seer_memory.declared_claims if isinstance(item, Mapping)):
            signals.append("同一发言者多次自述")
        if any(self._recent_mention_count(target_id) >= 3 for target_id in self._seer_memory.challenged_players[:_SUMMARY_LIST_LIMIT]):
            signals.append("点名反复")
        if self._seer_memory.self_risk_mode != "normal":
            signals.append(f"自保模式:{self._seer_memory.self_risk_mode}")
        return self._unique_stable(signals)

    def _decide_mode(self, turn_packet: Mapping[str, Any]) -> str:
        request = turn_packet.get("request") or {}
        allowed_actions = request.get("allowed_actions") if isinstance(request, Mapping) else None
        allowed_kinds = [str(item.get("kind") or "").lower() for item in allowed_actions if isinstance(item, Mapping)] if isinstance(allowed_actions, Sequence) and not isinstance(allowed_actions, (str, bytes, bytearray)) else []
        phase = (self._extract_phase(turn_packet) or "").lower()
        channel = str(request.get("channel") or "").lower() if isinstance(request, Mapping) else ""
        if any(self._kind_looks_like_inspect(kind) for kind in allowed_kinds):
            return "night_inspect"
        if any(kind in _SPEECH_ACTION_KINDS for kind in allowed_kinds):
            if self._seer_memory.self_risk_mode == "high" or self.player_id in self._seer_memory.challenged_players:
                return "self_risk"
            if any(token in phase for token in ("sheriff", "election", "竞选", "警长")) or any(
                token in channel for token in ("sheriff", "election", "竞选", "警长")
            ):
                return "sheriff_election_speech"
            return "day_speech"
        if any(self._kind_looks_like_vote(kind) for kind in allowed_kinds):
            return "vote"
        return "general"

    def _kind_looks_like_inspect(self, kind: str) -> bool:
        kind = kind.lower()
        return any(token in kind for token in ("inspect", "check", "seer", "查验", "验", "探查"))

    def _kind_looks_like_vote(self, kind: str) -> bool:
        kind = kind.lower()
        return any(token in kind for token in ("vote", "ballot", "lynch", "投票", "归票"))

    def _allowed_target_ids(self, allowed_actions: Any) -> list[str]:
        target_ids: list[str] = []
        if not isinstance(allowed_actions, Sequence) or isinstance(allowed_actions, (str, bytes, bytearray)):
            return target_ids
        for item in allowed_actions:
            if not isinstance(item, Mapping):
                continue
            ids = item.get("target_ids")
            if isinstance(ids, Sequence) and not isinstance(ids, (str, bytes, bytearray)):
                for target_id in ids:
                    if isinstance(target_id, str) and target_id:
                        target_ids.append(target_id)
        return self._unique_stable(target_ids)

    def _rank_inspect_targets(
        self, turn_packet: Mapping[str, Any], candidate_ids: Sequence[str]
    ) -> tuple[list[str], list[str]]:
        alive = self._seer_memory.alive or self._extract_player_ids(
            self._first_present(turn_packet, (("public_state", "alive"), ("public_state", "alive_players")))
        )
        seen_targets = self._seen_inspection_targets()
        candidates = [
            target_id
            for target_id in self._unique_stable(candidate_ids)
            if target_id and target_id != self.player_id and target_id not in seen_targets
        ]
        if not candidates:
            candidates = [
                target_id
                for target_id in alive
                if target_id and target_id != self.player_id and target_id not in seen_targets
            ]
        scores: dict[str, int] = {target_id: 0 for target_id in candidates}
        reasons: dict[str, list[str]] = {target_id: [] for target_id in candidates}

        sheriff_like = self._unique_stable([self._seer_memory.sheriff_id] + self._seer_memory.sheriff_candidates)
        pressure_targets = {self._normalize_reference(item.get("target")) for item in self._seer_memory.pressure_targets}
        unresolved_claims = {self._normalize_reference(item.get("speaker")) for item in self._seer_memory.declared_claims if isinstance(item, Mapping)}
        vote_hotspots = {self._normalize_reference(item.get("target")) for item in self._seer_memory.vote_timeline[-_SUMMARY_LIST_LIMIT:] if isinstance(item, Mapping)}
        for target_id in candidates:
            if target_id in sheriff_like:
                scores[target_id] += 45
                reasons[target_id].append("警长链关键位")
            if target_id in self._seer_memory.challenged_players:
                scores[target_id] += 34
                reasons[target_id].append("未解决对跳")
            if target_id in pressure_targets:
                scores[target_id] += 22
                reasons[target_id].append("公共压力位")
            if target_id in vote_hotspots:
                scores[target_id] += 18
                reasons[target_id].append("票型关键位")
            if target_id in unresolved_claims or self._has_declared_claim(target_id):
                scores[target_id] += 16
                reasons[target_id].append("公开表态")
            if self._claim_conflicts_with_known_inspection(target_id):
                scores[target_id] += 24
                reasons[target_id].append("与查验矛盾")
            if self._speaks_a_lot(target_id):
                scores[target_id] += 8
                reasons[target_id].append("发言密集")

            pressure_reasons = self._target_pressure_signals(target_id)
            if pressure_reasons:
                scores[target_id] += min(20, 4 * len(pressure_reasons))
                reasons[target_id].extend(pressure_reasons)

        ranked = sorted(candidates, key=lambda item: (-scores[item], candidates.index(item)))
        ranked_reasons = [
            f"{target_id}:{'、'.join(self._unique_stable(reasons[target_id])) or '默认信息增益'}"
            for target_id in ranked
        ]
        return ranked, ranked_reasons

    def _seen_inspection_targets(self) -> set[str]:
        seen: set[str] = set()
        for item in self._seer_memory.inspected:
            if isinstance(item, Mapping):
                target = self._normalize_reference(item.get("target"))
                if target:
                    seen.add(target)
        return seen

    def _has_declared_claim(self, target_id: str) -> bool:
        for claim in self._seer_memory.declared_claims:
            if not isinstance(claim, Mapping):
                continue
            speaker = self._normalize_reference(claim.get("speaker"))
            claim_text = self._normalize_alias_text(self._stringify(claim.get("claim")))
            if speaker == target_id or self._text_mentions_reference(speaker, target_id):
                return True
            if self._text_mentions_reference(claim_text, target_id):
                return True
        return False

    def _appears_in_recent_vote(self, target_id: str) -> bool:
        vote_detail = self._seer_memory.recent_vote_detail
        if not vote_detail:
            return False
        text = json.dumps(vote_detail, ensure_ascii=False, separators=(",", ":"))
        return self._text_mentions_reference(text, target_id)

    def _recent_mention_count(self, target_id: str) -> int:
        count = 0
        for line in self._seer_memory.dialogue_digest:
            if self._text_mentions_reference(line, target_id):
                count += 1
        for claim in self._seer_memory.declared_claims:
            if not isinstance(claim, Mapping):
                continue
            if self._text_mentions_reference(self._stringify(claim.get("speaker")), target_id):
                count += 1
            if self._text_mentions_reference(self._stringify(claim.get("claim")), target_id):
                count += 1
        if self._text_mentions_reference(self._seer_memory.recent_vote_summary, target_id):
            count += 1
        if self._appears_in_recent_vote(target_id):
            count += 1
        if target_id in self._seer_memory.challenged_players:
            count += 1
        return count

    def _speaker_claim_variation_count(self, target_id: str) -> int:
        claims: list[str] = []
        for claim in self._seer_memory.declared_claims:
            if not isinstance(claim, Mapping):
                continue
            speaker = self._normalize_reference(claim.get("speaker"))
            if speaker == target_id:
                claim_text = self._truncate_text(self._normalize_alias_text(self._stringify(claim.get("claim"))), 80)
                if claim_text:
                    claims.append(claim_text)
        return len(self._unique_stable(claims))

    def _target_pressure_signals(self, target_id: str) -> list[str]:
        signals: list[str] = []
        mention_count = self._recent_mention_count(target_id)
        if mention_count >= 3:
            signals.append(f"近期被点名{mention_count}次")
        variation_count = self._speaker_claim_variation_count(target_id)
        if variation_count >= 2:
            signals.append("跨轮声明不一致")
        if target_id in self._seer_memory.challenged_players and self._has_declared_claim(target_id):
            signals.append("对跳与自述同在")
        if self._appears_in_recent_vote(target_id) and self._text_mentions_reference(self._seer_memory.recent_vote_summary, target_id):
            signals.append("票型转折焦点")
        if self._claim_conflicts_with_known_inspection(target_id):
            signals.append("与已知查验矛盾")
        return self._unique_stable(signals)

    def _claim_conflicts_with_known_inspection(self, target_id: str) -> bool:
        target_result = self._normalize_inspection_result(self._seer_memory.inspection_map.get(target_id, ""))
        if not target_result:
            for item in self._seer_memory.inspected:
                if not isinstance(item, Mapping):
                    continue
                if self._normalize_reference(item.get("target")) == target_id:
                    target_result = self._normalize_inspection_result(item.get("result"))
        if not target_result:
            return False
        claim_texts = []
        for claim in self._seer_memory.declared_claims:
            if not isinstance(claim, Mapping):
                continue
            if self._normalize_reference(claim.get("speaker")) == target_id:
                claim_texts.append(self._normalize_alias_text(self._stringify(claim.get("claim"))).lower())
        if not claim_texts:
            return False
        wolf_like = target_result == "wolf"
        villager_like = target_result == "villager"
        if wolf_like:
            return any(token in text for text in claim_texts for token in ("好人", "金水", "查杀", "验好", "村民", "平民", "villager"))
        if villager_like:
            return any(token in text for text in claim_texts for token in ("狼人", "狼", "查杀", "黑", "坏人", "wolf"))
        return False

    def _speaks_a_lot(self, target_id: str) -> bool:
        count = 0
        for line in self._seer_memory.dialogue_digest:
            if self._text_mentions_reference(line, target_id):
                count += 1
        return count >= 2

    def _compact_request(self, request: Mapping[str, Any]) -> dict[str, Any]:
        allowed_actions = request.get("allowed_actions")
        compact_actions: list[dict[str, Any]] = []
        if isinstance(allowed_actions, Sequence) and not isinstance(allowed_actions, (str, bytes, bytearray)):
            for allowed in allowed_actions[:_SUMMARY_LIST_LIMIT]:
                if not isinstance(allowed, Mapping):
                    continue
                item: dict[str, Any] = {"kind": self._stringify(allowed.get("kind"))}
                target_ids = allowed.get("target_ids")
                if isinstance(target_ids, Sequence) and not isinstance(target_ids, (str, bytes, bytearray)):
                    ids = [self._stringify(v) for v in list(target_ids)[:_SUMMARY_LIST_LIMIT] if self._stringify(v)]
                    if ids:
                        item["target_ids"] = ids
                if allowed.get("max_chars") is not None:
                    item["max_chars"] = allowed.get("max_chars")
                if allowed.get("require_chinese") is not None:
                    item["require_chinese"] = bool(allowed.get("require_chinese"))
                compact_actions.append(item)
        return {
            "request_id": self._stringify(request.get("request_id")),
            "phase": self._stringify(request.get("phase")),
            "channel": self._stringify(request.get("channel")),
            "allowed_actions": compact_actions,
        }

    def _compact_private_information(self, private: Mapping[str, Any]) -> dict[str, Any]:
        compact = self._compact_value(private, max_depth=2)
        if isinstance(compact, dict):
            compact["seer_memory_hint"] = self._compact_seer_memory()
        return compact

    def _compact_seer_memory(self) -> dict[str, Any]:
        memory = {
            "game_id": self._seer_memory.game_id,
            "round": self._seer_memory.round_no,
            "phase": self._seer_memory.phase,
            "alive": self._seer_memory.alive[:_SUMMARY_LIST_LIMIT],
            "dead": self._seer_memory.dead[:_SUMMARY_LIST_LIMIT],
            "sheriff_id": self._seer_memory.sheriff_id,
            "sheriff_candidates": self._seer_memory.sheriff_candidates[:_SUMMARY_LIST_LIMIT],
            "seat_to_player_id": dict(list(self._seer_memory.seat_to_player_id.items())[:_SUMMARY_LIST_LIMIT]),
            "player_to_seat": dict(list(self._seer_memory.player_to_seat.items())[:_SUMMARY_LIST_LIMIT]),
            "inspected": self._seer_memory.inspected[-_SUMMARY_LIST_LIMIT:],
            "inspection_map": dict(list(self._seer_memory.inspection_map.items())[:_SUMMARY_LIST_LIMIT]),
            "declared_claims": self._seer_memory.declared_claims[-_SUMMARY_LIST_LIMIT:],
            "claim_timeline": self._seer_memory.claim_timeline[-_SUMMARY_LIST_LIMIT:],
            "vote_timeline": self._seer_memory.vote_timeline[-_SUMMARY_LIST_LIMIT:],
            "pressure_targets": self._seer_memory.pressure_targets[-_SUMMARY_LIST_LIMIT:],
            "self_risk_mode": self._seer_memory.self_risk_mode,
            "challenged_players": self._seer_memory.challenged_players[:_SUMMARY_LIST_LIMIT],
            "recent_vote_summary": self._seer_memory.recent_vote_summary,
            "dialogue_digest": self._seer_memory.dialogue_digest[-_SUMMARY_LIST_LIMIT:],
        }
        return memory

    def _compact_value(self, value: Any, *, max_depth: int, _depth: int = 0) -> Any:
        if value is None or isinstance(value, (bool, int, float)):
            return value
        if isinstance(value, str):
            return self._truncate_text(value, _SUMMARY_TEXT_LIMIT)
        if _depth >= max_depth:
            return self._truncate_text(self._stringify(value), _SUMMARY_TEXT_LIMIT)
        if isinstance(value, Mapping):
            items: dict[str, Any] = {}
            for idx, (key, nested) in enumerate(value.items()):
                if idx >= _SUMMARY_LIST_LIMIT:
                    items["__truncated__"] = len(value) - _SUMMARY_LIST_LIMIT
                    break
                items[self._stringify(key)] = self._compact_value(nested, max_depth=max_depth, _depth=_depth + 1)
            return items
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            items = [
                self._compact_value(item, max_depth=max_depth, _depth=_depth + 1)
                for item in list(value)[:_SUMMARY_LIST_LIMIT]
            ]
            if len(value) > _SUMMARY_LIST_LIMIT:
                items.append(f"...(+{len(value) - _SUMMARY_LIST_LIMIT})")
            return items
        return self._truncate_text(self._stringify(value), _SUMMARY_TEXT_LIMIT)

    def _unique_stable(self, values: Sequence[str]) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for value in values:
            text = self._stringify(value)
            if not text or text in seen:
                continue
            seen.add(text)
            ordered.append(text)
        return ordered

    def _dedupe_dict_list(self, values: list[dict[str, str]]) -> list[dict[str, str]]:
        seen: set[tuple[str, str]] = set()
        deduped: list[dict[str, str]] = []
        for item in values:
            if not isinstance(item, Mapping):
                continue
            speaker = self._stringify(item.get("speaker"))
            claim = self._stringify(item.get("claim"))
            key = (speaker, claim)
            if key in seen:
                continue
            seen.add(key)
            deduped.append({"speaker": speaker, "claim": claim})
        return deduped

    def _speaker_from_text(self, text: str) -> str:
        ids = _PLAYER_ID_PATTERN.findall(text)
        if ids:
            return ids[0]
        resolved = self._resolved_player_ids_from_text(text)
        return resolved[0] if resolved else ""

    @staticmethod
    def _truncate_text(text: str, limit: int) -> str:
        text = text.strip()
        if len(text) <= limit:
            return text
        return text[: max(0, limit - 1)] + "…"

    @staticmethod
    def _stringify(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, bool):
            return "true" if value else "false"
        try:
            return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        except TypeError:
            return str(value)

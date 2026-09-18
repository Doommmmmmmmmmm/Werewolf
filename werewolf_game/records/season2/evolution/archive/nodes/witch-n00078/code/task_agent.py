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
        self._witch_suspicion_ledger: dict[str, dict[str, Any]] = {}
        self._witch_recent_public_deaths: list[str] = []
        self._witch_recent_vote_pressure: list[str] = []
        self._witch_recent_counterclaims: list[str] = []
        self._witch_recent_role_chain: list[str] = []
        self._witch_public_evidence: dict[str, Any] = {
            "claims": {},
            "reveals": {},
            "votes": {},
            "deaths": [],
            "contradictions": [],
            "sheriff_votes": {},
        }

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
                self._recent_public_summaries = self._recent_public_summaries[-6:]
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
        planned_action = self._plan_witch_action(turn_packet, private, witch_context)
        if planned_action is not None:
            self._last_decision_audit = {
                "planned_action": planned_action,
                "witch_decision_context": witch_context,
            }
            return planned_action

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
            "recent_public": self._recent_public_summaries[-4:],
        }
        if self._last_public_round_phase:
            snapshot["last_public_round_phase"] = self._last_public_round_phase
        if self._last_private_resource_summary:
            snapshot["private_resource_state"] = self._last_private_resource_summary
        if witch_context:
            snapshot["witch_decision_summary"] = self._compact_witch_context(witch_context)
        if self.profile.role == "witch" and self._witch_suspicion_ledger:
            snapshot["witch_top_suspects"] = self._build_witch_ledger_snapshot()
        return snapshot

    def _build_private_resource_summary(self, private: Mapping[str, Any]) -> str:
        if self.profile.role != "witch" and str(private.get("role")) != "witch":
            return ""
        keys = {
            "antidote_available": self._extract_first_scalar(private, ("antidote_available", "save_available", "heal_available")),
            "poison_available": self._extract_first_scalar(private, ("poison_available", "poison_left", "kill_available")),
            "antidote_used": self._extract_first_scalar(private, ("antidote_used", "used_antidote", "save_used")),
            "poison_used": self._extract_first_scalar(private, ("poison_used", "used_poison", "kill_used")),
        }
        parts: list[str] = []
        for key, value in keys.items():
            if value is None:
                continue
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

        resource_state = {
            "antidote_available": self._coerce_bool_scalar(self._extract_first_scalar(private, ("antidote_available", "save_available", "heal_available"))),
            "poison_available": self._coerce_bool_scalar(self._extract_first_scalar(private, ("poison_available", "poison_left", "kill_available"))),
            "antidote_used": self._coerce_bool_scalar(self._extract_first_scalar(private, ("antidote_used", "used_antidote", "save_used"))),
            "poison_used": self._coerce_bool_scalar(self._extract_first_scalar(private, ("poison_used", "used_poison", "kill_used"))),
        }

        serialized_sources: list[str] = []
        for value in (
            private,
            turn_packet.get("public_state"),
            turn_packet.get("game"),
            request,
            turn_packet.get("tool_context"),
            self._recent_public_summaries[-4:],
        ):
            if value is not None:
                serialized_sources.append(self._safe_json_dump(value))
        combined_text = " \n ".join(serialized_sources)

        night_death_targets = self._extract_targets_from_text(
            combined_text,
            (
                "夜杀",
                "狼刀",
                "刀口",
                "死亡",
                "出局",
                "淘汰",
                "night kill",
                "wolf kill",
                "killed",
                "dead",
            ),
        )
        high_value_rescue_targets = self._extract_targets_from_text(
            combined_text,
            (
                "警长",
                "警徽",
                "上警",
                "查验",
                "验出",
                "查杀",
                "金水",
                "对跳",
                "悍跳",
                "claim",
                "counterclaim",
                "sheriff",
                "seer",
                "investigate",
                "reveal",
            ),
        )

        ledger_snapshot = self._build_witch_ledger_snapshot()
        high_confidence_targets = sorted(
            self._high_confidence_targets(turn_packet, ledger_snapshot=ledger_snapshot)
        )
        if high_confidence_targets:
            high_confidence_targets = high_confidence_targets[:5]
        rescue_targets = sorted((night_death_targets | high_value_rescue_targets))[:5]
        recommendation = "pass"
        poison_target = high_confidence_targets[0] if high_confidence_targets else ""
        rescue_target = rescue_targets[0] if rescue_targets else ""
        if resource_state.get("antidote_available") and rescue_target:
            recommendation = f"heal:{rescue_target}"
        elif resource_state.get("poison_available") and poison_target:
            recommendation = f"poison:{poison_target}"

        return {
            "resource_state": resource_state,
            "current_action_kinds": action_kinds[:4],
            "current_action_targets": action_targets[:6],
            "night_death_targets": sorted(night_death_targets)[:5],
            "high_value_rescue_targets": rescue_targets,
            "high_confidence_targets": high_confidence_targets,
            "recent_public_deaths": self._witch_recent_public_deaths[-4:],
            "recent_vote_pressure": self._witch_recent_vote_pressure[-4:],
            "counterclaim_summary": self._witch_recent_counterclaims[-4:],
            "role_chain_summary": self._witch_recent_role_chain[-4:],
            "ledger_top_suspects": ledger_snapshot,
            "recommendation": recommendation,
        }

    def _update_witch_memory(self, sync_packet: Mapping[str, Any]) -> None:
        serialized = self._safe_json_dump(sync_packet)
        if not serialized:
            return
        signal_specs = (
            ("death_chain", 3, self._witch_recent_public_deaths, ("死亡", "出局", "淘汰", "night kill", "wolf kill", "被刀", "刀口")),
            ("vote_pressure", 2, self._witch_recent_vote_pressure, ("投票", "票型", "归票", "vote", "ballot", "pressure")),
            ("claim_conflict", 2, self._witch_recent_counterclaims, ("悍跳", "对跳", "查杀", "金水", "claim", "counterclaim")),
            ("role_chain", 1, self._witch_recent_role_chain, ("警长", "警徽", "验出", "查验", "sheriff", "seer", "reveal", "确认")),
        )
        for signal, weight, sink, keywords in signal_specs:
            snippet = self._extract_keyword_snippet(serialized, keywords)
            if not snippet:
                continue
            players = self._extract_player_ids(snippet)
            if players:
                self._bump_witch_ledger(players, signal, weight, snippet)
            self._append_compact_entry(sink, f"{signal}:{snippet}")

    def _bump_witch_ledger(self, players: list[str], signal: str, weight: int, note: str) -> None:
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
            entry = self._witch_suspicion_ledger.setdefault(
                player_id,
                {
                    "score": 0,
                    "vote_pressure": 0,
                    "claim_conflict": 0,
                    "death_chain": 0,
                    "role_chain": 0,
                    "notes": [],
                },
            )
            entry["score"] = int(entry.get("score", 0)) + weight
            entry[field_name] = int(entry.get(field_name, 0)) + weight
            notes = entry.setdefault("notes", [])
            if note and (not notes or notes[-1] != note):
                notes.append(note)
                del notes[:-3]

    def _append_compact_entry(self, sink: list[str], entry: str, limit: int = 4) -> None:
        compact_entry = self._truncate_text(self._squash_whitespace(entry), 180)
        if not compact_entry:
            return
        if sink and sink[-1] == compact_entry:
            return
        sink.append(compact_entry)
        if len(sink) > limit:
            del sink[:-limit]

    def _build_witch_ledger_snapshot(self) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for player_id, payload in self._witch_suspicion_ledger.items():
            items.append(
                {
                    "player_id": player_id,
                    "score": int(payload.get("score", 0)),
                    "vote_pressure": int(payload.get("vote_pressure", 0)),
                    "claim_conflict": int(payload.get("claim_conflict", 0)),
                    "death_chain": int(payload.get("death_chain", 0)),
                    "role_chain": int(payload.get("role_chain", 0)),
                    "notes": list(payload.get("notes", []))[-2:],
                }
            )
        items.sort(
            key=lambda item: (
                -int(item.get("score", 0)),
                str(item.get("player_id", "")),
            )
        )
        return items[:4]

    def _compact_witch_context(self, witch_context: Mapping[str, Any]) -> str:
        summary = {
            "resource_state": witch_context.get("resource_state"),
            "night_death_targets": witch_context.get("night_death_targets", []),
            "high_value_rescue_targets": witch_context.get("high_value_rescue_targets", []),
            "high_confidence_targets": witch_context.get("high_confidence_targets", []),
            "recommendation": witch_context.get("recommendation", "pass"),
        }
        return self._truncate_text(self._safe_json_dump(summary), 900)

    def _build_safety_notes(self, private: Mapping[str, Any], witch_context: Mapping[str, Any] | None = None) -> str:
        if self.profile.role != "witch" and str(private.get("role")) != "witch":
            return ""
        notes = [
            "公开发言只允许基于公开事实；",
            "不得自报女巫、药水余量、夜刀目标、夜间选择、是否已用药、未公开身份链；",
            "任何内部状态都不能被说成已公开事实。",
        ]
        if witch_context:
            notes.append("决策时先核对 night_death_targets、high_value_rescue_targets、high_confidence_targets；")
            notes.append("救人默认只在确有夜死且能保住关键公开链时使用，毒药默认只在高置信候选集内使用。")
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

        confidence_targets = self._high_confidence_targets(turn_packet, witch_context=witch_context)
        rescue_targets = self._high_value_rescue_targets(turn_packet, witch_context=witch_context)
        if is_poison:
            if target_id in confidence_targets:
                self._last_decision_audit = {
                    "raw_response": raw_response,
                    "normalized_action": action,
                    "high_confidence_targets": sorted(confidence_targets),
                }
                return action
            if pass_action is not None:
                self._last_decision_audit = {
                    "raw_response": raw_response,
                    "normalized_action": action,
                    "fallback_action": pass_action,
                    "fallback_reason": "witch_poison_low_confidence",
                    "high_confidence_targets": sorted(confidence_targets),
                }
                return pass_action
            self._last_decision_audit = {
                "raw_response": raw_response,
                "normalized_action": action,
                "high_confidence_targets": sorted(confidence_targets),
            }
            return action

        if is_heal:
            if target_id in rescue_targets:
                self._last_decision_audit = {
                    "raw_response": raw_response,
                    "normalized_action": action,
                    "high_value_rescue_targets": sorted(rescue_targets),
                }
                return action
            if pass_action is not None:
                self._last_decision_audit = {
                    "raw_response": raw_response,
                    "normalized_action": action,
                    "fallback_action": pass_action,
                    "fallback_reason": "witch_heal_low_value",
                    "high_value_rescue_targets": sorted(rescue_targets),
                }
                return pass_action
        self._last_decision_audit = {
            "raw_response": raw_response,
            "normalized_action": action,
        }
        return action

    def _high_confidence_targets(
        self,
        turn_packet: dict[str, Any],
        *,
        witch_context: Mapping[str, Any] | None = None,
        ledger_snapshot: list[dict[str, Any]] | None = None,
    ) -> set[str]:
        confidence_targets: set[str] = set()
        ledger_snapshot = ledger_snapshot if ledger_snapshot is not None else self._build_witch_ledger_snapshot()
        for item in ledger_snapshot:
            player_id = str(item.get("player_id") or "").strip()
            if not player_id:
                continue
            score = int(item.get("score", 0))
            vote_pressure = int(item.get("vote_pressure", 0))
            claim_conflict = int(item.get("claim_conflict", 0))
            death_chain = int(item.get("death_chain", 0))
            role_chain = int(item.get("role_chain", 0))
            if score >= 4 or (vote_pressure >= 2 and claim_conflict >= 1) or (claim_conflict >= 2 and death_chain >= 1) or (death_chain >= 2 and role_chain >= 1):
                confidence_targets.add(player_id)
        if witch_context:
            for item in witch_context.get("ledger_top_suspects", []):
                if not isinstance(item, Mapping):
                    continue
                player_id = str(item.get("player_id") or "").strip()
                if not player_id:
                    continue
                score = int(item.get("score", 0))
                if score >= 4:
                    confidence_targets.add(player_id)
            for player_id in witch_context.get("high_confidence_targets", []):
                player_text = str(player_id).strip()
                if player_text:
                    confidence_targets.add(player_text)

        texts: list[str] = []
        texts.extend(self._recent_public_summaries[-4:])
        texts.append(self._last_public_round_phase)
        texts.append(self._last_private_resource_summary)
        for key in ("public_state", "game", "request"):
            value = turn_packet.get(key)
            if value is not None:
                texts.append(self._safe_json_dump(value))
        tool_context = turn_packet.get("tool_context") or {}
        current_dialogue = tool_context.get("current_round_dialogue")
        if current_dialogue:
            texts.append(self._safe_json_dump(current_dialogue))
        combined = " \n ".join(text for text in texts if text)
        combined_lower = combined.lower()
        if not combined:
            return confidence_targets
        if any(marker in combined_lower for marker in ("查杀", "悍跳", "对跳", "金水", "claim", "counterclaim", "sheriff", "seer", "vote", "票型", "归票")):
            for snippet_keywords in (("查杀", "悍跳", "对跳", "金水", "claim", "counterclaim", "sheriff", "seer", "vote", "票型", "归票"),):
                confidence_targets.update(self._extract_targets_from_text(combined, snippet_keywords))
        return confidence_targets

    def _high_value_rescue_targets(
        self,
        turn_packet: dict[str, Any],
        *,
        witch_context: Mapping[str, Any] | None = None,
    ) -> set[str]:
        rescue_targets: set[str] = set()
        if witch_context:
            for player_id in witch_context.get("night_death_targets", []):
                player_text = str(player_id).strip()
                if player_text:
                    rescue_targets.add(player_text)
            for player_id in witch_context.get("high_value_rescue_targets", []):
                player_text = str(player_id).strip()
                if player_text:
                    rescue_targets.add(player_text)
        texts: list[str] = []
        for key in ("private_information", "public_state", "game", "request"):
            value = turn_packet.get(key)
            if value is not None:
                texts.append(self._safe_json_dump(value))
        tool_context = turn_packet.get("tool_context") or {}
        current_dialogue = tool_context.get("current_round_dialogue")
        if current_dialogue:
            texts.append(self._safe_json_dump(current_dialogue))
        combined = " \n ".join(texts)
        rescue_targets.update(
            self._extract_targets_from_text(
                combined,
                (
                    "夜杀",
                    "狼刀",
                    "刀口",
                    "死亡",
                    "出局",
                    "淘汰",
                    "警长",
                    "警徽",
                    "上警",
                    "查验",
                    "验出",
                    "查杀",
                    "金水",
                    "对跳",
                    "悍跳",
                    "claim",
                    "counterclaim",
                    "sheriff",
                    "seer",
                    "investigate",
                    "reveal",
                ),
            )
        )
        return rescue_targets

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
        self._witch_suspicion_ledger = {}
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

    def _compact_memory_snapshot(self, witch_context: Mapping[str, Any] | None = None) -> dict[str, Any]:
        snapshot: dict[str, Any] = {
            "recent_public": self._recent_public_summaries[-4:],
        }
        if self._last_public_round_phase:
            snapshot["last_public_round_phase"] = self._last_public_round_phase
        if self._last_private_resource_summary:
            snapshot["private_resource_state"] = self._last_private_resource_summary
        if witch_context:
            snapshot["witch_decision_summary"] = self._compact_witch_context(witch_context)
        if self.profile.role == "witch" and any(self._witch_public_evidence.values()):
            snapshot["witch_public_evidence"] = self._build_witch_public_evidence_snapshot()
            snapshot["witch_top_suspects"] = self._build_witch_ledger_snapshot()
        return snapshot

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

        resource_state = {
            "antidote_available": self._coerce_bool_scalar(self._extract_first_scalar(private, ("antidote_available", "save_available", "heal_available"))),
            "poison_available": self._coerce_bool_scalar(self._extract_first_scalar(private, ("poison_available", "poison_left", "kill_available"))),
            "antidote_used": self._coerce_bool_scalar(self._extract_first_scalar(private, ("antidote_used", "used_antidote", "save_used"))),
            "poison_used": self._coerce_bool_scalar(self._extract_first_scalar(private, ("poison_used", "used_poison", "kill_used"))),
        }

        night_death_targets = self._ranked_witch_targets(
            candidates=self._collect_witch_candidates(turn_packet, ("night_death", "death", "wolf", "kill")),
            target_kind="rescue",
            turn_packet=turn_packet,
        )
        high_value_rescue_targets = [item["player_id"] for item in night_death_targets if item["score"] >= 4]
        poison_ranked = self._ranked_witch_targets(
            candidates=self._collect_witch_candidates(turn_packet, ("vote", "claim", "reveal", "sheriff", "seer", "pressure")),
            target_kind="poison",
            turn_packet=turn_packet,
        )
        high_confidence_targets = [item["player_id"] for item in poison_ranked if item["score"] >= 4]
        recommendation = "pass"
        if resource_state.get("antidote_available") and high_value_rescue_targets:
            recommendation = f"heal:{high_value_rescue_targets[0]}"
        elif resource_state.get("poison_available") and high_confidence_targets:
            recommendation = f"poison:{high_confidence_targets[0]}"

        return {
            "resource_state": resource_state,
            "current_action_kinds": action_kinds[:4],
            "current_action_targets": action_targets[:6],
            "night_death_targets": high_value_rescue_targets[:5],
            "high_value_rescue_targets": high_value_rescue_targets[:5],
            "high_confidence_targets": high_confidence_targets[:5],
            "recent_public_deaths": self._witch_recent_public_deaths[-4:],
            "recent_vote_pressure": self._witch_recent_vote_pressure[-4:],
            "counterclaim_summary": self._witch_recent_counterclaims[-4:],
            "role_chain_summary": self._witch_recent_role_chain[-4:],
            "ledger_top_suspects": self._build_witch_ledger_snapshot(),
            "recommendation": recommendation,
        }

    def _update_witch_memory(self, sync_packet: Mapping[str, Any]) -> None:
        round_tag = self._build_round_phase_tag(sync_packet)
        for record in self._extract_witch_structured_records(sync_packet, round_tag):
            kind = str(record.get("kind") or "")
            summary = self._truncate_text(self._squash_whitespace(str(record.get("summary") or "")), 180)
            players = list(record.get("players") or [])
            if kind == "death":
                self._append_compact_entry(self._witch_recent_public_deaths, f"death:{summary}")
                self._register_witch_death(record)
            elif kind == "vote":
                self._append_compact_entry(self._witch_recent_vote_pressure, f"vote:{summary}")
                self._register_witch_vote(record)
            elif kind == "reveal":
                self._append_compact_entry(self._witch_recent_role_chain, f"reveal:{summary}")
                self._register_witch_reveal(record)
            elif kind == "claim":
                self._append_compact_entry(self._witch_recent_counterclaims, f"claim:{summary}")
                self._register_witch_claim(record)
            elif kind == "contradiction":
                self._append_compact_entry(self._witch_recent_counterclaims, f"contradiction:{summary}")
                self._register_witch_contradiction(record)
            for player_id in players:
                if player_id and player_id in self._witch_suspicion_ledger:
                    self._witch_suspicion_ledger[player_id]["last_round"] = round_tag

    def _build_witch_ledger_snapshot(self) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for player_id, payload in self._witch_suspicion_ledger.items():
            score, reasons = self._score_witch_entry(payload)
            items.append(
                {
                    "player_id": player_id,
                    "score": score,
                    "vote_pressure": int(payload.get("vote_pressure", 0)),
                    "claim_conflict": int(payload.get("claim_conflict", 0)),
                    "death_chain": int(payload.get("death_chain", 0)),
                    "role_chain": int(payload.get("role_chain", 0)),
                    "sheriff_votes": int(payload.get("sheriff_votes", 0)),
                    "reasons": reasons[:3],
                }
            )
        items.sort(key=lambda item: (-int(item.get("score", 0)), str(item.get("player_id", ""))))
        return items[:4]

    def _compact_witch_context(self, witch_context: Mapping[str, Any]) -> str:
        summary = {
            "resource_state": witch_context.get("resource_state"),
            "night_death_targets": witch_context.get("night_death_targets", []),
            "high_value_rescue_targets": witch_context.get("high_value_rescue_targets", []),
            "high_confidence_targets": witch_context.get("high_confidence_targets", []),
            "recommendation": witch_context.get("recommendation", "pass"),
        }
        return self._truncate_text(self._safe_json_dump(summary), 900)

    def _build_safety_notes(self, private: Mapping[str, Any], witch_context: Mapping[str, Any] | None = None) -> str:
        if self.profile.role != "witch" and str(private.get("role")) != "witch":
            return ""
        notes = [
            "公开发言只允许基于公开事实；",
            "不得自报女巫、药水余量、夜刀目标、夜间选择、是否已用药、未公开身份链；",
            "内部判断要先看 structured evidence，再看少量模型推断。",
        ]
        if witch_context:
            notes.append("决策时先核对 night_death_targets、high_value_rescue_targets、high_confidence_targets；")
            notes.append("救人只在确有夜死且能保住关键公开链/警长/确认身份时优先；毒人只在明确冲突或票型收拢时优先。")
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
        ranked_rescue = self._ranked_witch_targets(candidates=self._collect_witch_candidates(turn_packet, ("night_death", "death", "wolf", "kill")), target_kind="rescue", turn_packet=turn_packet, witch_context=witch_context)
        ranked_poison = self._ranked_witch_targets(candidates=self._collect_witch_candidates(turn_packet, ("vote", "claim", "reveal", "sheriff", "seer", "pressure")), target_kind="poison", turn_packet=turn_packet, witch_context=witch_context)
        rescue_targets = {item["player_id"] for item in ranked_rescue if item["score"] >= 4}
        confidence_targets = {item["player_id"] for item in ranked_poison if item["score"] >= 4}

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

        if is_poison:
            if target_id in confidence_targets:
                self._last_decision_audit = {
                    "raw_response": raw_response,
                    "normalized_action": action,
                    "high_confidence_targets": sorted(confidence_targets),
                }
                return action
            if pass_action is not None:
                self._last_decision_audit = {
                    "raw_response": raw_response,
                    "normalized_action": action,
                    "fallback_action": pass_action,
                    "fallback_reason": "witch_poison_low_confidence",
                    "high_confidence_targets": sorted(confidence_targets),
                }
                return pass_action
            self._last_decision_audit = {
                "raw_response": raw_response,
                "normalized_action": action,
                "high_confidence_targets": sorted(confidence_targets),
            }
            return action

        if is_heal:
            if target_id in rescue_targets:
                self._last_decision_audit = {
                    "raw_response": raw_response,
                    "normalized_action": action,
                    "high_value_rescue_targets": sorted(rescue_targets),
                }
                return action
            if pass_action is not None:
                self._last_decision_audit = {
                    "raw_response": raw_response,
                    "normalized_action": action,
                    "fallback_action": pass_action,
                    "fallback_reason": "witch_heal_low_value",
                    "high_value_rescue_targets": sorted(rescue_targets),
                }
                return pass_action
        self._last_decision_audit = {
            "raw_response": raw_response,
            "normalized_action": action,
        }
        return action

    def _high_confidence_targets(
        self,
        turn_packet: dict[str, Any],
        *,
        witch_context: Mapping[str, Any] | None = None,
        ledger_snapshot: list[dict[str, Any]] | None = None,
    ) -> set[str]:
        confidence_targets: set[str] = set()
        ranked = self._ranked_witch_targets(
            candidates=self._collect_witch_candidates(turn_packet, ("vote", "claim", "reveal", "sheriff", "seer", "pressure")),
            target_kind="poison",
            turn_packet=turn_packet,
            witch_context=witch_context,
            ledger_snapshot=ledger_snapshot,
        )
        for item in ranked:
            if int(item.get("score", 0)) >= 4:
                confidence_targets.add(str(item.get("player_id") or "").strip())
        return {player_id for player_id in confidence_targets if player_id}

    def _high_value_rescue_targets(
        self,
        turn_packet: dict[str, Any],
        *,
        witch_context: Mapping[str, Any] | None = None,
    ) -> set[str]:
        rescue_targets: set[str] = set()
        ranked = self._ranked_witch_targets(
            candidates=self._collect_witch_candidates(turn_packet, ("night_death", "death", "wolf", "kill")),
            target_kind="rescue",
            turn_packet=turn_packet,
            witch_context=witch_context,
        )
        for item in ranked:
            if int(item.get("score", 0)) >= 4:
                player_id = str(item.get("player_id") or "").strip()
                if player_id:
                    rescue_targets.add(player_id)
        return rescue_targets

    def _plan_witch_action(
        self,
        turn_packet: Mapping[str, Any],
        private: Mapping[str, Any],
        witch_context: Mapping[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        request = turn_packet.get("request") if isinstance(turn_packet.get("request"), Mapping) else {}
        allowed_actions = request.get("allowed_actions") if isinstance(request, Mapping) else []
        if not isinstance(allowed_actions, list):
            return None
        heal_action = None
        poison_action = None
        for allowed in allowed_actions:
            if not isinstance(allowed, Mapping):
                continue
            kind = str(allowed.get("kind") or "")
            if self._is_heal_action_kind(kind) and heal_action is None:
                heal_action = allowed
            elif self._is_poison_action_kind(kind) and poison_action is None:
                poison_action = allowed
        if heal_action is None and poison_action is None:
            return None

        ranked_rescue = self._ranked_witch_targets(
            candidates=self._collect_witch_candidates(turn_packet, ("night_death", "death", "wolf", "kill")),
            target_kind="rescue",
            turn_packet=turn_packet,
            witch_context=witch_context,
        )
        ranked_poison = self._ranked_witch_targets(
            candidates=self._collect_witch_candidates(turn_packet, ("vote", "claim", "reveal", "sheriff", "seer", "pressure")),
            target_kind="poison",
            turn_packet=turn_packet,
            witch_context=witch_context,
        )
        self_id = str(turn_packet.get("self", {}).get("player_id") or self.player_id)
        valid_heal_targets = set(str(item) for item in (heal_action.get("target_ids") or []) if str(item))
        heal_target_id = str(heal_action.get("target_id") or "").strip()
        if heal_target_id:
            valid_heal_targets.add(heal_target_id)
        valid_poison_targets = set(str(item) for item in (poison_action.get("target_ids") or []) if str(item))
        poison_target_id = str(poison_action.get("target_id") or "").strip()
        if poison_target_id:
            valid_poison_targets.add(poison_target_id)

        if heal_action is not None:
            for item in ranked_rescue:
                target_id = str(item.get("player_id") or "").strip()
                if not target_id or target_id not in valid_heal_targets:
                    continue
                if int(item.get("score", 0)) >= 4:
                    planned = {"request_id": request["request_id"], "player_id": request["player_id"], "kind": str(heal_action.get("kind") or "")}
                    planned["target_id"] = target_id
                    return planned
                if target_id == self_id and int(item.get("score", 0)) >= 3:
                    planned = {"request_id": request["request_id"], "player_id": request["player_id"], "kind": str(heal_action.get("kind") or "")}
                    planned["target_id"] = target_id
                    return planned

        if poison_action is not None:
            for item in ranked_poison:
                target_id = str(item.get("player_id") or "").strip()
                if not target_id or target_id not in valid_poison_targets:
                    continue
                if int(item.get("score", 0)) >= 4 and (int(item.get("claim_conflict", 0)) >= 1 or int(item.get("vote_pressure", 0)) >= 2 or int(item.get("sheriff_votes", 0)) >= 1):
                    planned = {"request_id": request["request_id"], "player_id": request["player_id"], "kind": str(poison_action.get("kind") or "")}
                    planned["target_id"] = target_id
                    return planned
        return None

    def _ranked_witch_targets(
        self,
        *,
        candidates: set[str],
        target_kind: str,
        turn_packet: Mapping[str, Any],
        witch_context: Mapping[str, Any] | None = None,
        ledger_snapshot: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        ledger_snapshot = ledger_snapshot if ledger_snapshot is not None else self._build_witch_ledger_snapshot()
        candidate_set = {str(candidate).strip() for candidate in candidates if str(candidate).strip()}
        if witch_context:
            for key in ("night_death_targets", "high_value_rescue_targets", "high_confidence_targets"):
                for player_id in witch_context.get(key, []):
                    player_text = str(player_id).strip()
                    if player_text:
                        candidate_set.add(player_text)
        for item in ledger_snapshot:
            player_id = str(item.get("player_id") or "").strip()
            if player_id:
                candidate_set.add(player_id)
        if not candidate_set:
            return []

        ranked: list[dict[str, Any]] = []
        ledger_map = {str(item.get("player_id") or "").strip(): item for item in ledger_snapshot if str(item.get("player_id") or "").strip()}
        public_evidence = self._witch_public_evidence
        for player_id in candidate_set:
            payload = self._witch_suspicion_ledger.get(player_id, {})
            evidence = ledger_map.get(player_id, payload)
            score, reasons = self._score_witch_entry(payload)
            record = {
                "player_id": player_id,
                "score": score,
                "reasons": reasons,
                "claim_conflict": int(evidence.get("claim_conflict", 0)) if isinstance(evidence, Mapping) else 0,
                "vote_pressure": int(evidence.get("vote_pressure", 0)) if isinstance(evidence, Mapping) else 0,
                "death_chain": int(evidence.get("death_chain", 0)) if isinstance(evidence, Mapping) else 0,
                "role_chain": int(evidence.get("role_chain", 0)) if isinstance(evidence, Mapping) else 0,
                "sheriff_votes": int(evidence.get("sheriff_votes", 0)) if isinstance(evidence, Mapping) else 0,
            }
            if target_kind == "rescue":
                rescue_score = self._score_rescue_candidate(player_id, record, turn_packet, public_evidence)
                record["score"] = rescue_score
                if rescue_score > 0:
                    record["reasons"] = self._score_reasons_for_rescue(player_id, record, turn_packet, public_evidence)
            else:
                poison_score = self._score_poison_candidate(player_id, record, turn_packet, public_evidence)
                record["score"] = poison_score
                if poison_score > 0:
                    record["reasons"] = self._score_reasons_for_poison(player_id, record, turn_packet, public_evidence)
            ranked.append(record)
        ranked.sort(key=lambda item: (-int(item.get("score", 0)), str(item.get("player_id", ""))))
        return ranked[:5]

    def _collect_witch_candidates(self, turn_packet: Mapping[str, Any], keywords: tuple[str, ...]) -> set[str]:
        texts: list[str] = []
        for value in (
            self._witch_public_evidence,
            self._recent_public_summaries[-4:],
            self._last_public_round_phase,
            self._last_private_resource_summary,
            turn_packet.get("public_state"),
            turn_packet.get("game"),
            turn_packet.get("request"),
            turn_packet.get("tool_context"),
        ):
            if value is not None:
                texts.append(self._safe_json_dump(value))
        combined = " \n ".join(texts)
        return self._extract_targets_from_text(combined, keywords)

    def _build_witch_public_evidence_snapshot(self) -> dict[str, Any]:
        claims = []
        for player_id, payload in self._witch_public_evidence.get("claims", {}).items():
            claims.append({"player_id": player_id, "count": len(payload.get("items", [])), "roles": payload.get("roles", [])[-2:], "targets": payload.get("targets", [])[-2:]})
        reveals = []
        for player_id, payload in self._witch_public_evidence.get("reveals", {}).items():
            reveals.append({"player_id": player_id, "count": len(payload.get("items", [])), "roles": payload.get("roles", [])[-2:], "flags": payload.get("flags", [])[-2:]})
        votes = []
        for target_id, payload in self._witch_public_evidence.get("votes", {}).items():
            votes.append({"target_id": target_id, "count": int(payload.get("count", 0)), "sheriff_votes": int(payload.get("sheriff_votes", 0)), "voters": list(payload.get("voters", []))[-3:]})
        votes.sort(key=lambda item: (-int(item.get("count", 0)), str(item.get("target_id", ""))))
        contradictions = list(self._witch_public_evidence.get("contradictions", []))[-4:]
        deaths = list(self._witch_public_evidence.get("deaths", []))[-4:]
        return {
            "claims": claims[:4],
            "reveals": reveals[:4],
            "votes": votes[:4],
            "deaths": deaths,
            "contradictions": contradictions,
        }

    def _register_witch_claim(self, record: Mapping[str, Any]) -> None:
        subject = str(record.get("speaker") or record.get("player_id") or "").strip()
        if not subject:
            return
        entry = self._witch_suspicion_ledger.setdefault(subject, self._empty_witch_entry())
        claim_bucket = self._witch_public_evidence.setdefault("claims", {}).setdefault(subject, {"items": [], "roles": [], "targets": []})
        claim_bucket["items"].append(self._compact_record(record))
        if record.get("role"):
            claim_bucket.setdefault("roles", []).append(str(record.get("role")))
            entry["claim_conflict"] = int(entry.get("claim_conflict", 0)) + (2 if record.get("hard_conflict") else 1)
        target = str(record.get("target") or "").strip()
        if target:
            claim_bucket.setdefault("targets", []).append(target)
        if record.get("hard_conflict"):
            self._register_witch_contradiction(record)
        entry["score"] = int(entry.get("score", 0)) + (2 if record.get("hard_conflict") else 1)

    def _register_witch_reveal(self, record: Mapping[str, Any]) -> None:
        subject = str(record.get("speaker") or record.get("player_id") or "").strip()
        if not subject:
            return
        entry = self._witch_suspicion_ledger.setdefault(subject, self._empty_witch_entry())
        reveal_bucket = self._witch_public_evidence.setdefault("reveals", {}).setdefault(subject, {"items": [], "roles": [], "flags": []})
        reveal_bucket["items"].append(self._compact_record(record))
        role_text = str(record.get("role") or "").strip()
        if role_text:
            reveal_bucket.setdefault("roles", []).append(role_text)
        flags = reveal_bucket.setdefault("flags", [])
        if record.get("sheriff"):
            flags.append("sheriff")
            entry["role_chain"] = int(entry.get("role_chain", 0)) + 2
            entry["sheriff_votes"] = int(entry.get("sheriff_votes", 0)) + 1
        else:
            entry["role_chain"] = int(entry.get("role_chain", 0)) + 1
        entry["score"] = int(entry.get("score", 0)) + (2 if record.get("sheriff") else 1)

    def _register_witch_vote(self, record: Mapping[str, Any]) -> None:
        voter = str(record.get("speaker") or record.get("player_id") or "").strip()
        target = str(record.get("target") or "").strip()
        if not target and not voter:
            return
        target_bucket = self._witch_public_evidence.setdefault("votes", {}).setdefault(target or voter, {"count": 0, "voters": [], "sheriff_votes": 0, "items": []})
        target_bucket["count"] = int(target_bucket.get("count", 0)) + 1
        if voter:
            voters = target_bucket.setdefault("voters", [])
            if voter not in voters:
                voters.append(voter)
        if record.get("sheriff"):
            target_bucket["sheriff_votes"] = int(target_bucket.get("sheriff_votes", 0)) + 1
        target_bucket.setdefault("items", []).append(self._compact_record(record))
        entry = self._witch_suspicion_ledger.setdefault(target or voter, self._empty_witch_entry())
        entry["vote_pressure"] = int(entry.get("vote_pressure", 0)) + (2 if record.get("sheriff") else 1)
        entry["score"] = int(entry.get("score", 0)) + (2 if record.get("sheriff") else 1)

    def _register_witch_death(self, record: Mapping[str, Any]) -> None:
        target = str(record.get("target") or record.get("speaker") or record.get("player_id") or "").strip()
        if not target:
            return
        entry = self._witch_suspicion_ledger.setdefault(target, self._empty_witch_entry())
        payload = {
            "player_id": target,
            "round": record.get("round_tag", ""),
            "summary": self._compact_record(record),
        }
        deaths = self._witch_public_evidence.setdefault("deaths", [])
        if not deaths or deaths[-1] != payload:
            deaths.append(payload)
            del deaths[:-6]
        entry["death_chain"] = int(entry.get("death_chain", 0)) + 1
        entry["score"] = int(entry.get("score", 0)) + 1

    def _register_witch_contradiction(self, record: Mapping[str, Any]) -> None:
        source = str(record.get("speaker") or record.get("player_id") or "").strip()
        target = str(record.get("target") or "").strip()
        payload = {
            "source": source,
            "target": target,
            "round": record.get("round_tag", ""),
            "summary": self._compact_record(record),
        }
        contradictions = self._witch_public_evidence.setdefault("contradictions", [])
        if not contradictions or contradictions[-1] != payload:
            contradictions.append(payload)
            del contradictions[:-6]
        for player_id in (source, target):
            if not player_id:
                continue
            entry = self._witch_suspicion_ledger.setdefault(player_id, self._empty_witch_entry())
            entry["claim_conflict"] = int(entry.get("claim_conflict", 0)) + 2
            entry["score"] = int(entry.get("score", 0)) + 2

    def _extract_witch_structured_records(self, node: Any, round_tag: str) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []

        def walk(value: Any) -> None:
            if isinstance(value, Mapping):
                record = self._classify_witch_record(value, round_tag)
                if record is not None:
                    records.append(record)
                for item in value.values():
                    walk(item)
            elif isinstance(value, list):
                for item in value:
                    walk(item)

        walk(node)
        return records

    def _classify_witch_record(self, mapping: Mapping[str, Any], round_tag: str) -> dict[str, Any] | None:
        serialized = self._safe_json_dump(mapping)
        lowered = serialized.lower()
        players = self._extract_player_ids(serialized)
        if not players and not any(marker in lowered for marker in ("claim", "vote", "reveal", "sheriff", "seer", "death", "kill", "dead", "死亡", "投票", "查验", "警长", "悍跳", "对跳", "查杀", "金水")):
            return None

        speaker = self._pick_player_from_mapping(mapping, ("speaker", "player", "voter", "source", "from", "author", "claimer", "actor"))
        target = self._pick_player_from_mapping(mapping, ("target", "victim", "dead", "died", "accused", "vote_target", "lynch"))
        if not speaker and players:
            speaker = players[0]
        if not target and len(players) > 1:
            target = players[1]
        if speaker and target == speaker and len(players) > 1:
            target = players[1]

        kind = ""
        if any(marker in lowered for marker in ("death", "night kill", "wolf kill", "dead", "died", "被刀", "死亡", "出局", "淘汰", "刀口")):
            kind = "death"
        elif any(marker in lowered for marker in ("vote", "ballot", "归票", "投票", "票型")):
            kind = "vote"
        elif any(marker in lowered for marker in ("reveal", "revealed", "sheriff", "seer", "警长", "警徽", "验出", "查验", "确认")):
            kind = "reveal"
        elif any(marker in lowered for marker in ("claim", "counterclaim", "悍跳", "对跳", "查杀", "金水", "自称")):
            kind = "claim"
        elif any(marker in lowered for marker in ("contradiction", "冲突", "conflict", "对撞")):
            kind = "contradiction"
        if not kind:
            return None

        flags = {
            "hard_conflict": any(marker in lowered for marker in ("counterclaim", "悍跳", "对跳", "冲突", "conflict", "对撞", "查杀")),
            "sheriff": any(marker in lowered for marker in ("sheriff", "警长", "警徽")),
            "role_reveal": any(marker in lowered for marker in ("reveal", "seer", "sheriff", "警长", "警徽", "验出", "查验")),
        }
        role = self._pick_role_from_text(serialized)
        summary = self._truncate_text(self._squash_whitespace(serialized), 220)
        return {
            "kind": kind,
            "speaker": speaker or "",
            "target": target or "",
            "players": players[:3],
            "role": role,
            "summary": summary,
            "round_tag": round_tag,
            **flags,
        }

    def _pick_player_from_mapping(self, mapping: Mapping[str, Any], key_hints: tuple[str, ...]) -> str:
        for key, value in mapping.items():
            key_lower = str(key).lower()
            if not any(hint in key_lower for hint in key_hints):
                continue
            ids = self._extract_player_ids(self._safe_json_dump(value))
            if ids:
                return ids[0]
            if isinstance(value, str):
                text = value.strip()
                if _PLAYER_ID_PATTERN.fullmatch(text):
                    return text
        return ""

    @staticmethod
    def _pick_role_from_text(text: str) -> str:
        lowered = text.lower()
        for marker in ("seer", "sheriff", "witch", "狼人", "女巫", "预言家", "警长", "守卫", "民", "狼"):
            if marker.lower() in lowered:
                return marker
        return ""

    def _compact_record(self, record: Mapping[str, Any]) -> str:
        return self._truncate_text(self._safe_json_dump(dict(record)), 220)

    def _empty_witch_entry(self) -> dict[str, Any]:
        return {
            "score": 0,
            "vote_pressure": 0,
            "claim_conflict": 0,
            "death_chain": 0,
            "role_chain": 0,
            "sheriff_votes": 0,
            "notes": [],
            "claims": [],
            "reveals": [],
            "votes": [],
            "deaths": [],
            "contradictions": [],
            "last_round": "",
        }

    def _score_witch_entry(self, payload: Mapping[str, Any]) -> tuple[int, list[str]]:
        score = int(payload.get("score", 0))
        score += int(payload.get("vote_pressure", 0)) * 2
        score += int(payload.get("claim_conflict", 0)) * 2
        score += int(payload.get("death_chain", 0))
        score += int(payload.get("role_chain", 0))
        score += int(payload.get("sheriff_votes", 0)) * 2
        reasons: list[str] = []
        if int(payload.get("vote_pressure", 0)):
            reasons.append(f"vote_pressure={int(payload.get('vote_pressure', 0))}")
        if int(payload.get("claim_conflict", 0)):
            reasons.append(f"claim_conflict={int(payload.get('claim_conflict', 0))}")
        if int(payload.get("death_chain", 0)):
            reasons.append(f"death_chain={int(payload.get('death_chain', 0))}")
        if int(payload.get("role_chain", 0)):
            reasons.append(f"role_chain={int(payload.get('role_chain', 0))}")
        if int(payload.get("sheriff_votes", 0)):
            reasons.append(f"sheriff_votes={int(payload.get('sheriff_votes', 0))}")
        return score, reasons

    def _score_rescue_candidate(
        self,
        player_id: str,
        record: Mapping[str, Any],
        turn_packet: Mapping[str, Any],
        public_evidence: Mapping[str, Any],
    ) -> int:
        score = 0
        death_targets = {str(item).strip() for item in self._extract_targets_from_text(self._safe_json_dump(turn_packet), ("night_death", "death", "wolf", "kill", "刀口", "被刀", "死亡", "出局", "淘汰"))}
        if player_id in death_targets:
            score += 4
        if player_id == str(turn_packet.get("self", {}).get("player_id") or self.player_id):
            if player_id in death_targets:
                score += 2
        if int(record.get("role_chain", 0)):
            score += 2
        if int(record.get("death_chain", 0)):
            score += 1
        if self._is_key_public_chain(player_id, public_evidence):
            score += 2
        return score

    def _score_poison_candidate(
        self,
        player_id: str,
        record: Mapping[str, Any],
        turn_packet: Mapping[str, Any],
        public_evidence: Mapping[str, Any],
    ) -> int:
        score = 0
        vote_count = self._vote_pressure_count(player_id, public_evidence)
        if vote_count >= 2:
            score += 3
        if int(record.get("claim_conflict", 0)) >= 2:
            score += 4
        elif int(record.get("claim_conflict", 0)) >= 1:
            score += 2
        if int(record.get("sheriff_votes", 0)) >= 1:
            score += 2
        if int(record.get("death_chain", 0)) >= 1 and int(record.get("claim_conflict", 0)) >= 1:
            score += 2
        if self._has_direct_public_conflict(player_id, public_evidence):
            score += 2
        return score

    def _score_reasons_for_rescue(self, player_id: str, record: Mapping[str, Any], turn_packet: Mapping[str, Any], public_evidence: Mapping[str, Any]) -> list[str]:
        reasons: list[str] = []
        if player_id == str(turn_packet.get("self", {}).get("player_id") or self.player_id):
            reasons.append("self_save")
        if int(record.get("role_chain", 0)):
            reasons.append("role_chain")
        if self._is_key_public_chain(player_id, public_evidence):
            reasons.append("key_public_chain")
        if int(record.get("death_chain", 0)):
            reasons.append("death_chain")
        return reasons

    def _score_reasons_for_poison(self, player_id: str, record: Mapping[str, Any], turn_packet: Mapping[str, Any], public_evidence: Mapping[str, Any]) -> list[str]:
        reasons: list[str] = []
        if int(record.get("claim_conflict", 0)) >= 2:
            reasons.append("hard_conflict")
        if self._vote_pressure_count(player_id, public_evidence) >= 2:
            reasons.append("vote_convergence")
        if int(record.get("sheriff_votes", 0)) >= 1:
            reasons.append("sheriff_pressure")
        if self._has_direct_public_conflict(player_id, public_evidence):
            reasons.append("public_contradiction")
        return reasons

    def _vote_pressure_count(self, player_id: str, public_evidence: Mapping[str, Any]) -> int:
        votes = public_evidence.get("votes", {}) if isinstance(public_evidence, Mapping) else {}
        if not isinstance(votes, Mapping):
            return 0
        payload = votes.get(player_id)
        if not isinstance(payload, Mapping):
            return 0
        return int(payload.get("count", 0))

    def _has_direct_public_conflict(self, player_id: str, public_evidence: Mapping[str, Any]) -> bool:
        contradictions = public_evidence.get("contradictions", []) if isinstance(public_evidence, Mapping) else []
        if not isinstance(contradictions, list):
            return False
        for item in contradictions:
            if not isinstance(item, Mapping):
                continue
            if player_id and player_id in {str(item.get("source") or ""), str(item.get("target") or "")}:
                return True
        return False

    def _is_key_public_chain(self, player_id: str, public_evidence: Mapping[str, Any]) -> bool:
        reveals = public_evidence.get("reveals", {}) if isinstance(public_evidence, Mapping) else {}
        if not isinstance(reveals, Mapping):
            return False
        payload = reveals.get(player_id)
        if not isinstance(payload, Mapping):
            return False
        roles = [str(role).lower() for role in payload.get("roles", []) if str(role).strip()]
        return any(role in {"sheriff", "seer", "警长", "预言家"} for role in roles)

    def _extract_targets_from_text(self, text: str, keywords: tuple[str, ...]) -> set[str]:
        snippet = self._extract_keyword_snippet(text, keywords)
        if not snippet:
            return set()
        return set(self._extract_player_ids(snippet))

    def _reset_compact_memory(self, game_id: str | None) -> None:
        self._memory_game_id = game_id
        self._recent_public_summaries = []
        self._last_public_round_phase = ""
        self._last_private_resource_summary = ""
        self._last_decision_audit = None
        self._witch_suspicion_ledger = {}
        self._witch_recent_public_deaths = []
        self._witch_recent_vote_pressure = []
        self._witch_recent_counterclaims = []
        self._witch_recent_role_chain = []
        self._witch_public_evidence = {
            "claims": {},
            "reveals": {},
            "votes": {},
            "deaths": [],
            "contradictions": [],
            "sheriff_votes": {},
        }

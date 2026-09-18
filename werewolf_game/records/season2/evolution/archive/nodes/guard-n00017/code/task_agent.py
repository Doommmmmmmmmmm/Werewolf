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
        prompt = {
            "game": turn_packet["game"],
            "public_rules": turn_packet["public_rules"],
            "self": turn_packet["self"],
            "private_information": private,
            "public_state": turn_packet["public_state"],
            "request": turn_packet["request"],
            "guard_memory": self._guard_memory_snapshot(turn_packet),
            "history_policy": {
                "default_context": "current_state_only",
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

    def _turn_instruction(self, request: Mapping[str, Any], feedback: str) -> str:
        instruction = render_prompt(
            "player_turn_instruction.txt",
            action_contract=render_action_contract(request),
            validation_feedback=(f"上一次输出未通过校验：{feedback}" if feedback else ""),
        )
        extra_notes: list[str] = []
        if self.profile.role == _GUARD_ROLE_NAME:
            extra_notes.append(
                "守卫专用：最终 target_id 只能从本次 contract 的 target_ids 中选择；"
                "不要因为警长身份自动守护；可信查杀目标、狼嫌警长和上一夜守过的人要降权；"
                "不确定时选择当前合法的高价值目标，不要臆造目标。"
            )
            if self.self_publicly_claimed_guard:
                extra_notes.append(
                    "你已经公开跳过守卫时，后续不要再把自己说成平民或无夜间信息；"
                    "可以说你只知道自己守过谁，但不能确认是否守中刀。"
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
        request = turn_packet.get("request") if isinstance(turn_packet, Mapping) else {}
        public_state = turn_packet.get("public_state") if isinstance(turn_packet, Mapping) else {}
        sheriff_id = self._extract_first_player_id(
            self._first_present(
                public_state,
                "sheriff_id",
                "current_sheriff_id",
                "police_id",
                "badge_owner_id",
            )
        )
        current_round = self._current_round_value(turn_packet)
        candidate_scores: list[tuple[int, int, str]] = []
        for index, candidate in enumerate(candidates):
            score = 0
            labels = self.public_claims.get(candidate, set())
            if candidate == self.last_protected_id and current_round is not None and self.last_protected_round == current_round:
                score -= 200
            elif candidate == self.last_protected_id:
                score -= 80
            if candidate in self.public_dead_ids:
                score -= 200
            if self.public_alive_ids and candidate not in self.public_alive_ids:
                score -= 60
            if candidate in self.suspected_wolves:
                score -= 140
            if "public_check_kill" in labels or "sheriff_from_suspect" in labels:
                score -= 110
            if candidate == sheriff_id:
                score += 25
                if candidate in self.suspected_wolves or "sheriff_from_suspect" in labels:
                    score -= 90
            if candidate in self.trusted_info_ids:
                score += 120
            if "seer_claim" in labels or "check_kill" in labels or "check_clear" in labels:
                score += 65
            if "witch_claim" in labels or "hunter_claim" in labels:
                score += 45
            if candidate == self.player_id:
                score += 18
            if "guard_claim" in labels:
                score += 8
            if "sheriff_transfer" in labels:
                score -= 35
            if candidate in self._alive_id_set(public_state):
                score += 0
            candidate_scores.append((score, index, candidate))
        candidate_scores.sort(key=lambda item: (-item[0], item[1]))
        return [candidate for _, _, candidate in candidate_scores]

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
            if speaker == self.player_id:
                self.self_publicly_claimed_guard = True
        if "我是预言家" in cleaned or ("预言家" in cleaned and ("查杀" in cleaned or "金水" in cleaned or "验" in cleaned)):
            self._note_public_claim(speaker, "seer_claim")
        if "我是女巫" in cleaned or ("女巫" in cleaned and "我是" in cleaned):
            self._note_public_claim(speaker, "witch_claim")
        if "我是猎人" in cleaned or ("猎人" in cleaned and "我是" in cleaned):
            self._note_public_claim(speaker, "hunter_claim")
        if "警长" in cleaned or "警徽" in cleaned:
            self._note_public_claim(speaker, "sheriff_related")
        if "移交" in cleaned and "警徽" in cleaned:
            self._note_public_claim(speaker, "sheriff_transfer")
            if speaker in self.suspected_wolves:
                for pid in ids_in_text:
                    if pid != speaker:
                        self._note_public_claim(pid, "sheriff_from_suspect")
        if "查杀" in cleaned:
            self._note_public_claim(speaker, "check_kill")
            if self._looks_like_info_provider(cleaned, speaker):
                self.trusted_info_ids.add(speaker)
            for pid in ids_in_text:
                if pid != speaker:
                    self.suspected_wolves.add(pid)
                    self._note_public_claim(pid, "public_check_kill")
        if "金水" in cleaned:
            self._note_public_claim(speaker, "check_clear")
            if self._looks_like_info_provider(cleaned, speaker):
                self.trusted_info_ids.add(speaker)
            for pid in ids_in_text:
                if pid != speaker:
                    self._note_public_claim(pid, "public_check_clear")
        if "对跳" in cleaned and "守卫" in cleaned:
            self._note_public_claim(speaker, "guard_counterclaim")
        if speaker and speaker == self.player_id and "守卫" in cleaned:
            self.self_publicly_claimed_guard = True

    def _looks_like_info_provider(self, text: str, speaker: str | None) -> bool:
        if speaker is None:
            return False
        if "预言家" in text or "验" in text:
            return True
        labels = self.public_claims.get(speaker, set())
        return "seer_claim" in labels

    def _note_public_claim(self, player_id: str | None, label: str) -> None:
        normalized = self._normalize_player_id(player_id)
        if not normalized or not label:
            return
        labels = self.public_claims.setdefault(normalized, set())
        labels.add(label)

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

    def _compact_public_claims(self, *, max_players: int) -> dict[str, list[str]]:
        compact: dict[str, list[str]] = {}
        for player_id in sorted(self.public_claims):
            labels = sorted(self.public_claims[player_id])
            compact[player_id] = labels[:4]
            if len(compact) >= max_players:
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

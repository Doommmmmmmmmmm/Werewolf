"""最小可运行的角色 Task-Agent。

每次行动从一个新的模型会话开始。模型默认只收到当前状态；本轮对话只能通过唯一的
受限工具按需读取。这里不包含策略进化、长期记忆、外部检索或其他 Harness。
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from collections import deque
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
_GUARD_ACTION_KIND = "guard_protect"
_PLAYER_ID_PATTERN = re.compile(r"\bp\d+\b")
_GUARD_CONTEXT_KEYWORDS = (
    "sheriff",
    "警长",
    "focus",
    "vote",
    "suspect",
    "怀疑",
    "公开",
    "public",
    "claim",
    "claimed",
    "target",
    "dialogue",
    "刀口",
    "站边",
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
        self.last_protected_id: str | None = None
        self.last_protected_round: int | None = None
        self.alive_player_cache: set[str] = set()
        self.recent_public_focus_ids: deque[str] = deque(maxlen=8)
        self.player_risk_profile: dict[str, dict[str, int]] = {}
        self._last_observed_round: int | None = None

    async def observe(self, sync_packet: dict[str, Any]) -> None:
        """接收公开同步，并提取最小守卫记忆。"""

        self._update_guard_memory(sync_packet)

    async def decide(self, turn_packet: dict[str, Any]) -> dict[str, Any]:
        if turn_packet["self"]["player_id"] != self.player_id:
            raise ValueError("行动包被发送给了错误玩家")
        private = turn_packet["private_information"]
        if str(private.get("role")) != self.profile.role:
            raise ValueError("Task-Agent 的角色档案与行动包身份不一致")

        self._update_guard_memory(turn_packet.get("public_state") or {})
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
        if current_dialogue:
            self._update_guard_memory({"current_round_dialogue": current_dialogue})

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
            action = self._repair_guard_decision(action, turn_packet["request"])
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
        role_task = self.profile.task
        if self.profile.role == "guard":
            role_task = role_task + (
                "\n\n守卫约束：只从当前 target_ids 里选；先判定今晚最可能被狼刀的好人；"
                "默认不要重复上一夜目标；优先保护公开神职、被可信来源背书的好人，而不是单纯热度位；"
                "白天默认不要主动跳守卫、不要报昨晚守护对象、不要把平安夜当作守中证明；"
                "只有濒临被放逐、出现假守卫对跳，或必须挽救关键好人链时才有限度亮身份。"
            )
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

    def _turn_instruction(self, request: Mapping[str, Any], feedback: str) -> str:
        validation_feedback = f"上一次输出未通过校验：{feedback}" if feedback else ""
        if self.profile.role == "guard":
            guard_note = (
                "守卫执行要点：只在当前 target_ids 内选；先找今晚最可能被刀的好人；"
                "优先保护公开神职、被可信来源背书的好人，而不是单纯热度位；没有更好目标时再自守。"
                "白天默认不要主动跳守卫、不要报昨晚守护对象、不要把平安夜说成守中证明；"
                "只有濒临被放逐、出现假守卫对跳，或必须挽救关键好人链时才有限度亮身份。"
            )
            validation_feedback = guard_note + ("\n" + validation_feedback if validation_feedback else "")
        return render_prompt(
            "player_turn_instruction.txt",
            action_contract=render_action_contract(request),
            validation_feedback=validation_feedback,
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

    def _record_successful_guard_action(self, action: Mapping[str, Any], packet: Mapping[str, Any]) -> None:
        if self.profile.role != "guard" or action.get("kind") != _GUARD_ACTION_KIND:
            return
        target_id = action.get("target_id")
        if not target_id:
            return
        self.last_protected_id = str(target_id)
        round_no = self._extract_round_number(packet)
        if round_no is None and isinstance(packet.get("game"), Mapping):
            round_no = self._extract_round_number(packet["game"])
        if round_no is not None:
            self.last_protected_round = round_no

    def _repair_guard_decision(self, action: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
        if self.profile.role != "guard" or action.get("kind") != _GUARD_ACTION_KIND:
            return action
        allowed = next(
            (
                item
                for item in request.get("allowed_actions", [])
                if isinstance(item, Mapping) and item.get("kind") == _GUARD_ACTION_KIND
            ),
            None,
        )
        if not isinstance(allowed, Mapping):
            return action
        metrics = self._guard_candidate_metrics(allowed, request)
        if not metrics:
            return action
        ordered_candidates = self._guard_candidate_targets(allowed, request)
        target_id = str(action.get("target_id") or "")
        if not any(protect_score > 0 for _, _, protect_score, _ in metrics):
            if target_id != (ordered_candidates[0] if ordered_candidates else target_id):
                repaired = dict(action)
                repaired["target_id"] = ordered_candidates[0]
                return repaired
            return action
        candidates = [candidate for candidate, _, _, _ in metrics]
        if target_id not in candidates:
            repaired = dict(action)
            repaired["target_id"] = candidates[0]
            return repaired
        chosen = next((item for item in metrics if item[0] == target_id), None)
        best = metrics[0]
        if chosen is not None and (candidates.index(target_id) > 1 or best[1] - chosen[1] >= 4):
            repaired = dict(action)
            repaired["target_id"] = best[0]
            return repaired
        return action

    def _guard_candidate_targets(self, allowed: Mapping[str, Any], request: Mapping[str, Any]) -> list[str]:
        metrics = self._guard_candidate_metrics(allowed, request)
        if not metrics:
            return []
        ordered = [candidate for candidate, _, _, _ in metrics]
        repeat_forbidden = self._guard_repeat_forbidden(allowed)
        if repeat_forbidden and self.last_protected_id:
            filtered = [candidate for candidate in ordered if candidate != self.last_protected_id]
            if filtered:
                ordered = filtered
        if not any(protect_score > 0 for _, _, protect_score, _ in metrics):
            if self.player_id in ordered:
                ordered = [self.player_id] + [candidate for candidate in ordered if candidate != self.player_id]
        return ordered

    def _guard_repeat_forbidden(self, allowed: Mapping[str, Any]) -> bool:
        if allowed.get("guard_can_repeat_protect") is True:
            return False
        if allowed.get("guard_can_repeat_protect") is False:
            return True
        return True

    def _guard_candidate_metrics(
        self, allowed: Mapping[str, Any], request: Mapping[str, Any]
    ) -> list[tuple[str, int, int, int]]:
        target_ids = allowed.get("target_ids")
        if not isinstance(target_ids, list):
            return []
        candidates: list[str] = []
        for target_id in target_ids:
            candidate = str(target_id)
            if candidate and candidate not in candidates:
                candidates.append(candidate)
        if not candidates:
            return []
        round_no = self._extract_round_number(request) or self._last_observed_round
        repeat_forbidden = self._guard_repeat_forbidden(allowed)
        scored: list[tuple[str, int, int, int]] = []
        for index, candidate in enumerate(candidates):
            protect_score, suspicion_penalty = self._player_guard_score_components(candidate, request, round_no)
            final_score = protect_score - suspicion_penalty
            if repeat_forbidden and self.last_protected_id and candidate == self.last_protected_id:
                final_score -= 5_000
            if self.last_protected_round is not None and round_no is not None and candidate == self.last_protected_id:
                final_score -= max(0, 200 - max(0, round_no - self.last_protected_round))
            scored.append((candidate, final_score, protect_score, index))
        scored.sort(key=lambda item: (-item[1], -item[2], item[3]))
        return scored

    def _looks_publicly_salient(self, candidate: str, request: Mapping[str, Any]) -> bool:
        return candidate in self._extract_focus_ids(request.get("public_state"))

    def _update_guard_memory(self, packet: Mapping[str, Any] | Any) -> None:
        if not isinstance(packet, Mapping):
            return
        round_no = self._extract_round_number(packet)
        if round_no is not None:
            self._last_observed_round = round_no
        alive_ids = self._extract_ids_by_keys(
            packet,
            ("alive", "alive_player", "alive_players", "alive_player_ids", "player_ids", "players"),
        )
        if alive_ids:
            self.alive_player_cache.update(alive_ids)
        focus_ids = self._extract_focus_ids(packet)
        for focus_id in focus_ids:
            if focus_id not in self.recent_public_focus_ids:
                self.recent_public_focus_ids.append(focus_id)
        self._update_guard_risk_memory(packet, round_no)

    def _update_guard_risk_memory(self, packet: Mapping[str, Any], round_no: int | None) -> None:
        self._scan_guard_signal_tree(packet, round_no)

    def _scan_guard_signal_tree(self, node: Mapping[str, Any] | list[Any] | Any, round_no: int | None) -> None:
        if isinstance(node, Mapping):
            for key, value in node.items():
                key_text = str(key).lower()
                if key_text in {"current_round_dialogue", "dialogue", "dialogues", "chat", "speech", "speeches", "transcript"}:
                    self._ingest_dialogue_entries(value, round_no)
                    continue
                signal = self._signal_for_key(key_text)
                if signal is not None:
                    signal_name, signal_weight = signal
                    for player_id in self._extract_player_ids(value):
                        self._bump_player_risk(player_id, signal_name, signal_weight, round_no)
                if isinstance(value, Mapping):
                    self._scan_guard_signal_tree(value, round_no)
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, (Mapping, list)):
                            self._scan_guard_signal_tree(item, round_no)
            return
        if isinstance(node, list):
            for item in node:
                if isinstance(item, (Mapping, list)):
                    self._scan_guard_signal_tree(item, round_no)

    def _signal_for_key(self, key_text: str) -> tuple[str, int] | None:
        vote_keys = ("vote", "voted", "投票", "票型", "归票", "票数")
        claim_keys = ("claim", "claimed", "身份", "报身份", "自报", "跳预", "跳女巫", "跳守卫", "跳猎人", "跳神", "神职")
        threat_keys = ("kill", "attack", "夜刀", "刀口", "击杀", "夜里要刀", "今晚刀")
        pressure_keys = ("suspect", "suspected", "怀疑", "pressure", "point", "点名", "push", "压票", "站边", "focus", "关注", "指认")
        exposure_keys = ("public", "公开", "center", "核心", "关键", "hot", "热度")
        if any(token in key_text for token in vote_keys):
            return ("vote_pressure", 4)
        if any(token in key_text for token in claim_keys):
            return ("power_claim", 5)
        if any(token in key_text for token in threat_keys):
            return ("threat_to_good", 4)
        if any(token in key_text for token in pressure_keys):
            return ("execution_pressure", 3)
        if any(token in key_text for token in exposure_keys):
            return ("exposure", 2)
        return None

    def _ingest_dialogue_entries(self, dialogue: Any, round_no: int | None) -> None:
        entries = dialogue if isinstance(dialogue, list) else [dialogue]
        for entry in entries:
            speaker_id = self._dialogue_entry_speaker_id(entry)
            text = self._dialogue_entry_text(entry)
            mentioned_ids = set(self._extract_player_ids(entry))
            if text:
                mentioned_ids.update(self._extract_player_ids(text))
            if speaker_id:
                mentioned_ids.discard(speaker_id)
            explicit_targets = self._dialogue_entry_targets(entry)
            if explicit_targets:
                mentioned_ids.update(explicit_targets)
            signals = self._dialogue_text_signals(text)
            if not signals and mentioned_ids:
                signals = {"execution_pressure": 1}
            if speaker_id and signals.get("power_claim"):
                self._bump_player_risk(speaker_id, "power_claim", signals["power_claim"], round_no)
            for player_id in mentioned_ids:
                for signal_name, signal_weight in signals.items():
                    if signal_name == "power_claim":
                        continue
                    self._bump_player_risk(player_id, signal_name, signal_weight, round_no)

    def _dialogue_entry_text(self, entry: Any) -> str:
        if isinstance(entry, str):
            return entry
        if isinstance(entry, Mapping):
            for key in ("text", "content", "message", "speech", "utterance", "dialogue"):
                value = entry.get(key)
                if isinstance(value, str) and value.strip():
                    return value
        return ""

    def _dialogue_entry_speaker_id(self, entry: Any) -> str | None:
        if not isinstance(entry, Mapping):
            return None
        for key in ("speaker_id", "speaker", "player_id", "from", "source"):
            value = entry.get(key)
            if isinstance(value, str) and value:
                return value
            if isinstance(value, Mapping):
                ids = self._extract_player_ids(value)
                if ids:
                    return next(iter(ids))
        return None

    def _dialogue_entry_targets(self, entry: Any) -> set[str]:
        if not isinstance(entry, Mapping):
            return set()
        targets: set[str] = set()
        for key in ("target_id", "target_ids", "vote_target", "vote_targets", "suspect_id", "suspect_ids", "focus_id", "focus_ids"):
            value = entry.get(key)
            if value is not None:
                targets.update(self._extract_player_ids(value))
        return targets

    def _dialogue_text_signals(self, text: str) -> dict[str, int]:
        if not text:
            return {}
        lowered = text.lower()
        signals: dict[str, int] = {}
        if any(token in lowered for token in ("我是预言家", "我是女巫", "我是守卫", "我是猎人", "我是白痴", "seer", "witch", "guard", "hunter", "claim", "报身份", "自报", "跳预", "跳女巫", "跳守卫", "跳猎人", "跳神", "神职", "预言家", "女巫", "守卫", "猎人", "白痴")):
            signals["power_claim"] = 5
        if "不是好人" not in lowered and "不是金水" not in lowered and "别保" not in lowered and "不要保" not in lowered and "不保" not in lowered and any(token in lowered for token in ("金水", "好人", "保", "trusted", "good")):
            signals["trusted_by_claim"] = 4
            signals["good_claim"] = 3
        if "不是狼" not in lowered and "非狼" not in lowered and any(token in lowered for token in ("查杀", "是狼", "wolf", "狼坑", "验", "出狼", "狼刀位")):
            signals["wolf_accused"] = 5
        if any(token in lowered for token in ("今天出", "票", "压", "归票", "vote", "出局", "点名")):
            signals["vote_pressure"] = 4
        if any(token in lowered for token in ("怀疑", "可疑", "pressure", "push", "focus", "关注", "指认", "站边")):
            signals["execution_pressure"] = 3
        if any(token in lowered for token in ("今晚刀", "夜刀", "刀口", "击杀", "attack", "kill", "被刀", "刀到")):
            signals["threat_to_good"] = 4
        if any(token in lowered for token in ("公开", "关键", "核心", "热度", "中心", "public")):
            signals["exposure"] = 2
        return signals

    def _player_risk_bucket(self, player_id: str) -> dict[str, int]:
        bucket = self.player_risk_profile.get(player_id)
        if bucket is None:
            bucket = {
                "power_claim": 0,
                "trusted_by_claim": 0,
                "good_claim": 0,
                "threat_to_good": 0,
                "wolf_accused": 0,
                "execution_pressure": 0,
                "vote_pressure": 0,
                "pressure": 0,
                "vote": 0,
                "claim": 0,
                "threat": 0,
                "exposure": 0,
                "last_round": -1,
            }
            self.player_risk_profile[player_id] = bucket
        return bucket

    def _bump_player_risk(self, player_id: str, signal_name: str, amount: int, round_no: int | None) -> None:
        if not player_id:
            return
        bucket = self._player_risk_bucket(player_id)
        bucket[signal_name] = bucket.get(signal_name, 0) + max(1, int(amount))
        if round_no is not None:
            bucket["last_round"] = max(bucket.get("last_round", -1), round_no)

    def _player_guard_score_components(
        self, candidate: str, request: Mapping[str, Any], round_no: int | None
    ) -> tuple[int, int]:
        del request, round_no
        bucket = self.player_risk_profile.get(candidate) or {}
        protect_score = 0
        protect_score += int(bucket.get("power_claim", 0)) * 6
        protect_score += int(bucket.get("trusted_by_claim", 0)) * 5
        protect_score += int(bucket.get("good_claim", 0)) * 3
        protect_score += int(bucket.get("threat_to_good", 0)) * 4
        suspicion_penalty = 0
        suspicion_penalty += int(bucket.get("wolf_accused", 0)) * 6
        suspicion_penalty += int(bucket.get("execution_pressure", 0)) * 4
        suspicion_penalty += int(bucket.get("vote_pressure", 0)) * 3
        suspicion_penalty += int(bucket.get("pressure", 0)) * 2
        suspicion_penalty += int(bucket.get("vote", 0)) * 2
        suspicion_penalty += int(bucket.get("claim", 0)) * 1
        suspicion_penalty += int(bucket.get("threat", 0)) * 1
        return protect_score, suspicion_penalty

    def _player_risk_score(self, candidate: str, request: Mapping[str, Any], round_no: int | None) -> int:
        protect_score, suspicion_penalty = self._player_guard_score_components(candidate, request, round_no)
        return protect_score - suspicion_penalty

    def _public_heat_score(self, candidate: str, request: Mapping[str, Any]) -> int:
        bucket = self.player_risk_profile.get(candidate) or {}
        score = 0
        if candidate in self.recent_public_focus_ids:
            score += 4
        if self._looks_publicly_salient(candidate, request):
            score += 3
        score += int(bucket.get("exposure", 0))
        score += int(bucket.get("pressure", 0))
        return score

    def _extract_round_number(self, packet: Mapping[str, Any] | Any) -> int | None:
        if not isinstance(packet, Mapping):
            return None
        for key in ("round", "round_no", "round_number"):
            value = packet.get(key)
            if isinstance(value, int) and value >= 0:
                return value
            if isinstance(value, str) and value.isdigit():
                return int(value)
        game = packet.get("game")
        if isinstance(game, Mapping):
            for key in ("round", "round_no", "round_number"):
                value = game.get(key)
                if isinstance(value, int) and value >= 0:
                    return value
                if isinstance(value, str) and value.isdigit():
                    return int(value)
        return None

    def _extract_focus_ids(self, packet: Mapping[str, Any] | Any) -> set[str]:
        focus_ids: set[str] = set()
        if not isinstance(packet, Mapping):
            return focus_ids
        for key, value in packet.items():
            key_text = str(key).lower()
            if any(keyword in key_text for keyword in _GUARD_CONTEXT_KEYWORDS):
                focus_ids.update(self._extract_player_ids(value))
            if isinstance(value, Mapping):
                focus_ids.update(self._extract_focus_ids(value))
            elif isinstance(value, list):
                for item in value:
                    focus_ids.update(self._extract_player_ids(item))
                    if isinstance(item, Mapping):
                        focus_ids.update(self._extract_focus_ids(item))
        return focus_ids

    def _extract_ids_by_keys(self, packet: Mapping[str, Any], keys: tuple[str, ...]) -> set[str]:
        values: set[str] = set()
        for key, value in packet.items():
            key_text = str(key).lower()
            if any(token in key_text for token in keys):
                values.update(self._extract_player_ids(value))
            if isinstance(value, Mapping):
                values.update(self._extract_ids_by_keys(value, keys))
            elif isinstance(value, list):
                for item in value:
                    values.update(self._extract_player_ids(item))
                    if isinstance(item, Mapping):
                        values.update(self._extract_ids_by_keys(item, keys))
        return values

    def _extract_player_ids(self, value: Any) -> set[str]:
        player_ids: set[str] = set()
        if isinstance(value, str):
            player_ids.update(_PLAYER_ID_PATTERN.findall(value))
            return player_ids
        if isinstance(value, Mapping):
            for inner in value.values():
                player_ids.update(self._extract_player_ids(inner))
            return player_ids
        if isinstance(value, list):
            for item in value:
                player_ids.update(self._extract_player_ids(item))
            return player_ids
        return player_ids

    @staticmethod
    def _nonnegative_int(value: object, *, fallback: int | None) -> int | None:
        if isinstance(value, bool):
            return fallback
        try:
            normalized = int(value)
        except (TypeError, ValueError):
            return fallback
        return normalized if normalized >= 0 else fallback

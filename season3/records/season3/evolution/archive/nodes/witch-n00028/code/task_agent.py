"""精简版女巫 Task-Agent。

每次决策只向模型提供当前必要状态、极短跨回合摘要和一个受限的当前轮对话工具。
不读取网络、文件系统、环境变量、进程或隐藏状态；所有行动仍以行动包 allowed_actions 为唯一合法来源。
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
DEFAULT_MAX_DECISION_RETRIES = 2
_SPEECH_ACTION_KINDS = frozenset({"speak", "last_words"})

CURRENT_ROUND_DIALOGUE_TOOL: dict[str, Any] = {
    "name": CURRENT_ROUND_DIALOGUE_TOOL_NAME,
    "description": "读取当前昼夜轮次中、当前玩家依法可见的已发生发言。",
    "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
}


def render_action_contract(request: Mapping[str, Any]) -> str:
    """把本次 allowed_actions 转成模型容易遵守的最终 JSON 契约。"""

    raw_actions = request.get("allowed_actions")
    allowed_actions = [item for item in raw_actions if isinstance(item, Mapping)] if isinstance(raw_actions, list) else []
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
        valid_targets = [str(t) for t in target_ids] if isinstance(target_ids, list) and target_ids else []
        if kind in _SPEECH_ACTION_KINDS:
            example["text"] = "好"
        elif valid_targets:
            example["target_id"] = valid_targets[0]
        lines.append("- `" + json.dumps(example, ensure_ascii=False, separators=(",", ":")) + "`")
        if valid_targets:
            lines.append("  target_id 候选值：" + "、".join(valid_targets) + "。")
        if kind in _SPEECH_ACTION_KINDS:
            constraints: list[str] = []
            if allowed.get("max_chars") is not None:
                constraints.append(f"text 最多 {allowed['max_chars']} 个字符")
            if allowed.get("require_chinese"):
                constraints.append("text 至少包含一个中文字符")
            constraints.append("text 可以包含英文、数字和玩家编号")
            lines.append("  " + "；".join(constraints) + "。")
    lines.append("kind 和 target_id 是协议字段，必须使用英文原值；不要翻译。")
    return "\n".join(lines)


def normalize_decision(raw: dict[str, Any] | None, request: dict[str, Any]) -> dict[str, Any]:
    source = raw.get("decision", raw) if isinstance(raw, dict) else {}
    action = {
        "request_id": request["request_id"],
        "player_id": request["player_id"],
        "kind": str(source.get("kind", source.get("action", ""))),
    }
    target_id = source.get("target_id", source.get("targetId"))
    if target_id is not None and str(target_id):
        action["target_id"] = str(target_id)
    if isinstance(source.get("text"), str):
        action["text"] = source["text"]
    return action


def decision_error(action: dict[str, Any], request: dict[str, Any]) -> str | None:
    allowed = next((item for item in request["allowed_actions"] if item["kind"] == action["kind"]), None)
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
    """女巫角色在一局中的精简任务求解器。"""

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
        self.max_prompt_chars = max(1, min(int(max_prompt_chars), 15000))
        self._model_token_usage: dict[str, int] = {
            "successful_response_count": 0,
            "api_attempt_count": 0,
            "reported_usage_response_count": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }
        self._current_game_id: str | None = None
        self._compact_history: list[dict[str, Any]] = []
        self._last_observe_signature = ""
        self._known_private_state: dict[str, Any] = {}
        self._last_witch_action: dict[str, Any] = {}

    def _extract_game_id(self, packet: Mapping[str, Any]) -> str | None:
        sources: list[Mapping[str, Any]] = [packet]
        for key in ("game", "public_state", "request", "private_information"):
            value = packet.get(key)
            if isinstance(value, Mapping):
                sources.append(value)
        for source in sources:
            value = source.get("game_id")
            if value is not None and str(value).strip():
                return str(value)
        return None

    def _ensure_game_context(self, packet: Mapping[str, Any]) -> None:
        game_id = self._extract_game_id(packet)
        if game_id is None or game_id == self._current_game_id:
            return
        self._current_game_id = game_id
        self._compact_history.clear()
        self._last_observe_signature = ""
        self._known_private_state.clear()
        self._last_witch_action.clear()

    async def observe(self, sync_packet: dict[str, Any]) -> None:
        """只保存最近少量公开摘要，避免把完整历史塞进提示词。"""

        if not isinstance(sync_packet, Mapping):
            return
        self._ensure_game_context(sync_packet)
        snapshot = self._summarize_public_sync(sync_packet)
        if not snapshot:
            return
        signature = json.dumps(snapshot, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
        if signature == self._last_observe_signature:
            return
        self._last_observe_signature = signature
        self._compact_history.append(snapshot)
        if len(self._compact_history) > 4:
            self._compact_history = self._compact_history[-4:]

    async def decide(self, turn_packet: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(turn_packet, Mapping):
            raise ValueError("行动包必须是对象")
        self._ensure_game_context(turn_packet)
        if turn_packet["self"]["player_id"] != self.player_id:
            raise ValueError("行动包被发送给了错误玩家")
        private = turn_packet["private_information"]
        if str(private.get("role")) != self.profile.role:
            raise ValueError("Task-Agent 的角色档案与行动包身份不一致")

        self._update_private_state(private)
        recommendation = self._recommend_witch_action(turn_packet)
        system = self._system_prompt(private) + (
            "\n\n【女巫快速策略】用药阶段先执行合法性，再按建议：有明确夜刀且解药可用时前两夜积极救；"
            "毒药用于公开查杀、明显悍跳或票型高度集中铁狼。白天发言短、清楚、归票；不泄露夜间私密信息。"
        )
        prompt = self._build_prompt_packet(turn_packet, recommendation)
        tool_context = turn_packet.get("tool_context") or {}
        current_dialogue = tool_context.get("current_round_dialogue") or []

        def execute_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
            del arguments
            if name != CURRENT_ROUND_DIALOGUE_TOOL_NAME:
                return {"error": f"不支持的工具：{name}"}
            return {
                "round": (turn_packet.get("game") or {}).get("round"),
                "phase": (turn_packet.get("game") or {}).get("public_phase", (turn_packet.get("game") or {}).get("phase")),
                "dialogue": self._clip_to_chars(current_dialogue, self.max_tool_result_tokens),
            }

        feedback = ""
        last_error = ""
        for attempt in range(self.max_decision_retries + 1):
            allow_tool = attempt == 0 and self.max_tool_calls_per_decision > 0
            user_content = {
                "instruction": self._turn_instruction(turn_packet["request"], feedback),
                "packet": prompt,
            }
            try:
                raw = await self._complete_json(
                    system=system,
                    messages=[{"role": "user", "content": json.dumps(user_content, ensure_ascii=False, separators=(",", ":"))}],
                    tools=[CURRENT_ROUND_DIALOGUE_TOOL] if allow_tool else None,
                    tool_executor=execute_tool if allow_tool else None,
                    max_tool_calls=self.max_tool_calls_per_decision if allow_tool else 0,
                    max_tool_result_tokens=self.max_tool_result_tokens,
                    max_prompt_chars=self.max_prompt_chars,
                )
            except ModelClientError:
                if attempt >= self.max_decision_retries:
                    return self._fallback_legal_action(turn_packet["request"], recommendation)
                feedback = "上一轮模型调用失败，请直接给出最短合法 JSON。"
                continue
            self._record_token_usage(raw)
            action = normalize_decision(raw, turn_packet["request"])
            action = self._safe_witch_action(action, turn_packet["request"], turn_packet)
            error = decision_error(action, turn_packet["request"])
            if error is None:
                self._remember_witch_action(turn_packet, action)
                return action
            last_error = error
            if attempt < self.max_decision_retries:
                feedback = error

        fallback = self._fallback_legal_action(turn_packet["request"], recommendation)
        if decision_error(fallback, turn_packet["request"]) is None:
            self._remember_witch_action(turn_packet, fallback)
            return fallback
        raise RuleViolationError(f"模型返回非法行动：{last_error}", attempted_action=fallback, validation_error=last_error)

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
            "task_fingerprint": hashlib.sha256(self.profile.task.encode("utf-8")).hexdigest(),
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
            compact_memory=self._render_compact_memory(),
        )

    @staticmethod
    def _turn_instruction(request: Mapping[str, Any], feedback: str) -> str:
        return render_prompt(
            "player_turn_instruction.txt",
            action_contract=render_action_contract(request),
            validation_feedback=(f"上一次输出未通过校验：{feedback}" if feedback else ""),
        )

    def _build_prompt_packet(self, turn_packet: Mapping[str, Any], recommendation: dict[str, Any]) -> dict[str, Any]:
        packet = {
            "game": self._pick_keys(turn_packet.get("game"), ("game_id", "round", "phase", "public_phase", "day", "night")),
            "self": self._pick_keys(turn_packet.get("self"), ("player_id", "alive", "is_sheriff", "sheriff")),
            "private_witch": {
                "role": (turn_packet.get("private_information") or {}).get("role"),
                "team": (turn_packet.get("private_information") or {}).get("team"),
                "antidote_available": self._known_private_state.get("antidote_available"),
                "poison_available": self._known_private_state.get("poison_available"),
                "current_attack_target": self._known_private_state.get("current_attack_target"),
            },
            "public_state": self._compact_public_state(turn_packet.get("public_state")),
            "request": self._compact_request(turn_packet.get("request")),
            "memory": self._compact_history[-3:],
            "recommendation": recommendation,
            "rules": [
                "最终只输出一个合法 JSON。",
                "需要更多本轮发言时调用 read_current_round_dialogue；不要臆造历史。",
                "白天只谈公开信息，绝不泄露夜刀目标和药水状态。",
            ],
        }
        return self._shrink_prompt_packet(packet, 6000)

    def _compact_request(self, request: Any) -> dict[str, Any]:
        if not isinstance(request, Mapping):
            return {}
        compact: dict[str, Any] = self._pick_keys(request, ("request_id", "player_id", "phase", "channel", "kind"))
        actions: list[dict[str, Any]] = []
        raw_actions = request.get("allowed_actions")
        if isinstance(raw_actions, list):
            for item in raw_actions:
                if not isinstance(item, Mapping):
                    continue
                action = self._pick_keys(item, ("kind", "target_ids", "max_chars", "require_chinese"))
                if isinstance(action.get("target_ids"), list):
                    action["target_ids"] = [str(t) for t in action["target_ids"][:12]]
                actions.append(action)
        compact["allowed_actions"] = actions
        return compact

    def _compact_public_state(self, public_state: Any) -> dict[str, Any]:
        if not isinstance(public_state, Mapping):
            return {}
        keys = (
            "round", "phase", "public_phase", "alive_players", "dead_players", "players",
            "deaths", "votes", "vote_results", "claims", "reveals", "public_claims",
            "role_claims", "counterclaims", "sheriff", "captain",
        )
        result: dict[str, Any] = {}
        for key in keys:
            if key in public_state:
                result[key] = self._compact_value(public_state[key])
        alive = self._extract_player_ids(public_state.get("alive_players")) or self._extract_alive_from_players(public_state.get("players"))
        if alive:
            result["alive_count"] = len(alive)
            result["alive_players"] = alive[:12]
        return result

    def _shrink_prompt_packet(self, packet: dict[str, Any], limit: int) -> dict[str, Any]:
        if len(json.dumps(packet, ensure_ascii=False, separators=(",", ":"), default=str)) <= limit:
            return packet
        packet = dict(packet)
        packet["memory"] = packet.get("memory", [])[-1:]
        public_state = dict(packet.get("public_state") or {})
        for noisy in ("claims", "reveals", "public_claims", "role_claims", "counterclaims", "votes", "vote_results", "players"):
            if isinstance(public_state.get(noisy), list):
                public_state[noisy] = public_state[noisy][-4:]
        packet["public_state"] = public_state
        if len(json.dumps(packet, ensure_ascii=False, separators=(",", ":"), default=str)) <= limit:
            return packet
        packet["public_state"] = self._pick_keys(public_state, ("round", "phase", "public_phase", "alive_count", "alive_players", "dead_players", "deaths", "sheriff", "captain"))
        packet["memory"] = []
        return packet

    def _update_private_state(self, private: Mapping[str, Any]) -> None:
        def first_present(names: tuple[str, ...]) -> tuple[bool, Any]:
            for name in names:
                if name in private:
                    return True, private[name]
            return False, None

        def as_bool(value: Any) -> bool | None:
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)) and value in (0, 1):
                return bool(value)
            if isinstance(value, str):
                text = value.strip().lower()
                if text in {"true", "yes", "1", "available", "可用"}:
                    return True
                if text in {"false", "no", "0", "unavailable", "不可用"}:
                    return False
            return None

        def resource(available_names: tuple[str, ...], used_names: tuple[str, ...]) -> tuple[bool | None, bool | None]:
            available_found, available_value = first_present(available_names)
            used_found, used_value = first_present(used_names)
            available = as_bool(available_value) if available_found else None
            used = as_bool(used_value) if used_found else None
            if used is True or available is False:
                available = False
            return available, used

        antidote_available, antidote_used = resource(
            ("antidote_available", "heal_available", "can_heal", "has_antidote"),
            ("antidote_used", "heal_used", "used_heal", "healed"),
        )
        poison_available, poison_used = resource(
            ("poison_available", "can_poison", "has_poison"),
            ("poison_used", "used_poison", "poisoned"),
        )
        attack_found, attack_value = first_present(
            ("current_attack_target", "current_night_attack_target", "night_attack_target", "attacked_player_id", "wolf_attack_target")
        )
        state: dict[str, Any] = {
            "antidote_available": antidote_available,
            "poison_available": poison_available,
            "antidote_used": antidote_used,
            "poison_used": poison_used,
            "current_attack_target": str(attack_value) if attack_found and attack_value is not None else None,
        }
        for key in ("role", "team", "status"):
            if key in private:
                state[key] = private[key]
        self._known_private_state.update(state)

    def _render_compact_memory(self) -> str:
        if not self._compact_history:
            return "暂无跨回合摘要。"
        return json.dumps(self._compact_history[-3:], ensure_ascii=False, separators=(",", ":"), default=str)

    def _summarize_public_sync(self, sync_packet: Mapping[str, Any]) -> dict[str, Any]:
        summary: dict[str, Any] = {}
        for key in ("round", "game_round", "day_round", "phase", "public_phase", "stage"):
            if key in sync_packet:
                out_key = "round" if "round" in key else "phase"
                summary.setdefault(out_key, sync_packet[key])
        for source_key, out_key in (
            ("deaths", "deaths"), ("dead_players", "dead_players"), ("night_deaths", "deaths"),
            ("votes", "votes"), ("vote_results", "votes"), ("vote_events", "votes"),
            ("claims", "claims"), ("reveals", "claims"), ("public_claims", "claims"),
            ("role_claims", "claims"), ("counterclaims", "claims"),
        ):
            value = sync_packet.get(source_key)
            if isinstance(value, list) and value:
                summary[out_key] = self._compact_value(value[-6:])
        for key in ("sheriff", "captain"):
            if key in sync_packet:
                summary[key] = self._compact_value(sync_packet[key])
        return summary

    def _recommend_witch_action(self, turn_packet: Mapping[str, Any]) -> dict[str, Any]:
        request = turn_packet.get("request") if isinstance(turn_packet.get("request"), Mapping) else {}
        game = turn_packet.get("game") if isinstance(turn_packet.get("game"), Mapping) else {}
        public_state = turn_packet.get("public_state") if isinstance(turn_packet.get("public_state"), Mapping) else {}
        allowed = request.get("allowed_actions") if isinstance(request, Mapping) else []
        phase = str(game.get("public_phase", game.get("phase", request.get("phase", "")))).lower()
        kinds = [str(item.get("kind") or "") for item in allowed or [] if isinstance(item, Mapping)]
        pass_kind = next((k for k in kinds if "pass" in k or k == "wait"), None)
        heal_kind = next((k for k in kinds if "heal" in k), None)
        poison_kind = next((k for k in kinds if "poison" in k), None)
        fallback_kind = pass_kind or (kinds[0] if kinds else "pass")
        if not ("night_witch" in phase or ("night" in phase and "witch" in phase)):
            return {"kind": fallback_kind, "reason": "not_witch_night", "confidence": "low"}

        heal_targets = self._allowed_targets_for_kind(allowed, "heal")
        poison_targets = self._allowed_targets_for_kind(allowed, "poison")
        round_no = self._round_number(game, public_state)
        alive = self._alive_players(public_state)
        alive_count = len(alive) if alive else self._nonnegative_int(public_state.get("alive_count"), fallback=0) or 0
        attack = self._known_private_state.get("current_attack_target")
        heal_available = bool(heal_kind) and self._known_private_state.get("antidote_available") is True and self._known_private_state.get("antidote_used") is not True
        poison_available = bool(poison_kind) and self._known_private_state.get("poison_available") is True and self._known_private_state.get("poison_used") is not True

        if heal_available and attack is not None:
            target = self._select_legal_target(attack, heal_targets)
            if target:
                high_value = self._is_public_high_value(target, public_state) or target == self.player_id
                return {
                    "kind": heal_kind,
                    "target_id": target,
                    "reason": "save_known_night_kill",
                    "confidence": "high" if round_no <= 2 or high_value or alive_count <= 7 else "medium",
                }

        if poison_available:
            poison = self._find_poison_candidate(poison_targets, public_state)
            if poison:
                return {"kind": poison_kind, "target_id": poison["target_id"], "reason": poison["reason"], "confidence": poison["confidence"]}

        return {"kind": fallback_kind, "reason": "no_safe_medicine", "confidence": "low"}

    def _find_poison_candidate(self, legal_targets: list[str], public_state: Mapping[str, Any]) -> dict[str, str] | None:
        legal = [target for target in legal_targets if target != self.player_id]
        if not legal:
            return None
        scores = {target: 0 for target in legal}
        reasons: dict[str, list[str]] = {target: [] for target in legal}
        protected: set[str] = set()

        events: list[Any] = []
        for snapshot in self._compact_history[-3:]:
            for key in ("claims", "votes"):
                value = snapshot.get(key)
                if isinstance(value, list):
                    events.extend(value[-6:])
        for key in ("claims", "reveals", "public_claims", "role_claims", "counterclaims", "votes", "vote_results"):
            value = public_state.get(key)
            if isinstance(value, list):
                events.extend(value[-8:])

        vote_counts: dict[str, int] = {target: 0 for target in legal}
        seer_claimers: set[str] = set()
        for event in events:
            if not isinstance(event, Mapping):
                continue
            text = self._event_text(event)
            target = self._event_target(event)
            speaker = str(event.get("speaker_id") or event.get("claimant_id") or event.get("source_id") or "")
            if any(token in text for token in ("预言家", "seer", "prophet")) and speaker in legal:
                seer_claimers.add(speaker)
            if target in vote_counts and ("vote" in text or "投" in text or event.get("vote_target_id") is not None):
                vote_counts[target] += 1
            if target not in scores:
                continue
            if any(token in text for token in ("金水", "good", "好人", "银水")) and any(token in text for token in ("verified", "验证", "查验", "公开")):
                protected.add(target)
            if any(token in text for token in ("查杀", "狼人", "wolf", "werewolf", "黑查")):
                scores[target] += 4
                reasons[target].append("public_wolf_check")
            if any(token in text for token in ("悍跳", "对跳", "counter", "conflict", "矛盾")):
                scores[target] += 2
                reasons[target].append("counterclaim_conflict")

        if len(seer_claimers) >= 2:
            for pid in seer_claimers:
                if pid in scores:
                    scores[pid] += 1
                    reasons[pid].append("prophet_counterclaim")
        for target, count in vote_counts.items():
            if count >= 3:
                scores[target] += 2
                reasons[target].append("concentrated_vote_pressure")
            elif count >= 2:
                scores[target] += 1

        best = sorted(legal, key=lambda pid: (-scores[pid], pid))[0]
        if best in protected:
            return None
        if scores[best] >= 4 or (scores[best] >= 3 and vote_counts.get(best, 0) >= 2):
            return {"target_id": best, "reason": ",".join(reasons[best][:3]) or "strong_public_wolf_signal", "confidence": "high"}
        return None

    def _safe_witch_action(self, action: dict[str, Any], request: Mapping[str, Any], turn_packet: Mapping[str, Any]) -> dict[str, Any]:
        kind = str(action.get("kind") or "")
        if "heal" not in kind and "poison" not in kind:
            return action
        game = turn_packet.get("game") if isinstance(turn_packet.get("game"), Mapping) else {}
        phase = str(game.get("public_phase", game.get("phase", request.get("phase", "")))).lower()
        if not ("night_witch" in phase or ("night" in phase and "witch" in phase)):
            return self._fallback_legal_action(request, {"kind": "pass"})
        allowed = request.get("allowed_actions") if isinstance(request, Mapping) else []
        if "heal" in kind:
            legal = self._allowed_targets_for_kind(allowed, "heal")
            available = self._known_private_state.get("antidote_available") is True and self._known_private_state.get("antidote_used") is not True
            if available and str(action.get("target_id")) in legal:
                return action
        if "poison" in kind:
            legal = self._allowed_targets_for_kind(allowed, "poison")
            available = self._known_private_state.get("poison_available") is True and self._known_private_state.get("poison_used") is not True
            if available and str(action.get("target_id")) in legal and str(action.get("target_id")) != self.player_id:
                return action
        return self._fallback_legal_action(request, {"kind": "pass"})

    def _fallback_legal_action(self, request: Mapping[str, Any], recommendation: Mapping[str, Any] | None = None) -> dict[str, Any]:
        allowed = request.get("allowed_actions") if isinstance(request, Mapping) else []
        base = {"request_id": request.get("request_id"), "player_id": request.get("player_id", self.player_id)}
        if isinstance(recommendation, Mapping):
            kind = str(recommendation.get("kind") or "")
            target = recommendation.get("target_id")
            action = {**base, "kind": kind}
            if target is not None:
                action["target_id"] = str(target)
            if kind and decision_error(action, request) is None:
                return action
        if isinstance(allowed, list):
            pass_item = next((a for a in allowed if isinstance(a, Mapping) and ("pass" in str(a.get("kind")) or str(a.get("kind")) == "wait")), None)
            item = pass_item or next((a for a in allowed if isinstance(a, Mapping)), None)
            if isinstance(item, Mapping):
                kind = str(item.get("kind") or "")
                action = {**base, "kind": kind}
                targets = item.get("target_ids")
                if isinstance(targets, list) and targets:
                    action["target_id"] = str(targets[0])
                if kind in _SPEECH_ACTION_KINDS:
                    max_chars = int(item.get("max_chars") or 80)
                    text = "我先按公开发言和票型判断，今天请大家集中票型，不分票。"
                    action["text"] = text[:max_chars]
                    if item.get("require_chinese") and not _CHINESE_CHARACTER.search(action["text"]):
                        action["text"] = "好"
                return action
        return {**base, "kind": "pass"}

    def _remember_witch_action(self, turn_packet: Mapping[str, Any], action: dict[str, Any]) -> None:
        request = turn_packet.get("request") or {}
        phase = str((turn_packet.get("game") or {}).get("public_phase", (turn_packet.get("game") or {}).get("phase", "")))
        if "witch" not in phase and "witch" not in str(request.get("kind") or ""):
            return
        self._last_witch_action = {"phase": phase, "kind": action.get("kind"), "target_id": action.get("target_id")}
        self._known_private_state["last_witch_action"] = dict(self._last_witch_action)

    async def _complete_json(self, **kwargs: Any) -> dict[str, Any]:
        if self.request_coordinator is not None:
            return await self.request_coordinator.complete_json(self.model_client, max_tokens=self.max_tokens, **kwargs)
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
        values = {key: self._nonnegative_int(usage.get(key), fallback=None) for key in ("input_tokens", "output_tokens", "total_tokens")}
        if all(value is None for value in values.values()):
            return
        self._model_token_usage["reported_usage_response_count"] += 1
        for key, value in values.items():
            if value is not None:
                self._model_token_usage[key] += value

    def _compact_value(self, value: Any) -> Any:
        if isinstance(value, Mapping):
            compact: dict[str, Any] = {}
            for key in (
                "player_id", "id", "target_id", "speaker_id", "claimant_id", "source_id", "claimed_player_id",
                "claimed_role", "claimed_team", "voter_id", "vote_target_id", "result", "role", "team", "type",
                "action", "kind", "alive", "reason", "count", "round", "phase", "verified", "verification",
                "conflict", "conflicts", "counterclaim_to", "contradicts", "text", "content",
            ):
                if key in value:
                    item = value[key]
                    compact[key] = self._short_text(item) if isinstance(item, str) else item
            if not compact:
                for key, item in list(value.items())[:4]:
                    compact[str(key)] = self._short_text(item) if isinstance(item, str) else item
            return compact
        if isinstance(value, list):
            return [self._compact_value(item) for item in value[:8]]
        return self._short_text(value) if isinstance(value, str) else value

    @staticmethod
    def _pick_keys(value: Any, keys: tuple[str, ...]) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            return {}
        return {key: value[key] for key in keys if key in value}

    @staticmethod
    def _short_text(text: str, limit: int = 120) -> str:
        return text if len(text) <= limit else text[:limit] + "…"

    def _clip_to_chars(self, value: Any, limit: int) -> Any:
        text = json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
        if len(text) <= limit:
            return value
        return text[: max(1, limit - 1)] + "…"

    def _event_text(self, event: Mapping[str, Any]) -> str:
        parts: list[str] = []
        for key in ("text", "content", "reason", "result", "claimed_role", "claimed_team", "type", "kind", "action", "verification", "conflict", "conflicts", "contradicts"):
            if key in event and event[key] is not None:
                parts.append(str(event[key]))
        return " ".join(parts).lower()

    def _event_target(self, event: Mapping[str, Any]) -> str | None:
        for key in ("target_id", "claimed_player_id", "vote_target_id", "subject_id", "candidate_id"):
            if key in event and event[key] is not None:
                return str(event[key])
        return None

    def _round_number(self, game: Mapping[str, Any], public_state: Mapping[str, Any]) -> int:
        for source in (game, public_state):
            for key in ("round", "day_round", "night_round"):
                value = self._nonnegative_int(source.get(key), fallback=None)
                if value is not None:
                    return value
        return 0

    def _alive_players(self, public_state: Mapping[str, Any]) -> list[str]:
        return self._extract_player_ids(public_state.get("alive_players")) or self._extract_alive_from_players(public_state.get("players"))

    def _is_public_high_value(self, player_id: Any, public_state: Mapping[str, Any]) -> bool:
        target = str(player_id)
        sheriff = public_state.get("sheriff", public_state.get("captain"))
        if isinstance(sheriff, Mapping) and str(sheriff.get("player_id") or sheriff.get("id")) == target:
            return True
        if str(sheriff) == target:
            return True
        events: list[Any] = []
        for snapshot in self._compact_history[-3:]:
            value = snapshot.get("claims")
            if isinstance(value, list):
                events.extend(value)
        for key in ("claims", "reveals", "public_claims", "role_claims"):
            value = public_state.get(key)
            if isinstance(value, list):
                events.extend(value[-8:])
        for event in events:
            if not isinstance(event, Mapping):
                continue
            target_id = self._event_target(event) or str(event.get("speaker_id") or event.get("claimant_id") or "")
            if target_id != target:
                continue
            text = self._event_text(event)
            if any(token in text for token in ("预言家", "女巫", "守卫", "seer", "prophet", "witch", "guard", "guardian", "神")):
                return True
            if any(token in text for token in ("金水", "银水", "good", "好人")):
                return True
        return False

    def _extract_player_ids(self, value: Any) -> list[str]:
        if isinstance(value, list):
            result: list[str] = []
            for item in value:
                if isinstance(item, Mapping):
                    for key in ("player_id", "id", "target_id", "speaker_id"):
                        if key in item and item[key] is not None:
                            result.append(str(item[key]))
                            break
                elif item is not None:
                    result.append(str(item))
            return result
        if isinstance(value, Mapping):
            for key in ("player_id", "id", "target_id"):
                if key in value and value[key] is not None:
                    return [str(value[key])]
        return []

    def _extract_alive_from_players(self, value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        alive: list[str] = []
        for item in value:
            if not isinstance(item, Mapping) or item.get("alive") is False:
                continue
            for key in ("player_id", "id"):
                if key in item and item[key] is not None:
                    alive.append(str(item[key]))
                    break
        return alive

    def _allowed_targets_for_kind(self, allowed_actions: Any, needle: str) -> list[str]:
        if not isinstance(allowed_actions, list):
            return []
        targets: list[str] = []
        for item in allowed_actions:
            if not isinstance(item, Mapping):
                continue
            if needle not in str(item.get("kind") or ""):
                continue
            raw_targets = item.get("target_ids")
            if isinstance(raw_targets, list):
                for target in raw_targets:
                    target_id = str(target)
                    if target_id not in targets:
                        targets.append(target_id)
        return targets

    @staticmethod
    def _select_legal_target(preferred: Any, legal_targets: list[str]) -> str | None:
        if preferred is None:
            return None
        preferred_id = str(preferred)
        return preferred_id if preferred_id in legal_targets else None

    @staticmethod
    def _nonnegative_int(value: object, *, fallback: int | None) -> int | None:
        if isinstance(value, bool):
            return fallback
        try:
            normalized = int(value)
        except (TypeError, ValueError):
            return fallback
        return normalized if normalized >= 0 else fallback

"""最小可运行的角色 Task-Agent。

每次行动从一个新的模型会话开始。模型默认只收到当前状态；本轮对话只能通过唯一的
受限工具按需读取。这里不包含策略进化、长期记忆、外部检索或其他 Harness。
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
        self._compact_history: list[dict[str, Any]] = []
        self._last_observe_signature = ""
        self._known_private_state: dict[str, Any] = {}
        self._last_witch_action: dict[str, Any] = {}

    async def observe(self, sync_packet: dict[str, Any]) -> None:
        """接收公开同步，并保留极简跨回合摘要。"""

        if not isinstance(sync_packet, Mapping):
            return
        snapshot = self._summarize_public_sync(sync_packet)
        if not snapshot:
            return
        signature = json.dumps(snapshot, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if signature == self._last_observe_signature:
            return
        self._last_observe_signature = signature
        self._compact_history.append(snapshot)
        if len(self._compact_history) > 6:
            self._compact_history = self._compact_history[-6:]

    async def decide(self, turn_packet: dict[str, Any]) -> dict[str, Any]:
        if turn_packet["self"]["player_id"] != self.player_id:
            raise ValueError("行动包被发送给了错误玩家")
        private = turn_packet["private_information"]
        if str(private.get("role")) != self.profile.role:
            raise ValueError("Task-Agent 的角色档案与行动包身份不一致")

        self._update_private_state(private)
        witch_summary = self._witch_decision_summary(turn_packet)
        system = self._system_prompt(private)
        prompt = {
            "game": turn_packet["game"],
            "public_rules": turn_packet["public_rules"],
            "self": turn_packet["self"],
            "private_information": private,
            "public_state": turn_packet["public_state"],
            "request": turn_packet["request"],
            "compact_memory": self._render_compact_memory(),
            "self_resource_state": dict(self._known_private_state),
            "witch_decision_summary": witch_summary,
            "speech_safety": (
                "所有发言只能基于公开事实；不要把药水余量、夜刀目标、自己未公开的用药结果、"
                "或未公开身份链说成既成事实。"
            ),
            "history_policy": {
                "default_context": "current_state_only",
                "current_round_dialogue_tool": CURRENT_ROUND_DIALOGUE_TOOL_NAME,
                "max_tool_calls": self.max_tool_calls_per_decision,
                "max_tool_result_tokens": self.max_tool_result_tokens,
            },
        }

        tool_context = turn_packet.get("tool_context") or {}
        current_dialogue = tool_context.get("current_round_dialogue") or []

        system = system + "\n\n【女巫专用决策规约】\n先复盘最近票型与死亡，再看自己药水状态。救药只救高价值且能改变轮次的目标；毒药只打高置信狼人或悍跳核心；证据不足时默认保留资源。公开发言只谈公开信息，不得泄露私密夜间信息。"

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
            action = self._apply_witch_safety_fallback(turn_packet, action, witch_summary)
            error = decision_error(action, turn_packet["request"])
            if error is None:
                self._remember_witch_action(turn_packet, action)
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
        memory_hint = self._render_compact_memory()
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
            compact_memory=memory_hint,
        )

    @staticmethod
    def _turn_instruction(request: Mapping[str, Any], feedback: str) -> str:
        return render_prompt(
            "player_turn_instruction.txt",
            action_contract=render_action_contract(request),
            validation_feedback=(f"上一次输出未通过校验：{feedback}" if feedback else ""),
        )

    def _update_private_state(self, private: Mapping[str, Any]) -> None:
        extracted: dict[str, Any] = {}
        for key in (
            "role",
            "team",
            "status",
            "can_heal",
            "can_poison",
            "heal_used",
            "poison_used",
            "heal_available",
            "poison_available",
            "antidote_available",
            "poisoner_available",
            "heal_remaining",
            "poison_remaining",
        ):
            if key in private:
                extracted[key] = private[key]
        for key in ("healed", "poisoned", "used_heal", "used_poison", "last_heal_target", "last_poison_target"):
            if key in private:
                extracted[key] = private[key]
        for key, value in extracted.items():
            self._known_private_state[key] = value

    def _render_compact_memory(self) -> str:
        if not self._compact_history:
            return "暂无跨回合摘要。"
        recent = self._compact_history[-4:]
        return json.dumps(recent, ensure_ascii=False, separators=(",", ":"))

    def _summarize_public_sync(self, sync_packet: Mapping[str, Any]) -> dict[str, Any]:
        def pick(source: Mapping[str, Any], *names: str) -> Any:
            for name in names:
                if name in source:
                    return source[name]
            return None

        summary: dict[str, Any] = {}
        round_info = pick(sync_packet, "round", "game_round", "day_round")
        phase_info = pick(sync_packet, "phase", "public_phase", "stage")
        if round_info is not None:
            summary["round"] = round_info
        if phase_info is not None:
            summary["phase"] = phase_info

        deaths = pick(sync_packet, "deaths", "dead_players", "night_deaths", "death_events")
        if isinstance(deaths, list) and deaths:
            summary["deaths"] = self._compact_entities(deaths)

        votes = pick(sync_packet, "votes", "voting", "vote_results", "vote_events")
        if isinstance(votes, list) and votes:
            summary["votes"] = self._compact_entities(votes)

        sheri = pick(sync_packet, "sheriff", "police", "警长", "captain")
        if sheri is not None:
            summary["sheriff"] = self._compact_value(sheri)

        claims = pick(sync_packet, "claims", "reveals", "public_claims", "role_claims", "counterclaims")
        if isinstance(claims, list) and claims:
            summary["claims"] = self._compact_entities(claims)

        if not summary:
            for key in ("round", "phase", "deaths", "votes", "sheriff", "claims"):
                if key in sync_packet:
                    value = sync_packet[key]
                    if isinstance(value, list):
                        summary[key] = self._compact_entities(value)
                    else:
                        summary[key] = self._compact_value(value)
        return summary

    def _compact_entities(self, items: list[Any]) -> list[Any]:
        return [self._compact_value(item) for item in items[:8]]

    def _compact_value(self, value: Any) -> Any:
        if isinstance(value, Mapping):
            compact: dict[str, Any] = {}
            for key in (
                "player_id",
                "id",
                "target_id",
                "speaker_id",
                "voter_id",
                "vote_target_id",
                "result",
                "role",
                "team",
                "type",
                "action",
                "kind",
                "alive",
                "reason",
                "count",
                "heal_ready",
                "poison_ready",
                "heal_available",
                "poison_available",
                "antidote_available",
                "poisoner_available",
                "heal_remaining",
                "poison_remaining",
                "heal_used",
                "poison_used",
                "last_heal_target",
                "last_poison_target",
            ):
                if key in value:
                    compact[key] = value[key]
            if not compact:
                for key, item in list(value.items())[:4]:
                    compact[key] = item
            return compact
        if isinstance(value, list):
            return [self._compact_value(item) for item in value[:8]]
        return value

    def _remember_witch_action(self, turn_packet: Mapping[str, Any], action: dict[str, Any]) -> None:
        request = turn_packet.get("request") or {}
        phase = str(turn_packet.get("game", {}).get("public_phase", turn_packet.get("game", {}).get("phase", "")))
        if "witch" not in phase and "witch" not in str(request.get("kind") or ""):
            return
        self._last_witch_action = {
            "phase": phase,
            "kind": action.get("kind"),
            "target_id": action.get("target_id"),
        }
        self._known_private_state["last_witch_action"] = dict(self._last_witch_action)

    def _witch_decision_summary(self, turn_packet: Mapping[str, Any]) -> dict[str, Any]:
        context = self._build_witch_decision_context(turn_packet)
        recommendation = self._recommend_witch_action(context)
        context["recommendation"] = recommendation
        return context

    def _build_witch_decision_context(self, turn_packet: Mapping[str, Any]) -> dict[str, Any]:
        request = turn_packet.get("request") or {}
        private = turn_packet.get("private_information") or {}
        game = turn_packet.get("game") or {}
        public_state = turn_packet.get("public_state") or {}
        tool_context = turn_packet.get("tool_context") or {}
        phase = str(game.get("public_phase", game.get("phase", "")))
        allowed_actions = [item for item in request.get("allowed_actions", []) if isinstance(item, Mapping)]
        action_kinds = [str(item.get("kind") or "") for item in allowed_actions if item.get("kind")]
        heal_kind = next((kind for kind in action_kinds if "heal" in kind), None)
        poison_kind = next((kind for kind in action_kinds if "poison" in kind), None)
        pass_kind = next((kind for kind in action_kinds if "pass" in kind or kind == "wait"), None)

        resource_state = self._extract_witch_resource_state(private)
        recent_public_deaths = self._collect_compact_events(public_state, ("deaths", "dead_players", "night_deaths", "death_events"))
        recent_votes = self._collect_compact_events(public_state, ("votes", "voting", "vote_results", "vote_events"))
        recent_claims = self._collect_compact_events(public_state, ("claims", "reveals", "public_claims", "role_claims", "counterclaims"))
        current_dialogue = self._collect_compact_events(tool_context.get("current_round_dialogue") or [])
        high_confidence_targets = self._rank_high_confidence_targets(recent_votes, recent_claims, current_dialogue)
        heal_candidate = self._infer_heal_candidate(resource_state, recent_public_deaths, high_confidence_targets, allowed_actions)
        poison_candidate = self._infer_poison_candidate(resource_state, recent_votes, recent_claims, current_dialogue, allowed_actions)
        info_level = self._judge_witch_information_level(resource_state, recent_public_deaths, recent_votes, recent_claims, current_dialogue)

        summary: dict[str, Any] = {
            "phase": phase,
            "request_kind": request.get("kind"),
            "allowed_action_kinds": [kind for kind in action_kinds if kind],
            "resource_state": resource_state,
            "recent_public_deaths": recent_public_deaths[-2:],
            "recent_vote_pressure": recent_votes[-3:],
            "counterclaim_summary": recent_claims[-3:],
            "public_high_confidence_targets": high_confidence_targets[:3],
            "current_round_dialogue_signals": current_dialogue[:4],
            "last_witch_action": dict(self._last_witch_action) if self._last_witch_action else {},
            "information_level": info_level,
            "decision_rule": {
                "low_info_default": "pass",
                "heal_rule": "only when explicit save target exists or recent death is clearly actionable",
                "poison_rule": "only when target has double public support or hard contradiction",
                "fallback": pass_kind or "pass",
            },
        }
        if heal_kind:
            summary["heal_candidate"] = heal_candidate or {"kind": heal_kind, "target_id": None, "confidence": "low", "reason": "no_clear_save_target"}
        else:
            summary["heal_candidate"] = None
        if poison_kind:
            summary["poison_candidate"] = poison_candidate or {"kind": poison_kind, "target_id": None, "confidence": "low", "reason": "no_clear_poison_target"}
        else:
            summary["poison_candidate"] = None
        summary["pass_candidate"] = {"kind": pass_kind or "pass", "reason": "default_safety", "confidence": "high" if info_level == "low" else "medium"}
        return summary

    def _extract_witch_resource_state(self, private: Mapping[str, Any]) -> dict[str, Any]:
        state = dict(self._known_private_state)
        for key in (
            "can_heal",
            "can_poison",
            "heal_available",
            "poison_available",
            "antidote_available",
            "poisoner_available",
            "heal_remaining",
            "poison_remaining",
            "heal_used",
            "poison_used",
            "last_heal_target",
            "last_poison_target",
            "wolf_target",
            "kill_target",
            "night_target",
            "attack_target",
            "attacked_target",
            "saved_target",
            "heal_target",
            "poison_target",
            "target_id",
            "healed",
            "poisoned",
            "used_heal",
            "used_poison",
        ):
            if key in private:
                state[key] = private[key]
        for key in ("heal_available", "antidote_available", "can_heal"):
            if key in state:
                state["heal_ready"] = bool(state[key])
                break
        for key in ("poison_available", "can_poison", "poisoner_available"):
            if key in state:
                state["poison_ready"] = bool(state[key])
                break
        return self._compact_value(state) if isinstance(state, Mapping) else state

    def _collect_compact_events(self, source: Any, keys: tuple[str, ...] | tuple[()] = ()) -> list[Any]:
        items: Any = source
        if isinstance(source, Mapping) and keys:
            for key in keys:
                if key in source:
                    items = source[key]
                    break
            else:
                return []
        if not isinstance(items, list):
            return []
        return [self._compact_value(item) for item in items[:8]]

    def _rank_high_confidence_targets(self, votes: list[Any], claims: list[Any], dialogue: list[Any]) -> list[dict[str, Any]]:
        scores: dict[str, dict[str, Any]] = {}
        for item in votes:
            target = self._extract_target_id(item)
            if not target:
                continue
            entry = scores.setdefault(target, {"target_id": target, "score": 0, "reasons": []})
            entry["score"] += 2
            entry["reasons"].append("vote_pressure")
        for item in claims:
            text = self._stringify_item(item)
            targets = self._extract_target_ids_from_text(text)
            for target in targets:
                entry = scores.setdefault(target, {"target_id": target, "score": 0, "reasons": []})
                if any(keyword in text for keyword in ("对跳", "悍跳", "查杀", "狼人", "狼刀", "毒")):
                    entry["score"] += 2
                    entry["reasons"].append("hard_claim")
                else:
                    entry["score"] += 1
                    entry["reasons"].append("public_claim")
        for item in dialogue:
            text = self._stringify_item(item)
            targets = self._extract_target_ids_from_text(text)
            if not targets:
                continue
            hard_signal = any(keyword in text for keyword in ("查杀", "悍跳", "狼人", "对跳", "刀", "毒"))
            for target in targets:
                entry = scores.setdefault(target, {"target_id": target, "score": 0, "reasons": []})
                entry["score"] += 2 if hard_signal else 1
                entry["reasons"].append("dialogue_signal")
        ranked = sorted(scores.values(), key=lambda item: (-int(item.get("score", 0)), str(item.get("target_id", ""))))
        for item in ranked:
            item["confidence"] = "high" if int(item.get("score", 0)) >= 4 else "medium" if int(item.get("score", 0)) >= 2 else "low"
            reasons = item.get("reasons")
            if isinstance(reasons, list):
                item["reasons"] = reasons[:3]
        return ranked

    def _infer_heal_candidate(
        self,
        resource_state: Mapping[str, Any],
        deaths: list[Any],
        high_confidence_targets: list[dict[str, Any]],
        allowed_actions: list[Mapping[str, Any]],
    ) -> dict[str, Any] | None:
        if not self._resource_ready(resource_state, "heal"):
            return None
        heal_action = next((item for item in allowed_actions if "heal" in str(item.get("kind") or "")), None)
        if heal_action is None:
            return None
        explicit_target = self._extract_first_target_from_sources(
            resource_state,
            deaths,
            high_confidence_targets,
            ("wolf_target", "kill_target", "night_target", "attacked_target", "attack_target", "target_id", "saved_target", "heal_target"),
        )
        if explicit_target:
            confidence = "high" if deaths or self._has_strong_signal(high_confidence_targets, explicit_target) else "medium"
            if confidence == "high" or deaths:
                return {"kind": str(heal_action.get("kind")), "target_id": explicit_target, "confidence": confidence, "reason": "explicit_save_target"}
        if deaths and len(deaths) == 1:
            target = self._extract_target_id(deaths[0])
            if target:
                return {"kind": str(heal_action.get("kind")), "target_id": target, "confidence": "medium", "reason": "single_public_death"}
        return None

    def _infer_poison_candidate(
        self,
        resource_state: Mapping[str, Any],
        votes: list[Any],
        claims: list[Any],
        dialogue: list[Any],
        allowed_actions: list[Mapping[str, Any]],
    ) -> dict[str, Any] | None:
        if not self._resource_ready(resource_state, "poison"):
            return None
        poison_action = next((item for item in allowed_actions if "poison" in str(item.get("kind") or "")), None)
        if poison_action is None:
            return None
        vote_targets = [self._extract_target_id(item) for item in votes]
        vote_counts = Counter(target for target in vote_targets if target)
        claim_counts = Counter(target for target in self._targets_from_items(claims) if target)
        dialogue_counts = Counter(target for target in self._targets_from_items(dialogue) if target)
        combined: list[tuple[str, int, list[str]]] = []
        for target in set(vote_counts) | set(claim_counts) | set(dialogue_counts):
            score = 0
            reasons: list[str] = []
            if vote_counts.get(target, 0) >= 2:
                score += 2
                reasons.append("vote_pressure")
            if claim_counts.get(target, 0) >= 1:
                score += 2
                reasons.append("claim_conflict")
            if dialogue_counts.get(target, 0) >= 1:
                score += 1
                reasons.append("dialogue_signal")
            combined.append((target, score, reasons))
        combined.sort(key=lambda item: (-item[1], item[0]))
        if not combined:
            return None
        best_target, best_score, best_reasons = combined[0]
        if best_score < 4:
            return None
        confidence = "high" if best_score >= 4 else "medium"
        return {"kind": str(poison_action.get("kind")), "target_id": best_target, "confidence": confidence, "reason": "+".join(best_reasons[:3]) or "double_public_support"}

    def _judge_witch_information_level(
        self,
        resource_state: Mapping[str, Any],
        deaths: list[Any],
        votes: list[Any],
        claims: list[Any],
        dialogue: list[Any],
    ) -> str:
        signals = 0
        if deaths:
            signals += 1
        if len({self._extract_target_id(item) for item in votes if self._extract_target_id(item)}) >= 2:
            signals += 1
        if len(self._targets_from_items(claims)) >= 2:
            signals += 1
        if len(self._targets_from_items(dialogue)) >= 2:
            signals += 1
        if resource_state.get("heal_used") or resource_state.get("poison_used"):
            signals += 1
        if signals <= 1:
            return "low"
        if signals == 2:
            return "medium"
        return "high"

    def _recommend_witch_action(self, context: Mapping[str, Any]) -> dict[str, Any]:
        info_level = str(context.get("information_level") or "low")
        heal_candidate = context.get("heal_candidate") if isinstance(context.get("heal_candidate"), Mapping) else None
        poison_candidate = context.get("poison_candidate") if isinstance(context.get("poison_candidate"), Mapping) else None
        pass_candidate = context.get("pass_candidate") if isinstance(context.get("pass_candidate"), Mapping) else {"kind": "pass"}
        if info_level == "low":
            return {
                "primary_action": str(pass_candidate.get("kind") or "pass"),
                "selected_target_id": None,
                "heal_candidate": heal_candidate,
                "poison_candidate": poison_candidate,
                "confidence_level": "low",
                "pass_reason": "low_info_default_pass",
            }
        if heal_candidate and str(heal_candidate.get("confidence")) == "high" and heal_candidate.get("target_id"):
            return {
                "primary_action": str(heal_candidate.get("kind") or "heal"),
                "selected_target_id": str(heal_candidate.get("target_id")),
                "heal_candidate": heal_candidate,
                "poison_candidate": poison_candidate,
                "confidence_level": "high",
                "pass_reason": "",
            }
        if poison_candidate and str(poison_candidate.get("confidence")) == "high" and poison_candidate.get("target_id"):
            return {
                "primary_action": str(poison_candidate.get("kind") or "poison"),
                "selected_target_id": str(poison_candidate.get("target_id")),
                "heal_candidate": heal_candidate,
                "poison_candidate": poison_candidate,
                "confidence_level": "high",
                "pass_reason": "",
            }
        return {
            "primary_action": str(pass_candidate.get("kind") or "pass"),
            "selected_target_id": None,
            "heal_candidate": heal_candidate,
            "poison_candidate": poison_candidate,
            "confidence_level": "medium",
            "pass_reason": "insufficient_public_convergence",
        }

    def _apply_witch_safety_fallback(self, turn_packet: Mapping[str, Any], action: dict[str, Any], summary: Mapping[str, Any]) -> dict[str, Any]:
        request = turn_packet.get("request") or {}
        allowed_actions = [item for item in request.get("allowed_actions", []) if isinstance(item, Mapping)]
        if not allowed_actions:
            return action
        action_kind = str(action.get("kind") or "")
        pass_kind = next((str(item.get("kind")) for item in allowed_actions if "pass" in str(item.get("kind") or "") or str(item.get("kind") or "") == "wait"), None)
        if action_kind not in {str(item.get("kind") or "") for item in allowed_actions}:
            return action
        if "witch" not in str(turn_packet.get("game", {}).get("public_phase", turn_packet.get("game", {}).get("phase", ""))) and "witch" not in str(request.get("kind") or ""):
            return action
        info_level = str(summary.get("information_level") or "low")
        heal_candidate = summary.get("heal_candidate") if isinstance(summary.get("heal_candidate"), Mapping) else None
        poison_candidate = summary.get("poison_candidate") if isinstance(summary.get("poison_candidate"), Mapping) else None
        pass_candidate = summary.get("pass_candidate") if isinstance(summary.get("pass_candidate"), Mapping) else None
        if action_kind and "heal" in action_kind:
            if info_level == "low" or not heal_candidate or str(heal_candidate.get("confidence")) != "high":
                if pass_kind:
                    return {"request_id": action["request_id"], "player_id": action["player_id"], "kind": pass_kind}
                return action
            if not action.get("target_id") and heal_candidate.get("target_id"):
                action["target_id"] = str(heal_candidate.get("target_id"))
            if heal_candidate.get("target_id") and action.get("target_id") and str(action.get("target_id")) != str(heal_candidate.get("target_id")):
                if pass_kind:
                    return {"request_id": action["request_id"], "player_id": action["player_id"], "kind": pass_kind}
                action["target_id"] = str(heal_candidate.get("target_id"))
            return action
        if action_kind and "poison" in action_kind:
            if info_level == "low" or not poison_candidate or str(poison_candidate.get("confidence")) != "high":
                if pass_kind:
                    return {"request_id": action["request_id"], "player_id": action["player_id"], "kind": pass_kind}
                return action
            if not action.get("target_id") and poison_candidate.get("target_id"):
                action["target_id"] = str(poison_candidate.get("target_id"))
            if poison_candidate.get("target_id") and action.get("target_id") and str(action.get("target_id")) != str(poison_candidate.get("target_id")):
                if pass_kind:
                    return {"request_id": action["request_id"], "player_id": action["player_id"], "kind": pass_kind}
                action["target_id"] = str(poison_candidate.get("target_id"))
            return action
        if pass_candidate and pass_kind and action_kind == str(pass_candidate.get("kind") or ""):
            return action
        return action

    def _resource_ready(self, resource_state: Mapping[str, Any], resource_name: str) -> bool:
        if resource_name == "heal":
            return bool(resource_state.get("heal_ready") or resource_state.get("can_heal") or resource_state.get("heal_available") or resource_state.get("antidote_available"))
        if resource_name == "poison":
            return bool(resource_state.get("poison_ready") or resource_state.get("can_poison") or resource_state.get("poison_available") or resource_state.get("poisoner_available"))
        return False

    def _extract_first_target_from_sources(self, *sources: Any) -> str | None:
        for source in sources:
            if isinstance(source, Mapping):
                for key in ("wolf_target", "kill_target", "night_target", "attacked_target", "attack_target", "saved_target", "heal_target", "poison_target", "target_id", "player_id", "id"):
                    value = source.get(key)
                    if value:
                        return str(value)
            elif isinstance(source, list):
                for item in source:
                    target = self._extract_target_id(item)
                    if target:
                        return target
        return None

    def _has_strong_signal(self, targets: list[dict[str, Any]], target_id: str) -> bool:
        for item in targets:
            if str(item.get("target_id")) == str(target_id) and int(item.get("score", 0)) >= 4:
                return True
        return False

    def _targets_from_items(self, items: list[Any]) -> list[str]:
        targets: list[str] = []
        for item in items:
            target = self._extract_target_id(item)
            if target:
                targets.append(target)
            else:
                targets.extend(self._extract_target_ids_from_text(self._stringify_item(item)))
        return targets

    def _extract_target_id(self, value: Any) -> str | None:
        if isinstance(value, Mapping):
            for key in ("target_id", "vote_target_id", "player_id", "id", "speaker_id", "voter_id"):
                target = value.get(key)
                if target:
                    return str(target)
            text = self._stringify_item(value)
            return self._extract_target_ids_from_text(text)[0] if self._extract_target_ids_from_text(text) else None
        if isinstance(value, str):
            matches = self._extract_target_ids_from_text(value)
            return matches[0] if matches else None
        return None

    def _extract_target_ids_from_text(self, text: str) -> list[str]:
        if not text:
            return []
        matches = re.findall(r"\bp\d+\b", text, flags=re.IGNORECASE)
        return [match.lower() for match in matches]

    def _stringify_item(self, value: Any) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, Mapping):
            try:
                return json.dumps(value, ensure_ascii=False, sort_keys=True)
            except TypeError:
                return str(value)
        if isinstance(value, list):
            return json.dumps([self._compact_value(item) for item in value[:4]], ensure_ascii=False)
        return str(value)

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

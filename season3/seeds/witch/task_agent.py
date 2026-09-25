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
        self._current_game_id: str | None = None
        self._compact_history: list[dict[str, Any]] = []
        self._last_observe_signature = ""
        self._known_private_state: dict[str, Any] = {}
        self._last_witch_action: dict[str, Any] = {}

    def _extract_game_id(self, packet: Mapping[str, Any]) -> str | None:
        """只接受明确的 game_id，避免把局内的普通 id 当成新局标识。"""
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
        # 缺失 game_id 时不清空；只有确认收到不同的非空标识才开始新局。
        if game_id is None or game_id == self._current_game_id:
            return
        self._current_game_id = game_id
        self._compact_history.clear()
        self._last_observe_signature = ""
        self._known_private_state.clear()
        self._last_witch_action.clear()

    async def observe(self, sync_packet: dict[str, Any]) -> None:
        """接收公开同步，并保留极简跨回合摘要。"""

        if not isinstance(sync_packet, Mapping):
            return
        self._ensure_game_context(sync_packet)
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
        if not isinstance(turn_packet, Mapping):
            raise ValueError("行动包必须是对象")
        self._ensure_game_context(turn_packet)
        if turn_packet["self"]["player_id"] != self.player_id:
            raise ValueError("行动包被发送给了错误玩家")
        private = turn_packet["private_information"]
        if str(private.get("role")) != self.profile.role:
            raise ValueError("Task-Agent 的角色档案与行动包身份不一致")

        self._update_private_state(private)
        system = self._system_prompt(private)
        decision_brief = self._witch_decision_summary(turn_packet)
        prompt = {
            "game": turn_packet["game"],
            "public_rules": turn_packet["public_rules"],
            "self": turn_packet["self"],
            "private_information": private,
            "public_state": turn_packet["public_state"],
            "request": turn_packet["request"],
            "compact_memory": self._render_compact_memory(),
            "self_resource_state": dict(self._known_private_state),
            "decision_brief": decision_brief,
            "witch_decision_summary": decision_brief,
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

        system = system + "\n\n【女巫专用决策规约】\n先看 decision_brief，再复盘最近票型与死亡。救药只救高价值且能改变轮次的目标；毒药只打高置信狼人或悍跳核心；证据不足时默认保留资源。强对跳、平安夜、关键神职死亡或残局时必须重排优先级。公开发言只谈公开信息，不得泄露私密夜间信息。"

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
            action = self._safe_witch_action(
                action,
                turn_packet["request"],
                turn_packet,
                decision_brief.get("default_action") if isinstance(decision_brief, Mapping) else None,
            )
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
        """把运行时可能使用的字段名归一化；不确定的字段不解释成夜刀。"""
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
            # 明确使用过，或“已使用”和“可用”冲突，都按不可用处理。
            if used is True or (used is False and available is False):
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
        state: dict[str, Any] = {
            "antidote_available": antidote_available,
            "poison_available": poison_available,
            "antidote_used": antidote_used,
            "poison_used": poison_used,
        }
        for key in ("role", "team", "status"):
            if key in private:
                state[key] = private[key]

        attack_found, attack_value = first_present(
            ("current_attack_target", "current_night_attack_target", "night_attack_target",
             "attacked_player_id", "wolf_attack_target")
        )
        # 每个行动包都代表当前时点；目标缺失不能沿用上一夜的目标。
        state["current_attack_target"] = str(attack_value) if attack_found and attack_value is not None else None
        self._known_private_state.update(state)

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
            summary["claims"] = [self._compact_claim_event(item, round_info, phase_info) for item in claims[:8]]
            claim_threads = self._summarize_claim_threads(summary["claims"])
            if claim_threads:
                summary["claim_threads"] = claim_threads

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

    def _compact_claim_event(self, value: Any, round_info: Any = None, phase_info: Any = None) -> Any:
        if not isinstance(value, Mapping):
            return self._compact_value(value)
        compact = self._compact_value(value)
        if isinstance(compact, dict):
            for key, aliases in (
                ("speaker_id", ("speaker_id", "player_id", "claimant_id", "source_id")),
                ("target_id", ("target_id", "claimed_player_id", "subject_id", "candidate_id")),
                ("claimed_role", ("claimed_role", "role", "claim_role", "result")),
                ("claimed_team", ("claimed_team", "team", "claim_team")),
                ("verified", ("verified", "verification", "is_verified")),
                ("verified_by", ("verified_by", "verified_with", "checked_by")),
                ("conflicts", ("conflicts", "contradicts", "counterclaim_to", "conflict_with")),
                ("supports", ("supports", "supported_by", "same_as")),
                ("round", ("round", "game_round", "day_round")),
                ("phase", ("phase", "public_phase", "stage")),
            ):
                if key in compact:
                    continue
                for alias in aliases:
                    if alias in value:
                        compact[key] = value[alias]
                        break
            if round_info is not None and "round" not in compact:
                compact["round"] = round_info
            if phase_info is not None and "phase" not in compact:
                compact["phase"] = phase_info
            return compact
        return compact

    def _compact_value(self, value: Any) -> Any:
        if isinstance(value, Mapping):
            compact: dict[str, Any] = {}
            for key in (
                "player_id",
                "id",
                "target_id",
                "speaker_id",
                "claimant_id",
                "source_id",
                "claimed_player_id",
                "claimed_role",
                "claimed_team",
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
                "round",
                "phase",
                "verified",
                "verified_by",
                "verification",
                "support",
                "supports",
                "conflict",
                "conflicts",
                "counterclaim_to",
                "contradicts",
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

    def _summarize_claim_threads(self, claims: list[Any]) -> list[dict[str, Any]]:
        threads: dict[str, dict[str, Any]] = {}
        for claim in claims:
            if not isinstance(claim, Mapping):
                continue
            target = str(claim.get("target_id") or claim.get("claimed_player_id") or claim.get("speaker_id") or "")
            if not target:
                continue
            bucket = threads.setdefault(
                target,
                {
                    "target_id": target,
                    "speakers": [],
                    "claimed_roles": [],
                    "claimed_teams": [],
                    "verified_count": 0,
                    "conflict_count": 0,
                    "support_count": 0,
                    "rounds": [],
                },
            )
            speaker = claim.get("speaker_id")
            if speaker is not None:
                speaker_id = str(speaker)
                if speaker_id not in bucket["speakers"]:
                    bucket["speakers"].append(speaker_id)
            claimed_role = claim.get("claimed_role")
            if claimed_role is not None:
                role_text = str(claimed_role)
                if role_text not in bucket["claimed_roles"]:
                    bucket["claimed_roles"].append(role_text)
            claimed_team = claim.get("claimed_team")
            if claimed_team is not None:
                team_text = str(claimed_team)
                if team_text not in bucket["claimed_teams"]:
                    bucket["claimed_teams"].append(team_text)
            if claim.get("verified") is True:
                bucket["verified_count"] += 1
            if claim.get("conflicts"):
                bucket["conflict_count"] += 1
            if claim.get("supports"):
                bucket["support_count"] += 1
            claim_round = claim.get("round")
            if claim_round is not None:
                round_text = str(claim_round)
                if round_text not in bucket["rounds"]:
                    bucket["rounds"].append(round_text)
        ranked = sorted(
            threads.values(),
            key=lambda item: (-int(item["conflict_count"]) - int(item["verified_count"]), str(item["target_id"])),
        )
        return ranked[:4]

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
        request = turn_packet.get("request") or {}
        game = turn_packet.get("game") or {}
        phase = str(game.get("public_phase", game.get("phase", request.get("phase", ""))))
        allowed_actions = request.get("allowed_actions") if isinstance(request, Mapping) else None
        action_kinds = [str(item.get("kind")) for item in allowed_actions or [] if isinstance(item, Mapping)]
        public_state = turn_packet.get("public_state") if isinstance(turn_packet.get("public_state"), Mapping) else {}
        board_state = self._board_state_snapshot(turn_packet, public_state)
        if phase:
            board_state["phase"] = phase
        candidates = self._rank_witch_candidates(turn_packet, board_state)
        save_priority = self._build_save_priority(turn_packet, board_state)
        poison_priority = [item for item in candidates if item.get("kind") == "poison"][:3]
        heal_kind = next((kind for kind in action_kinds if "heal" in kind), None)
        poison_kind = next((kind for kind in action_kinds if "poison" in kind), None)
        pass_kind = next((kind for kind in action_kinds if "pass" in kind or kind == "wait"), None)
        recommended = self._recommend_witch_action(
            allowed_actions,
            board_state,
            save_priority,
            poison_priority,
            heal_kind,
            poison_kind,
            pass_kind,
            action_kinds[0] if action_kinds else None,
        )
        verified = [
            {"player_id": item["player_id"], "reasons": item["reasons"]}
            for item in candidates if item.get("hard_evidence", 0) > 0
        ]
        weak = [
            {"player_id": item["player_id"], "reasons": item["reasons"]}
            for item in candidates if item.get("hard_evidence", 0) == 0
        ]
        summary: dict[str, Any] = {
            "phase": phase,
            "round": board_state.get("round"),
            "alive_count": board_state.get("alive_count"),
            "resource_state": {
                key: self._known_private_state.get(key)
                for key in ("antidote_available", "poison_available", "antidote_used", "poison_used")
            },
            "current_attack_target": self._known_private_state.get("current_attack_target"),
            "legal_heal_targets": self._allowed_targets_for_kind(allowed_actions, "heal"),
            "legal_poison_targets": self._allowed_targets_for_kind(allowed_actions, "poison"),
            "verified_public_evidence": verified[:3],
            "weak_public_evidence": weak[:3],
            "save_candidate": save_priority[0] if save_priority else None,
            "poison_candidate": poison_priority[0] if poison_priority and poison_priority[0].get("poison_eligible") else None,
            "default_action": recommended,
            "evidence_notice": "玩家自称不等于验证；普通投票是弱证据；重复复述不增加证据；私密夜杀不可公开发言。",
        }
        if "night_witch" not in phase and not ("night" in phase and "witch" in phase):
            summary["default_action"] = {"kind": pass_kind or "pass", "reason": "not_night_witch_phase"}
        elif not action_kinds:
            summary["default_action"] = {"kind": "pass", "reason": "no_allowed_actions"}
        return summary

    def _safe_witch_action(
        self,
        action: dict[str, Any],
        request: Mapping[str, Any],
        turn_packet: Mapping[str, Any],
        recommendation: Any,
    ) -> dict[str, Any]:
        """模型输出之后的本地女巫闸门；只收紧女巫行动，不改变引擎 schema。"""
        kind = str(action.get("kind") or "")
        if "heal" not in kind and "poison" not in kind:
            return action
        game = turn_packet.get("game") if isinstance(turn_packet.get("game"), Mapping) else {}
        phase = str(game.get("public_phase", game.get("phase", request.get("phase", "")))).lower()
        is_witch_phase = "night_witch" in phase or ("night" in phase and "witch" in phase)
        allowed = request.get("allowed_actions") if isinstance(request, Mapping) else []
        fallback_kind = next((str(item.get("kind")) for item in allowed or []
                              if isinstance(item, Mapping) and ("pass" in str(item.get("kind")) or str(item.get("kind")) == "wait")), None)
        if not is_witch_phase:
            return self._safe_pass_action(action, request, fallback_kind)

        if "heal" in kind:
            target = self._known_private_state.get("current_attack_target")
            legal = self._allowed_targets_for_kind(allowed, "heal")
            available = self._known_private_state.get("antidote_available") is True and self._known_private_state.get("antidote_used") is not True
            if available and target is not None and str(target) == str(action.get("target_id")) and str(target) in legal:
                return action
        if "poison" in kind:
            candidate = (recommendation or {}).get("target_id")
            legal = self._allowed_targets_for_kind(allowed, "poison")
            available = self._known_private_state.get("poison_available") is True and self._known_private_state.get("poison_used") is not True
            if available and (recommendation or {}).get("confidence") == "high" and candidate is not None and str(candidate) == str(action.get("target_id")) and str(candidate) in legal:
                return action
        return self._safe_pass_action(action, request, fallback_kind)

    def _safe_pass_action(self, action: dict[str, Any], request: Mapping[str, Any], fallback_kind: str | None) -> dict[str, Any]:
        if not fallback_kind:
            return action
        return {"request_id": request.get("request_id"), "player_id": request.get("player_id", self.player_id), "kind": fallback_kind}

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

    def _board_state_snapshot(self, turn_packet: Mapping[str, Any], public_state: Mapping[str, Any]) -> dict[str, Any]:
        game = turn_packet.get("game") if isinstance(turn_packet.get("game"), Mapping) else {}
        summary: dict[str, Any] = {}
        for source in (public_state, game):
            if not isinstance(source, Mapping):
                continue
            for key in ("round", "day_round", "night_round"):
                if key in source and "round" not in summary:
                    summary["round"] = source[key]
                    break
            for key in ("phase", "public_phase", "stage"):
                if key in source and "phase" not in summary:
                    summary["phase"] = source[key]
                    break
            for key in ("alive_players", "dead_players", "players", "votes", "deaths", "claims", "sheriff"):
                if key in source and key not in summary:
                    value = source[key]
                    summary[key] = self._compact_value(value) if not isinstance(value, list) else self._compact_entities(value)
        alive = self._extract_player_ids(public_state.get("alive_players"))
        if not alive:
            alive = self._extract_alive_from_players(public_state.get("players"))
        if alive:
            summary["alive_count"] = len(alive)
            summary["alive_players"] = alive[:12]
        dead = self._extract_player_ids(public_state.get("dead_players"))
        if not dead:
            dead = self._extract_player_ids(public_state.get("deaths"))
        if dead:
            summary["dead_count"] = len(dead)
            summary["dead_players"] = dead[:12]
        return summary

    def _collect_recent_claim_threads(self) -> list[dict[str, Any]]:
        threads: list[dict[str, Any]] = []
        for snapshot in self._compact_history[-4:]:
            if not isinstance(snapshot, Mapping):
                continue
            for claim in snapshot.get("claims", []):
                if not isinstance(claim, Mapping):
                    continue
                threads.append({
                    key: claim[key]
                    for key in (
                        "speaker_id",
                        "target_id",
                        "claimed_role",
                        "claimed_roles",
                        "claimed_team",
                        "verified",
                        "conflicts",
                        "supports",
                        "round",
                        "phase",
                    )
                    if key in claim
                })
            for thread in snapshot.get("claim_threads", []):
                if not isinstance(thread, Mapping):
                    continue
                threads.append({
                    key: thread[key]
                    for key in (
                        "speaker_id",
                        "target_id",
                        "claimed_role",
                        "claimed_roles",
                        "claimed_team",
                        "verified",
                        "conflicts",
                        "supports",
                        "round",
                        "phase",
                    )
                    if key in thread
                })
        return threads[-8:]

    def _rank_witch_candidates(self, turn_packet: Mapping[str, Any], board_state: Mapping[str, Any]) -> list[dict[str, Any]]:
        """只把分层且相互独立的公开证据交给毒药建议。"""
        request = turn_packet.get("request") if isinstance(turn_packet.get("request"), Mapping) else {}
        allowed_actions = request.get("allowed_actions") if isinstance(request, Mapping) else None
        poison_kind = next((str(item.get("kind")) for item in allowed_actions or []
                            if isinstance(item, Mapping) and "poison" in str(item.get("kind") or "")), None)
        if not poison_kind:
            return []
        legal = set(self._allowed_targets_for_kind(allowed_actions, "poison"))
        alive = set(self._extract_player_ids(board_state.get("alive_players")))
        dead = set(self._extract_player_ids(board_state.get("dead_players")))
        if not alive:
            alive = legal | {self.player_id}

        candidates: dict[str, dict[str, Any]] = {}
        seen_claims: set[str] = set()
        seen_votes: set[str] = set()

        def bucket_for(target: Any) -> dict[str, Any] | None:
            pid = str(target) if target is not None else ""
            if not pid or pid == "None" or pid == self.player_id or pid not in legal or pid in dead or pid not in alive:
                return None
            return candidates.setdefault(pid, {
                "player_id": pid, "kind": "poison", "score": 0,
                "hard_evidence": 0, "independent_support": 0, "weak_evidence": 0,
                "reasons": [], "sources": [], "poison_eligible": False,
            })

        def add(bucket: dict[str, Any], points: int, field: str, reason: str, source: Any = None) -> None:
            bucket["score"] += points
            bucket[field] += 1
            if reason not in bucket["reasons"]:
                bucket["reasons"].append(reason)
            if source is not None and str(source) not in bucket["sources"]:
                bucket["sources"].append(str(source))

        claim_sources: list[Mapping[str, Any]] = []
        for snapshot in self._compact_history[-4:]:
            if isinstance(snapshot, Mapping):
                claim_sources.extend(item for item in snapshot.get("claims", []) if isinstance(item, Mapping))
        if isinstance(board_state.get("claims"), list):
            claim_sources.extend(item for item in board_state["claims"] if isinstance(item, Mapping))
        vote_sources: list[Mapping[str, Any]] = []
        for snapshot in self._compact_history[-4:]:
            if isinstance(snapshot, Mapping):
                vote_sources.extend(item for item in snapshot.get("votes", []) if isinstance(item, Mapping))
        if isinstance(board_state.get("votes"), list):
            vote_sources.extend(item for item in board_state["votes"] if isinstance(item, Mapping))

        for claim in claim_sources:
            fingerprint = json.dumps(self._compact_value(claim), ensure_ascii=False, sort_keys=True, default=str)
            if fingerprint in seen_claims:
                continue
            seen_claims.add(fingerprint)
            target = claim.get("target_id") or claim.get("claimed_player_id")
            bucket = bucket_for(target)
            if bucket is None:
                continue
            text = " ".join(str(claim.get(key) or "") for key in
                            ("claimed_role", "claimed_team", "result", "type", "reason")).lower()
            wolf = any(token in text for token in ("wolf", "狼人", "查杀", "kill"))
            source = claim.get("speaker_id") or claim.get("claimant_id") or claim.get("source_id")
            verified = claim.get("verified") is True or claim.get("verification") in (True, "verified")
            explicit_check = any(token in text for token in ("查杀", "seer_check", "prophet_check", "检查结果"))
            conflict = bool(claim.get("conflicts") or claim.get("conflict") or claim.get("contradicts"))
            if wolf and (verified or explicit_check):
                add(bucket, 3, "hard_evidence", "verified_public_wolf_check", source)
            elif wolf and conflict:
                add(bucket, 2, "independent_support", "strong_counterclaim_conflict", source)
            elif wolf:
                # 普通自称/单人质疑只是弱证据，不能单独消耗毒药。
                add(bucket, 1, "weak_evidence", "single_public_suspicion", source)

        # 票型只计作独立来源；同一投票事件重复出现在多个摘要中会被去重。
        for vote in vote_sources:
            fingerprint = json.dumps(self._compact_value(vote), ensure_ascii=False, sort_keys=True, default=str)
            if fingerprint in seen_votes:
                continue
            seen_votes.add(fingerprint)
            target = vote.get("target_id") or vote.get("vote_target_id")
            bucket = bucket_for(target)
            if bucket is None:
                continue
            add(bucket, 1, "weak_evidence", "ordinary_vote_pressure", vote.get("round"))

        result: list[dict[str, Any]] = []
        for candidate in candidates.values():
            # 至少一个硬证据，或两个真正独立的中强来源；普通票和单人质疑不能触发。
            candidate["poison_eligible"] = bool(
                candidate["hard_evidence"] >= 1 or candidate["independent_support"] >= 2
            )
            candidate["confidence"] = "high" if candidate["poison_eligible"] else "low"
            result.append(candidate)
        return sorted(result, key=lambda item: (-int(item["poison_eligible"]), -int(item["score"]), str(item["player_id"])))[:4]

    def _build_save_priority(self, turn_packet: Mapping[str, Any], board_state: Mapping[str, Any]) -> list[dict[str, Any]]:
        private = self._known_private_state
        target = private.get("current_attack_target")
        heal_targets = self._allowed_targets_for_kind(
            (turn_packet.get("request") or {}).get("allowed_actions"), "heal"
        )
        alive = set(self._extract_player_ids(board_state.get("alive_players")))
        if target is None or str(target) not in heal_targets or str(target) == self.player_id:
            return []
        if str(target) in set(self._extract_player_ids(board_state.get("dead_players"))):
            return []
        if alive and str(target) not in alive:
            return []
        reasons = ["current_private_night_attack"]
        if int(board_state.get("alive_count") or 0) <= 5:
            reasons.append("endgame_pressure")
        if self._is_high_value_save_target(target, board_state):
            reasons.append("high_value_target")
        return [{"kind": "heal", "player_id": str(target), "score": 5, "reasons": reasons}]

    def _recommend_witch_action(
        self,
        allowed_actions: Any,
        board_state: Mapping[str, Any],
        save_priority: list[dict[str, Any]],
        poison_priority: list[dict[str, Any]],
        heal_kind: str | None,
        poison_kind: str | None,
        pass_kind: str | None,
        fallback_kind: str | None,
    ) -> dict[str, Any]:
        private = self._known_private_state
        phase = str(board_state.get("phase") or "").lower()
        if not ("night_witch" in phase or ("night" in phase and "witch" in phase)):
            return {"kind": pass_kind or "pass", "reason": "not_night_witch_phase", "confidence": "low", "priority": 3}
        # 未知资源状态按不可用处理，避免字段缺失时误用一次性药水。
        heal_available = bool(heal_kind) and private.get("antidote_available") is True and private.get("antidote_used") is not True
        poison_available = bool(poison_kind) and private.get("poison_available") is True and private.get("poison_used") is not True
        heal_targets = self._allowed_targets_for_kind(allowed_actions, "heal")
        poison_targets = self._allowed_targets_for_kind(allowed_actions, "poison")
        endgame_pressure = self._endgame_pressure(board_state)
        if heal_available and save_priority:
            top_save = save_priority[0]
            target_id = self._select_legal_target(top_save.get("player_id"), heal_targets)
            if target_id and (endgame_pressure in ("high", "critical") or self._is_high_value_save_target(target_id, board_state)):
                return {"kind": heal_kind, "target_id": target_id, "reason": "high_value_save", "confidence": "high", "priority": 1}
        if poison_available and poison_priority:
            top_poison = poison_priority[0]
            if top_poison.get("poison_eligible") is True:
                target_id = self._select_legal_target(top_poison.get("player_id"), poison_targets)
                if target_id:
                    return {"kind": poison_kind, "target_id": target_id, "reason": ",".join(top_poison.get("reasons", [])) or "high_confidence_wolf", "confidence": "high", "priority": 1}
        if heal_available and save_priority:
            top_save = save_priority[0]
            if int(board_state.get("alive_count") or 0) <= 7:
                target_id = self._select_legal_target(top_save.get("player_id"), heal_targets)
                if target_id:
                    return {"kind": heal_kind, "target_id": target_id, "reason": "endgame_preserve_life", "confidence": "medium", "priority": 2}
        # 没有高置信候选时保留唯一毒药，即使处于残局也不因普通票型强行出手。
        if pass_kind:
            return {"kind": pass_kind, "reason": "evidence_insufficient", "confidence": "low", "priority": 3}
        if fallback_kind:
            fallback_targets = self._allowed_targets_for_kind(allowed_actions, fallback_kind)
            fallback_target = fallback_targets[0] if fallback_targets else None
            action = {"kind": fallback_kind, "reason": "evidence_insufficient", "confidence": "low", "priority": 3}
            if fallback_target is not None:
                action["target_id"] = fallback_target
            return action
        return {"kind": "pass", "reason": "evidence_insufficient", "confidence": "low", "priority": 3}

    def _endgame_pressure(self, board_state: Mapping[str, Any]) -> str:
        alive = self._nonnegative_int(board_state.get("alive_count"), fallback=0) or 0
        round_no = self._nonnegative_int(board_state.get("round"), fallback=0) or 0
        if alive <= 4 or round_no >= 5:
            return "critical"
        if alive <= 6 or round_no >= 4:
            return "high"
        if alive <= 8 or round_no >= 3:
            return "medium"
        return "low"

    def _is_high_value_save_target(self, player_id: Any, board_state: Mapping[str, Any]) -> bool:
        if player_id is None:
            return False
        target = str(player_id)
        for claim in self._collect_recent_claim_threads():
            if str(claim.get("target_id")) != target:
                continue
            roles = claim.get("claimed_roles")
            if not isinstance(roles, list):
                roles = [claim.get("claimed_role")]
            normalized_roles = {str(role).lower() for role in roles if role is not None}
            if normalized_roles & {"seer", "prophet", "witch", "guardian", "sheriff"}:
                return True
        sheriff = board_state.get("sheriff")
        if isinstance(sheriff, Mapping):
            if str(sheriff.get("player_id") or sheriff.get("id")) == target:
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

    def _allowed_targets_for_kind(self, allowed_actions: Any, needle: str) -> list[str]:
        if not isinstance(allowed_actions, list):
            return []
        targets: list[str] = []
        for item in allowed_actions:
            if not isinstance(item, Mapping):
                continue
            kind = str(item.get("kind") or "")
            if needle not in kind:
                continue
            raw_targets = item.get("target_ids")
            if isinstance(raw_targets, list):
                for target in raw_targets:
                    if target is not None:
                        target_id = str(target)
                        if target_id not in targets:
                            targets.append(target_id)
        return targets

    def _select_legal_target(self, preferred: Any, legal_targets: list[str]) -> str | None:
        if preferred is None or not legal_targets:
            return None
        preferred_id = str(preferred)
        return preferred_id if preferred_id in legal_targets else None

    def _extract_alive_from_players(self, value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        alive: list[str] = []
        for item in value:
            if not isinstance(item, Mapping):
                continue
            if item.get("alive") is False:
                continue
            for key in ("player_id", "id"):
                if key in item and item[key] is not None:
                    alive.append(str(item[key]))
                    break
        return alive

    @staticmethod
    def _nonnegative_int(value: object, *, fallback: int | None) -> int | None:
        if isinstance(value, bool):
            return fallback
        try:
            normalized = int(value)
        except (TypeError, ValueError):
            return fallback
        return normalized if normalized >= 0 else fallback

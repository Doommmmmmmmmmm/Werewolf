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
        self._compact_history: list[dict[str, Any]] = []
        self._last_observe_signature = ""
        self._current_game_id: str | None = None
        self._known_private_state: dict[str, Any] = {}
        self._last_witch_action: dict[str, Any] = {}
        # 仅保存行动包中明确给出的当前/上一夜狼刀目标；不从公开信息反推。
        self._current_night_attack_target: str | None = None
        self._last_night_attack_target: str | None = None

    async def observe(self, sync_packet: dict[str, Any]) -> None:
        """接收公开同步，并保留极简跨回合摘要。"""

        if not isinstance(sync_packet, Mapping):
            return
        game_id = self._extract_game_id(sync_packet)
        if game_id is not None and game_id != self._current_game_id:
            self._reset_game_memory(game_id)
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
            "witch_decision_summary": self._witch_decision_summary(turn_packet),
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

        system = system + "\n\n【女巫专用决策规约】\n决策顺序固定为：先读取本夜私有狼刀目标，再核对当前合法行动与药水状态，再判断公开高价值目标和残局压力，最后才考虑 pass。若本夜狼刀目标是公开警长、公开有连续查验链的预言家或其他有独立公开支持的关键好人，且 witch_heal 合法，必须先比较救援收益，不能仅因保留资源直接 pass。poison 目标若等于本夜狼刀目标，视为重复击杀；除非规则明确提供额外收益，否则不得推荐，且有合法 pass 时应 pass。残局压力只降低救援评估的 pass 门槛，不降低毒药所需的强对跳、双重独立公开证据或稳定公开冲突门槛。公开发言只谈公开事实，不得泄露药水余量、夜刀目标或自己未公开的用药结果。"

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
            witch_summary = prompt["witch_decision_summary"]
            action = self._apply_witch_safety_fallback(action, turn_packet, witch_summary)
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
        for key in ("role", "team", "status"):
            if key in private:
                extracted[key] = private[key]

        alias_groups = {
            "can_heal": ("can_heal", "antidote_available", "heal_available", "has_antidote"),
            "can_poison": ("can_poison", "poison_available", "has_poison"),
            "heal_used": ("heal_used", "antidote_used", "used_heal", "used_antidote"),
            "poison_used": ("poison_used", "used_poison"),
        }
        for canonical, aliases in alias_groups.items():
            value = self._pick_first_mapping_value(private, *aliases)
            if value is not None:
                normalized = self._coerce_bool(value)
                extracted[canonical] = normalized if normalized is not None else value

        if "can_heal" in extracted:
            extracted["antidote_available"] = extracted["can_heal"]
        if "can_poison" in extracted:
            extracted["poison_available"] = extracted["can_poison"]
        if "heal_used" in extracted:
            extracted["antidote_used"] = extracted["heal_used"]
            extracted["used_heal"] = extracted["heal_used"]
        if "poison_used" in extracted:
            extracted["used_poison"] = extracted["poison_used"]

        for key, value in extracted.items():
            self._known_private_state[key] = value

        # 字段名可能随行动包版本变化，但只接受这组明确的直接别名。不要扫描
        # 任意文本或递归对象，以免把别的私有信息误当成狼刀目标。
        attack_aliases = (
            "wolf_target",
            "wolf_attack_target",
            "night_attack_target",
            "attack_target",
            "attacked_player_id",
            "kill_target",
            "night_kill_target",
        )
        raw_attack_target = self._pick_first_mapping_value(private, *attack_aliases)
        self._current_night_attack_target = self._normalize_player_id(raw_attack_target)
        self._known_private_state["current_night_attack_target"] = self._current_night_attack_target

    def _extract_game_id(self, packet: Mapping[str, Any]) -> str | None:
        sources: list[tuple[Mapping[str, Any], tuple[str, ...]]] = [(packet, ("game_id", "gameId", "match_id", "matchId"))]
        game = packet.get("game")
        if isinstance(game, Mapping):
            sources.append((game, ("game_id", "gameId", "match_id", "matchId", "id")))
        public_state = packet.get("public_state")
        if isinstance(public_state, Mapping):
            sources.append((public_state, ("game_id", "gameId", "match_id", "matchId")))
        for source, keys in sources:
            for key in keys:
                value = source.get(key)
                if value not in (None, ""):
                    return str(value)
        return None

    def _reset_game_memory(self, game_id: str | None) -> None:
        self._current_game_id = game_id
        self._compact_history.clear()
        self._last_observe_signature = ""
        self._known_private_state.clear()
        self._last_witch_action.clear()
        self._current_night_attack_target = None
        self._last_night_attack_target = None

    @staticmethod
    def _coerce_bool(value: Any) -> bool | None:
        if isinstance(value, bool):
            return value
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "1", "yes", "y", "on", "available", "alive"}:
                return True
            if normalized in {"false", "0", "no", "n", "off", "used", "spent", "dead"}:
                return False
        return bool(value)

    @staticmethod
    def _pick_first_mapping_value(source: Mapping[str, Any], *names: str) -> Any:
        for name in names:
            if name in source:
                return source[name]
        return None

    @staticmethod
    def _normalize_player_id(value: Any) -> str | None:
        """只接受简单编号或目标字段中的明确 id，不扫描任意嵌套内容。"""
        if isinstance(value, Mapping):
            source = value
            value = None
            for key in ("target_id", "player_id", "id"):
                candidate = source.get(key)
                if candidate not in (None, ""):
                    value = candidate
                    break
            if value is None:
                return None
        if value is None or isinstance(value, (bool, Mapping, list, tuple, set)):
            return None
        if isinstance(value, (str, int)):
            normalized = str(value).strip()
            return normalized or None
        return None

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
                "text",
                "message",
                "content",
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
        current_target = self._current_night_attack_target
        target_id = self._normalize_player_id(action.get("target_id"))
        resource_before = {
            key: self._known_private_state[key]
            for key in ("can_heal", "can_poison", "heal_used", "poison_used")
            if key in self._known_private_state
        }
        resource_after = dict(resource_before)
        if "heal" in str(action.get("kind") or ""):
            if "can_heal" in resource_after:
                resource_after["can_heal"] = False
            if "heal_used" in resource_after:
                resource_after["heal_used"] = True
        if "poison" in str(action.get("kind") or ""):
            if "can_poison" in resource_after:
                resource_after["can_poison"] = False
            if "poison_used" in resource_after:
                resource_after["poison_used"] = True
        self._last_night_attack_target = current_target
        self._last_witch_action = {
            "phase": phase,
            "kind": action.get("kind"),
            "target_id": target_id,
            "was_current_attack_target": bool(target_id and target_id == current_target),
            "resource_before": resource_before,
            "resource_after": resource_after,
        }
        self._known_private_state["last_night_attack_target"] = current_target
        self._known_private_state["last_night_action_kind"] = action.get("kind")
        self._known_private_state["last_night_action_target"] = target_id
        self._known_private_state["last_witch_action"] = dict(self._last_witch_action)

    def _public_high_value_targets(
        self,
        turn_packet: Mapping[str, Any],
        public_state: Mapping[str, Any],
        game: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        """从公开资料提取救援价值，不把身份推测升级为系统确认。"""
        sources = [public_state, game, turn_packet]
        target_keys = ("player_id", "id", "target_id", "speaker_id", "author_id")

        def first_value(names: tuple[str, ...]) -> Any:
            for source in sources:
                for name in names:
                    if isinstance(source, Mapping) and source.get(name) not in (None, ""):
                        return source[name]
            return None

        def get_id(value: Any) -> str | None:
            if isinstance(value, Mapping):
                for key in target_keys:
                    candidate = self._normalize_player_id(value.get(key))
                    if candidate:
                        return candidate
                return None
            return self._normalize_player_id(value)

        values: dict[str, dict[str, Any]] = {}
        def add(player_id: str | None, reason: str) -> None:
            if not player_id:
                return
            entry = values.setdefault(player_id, {"target_id": player_id, "reasons": []})
            if reason not in entry["reasons"]:
                entry["reasons"].append(reason)

        sheriff = first_value(("sheriff", "police", "captain", "警长"))
        sheriff_id = get_id(sheriff)
        if sheriff_id:
            add(sheriff_id, "公开警长")

        record_lists: list[list[Any]] = []
        tool_context = turn_packet.get("tool_context")
        if isinstance(tool_context, Mapping) and isinstance(tool_context.get("current_round_dialogue"), list):
            record_lists.append(tool_context["current_round_dialogue"])
        for source in sources:
            if not isinstance(source, Mapping):
                continue
            for key in ("claims", "reveals", "public_claims", "role_claims", "counterclaims", "dialogue", "speeches", "messages", "public_dialogue"):
                value = source.get(key)
                if isinstance(value, list):
                    record_lists.append(value)
        # compact_memory 只保留公开同步摘要；它不含私有夜刀字段。
        for snapshot in self._compact_history[-4:]:
            if isinstance(snapshot, Mapping):
                for key in ("claims", "dialogue", "speeches", "messages"):
                    value = snapshot.get(key)
                    if isinstance(value, list):
                        record_lists.append(value)

        seer_counts: dict[str, int] = {}
        seen_records: set[str] = set()
        for records in record_lists:
            for record in records[:12]:
                if not isinstance(record, Mapping):
                    continue
                record_signature = json.dumps(record, ensure_ascii=False, sort_keys=True, default=str)
                if record_signature in seen_records:
                    continue
                seen_records.add(record_signature)
                player_id = get_id(record)
                if not player_id:
                    continue
                blob = json.dumps(record, ensure_ascii=False, separators=(",", ":")).lower()
                seer_claim = any(token in blob for token in ("seer", "fortune_teller", "oracle", "预言家", "占卜师"))
                check_count = sum(blob.count(token) for token in ("查验", "验人", "checked", "check_result", "seer_result"))
                result_value = record.get("results", record.get("checks", record.get("check_results")))
                if isinstance(result_value, list):
                    check_count += len(result_value)
                if seer_claim:
                    seer_counts[player_id] = seer_counts.get(player_id, 0) + max(1, check_count)
                if any(token in blob for token in ("confirmed", "verified", "revealed", "trusted", "锁定", "确认")) and any(
                    token in blob for token in ("seer", "guard", "医生", "预言家", "守卫", "女巫")
                ):
                    add(player_id, "公开确认的关键神职声明")

        for player_id, count in seer_counts.items():
            if count >= 2:
                add(player_id, "公开预言家且有连续查验链")
        result = list(values.values())
        for entry in result:
            entry["high_value_for_heal"] = True
        return result[:6]

    def _endgame_pressure(self, alive_count: Any, recent_deaths: list[Any], confirmed: list[dict[str, Any]]) -> str:
        alive = self._nonnegative_int(alive_count, fallback=None)
        wolf_count = 0
        for record in confirmed:
            blob = json.dumps(record, ensure_ascii=False).lower()
            if any(token in blob for token in ("wolf", "werewolf", "狼人", "狼")):
                wolf_count += 1
        if alive is not None and (alive <= 5 or (wolf_count >= 2 and alive <= 7)):
            return "high"
        if alive is not None and (alive <= 8 or len(recent_deaths) >= 3):
            return "medium"
        return "low"

    def _witch_decision_summary(self, turn_packet: Mapping[str, Any]) -> dict[str, Any]:
        request = turn_packet.get("request") or {}
        game = turn_packet.get("game") or {}
        public_state = turn_packet.get("public_state") or {}
        phase = str(game.get("public_phase", game.get("phase", "")))
        current_game_id = self._current_game_id or self._extract_game_id(turn_packet)

        allowed_actions = request.get("allowed_actions") if isinstance(request, Mapping) else None
        action_kinds = [str(item.get("kind")) for item in allowed_actions or [] if isinstance(item, Mapping) and item.get("kind")]
        heal_kind = next((kind for kind in action_kinds if "heal" in kind or "antidote" in kind), None)
        poison_kind = next((kind for kind in action_kinds if "poison" in kind), None)
        pass_kind = next((kind for kind in action_kinds if "pass" in kind or kind == "wait"), None)

        allowed_targets_by_kind: dict[str, set[str]] = {}
        for item in allowed_actions or []:
            if not isinstance(item, Mapping):
                continue
            kind = str(item.get("kind") or "")
            target_ids = item.get("target_ids")
            if kind and isinstance(target_ids, list) and target_ids:
                allowed_targets_by_kind.setdefault(kind, set()).update(
                    str(target_id) for target_id in target_ids if target_id not in (None, "")
                )

        def first_value(sources: list[Any], *names: str) -> Any:
            for source in sources:
                if not isinstance(source, Mapping):
                    continue
                for name in names:
                    value = source.get(name)
                    if value not in (None, ""):
                        return value
            return None

        def as_list(value: Any) -> list[Any]:
            return list(value) if isinstance(value, list) else []

        def target_id(record: Any) -> str | None:
            if not isinstance(record, Mapping):
                return None
            for key in ("target_id", "player_id", "id", "speaker_id", "voter_id", "vote_target_id", "target"):
                value = record.get(key)
                if value not in (None, ""):
                    return str(value)
            return None

        def alive_state(record: Any) -> bool | None:
            if not isinstance(record, Mapping):
                return None
            if "alive" in record:
                coerced = self._coerce_bool(record.get("alive"))
                if coerced is not None:
                    return coerced
            status = str(record.get("status") or "").strip().lower()
            if status in {"alive", "living", "surviving", "survivor"}:
                return True
            if status in {"dead", "eliminated", "killed", "deceased"}:
                return False
            return None

        def text_blob(record: Mapping[str, Any]) -> str:
            return json.dumps(self._compact_value(record), ensure_ascii=False, separators=(",", ":")).lower()

        summary: dict[str, Any] = {
            "phase": phase,
            "request_kind": request.get("kind"),
            "current_game_id": current_game_id,
            "last_witch_action": dict(self._last_witch_action) if self._last_witch_action else {},
        }

        alive_count = first_value([public_state, game, turn_packet], "alive_count", "living_count", "survivor_count")
        alive_players = as_list(first_value([public_state, game, turn_packet], "alive_players", "living_players", "survivors"))
        if alive_count is None and alive_players:
            alive_flags = [alive_state(item) for item in alive_players]
            known_alive_flags = [flag for flag in alive_flags if flag is not None]
            if known_alive_flags:
                alive_count = sum(1 for flag in known_alive_flags if flag)
            else:
                alive_count = len(alive_players)
        if alive_count is not None:
            normalized_alive_count = self._nonnegative_int(alive_count, fallback=None)
            if normalized_alive_count is not None:
                alive_count = normalized_alive_count

        recent_deaths = as_list(first_value([public_state, game, turn_packet], "deaths", "dead_players", "night_deaths", "death_events"))
        vote_records = as_list(first_value([public_state, game, turn_packet], "votes", "voting", "vote_results", "vote_events"))
        claim_records = as_list(first_value([public_state, game, turn_packet], "claims", "reveals", "public_claims", "role_claims", "counterclaims"))
        current_round = first_value([public_state, game, turn_packet], "round", "game_round", "day_round")
        normalized_round = self._nonnegative_int(current_round, fallback=None)
        if normalized_round is not None:
            current_round = normalized_round
        high_value_targets = self._public_high_value_targets(turn_packet, public_state, game)
        high_value_ids = {item["target_id"] for item in high_value_targets}

        vote_totals: dict[str, int] = {}
        for record in vote_records:
            if not isinstance(record, Mapping):
                continue
            tid = target_id(record)
            if not tid:
                continue
            count = self._nonnegative_int(record.get("count"), fallback=1) or 1
            vote_totals[tid] = vote_totals.get(tid, 0) + count
        vote_clusters = [
            {"target_id": tid, "count": count, "reason": "票型聚集"}
            for tid, count in sorted(vote_totals.items(), key=lambda item: (-item[1], item[0]))[:3]
        ]

        public_counterclaims: list[dict[str, Any]] = []
        confirmed_identities: list[dict[str, Any]] = []
        for record in claim_records:
            if not isinstance(record, Mapping):
                continue
            blob = text_blob(record)
            compact = self._compact_value(record)
            claim_target = target_id(record)
            is_counterclaim = any(token in blob for token in ("counterclaim", "challenge", "conflict", "悍跳", "对跳", "冲突"))
            is_confirmed = any(token in blob for token in ("confirmed", "verified", "revealed", "trusted", "lock", "锁定"))
            if is_counterclaim and claim_target:
                public_counterclaims.append({"target_id": claim_target, "evidence": compact})
            if is_confirmed and (record.get("role") is not None or record.get("team") is not None):
                confirmed_identities.append(compact)

        resource_state = {
            key: self._known_private_state[key]
            for key in (
                "role",
                "team",
                "status",
                "can_heal",
                "can_poison",
                "antidote_available",
                "poison_available",
                "heal_used",
                "poison_used",
                "antidote_used",
                "used_heal",
                "used_poison",
            )
            if key in self._known_private_state
        }
        endgame_pressure = self._endgame_pressure(alive_count, recent_deaths, confirmed_identities)
        board_snapshot = {
            "alive_count": alive_count,
            "recent_deaths": [self._compact_value(item) for item in recent_deaths[:3]],
            "sheriff": self._compact_value(first_value([public_state, game, turn_packet], "sheriff", "police", "警长", "captain")),
            "public_counterclaims": public_counterclaims[:3],
            "vote_clusters": vote_clusters,
            "confirmed_identities": confirmed_identities[:3],
            "resource_state": resource_state,
            "public_high_value_targets": high_value_targets,
            "endgame_pressure": endgame_pressure,
        }

        def add_poison_candidate(store: dict[str, dict[str, Any]], candidate_target: str | None, score: int, reason: str) -> None:
            if not candidate_target:
                return
            allowed_targets = allowed_targets_by_kind.get(poison_kind or "", set())
            if allowed_targets and candidate_target not in allowed_targets:
                return
            entry = store.setdefault(
                candidate_target,
                {"kind": poison_kind, "target_id": candidate_target, "score": score, "reasons": []},
            )
            entry["score"] = max(int(entry["score"]), score)
            reasons = entry.setdefault("reasons", [])
            if reason not in reasons:
                reasons.append(reason)

        heal_candidates: list[dict[str, Any]] = []
        heal_targets = allowed_targets_by_kind.get(heal_kind or "", set())
        current_target = self._current_night_attack_target
        if current_target and current_target in high_value_ids and (not heal_targets or current_target in heal_targets):
            heal_candidates.append({
                "kind": heal_kind,
                "candidate_type": "heal_current_target",
                "target_id": current_target,
                "score": 150,
                "high_value_for_heal": True,
                "reason": "本夜私有狼刀目标；公开警长/可信预言家或连续查验链，优先比较救药收益",
            })
        for index, record in enumerate(recent_deaths[:3]):
            if not isinstance(record, Mapping):
                continue
            candidate = target_id(record)
            if not candidate:
                continue
            if heal_targets and candidate not in heal_targets:
                continue
            score = 100 - index * 10
            reason = "最近死亡，优先救援" if index == 0 else "较新的死亡记录，可作为备选救援"
            blob = text_blob(record)
            if any(token in blob for token in ("night", "night_kill", "kill", "killed", "刀", "狼刀")):
                score += 5
                reason += "；疑似夜刀"
            if candidate == current_target:
                continue
            heal_candidates.append({
                "kind": heal_kind,
                "candidate_type": "heal_recent_death",
                "target_id": candidate,
                "score": score,
                "high_value_for_heal": candidate in high_value_ids,
                "reason": reason + ("；公开高价值目标" if candidate in high_value_ids else ""),
            })

        poison_store: dict[str, dict[str, Any]] = {}
        for item in vote_clusters:
            candidate = item.get("target_id")
            if not candidate:
                continue
            score = int(item.get("count", 0)) * 10 + 50
            add_poison_candidate(poison_store, candidate, score, f"票型聚集 {item.get('count', 0)} 票")
        for item in public_counterclaims:
            add_poison_candidate(poison_store, item.get("target_id"), 85, "公开对跳/身份冲突")
        poison_candidates = []
        for entry in sorted(poison_store.values(), key=lambda item: (-int(item["score"]), str(item["target_id"]))):
            candidate = entry.get("target_id")
            redundant = bool(candidate and candidate == current_target)
            reasons = entry.get("reasons", [])
            strong_conflict = any("对跳" in str(reason) or "身份冲突" in str(reason) for reason in reasons)
            independently_supported = len(reasons) >= 2 or strong_conflict
            recommended = not redundant and independently_supported
            poison_candidates.append(
                {
                    "kind": entry.get("kind"),
                    "candidate_type": "poison_candidate",
                    "target_id": candidate,
                    "score": entry.get("score") if recommended else 0,
                    "evidence_classes": len(reasons),
                    "poison_redundant": redundant,
                    "recommended": recommended,
                    "reason": "；".join(reasons) + (
                        "；与本夜狼刀目标重复，不推荐" if redundant else
                        "；证据类别不足，不推荐" if not independently_supported else ""
                    ),
                }
            )

        available_actions = [
            {"kind": str(item.get("kind")), "target_ids": [str(value) for value in item.get("target_ids", [])]}
            for item in allowed_actions or []
            if isinstance(item, Mapping) and item.get("kind")
        ]
        summary.update({
            "current_night_attack_target": current_target,
            "last_night_attack_target": self._last_night_attack_target,
            "current_round": current_round,
            "alive_count": alive_count,
            "available_actions": available_actions,
            "resource_state": resource_state,
            "public_high_value_targets": high_value_targets,
            "public_counterclaims": public_counterclaims[:3],
            "recent_vote_pressure": vote_clusters,
            "endgame_pressure": endgame_pressure,
        })

        pass_reason = "证据不足时默认保留资源；先评估本夜狼刀目标的救援收益"
        if heal_candidates or poison_candidates:
            pass_reason = "作为保守备选，证据不足或收益不高时再 pass"

        summary["board_snapshot"] = board_snapshot
        summary["candidate_ranking"] = {
            "heal": heal_candidates[:2],
            "poison": poison_candidates[:3],
            "pass": {
                "kind": pass_kind or "pass",
                "score": 40 if not (heal_candidates or any(item.get("recommended") for item in poison_candidates)) else 10,
                "reason": pass_reason,
            },
        }
        return summary

    def _apply_witch_safety_fallback(
        self,
        action: dict[str, Any],
        turn_packet: Mapping[str, Any],
        summary: Mapping[str, Any],
    ) -> dict[str, Any]:
        """仅修正两种明确的女巫收益/安全错误，仍交由原有校验确认合法性。"""
        request = turn_packet.get("request") or {}
        allowed = request.get("allowed_actions") if isinstance(request, Mapping) else []
        if not isinstance(allowed, list):
            return action

        def kind_named(fragment: str) -> str | None:
            for item in allowed:
                if isinstance(item, Mapping):
                    kind = str(item.get("kind") or "")
                    if fragment in kind:
                        return kind
            return None

        pass_kind = next(
            (str(item.get("kind")) for item in allowed if isinstance(item, Mapping)
             and ("pass" in str(item.get("kind") or "") or str(item.get("kind") or "") == "wait")),
            None,
        )
        poison_kind = kind_named("poison")
        heal_kind = next(
            (str(item.get("kind")) for item in allowed if isinstance(item, Mapping)
             and ("heal" in str(item.get("kind") or "") or "antidote" in str(item.get("kind") or ""))),
            None,
        )
        current_target = self._normalize_player_id(summary.get("current_night_attack_target"))

        def targets(kind: str | None) -> set[str]:
            if not kind:
                return set()
            item = next((item for item in allowed if isinstance(item, Mapping) and str(item.get("kind")) == kind), None)
            values = item.get("target_ids") if isinstance(item, Mapping) else None
            return {str(value) for value in values if value not in (None, "")} if isinstance(values, list) else set()

        # 狼刀与毒目标重合时，毒药没有额外击杀收益。只有引擎明确允许 pass
        # 才做替换；否则保持模型答案，交给现有 schema/重试流程处理。
        if poison_kind and action.get("kind") == poison_kind and action.get("target_id") == current_target and pass_kind:
            safe = dict(action)
            safe["kind"] = pass_kind
            safe.pop("target_id", None)
            safe.pop("text", None)
            return safe

        high_value_ids = {
            str(item.get("target_id"))
            for item in summary.get("public_high_value_targets", [])
            if isinstance(item, Mapping) and item.get("high_value_for_heal") and item.get("target_id")
        }
        if (
            pass_kind
            and action.get("kind") == pass_kind
            and heal_kind
            and current_target
            and current_target in high_value_ids
            and current_target in targets(heal_kind)
        ):
            safe = dict(action)
            safe["kind"] = heal_kind
            safe["target_id"] = current_target
            safe.pop("text", None)
            return safe
        return action

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

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
        witch_summary = self._witch_decision_summary(turn_packet)
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
            "decision_brief": witch_summary.get("decision_brief", {}),
            "speech_guidance": witch_summary.get("speech_guidance", ""),
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

        system = system + "\n\n【女巫专用决策规约】\n先复盘最近票型与死亡，再看自己药水状态。救药优先保留给能明显改变轮次的高价值夜死；毒药只在强对跳、稳定票型压缩、公开身份链冲突或终局倒计时等高置信情形下优先使用。证据不足时默认保留资源。若 decision_brief 标记 terminal_pressure、high_value_claim_conflict 或 self_at_risk，应优先解释并执行其中建议，不能机械 pass。公开发言只谈公开事实，不得泄露药水余量、夜间目标或自己未公开的用药结果。"

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
            error = decision_error(action, turn_packet["request"])
            if error is None:
                action = self._apply_witch_critical_fallback(turn_packet, action, witch_summary)
                fallback_error = decision_error(action, turn_packet["request"])
                if fallback_error is not None:
                    # 安全修正只能产生当前请求已经允许的行动；若候选摘要有误，
                    # 保留模型原答而不是让纠错逻辑破坏行动契约。
                    action = normalize_decision(raw, turn_packet["request"])
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

    def _render_compact_memory(self) -> str:
        if not self._compact_history:
            return "暂无跨回合摘要。"
        recent = self._compact_history[-4:]
        return json.dumps(recent, ensure_ascii=False, separators=(",", ":"))

    def _legacy_summarize_public_sync(self, sync_packet: Mapping[str, Any]) -> dict[str, Any]:
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

    def _legacy_witch_decision_summary(self, turn_packet: Mapping[str, Any]) -> dict[str, Any]:
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

        board_snapshot = {
            "alive_count": alive_count,
            "recent_deaths": [self._compact_value(item) for item in recent_deaths[:3]],
            "sheriff": self._compact_value(first_value([public_state, game, turn_packet], "sheriff", "police", "警长", "captain")),
            "public_counterclaims": public_counterclaims[:3],
            "vote_clusters": vote_clusters,
            "confirmed_identities": confirmed_identities[:3],
            "resource_state": {
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
            },
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
            heal_candidates.append({"kind": heal_kind, "target_id": candidate, "score": score, "reason": reason})

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
            poison_candidates.append(
                {
                    "kind": entry.get("kind"),
                    "target_id": entry.get("target_id"),
                    "score": entry.get("score"),
                    "reason": "；".join(entry.get("reasons", [])),
                }
            )

        pass_reason = "证据不足时默认保留资源"
        if heal_candidates or poison_candidates:
            pass_reason = "作为保守备选，证据不足或收益不高时再 pass"

        summary["board_snapshot"] = board_snapshot
        summary["candidate_ranking"] = {
            "heal": heal_candidates[:2],
            "poison": poison_candidates[:3],
            "pass": {
                "kind": pass_kind or "pass",
                "score": 40 if not (heal_candidates or poison_candidates) else 10,
                "reason": pass_reason,
            },
        }
        return summary

    @staticmethod
    def _public_sources(packet: Mapping[str, Any]) -> list[Mapping[str, Any]]:
        """返回公开包的几个合法载体；不递归读取私有信息。"""
        sources: list[Mapping[str, Any]] = []
        for source in (
            packet.get("public_state"),
            packet.get("payload"),
            packet.get("game"),
            packet,
        ):
            if isinstance(source, Mapping):
                sources.append(source)
                nested = source.get("public_state")
                if isinstance(nested, Mapping):
                    sources.append(nested)
                nested_payload = source.get("payload")
                if isinstance(nested_payload, Mapping):
                    sources.append(nested_payload)
        return sources

    @staticmethod
    def _public_value(sources: list[Mapping[str, Any]], *names: str) -> Any:
        for source in sources:
            for name in names:
                value = source.get(name)
                if value not in (None, ""):
                    return value
        return None

    def _summarize_public_sync(self, sync_packet: Mapping[str, Any]) -> dict[str, Any]:
        """压缩公开同步；PUBLIC_STATE_SYNC 可能把内容放在三层不同载体中。"""
        sources = self._public_sources(sync_packet)
        summary: dict[str, Any] = {}
        aliases = {
            "round": ("round", "game_round", "day_round"),
            "phase": ("phase", "public_phase", "stage"),
            "deaths": ("deaths", "dead_players", "night_deaths", "death_events"),
            "votes": ("votes", "voting", "vote_results", "vote_events"),
            "sheriff": ("sheriff", "police", "警长", "captain", "sheriff_id"),
            "claims": ("claims", "reveals", "public_claims", "role_claims", "counterclaims"),
            "players": ("players", "alive_players", "living_players", "survivors"),
        }
        for output_key, names in aliases.items():
            value = self._public_value(sources, *names)
            if value in (None, "", []):
                continue
            summary[output_key] = (
                self._compact_entities(value) if isinstance(value, list) else self._compact_value(value)
            )
        return summary

    def _alive_player_ids(self, public_state: Mapping[str, Any]) -> list[str]:
        records = self._public_value(
            self._public_sources(public_state),
            "alive_players", "living_players", "survivors", "players",
        )
        if not isinstance(records, list):
            return []
        result: list[str] = []
        for record in records:
            if not isinstance(record, Mapping):
                continue
            player_id = record.get("player_id", record.get("id"))
            if player_id in (None, ""):
                continue
            alive = record.get("alive")
            if alive is None:
                status = str(record.get("status") or "").lower()
                alive = status not in {"dead", "dead_player", "eliminated", "killed", "deceased"}
            if self._coerce_bool(alive):
                result.append(str(player_id))
        return list(dict.fromkeys(result))

    def _known_public_dead_roles(self, public_state: Mapping[str, Any]) -> list[str]:
        sources = self._public_sources(public_state)
        records = self._public_value(sources, "deaths", "dead_players", "death_events", "players")
        if not isinstance(records, list):
            return []
        roles: list[str] = []
        for record in records:
            if not isinstance(record, Mapping):
                continue
            alive = record.get("alive")
            status = str(record.get("status") or "").lower()
            if alive is True or status in {"alive", "living"}:
                continue
            role = record.get("revealed_role", record.get("public_role", record.get("role")))
            if role not in (None, ""):
                roles.append(str(role))
        return list(dict.fromkeys(roles))

    def _allowed_action_info(self, request: Mapping[str, Any]) -> dict[str, Any]:
        info: dict[str, Any] = {"kinds": [], "targets": {}}
        actions = request.get("allowed_actions")
        if not isinstance(actions, list):
            return info
        for item in actions:
            if not isinstance(item, Mapping) or not item.get("kind"):
                continue
            kind = str(item["kind"])
            info["kinds"].append(kind)
            targets = item.get("target_ids")
            info["targets"][kind] = (
                {str(target) for target in targets if target not in (None, "")}
                if isinstance(targets, list) else set()
            )
        return info

    def _dialogue_signals(self, dialogue: Any) -> dict[str, Any]:
        """只抽取当前轮的短公开信号，原文仍由受限工具提供给模型核对。"""
        records = dialogue if isinstance(dialogue, list) else []
        role_words = {
            "seer": ("预言家", "跳预", "真预", "假预", "seer"),
            "hunter": ("猎人", "hunter"),
            "guard": ("守卫", "guard"),
            "witch": ("女巫", "witch"),
            "villager": ("平民", "民牌", "villager"),
        }
        speakers: list[dict[str, Any]] = []
        conflicts: list[dict[str, Any]] = []
        attacks: dict[str, int] = {}
        for item in records[:8]:
            if isinstance(item, Mapping):
                speaker = item.get("speaker_id", item.get("player_id", item.get("id", "")))
                text = str(item.get("text", item.get("content", item.get("message", ""))))
            else:
                speaker, text = "", str(item)
            text = text[:240]
            lower = text.lower()
            claims = [role for role, words in role_words.items() if any(word in lower for word in words)]
            evidence = [term for term in ("查验", "金水", "查杀", "验人", "查到") if term in text]
            denial = any(term in text for term in ("没跳过", "没有跳", "不是", "否认", "对不上", "不成立"))
            conflict = denial or any(term in text for term in ("悍跳", "对跳", "冲突", "假预", "伪造"))
            targets = re.findall(r"(?:攻击|怀疑|投|归票|查杀|金水)[：: ]*([A-Za-z]?\d+)", text)
            for target in targets:
                attacks[target] = attacks.get(target, 0) + 1
            signal = {
                "speaker_id": str(speaker) if speaker not in (None, "") else "",
                "claims": claims[:3],
                "evidence": evidence[:3],
                "denial_or_conflict": conflict,
                "attack_targets": targets[:3],
            }
            if signal["speaker_id"] or claims or evidence or conflict or targets:
                speakers.append(signal)
            if conflict:
                conflicts.append(signal)
        top_suspects = [
            {"target_id": target, "mentions": count}
            for target, count in sorted(attacks.items(), key=lambda pair: (-pair[1], pair[0]))[:3]
        ]
        return {"speakers": speakers[:8], "conflicts": conflicts[:5], "top_suspects": top_suspects}

    def _witch_decision_summary(self, turn_packet: Mapping[str, Any]) -> dict[str, Any]:
        request = turn_packet.get("request") or {}
        game = turn_packet.get("game") or {}
        public_state = turn_packet.get("public_state") or {}
        sources = self._public_sources(turn_packet)
        phase = str(game.get("public_phase", game.get("phase", "")))
        action_info = self._allowed_action_info(request)
        kinds = action_info["kinds"]
        heal_kind = next((kind for kind in kinds if "heal" in kind or "antidote" in kind), None)
        poison_kind = next((kind for kind in kinds if "poison" in kind), None)
        pass_kind = next((kind for kind in kinds if "pass" in kind or kind == "wait"), None)
        deaths = self._public_value(sources, "deaths", "dead_players", "night_deaths", "death_events")
        votes = self._public_value(sources, "votes", "voting", "vote_results", "vote_events")
        claims = self._public_value(sources, "claims", "reveals", "public_claims", "role_claims", "counterclaims")
        deaths = deaths if isinstance(deaths, list) else []
        votes = votes if isinstance(votes, list) else []
        claims = claims if isinstance(claims, list) else []
        dialogue = ((turn_packet.get("tool_context") or {}).get("current_round_dialogue") or [])
        dialogue_signals = self._dialogue_signals(dialogue)
        alive_ids = self._alive_player_ids(public_state)
        alive_count = self._public_value(sources, "alive_count", "living_count", "survivor_count")
        if alive_count is None and alive_ids:
            alive_count = len(alive_ids)
        alive_count = self._nonnegative_int(alive_count, fallback=None)
        dead_roles = self._known_public_dead_roles(public_state)
        dead_power_roles = [role for role in dead_roles if any(word in role.lower() for word in ("seer", "预言", "hunter", "猎人", "guard", "守卫"))]

        def record_id(record: Any, *keys: str) -> str | None:
            if not isinstance(record, Mapping):
                return None
            for key in keys + ("target_id", "player_id", "id", "speaker_id", "voter_id", "vote_target_id", "target"):
                value = record.get(key)
                if value not in (None, ""):
                    return str(value)
            return None

        vote_totals: dict[str, int] = {}
        for record in votes:
            target = record_id(record, "vote_target_id", "target_id", "target")
            if target:
                count = self._nonnegative_int(record.get("count") if isinstance(record, Mapping) else 1, fallback=1) or 1
                vote_totals[target] = vote_totals.get(target, 0) + count
        self_votes = vote_totals.get(self.player_id, 0)
        max_votes = max(vote_totals.values(), default=0)
        self_at_risk = self_votes >= 2 or (self_votes > 0 and self_votes == max_votes)

        claimed_roles: dict[str, set[str]] = {}
        claim_conflicts: list[dict[str, Any]] = []
        for record in claims:
            if not isinstance(record, Mapping):
                continue
            speaker = record_id(record, "claimant_id", "owner_id")
            blob = json.dumps(record, ensure_ascii=False).lower()
            role = str(record.get("role", record.get("claimed_role", ""))).lower()
            for role_name, words in {
                "seer": ("seer", "预言家", "预言"), "hunter": ("hunter", "猎人"),
                "guard": ("guard", "守卫"), "witch": ("witch", "女巫"),
            }.items():
                if any(word in role or word in blob for word in words) and speaker:
                    claimed_roles.setdefault(role_name, set()).add(speaker)
            if any(term in blob for term in ("counterclaim", "challenge", "conflict", "悍跳", "对跳", "冲突")):
                claim_conflicts.append({"speaker_id": speaker, "evidence": self._compact_value(record)})
        for signal in dialogue_signals["conflicts"]:
            claim_conflicts.append(signal)
            speaker = signal.get("speaker_id")
            if speaker:
                claimed_roles.setdefault("conflict", set()).add(speaker)

        high_value_roles = {player for role in ("seer", "hunter", "guard") for player in claimed_roles.get(role, set())}
        conflict_speakers = {item.get("speaker_id") for item in claim_conflicts if item.get("speaker_id")}
        attack_counts = {item["target_id"]: item["mentions"] for item in dialogue_signals["top_suspects"]}
        poison_store: dict[str, dict[str, Any]] = {}

        def add_candidate(target: str | None, points: int, reason: str) -> None:
            if not target or target == self.player_id:
                return
            allowed = action_info["targets"].get(poison_kind or "", set())
            if allowed and target not in allowed:
                return
            item = poison_store.setdefault(target, {"kind": poison_kind, "target_id": target, "score": 0, "reasons": []})
            item["score"] += points
            if reason not in item["reasons"]:
                item["reasons"].append(reason)

        for target, count in vote_totals.items():
            # 票多只是弱信号，不能单独把好人送进毒药候选。
            add_candidate(target, min(15, count * 5), f"票型聚集 {count} 票")
        for target in conflict_speakers:
            add_candidate(str(target), 45, "公开身份链冲突/对跳")
            if target in high_value_roles:
                add_candidate(str(target), 30, "公开神职声明牵涉冲突")
        for signal in dialogue_signals["speakers"]:
            speaker = signal.get("speaker_id")
            if not speaker:
                continue
            if signal.get("evidence") and signal.get("denial_or_conflict"):
                add_candidate(speaker, 35, "验人链与否认互相矛盾")
            if "seer" in signal.get("claims", []) and signal.get("evidence"):
                add_candidate(speaker, 15, "公开跳预并给出验人链")
        for target, mentions in attack_counts.items():
            if mentions >= 2:
                add_candidate(target, 15, "当前轮被集中点名")
        for target, roles in claimed_roles.items():
            if target == "conflict":
                continue
            for speaker in roles:
                if target == "seer" or target == "hunter":
                    continue
                add_candidate(speaker, 30, "公开身份声明值得核对")
        poison_candidates = [
            {"kind": item["kind"], "target_id": item["target_id"], "score": min(150, item["score"]), "reason": "；".join(item["reasons"])}
            for item in sorted(poison_store.values(), key=lambda value: (-value["score"], value["target_id"]))[:3]
        ]

        heal_candidates: list[dict[str, Any]] = []
        heal_allowed = action_info["targets"].get(heal_kind or "", set())
        for index, record in enumerate(deaths[:3]):
            target = record_id(record)
            if not target or (heal_allowed and target not in heal_allowed):
                continue
            public_role = str(record.get("revealed_role", record.get("public_role", record.get("role", "")))) if isinstance(record, Mapping) else ""
            high_value = any(word in public_role.lower() for word in ("seer", "预言", "hunter", "猎人", "guard", "守卫"))
            heal_candidates.append({"kind": heal_kind, "target_id": target, "score": 110 if high_value else 80 - index * 10, "high_value_public_role": high_value, "reason": "公开高价值神职" if high_value else "最近夜死"})

        poison_available = self._coerce_bool(self._known_private_state.get("poison_available", self._known_private_state.get("can_poison"))) is True
        terminal_pressure = bool(
            poison_available and ((alive_count is not None and alive_count <= 9) or len(dead_power_roles) >= 2 or self_at_risk)
        )
        speech_guidance = "仅基于公开事实发言。"
        if self_at_risk or terminal_pressure or claim_conflicts:
            speech_guidance = "以自然神职/强好人视角明确复盘公开票型，给出归票目标并反对无依据跟风；不要透露夜刀、药水余量或未公开用药。"
        if str(request.get("kind") or "") in {"speak", "last_words"}:
            speech_guidance += " 遗言也要交代公开逻辑链和怀疑列表，不要说‘没什么可留’。"
        board_snapshot = {
            "alive_count": alive_count,
            "estimated_max_wolves": 4,
            "wolf_pressure": "high" if (alive_count is not None and alive_count <= 9) else "normal",
            "public_dead_power_roles": dead_power_roles[:5],
            "sheriff_id": self._public_value(sources, "sheriff_id", "sheriff", "police", "captain"),
            "self_vote_pressure": self_votes,
            "self_at_risk": self_at_risk,
            "seer_claims": sorted(claimed_roles.get("seer", set()))[:4],
            "hunter_claims": sorted(claimed_roles.get("hunter", set()))[:4],
            "guard_claims": sorted(claimed_roles.get("guard", set()))[:4],
            "witch_claims": sorted(claimed_roles.get("witch", set()))[:4],
            "claim_conflicts": claim_conflicts[:5],
            "dialogue_signals": dialogue_signals,
            "recent_deaths": [self._compact_value(item) for item in deaths[:3]],
            "vote_clusters": [{"target_id": target, "count": count} for target, count in sorted(vote_totals.items(), key=lambda pair: (-pair[1], pair[0]))[:3]],
            "resource_state": dict(self._known_private_state),
        }
        decision_brief = {
            "terminal_pressure": terminal_pressure,
            "high_value_claim_conflict": bool(claim_conflicts),
            "self_at_risk": self_at_risk,
            "recommended_priority": "检查高分毒候选，残局不要机械 pass" if terminal_pressure else "核对公开链后再决定，低信息可保留资源",
        }
        return {
            "phase": phase,
            "request_kind": request.get("kind"),
            "board_snapshot": board_snapshot,
            "decision_brief": decision_brief,
            "speech_guidance": speech_guidance,
            "candidate_ranking": {
                "heal": heal_candidates[:2],
                "poison": poison_candidates,
                "pass": {"kind": pass_kind or "pass", "score": 10 if (terminal_pressure or poison_candidates) else 40, "reason": "残局或高冲突时不要机械 pass" if terminal_pressure else "证据不足时保留资源"},
            },
        }

    def _apply_witch_critical_fallback(
        self, turn_packet: Mapping[str, Any], action: dict[str, Any], witch_summary: Mapping[str, Any]
    ) -> dict[str, Any]:
        """只修正最危险的残局留毒；不接管模型已经作出的合法非 pass 决策。"""
        if action.get("kind") not in {"pass", "wait"} and "pass" not in str(action.get("kind") or "").lower():
            return action
        request = turn_packet.get("request") or {}
        phase = str((turn_packet.get("game") or {}).get("public_phase", (turn_packet.get("game") or {}).get("phase", ""))).lower()
        if "witch" not in phase and "witch" not in str(request.get("kind") or "").lower():
            return action
        brief = witch_summary.get("decision_brief") or {}
        if not brief.get("terminal_pressure"):
            return action
        ranking = witch_summary.get("candidate_ranking") or {}
        for candidate in ranking.get("poison", []) if isinstance(ranking.get("poison"), list) else []:
            if int(candidate.get("score", 0)) < 90:
                continue
            target = candidate.get("target_id")
            if not target:
                continue
            for allowed in request.get("allowed_actions", []):
                if isinstance(allowed, Mapping) and "poison" in str(allowed.get("kind", "")).lower():
                    targets = allowed.get("target_ids")
                    if isinstance(targets, list) and target in targets:
                        return {"request_id": request.get("request_id", action.get("request_id")), "player_id": request.get("player_id", action.get("player_id", self.player_id)), "kind": str(allowed["kind"]), "target_id": str(target)}
        heal_available = self._coerce_bool(
            self._known_private_state.get("antidote_available", self._known_private_state.get("can_heal"))
        ) is True
        for candidate in ranking.get("heal", []) if isinstance(ranking.get("heal"), list) else []:
            if not heal_available or int(candidate.get("score", 0)) < 100 or not candidate.get("high_value_public_role"): 
                continue
            target = candidate.get("target_id")
            for allowed in request.get("allowed_actions", []):
                if isinstance(allowed, Mapping) and ("heal" in str(allowed.get("kind", "")).lower() or "antidote" in str(allowed.get("kind", "")).lower()):
                    targets = allowed.get("target_ids")
                    if isinstance(targets, list) and target in targets:
                        return {"request_id": request.get("request_id", action.get("request_id")), "player_id": request.get("player_id", action.get("player_id", self.player_id)), "kind": str(allowed["kind"]), "target_id": str(target)}
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

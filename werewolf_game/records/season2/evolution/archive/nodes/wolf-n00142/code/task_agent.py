"""最小可运行的角色 Task-Agent。

每次行动从一个新的模型会话开始；这里额外维护一小段跨回合公开记忆，用于压缩上下文、
稳定狼人夜刀收敛与白天叙事，但不访问任何外部状态。
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
_MAX_HISTORY_ITEMS = 6
_MAX_SNIPPET_LENGTH = 80
_MAX_COMPACT_DEPTH = 2
_TEMPLATE_PHRASES = (
    "先听一圈",
    "看票型",
    "中间位",
    "别划水",
    "稳一点",
    "我先过",
    "先稳",
    "刀中间位",
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
        self._game_memory: dict[str, Any] = {
            "round": None,
            "phase": None,
            "alive_players": [],
            "dead_players": [],
            "sheriff_id": None,
            "death_order": [],
            "recent_votes": [],
        }
        self._public_history: list[dict[str, Any]] = []
        self._self_history: list[dict[str, Any]] = []
        self._recent_public_texts: list[str] = []
        self._recent_self_texts: list[str] = []
        # 只由私有身份信息、自身行动和公开同步结果派生；不读取引擎之外的状态。
        self._wolf_memory: dict[str, Any] = {
            "recent_kill_votes": [],
            "last_wolf_target": None,
            "last_kill_round": None,
            "last_night_deaths": [],
            "failed_kill_targets": [],
            "known_wolf_ids_cache": [],
            "alive_wolf_ids_cache": [],
            "no_death_rounds": [],
            # 每条记录只描述一个已经形成的狼刀计划；结算必须发生在明确的夜->昼边界。
            "night_resolution_ledger": [],
        }
        self._public_ledger: dict[str, list[dict[str, Any]]] = {
            "claims": [],
            "votes": [],
            "deaths": [],
            "sheriff_changes": [],
        }

    async def observe(self, sync_packet: dict[str, Any]) -> None:
        """接收公开同步并压缩成轻量记忆。"""

        summary = self._summarize_sync_packet(sync_packet)
        if not summary:
            return

        old_dead = {str(item) for item in (self._game_memory.get("dead_players") or [])}
        new_dead = {str(item) for item in (summary.get("dead_players") or [])}
        old_phase = str(self._game_memory.get("phase") or "").lower()
        new_phase = str(summary.get("phase") or "").lower()
        old_round = self._game_memory.get("round")
        new_round = summary.get("round")
        # 先结算上一观察周期形成的计划，再写入本次同步；没有明确夜->昼边界时宁可不猜。
        self._resolve_night_plans(
            summary,
            old_phase=old_phase,
            new_phase=new_phase,
            old_round=old_round,
            new_round=new_round,
            old_dead=old_dead,
            new_dead=new_dead,
        )
        if summary.get("no_death") and new_phase:
            round_value = new_round
            if round_value is not None and round_value not in self._wolf_memory["no_death_rounds"]:
                self._wolf_memory["no_death_rounds"].append(round_value)
                self._wolf_memory["no_death_rounds"] = self._wolf_memory["no_death_rounds"][-4:]
        if new_dead and new_dead != old_dead:
            self._wolf_memory["last_night_deaths"] = sorted(new_dead)[-4:]

        self._update_public_ledger(summary)
        for key in ("round", "phase", "sheriff_id"):
            value = summary.get(key)
            if value is not None:
                self._game_memory[key] = value
        for key in ("alive_players", "dead_players", "death_order"):
            value = summary.get(key)
            if value:
                self._game_memory[key] = value
        votes = summary.get("recent_votes")
        if votes:
            self._game_memory["recent_votes"] = votes[-_MAX_HISTORY_ITEMS:]

        public_texts = summary.get("public_texts") or []
        self._recent_public_texts.extend(public_texts)
        self._recent_public_texts = self._recent_public_texts[-_MAX_HISTORY_ITEMS:]

        self._public_history.append(summary)
        self._public_history = self._public_history[-_MAX_HISTORY_ITEMS:]

    async def decide(self, turn_packet: dict[str, Any]) -> dict[str, Any]:
        if turn_packet["self"]["player_id"] != self.player_id:
            raise ValueError("行动包被发送给了错误玩家")
        private = turn_packet["private_information"]
        if str(private.get("role")) != self.profile.role:
            raise ValueError("Task-Agent 的角色档案与行动包身份不一致")
        wolf_ids = self._extract_wolf_ids(private)
        if wolf_ids:
            self._wolf_memory["known_wolf_ids_cache"] = wolf_ids
            alive = self._alive_ids_from_state(turn_packet.get("public_state"), turn_packet.get("game"))
            self._wolf_memory["alive_wolf_ids_cache"] = [pid for pid in wolf_ids if not alive or pid in alive]

        memory_snapshot = self._memory_snapshot()
        current_dialogue = self._extract_current_round_dialogue(turn_packet)
        system = self._system_prompt(private)
        prompt = self._build_compact_turn_prompt(turn_packet, memory_snapshot, current_dialogue)

        tool_context = turn_packet.get("tool_context") or {}
        current_dialogue_tool = tool_context.get("current_round_dialogue") or []

        def execute_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
            del arguments
            if name != CURRENT_ROUND_DIALOGUE_TOOL_NAME:
                return {"error": f"不支持的工具：{name}"}
            return {
                "round": turn_packet["game"].get("round"),
                "phase": turn_packet["game"].get("public_phase", turn_packet["game"].get("phase")),
                "dialogue": current_dialogue_tool,
            }

        feedback = ""
        for _attempt in range(self.max_decision_retries + 1):
            # 一次玩家决策的工具预算不能因纠错重试而被放大。首个模型尝试可读取
            # 当前轮对话；所有后续纠错尝试都是没有工具的新会话。
            allow_tool = _attempt == 0 and self.max_tool_calls_per_decision > 0
            instruction = self._turn_instruction(turn_packet["request"], feedback)
            user_content: dict[str, Any] = {
                "instruction": instruction,
                "packet": self._fit_prompt_payload(prompt, system=system, instruction=instruction),
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
                self._record_self_action(turn_packet, action)
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

    @staticmethod
    def _nonnegative_int(value: object, *, fallback: int | None) -> int | None:
        if isinstance(value, bool):
            return fallback
        try:
            normalized = int(value)
        except (TypeError, ValueError):
            return fallback
        return normalized if normalized >= 0 else fallback

    def _record_self_action(self, turn_packet: Mapping[str, Any], action: Mapping[str, Any]) -> None:
        summary = {
            "round": self._safe_get(turn_packet.get("game"), "round"),
            "phase": self._safe_get(turn_packet.get("request"), "phase"),
            "kind": action.get("kind"),
        }
        target_id = action.get("target_id")
        if target_id:
            summary["target_id"] = str(target_id)
            if str(action.get("kind")) == "wolf_kill_vote":
                kill = {
                    "round": summary.get("round"),
                    "target_id": str(target_id),
                }
                self._wolf_memory["recent_kill_votes"].append(kill)
                self._wolf_memory["recent_kill_votes"] = self._wolf_memory["recent_kill_votes"][-3:]
                self._wolf_memory["last_wolf_target"] = str(target_id)
                self._wolf_memory["last_kill_round"] = summary.get("round")
                baseline = sorted({str(item) for item in (self._game_memory.get("dead_players") or [])})
                plan_round = summary.get("round")
                key = f"{plan_round}|{target_id}|{','.join(baseline)}"
                plans = self._wolf_memory["night_resolution_ledger"]
                if not any(str(item.get("key")) == key for item in plans if isinstance(item, Mapping)):
                    plans.append({
                        "key": key,
                        "plan_round": plan_round,
                        "target_id": str(target_id),
                        "baseline_dead_players": baseline,
                        "resolved_round": None,
                        "new_deaths": [],
                        "result": "pending",
                    })
                    self._wolf_memory["night_resolution_ledger"] = plans[-6:]
        text = action.get("text")
        if isinstance(text, str) and text.strip():
            clipped = self._clip_text(text, _MAX_SNIPPET_LENGTH)
            summary["text"] = clipped
            self._recent_self_texts.append(clipped)
            self._recent_self_texts = self._recent_self_texts[-_MAX_HISTORY_ITEMS:]
        self._self_history.append(summary)
        self._self_history = self._self_history[-_MAX_HISTORY_ITEMS:]

    def _memory_snapshot(self) -> dict[str, Any]:
        return {
            "game": {
                key: self._game_memory.get(key)
                for key in ("round", "phase", "sheriff_id", "alive_players", "dead_players", "death_order")
            },
            "recent_public_history": self._public_history[-3:],
            "recent_self_history": self._self_history[-3:],
            "recent_vote_summary": self._game_memory.get("recent_votes", [])[-2:],
            "public_ledger": {
                key: list(values[-6:]) for key, values in self._public_ledger.items()
            },
            "wolf_memory": {
                "recent_kill_votes": self._wolf_memory.get("recent_kill_votes", [])[-3:],
                "last_wolf_target": self._wolf_memory.get("last_wolf_target"),
                "last_kill_round": self._wolf_memory.get("last_kill_round"),
                "last_night_deaths": self._wolf_memory.get("last_night_deaths", [])[-3:],
                "failed_kill_targets": self._wolf_memory.get("failed_kill_targets", [])[-3:],
                "known_wolf_ids": self._wolf_memory.get("known_wolf_ids_cache", []),
                "alive_wolf_ids": self._wolf_memory.get("alive_wolf_ids_cache", []),
                "no_death_rounds": self._wolf_memory.get("no_death_rounds", [])[-3:],
                "night_resolution_ledger": self._wolf_memory.get("night_resolution_ledger", [])[-4:],
            },
        }

    def _build_compact_turn_prompt(
        self,
        turn_packet: Mapping[str, Any],
        memory_snapshot: Mapping[str, Any],
        current_dialogue: list[dict[str, Any]],
    ) -> dict[str, Any]:
        public_rules = self._compact_snapshot(
            turn_packet.get("public_rules"),
            preferred_keys=(
                "phase_rules",
                "vote_rules",
                "speech_limits",
                "action_limits",
                "public_information",
                "night_rules",
                "day_rules",
            ),
        )
        game = self._compact_snapshot(
            turn_packet.get("game"),
            preferred_keys=(
                "round",
                "phase",
                "public_phase",
                "day",
                "night",
                "alive_count",
                "dead_count",
                "players_alive",
                "players_dead",
            ),
        )
        public_state = self._compact_public_state(turn_packet.get("public_state"))
        if isinstance(public_state, dict):
            public_state["player_index"] = self._build_player_index(
                turn_packet.get("public_state"),
                turn_packet.get("game"),
                memory_snapshot,
                turn_packet.get("request"),
            )
        request = self._compact_request(turn_packet.get("request"))
        self_state = self._compact_snapshot(
            turn_packet.get("self"),
            preferred_keys=(
                "player_id",
                "seat",
                "seat_id",
                "alive",
                "status",
                "vote_target",
                "speak_order",
            ),
        )
        private_information = self._compact_snapshot(
            turn_packet.get("private_information"),
            preferred_keys=("role", "team", "identity", "faction", "ability"),
        )
        wolf_strategy = self._build_wolf_strategy(turn_packet, memory_snapshot, current_dialogue)
        return {
            "game": game,
            "public_rules": public_rules,
            "self": self_state,
            "private_information": private_information,
            "public_state": public_state,
            "request": request,
            "memory": memory_snapshot,
            "wolf_strategy": wolf_strategy,
            "current_round_dialogue": current_dialogue,
        }

    def _build_wolf_strategy(
        self,
        turn_packet: Mapping[str, Any],
        memory_snapshot: Mapping[str, Any],
        current_dialogue: list[dict[str, Any]],
    ) -> dict[str, Any]:
        request = turn_packet.get("request") or {}
        phase = str(request.get("phase") or turn_packet.get("game", {}).get("phase") or "unknown")
        channel = str(request.get("channel") or "unknown")
        allowed_actions = request.get("allowed_actions") if isinstance(request.get("allowed_actions"), list) else []
        target_actions = [item for item in allowed_actions if isinstance(item, Mapping) and item.get("target_ids")]
        speech_actions = [item for item in allowed_actions if isinstance(item, Mapping) and item.get("kind") in _SPEECH_ACTION_KINDS]
        phase_lower = phase.lower()
        channel_lower = channel.lower()
        is_night = any(token in phase_lower for token in ("night", "夜")) or any(
            token in channel_lower for token in ("night", "wolf")
        )
        recent_vote_summary = memory_snapshot.get("recent_vote_summary") or []
        dialogue_slice = current_dialogue[-3:]
        if is_night and target_actions:
            ranked_targets = self._rank_targets(turn_packet, memory_snapshot, target_actions)
            primary = ranked_targets[0] if ranked_targets else None
            backup = self._pick_backup_target(ranked_targets)
            return {
                "stage": "night",
                "phase_bucket": self._night_stage(memory_snapshot),
                "objective": "收敛到一个主刀和一个来自不同威胁层级的备选刀",
                "primary_target": primary,
                "backup_target": backup,
                "target_diversity_required": True,
                "tie_break": "先看真神职暴露/对跳/验人链，其次警徽与票型核心，再看公共威胁；若同分，保留更接近既有一致目标者",
                "recent_vote_summary": recent_vote_summary,
                "current_round_dialogue": dialogue_slice,
                "notes": self._night_notes(memory_snapshot, current_dialogue),
            }
        day_target_actions = [
            item for item in target_actions
            if str(item.get("kind") or "").lower() in {"day_vote", "sheriff_vote", "vote", "vote_cast"}
        ]
        if day_target_actions:
            return self._build_day_vote_strategy(turn_packet, memory_snapshot, current_dialogue, day_target_actions)
        if speech_actions:
            focus_target = self._pick_day_focus_target(memory_snapshot, current_dialogue)
            claims = self._detect_claims_from_dialogue(current_dialogue, memory_snapshot)
            focus_reason = self._day_focus_reason(focus_target, recent_vote_summary, dialogue_slice)
            return {
                "stage": "day",
                "phase_bucket": self._night_stage(memory_snapshot),
                "objective": "围绕当前争议对象持续修正叙事",
                "focus_target": focus_target,
                "focus_reason": focus_reason,
                "narrative_contract": {
                    "object": "必须点名一个具体对象",
                    "reason": "必须引用 recent_vote_summary 或 current_round_dialogue 中的具体票型/发言",
                    "action": "必须给出下一步动作（继续压、转向、暂缓）",
                },
                "recent_vote_summary": recent_vote_summary,
                "current_round_dialogue": dialogue_slice,
                "pressure_points": self._day_pressure_points(memory_snapshot),
                "claims_internal": claims,
                "wolf_narrative": self._wolf_narrative(memory_snapshot, claims, current_dialogue),
                "avoid_templates": self._anti_template_notes(),
            }
        return {
            "stage": "other",
            "objective": "只做当前合法行动，不额外发散",
            "avoid_templates": self._anti_template_notes(),
        }

    def _build_day_vote_strategy(
        self,
        turn_packet: Mapping[str, Any],
        memory_snapshot: Mapping[str, Any],
        current_dialogue: list[dict[str, Any]],
        target_actions: list[Mapping[str, Any]],
    ) -> dict[str, Any]:
        claims = self._detect_claims_from_dialogue(current_dialogue, memory_snapshot)
        public_ledger = memory_snapshot.get("public_ledger") or {}
        wolf_memory = memory_snapshot.get("wolf_memory") or {}
        alive_wolves = [str(item) for item in (wolf_memory.get("alive_wolf_ids") or [])]
        known_wolves = {str(item) for item in (wolf_memory.get("known_wolf_ids") or [])}
        alive = self._alive_ids_from_state(turn_packet.get("public_state"), turn_packet.get("game"))
        if alive:
            alive_wolves = [pid for pid in alive_wolves if pid in alive]
        dead_wolves = sorted(known_wolves - set(alive_wolves))
        current_round = self._safe_get(turn_packet.get("game"), "round")
        recent_claims = [item for item in (public_ledger.get("claims") or [])
                         if item.get("round") in {current_round, (current_round - 1 if isinstance(current_round, int) else None)}]
        pressured = []
        for pid in alive_wolves:
            if any(str(item.get("speaker")) != pid and pid in str(item.get("text") or "")
                   and any(word in str(item.get("text") or "") for word in ("查杀", "出局"))
                   for item in recent_claims):
                pressured.append(pid)
        hunter_claimants = [pid for pid, roles in claims.items() if "猎人" in roles]
        dangerous_claims = {
            pid: roles for pid, roles in claims.items()
            if any(role in roles for role in ("预言家", "女巫", "守卫", "猎人"))
        }
        for item in recent_claims:
            speaker = str(item.get("speaker"))
            role = item.get("claimed_role")
            if role in ("预言家", "女巫", "守卫", "猎人"):
                dangerous_claims.setdefault(speaker, []).append(role)
        candidate_ids: list[str] = []
        for action in target_actions:
            for target in action.get("target_ids") or []:
                pid = str(target)
                if pid not in candidate_ids and pid != self.player_id:
                    candidate_ids.append(pid)
        # 队友被查杀时，默认把票分给查杀者/对跳者，而不是主动补刀队友。
        recommended: list[str] = []
        for pid in candidate_ids:
            if pid in pressured and len(alive_wolves) <= 2:
                continue
            if hunter_claimants and pid in hunter_claimants and len(alive_wolves) <= 2:
                continue
            if pid in dangerous_claims and pid not in pressured:
                recommended.append(pid)
        recommended.extend(pid for pid in candidate_ids if pid not in recommended and pid not in pressured)
        if not recommended:
            recommended = candidate_ids[:3]
        return {
            "stage": "day_vote",
            "objective": "从狼人视角保票权、造分票并避免把真猎人送上焦点",
            "alive_wolf_ids": alive_wolves,
            "dead_wolf_ids_count": len(dead_wolves),
            "endgame_pressure": len(alive_wolves) <= 2 or len(alive) <= 5,
            "dangerous_claims": dangerous_claims,
            "teammate_under_pressure": pressured[:3],
            "hunter_claimants": hunter_claimants[:3],
            "recommended_vote_targets": recommended[:4],
            "rules": [
                "活狼<=2时除非队友必死，不主动投队友；队友被查杀且不可救时优先制造分票",
                "猎人对跳未确认真伪时，避免把确定真猎人送出，评估其出局带枪风险",
                "狼多时可少量倒钩，但不要全队同票；优先投警长/对跳者或低信任好人",
            ],
            "claims_internal": claims,
            "current_round_dialogue": current_dialogue[-3:],
        }

    def _wolf_narrative(
        self,
        memory_snapshot: Mapping[str, Any],
        claims: Mapping[str, list[str]],
        current_dialogue: list[dict[str, Any]],
    ) -> list[str]:
        alive_wolves = (memory_snapshot.get("wolf_memory") or {}).get("alive_wolf_ids") or []
        if len(alive_wolves) <= 2:
            return ["活狼少，默认软切割而非硬冲队友", "猎人对跳时攻击对跳动机，避免直接站真猎人"]
        if any("预言家" in roles and "查杀" in roles for roles in claims.values()):
            return ["队友被查杀时质疑查验时机/警徽压人，制造分票，不要公开泄露队友信息"]
        return ["公开绑定具体对象、票型理由和下一步动作，避免机械模板"]

    def _extract_wolf_ids(self, private_information: Any) -> list[str]:
        if not isinstance(private_information, Mapping):
            return []
        values: list[Any] = []
        for key in ("wolf_ids", "wolves", "teammates", "wolf_team", "werewolves"):
            raw = private_information.get(key)
            if isinstance(raw, Mapping):
                values.extend(raw.keys())
            elif isinstance(raw, (list, tuple, set)):
                values.extend(raw)
            elif raw is not None:
                values.append(raw)
        result: list[str] = []
        for value in values:
            if isinstance(value, Mapping):
                value = self._first_value(value, ("player_id", "id", "seat", "seat_id"))
            if value is not None and str(value) != self.player_id and str(value) not in result:
                result.append(str(value))
        return result[:8]

    def _alive_ids_from_state(self, public_state: Any, game: Any = None) -> set[str]:
        for source in (public_state, game):
            if isinstance(source, Mapping):
                ids = self._extract_id_list(
                    source, preferred_keys=("alive_players", "alive", "living_players", "survivors")
                )
                if ids:
                    return set(ids)
        return set()

    def _dead_ids_from_state(self, public_state: Any, game: Any = None) -> set[str]:
        result: set[str] = set()
        for source in (public_state, game):
            if isinstance(source, Mapping):
                result.update(self._extract_id_list(
                    source, preferred_keys=("dead_players", "dead", "eliminated_players", "casualties")
                ))
        return result

    def _detect_claims_from_dialogue(
        self,
        dialogue: Any,
        memory_snapshot: Mapping[str, Any],
    ) -> dict[str, list[str]]:
        claims: dict[str, list[str]] = {}
        ledger = memory_snapshot.get("public_ledger") or {}
        for record in ledger.get("claims") or []:
            if not isinstance(record, Mapping) or record.get("speaker") is None:
                continue
            speaker = str(record["speaker"])
            role = record.get("claimed_role")
            text = str(record.get("text") or "")
            values = [role] if role else []
            for marker, label in (("查杀", "查杀"), ("金水", "金水"), ("对跳", "对跳")):
                if marker in text:
                    values.append(label)
            if values:
                claims.setdefault(speaker, [])
                for value in values:
                    if value and value not in claims[speaker]:
                        claims[speaker].append(value)
        items: list[Any] = []
        if isinstance(dialogue, list):
            items.extend(dialogue)
        # 当前窗口是补充，不覆盖账本；历史原文只保留极短窗口。
        items.extend(self._public_history[-1:])
        for item in items:
            if isinstance(item, Mapping):
                speaker = item.get("speaker") or item.get("speaker_id") or item.get("player_id")
                text = item.get("text") or item.get("content") or item.get("message") or ""
            else:
                speaker, text = None, str(item)
            if speaker is None:
                match = re.search(r"(p\d+)", str(text))
                speaker = match.group(1) if match else None
            if speaker is None:
                continue
            text = str(text)
            roles = [role for role in ("预言家", "女巫", "守卫", "猎人", "平民") if role in text]
            if "查杀" in text:
                roles.append("查杀")
            if "金水" in text:
                roles.append("金水")
            if "对跳" in text:
                roles.append("对跳")
            if roles:
                claims.setdefault(str(speaker), [])
                for role in roles:
                    if role not in claims[str(speaker)]:
                        claims[str(speaker)].append(role)
        return {pid: roles[:5] for pid, roles in list(claims.items())[:8]}

    def _night_stage(self, memory_snapshot: Mapping[str, Any]) -> str:
        game = memory_snapshot.get("game", {})
        round_value = self._nonnegative_int(game.get("round"), fallback=0) or 0
        alive_players = game.get("alive_players") or []
        alive_count = len(alive_players) if isinstance(alive_players, list) else 0
        if round_value <= 1 and alive_count >= 9:
            return "early"
        if alive_count <= 6 or round_value >= 4:
            return "end"
        return "mid"

    def _target_threat_profile(
        self,
        target_id: str,
        record: Mapping[str, Any],
        memory_snapshot: Mapping[str, Any],
    ) -> dict[str, Any]:
        stage = self._night_stage(memory_snapshot)
        stage_weights = {
            "early": {"exposed": 10, "counterclaim": 8, "sheriff": 12, "vote_lock": 4, "public": 3},
            "mid": {"exposed": 9, "counterclaim": 10, "sheriff": 9, "vote_lock": 8, "public": 4},
            "end": {"exposed": 8, "counterclaim": 10, "sheriff": 7, "vote_lock": 12, "public": 5},
        }[stage]
        score = 0
        reasons: list[str] = []
        layer = 1
        if target_id == self.player_id:
            return {"score": -100, "layer": 0, "reason": "自己不能作为刀口"}
        if record:
            alive_state = self._is_alive_record(record)
            if alive_state is False:
                score -= 80
                reasons.append("已死亡")
                layer = max(layer, 0)
            claim_text = " ".join(
                str(item)
                for item in (
                    record.get("public_role"),
                    record.get("claim"),
                    record.get("claimed_role"),
                )
                if item
            )
            ledger = memory_snapshot.get("public_ledger") or {}
            ledger_text = " ".join(
                str(item.get("text") or "") for item in (ledger.get("claims") or [])
                if isinstance(item, Mapping)
            )
            corpus = " ".join(
                self._extract_text_snippets(memory_snapshot.get("recent_public_history"))
                + self._extract_text_snippets(self._recent_public_texts[-4:])
                + [ledger_text]
            )
            if claim_text:
                score += stage_weights["exposed"]
                reasons.append("公开跳身份")
                layer = max(layer, 4)
                if self._has_counterclaim_signal(target_id, claim_text, corpus):
                    score += stage_weights["counterclaim"]
                    reasons.append("对跳/验人链")
                    layer = max(layer, 5)
            sheriff_id = memory_snapshot.get("game", {}).get("sheriff_id")
            if record.get("is_sheriff") or record.get("sheriff") or str(target_id) == str(sheriff_id):
                score += stage_weights["sheriff"]
                reasons.append("警徽位")
                layer = max(layer, 4)
            elif self._has_badge_transfer_signal(target_id, corpus):
                score += max(4, stage_weights["sheriff"] - 2)
                reasons.append("疑似警徽传递位")
                layer = max(layer, 4)
            vote_count = record.get("vote_count", record.get("votes"))
            vote_pressure = self._vote_pressure_for_target(memory_snapshot, target_id)
            locked_votes = max(
                self._nonnegative_int(vote_count, fallback=0) or 0,
                vote_pressure,
            )
            if locked_votes:
                score += min(stage_weights["vote_lock"], 2 + locked_votes * 2)
                reasons.append("票型锁位")
                layer = max(layer, 3)
            speak_count = record.get("speak_count", record.get("speech_count"))
            if isinstance(speak_count, int) and speak_count > 0:
                score += min(stage_weights["public"], 1 + speak_count // 2)
                reasons.append("公共发言活跃")
                layer = max(layer, 2)
        mentions = self._recent_mention_count(target_id)
        if mentions:
            score += min(3, mentions)
            reasons.append(f"近期被提及{mentions}次")
            layer = max(layer, 2)
        if self._is_alive_record(record) is False and "已死亡" not in reasons:
            score -= 40
            reasons.append("不在存活名单")
        death_order = memory_snapshot.get("game", {}).get("death_order") or []
        if isinstance(death_order, list) and target_id in {str(item) for item in death_order[-2:]}:
            score += 1
            reasons.append("延续已有压力")
            layer = max(layer, 2)
        wolf_memory = memory_snapshot.get("wolf_memory") or {}
        failed_targets = {str(item) for item in (wolf_memory.get("failed_kill_targets") or [])}
        if target_id == str(wolf_memory.get("last_wolf_target")) and target_id in failed_targets:
            # 只惩罚最近一次失败刀口，避免把一次平安夜变成永久禁刀。
            score -= 6 if stage == "end" else 12
            reasons.append("上一刀失败，疑似被救/被守")
        if not reasons:
            reasons.append("默认公共威胁")
        return {"score": score, "layer": layer, "reason": ",".join(reasons[:4])}

    def _rank_targets(
        self,
        turn_packet: Mapping[str, Any],
        memory_snapshot: Mapping[str, Any],
        target_actions: list[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        public_state = turn_packet.get("public_state")
        players = self._extract_player_records(public_state, turn_packet.get("game"), memory_snapshot)
        candidates: list[dict[str, Any]] = []
        seen: set[str] = set()
        wolf_memory = memory_snapshot.get("wolf_memory") or {}
        known_wolves = {str(item) for item in (wolf_memory.get("known_wolf_ids") or [])}
        dead_ids = {str(item) for item in (memory_snapshot.get("game", {}).get("dead_players") or [])}
        for action in target_actions:
            for target_id in action.get("target_ids") or []:
                target = str(target_id)
                if target in seen or target == self.player_id or target in known_wolves or target in dead_ids:
                    continue
                seen.add(target)
                record = players.get(target, {})
                profile = self._target_threat_profile(target, record, memory_snapshot)
                candidates.append(
                    {
                        "player_id": target,
                        "score": profile["score"],
                        "layer": profile["layer"],
                        "reason": profile["reason"],
                        "seat": self._seat_sort_value(record, target),
                    }
                )
        candidates.sort(key=lambda item: (-int(item["score"]), -int(item["layer"]), item["seat"], item["player_id"]))
        primary_layer = candidates[0]["layer"] if candidates else None
        ranked: list[dict[str, Any]] = []
        if candidates:
            ranked.append(
                {
                    "player_id": candidates[0]["player_id"],
                    "layer": candidates[0]["layer"],
                    "reason": candidates[0]["reason"],
                }
            )
        backup = self._pick_backup_from_ranked(candidates, primary_layer)
        if backup is not None:
            ranked.append(backup)
        for item in candidates:
            if len(ranked) >= 3:
                break
            if item["player_id"] in {entry["player_id"] for entry in ranked}:
                continue
            ranked.append(
                {
                    "player_id": item["player_id"],
                    "layer": item["layer"],
                    "reason": item["reason"],
                }
            )
        return ranked

    def _score_target(
        self,
        target_id: str,
        record: Mapping[str, Any],
        memory_snapshot: Mapping[str, Any],
    ) -> tuple[int, str]:
        profile = self._target_threat_profile(target_id, record, memory_snapshot)
        return int(profile["score"]), str(profile["reason"])

    def _pick_backup_from_ranked(
        self,
        candidates: list[dict[str, Any]],
        primary_layer: int | None,
    ) -> dict[str, Any] | None:
        if not candidates:
            return None
        if primary_layer is None:
            return {
                "player_id": candidates[1]["player_id"],
                "layer": candidates[1]["layer"],
                "reason": candidates[1]["reason"],
            } if len(candidates) > 1 else None
        for item in candidates[1:]:
            if item["layer"] != primary_layer:
                return {
                    "player_id": item["player_id"],
                    "layer": item["layer"],
                    "reason": item["reason"],
                }
        if len(candidates) > 1:
            item = candidates[1]
            return {"player_id": item["player_id"], "layer": item["layer"], "reason": item["reason"]}
        return None

    def _night_notes(
        self,
        memory_snapshot: Mapping[str, Any],
        current_dialogue: list[dict[str, Any]],
    ) -> list[str]:
        notes: list[str] = []
        stage = self._night_stage(memory_snapshot)
        if stage == "early":
            notes.append("首夜优先真神职暴露与对跳位")
        elif stage == "mid":
            notes.append("中期优先断验人链与警徽线")
        else:
            notes.append("残局优先票型锁位与定点收口")
        game = memory_snapshot.get("game", {})
        sheriff_id = game.get("sheriff_id")
        if sheriff_id:
            notes.append(f"关注警长线：{self._clip_text(str(sheriff_id), 18)}")
        recent_votes = memory_snapshot.get("recent_vote_summary") or []
        if recent_votes:
            notes.append("结合最近票型收敛刀口")
        if current_dialogue:
            notes.append("从当前可见发言里找能串证据链的人")
        wolf_memory = memory_snapshot.get("wolf_memory") or {}
        failed = wolf_memory.get("failed_kill_targets") or []
        if failed:
            notes.append(
                f"上一刀{self._format_id_list([failed[-1]])}疑似失败，默认换刀警徽流/金水/疑似神职"
            )
        return notes[:5]

    def _day_pressure_points(self, memory_snapshot: Mapping[str, Any]) -> list[str]:
        points: list[str] = []
        game = memory_snapshot.get("game", {})
        dead_players = game.get("dead_players") or []
        if dead_players:
            points.append(f"围绕最近死亡：{self._format_id_list(dead_players[-2:])}")
        sheriff_id = game.get("sheriff_id")
        if sheriff_id:
            points.append(f"盯住警长位：{sheriff_id}")
        recent_votes = memory_snapshot.get("recent_vote_summary") or []
        if recent_votes:
            points.append(f"上一轮票型：{self._format_id_list(recent_votes[-2:])}")
        return points[:3]

    def _dialogue_focus(self, current_dialogue: list[dict[str, Any]]) -> list[str]:
        focus: list[str] = []
        for item in current_dialogue[-3:]:
            speaker = item.get("speaker") or item.get("player_id") or item.get("from")
            text = item.get("text") or item.get("content") or item.get("message")
            if speaker is not None and text:
                focus.append(f"{speaker}:{self._clip_text(str(text), 28)}")
        return focus

    def _anti_template_notes(self) -> list[str]:
        corpus = self._recent_public_texts[-4:] + self._recent_self_texts[-4:]
        repeated: list[str] = []
        for phrase in _TEMPLATE_PHRASES:
            hits = sum(1 for text in corpus if phrase in text)
            if hits >= 2:
                repeated.append(phrase)
        if repeated:
            return repeated[:3]
        if self._recent_self_texts:
            last = self._recent_self_texts[-1]
            if last:
                repeated.append(self._clip_text(last, 24))
        return repeated[:3]

    def _recent_mention_count(self, target_id: str) -> int:
        if not target_id:
            return 0
        corpus = " ".join(self._recent_public_texts[-6:] + self._recent_self_texts[-3:])
        if not corpus:
            return 0
        return corpus.count(str(target_id))

    def _has_counterclaim_signal(self, target_id: str, claim_text: str, corpus: str) -> bool:
        haystack = f"{target_id} {claim_text} {corpus}"
        keywords = ("对跳", "验人", "查杀", "预言家", "女巫", "猎人", "守卫")
        return any(keyword in haystack for keyword in keywords)

    def _has_badge_transfer_signal(self, target_id: str, corpus: str) -> bool:
        if not corpus:
            return False
        badge_keywords = ("警徽", "警长", "移交", "接警", "归票")
        return str(target_id) in corpus and any(keyword in corpus for keyword in badge_keywords)

    def _vote_pressure_for_target(self, memory_snapshot: Mapping[str, Any], target_id: str) -> int:
        votes = memory_snapshot.get("recent_vote_summary") or []
        target = str(target_id)
        pressure = 0
        for item in votes:
            if isinstance(item, str) and (item.endswith(f"->{target}") or f"->{target}," in item or f"->{target} " in item):
                pressure += 1
        for item in (memory_snapshot.get("public_ledger", {}).get("votes") or []):
            if isinstance(item, Mapping) and str(item.get("target")) == target:
                pressure += 1
        return pressure

    def _pick_backup_target(self, ranked_targets: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not ranked_targets:
            return None
        primary_layer = ranked_targets[0].get("layer")
        for item in ranked_targets[1:]:
            if item.get("layer") != primary_layer:
                return item
        return ranked_targets[1] if len(ranked_targets) > 1 else None

    def _pick_day_focus_target(self, memory_snapshot: Mapping[str, Any], current_dialogue: list[dict[str, Any]]) -> str | None:
        counts: Counter[str] = Counter()
        alive_players = {
            str(player_id)
            for player_id in (memory_snapshot.get("game", {}).get("alive_players") or [])
            if player_id is not None
        }
        self_id = str(self.player_id)
        for item in memory_snapshot.get("recent_vote_summary") or []:
            if not isinstance(item, str) or "->" not in item:
                continue
            _, target = item.split("->", 1)
            target = target.strip()
            if target and target != self_id:
                counts[target] += 2
        for item in (memory_snapshot.get("public_ledger", {}).get("votes") or []):
            if isinstance(item, Mapping):
                target = str(item.get("target") or "")
                if target and target != self_id:
                    counts[target] += 2
        for item in (memory_snapshot.get("public_ledger", {}).get("claims") or []):
            if isinstance(item, Mapping):
                text = str(item.get("text") or "")
                for target in re.findall(r"p\d+", text):
                    if target != self_id and (not alive_players or target in alive_players):
                        counts[target] += 1
        for item in current_dialogue[-3:]:
            text = str(item.get("text") or item.get("content") or item.get("message") or "")
            for target in re.findall(r"p\d+", text):
                if target != self_id and (not alive_players or target in alive_players):
                    counts[target] += 1
        if not counts:
            return None
        return counts.most_common(1)[0][0]

    def _day_focus_reason(
        self,
        focus_target: str | None,
        recent_vote_summary: list[str],
        current_dialogue: list[dict[str, Any]],
    ) -> str:
        if focus_target is None:
            return "当前没有明确票型焦点，先围绕最近发言和票型收窄判断"
        reasons: list[str] = []
        if any(isinstance(item, str) and focus_target in item for item in recent_vote_summary):
            reasons.append("最近票型直接指向该对象")
        if any(
            focus_target in str(item.get("text") or item.get("content") or item.get("message") or "")
            for item in current_dialogue
        ):
            reasons.append("当前对话正在围绕该对象展开")
        if not reasons:
            reasons.append("它是本轮最容易形成争议收口的对象")
        return "；".join(reasons)

    def _extract_current_round_dialogue(self, turn_packet: Mapping[str, Any]) -> list[dict[str, Any]]:
        tool_context = turn_packet.get("tool_context") or {}
        raw_dialogue = tool_context.get("current_round_dialogue") or []
        return self._summarize_dialogue(raw_dialogue)

    def _resolve_night_plans(
        self,
        summary: Mapping[str, Any],
        *,
        old_phase: str,
        new_phase: str,
        old_round: Any,
        new_round: Any,
        old_dead: set[str],
        new_dead: set[str],
    ) -> None:
        night = any(token in old_phase for token in ("night", "夜", "wolf"))
        day = any(token in new_phase for token in ("day", "昼", "morning", "白天"))
        if not (night and day):
            return
        boundary_round = new_round if new_round is not None else old_round
        plans = self._wolf_memory.get("night_resolution_ledger") or []
        for plan in plans:
            if not isinstance(plan, dict) or plan.get("result") != "pending":
                continue
            plan_round = plan.get("plan_round")
            # 某些内核在黎明递增 round，某些内核在白天沿用 round；两者都须邻接本次边界。
            if plan_round not in {old_round, new_round}:
                continue
            baseline = {str(item) for item in (plan.get("baseline_dead_players") or [])}
            observed_new = sorted((new_dead or set()) - baseline)
            # 只有同步确认了本夜结果才落账；累计死亡集合的重复同步自然幂等。
            target = str(plan.get("target_id"))
            plan["resolved_round"] = boundary_round
            plan["new_deaths"] = observed_new[-6:]
            if target in set(observed_new):
                plan["result"] = "success"
                self._wolf_memory["last_night_deaths"] = observed_new[-4:]
            elif summary.get("no_death") or new_dead != old_dead or day:
                plan["result"] = "no_death" if summary.get("no_death") or new_dead == old_dead else "failed"
                failed = self._wolf_memory.setdefault("failed_kill_targets", [])
                if target not in {str(item) for item in failed}:
                    failed.append(target)
                    self._wolf_memory["failed_kill_targets"] = failed[-4:]

    def _update_public_ledger(self, summary: Mapping[str, Any]) -> None:
        round_value = summary.get("round")
        for claim in summary.get("public_claims") or []:
            if not isinstance(claim, Mapping) or claim.get("speaker") is None:
                continue
            record = dict(claim)
            record["round"] = round_value
            speaker = str(record["speaker"])
            # 同一轮同一发言者的声明以最新可见声明为准，避免重复提及伪造证据强度。
            claims = self._public_ledger["claims"]
            claims[:] = [item for item in claims if not (
                str(item.get("speaker")) == speaker and item.get("round") == round_value
            )]
            claims.append(record)
            self._public_ledger["claims"] = claims[-8:]
        for vote in summary.get("public_vote_records") or []:
            if isinstance(vote, Mapping):
                record = dict(vote)
                record["round"] = round_value
                self._public_ledger["votes"].append(record)
        self._public_ledger["votes"] = self._public_ledger["votes"][-8:]
        dead_now = [str(item) for item in (summary.get("dead_players") or [])]
        previous = {str(item) for item in (self._game_memory.get("dead_players") or [])}
        for player_id in dead_now:
            if player_id not in previous:
                self._public_ledger["deaths"].append({"round": round_value, "player_id": player_id})
        self._public_ledger["deaths"] = self._public_ledger["deaths"][-8:]
        sheriff = summary.get("sheriff_id")
        if sheriff is not None and str(sheriff) != str(self._game_memory.get("sheriff_id")):
            self._public_ledger["sheriff_changes"].append({
                "round": round_value,
                "player_id": str(sheriff),
                "action_type": "sheriff_change",
            })
            self._public_ledger["sheriff_changes"] = self._public_ledger["sheriff_changes"][-8:]

    def _extract_public_claims(self, packet: Mapping[str, Any], combined: Mapping[str, Any]) -> list[dict[str, Any]]:
        raw = None
        for source in (packet, combined):
            for key in ("dialogue", "public_dialogue", "events", "history"):
                if isinstance(source, Mapping) and source.get(key):
                    raw = source[key]
                    break
            if raw is not None:
                break
        items = raw if isinstance(raw, list) else ([raw] if raw is not None else [])
        result: list[dict[str, Any]] = []
        for item in items[-_MAX_HISTORY_ITEMS:]:
            if not isinstance(item, Mapping):
                continue
            speaker = self._first_value(item, ("speaker", "speaker_id", "player_id", "from", "by"))
            text = self._first_value(item, ("text", "content", "utterance", "speech", "message"))
            if speaker is None or text is None:
                continue
            text = self._clip_text(str(text), _MAX_SNIPPET_LENGTH)
            role = next((r for r in ("预言家", "女巫", "守卫", "猎人", "平民") if r in text), None)
            if role is None and not any(word in text for word in ("查杀", "金水", "对跳", "警徽", "警长")):
                continue
            claim_type = "role_claim" if role else "relation_claim"
            result.append({"speaker": str(speaker), "claim_type": claim_type, "claimed_role": role, "text": text})
        return result[-6:]

    def _extract_public_vote_records(self, combined: Mapping[str, Any]) -> list[dict[str, Any]]:
        raw = combined.get("votes") or combined.get("vote_history")
        result: list[dict[str, Any]] = []
        if isinstance(raw, Mapping):
            raw = [{"voter": voter, "target": target} for voter, target in raw.items()]
        for item in raw if isinstance(raw, list) else []:
            if isinstance(item, Mapping):
                voter = self._first_value(item, ("voter", "from", "player_id"))
                target = self._first_value(item, ("target", "target_id", "to"))
                if voter is not None and target is not None:
                    result.append({"voter": str(voter), "target": str(target), "action_type": "vote"})
        return result[-_MAX_HISTORY_ITEMS:]

    def _summarize_sync_packet(self, sync_packet: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(sync_packet, Mapping):
            return {}
        game = sync_packet.get("game") if isinstance(sync_packet.get("game"), Mapping) else {}
        public_state = (
            sync_packet.get("public_state") if isinstance(sync_packet.get("public_state"), Mapping) else {}
        )
        combined: dict[str, Any] = {}
        combined.update(self._flatten_sync_fields(sync_packet))
        combined.update(self._flatten_sync_fields(game))
        combined.update(self._flatten_sync_fields(public_state))
        summary: dict[str, Any] = {}
        round_value = combined.get("round")
        if round_value is not None:
            summary["round"] = round_value
        phase_value = combined.get("phase", combined.get("public_phase"))
        if phase_value is not None:
            summary["phase"] = phase_value
        sheriff_value = combined.get("sheriff_id", combined.get("chairman_id"))
        if sheriff_value is None and isinstance(combined.get("sheriff"), Mapping):
            sheriff_value = self._first_value(combined["sheriff"], ("player_id", "id", "seat", "seat_id"))
        if sheriff_value is not None:
            summary["sheriff_id"] = sheriff_value
        alive_players = self._extract_id_list(
            combined,
            preferred_keys=("alive_players", "alive", "living_players", "survivors"),
        )
        if alive_players:
            summary["alive_players"] = alive_players
        dead_players = self._extract_id_list(
            combined,
            preferred_keys=("dead_players", "dead", "eliminated_players", "casualties"),
        )
        if not dead_players:
            dead_players = self._extract_event_deaths(combined.get("events") or combined.get("history"))
        if dead_players:
            summary["dead_players"] = dead_players
            summary["death_order"] = dead_players
        votes = self._summarize_votes(combined)
        if votes:
            summary["recent_votes"] = votes
        public_claims = self._extract_public_claims(sync_packet, combined)
        if public_claims:
            summary["public_claims"] = public_claims
        public_vote_records = self._extract_public_vote_records(combined)
        if public_vote_records:
            summary["public_vote_records"] = public_vote_records
        events = combined.get("events") or combined.get("history") or []
        if combined.get("no_death") is True or self._contains_no_death_signal(events):
            summary["no_death"] = True
        texts = self._extract_text_snippets(sync_packet)
        if texts:
            summary["public_texts"] = texts
        if not summary:
            summary["round"] = self._safe_get(game, "round") or self._safe_get(public_state, "round")
            summary["phase"] = self._safe_get(game, "phase") or self._safe_get(public_state, "phase")
        return summary

    def _flatten_sync_fields(self, value: Any) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            return {}
        flattened: dict[str, Any] = {}
        for key in (
            "round",
            "phase",
            "public_phase",
            "sheriff_id",
            "sheriff",
            "chairman_id",
            "chairman",
            "alive_players",
            "dead_players",
            "alive",
            "dead",
            "players",
            "votes",
            "vote_history",
            "dialogue",
            "public_dialogue",
            "events",
            "history",
            "death_order",
            "no_death",
        ):
            if key in value:
                flattened[key] = value[key]
        for key in ("sheriff", "chairman"):
            if key in flattened and isinstance(flattened[key], Mapping):
                flattened[f"{key}_id"] = self._first_value(flattened[key], ("player_id", "id", "seat", "seat_id"))
        return flattened

    def _summarize_dialogue(self, dialogue: Any) -> list[dict[str, Any]]:
        if not dialogue:
            return []
        items = dialogue if isinstance(dialogue, list) else [dialogue]
        snippets: list[dict[str, Any]] = []
        for item in items[-_MAX_HISTORY_ITEMS:]:
            if isinstance(item, Mapping):
                speaker = self._first_value(item, ("speaker", "speaker_id", "player_id", "from", "by"))
                text = self._first_value(item, ("text", "content", "utterance", "speech", "message"))
                phase = self._first_value(item, ("phase", "public_phase"))
                snippet: dict[str, Any] = {}
                if speaker is not None:
                    snippet["speaker"] = str(speaker)
                if phase is not None:
                    snippet["phase"] = str(phase)
                if text is not None:
                    snippet["text"] = self._clip_text(str(text), _MAX_SNIPPET_LENGTH)
                if snippet:
                    snippets.append(snippet)
            elif isinstance(item, str):
                snippets.append({"text": self._clip_text(item, _MAX_SNIPPET_LENGTH)})
            else:
                snippets.append({"text": self._clip_text(str(item), _MAX_SNIPPET_LENGTH)})
        return snippets[-_MAX_HISTORY_ITEMS:]

    def _extract_text_snippets(self, value: Any) -> list[str]:
        snippets: list[str] = []
        if isinstance(value, Mapping):
            for key in ("dialogue", "public_dialogue", "events", "history"):
                if key in value:
                    snippets.extend(self._extract_text_snippets(value[key]))
            for key in ("text", "content", "message", "utterance"):
                if key in value and isinstance(value[key], str):
                    snippets.append(self._clip_text(value[key], _MAX_SNIPPET_LENGTH))
        elif isinstance(value, list):
            for item in value[-_MAX_HISTORY_ITEMS:]:
                snippets.extend(self._extract_text_snippets(item))
        elif isinstance(value, str):
            snippets.append(self._clip_text(value, _MAX_SNIPPET_LENGTH))
        return snippets[-_MAX_HISTORY_ITEMS:]

    def _extract_event_deaths(self, value: Any) -> list[str]:
        result: list[str] = []
        items = value if isinstance(value, list) else [value]
        for item in items:
            if not isinstance(item, Mapping):
                continue
            event_type = str(item.get("type") or item.get("event") or item.get("kind") or "").upper()
            if not any(token in event_type for token in ("PLAYER_DIED", "PLAYER_DEAD", "ELIMINATED", "DEATH")):
                continue
            pid = self._first_value(item, ("player_id", "target_id", "victim", "dead_player", "id"))
            if pid is not None and str(pid) not in result:
                result.append(str(pid))
        return result[-_MAX_HISTORY_ITEMS:]

    def _contains_no_death_signal(self, value: Any) -> bool:
        if isinstance(value, Mapping):
            event_type = str(value.get("type") or value.get("event") or value.get("kind") or "").upper()
            if "NO_ONE_DIED" in event_type or "NO_DEATH" in event_type:
                return True
            return any(self._contains_no_death_signal(item) for item in value.values())
        if isinstance(value, list):
            return any(self._contains_no_death_signal(item) for item in value[-_MAX_HISTORY_ITEMS:])
        return "无人死亡" in str(value) or "平安夜" in str(value)

    def _summarize_votes(self, value: Mapping[str, Any]) -> list[str]:
        votes = value.get("votes") or value.get("vote_history")
        if not votes:
            return []
        items: list[str] = []
        if isinstance(votes, Mapping):
            for voter, target in votes.items():
                items.append(f"{voter}->{target}")
        elif isinstance(votes, list):
            for entry in votes[-_MAX_HISTORY_ITEMS:]:
                if isinstance(entry, Mapping):
                    voter = self._first_value(entry, ("voter", "from", "player_id"))
                    target = self._first_value(entry, ("target", "target_id", "to"))
                    if voter is not None or target is not None:
                        items.append(f"{voter}->{target}")
                else:
                    items.append(self._clip_text(str(entry), 40))
        return items[-_MAX_HISTORY_ITEMS:]

    def _extract_player_records(
        self,
        public_state: Any,
        game: Any,
        memory_snapshot: Mapping[str, Any],
    ) -> dict[str, dict[str, Any]]:
        records: dict[str, dict[str, Any]] = {}

        def ensure_record(player_id: Any) -> dict[str, Any]:
            pid = str(player_id)
            record = records.setdefault(pid, {"player_id": pid})
            return record

        def ingest_players(players: Any, alive: bool | None = None) -> None:
            if isinstance(players, Mapping):
                for pid, info in players.items():
                    record = ensure_record(pid)
                    if isinstance(info, Mapping):
                        record.update(self._compact_snapshot(info, preferred_keys=("seat", "seat_id", "alive", "status", "role", "claim", "claimed_role", "public_role", "vote_count", "votes", "speak_count", "speech_count")))
                    if alive is not None and "alive" not in record:
                        record["alive"] = alive
            elif isinstance(players, list):
                for item in players:
                    if isinstance(item, Mapping):
                        pid = self._first_value(item, ("player_id", "id", "seat", "seat_id"))
                        if pid is None:
                            continue
                        record = ensure_record(pid)
                        record.update(self._compact_snapshot(item, preferred_keys=("player_id", "id", "seat", "seat_id", "alive", "status", "role", "claim", "claimed_role", "public_role", "vote_count", "votes", "speak_count", "speech_count")))
                        if alive is not None and "alive" not in record:
                            record["alive"] = alive
                    else:
                        record = ensure_record(item)
                        if alive is not None:
                            record["alive"] = alive

        if isinstance(public_state, Mapping):
            ingest_players(public_state.get("players"))
            ingest_players(public_state.get("alive_players"), alive=True)
            ingest_players(public_state.get("dead_players"), alive=False)
            ingest_players(public_state.get("living_players"), alive=True)
            ingest_players(public_state.get("eliminated_players"), alive=False)
            sheriff = self._first_value(public_state, ("sheriff_id", "chairman_id"))
            if sheriff is None and isinstance(public_state.get("sheriff"), Mapping):
                sheriff = self._first_value(public_state["sheriff"], ("player_id", "id", "seat", "seat_id"))
            if sheriff is None and isinstance(public_state.get("chairman"), Mapping):
                sheriff = self._first_value(public_state["chairman"], ("player_id", "id", "seat", "seat_id"))
            if sheriff is not None:
                ensure_record(sheriff)["is_sheriff"] = True
            votes = public_state.get("votes")
            if isinstance(votes, Mapping):
                for voter, target in votes.items():
                    if isinstance(target, Mapping):
                        target_id = self._first_value(target, ("player_id", "id", "seat", "seat_id"))
                    else:
                        target_id = target
                    ensure_record(voter).setdefault("vote_target", target_id)
                    if target_id is None:
                        continue
                    target_record = ensure_record(target_id)
                    current_votes = self._nonnegative_int(target_record.get("vote_count"), fallback=0) or 0
                    target_record["vote_count"] = current_votes + 1
        if isinstance(game, Mapping):
            ingest_players(game.get("alive_players"), alive=True)
            ingest_players(game.get("dead_players"), alive=False)
            ingest_players(game.get("players"))
            if "death_order" in game:
                for player_id in self._extract_id_list(game, preferred_keys=("death_order",)):
                    ensure_record(player_id).setdefault("death_order", True)

        for player_id in memory_snapshot.get("game", {}).get("alive_players") or []:
            ensure_record(player_id).setdefault("alive", True)
        for player_id in memory_snapshot.get("game", {}).get("dead_players") or []:
            ensure_record(player_id).setdefault("alive", False)
        if memory_snapshot.get("game", {}).get("sheriff_id") is not None:
            ensure_record(memory_snapshot["game"]["sheriff_id"])["is_sheriff"] = True
        return records

    def _compact_request(self, value: Any) -> dict[str, Any]:
        raw = value if isinstance(value, Mapping) else {}
        result: dict[str, Any] = {}
        for key in ("request_id", "phase", "channel", "max_chars", "speaker_id"):
            if key in raw:
                result[key] = self._compact_snapshot(raw[key], depth=1)
        actions: list[dict[str, Any]] = []
        for action in raw.get("allowed_actions") or []:
            if not isinstance(action, Mapping):
                continue
            item: dict[str, Any] = {}
            for key, item_value in action.items():
                if key == "target_ids" and isinstance(item_value, list):
                    item[key] = [str(target) for target in item_value]
                elif key in ("kind", "max_chars", "require_chinese"):
                    item[key] = item_value
            actions.append(item)
        result["allowed_actions"] = actions
        return result

    def _build_player_index(
        self,
        public_state: Any,
        game: Any,
        memory_snapshot: Mapping[str, Any],
        request: Any,
    ) -> list[dict[str, Any]]:
        records = self._extract_player_records(public_state, game, memory_snapshot)
        allowed_ids: list[str] = []
        if isinstance(request, Mapping):
            for action in request.get("allowed_actions") or []:
                if isinstance(action, Mapping):
                    allowed_ids.extend(str(pid) for pid in (action.get("target_ids") or []))
        all_ids = set(records) | set(allowed_ids)
        for pid in (memory_snapshot.get("game", {}).get("alive_players") or []):
            all_ids.add(str(pid))
        for pid in (memory_snapshot.get("game", {}).get("dead_players") or []):
            all_ids.add(str(pid))
        claims = memory_snapshot.get("public_ledger", {}).get("claims") or []
        claim_map: dict[str, str] = {}
        for claim in claims:
            if isinstance(claim, Mapping) and claim.get("speaker") is not None:
                claim_map[str(claim["speaker"])] = self._clip_text(str(claim.get("text") or ""), 40)
        result: list[dict[str, Any]] = []
        for pid in sorted(all_ids, key=lambda item: self._seat_sort_value(records.get(item, {}), item)):
            record = records.get(pid, {})
            entry: dict[str, Any] = {"player_id": pid, "alive": self._is_alive_record(record)}
            seat = record.get("seat", record.get("seat_id"))
            if seat is not None:
                entry["seat"] = seat
            if pid == str(memory_snapshot.get("game", {}).get("sheriff_id")) or record.get("is_sheriff"):
                entry["sheriff"] = True
            if pid in claim_map:
                entry["public_claim"] = claim_map[pid]
            result.append(entry)
        return result

    def _compact_public_state(self, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return self._compact_snapshot(value)
        return self._compact_snapshot(
            value,
            preferred_keys=(
                "round",
                "phase",
                "public_phase",
                "alive_players",
                "dead_players",
                "votes",
                "vote_history",
                "sheriff",
                "sheriff_id",
                "chairman",
                "chairman_id",
                "death_order",
                "dialogue",
                "public_dialogue",
            ),
        )

    def _compact_snapshot(
        self,
        value: Any,
        *,
        preferred_keys: tuple[str, ...] = (),
        depth: int = _MAX_COMPACT_DEPTH,
        max_items: int = 8,
    ) -> Any:
        if depth <= 0:
            return self._clip_primitive(value)
        if isinstance(value, Mapping):
            compact: dict[str, Any] = {}
            for key in preferred_keys:
                if key in value and key not in compact:
                    compact[str(key)] = self._compact_snapshot(
                        value[key], preferred_keys=(), depth=depth - 1, max_items=max_items
                    )
            for key, item in value.items():
                if len(compact) >= max_items:
                    break
                if key in compact:
                    continue
                compact[str(key)] = self._compact_snapshot(
                    item, preferred_keys=(), depth=depth - 1, max_items=max_items
                )
            return compact
        if isinstance(value, list):
            compact_list = [
                self._compact_snapshot(item, preferred_keys=(), depth=depth - 1, max_items=max_items)
                for item in value[:max_items]
            ]
            if len(value) > max_items:
                compact_list.append(f"...(+{len(value) - max_items})")
            return compact_list
        if isinstance(value, tuple):
            return self._compact_snapshot(list(value), preferred_keys=preferred_keys, depth=depth, max_items=max_items)
        return self._clip_primitive(value)

    def _clip_primitive(self, value: Any) -> Any:
        if isinstance(value, str):
            return self._clip_text(value, 120)
        return value

    @staticmethod
    def _clip_text(text: str, limit: int) -> str:
        text = str(text)
        if len(text) <= limit:
            return text
        return text[: max(0, limit - 1)] + "…"

    @staticmethod
    def _safe_get(value: Any, key: str) -> Any:
        if isinstance(value, Mapping):
            return value.get(key)
        return None

    def _first_value(self, value: Any, keys: tuple[str, ...]) -> Any:
        if not isinstance(value, Mapping):
            return None
        for key in keys:
            if key in value and value[key] is not None:
                return value[key]
        return None

    def _extract_id_list(self, value: Mapping[str, Any], *, preferred_keys: tuple[str, ...]) -> list[str]:
        for key in preferred_keys:
            if key not in value:
                continue
            raw = value[key]
            if raw is None:
                continue
            if isinstance(raw, list):
                return [str(item) for item in raw if item is not None]
            if isinstance(raw, Mapping):
                return [str(item) for item in raw.keys() if item is not None]
            return [str(raw)]
        return []

    def _fit_prompt_payload(
        self,
        payload: Mapping[str, Any],
        *,
        system: str,
        instruction: str,
    ) -> dict[str, Any]:
        """按总字符预算裁剪，始终保留当前行动契约和玩家索引。"""
        fitted = json.loads(json.dumps(payload, ensure_ascii=False))
        # 为 JSON wrapper、服务端标记和边界留出余量；不是只测 packet 自身。
        budget = max(256, self.max_prompt_chars - len(system) - len(instruction) - 700)
        def size() -> int:
            wrapper = {"instruction": instruction, "packet": fitted}
            return len(json.dumps(wrapper, ensure_ascii=False, separators=(",", ":")))
        if size() <= budget:
            return fitted
        # 优先删除重复/低价值原文，保留结构化账本和行动包。
        for key in ("current_round_dialogue", "recent_public_history", "recent_self_history"):
            if isinstance(fitted.get("memory"), dict):
                fitted["memory"].pop(key, None)
            fitted.pop(key, None)
            if size() <= budget:
                return fitted
        if isinstance(fitted.get("wolf_strategy"), dict):
            strategy = fitted["wolf_strategy"]
            for key in ("current_round_dialogue", "recent_vote_summary", "notes", "pressure_points", "claims_internal"):
                strategy.pop(key, None)
                if size() <= budget:
                    return fitted
        if isinstance(fitted.get("memory"), dict):
            fitted["memory"]["public_ledger"] = {
                key: value[-3:] if isinstance(value, list) else value
                for key, value in (fitted["memory"].get("public_ledger") or {}).items()
            }
        if size() <= budget:
            return fitted
        # 最后只压缩非契约字段；request.allowed_actions/target_ids、player_index、
        # 狼队身份和最近结算结果不在裁剪清单中。
        for key in ("public_rules", "game", "self", "private_information"):
            fitted.pop(key, None)
            if size() <= budget:
                return fitted
        return fitted

    @staticmethod
    def _is_alive_record(record: Mapping[str, Any]) -> bool | None:
        if not record:
            return None
        if record.get("alive") is True:
            return True
        status = record.get("status")
        if status in {"dead", "eliminated"}:
            return False
        if status in {"alive", "living"}:
            return True
        if record.get("is_dead") is True:
            return False
        if record.get("is_alive") is True:
            return True
        return None

    @staticmethod
    def _seat_sort_value(record: Mapping[str, Any], target_id: str) -> tuple[int, str]:
        seat = record.get("seat") if isinstance(record, Mapping) else None
        if seat is None and isinstance(record, Mapping):
            seat = record.get("seat_id")
        if seat is None:
            match = re.search(r"\d+", str(target_id))
            if match:
                return int(match.group()), str(target_id)
            return 10**9, str(target_id)
        try:
            return int(seat), str(target_id)
        except (TypeError, ValueError):
            match = re.search(r"\d+", str(seat))
            if match:
                return int(match.group()), str(target_id)
            return 10**9, str(target_id)

    @staticmethod
    def _format_id_list(values: list[Any]) -> str:
        return "、".join(str(item) for item in values if item is not None)



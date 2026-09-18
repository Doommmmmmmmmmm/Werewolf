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
            # 每条记录均来自已观测的公开结果；不把模型猜测写入这里。
            "night_failure_ledger": [],
            "known_wolf_ids_cache": [],
            "alive_wolf_ids_cache": [],
            "no_death_rounds": [],
        }

    async def observe(self, sync_packet: dict[str, Any]) -> None:
        """接收公开同步并压缩成轻量记忆。"""

        summary = self._summarize_sync_packet(sync_packet)
        if not summary:
            return

        old_dead = {str(item) for item in (self._game_memory.get("dead_players") or [])}
        new_dead = {str(item) for item in (summary.get("dead_players") or [])}
        if summary.get("no_death"):
            round_value = summary.get("round")
            if round_value is not None and round_value not in self._wolf_memory["no_death_rounds"]:
                self._wolf_memory["no_death_rounds"].append(round_value)
                self._wolf_memory["no_death_rounds"] = self._wolf_memory["no_death_rounds"][-4:]
            self._record_night_failure(round_value)
        elif new_dead and new_dead != old_dead:
            # 夜间目标死亡才算成功；白天死了别人时，上一刀目标仍应进入失败账本。
            self._record_night_failure(summary.get("round"), dead_ids=new_dead)
            self._wolf_memory["last_night_deaths"] = list(new_dead)[-4:]

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

    def _record_night_failure(self, round_value: Any, dead_ids: set[str] | None = None) -> None:
        """把可见的平安夜/目标未死亡结果写入按目标计数的账本。"""
        target = self._wolf_memory.get("last_wolf_target")
        if target is None:
            return
        target = str(target)
        if dead_ids is not None and target in {str(item) for item in dead_ids}:
            return
        ledger = self._wolf_memory.setdefault("night_failure_ledger", [])
        existing = next((item for item in ledger if str(item.get("target")) == target), None)
        if isinstance(existing, Mapping) and existing.get("last_failed_round") == round_value:
            return
        count = int(existing.get("failure_count", 0)) + 1 if isinstance(existing, Mapping) else 1
        record = {
            "round": round_value,
            "target": target,
            "failure_count": count,
            "last_failed_round": round_value,
        }
        if isinstance(existing, Mapping):
            ledger[:] = [item for item in ledger if item is not existing]
        ledger.append(record)
        self._wolf_memory["night_failure_ledger"] = ledger[-8:]
        failed = self._wolf_memory.setdefault("failed_kill_targets", [])
        if target not in {str(item) for item in failed}:
            failed.append(target)
        self._wolf_memory["failed_kill_targets"] = [str(item) for item in failed][-8:]

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
            "wolf_memory": {
                "recent_kill_votes": self._wolf_memory.get("recent_kill_votes", [])[-3:],
                "last_wolf_target": self._wolf_memory.get("last_wolf_target"),
                "last_kill_round": self._wolf_memory.get("last_kill_round"),
                "last_night_deaths": self._wolf_memory.get("last_night_deaths", [])[-3:],
                "failed_kill_targets": self._wolf_memory.get("failed_kill_targets", [])[-5:],
                "night_failure_ledger": self._wolf_memory.get("night_failure_ledger", [])[-5:],
                "known_wolf_ids": self._wolf_memory.get("known_wolf_ids_cache", []),
                "alive_wolf_ids": self._wolf_memory.get("alive_wolf_ids_cache", []),
                "no_death_rounds": self._wolf_memory.get("no_death_rounds", [])[-3:],
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
        # 玩家索引不按座位截断，避免高座位玩家从模型视野中消失；删除重复的详细 players。
        if isinstance(turn_packet.get("public_state"), Mapping) and isinstance(public_state, dict):
            public_state.pop("players", None)
            public_state["player_index"] = self._public_player_index(
                turn_packet.get("public_state"), turn_packet.get("game"), memory_snapshot
            )
        prompt_memory = dict(memory_snapshot)
        prompt_memory["recent_claim_edges"] = self._extract_public_claim_edges(
            current_dialogue, memory_snapshot
        )[-8:]
        prompt_memory["recent_public_history"] = [
            {"round": item.get("round"), "phase": item.get("phase"),
             "votes": item.get("recent_votes", [])[-4:],
             "deaths": item.get("dead_players", [])[-4:],
             "texts": (item.get("public_texts") or [])[-2:]}
            for item in (memory_snapshot.get("recent_public_history") or [])[-3:]
            if isinstance(item, Mapping)
        ]
        request = self._compact_snapshot(
            turn_packet.get("request"),
            preferred_keys=(
                "request_id",
                "phase",
                "channel",
                "allowed_actions",
                "max_chars",
                "target_ids",
                "speaker_id",
            ),
        )
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
        wolf_strategy = self._build_wolf_strategy(turn_packet, prompt_memory, current_dialogue)
        payload = {
            "game": game,
            "public_rules": public_rules,
            "self": self_state,
            "private_information": private_information,
            "public_state": public_state,
            "request": request,
            "memory": prompt_memory,
            "wolf_strategy": wolf_strategy,
            # 当前可见原文只注入一次；wolf_strategy 只保留由它提炼出的结构化结果。
            "visible_dialogue": current_dialogue[-3:],
        }
        return self._fit_prompt_payload(payload)

    def _fit_prompt_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        """给 system/task 留出余量；只删低价值原文，不删合法目标和事实索引。"""
        def size() -> int:
            return len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        if size() <= 7000:
            return payload
        payload["memory"]["recent_public_history"] = payload["memory"].get("recent_public_history", [])[-2:]
        payload["memory"]["recent_claim_edges"] = payload["memory"].get("recent_claim_edges", [])[-5:]
        payload["visible_dialogue"] = payload.get("visible_dialogue", [])[-2:]
        strategy = payload.get("wolf_strategy")
        if isinstance(strategy, dict):
            strategy.pop("notes", None)
            strategy.pop("pressure_points", None)
            strategy.pop("claim_edges", None)
        if size() <= 7000:
            return payload
        # 最后只压缩文本字段；全体 player_index、allowed_actions 和关系方向仍保留。
        for key in ("public_rules", "memory", "wolf_strategy"):
            value = payload.get(key)
            if isinstance(value, dict):
                for nested_key in ("recent_public_history", "recent_self_history", "recent_claim_edges", "rules"):
                    if nested_key in value:
                        value[nested_key] = value[nested_key][-2:] if isinstance(value[nested_key], list) else value[nested_key]

        def trim_text(value: Any) -> Any:
            if isinstance(value, str):
                return self._clip_text(value, 64)
            if isinstance(value, list):
                return [trim_text(item) for item in value[-4:]]
            if isinstance(value, dict):
                return {key: trim_text(item) for key, item in value.items()}
            return value
        return trim_text(payload)

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
            consensus = self._team_consensus_target(current_dialogue, target_actions, memory_snapshot)
            ranked_targets = self._rank_targets(
                turn_packet, memory_snapshot, target_actions, current_dialogue=current_dialogue
            )
            primary = ranked_targets[0] if ranked_targets else None
            if consensus is not None:
                consensus_entry = next(
                    (item for item in ranked_targets if item.get("player_id") == consensus["target"]), None
                )
                if consensus_entry is not None and not consensus_entry.get("avoid"):
                    primary = dict(consensus_entry)
                    primary["reason"] = f"狼队共识{consensus['support']}票；{primary.get('reason', '')}"
            backup = self._pick_backup_target(ranked_targets)
            return {
                "stage": "night",
                "phase_bucket": self._night_stage(memory_snapshot),
                "objective": "收敛到一个主刀和一个来自不同威胁层级的备选刀",
                "primary_target": primary,
                "backup_target": backup,
                "team_consensus_target": consensus,
                "target_diversity_required": True,
                "tie_break": "先看成功概率与失败冷却，再看真神职/警徽/票型收益；冷却目标仅在残局或无替代时 override",
                "recent_vote_summary": recent_vote_summary,
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
            claim_edges = self._extract_public_claim_edges(current_dialogue, memory_snapshot)
            claims = self._claims_by_speaker(claim_edges)
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
                "pressure_points": self._day_pressure_points(memory_snapshot),
                "claim_edges": claim_edges[-8:],
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
        claim_edges = self._extract_public_claim_edges(current_dialogue, memory_snapshot)
        claims = self._claims_by_speaker(claim_edges)
        wolf_memory = memory_snapshot.get("wolf_memory") or {}
        alive_wolves = [str(item) for item in (wolf_memory.get("alive_wolf_ids") or [])]
        known_wolves = {str(item) for item in (wolf_memory.get("known_wolf_ids") or [])}
        alive = self._alive_ids_from_state(turn_packet.get("public_state"), turn_packet.get("game"))
        if alive:
            alive_wolves = [pid for pid in alive_wolves if pid in alive]
        dead_wolves = sorted(known_wolves - set(alive_wolves))
        candidate_ids = self._legal_candidate_ids(target_actions, exclude={self.player_id})
        pressure = self._estimate_vote_pressure(current_dialogue, turn_packet, candidate_ids)
        teammate_state: dict[str, str] = {}
        doomed: list[str] = []
        pressured: list[str] = []
        for wolf in alive_wolves:
            if wolf not in candidate_ids:
                continue
            kill_edges = [edge for edge in claim_edges if edge.get("claim_type") == "查杀" and edge.get("target_id") == wolf]
            support = {str(edge.get("speaker_id")) for edge in kill_edges if edge.get("confidence", 0) >= 0.75}
            weighted = float((pressure.get(wolf) or {}).get("weighted_votes", 0))
            if len(support) >= 2 or (kill_edges and (weighted >= 2.5 or self._has_sheriff_support(kill_edges, turn_packet))):
                teammate_state[wolf] = "mathematically_doomed"
                doomed.append(wolf)
            elif kill_edges or weighted >= 1.5:
                teammate_state[wolf] = "pressured"
                pressured.append(wolf)
            else:
                teammate_state[wolf] = "safe"
        wolf_set = set(alive_wolves)
        dangerous_claims = self._claims_by_speaker(
            [edge for edge in claim_edges if edge.get("claim_type") in {"role_claim", "查杀", "金水", "对跳", "守护", "猎人"}]
        )
        hunter_claimants = [str(edge["speaker_id"]) for edge in claim_edges if edge.get("claim_type") == "猎人"]
        # 数学必死时切割，其他情况下不把队友列入正常候选；优先投查杀者或其对立位。
        opposing = []
        for wolf in doomed:
            opposing.extend(
                str(edge.get("speaker_id")) for edge in claim_edges
                if edge.get("claim_type") == "查杀" and edge.get("target_id") == wolf
            )
        recommended: list[str] = []
        recommended.extend(pid for pid in doomed if pid in candidate_ids)
        recommended.extend(pid for pid in opposing if pid in candidate_ids and pid not in wolf_set)
        non_wolves = [pid for pid in candidate_ids if pid not in wolf_set]
        non_wolves.sort(key=lambda pid: (-float((pressure.get(pid) or {}).get("weighted_votes", 0)), pid))
        recommended.extend(pid for pid in non_wolves if pid not in recommended)
        # 没有必死队友时，少量倒钩只取一名狼队以外目标，避免机械同票。
        if not doomed:
            recommended = [pid for pid in recommended if pid not in wolf_set]
        if not recommended:
            recommended = [pid for pid in candidate_ids if pid not in wolf_set] or candidate_ids[:3]
        return {
            "stage": "day_vote",
            "objective": "用预计票型保留狼队票权；队友接近必死时自然切割，否则不主动补刀",
            "alive_wolf_ids": alive_wolves,
            "dead_wolf_ids_count": len(dead_wolves),
            "endgame_pressure": len(alive_wolves) <= 2 or len(alive) <= 5,
            "claim_edges": claim_edges[-8:],
            "dangerous_claims": dangerous_claims,
            "teammate_status": teammate_state,
            "teammate_under_pressure": pressured[:3],
            "mathematically_doomed": doomed[:3],
            "hunter_claimants": list(dict.fromkeys(hunter_claimants))[:3],
            "predicted_vote_pressure": pressure,
            "recommended_vote_targets": recommended[:5],
            "rules": [
                "明确查杀+多源支持或票数不可逆时允许切割；单人怀疑不能触发硬保/硬切",
                "无明确查杀时不投活狼；必要倒钩只由票型决定，不要全狼同票",
                "猎人对跳未确认时避免把高风险猎人送出，优先攻击其声明的依据",
            ],
            "claims_internal": claims,
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

    def _claim_items(self, dialogue: Any) -> list[tuple[str | None, str, str]]:
        """返回 (speaker, text, phase)，只使用当前可见或已同步的公开文本。"""
        items: list[Any] = list(dialogue) if isinstance(dialogue, list) else []
        for history_item in self._public_history[-3:]:
            if isinstance(history_item, Mapping):
                items.extend(history_item.get("public_dialogue") or [])
                if not history_item.get("public_dialogue"):
                    items.extend(history_item.get("public_texts") or [])
        result: list[tuple[str | None, str, str]] = []
        for item in items:
            if isinstance(item, Mapping):
                speaker = self._first_value(item, ("speaker", "speaker_id", "player_id", "from", "by"))
                text = self._first_value(item, ("text", "content", "message", "utterance", "speech")) or ""
                phase = self._first_value(item, ("phase", "public_phase")) or "public"
            else:
                speaker, text, phase = None, str(item), "public"
            text = str(text)
            if speaker is None:
                # 无 speaker 的文本不能安全归属为某人的硬声明。
                result.append((None, text, str(phase)))
            else:
                result.append((str(speaker), text, str(phase)))
        return result[-12:]

    def _extract_public_claim_edges(
        self, dialogue: Any, memory_snapshot: Mapping[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """提取 speaker -> target 的软/硬关系边；解析结果不是引擎事实。"""
        edges: list[dict[str, Any]] = []
        for speaker, text, phase in self._claim_items(dialogue):
            if not speaker or not text:
                continue
            ids = list(dict.fromkeys(re.findall(r"p\d+", text)))
            def add(kind: str, target: str | None, result: str = "", confidence: float = 0.7) -> None:
                if target is None or target == speaker:
                    return
                edges.append({
                    "claim_type": kind, "speaker_id": speaker, "target_id": target,
                    "result": result, "round": (memory_snapshot or {}).get("game", {}).get("round"),
                    "source_phase": phase, "confidence": confidence,
                    "source_text": self._clip_text(text, 100),
                })
            # 明确关系优先于角色关键词，避免把“他说查杀p2”误当成系统信息。
            for match in re.finditer(r"查杀\s*(p\d+)|(p\d+)\s*(?:是|为)?\s*查杀", text):
                add("查杀", match.group(1) or match.group(2), "wolf", 0.9)
            for match in re.finditer(r"(?:验了?|查验|验人|给)\s*(p\d+)\s*(?:是)?\s*(金水|查杀)?", text):
                target, result = match.group(1), match.group(2) or "unknown"
                add("金水" if result == "金水" else ("查杀" if result == "查杀" else "验人"), target, result, 0.85)
            for match in re.finditer(r"(?:金水|保)\s*(p\d+)|(p\d+)\s*(?:是|为)?\s*金水", text):
                add("金水", match.group(1) or match.group(2), "good", 0.85)
            for match in re.finditer(r"(?:守了?|保护|盾)\s*(p\d+)|(p\d+)\s*(?:被守|有盾)", text):
                add("守护", match.group(1) or match.group(2), "protected", 0.75)
            for match in re.finditer(r"(?:对跳|和|跟)\s*(p\d+)\s*(?:对跳|跳)", text):
                add("对跳", match.group(1), "counterclaim", 0.7)
            if "对跳" in text and len(ids) >= 2:
                add("对跳", ids[-1], "counterclaim", 0.65)
            for match in re.finditer(r"(?:我是|自称|跳)\s*(预言家|女巫|守卫|猎人|平民)", text):
                role = match.group(1)
                edges.append({"claim_type": "role_claim", "speaker_id": speaker, "target_id": None,
                              "result": role, "round": (memory_snapshot or {}).get("game", {}).get("round"),
                              "source_phase": phase, "confidence": 0.8,
                              "source_text": self._clip_text(text, 100)})
            if "猎人" in text and ("我是" in text or "跳" in text or "自称" in text):
                edges.append({"claim_type": "猎人", "speaker_id": speaker, "target_id": None,
                              "result": "hunter", "round": (memory_snapshot or {}).get("game", {}).get("round"),
                              "source_phase": phase, "confidence": 0.8, "source_text": self._clip_text(text, 100)})
        # 同一来源同一关系去重，保留第一次的来源文本。
        unique: dict[tuple[Any, ...], dict[str, Any]] = {}
        for edge in edges:
            key = (edge.get("claim_type"), edge.get("speaker_id"), edge.get("target_id"), edge.get("result"))
            unique.setdefault(key, edge)
        return list(unique.values())[-16:]

    def _claims_by_speaker(self, edges: list[Mapping[str, Any]]) -> dict[str, list[str]]:
        claims: dict[str, list[str]] = {}
        for edge in edges:
            speaker = edge.get("speaker_id")
            kind = edge.get("claim_type")
            if speaker is None or not kind:
                continue
            value = str(edge.get("result") or kind)
            label = value if kind == "role_claim" else str(kind)
            claims.setdefault(str(speaker), [])
            if label not in claims[str(speaker)]:
                claims[str(speaker)].append(label)
        return {pid: values[:6] for pid, values in list(claims.items())[:12]}

    def _detect_claims_from_dialogue(self, dialogue: Any, memory_snapshot: Mapping[str, Any]) -> dict[str, list[str]]:
        """兼容旧调用者；新策略使用 claim edges。"""
        return self._claims_by_speaker(self._extract_public_claim_edges(dialogue, memory_snapshot))

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
            corpus = " ".join(
                self._extract_text_snippets(memory_snapshot.get("recent_public_history"))
                + self._extract_text_snippets(self._recent_public_texts[-6:])
            )
            if claim_text:
                score += stage_weights["exposed"]
                reasons.append("公开跳身份")
                layer = max(layer, 4)
                if "猎人" in claim_text:
                    # 未知猎人出局可能反杀狼人，除非残局收口，不把它当无风险刀口。
                    score -= 7 if stage != "end" else 2
                    reasons.append("未知猎人反杀风险")
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
        current_round = self._nonnegative_int(memory_snapshot.get("game", {}).get("round"), fallback=0) or 0
        ledger = wolf_memory.get("night_failure_ledger") or []
        failure = next((item for item in ledger if str(item.get("target")) == target_id), None)
        if failure is None and target_id in {str(item) for item in (wolf_memory.get("failed_kill_targets") or [])}:
            failure = {"failure_count": 1, "last_failed_round": current_round - 1}
        if isinstance(failure, Mapping):
            count = self._nonnegative_int(failure.get("failure_count"), fallback=1) or 1
            last_failed = self._nonnegative_int(failure.get("last_failed_round"), fallback=current_round - 1)
            age = max(0, current_round - last_failed)
            # 失败账本按轮次衰减，但至少让下一夜换层；连续失败两次在非残局直接避开。
            if age <= 1:
                score -= 12 if stage != "end" else 6
                reasons.append(f"失败冷却{count}次")
            elif age <= 3:
                score -= max(2, 8 - age * 2)
                reasons.append("近期失败余波")
            if count >= 2 and stage != "end" and age <= 2:
                score -= 28
                reasons.append("连续失败，默认避开")
                layer = 0
        no_death_rounds = wolf_memory.get("no_death_rounds") or []
        if len(no_death_rounds) >= 2 and stage != "end" and failure is not None:
            score -= 8
            reasons.append("连续平安夜后换威胁层")
        if not reasons:
            reasons.append("默认公共威胁")
        return {
            "score": score,
            "layer": layer,
            "reason": ",".join(reasons[:5]),
            "failure_cooldown": bool(failure) if 'failure' in locals() else False,
            "guard_pattern_risk": len(no_death_rounds) >= 2 if 'no_death_rounds' in locals() else False,
            "known_claimed_power_role": any(role in claim_text for role in ("预言家", "女巫", "守卫", "猎人")) if 'claim_text' in locals() else False,
            "sheriff_vote_value": bool(record.get("is_sheriff") or str(target_id) == str(memory_snapshot.get("game", {}).get("sheriff_id"))) if record else False,
            "endgame_vote_swing": stage == "end" and bool(record),
            "hunter_exposure_risk": bool(record and "猎人" in claim_text),
        }

    def _rank_targets(
        self,
        turn_packet: Mapping[str, Any],
        memory_snapshot: Mapping[str, Any],
        target_actions: list[Mapping[str, Any]],
        *,
        current_dialogue: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        public_state = turn_packet.get("public_state")
        players = self._extract_player_records(public_state, turn_packet.get("game"), memory_snapshot)
        candidates: list[dict[str, Any]] = []
        seen: set[str] = set()
        for action in target_actions:
            for target_id in action.get("target_ids") or []:
                target = str(target_id)
                if target in seen:
                    continue
                seen.add(target)
                record = players.get(target, {})
                profile = self._target_threat_profile(target, record, memory_snapshot)
                edges = self._extract_public_claim_edges(current_dialogue or [], memory_snapshot)
                role_edges = [edge for edge in edges if str(edge.get("speaker_id")) == target and edge.get("claim_type") == "role_claim"]
                if role_edges:
                    profile["score"] += 7
                    profile["reason"] += ",公开神职声明"
                    profile["layer"] = max(profile["layer"], 4)
                if any(edge.get("claim_type") == "守护" and str(edge.get("target_id")) == target for edge in edges):
                    profile["score"] += 4
                    profile["reason"] += ",守护锚点"
                candidates.append(
                    {
                        "player_id": target,
                        "score": profile["score"],
                        "layer": profile["layer"],
                        "reason": profile["reason"],
                        "avoid": "连续失败，默认避开" in str(profile["reason"]),
                        "seat": self._seat_sort_value(record, target),
                    }
                )
        # 共识只加分，不得覆盖冷却、死亡目标或明显的高反杀风险。
        consensus = self._team_consensus_target(current_dialogue or [], target_actions, memory_snapshot)
        if consensus:
            for item in candidates:
                if item["player_id"] == consensus["target"] and not item.get("avoid"):
                    item["score"] += min(8, 2 + consensus["support"] * 2)
                    item["reason"] += f",队友共识{consensus['support']}票"
        safe = [item for item in candidates if not item.get("avoid")]
        if safe and any(item.get("avoid") for item in candidates):
            candidates = safe
        candidates.sort(key=lambda item: (-int(item["score"]), -int(item["layer"]), item["seat"], item["player_id"]))
        primary_layer = candidates[0]["layer"] if candidates else None
        ranked: list[dict[str, Any]] = []
        if candidates:
            ranked.append(
                {
                    "player_id": candidates[0]["player_id"],
                    "layer": candidates[0]["layer"],
                    "reason": candidates[0]["reason"],
                    "avoid": candidates[0].get("avoid", False),
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
                    "avoid": item.get("avoid", False),
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
                    "avoid": item.get("avoid", False),
                }
        if len(candidates) > 1:
            item = candidates[1]
            return {"player_id": item["player_id"], "layer": item["layer"], "reason": item["reason"], "avoid": item.get("avoid", False)}
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
        if not isinstance(votes, list):
            return 0
        target = str(target_id)
        pressure = 0
        for item in votes:
            if not isinstance(item, str):
                continue
            if item.endswith(f"->{target}") or f"->{target}," in item or f"->{target} " in item:
                pressure += 1
        return pressure

    def _legal_candidate_ids(
        self, target_actions: list[Mapping[str, Any]], exclude: set[str] | None = None
    ) -> list[str]:
        result: list[str] = []
        excluded = {str(item) for item in (exclude or set())}
        for action in target_actions:
            for target in action.get("target_ids") or []:
                target = str(target)
                if target not in excluded and target not in result:
                    result.append(target)
        return result

    def _estimate_vote_pressure(
        self, dialogue: Any, turn_packet: Mapping[str, Any], candidate_ids: list[str] | None = None
    ) -> dict[str, dict[str, Any]]:
        """只统计明确的本轮意向；否定句不会被误算为投票。"""
        candidates = set(str(item) for item in (candidate_ids or []))
        sheriff = self._safe_get(turn_packet.get("game"), "sheriff_id")
        if sheriff is None:
            state = turn_packet.get("public_state")
            sheriff = self._safe_get(state, "sheriff_id")
        result: dict[str, dict[str, Any]] = {}
        for speaker, text, _phase in self._claim_items(dialogue):
            if not speaker or not text:
                continue
            for target in re.findall(r"p\d+", text):
                if candidates and target not in candidates:
                    continue
                before = text[max(0, text.find(target) - 4):text.find(target)]
                if any(word in before for word in ("不投", "别投", "不出", "不跟")):
                    continue
                if not any(word in before or word in text[:text.find(target) + 1] for word in ("投", "出", "票", "跟", "优先", "归票")):
                    continue
                weight = 1.5 if str(speaker) == str(sheriff) else 1.0
                entry = result.setdefault(target, {"votes": 0, "weighted_votes": 0.0, "supporters": [], "reasons": []})
                entry["votes"] += 1
                entry["weighted_votes"] += weight
                if str(speaker) not in entry["supporters"]:
                    entry["supporters"].append(str(speaker))
                entry["reasons"].append(self._clip_text(text, 50))
        # 有些同步包已经给出本轮票数；它是公开系统事实，优先作为基础票压。
        records = self._extract_player_records(
            turn_packet.get("public_state"), turn_packet.get("game"), {"game": {}}
        )
        for target in candidates or records.keys():
            record = records.get(str(target), {})
            count = self._nonnegative_int(record.get("vote_count", record.get("votes")), fallback=0) or 0
            if count:
                entry = result.setdefault(str(target), {"votes": 0, "weighted_votes": 0.0, "supporters": [], "reasons": []})
                entry["votes"] = max(int(entry["votes"]), count)
                entry["weighted_votes"] = max(float(entry["weighted_votes"]), float(count))
                entry["reasons"].append("系统票数")
        return result

    def _has_sheriff_support(self, edges: list[Mapping[str, Any]], turn_packet: Mapping[str, Any]) -> bool:
        sheriff = self._safe_get(turn_packet.get("game"), "sheriff_id")
        if sheriff is None:
            sheriff = self._safe_get(turn_packet.get("public_state"), "sheriff_id")
        return sheriff is not None and any(str(edge.get("speaker_id")) == str(sheriff) for edge in edges)

    def _team_consensus_target(
        self, dialogue: Any, target_actions: list[Mapping[str, Any]], memory_snapshot: Mapping[str, Any]
    ) -> dict[str, Any] | None:
        legal = set(self._legal_candidate_ids(target_actions, exclude={self.player_id}))
        wolves = {str(item) for item in (memory_snapshot.get("wolf_memory", {}).get("known_wolf_ids") or [])}
        supports: Counter[str] = Counter()
        opponents: Counter[str] = Counter()
        last_speaker: dict[str, str] = {}
        for speaker, text, _phase in self._claim_items(dialogue):
            if not speaker or str(speaker) == self.player_id:
                continue
            for target in re.findall(r"p\d+", text):
                if target not in legal or target in wolves:
                    continue
                if re.search(rf"(?:刀|杀|击杀|建议|主刀|投)\s*{re.escape(target)}|{re.escape(target)}\s*(?:刀|可刀|该杀)", text):
                    supports[target] += 1
                    last_speaker[target] = str(speaker)
                if re.search(rf"(?:别|不要|不)\s*(?:刀|杀|投)?\s*{re.escape(target)}", text):
                    opponents[target] += 1
        if not supports:
            return None
        target, count = supports.most_common(1)[0]
        if count < 2 and count <= opponents[target]:
            return None
        return {"target": target, "support": count, "opposition": opponents[target], "last_speaker": last_speaker.get(target)}

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
        events = combined.get("events") or combined.get("history") or []
        if combined.get("no_death") is True or self._contains_no_death_signal(events):
            summary["no_death"] = True
        texts = self._extract_text_snippets(sync_packet)
        if texts:
            summary["public_texts"] = texts
        dialogue = combined.get("dialogue") or combined.get("public_dialogue")
        structured_dialogue = self._summarize_dialogue(dialogue)
        if structured_dialogue:
            summary["public_dialogue"] = structured_dialogue
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

    def _public_player_index(
        self, public_state: Any, game: Any, memory_snapshot: Mapping[str, Any]
    ) -> list[dict[str, Any]]:
        records = self._extract_player_records(public_state, game, memory_snapshot)
        result: list[dict[str, Any]] = []
        for pid, record in records.items():
            seat = record.get("seat", record.get("seat_id"))
            alive = self._is_alive_record(record)
            result.append({"id": str(pid), "seat": seat, "alive": alive})
        result.sort(key=lambda item: self._seat_sort_value({"seat": item.get("seat")}, item["id"]))
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
                "players",
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



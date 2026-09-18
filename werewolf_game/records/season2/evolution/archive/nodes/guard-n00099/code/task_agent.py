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
        # Enforce the external Task-Agent budget even if a caller supplies looser values.
        self.max_tokens = min(900, max(1, int(max_tokens)))
        self.max_decision_retries = max(0, int(max_decision_retries))
        self.max_tool_calls_per_decision = min(5, max(0, int(max_tool_calls_per_decision)))
        self.max_tool_result_tokens = min(1000, max(1, int(max_tool_result_tokens)))
        self.max_prompt_chars = min(12000, max(1, int(max_prompt_chars)))
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
        self._guard_signal_round_counts: dict[tuple[str, int], dict[str, int]] = {}
        self._last_observed_round: int | None = None
        self._guard_memory_round: int | None = None

    async def observe(self, sync_packet: dict[str, Any]) -> None:
        """接收公开同步，并提取最小守卫记忆。"""

        self._update_guard_memory(sync_packet)

    async def decide(self, turn_packet: dict[str, Any]) -> dict[str, Any]:
        if turn_packet["self"]["player_id"] != self.player_id:
            raise ValueError("行动包被发送给了错误玩家")
        private = turn_packet["private_information"]
        if str(private.get("role")) != self.profile.role:
            raise ValueError("Task-Agent 的角色档案与行动包身份不一致")

        self._update_guard_memory(turn_packet)
        system = self._system_prompt(private)
        prompt = {
            "game": turn_packet["game"],
            "public_rules": turn_packet["public_rules"],
            "self": turn_packet["self"],
            "private_information": private,
            "public_state": turn_packet["public_state"],
            "request": turn_packet["request"],
            "guard_memory_summary": self._guard_memory_summary(
                turn_packet["request"], turn_packet
            ) if self.profile.role == "guard" else None,
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
            self._update_guard_memory({
                "round": turn_packet["game"].get("round"),
                "current_round_dialogue": current_dialogue,
            })
            if self.profile.role == "guard":
                prompt["guard_memory_summary"] = self._guard_memory_summary(
                    turn_packet["request"], turn_packet
                )

        def execute_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
            del arguments
            if name != CURRENT_ROUND_DIALOGUE_TOOL_NAME:
                return {"error": f"不支持的工具：{name}"}
            return {
                "round": turn_packet["game"].get("round"),
                "phase": turn_packet["game"].get("public_phase", turn_packet["game"].get("phase")),
                "dialogue": self._bounded_tool_dialogue(current_dialogue),
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
        role_task = self.profile.task
        if self.profile.role == "guard":
            role_task = role_task + (
                "\n\n守卫约束：只从当前 target_ids 里选；先看近期公开可信神职、确认好人和警徽链核心；"
                "再看明确的近期狼人威胁，投票压力和热度只能辅助；话多或被点名不等于刀口。"
                "默认不要重复上一夜目标（仅在规则禁止时是硬限制）；没有明显更好的外部目标时可考虑自守；"
                "不要把守护成功当作已确认结果，也不要主动公开守卫身份或具体守护对象。"
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
                "守卫执行要点：只在当前 target_ids 内选；先看近期公开可信神职、确认好人和警徽链核心，"
                "再看明确的近期狼人威胁；投票压力、被怀疑和热度只能辅助。没有明显更好的外部目标时才考虑自守；"
                "不要公开守卫身份、具体守护对象或把平安夜当作守中证明。"
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

    def _bounded_tool_dialogue(self, dialogue: Any) -> list[Any]:
        """Keep the restricted tool response below the per-call length budget."""
        entries = dialogue if isinstance(dialogue, list) else [dialogue]
        kept: list[Any] = []
        for entry in reversed(entries):
            candidate = [entry] + kept
            if len(json.dumps(candidate, ensure_ascii=False)) > self.max_tool_result_tokens:
                break
            kept = candidate
        if not kept and entries:
            entry = entries[-1]
            if isinstance(entry, Mapping):
                compact = {
                    key: entry[key] for key in ("speaker_id", "player_id", "text")
                    if key in entry
                }
                if not compact:
                    compact = {"text": str(entry)}
                kept = [compact]
            else:
                kept = [{"text": str(entry)}]
        # Compact the final entry until the serialized tool result itself fits.
        while len(json.dumps(kept, ensure_ascii=False)) > self.max_tool_result_tokens:
            entry = kept[-1]
            if isinstance(entry, Mapping):
                text_key = next((key for key in ("text", "content", "message")
                                 if isinstance(entry.get(key), str)), None)
                if text_key is None:
                    kept = []
                    break
                compact = dict(entry)
                text = str(compact[text_key])
                compact[text_key] = text[: max(0, len(text) // 2)]
                kept[-1] = compact
            else:
                kept[-1] = str(entry)[: max(0, len(str(entry)) // 2)]
        return kept

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

    def _repair_guard_decision(
        self,
        action: dict[str, Any],
        request: dict[str, Any],
        packet: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
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
        ranked = self._guard_candidate_ranking(allowed, request, packet)
        if not ranked:
            return action
        chosen = action.get("target_id")
        ranked_targets = {item["candidate"] for item in ranked}
        if chosen in ranked_targets:
            # 合法目标代表模型已经完成了策略判断；本地画像绝不覆盖它。
            return action
        repaired = dict(action)
        repaired["target_id"] = ranked[0]["candidate"]
        return repaired

    def _guard_candidate_targets(
        self,
        allowed: Mapping[str, Any],
        request: Mapping[str, Any],
        packet: Mapping[str, Any] | None = None,
    ) -> list[str]:
        ranked = self._guard_candidate_ranking(allowed, request, packet)
        return [item["candidate"] for item in ranked]

    def _guard_candidate_ranking(
        self,
        allowed: Mapping[str, Any],
        request: Mapping[str, Any],
        packet: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
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
        round_no = self._extract_round_number(request)
        if round_no is None and packet is not None:
            round_no = self._extract_round_number(packet)
        if round_no is None:
            round_no = self._last_observed_round
        repeat_forbidden = self._guard_repeat_forbidden(allowed)
        ranked: list[dict[str, Any]] = []
        for index, candidate in enumerate(candidates):
            profile = self._guard_candidate_profile(candidate, request, packet, round_no)
            # self is a preference, never a protocol prohibition.  Repeat is handled
            # below only when the engine explicitly forbids it.
            if candidate == self.player_id:
                profile["wolf_attack_risk"] -= 3
                profile["information_value"] -= 2
            profile["score_key"] = (
                profile["protection_tier"],
                profile["wolf_attack_risk"],
                profile["trusted_claim"] + profile["confirmed_good"],
                profile["public_pressure"],
                profile["recency"],
                -index,
            )
            profile["total_score"] = (
                profile["protection_tier"] * 1000
                + profile["wolf_attack_risk"] * 10
                + profile["trusted_claim"] * 3
                + profile["confirmed_good"] * 3
                + profile["public_pressure"]
            )
            profile["candidate"] = candidate
            ranked.append(profile)
        ranked.sort(key=lambda item: item["score_key"], reverse=True)
        if repeat_forbidden and self.last_protected_id:
            filtered = [item for item in ranked if item["candidate"] != self.last_protected_id]
            if filtered:
                ranked = filtered
        for rank, item in enumerate(ranked):
            item["rank"] = rank
        return ranked

    def _guard_repeat_forbidden(self, allowed: Mapping[str, Any]) -> bool:
        if allowed.get("guard_can_repeat_protect") is False:
            return True
        # Missing metadata is not permission to rewrite a legal model choice.
        return False

    def _guard_candidate_profile(
        self,
        candidate: str,
        request: Mapping[str, Any],
        packet: Mapping[str, Any] | None,
        round_no: int | None,
    ) -> dict[str, int]:
        bucket = self.player_risk_profile.get(candidate) or {}
        alive_count = self._current_alive_count(request, packet)
        stage_bonus = self._guard_stage_bonus(alive_count)
        if round_no is None:
            round_no = self._last_observed_round
        last_round = bucket.get("last_round")
        recency = 0
        if isinstance(last_round, int) and last_round >= 0 and round_no is not None:
            gap = max(0, round_no - last_round)
            recency = max(0, 10 - gap * 4)
        recent_threat = int(bucket.get("recent_threat", bucket.get("threat", 0)))
        trusted_claim = int(bucket.get("trusted_claim", bucket.get("claim", 0)))
        confirmed_good = int(bucket.get("confirmed_good", 0))
        sheriff_or_badge = int(bucket.get("sheriff_or_badge", 0))
        pressure = int(bucket.get("pressure", 0))
        vote = int(bucket.get("vote", 0))
        exposure = int(bucket.get("exposure", 0))
        public_focus = 1 if candidate in self.recent_public_focus_ids else 0
        # 分层而非等权累计：公开核心和明确近期刀口是主信号，热度仅作微弱 tie-breaker。
        protection_tier = (
            sheriff_or_badge * 3 + confirmed_good * 2 + trusted_claim
        )
        wolf_attack_risk = recent_threat * 8 + protection_tier * 4
        information_value = trusted_claim * 3 + confirmed_good * 4 + sheriff_or_badge * 4
        public_pressure = min(8, pressure + vote + exposure + public_focus)
        wolf_attack_risk += min(3, pressure + vote)  # 辅助，不能制造主导风险
        if alive_count is not None and alive_count <= 5:
            wolf_attack_risk += 1
            information_value += 2
        elif alive_count is not None and alive_count <= 7:
            information_value += 1
        return {
            "wolf_attack_risk": wolf_attack_risk,
            "information_value": information_value,
            "public_pressure": public_pressure,
            "stage_bonus": stage_bonus,
            "recency": recency,
            "alive_count": alive_count or 0,
            "protection_tier": protection_tier,
            "recent_threat": recent_threat,
            "trusted_claim": trusted_claim,
            "confirmed_good": confirmed_good,
            "sheriff_or_badge": sheriff_or_badge,
            "pressure": pressure,
            "vote": vote,
            "exposure": exposure,
        }

    def _guard_stage_bonus(self, alive_count: int | None) -> int:
        if alive_count is None:
            return 0
        if alive_count <= 5:
            return 6
        if alive_count <= 7:
            return 3
        return 0

    def _current_alive_count(self, request: Mapping[str, Any], packet: Mapping[str, Any] | None) -> int | None:
        alive_ids: set[str] = set()
        public_state = request.get("public_state") if isinstance(request, Mapping) else None
        if isinstance(public_state, Mapping):
            alive_ids.update(
                self._extract_ids_by_keys(
                    public_state,
                    ("alive", "alive_player", "alive_players", "alive_player_ids", "player_ids", "players"),
                )
            )
        if not alive_ids and isinstance(packet, Mapping):
            public_state = packet.get("public_state")
            if isinstance(public_state, Mapping):
                alive_ids.update(
                    self._extract_ids_by_keys(
                        public_state,
                        ("alive", "alive_player", "alive_players", "alive_player_ids", "player_ids", "players"),
                    )
                )
        if not alive_ids:
            alive_ids.update(self.alive_player_cache)
        return len(alive_ids) if alive_ids else None

    def _looks_publicly_salient(
        self,
        candidate: str,
        request: Mapping[str, Any],
        packet: Mapping[str, Any] | None = None,
    ) -> bool:
        public_state = request.get("public_state") if isinstance(request, Mapping) else None
        if candidate in self._extract_focus_ids(public_state):
            return True
        if isinstance(packet, Mapping):
            if candidate in self._extract_focus_ids(packet.get("public_state")):
                return True
        return False

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
        if round_no is not None:
            stale_keys = [key for key in self._guard_signal_round_counts if key[1] < round_no - 3]
            for key in stale_keys:
                self._guard_signal_round_counts.pop(key, None)

    def _update_guard_risk_memory(self, packet: Mapping[str, Any], round_no: int | None) -> None:
        if round_no is not None and round_no != self._guard_memory_round:
            self._decay_guard_memory(round_no)
            self._guard_memory_round = round_no
        # Only known public structures are interpreted.  Unknown nested records are
        # deliberately ignored: a field named ``target`` is not evidence of a wolf attack.
        state = packet.get("public_state") if isinstance(packet.get("public_state"), Mapping) else packet
        if isinstance(state, Mapping):
            self._ingest_structured_guard_state(state, round_no)
            for key in ("current_round_dialogue", "dialogue"):
                if key in state:
                    self._ingest_dialogue_entries(state[key], round_no)
        if "current_round_dialogue" in packet:
            self._ingest_dialogue_entries(packet["current_round_dialogue"], round_no)

    def _decay_guard_memory(self, round_no: int) -> None:
        for bucket in self.player_risk_profile.values():
            for name in ("recent_threat", "threat", "pressure", "vote", "exposure", "claim"):
                if name in bucket:
                    # One-round memory remains useful, but old public pressure rapidly fades.
                    bucket[name] = int(bucket[name] * 0.35)
            if bucket.get("last_round", -1) < round_no - 2:
                for name in ("recent_threat", "threat", "pressure", "vote", "exposure", "claim"):
                    bucket[name] = 0

    def _ingest_structured_guard_state(self, state: Mapping[str, Any], round_no: int | None) -> None:
        aliases: dict[str, tuple[str, ...]] = {
            "sheriff_or_badge": ("sheriff_id", "sheriff", "sheriff_chain", "badge_holder_id", "badge_chain", "警长id", "警徽持有者"),
            "confirmed_good": ("confirmed_good_ids", "confirmed_good", "verified_good_ids", "known_good_ids"),
            "recent_threat": ("recent_threat_ids", "wolf_attack_candidates", "night_threat_ids", "explicit_threat_ids"),
            "pressure": ("current_pressure_ids", "suspect_ids", "recent_suspects", "pressure_ids"),
            "vote": ("current_vote_targets", "vote_targets", "recent_vote_targets"),
            "exposure": ("public_focus_ids", "key_player_ids"),
            "trusted_claim": ("trusted_claim_ids", "public_role_claim_ids", "claimed_good_role_ids", "public_seer_ids", "seer_claim_ids"),
        }
        for signal_name, keys in aliases.items():
            for key in keys:
                value = state.get(key)
                if value is None:
                    continue
                for player_id in self._extract_player_ids(value):
                    self._bump_player_risk(player_id, signal_name, 1, round_no)
        # Some kernels expose role claims as records.  A claim is still only a public
        # signal, but applying it to the claimant is less ambiguous than to mentioned IDs.
        claims = state.get("claims", state.get("public_claims", state.get("role_claims")))
        if isinstance(claims, list):
            for claim in claims:
                if not isinstance(claim, Mapping):
                    continue
                claimant = self._first_id(claim.get("player_id", claim.get("speaker_id", claim.get("claimant"))))
                role = str(claim.get("role", claim.get("claimed_role", ""))).lower()
                if claimant and role in {"seer", "witch", "guard", "hunter", "预言家", "女巫", "守卫", "猎人"}:
                    self._bump_player_risk(claimant, "trusted_claim", 1, round_no)
        elif isinstance(claims, Mapping):
            for claimant, role in claims.items():
                claimant_id = self._first_id(claimant)
                if claimant_id and str(role).lower() in {"seer", "witch", "guard", "hunter", "预言家", "女巫", "守卫", "猎人"}:
                    self._bump_player_risk(claimant_id, "trusted_claim", 1, round_no)

    def _scan_guard_signal_tree(self, node: Mapping[str, Any] | list[Any] | Any, round_no: int | None) -> None:
        """Compatibility shim: parse only a public packet's known top-level shape."""
        if isinstance(node, Mapping):
            self._update_guard_risk_memory(node, round_no)

    def _signal_for_key(self, key_text: str) -> tuple[str, int] | None:
        # Kept for callers of the old helper, but exact names prevent arbitrary
        # recursive key matching from manufacturing signals.
        return {
            "recent_threat_ids": ("recent_threat", 1),
            "wolf_attack_candidates": ("recent_threat", 1),
            "sheriff_id": ("sheriff_or_badge", 1),
            "badge_holder_id": ("sheriff_or_badge", 1),
            "confirmed_good_ids": ("confirmed_good", 1),
            "suspect_ids": ("pressure", 1),
            "vote_targets": ("vote", 1),
        }.get(key_text)

    def _is_guard_signal_container_key(self, key_text: str) -> bool:
        return key_text in {"public_state", "current_round_dialogue", "dialogue"}

    def _ingest_dialogue_entries(self, dialogue: Any, round_no: int | None) -> None:
        entries = dialogue if isinstance(dialogue, list) else [dialogue]
        for entry in entries:
            speaker_id = self._dialogue_entry_speaker_id(entry)
            text = self._dialogue_entry_text(entry)
            targets = self._dialogue_entry_targets(entry)
            if text:
                targets.update(self._extract_player_ids(text))
            if speaker_id:
                targets.discard(speaker_id)
            signals = self._dialogue_text_signals(text)
            if speaker_id and signals.get("claim"):
                self._bump_player_risk(speaker_id, "trusted_claim", 1, round_no)
            # A threat applies to the explicitly named/mentioned target, never to the
            # speaker.  Pressure and votes remain weak auxiliary evidence.
            for player_id in targets:
                if signals.get("threat"):
                    self._bump_player_risk(player_id, "recent_threat", 1, round_no)
                if signals.get("vote"):
                    self._bump_player_risk(player_id, "vote", 1, round_no)
                if signals.get("pressure"):
                    self._bump_player_risk(player_id, "pressure", 1, round_no)
                if signals.get("exposure"):
                    self._bump_player_risk(player_id, "exposure", 1, round_no)

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
        if (
            any(token in lowered for token in ("claim", "报身份", "自报", "自称", "跳预", "跳女巫", "跳守卫", "跳猎人", "跳神"))
            or any(token in lowered for token in ("我是预言家", "我是女巫", "我是守卫", "我是猎人", "我是白痴"))
        ):
            signals["claim"] = 1
        if any(token in lowered for token in ("投票", "归票", "vote", "票型", "点名")):
            signals["vote"] = 4
        if any(token in lowered for token in ("夜刀", "刀口", "今晚刀", "击杀", "attack", "kill")):
            signals["threat"] = 4
        if any(token in lowered for token in ("怀疑", "可疑", "狼", "站边", "压票", "focus", "关注", "push", "指认")):
            signals["pressure"] = max(signals.get("pressure", 0), 2)
        if any(token in lowered for token in ("公开", "关键", "核心", "热度", "中心", "public")):
            signals["exposure"] = max(signals.get("exposure", 0), 1)
        return signals

    def _player_risk_bucket(self, player_id: str) -> dict[str, int]:
        bucket = self.player_risk_profile.get(player_id)
        if bucket is None:
            bucket = {
                "pressure": 0,
                "vote": 0,
                "recent_threat": 0,
                "trusted_claim": 0,
                "confirmed_good": 0,
                "sheriff_or_badge": 0,
                "exposure": 0,
                "last_round": -1,
            }
            self.player_risk_profile[player_id] = bucket
        return bucket

    def _bump_player_risk(self, player_id: str, signal_name: str, amount: int, round_no: int | None) -> None:
        if not player_id:
            return
        if round_no is not None:
            key = (player_id, round_no)
            seen = self._guard_signal_round_counts.setdefault(key, {})
            cap = self._guard_signal_round_cap(signal_name)
            if seen.get(signal_name, 0) >= cap:
                return
            seen[signal_name] = seen.get(signal_name, 0) + 1
        bucket = self._player_risk_bucket(player_id)
        canonical = {"threat": "recent_threat", "claim": "trusted_claim"}.get(signal_name, signal_name)
        increment = max(1, min(3, int(amount)))
        if canonical in {"trusted_claim", "confirmed_good", "sheriff_or_badge"}:
            bucket[canonical] = min(2, bucket.get(canonical, 0) + increment)
        else:
            bucket[canonical] = bucket.get(canonical, 0) + increment
        if round_no is not None:
            bucket["last_round"] = max(bucket.get("last_round", -1), round_no)

    def _first_id(self, value: Any) -> str | None:
        ids = self._extract_player_ids(value)
        return next(iter(ids), None)

    def _guard_signal_round_cap(self, signal_name: str) -> int:
        if signal_name == "threat":
            return 2
        if signal_name in {"claim", "vote"}:
            return 2
        return 1

    def _player_risk_score(self, candidate: str, request: Mapping[str, Any], round_no: int | None) -> int:
        profile = self._guard_candidate_profile(candidate, request, None, round_no)
        return profile["wolf_attack_risk"]

    def _public_heat_score(self, candidate: str, request: Mapping[str, Any]) -> int:
        profile = self._guard_candidate_profile(candidate, request, None, self._last_observed_round)
        return profile["public_pressure"]

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
        """Return only explicitly structured public focus, not arbitrary mentioned IDs."""
        if not isinstance(packet, Mapping):
            return set()
        ids: set[str] = set()
        for key in (
            "sheriff_id", "badge_holder_id", "警长id", "警徽持有者",
            "public_focus_ids", "key_player_ids", "confirmed_good_ids",
            "recent_threat_ids", "wolf_attack_candidates",
        ):
            ids.update(self._extract_player_ids(packet.get(key)))
        return ids

    def _guard_memory_summary(
        self, request: Mapping[str, Any], packet: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        allowed = next(
            (item for item in request.get("allowed_actions", [])
             if isinstance(item, Mapping) and item.get("kind") == _GUARD_ACTION_KIND),
            None,
        )
        if not isinstance(allowed, Mapping):
            return {"note": "仅公开信号；无 guard_protect 候选。"}
        ranked = self._guard_candidate_ranking(allowed, request, packet)
        candidates = []
        for item in ranked:
            candidates.append({
                "candidate": item["candidate"],
                "recent_threat": item["recent_threat"],
                "trusted_claim": item["trusted_claim"],
                "confirmed_good": item["confirmed_good"],
                "sheriff_or_badge": item["sheriff_or_badge"],
                "recent_pressure": item["pressure"],
                "recent_vote": item["vote"],
                "heat_auxiliary": item["public_pressure"],
            })
        return {
            "note": "仅为最近公开信号的分层摘要，不是身份事实；守护成功也不构成确认。",
            "priority": "可信公开核心/确认好人、近期明确威胁 > 普通压力/投票/热度",
            "candidates": candidates,
            "last_protected": self.last_protected_id,
        }

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

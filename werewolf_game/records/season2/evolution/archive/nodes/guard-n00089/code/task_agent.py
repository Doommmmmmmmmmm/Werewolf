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
# 不用 Unicode \b：中文与 p5 相邻时也必须识别玩家编号，但不能识别更长英文标识的一部分。
_PLAYER_ID_PATTERN = re.compile(r"(?<![A-Za-z0-9_])p\d+(?![A-Za-z0-9_])", re.IGNORECASE)
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
_GUARD_THREAT_COMPONENTS = (
    "claim_pressure",
    "sheriff_linkage",
    "vote_concentration",
    "kill_mentions",
    "public_salience",
    "recency",
)
_GUARD_THREAT_WEIGHTS = {
    "claim_pressure": 6,
    "sheriff_linkage": 7,
    "vote_concentration": 5,
    "kill_mentions": 8,
    "public_salience": 1,
    "recency": 2,
}
_GUARD_THREAT_DECAY = {
    "claim_pressure": 1,
    "sheriff_linkage": 1,
    "vote_concentration": 1,
    "kill_mentions": 1,
    "public_salience": 2,
    "recency": 1,
}


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
        self.recent_public_focus_ids: deque[tuple[int | None, str]] = deque(maxlen=8)
        # 守卫威胁画像：只来自当前玩家依法可见的公开状态/当前轮对话。
        # 记录的是可见威胁信号的结构化累积，不是身份事实。
        # 画像只保存可审计的公开事件；旧的 component 字段保留为兼容视图，
        # 但排序使用 hard/soft/uncertainty 三层分数，不把所有提及混成一个总分。
        self.player_risk_profile: dict[str, dict[str, Any]] = {}
        self._guard_events: list[dict[str, Any]] = []
        self._guard_event_keys: set[str] = set()
        self._last_observed_round: int | None = None
        self._last_observed_phase: str | None = None

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
        if self.profile.role == "guard":
            prompt["guard_memory"] = self._guard_memory_summary(turn_packet["request"])

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
                "\n\n守卫约束：只从当前 target_ids 里选；遵守行动包给出的连守限制；"
                "核心目标是保护今晚最可能被狼刀的好人。先看可解释的 hard 信号："
                "可信 claim/警徽链/查验/明确刀口，soft 热度只能辅助；被提及不等于被刀。"
                "claim 可能悍跳；若规则允许且 self 在 target_ids 中，自守也是正常候选。"
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
                "守卫执行要点：只选当前 target_ids，并遵守行动包的连守限制；"
                "优先守今晚最可能被狼刀的好人。先核对 hard 信号的目标关系和来源："
                "可信身份/警徽链/查验/明确刀口优先，单纯话多或被提及只能作 soft 参考。"
                "公开 claim 可能悍跳；规则允许自守时不要自行排除 self。"
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
        """只修复协议错误；不要用不可靠画像接管模型的合法选择。"""
        if self.profile.role != "guard" or action.get("kind") != _GUARD_ACTION_KIND:
            return action
        allowed = next(
            (item for item in request.get("allowed_actions", [])
             if isinstance(item, Mapping) and item.get("kind") == _GUARD_ACTION_KIND),
            None,
        )
        if not isinstance(allowed, Mapping):
            return action
        rankings = self._guard_candidate_rankings(allowed, request)
        if not rankings:
            return action
        chosen_id = str(action.get("target_id") or "")
        candidate_ids = [item["player_id"] for item in rankings]
        if chosen_id not in candidate_ids:
            repaired = dict(action)
            repaired["target_id"] = candidate_ids[0]
            return repaired

        # 合法选择默认原样保留。只有“明确的结构化高压目标”且模型完全没有
        # 该层信号时才作安全网式纠正；普通热度永远不能触发覆盖。
        best = rankings[0]
        chosen = next(item for item in rankings if item["player_id"] == chosen_id)
        if (
            best["player_id"] != chosen_id
            and best["hard_score"] >= 12
            and chosen["hard_score"] == 0
            and best["hard_score"] >= max(12, (chosen["soft_score"] + 1) * 3)
        ):
            repaired = dict(action)
            repaired["target_id"] = best["player_id"]
            return repaired
        return action

    def _guard_candidate_rankings(self, allowed: Mapping[str, Any], request: Mapping[str, Any]) -> list[dict[str, Any]]:
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
        current_focus_ids = self._extract_focus_ids(request.get("public_state"))
        recent_focus_ids = self._recent_public_focus_ids(round_no)
        rankings: list[dict[str, Any]] = []
        for index, candidate in enumerate(candidates):
            profile = dict(self.player_risk_profile.get(candidate) or self._empty_guard_profile())
            hard_score = self._guard_hard_score(profile)
            soft_score = self._guard_soft_score(candidate, profile, current_focus_ids, recent_focus_ids)
            uncertainty_score = int(profile.get("uncertainty_score", 0))
            if self.last_protected_id and candidate == self.last_protected_id and repeat_forbidden:
                # 这是规则约束，不是威胁判断；保留候选以便合法性修复有明确依据。
                hard_score = -10_000
            rankings.append({
                "player_id": candidate,
                "hard_score": hard_score,
                "soft_score": soft_score,
                "uncertainty_score": uncertainty_score,
                "core_score": hard_score,
                "public_score": soft_score,
                "total_score": hard_score + soft_score,
                "signals": self._guard_signal_summary(profile),
                "index": index,
                "profile": profile,
            })
        rankings.sort(key=lambda item: (
            -item["hard_score"], -item["soft_score"], item["uncertainty_score"], item["index"]
        ))
        if repeat_forbidden and self.last_protected_id:
            filtered = [item for item in rankings if item["player_id"] != self.last_protected_id]
            if filtered:
                rankings = filtered
        return rankings

    def _guard_candidate_targets(self, allowed: Mapping[str, Any], request: Mapping[str, Any]) -> list[str]:
        return [item["player_id"] for item in self._guard_candidate_rankings(allowed, request)]

    def _guard_memory_summary(self, request: Mapping[str, Any]) -> dict[str, Any]:
        allowed = next(
            (
                item
                for item in request.get("allowed_actions", [])
                if isinstance(item, Mapping) and item.get("kind") == _GUARD_ACTION_KIND
            ),
            {},
        )
        rankings = self._guard_candidate_rankings(allowed, request) if isinstance(allowed, Mapping) else []
        top_threats = [
            {
                "player_id": item["player_id"],
                "hard_score": item["hard_score"],
                "soft_score": item["soft_score"],
                "uncertainty_score": item["uncertainty_score"],
                "signals": item["signals"],
                "last_round": item["profile"].get("last_round", -1),
                "repeat_last_night": item["player_id"] == self.last_protected_id,
            }
            for item in rankings[:4]
            if item["hard_score"] > 0 or item["soft_score"] > 0
        ]
        return {
            "meaning": "hard 是有明确关系的公开事件，soft 只是热度；二者都不是身份确认。被提及不等于会被刀。",
            "decision_brief": "先审查存活、合法性和连守；再看可信身份/查验/明确刀口。普通热度不能单独压过 hard，也不要自动相信悍跳 claim。",
            "self_protect": "若 self.player_id 在 target_ids 中且规则未明确禁止，自守是正常合法候选。",
            "last_protected_id": self.last_protected_id,
            "last_protected_round": self.last_protected_round,
            "recommended_order": [item["player_id"] for item in rankings[:5]],
            "threat_brief": top_threats,
        }

    def _guard_repeat_forbidden(self, allowed: Mapping[str, Any]) -> bool:
        if allowed.get("guard_can_repeat_protect") is True:
            return False
        if allowed.get("guard_can_repeat_protect") is False:
            return True
        return True

    def _looks_publicly_salient(self, candidate: str, request: Mapping[str, Any]) -> bool:
        current_focus_ids = self._extract_focus_ids(request.get("public_state"))
        if candidate in current_focus_ids:
            return True
        round_no = self._extract_round_number(request) or self._last_observed_round
        return candidate in self._recent_public_focus_ids(round_no)

    def _recent_public_focus_ids(self, round_no: int | None) -> set[str]:
        if round_no is None:
            return {player_id for _, player_id in self.recent_public_focus_ids}
        recent: set[str] = set()
        for seen_round, player_id in self.recent_public_focus_ids:
            if seen_round is None or round_no - seen_round <= 2:
                recent.add(player_id)
        return recent

    def _update_guard_memory(self, packet: Mapping[str, Any] | Any) -> None:
        if not isinstance(packet, Mapping):
            return
        round_no = self._extract_round_number(packet)
        phase = str(packet.get("phase") or packet.get("public_phase") or "")
        if round_no is not None and (self._last_observed_round is None or round_no > self._last_observed_round):
            self._decay_guard_threat_profile(round_no)
        if round_no is not None:
            self._last_observed_round = round_no
        if phase:
            self._last_observed_phase = phase

        # alive 是快照而非累加器；死亡字段只使用行动包中依法公开的字段。
        alive_ids = self._extract_ids_by_keys(packet, (
            "alive", "alive_player", "alive_players", "alive_player_ids"
        ))
        dead_ids = self._extract_ids_by_keys(packet, (
            "dead", "dead_player", "dead_players", "dead_player_ids", "eliminated"
        ))
        if alive_ids:
            self.alive_player_cache = set(alive_ids)
        self._forget_guard_players(dead_ids)
        focus_ids = self._extract_focus_ids(packet)
        if focus_ids:
            focus_round = round_no if round_no is not None else self._last_observed_round
            for focus_id in sorted(focus_ids):
                self.recent_public_focus_ids.append((focus_round, focus_id))
        self._update_guard_threat_profile(packet, focus_ids, round_no, phase)

    def _update_guard_threat_profile(
        self, packet: Mapping[str, Any], focus_ids: set[str], round_no: int | None,
        phase: str = "",
    ) -> None:
        # 只把显式 focus 当作 soft 热度，不把一个状态对象里的所有编号都升级成 hard。
        for focus_id in focus_ids:
            self._record_guard_event(round_no, phase, "普通讨论热度", None, focus_id, 1, 0.35)
        for speaker_id, text in self._iter_visible_speeches(packet):
            self._scan_guard_text_signal(speaker_id, text, round_no, phase)
        self._extract_structured_guard_events(packet, round_no, phase)
        for player_id, profile in list(self.player_risk_profile.items()):
            if not any(profile.get(component, 0) > 0 for component in _GUARD_THREAT_COMPONENTS):
                self.player_risk_profile.pop(player_id, None)
            else:
                for component in _GUARD_THREAT_COMPONENTS:
                    profile[component] = min(80, int(profile.get(component, 0)))

    def _decay_guard_threat_profile(self, current_round: int) -> None:
        for player_id, profile in list(self.player_risk_profile.items()):
            last_round = int(profile.get("last_round", -1))
            if last_round < 0:
                continue
            steps = current_round - last_round
            if steps <= 0:
                continue
            for component, decay in _GUARD_THREAT_DECAY.items():
                profile[component] = max(0, int(profile.get(component, 0)) - steps * decay)
            profile["hard_score"] = max(0, int(profile.get("hard_score", 0)) - steps)
            profile["soft_score"] = max(0, int(profile.get("soft_score", 0)) - steps * 2)
            profile["uncertainty_score"] = max(0, int(profile.get("uncertainty_score", 0)) - steps)
            profile["last_round"] = current_round
            if not any(int(profile.get(component, 0)) > 0 for component in _GUARD_THREAT_COMPONENTS):
                self.player_risk_profile.pop(player_id, None)

    def _forget_guard_players(self, player_ids: set[str]) -> None:
        if not player_ids:
            return
        self.alive_player_cache.difference_update(player_ids)
        self.recent_public_focus_ids = deque(
            ((round_no, player_id) for round_no, player_id in self.recent_public_focus_ids if player_id not in player_ids),
            maxlen=8,
        )
        for player_id in player_ids:
            self.player_risk_profile.pop(player_id, None)
        self._guard_events = [event for event in self._guard_events if event.get("target_id") not in player_ids]
        self._guard_event_keys = {str(event.get("event_key")) for event in self._guard_events}

    def _empty_guard_profile(self) -> dict[str, Any]:
        profile: dict[str, Any] = {component: 0 for component in _GUARD_THREAT_COMPONENTS}
        profile.update({"hard_score": 0, "soft_score": 0, "uncertainty_score": 0, "last_round": -1, "signal_kinds": []})
        return profile

    def _guard_profile_for(self, player_id: str) -> dict[str, Any]:
        profile = self.player_risk_profile.get(player_id)
        if not isinstance(profile, dict):
            profile = self._empty_guard_profile()
            self.player_risk_profile[player_id] = profile
        for component in _GUARD_THREAT_COMPONENTS:
            profile.setdefault(component, 0)
        for key, default in (("hard_score", 0), ("soft_score", 0), ("uncertainty_score", 0), ("last_round", -1), ("signal_kinds", [])):
            profile.setdefault(key, default.copy() if isinstance(default, list) else default)
        return profile

    def _bump_guard_threat(self, player_id: str | None, component: str, amount: int, round_no: int | None = None) -> None:
        # 兼容内部/旧测试调用；新的公开输入统一走 _record_guard_event。
        if not player_id or not _PLAYER_ID_PATTERN.fullmatch(player_id) or component not in _GUARD_THREAT_COMPONENTS or amount <= 0:
            return
        profile = self._guard_profile_for(player_id)
        profile[component] = min(80, int(profile.get(component, 0)) + int(amount))
        profile["recency"] = min(20, int(profile.get("recency", 0)) + 1)
        if component == "public_salience":
            profile["soft_score"] = min(100, int(profile.get("soft_score", 0)) + amount)
        else:
            profile["hard_score"] = min(100, int(profile.get("hard_score", 0)) + amount)
        if round_no is not None:
            profile["last_round"] = max(int(profile.get("last_round", -1)), round_no)

    def _record_guard_event(self, source_round: int | None, source_phase: str, source_kind: str,
                            speaker_id: str | None, target_id: str | None, strength: int,
                            confidence: float, *, event_key: str | None = None,
                            protective: bool = True) -> None:
        if not target_id or not _PLAYER_ID_PATTERN.fullmatch(str(target_id)) or strength <= 0:
            return
        target_id = str(target_id)
        key = event_key or "|".join((str(source_round), source_phase, source_kind, str(speaker_id), target_id, str(strength)))
        if key in self._guard_event_keys:
            return
        self._guard_event_keys.add(key)
        event = {"source_round": source_round, "source_kind": source_kind, "speaker_id": speaker_id,
                 "target_id": target_id, "strength": int(strength), "confidence": float(confidence),
                 "event_key": key}
        self._guard_events.append(event)
        profile = self._guard_profile_for(target_id)
        component = {
            "明确身份声明": "claim_pressure", "警长/警徽持有": "sheriff_linkage",
            "公开查验结果": "claim_pressure", "明确票型压力": "vote_concentration",
            "明确狼队刀口表述": "kill_mentions", "普通讨论热度": "public_salience",
        }.get(source_kind)
        if component:
            if protective:
                profile[component] = min(80, int(profile.get(component, 0)) + strength)
                if component == "public_salience":
                    profile["soft_score"] = min(100, int(profile.get("soft_score", 0)) + strength)
                else:
                    profile["hard_score"] = min(100, int(profile.get("hard_score", 0)) + strength)
            else:
                # “查杀/是狼”是公开事实，但它降低而不是提高守护价值。
                profile["uncertainty_score"] = min(50, int(profile.get("uncertainty_score", 0)) + strength)
            if source_kind == "明确身份声明":
                profile["uncertainty_score"] = min(50, int(profile.get("uncertainty_score", 0)) + 2)
            profile["signal_kinds"] = list(dict.fromkeys([*profile.get("signal_kinds", []), source_kind]))[:4]
            profile["last_round"] = source_round if source_round is not None else profile.get("last_round", -1)
            profile["recency"] = min(20, int(profile.get("recency", 0)) + 1)

    def _guard_hard_score(self, profile: Mapping[str, Any]) -> int:
        # 单独悍跳 claim 不足以成为“可信好人”硬信号；明确的狼向查验
        # 也应降低保护价值。uncertainty 只作扣分，不让 soft 反向补回 hard。
        raw = int(profile.get("hard_score", 0))
        uncertainty = int(profile.get("uncertainty_score", 0))
        return max(0, raw - uncertainty * 2)

    def _guard_soft_score(self, candidate: str, profile: Mapping[str, Any], current_focus_ids: set[str], recent_focus_ids: set[str]) -> int:
        score = int(profile.get("soft_score", 0))
        if candidate in current_focus_ids:
            score += 2
        if candidate in recent_focus_ids:
            score += 1
        return score

    def _guard_core_threat_score(self, profile: Mapping[str, Any]) -> int:
        return self._guard_hard_score(profile)

    def _guard_public_tie_score(self, candidate: str, profile: Mapping[str, Any], current_focus_ids: set[str], recent_focus_ids: set[str]) -> int:
        return self._guard_soft_score(candidate, profile, current_focus_ids, recent_focus_ids)

    def _guard_signal_summary(self, profile: Mapping[str, Any]) -> str:
        kinds = profile.get("signal_kinds")
        if isinstance(kinds, list) and kinds:
            return "+".join(str(kind) for kind in kinds[:4])
        return "none"

    def _guard_low_information(self, rankings: list[dict[str, Any]]) -> bool:
        return not rankings or int(rankings[0].get("hard_score", 0)) < 1

    def _scan_guard_text_signal(self, speaker_id: str | None, text: str, round_no: int | None = None, phase: str = "") -> None:
        """仅记录能确定 target 关系的句式；孤立动作词不再污染 speaker/所有编号。"""
        if not text:
            return
        clean = " ".join(str(text).split())[:500]
        ids = list(dict.fromkeys(_PLAYER_ID_PATTERN.findall(clean)))
        key_base = hashlib.sha256(clean.encode("utf-8")).hexdigest()[:16]
        for target in ids:
            # 明确的攻击建议/刀口威胁：只提升 target，绝不提升说话者。
            if re.search(r"(?:建议|提议|今晚|明晚|狼队|我们|他们|先)\s*(?:明确(?:说|表示)?\s*)?(?:先\s*)?(?:刀|砍|攻击|杀|下手)\s*(?:一下)?\s*" + re.escape(target), clean):
                self._record_guard_event(round_no, phase, "明确狼队刀口表述", speaker_id, target, 8, .70,
                                         event_key=f"{round_no}|{phase}|{speaker_id}|{key_base}|kill|{target}")
                continue
            # 查验关系必须是 target 紧邻结果词；“查验”本身不是查验事实。
            result = re.search(re.escape(target) + r"\s*(是狼|查杀|金水|银水|好人|查验结果)", clean)
            if result:
                wolf_result = result.group(1) in {"是狼", "查杀"}
                self._record_guard_event(round_no, phase, "公开查验结果", speaker_id, target, 7, .75,
                                         protective=not wolf_result,
                                         event_key=f"{round_no}|{phase}|{speaker_id}|{key_base}|check|{target}")
            # 保护/关注是弱讨论信号，不当作刀口事实。
            if re.search(r"(?:保护|守一下|守住|关注|重点看|盯着)\s*" + re.escape(target), clean):
                self._record_guard_event(round_no, phase, "普通讨论热度", speaker_id, target, 1, .35,
                                         event_key=f"{round_no}|{phase}|{speaker_id}|{key_base}|soft|{target}")

        # 身份声明和警长持有只归能确认的主体；不把同句出现的其他编号当成 claim。
        if speaker_id and re.search(r"(?:我是|我就是|跳|自称)\s*(?:预言家|女巫|猎人|守卫|白痴|神职)", clean):
            self._record_guard_event(round_no, phase, "明确身份声明", speaker_id, speaker_id, 4, .55,
                                     event_key=f"{round_no}|{phase}|{speaker_id}|{key_base}|claim")
        for target in ids:
            if re.search(re.escape(target) + r"\s*(?:是|当选|拿到)?\s*(?:警长|警徽)", clean):
                self._record_guard_event(round_no, phase, "警长/警徽持有", speaker_id, target, 5, .70,
                                         event_key=f"{round_no}|{phase}|{speaker_id}|{key_base}|sheriff|{target}")
        if speaker_id and re.search(r"(?:我|本人)(?:拿|有|是|竞选|上警)?\s*(?:警长|警徽|sheriff|badge)", clean, re.I):
            self._record_guard_event(round_no, phase, "警长/警徽持有", speaker_id, speaker_id, 4, .50,
                                     event_key=f"{round_no}|{phase}|{speaker_id}|{key_base}|sheriff")

    def _iter_visible_speeches(self, value: Any) -> list[tuple[str | None, str]]:
        speeches: list[tuple[str | None, str]] = []
        if isinstance(value, str):
            if _PLAYER_ID_PATTERN.search(value):
                speeches.append((None, value[:500]))
            return speeches
        if isinstance(value, Mapping):
            speaker = self._first_player_id(
                value.get("speaker_id") or value.get("player_id") or value.get("speaker") or value.get("actor_id")
            )
            text_parts: list[str] = []
            for key in ("text", "content", "message", "speech", "utterance", "claim", "reason"):
                item = value.get(key)
                if isinstance(item, str):
                    text_parts.append(item)
            if text_parts:
                speeches.append((speaker, " ".join(text_parts)[:500]))
            for item in value.values():
                if isinstance(item, (Mapping, list)):
                    speeches.extend(self._iter_visible_speeches(item))
            return speeches
        if isinstance(value, list):
            for item in value:
                speeches.extend(self._iter_visible_speeches(item))
        return speeches

    def _first_player_id(self, value: Any) -> str | None:
        ids = self._extract_player_ids(value)
        return next(iter(ids), None)

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
        explicit_focus_keys = (
            "vote", "votes", "voter", "voters", "target", "target_id", "target_ids",
            "claim", "claims", "claimed", "sheriff", "badge", "警长", "警徽", "警徽流",
        )
        for key, value in packet.items():
            key_text = str(key).lower()
            if not any(keyword in key_text for keyword in explicit_focus_keys):
                continue
            focus_ids.update(self._extract_player_ids(value))
            if isinstance(value, Mapping):
                focus_ids.update(self._extract_focus_ids(value))
            elif isinstance(value, list):
                for item in value:
                    focus_ids.update(self._extract_player_ids(item))
                    if isinstance(item, Mapping):
                        focus_ids.update(self._extract_focus_ids(item))
        return focus_ids

    def _extract_structured_guard_events(self, packet: Mapping[str, Any], round_no: int | None, phase: str) -> None:
        """读取行动包中有字段关系的公开事实；不递归扫描任意 public_state 文本。"""
        for key in ("vote", "votes", "vote_results", "voting"):
            value = packet.get(key)
            if isinstance(value, Mapping):
                for voter, target in value.items():
                    target_id = self._first_player_id(target)
                    if target_id:
                        self._record_guard_event(round_no, phase, "明确票型压力", self._first_player_id(voter), target_id, 3, .75,
                                                 event_key=f"{round_no}|{phase}|vote|{voter}|{target_id}")
            elif isinstance(value, list):
                for item in value:
                    if not isinstance(item, Mapping):
                        continue
                    voter = self._first_player_id(item.get("voter") or item.get("voter_id") or item.get("from"))
                    target = self._first_player_id(item.get("target") or item.get("target_id") or item.get("vote"))
                    if target:
                        self._record_guard_event(round_no, phase, "明确票型压力", voter, target, 3, .75,
                                                 event_key=f"{round_no}|{phase}|vote|{voter}|{target}")

    def _extract_ids_by_keys(self, packet: Mapping[str, Any], keys: tuple[str, ...]) -> set[str]:
        values: set[str] = set()
        if not isinstance(packet, Mapping):
            return values
        for key, value in packet.items():
            key_text = str(key).lower()
            if not any(token in key_text for token in keys):
                continue
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

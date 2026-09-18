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
                "\n\n守卫约束：只从当前 target_ids 里选；默认不要重复上一夜目标；"
                "核心目标是保护今晚最可能被狼刀的好人。要按威胁信号排序："
                "claim / 警徽链 / 票型 / 明确刀口提及优先，公开热度只作同分参考，"
                "不要把被讨论多、话多直接当成高刀口风险。"
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
                "守卫执行要点：只选当前 target_ids；尽量不要重复上一夜守护对象；"
                "优先守今晚最可能被狼刀的好人。排序依据是明确威胁信号："
                "跳身份/警徽链/票型压力/刀口提及优先，单纯话多或被频繁提及只作同分参考。"
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
        rankings = self._guard_candidate_rankings(allowed, request)
        if not rankings:
            return action
        chosen_id = str(action.get("target_id") or "")
        candidate_ids = [item["player_id"] for item in rankings]
        if chosen_id not in candidate_ids:
            repaired = dict(action)
            repaired["target_id"] = candidate_ids[0]
            return repaired

        chosen_index = candidate_ids.index(chosen_id)
        best_item = rankings[0]
        chosen_item = rankings[chosen_index]
        low_info = self._guard_low_information(rankings)

        if chosen_id == self.player_id and len(candidate_ids) > 1:
            repaired = dict(action)
            repaired["target_id"] = best_item["player_id"]
            return repaired

        if chosen_index >= 2 and low_info:
            repaired = dict(action)
            repaired["target_id"] = best_item["player_id"]
            return repaired

        if chosen_index > 0 and best_item["core_score"] >= chosen_item["core_score"] + 5:
            repaired = dict(action)
            repaired["target_id"] = best_item["player_id"]
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
            profile = self.player_risk_profile.get(candidate)
            if not isinstance(profile, Mapping):
                profile = self._empty_guard_profile()
            else:
                profile = dict(profile)
            core_score = self._guard_core_threat_score(profile)
            public_score = self._guard_public_tie_score(candidate, profile, current_focus_ids, recent_focus_ids)
            penalty = 0
            if candidate == self.player_id:
                penalty += 10_000
            if self.last_protected_id and candidate == self.last_protected_id:
                if repeat_forbidden:
                    penalty += 5_000
                elif round_no is not None and self.last_protected_round is not None:
                    age = max(0, round_no - self.last_protected_round)
                    penalty += max(0, 4 - age)
            adjusted_core = max(0, core_score - penalty)
            rankings.append(
                {
                    "player_id": candidate,
                    "core_score": adjusted_core,
                    "public_score": public_score,
                    "total_score": adjusted_core + public_score,
                    "signals": self._guard_signal_summary(profile),
                    "index": index,
                    "profile": profile,
                }
            )
        rankings.sort(
            key=lambda item: (
                -item["core_score"],
                -item["public_score"],
                -item["profile"].get("claim_pressure", 0),
                -item["profile"].get("sheriff_linkage", 0),
                -item["profile"].get("vote_concentration", 0),
                -item["profile"].get("kill_mentions", 0),
                item["index"],
            )
        )
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
                "core_score": item["core_score"],
                "public_score": item["public_score"],
                "signals": item["signals"],
            }
            for item in rankings[:4]
            if item["core_score"] > 0 or item["public_score"] > 0
        ]
        return {
            "meaning": "威胁分数表示公开可见信号累积出的今晚刀口风险，不是身份确认；公开热度只作同分参考。",
            "decision_brief": "优先看 claim / 警徽链 / 票型 / 刀口提及；公开热度只作同分参考。",
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
        if round_no is not None and (self._last_observed_round is None or round_no > self._last_observed_round):
            self._decay_guard_threat_profile(round_no)
        if round_no is not None:
            self._last_observed_round = round_no
        alive_ids = self._extract_ids_by_keys(
            packet,
            ("alive", "alive_player", "alive_players", "alive_player_ids", "player_ids", "players"),
        )
        if alive_ids:
            self.alive_player_cache.update(alive_ids)
        focus_ids = self._extract_focus_ids(packet)
        if focus_ids:
            focus_round = round_no if round_no is not None else self._last_observed_round
            for focus_id in focus_ids:
                self.recent_public_focus_ids.append((focus_round, focus_id))
        self._update_guard_threat_profile(packet, focus_ids, round_no)

    def _update_guard_threat_profile(self, packet: Mapping[str, Any], focus_ids: set[str], round_no: int | None) -> None:
        for focus_id in focus_ids:
            self._bump_guard_threat(focus_id, "public_salience", 1, round_no)
        for speaker_id, text in self._iter_visible_speeches(packet):
            self._scan_guard_text_signal(speaker_id, text, round_no)
        for player_id, profile in list(self.player_risk_profile.items()):
            if not any(profile.get(component, 0) > 0 for component in _GUARD_THREAT_COMPONENTS):
                self.player_risk_profile.pop(player_id, None)
            else:
                for component in _GUARD_THREAT_COMPONENTS:
                    if profile.get(component, 0) > 80:
                        profile[component] = 80

    def _decay_guard_threat_profile(self, current_round: int) -> None:
        for player_id, profile in list(self.player_risk_profile.items()):
            last_round = profile.get("last_round", -1)
            if last_round < 0:
                continue
            steps = current_round - last_round
            if steps <= 0:
                continue
            for component, decay in _GUARD_THREAT_DECAY.items():
                profile[component] = max(0, profile.get(component, 0) - steps * decay)
            profile["last_round"] = current_round
            if not any(profile.get(component, 0) > 0 for component in _GUARD_THREAT_COMPONENTS):
                self.player_risk_profile.pop(player_id, None)

    def _empty_guard_profile(self) -> dict[str, int]:
        profile = {component: 0 for component in _GUARD_THREAT_COMPONENTS}
        profile["last_round"] = -1
        return profile

    def _guard_profile_for(self, player_id: str) -> dict[str, int]:
        profile = self.player_risk_profile.get(player_id)
        if not isinstance(profile, dict):
            profile = self._empty_guard_profile()
            self.player_risk_profile[player_id] = profile
            return profile
        for component in _GUARD_THREAT_COMPONENTS:
            profile.setdefault(component, 0)
        profile.setdefault("last_round", -1)
        return profile

    def _bump_guard_threat(self, player_id: str | None, component: str, amount: int, round_no: int | None = None) -> None:
        if not player_id or not _PLAYER_ID_PATTERN.fullmatch(player_id):
            return
        if component not in _GUARD_THREAT_COMPONENTS:
            return
        if amount <= 0:
            return
        if player_id == self.player_id:
            amount = min(amount, 1)
        profile = self._guard_profile_for(player_id)
        profile[component] = min(80, profile.get(component, 0) + int(amount))
        profile["recency"] = min(20, profile.get("recency", 0) + 1)
        if round_no is not None:
            profile["last_round"] = max(profile.get("last_round", -1), round_no)

    def _guard_core_threat_score(self, profile: Mapping[str, Any]) -> int:
        return (
            int(profile.get("claim_pressure", 0)) * _GUARD_THREAT_WEIGHTS["claim_pressure"]
            + int(profile.get("sheriff_linkage", 0)) * _GUARD_THREAT_WEIGHTS["sheriff_linkage"]
            + int(profile.get("vote_concentration", 0)) * _GUARD_THREAT_WEIGHTS["vote_concentration"]
            + int(profile.get("kill_mentions", 0)) * _GUARD_THREAT_WEIGHTS["kill_mentions"]
            + int(profile.get("recency", 0)) * _GUARD_THREAT_WEIGHTS["recency"]
        )

    def _guard_public_tie_score(
        self,
        candidate: str,
        profile: Mapping[str, Any],
        current_focus_ids: set[str],
        recent_focus_ids: set[str],
    ) -> int:
        score = int(profile.get("public_salience", 0)) * _GUARD_THREAT_WEIGHTS["public_salience"]
        if candidate in current_focus_ids:
            score += 3
        if candidate in recent_focus_ids:
            score += 1
        return score

    def _guard_signal_summary(self, profile: Mapping[str, Any]) -> str:
        labels = []
        if int(profile.get("claim_pressure", 0)) > 0:
            labels.append("claim")
        if int(profile.get("sheriff_linkage", 0)) > 0:
            labels.append("sheriff")
        if int(profile.get("vote_concentration", 0)) > 0:
            labels.append("vote")
        if int(profile.get("kill_mentions", 0)) > 0:
            labels.append("kill")
        if int(profile.get("public_salience", 0)) > 0:
            labels.append("public")
        if int(profile.get("recency", 0)) > 0 and not labels:
            labels.append("fresh")
        return "+".join(labels) if labels else "none"

    def _guard_low_information(self, rankings: list[dict[str, Any]]) -> bool:
        if not rankings:
            return True
        top_one = int(rankings[0]["core_score"])
        top_two = top_one + int(rankings[1]["core_score"]) if len(rankings) > 1 else top_one
        return top_one < 10 or top_two < 18

    def _scan_guard_text_signal(self, speaker_id: str | None, text: str, round_no: int | None = None) -> None:
        if not text:
            return
        ids = list(dict.fromkeys(_PLAYER_ID_PATTERN.findall(text)))
        if not ids and speaker_id is None:
            return
        claim_patterns = (
            "我是预言家",
            "我预言家",
            "跳预言家",
            "我是女巫",
            "跳女巫",
            "我是猎人",
            "跳猎人",
            "我是守卫",
            "跳守卫",
            "神职",
            "拍身份",
            "报身份",
            "银水",
            "金水",
            "查验",
            "验了",
        )
        sheriff_patterns = (
            "警长",
            "警徽",
            "警徽流",
            "sheriff",
            "badge",
            "站边",
            "上警",
            "接票",
        )
        vote_patterns = (
            "查杀",
            "是狼",
            "狼坑",
            "票",
            "投",
            "归",
            "出",
            "抗推",
            "压票",
            "点名",
            "怀疑",
            "焦点",
        )
        kill_patterns = (
            "刀",
            "砍",
            "夜里",
            "晚上",
            "今晚",
            "明晚",
            "先刀",
            "刀口",
            "吃刀",
            "狼刀",
            "威胁",
            "留不住",
            "保护",
            "守一下",
        )

        if speaker_id and any(pattern in text for pattern in claim_patterns):
            self._bump_guard_threat(speaker_id, "claim_pressure", 8, round_no)
        if speaker_id and any(pattern in text for pattern in sheriff_patterns):
            self._bump_guard_threat(speaker_id, "sheriff_linkage", 4, round_no)
        if speaker_id and any(pattern in text for pattern in vote_patterns):
            self._bump_guard_threat(speaker_id, "vote_concentration", 3, round_no)
        if speaker_id and any(pattern in text for pattern in kill_patterns):
            self._bump_guard_threat(speaker_id, "kill_mentions", 2, round_no)

        for target_id in ids:
            if target_id == speaker_id:
                continue
            if any(pattern in text for pattern in claim_patterns):
                self._bump_guard_threat(target_id, "claim_pressure", 3, round_no)
            if any(pattern in text for pattern in sheriff_patterns):
                self._bump_guard_threat(target_id, "sheriff_linkage", 4, round_no)
            if any(pattern in text for pattern in vote_patterns):
                self._bump_guard_threat(target_id, "vote_concentration", 4, round_no)
            if any(pattern in text for pattern in kill_patterns):
                self._bump_guard_threat(target_id, "kill_mentions", 5, round_no)

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
            "alive",
            "alive_player",
            "alive_players",
            "alive_player_ids",
            "player_id",
            "player_ids",
            "players",
            "vote",
            "votes",
            "voter",
            "voters",
            "target",
            "target_id",
            "target_ids",
            "claim",
            "claims",
            "claimed",
            "sheriff",
            "badge",
            "警长",
            "警徽",
            "警徽流",
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

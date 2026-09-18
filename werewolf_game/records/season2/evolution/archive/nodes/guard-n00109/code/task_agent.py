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
        self.recent_public_focus_ids: deque[str] = deque(maxlen=8)
        self.player_risk_profile: dict[str, dict[str, int]] = {}
        self._guard_signal_round_counts: dict[tuple[str, int], dict[str, int]] = {}
        # These are deliberately small, public-only relations.  They are not a
        # second game state and never contain the agent's private role knowledge.
        self.claim_ledger: dict[str, list[dict[str, Any]]] = {}
        self.counterclaim_groups: dict[str, set[str]] = {}
        self.sheriff_history: list[dict[str, Any]] = []
        self.publicly_revealed_wolves: set[str] = set()
        self.public_confirmed_good_candidates: set[str] = set()
        self.explicit_attack_signals: dict[str, int] = {}
        self.public_conflicts: dict[str, set[str]] = {}
        self._guard_event_fingerprints: set[str] = set()
        self._last_observed_round: int | None = None
        self._last_decay_round: int | None = None

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
                "\n\n守卫约束：只从当前 target_ids 里选；禁止连守时先排除上一夜目标；"
                "只把明确夜刀/狼刀/击杀/今晚刀视为刀口；普通target、投票、查验和话多只是同分参考；"
                "身份声称不等于可信好人；已公开暴露或残局且无更高可信外部目标时可以自守；"
                "不要把守护成功或平安夜当作已确认结果。"
                "决策顺序固定为：合法且非重复；明确公开狼刀；可信且无冲突的公开好人/神职链；"
                "最后才参考单一身份声称、警徽、投票、被点名和热度。对跳或已知狼支持链必须降权；"
                "只有没有更可信外部目标、且本人公开暴露或进入残局时才自守。"
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
                "守卫执行要点：只在当前 target_ids 内选，禁止连守时排除上一夜目标；"
                "顺序是明确公开狼刀，其次可信且无冲突的公开好人/神职链，最后才是一般辅助信息；"
                "普通target、投票、查验、被提及或话多不等于刀口，身份声称本身不等于可信；"
                "本人已公开暴露或5人及以下残局且无更高可信外部目标时可以自守。"
                "模型若选择高冲突/狼支持链目标，只有在存在分数明显更高的无冲突候选时才会被有限纠偏；"
                "分差小则保留模型选择。"
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
        if chosen not in ranked_targets:
            repaired = dict(action)
            repaired["target_id"] = ranked[0]["candidate"]
            return repaired
        # This is intentionally a narrow strategic gate, not a general model
        # replacement.  A real, explicit attack signal always wins.  Otherwise
        # replace only a structurally conflicted choice when the public-only
        # ranking has a clearly better, non-conflicted alternative.
        chosen_item = next(item for item in ranked if item["candidate"] == chosen)
        best = ranked[0]
        if chosen_item is not best and self._guard_high_conflict(chosen_item):
            alternatives = [item for item in ranked if not self._guard_high_conflict(item)]
            if alternatives:
                better = alternatives[0]
                if (
                    not chosen_item.get("explicit_threat")
                    and better["total_score"] - chosen_item["total_score"] >= 25
                ):
                    repaired = dict(action)
                    repaired["target_id"] = better["candidate"]
                    return repaired
        return action

    def _guard_high_conflict(self, profile: Mapping[str, Any]) -> bool:
        return bool(
            int(profile.get("conflict_penalty", 0)) >= 8
            and (int(profile.get("counterclaim", 0)) or int(profile.get("wolf_supported", 0)))
        )

    def _guard_memory_summary(
        self, request: Mapping[str, Any], packet: Mapping[str, Any] | None = None
    ) -> str:
        allowed = next(
            (item for item in request.get("allowed_actions", [])
             if isinstance(item, Mapping) and item.get("kind") == _GUARD_ACTION_KIND),
            None,
        )
        if not isinstance(allowed, Mapping):
            return ""
        ranked = self._guard_candidate_ranking(allowed, request, packet)
        lines = ["仅依据公开状态/本轮发言的守卫摘要（只含当前可选目标）："]
        for item in ranked[:12]:
            candidate = item["candidate"]
            reasons: list[str] = []
            if self._guard_repeat_forbidden(allowed) and candidate == self.last_protected_id:
                reasons.append("repeat_forbidden")
            if item["explicit_threat"]:
                reasons.append("explicit_threat")
            if item["trusted_public_value"]:
                reasons.append("trusted_public_fact")
            records = self.claim_ledger.get(candidate, [])
            claim_status = "none"
            if records:
                claim_status = "counterclaim" if item["counterclaim"] else "single_claim"
            if item["wolf_supported"]:
                reasons.append("known_wolf_support_chain")
            if item["conflict_penalty"] and not item["wolf_supported"]:
                reasons.append("public_conflict_or_pressure")
            lines.append(
                f"{candidate}: stage={item['stage_bonus']}; threat={item['explicit_threat']}; "
                f"claim={claim_status}; conflict={item['conflict_penalty']}; "
                f"reasons={','.join(reasons) or 'ordinary_public_context'}"
            )
        return "\n".join(lines)[:3500]

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
            if repeat_forbidden and self.last_protected_id and candidate == self.last_protected_id:
                profile["wolf_attack_risk"] -= 6_000
                profile["information_value"] -= 3_000
                profile["public_pressure"] -= 1_000
            # Lexicographic layers are deliberate: ordinary heat and claims
            # must never outrank an explicit public attack signal or a clean
            # public confirmation merely because they occur more often.
            profile["score_key"] = (
                profile["wolf_attack_risk"],
                profile["trusted_public_value"],
                -profile["conflict_penalty"],
                profile["information_value"],
                profile["stage_bonus"],
                profile["public_pressure"],
                profile["recency"],
                -index,
            )
            profile["total_score"] = (
                profile["wolf_attack_risk"] * 100
                + profile["trusted_public_value"] * 30
                - profile["conflict_penalty"] * 35
                + profile["information_value"] * 6
                + profile["stage_bonus"] * 4
                + profile["public_pressure"] * 2
                + profile["recency"]
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
        if allowed.get("guard_can_repeat_protect") is True:
            return False
        if allowed.get("guard_can_repeat_protect") is False:
            return True
        return True

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
            recency = max(0, 10 - gap * 2)
        threat = int(bucket.get("threat", 0))
        claim = int(bucket.get("claim", 0))
        vote = int(bucket.get("vote", 0))
        pressure = int(bucket.get("pressure", 0))
        exposure = int(bucket.get("exposure", 0))
        public_focus = 1 if candidate in self.recent_public_focus_ids else 0
        salient = 1 if self._looks_publicly_salient(candidate, request, packet) else 0
        claim_records = self.claim_ledger.get(candidate, [])
        counterclaim = bool(self.public_conflicts.get(candidate)) or any(
            record.get("counterclaim") for record in claim_records
        )
        wolf_supported = (
            "known_wolf_support" in self.public_conflicts.get(candidate, set())
            or any(record.get("wolf_supported") for record in claim_records)
        )
        explicit_threat = min(3, int(self.explicit_attack_signals.get(candidate, threat)))
        # Only explicit night-attack facts enter this layer.  Pressure, votes,
        # mentions, and claims are intentionally not proxies for a wolf target.
        wolf_attack_risk = explicit_threat * 100
        conflict_penalty = (8 if counterclaim else 0) + (12 if wolf_supported else 0)
        if claim and (vote or pressure):
            conflict_penalty += min(claim, 4) + min(vote + pressure, 5)
        if pressure >= 6 and not explicit_threat:
            conflict_penalty += 2
        trusted_public_value = 0
        if candidate in self.public_confirmed_good_candidates:
            trusted_public_value += 10
        if claim and not counterclaim and not wolf_supported:
            # An unopposed claim is only a small tie-breaker, never a claim of
            # alignment or a night-attack risk.
            trusted_public_value += 1
        if any(record.get("target") == candidate for record in self.sheriff_history) and not counterclaim and not wolf_supported:
            trusted_public_value += 1
        information_value = min(exposure, 3) + public_focus + salient
        public_pressure = min(pressure, 5) * 2 + min(vote, 5) + public_focus + salient
        suspicion_penalty = conflict_penalty
        if explicit_threat:
            information_value += min(explicit_threat, 3)
        if alive_count is not None:
            if alive_count <= 5:
                information_value += 3
                public_pressure += 1
            elif alive_count <= 7:
                information_value += 1
        if candidate == self.player_id:
            self_exposure_bonus = 0
            if claim or exposure:
                self_exposure_bonus += min(claim + exposure, 8)
            if vote or pressure:
                self_exposure_bonus += min(vote + pressure, 6)
            if alive_count is not None and alive_count <= 5:
                self_exposure_bonus += 5
            elif alive_count is not None and alive_count <= 7 and (claim or exposure or vote or pressure):
                self_exposure_bonus += 2
            # Self-protection is a late-game/public-exposure option, not an
            # inferred wolf attack signal.  Keep it below explicit attacks.
            trusted_public_value += min(self_exposure_bonus, 3)
            information_value += min(self_exposure_bonus, 4)
        if candidate in self.alive_player_cache:
            information_value += 1
        if recency:
            information_value += recency // 3
        return {
            "wolf_attack_risk": wolf_attack_risk,
            "information_value": information_value,
            "public_pressure": public_pressure,
            "suspicion_penalty": suspicion_penalty,
            "conflict_penalty": conflict_penalty,
            "trusted_public_value": trusted_public_value,
            "explicit_threat": explicit_threat,
            "counterclaim": int(counterclaim),
            "wolf_supported": int(wolf_supported),
            "stage_bonus": stage_bonus,
            "recency": recency,
            "alive_count": alive_count or 0,
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
            alive_ids.update(self._extract_alive_ids(public_state))
        if not alive_ids and isinstance(packet, Mapping):
            public_state = packet.get("public_state")
            if isinstance(public_state, Mapping):
                alive_ids.update(self._extract_alive_ids(public_state))
            if not alive_ids:
                alive_ids.update(self._extract_alive_ids(packet))
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
        if round_no is not None:
            self._decay_guard_risk_memory(round_no)
        alive_ids = self._extract_alive_ids(packet)
        if alive_ids:
            # 明确 alive 字段代表当前存活集合，应替换缓存而非只增不减。
            self.alive_player_cache = set(alive_ids)
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
        self._scan_guard_signal_tree(packet, round_no)

    def _scan_guard_signal_tree(self, node: Mapping[str, Any] | list[Any] | Any, round_no: int | None) -> None:
        if isinstance(node, Mapping):
            self._ingest_public_facts(node, round_no)
            for key, value in node.items():
                key_text = str(key).lower()
                if key_text in {"current_round_dialogue", "dialogue", "dialogues", "chat", "speech", "speeches", "transcript"}:
                    self._ingest_dialogue_entries(value, round_no)
                    continue
                if key_text in {"sheriff_history", "sheriff_chain", "badge_history", "badge_chain", "警徽链"}:
                    items = value if isinstance(value, list) else [value]
                    for item in items:
                        self._ingest_sheriff_record(item, round_no)
                    continue
                signal = self._signal_for_key(key_text)
                if signal is not None:
                    signal_name, signal_weight = signal
                    if signal_name == "claim":
                        # A claim is an assertion by an actor.  A target nested
                        # in the assertion is a result/mention, not the claimant.
                        self._ingest_structured_claim(value, key_text, round_no)
                    else:
                        for player_id in self._extract_player_ids(value):
                            self._bump_player_risk(player_id, signal_name, signal_weight, round_no)
                            if signal_name == "threat":
                                self.explicit_attack_signals[player_id] = min(
                                    3, self.explicit_attack_signals.get(player_id, 0) + 1
                                )
                    continue
                if self._is_guard_signal_container_key(key_text) or isinstance(value, (Mapping, list)):
                    self._scan_guard_signal_tree(value, round_no)
            return
        if isinstance(node, list):
            for item in node:
                if isinstance(item, (Mapping, list)):
                    self._scan_guard_signal_tree(item, round_no)

    def _ingest_public_facts(self, node: Mapping[str, Any], round_no: int | None) -> None:
        """Read only explicit, public facts; never infer a role from behaviour."""
        for key, value in node.items():
            key_text = str(key).lower()
            ids = self._extract_player_ids(value)
            if any(token in key_text for token in ("confirmed_good", "confirmed_human", "good_result", "public_good", "好人确认", "金水")):
                self.public_confirmed_good_candidates.update(ids)
            if any(token in key_text for token in ("revealed_role", "public_role", "role_reveal", "公开身份", "身份揭示")):
                if self._contains_wolf_role(value):
                    for player_id in ids:
                        self.publicly_revealed_wolves.add(player_id)
                        self._mark_wolf_support(player_id)
                elif self._contains_public_good_role(value):
                    self.public_confirmed_good_candidates.update(ids)
            if any(token in key_text for token in ("sheriff_transfer", "badge_transfer", "警徽转移", "传警徽")):
                self._ingest_sheriff_record(value, round_no)
            if any(token in key_text for token in ("wolf_target", "wolf_kill", "night_kill", "attack_target", "attack_targets", "夜刀", "狼刀", "今晚刀")):
                for player_id in ids:
                    self.explicit_attack_signals[player_id] = min(3, self.explicit_attack_signals.get(player_id, 0) + 1)

    def _contains_public_good_role(self, value: Any) -> bool:
        if isinstance(value, str):
            text = value.lower()
            return any(token in text for token in ("villager", "seer", "witch", "guard", "hunter", "idiot", "好人", "村民", "预言家", "女巫", "守卫", "猎人", "白痴"))
        if isinstance(value, Mapping):
            return any(self._contains_public_good_role(item) for item in value.values())
        if isinstance(value, list):
            return any(self._contains_public_good_role(item) for item in value)
        return False

    def _contains_wolf_role(self, value: Any) -> bool:
        if isinstance(value, str):
            text = value.lower()
            return any(token in text for token in ("werewolf", "wolf", "狼人", "狼人牌"))
        if isinstance(value, Mapping):
            return any(self._contains_wolf_role(item) for item in value.values())
        if isinstance(value, list):
            return any(self._contains_wolf_role(item) for item in value)
        return False

    def _mark_wolf_support(self, wolf_id: str) -> None:
        for record in self.sheriff_history:
            if record.get("source") == wolf_id and record.get("target"):
                target = str(record["target"])
                for claim in self.claim_ledger.get(target, []):
                    claim["wolf_supported"] = True
                self.public_conflicts.setdefault(target, set()).add("known_wolf_support")

    def _ingest_sheriff_record(self, value: Any, round_no: int | None) -> None:
        if not isinstance(value, Mapping):
            return
        source_ids = self._extract_player_ids(value.get("from") or value.get("source") or value.get("old_holder"))
        target_ids = self._extract_player_ids(value.get("to") or value.get("target") or value.get("new_holder") or value.get("holder"))
        if not target_ids:
            ids = list(self._extract_player_ids(value))
            if len(ids) >= 2:
                source_ids, target_ids = {ids[0]}, {ids[-1]}
        for target in target_ids:
            record = {"source": next(iter(source_ids), None), "target": target, "round": round_no}
            fingerprint = self._fingerprint("sheriff", record)
            if fingerprint in self._guard_event_fingerprints:
                continue
            self._guard_event_fingerprints.add(fingerprint)
            self.sheriff_history.append(record)
            if record["source"] in self.publicly_revealed_wolves:
                self._mark_wolf_support(str(record["source"]))

    def _ingest_structured_claim(self, value: Any, key_text: str, round_no: int | None) -> None:
        if isinstance(value, list):
            for item in value:
                self._ingest_structured_claim(item, key_text, round_no)
            return
        if not isinstance(value, Mapping):
            return
        speaker_ids = self._extract_player_ids(
            value.get("speaker") or value.get("speaker_id") or value.get("player_id")
            or value.get("claimant") or value.get("from")
        )
        claimed_role = str(value.get("role") or value.get("claimed_role") or value.get("claim_type") or key_text)
        targets = self._extract_player_ids(
            value.get("target") or value.get("target_id") or value.get("mentioned_result") or value.get("result")
        )
        # Also accept the compact public form {"p2": "seer"}; this still
        # records only an explicit assertion and never grants alignment.
        if not speaker_ids:
            for possible_speaker, possible_role in value.items():
                ids = self._extract_player_ids(possible_speaker)
                if ids and isinstance(possible_role, str):
                    for speaker in ids:
                        self._record_claim(speaker, possible_role, round_no, set(), None)
            return
        for speaker in speaker_ids:
            self._record_claim(speaker, claimed_role, round_no, targets, value.get("result"))

    def _record_claim(self, speaker: str, role: str, round_no: int | None, targets: set[str] | None = None, result: Any = None) -> None:
        record = {"speaker": speaker, "claimed_role": role, "round": round_no, "mentioned_result": sorted(targets or set()), "result": str(result)[:40] if result is not None else "", "counterclaim": False, "wolf_supported": False}
        fingerprint = self._fingerprint("claim", record)
        if fingerprint in self._guard_event_fingerprints:
            return
        self._guard_event_fingerprints.add(fingerprint)
        self.claim_ledger.setdefault(speaker, []).append(record)
        self._bump_player_risk(speaker, "claim", 1, round_no)
        role_key = role.lower()
        group = self.counterclaim_groups.setdefault(role_key, set())
        group.add(speaker)
        if len(group) > 1:
            for claimant in group:
                self.public_conflicts.setdefault(claimant, set()).add("counterclaim")
                for old_record in self.claim_ledger.get(claimant, []):
                    if str(old_record.get("claimed_role", "")).lower() == role_key:
                        old_record["counterclaim"] = True

    def _fingerprint(self, kind: str, value: Mapping[str, Any]) -> str:
        return hashlib.sha1((kind + json.dumps(dict(value), ensure_ascii=False, sort_keys=True, default=str)).encode("utf-8")).hexdigest()

    def _signal_for_key(self, key_text: str) -> tuple[str, int] | None:
        # Match field names, not arbitrary substrings in a container such as
        # public_state/history.  This prevents a whole nested object from being
        # attributed to every player id found inside it.
        normalized = key_text.strip().replace("-", "_")
        vote_keys = {"vote", "voted", "vote_target", "vote_targets", "投票", "票型", "归票", "票数"}
        inspection_keys = {"inspect", "inspection", "check", "查验", "验人", "验出", "inspection_result"}
        claim_keys = {"claim", "claimed", "identity_claim", "claimant", "身份", "报身份", "自报", "跳预", "跳女巫", "跳守卫", "跳猎人", "跳神", "神职"}
        threat_keys = {
            "wolf_target", "wolf_targets", "wolf_kill",
            "night_kill", "nightkill", "attack_target", "attack_targets", "wolf_attack",
            "werewolf_attack", "夜刀", "刀口", "狼刀", "击杀", "今晚刀", "夜里要刀",
        }
        pressure_keys = {"suspect", "suspected", "怀疑", "pressure", "point", "点名", "push", "压票", "站边", "focus", "关注", "指认"}
        exposure_keys = {"public", "公开", "center", "核心", "关键", "hot", "热度"}
        if normalized in vote_keys:
            return ("vote", 3)
        if normalized in inspection_keys:
            return ("exposure", 1)
        if normalized in claim_keys:
            return ("claim", 3)
        if normalized in threat_keys:
            return ("threat", 4)
        if normalized in pressure_keys:
            return ("pressure", 2)
        if normalized in exposure_keys:
            return ("exposure", 2)
        return None

    def _is_guard_signal_container_key(self, key_text: str) -> bool:
        return key_text in {
            "players",
            "player_list",
            "events",
            "history",
            "rounds",
            "records",
            "logs",
            "entries",
            "dialogue_list",
            "dialogue_entries",
            "dialogues",
            "chat",
            "speech",
            "speeches",
            "transcript",
        }

    def _ingest_dialogue_entries(self, dialogue: Any, round_no: int | None) -> None:
        entries = dialogue if isinstance(dialogue, list) else [dialogue]
        for entry in entries:
            speaker_id = self._dialogue_entry_speaker_id(entry)
            text = self._dialogue_entry_text(entry)
            explicit_targets = self._dialogue_entry_targets(entry)
            mentioned_ids = set(explicit_targets)
            if text:
                mentioned_ids.update(self._extract_player_ids(text))
            if speaker_id:
                mentioned_ids.discard(speaker_id)
            signals = self._dialogue_text_signals(text)
            if not signals and explicit_targets:
                signals = {"pressure": 1}
            if not signals:
                continue
            if speaker_id and signals.get("claim"):
                role = self._claimed_role_from_text(text)
                result_targets = self._result_targets_from_text(text, mentioned_ids)
                self._record_claim(speaker_id, role, round_no, result_targets, text)
            if speaker_id and signals.get("threat"):
                # A speaker discussing a kill is pressure on the speaker, not
                # proof that every mentioned player is a night target.
                self._bump_player_risk(speaker_id, "pressure", 1, round_no)
            for player_id in mentioned_ids:
                # Claims belong exclusively to speaker_id.  Other signals are
                # kept separate and remain low-weight public context.
                for signal_name, signal_weight in signals.items():
                    if signal_name == "claim":
                        continue
                    self._bump_player_risk(player_id, signal_name, signal_weight, round_no)
                    if signal_name == "threat" and self._text_is_explicit_attack(text):
                        self.explicit_attack_signals[player_id] = min(3, self.explicit_attack_signals.get(player_id, 0) + 1)

    def _claimed_role_from_text(self, text: str) -> str:
        lowered = text.lower()
        for role, tokens in (("seer", ("预言家", "seer", "跳预")), ("witch", ("女巫", "witch")), ("guard", ("守卫", "guard")), ("hunter", ("猎人", "hunter"))):
            if any(token in lowered for token in tokens):
                return role
        return "unknown_claim"

    def _result_targets_from_text(self, text: str, mentioned: set[str]) -> set[str]:
        if not mentioned:
            return set()
        lowered = text.lower()
        result_words = ("金水", "银水", "查杀", "验出", "好人", "狼人", "是狼", "不是狼", "结果")
        return set(mentioned) if any(word in lowered for word in result_words) else set()

    def _text_is_explicit_attack(self, text: str) -> bool:
        lowered = text.lower()
        return any(token in lowered for token in ("今晚刀", "夜刀", "狼刀", "刀口", "击杀", "attack_target", "night_kill", "wolf_target", "wolf_kill"))

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
        if any(token in lowered for token in ("claim", "报身份", "自报", "我是预言家", "我是女巫", "我是守卫", "我是猎人", "我是白痴", "跳预", "跳女巫", "跳守卫", "跳猎人", "跳神", "神职", "预言家", "女巫", "守卫", "猎人", "白痴")):
            signals["claim"] = 5
        if any(token in lowered for token in ("投票", "归票", "vote", "票型", "点名")):
            signals["vote"] = 4
        if self._text_is_explicit_attack(text):
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
                "claim": 0,
                "threat": 0,
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
        bucket[signal_name] = bucket.get(signal_name, 0) + max(1, int(amount))
        if round_no is not None:
            bucket["last_round"] = max(bucket.get("last_round", -1), round_no)

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
        focus_ids: set[str] = set()
        if not isinstance(packet, Mapping):
            return focus_ids
        focus_keys = {
            "sheriff",
            "sheriff_id",
            "sheriff_chain",
            "警长",
            "警长id",
            "警徽",
            "警徽链",
            "vote",
            "voted",
            "vote_target",
            "vote_targets",
            "claim",
            "claimed",
            "suspect",
            "suspected",
            "pressure",
            "point",
            "focus",
            "threat",
            "公开",
            "public",
        }
        container_keys = {
            "players",
            "player_list",
            "events",
            "history",
            "rounds",
            "records",
            "logs",
            "entries",
            "dialogue",
            "dialogues",
            "chat",
            "speech",
            "speeches",
            "transcript",
        }
        for key, value in packet.items():
            key_text = str(key).lower()
            if key_text in focus_keys or any(token in key_text for token in ("sheriff", "警长", "警徽", "vote", "claim", "suspect", "pressure", "focus", "threat", "夜刀", "刀口", "公开", "public")):
                focus_ids.update(self._extract_player_ids(value))
                continue
            if key_text in container_keys:
                if isinstance(value, Mapping):
                    focus_ids.update(self._extract_focus_ids(value))
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, Mapping):
                            focus_ids.update(self._extract_focus_ids(item))
        return focus_ids

    def _extract_alive_ids(self, packet: Mapping[str, Any] | Any) -> set[str]:
        alive_ids: set[str] = set()
        if not isinstance(packet, Mapping):
            return alive_ids
        explicit_alive_keys = {"alive", "alive_players", "alive_player_ids", "alive_ids", "存活", "存活玩家"}
        for key, value in packet.items():
            key_text = str(key).lower()
            if key_text in explicit_alive_keys or key_text.endswith("_alive_players"):
                alive_ids.update(self._extract_player_ids(value))
                continue
            if key_text in {"players", "player_list"}:
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, Mapping) and item.get("alive") is True:
                            alive_ids.update(self._extract_player_ids(item.get("player_id") or item.get("id") or item))
                elif isinstance(value, Mapping):
                    for player_id, state in value.items():
                        if isinstance(state, Mapping) and state.get("alive") is True:
                            ids = self._extract_player_ids(player_id)
                            alive_ids.update(ids or self._extract_player_ids(state))
            if isinstance(value, Mapping):
                alive_ids.update(self._extract_alive_ids(value))
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, Mapping):
                        alive_ids.update(self._extract_alive_ids(item))
        return alive_ids

    def _decay_guard_risk_memory(self, round_no: int) -> None:
        if self._last_decay_round is None:
            self._last_decay_round = round_no
            return
        gap = round_no - self._last_decay_round
        if gap <= 0:
            return
        factor_numerator = 7 ** min(gap, 6)
        factor_denominator = 10 ** min(gap, 6)
        empty_players: list[str] = []
        for player_id, bucket in self.player_risk_profile.items():
            for signal_name in ("pressure", "vote", "claim", "threat", "exposure"):
                value = int(bucket.get(signal_name, 0))
                bucket[signal_name] = (value * factor_numerator) // factor_denominator
            last_round = bucket.get("last_round")
            if all(int(bucket.get(name, 0)) <= 0 for name in ("pressure", "vote", "claim", "threat", "exposure")):
                if not isinstance(last_round, int) or last_round < round_no - 3:
                    empty_players.append(player_id)
        for player_id in empty_players:
            self.player_risk_profile.pop(player_id, None)
        self._last_decay_round = round_no

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

    @classmethod
    def _guard_smoke_checks(cls) -> dict[str, bool]:
        """纯函数级公开信息边界与守卫排序 smoke。"""
        agent = object.__new__(cls)
        agent.player_id = "p6"
        agent.profile = type("Profile", (), {"role": "guard"})()
        agent.last_protected_id = "p1"
        agent.last_protected_round = 1
        agent.alive_player_cache = {"p1", "p2", "p3", "p6"}
        agent.recent_public_focus_ids = deque(maxlen=8)
        agent.player_risk_profile = {}
        agent._guard_signal_round_counts = {}
        agent.claim_ledger = {}
        agent.counterclaim_groups = {}
        agent.sheriff_history = []
        agent.publicly_revealed_wolves = set()
        agent.public_confirmed_good_candidates = set()
        agent.explicit_attack_signals = {}
        agent.public_conflicts = {}
        agent._guard_event_fingerprints = set()
        agent._last_observed_round = 3
        agent._last_decay_round = None

        vote_signal = agent._signal_for_key("vote_target")
        threat_signal = agent._signal_for_key("attack_target")
        agent._update_guard_memory({"round": 3, "vote_target": "p2", "alive_players": ["p1", "p2", "p3", "p6"]})
        no_vote_threat = int(agent.player_risk_profile.get("p2", {}).get("threat", 0)) == 0
        agent._update_guard_memory({"round": 4, "alive_players": ["p2", "p3", "p6"]})
        alive_count_ok = agent._current_alive_count({}, None) == 3

        # A speaker's claim must not leak to a mentioned player.
        agent._ingest_dialogue_entries([{"speaker_id": "p2", "text": "我是预言家，验 p3 金水"}], 4)
        claim_attribution_ok = (
            int(agent.player_risk_profile.get("p2", {}).get("claim", 0)) > 0
            and int(agent.player_risk_profile.get("p3", {}).get("claim", 0)) == 0
            and agent.claim_ledger.get("p2", [{}])[0].get("mentioned_result") == ["p3"]
        )
        # Two claims of the same role are a conflict, not two confirmations.
        agent._ingest_dialogue_entries([{"speaker_id": "p4", "text": "我是预言家"}], 4)
        counterclaim_ok = all("counterclaim" in agent.public_conflicts.get(pid, set()) for pid in ("p2", "p4"))
        # A publicly revealed wolf's badge transfer taints the recipient's chain.
        agent._update_guard_memory({
            "round": 4,
            "revealed_role": {"player_id": "p4", "role": "狼人"},
            "sheriff_transfer": {"from": "p4", "to": "p5"},
        })
        wolf_chain_penalty_ok = bool(agent._guard_candidate_profile("p5", {"round": 4}, None, 4)["wolf_supported"])

        agent.player_risk_profile["p6"] = {"pressure": 3, "vote": 2, "claim": 3, "threat": 0, "exposure": 3, "last_round": 4}
        agent.player_risk_profile["p2"] = {"pressure": 0, "vote": 0, "claim": 3, "threat": 0, "exposure": 0, "last_round": 4}
        agent.explicit_attack_signals["p2"] = 1
        allowed = {"kind": _GUARD_ACTION_KIND, "target_ids": ["p1", "p2", "p6"], "guard_can_repeat_protect": False}
        ranked = agent._guard_candidate_ranking(allowed, {"round": 4}, None)
        ranked_ids = [item["candidate"] for item in ranked]
        repeat_excluded = "p1" not in ranked_ids
        attack_priority_ok = ranked_ids and ranked_ids[0] == "p2"
        claim_not_attack = agent._guard_candidate_profile("p6", {"round": 4}, None, 4)["wolf_attack_risk"] == 0

        # A high-conflict legal choice is gated only when a clean public fact is
        # clearly better; an explicit threat remains a valid model choice.
        agent.public_conflicts["p6"] = {"counterclaim"}
        agent.claim_ledger["p6"] = [{"counterclaim": True, "wolf_supported": False}]
        gate_allowed = {"kind": _GUARD_ACTION_KIND, "target_ids": ["p6", "p3"], "guard_can_repeat_protect": True}
        agent.public_confirmed_good_candidates.add("p3")
        gate_request = {"round": 4, "allowed_actions": [gate_allowed]}
        action_request = {"round": 4, "allowed_actions": [allowed]}
        gated = agent._repair_guard_decision({"kind": _GUARD_ACTION_KIND, "target_id": "p6"}, gate_request, {"round": 4})
        no_gate = agent._repair_guard_decision({"kind": _GUARD_ACTION_KIND, "target_id": "p2"}, action_request, {"round": 4})["target_id"] == "p2"
        return {
            "vote_target_not_threat": vote_signal != ("threat", 4) and no_vote_threat,
            "explicit_attack_target_is_threat": threat_signal == ("threat", 4),
            "claim_not_propagated_to_mentioned_target": claim_attribution_ok,
            "counterclaim_is_conflict": counterclaim_ok,
            "wolf_sheriff_chain_penalized": wolf_chain_penalty_ok,
            "explicit_attack_prioritized": bool(attack_priority_ok),
            "repeat_target_excluded": repeat_excluded,
            "claim_not_repeated_as_attack_risk": claim_not_attack,
            "legal_conflict_target_is_gated": gated.get("target_id") == "p3",
            "model_choice_retained_when_not_high_conflict": no_gate,
            "dead_players_not_counted_alive": alive_count_ok,
        }

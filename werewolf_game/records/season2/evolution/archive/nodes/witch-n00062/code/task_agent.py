"""最小可运行的角色 Task-Agent。

每次行动从一个新的模型会话开始。模型默认只收到当前状态；本轮对话只能通过唯一的
受限工具按需读取。这里不包含策略进化、长期记忆、外部检索或其他 Harness。
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import re
from collections.abc import Mapping
from typing import Any

from ..core.errors import ModelClientError, RuleViolationError
from .llm.coordinator import ModelRequestCoordinator
from ..prompts import RoleProfile, render_prompt


_CHINESE_CHARACTER = re.compile(r"[\u3400-\u9fff]")
_PLAYER_ID_PATTERN = re.compile(r"p\d+", re.IGNORECASE)

CURRENT_ROUND_DIALOGUE_TOOL_NAME = "read_current_round_dialogue"
DEFAULT_MAX_TOOL_CALLS_PER_DECISION = 5
DEFAULT_MAX_TOOL_RESULT_TOKENS = 1000
DEFAULT_MAX_PROMPT_CHARS = 12000
# 这是 Task-Agent 层的“纠错重试”次数：首次模型调用之外，最多再请求两次。
# 传输层的 HTTP/网络重试仍由 ModelClient 的 MODEL_MAX_RETRIES 单独控制。
DEFAULT_MAX_DECISION_RETRIES = 2
_SPEECH_ACTION_KINDS = frozenset({"speak", "last_words"})
_PASS_ACTION_KINDS = frozenset({"pass", "skip", "noop", "do_nothing", "none"})


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
        self._memory_game_id: str | None = None
        self._recent_public_summaries: list[str] = []
        self._last_public_round_phase: str = ""
        self._last_private_resource_summary: str = ""
        self._last_decision_audit: dict[str, Any] | None = None

    async def observe(self, sync_packet: dict[str, Any]) -> None:
        """接收公开同步，并压缩保留少量与后续决策有关的摘要。"""

        game_id = self._extract_first_scalar(
            sync_packet,
            ("game_id", "match_id", "session_id", "table_id", "room_id"),
        )
        if game_id and game_id != self._memory_game_id:
            self._reset_compact_memory(game_id)

        digest = self._build_public_digest(sync_packet)
        if digest:
            if not self._recent_public_summaries or self._recent_public_summaries[-1] != digest:
                self._recent_public_summaries.append(digest)
                self._recent_public_summaries = self._recent_public_summaries[-6:]
        self._last_public_round_phase = self._build_round_phase_tag(sync_packet)

    async def decide(self, turn_packet: dict[str, Any]) -> dict[str, Any]:
        if turn_packet["self"]["player_id"] != self.player_id:
            raise ValueError("行动包被发送给了错误玩家")
        private = turn_packet["private_information"]
        if str(private.get("role")) != self.profile.role:
            raise ValueError("Task-Agent 的角色档案与行动包身份不一致")

        tool_context = turn_packet.get("tool_context") or {}
        current_dialogue = tool_context.get("current_round_dialogue") or []

        private_state = self._build_private_resource_summary(private)
        if private_state:
            self._last_private_resource_summary = private_state
        compact_memory = self._compact_memory_snapshot(turn_packet, current_dialogue)
        safety_notes = self._build_safety_notes(private)
        witch_state_snapshot = self._build_witch_state_snapshot(turn_packet, current_dialogue)

        if self.profile.role == "witch":
            direct_plan = self._plan_witch_action(turn_packet, current_dialogue)
            if direct_plan is not None:
                plan_error = decision_error(direct_plan, turn_packet["request"])
                if plan_error is None:
                    self._last_decision_audit = {
                        "planned_action": direct_plan,
                        "decision_source": "witch_planner",
                    }
                    return direct_plan

        system = self._system_prompt(private)
        prompt = {
            "game": turn_packet["game"],
            "public_rules": turn_packet["public_rules"],
            "self": turn_packet["self"],
            "private_information": private,
            "public_state": turn_packet["public_state"],
            "current_round_dialogue": current_dialogue,
            "request": turn_packet["request"],
            "compact_memory": compact_memory,
            "witch_state_snapshot": witch_state_snapshot,
            "private_resource_state": private_state,
            "safety_notes": safety_notes,
            "history_policy": {
                "default_context": "current_state_only",
                "current_round_dialogue_tool": CURRENT_ROUND_DIALOGUE_TOOL_NAME,
                "max_tool_calls": self.max_tool_calls_per_decision,
                "max_tool_result_tokens": self.max_tool_result_tokens,
            },
        }

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
            action = self._apply_role_safe_fallback(action, turn_packet, raw)
            error = decision_error(action, turn_packet["request"])
            if error is None:
                self._last_decision_audit = {
                    "raw_response": raw,
                    "normalized_action": action,
                }
                return action
            if _attempt >= self.max_decision_retries:
                # 只保留规范化后的行动字段；完整原始响应和推理文本不会进入记录。
                # Runner 可将该行动存入管理员审计记录，便于定位 schema / 长度等问题，
                # 而不会泄露给公开记录。
                self._last_decision_audit = {
                    "raw_response": raw,
                    "normalized_action": action,
                    "validation_error": error,
                }
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

    def _compact_memory_snapshot(self, turn_packet: Mapping[str, Any], current_dialogue: Any) -> dict[str, Any]:
        snapshot: dict[str, Any] = {
            "recent_public": self._recent_public_summaries[-4:],
        }
        if self._last_public_round_phase:
            snapshot["last_public_round_phase"] = self._last_public_round_phase
        if self._last_private_resource_summary:
            snapshot["private_resource_state"] = self._last_private_resource_summary
        public_summary = self._build_public_state_summary(turn_packet.get("public_state"), current_dialogue)
        if public_summary:
            snapshot["witch_state"] = public_summary
        return snapshot

    def _build_private_resource_summary(self, private: Mapping[str, Any]) -> str:
        if self.profile.role != "witch" and str(private.get("role")) != "witch":
            return ""
        keys = {
            "antidote_available": self._extract_first_scalar(private, ("antidote_available", "save_available", "heal_available")),
            "poison_available": self._extract_first_scalar(private, ("poison_available", "poison_left", "kill_available")),
            "antidote_used": self._extract_first_scalar(private, ("antidote_used", "used_antidote", "save_used")),
            "poison_used": self._extract_first_scalar(private, ("poison_used", "used_poison", "kill_used")),
        }
        parts: list[str] = []
        for key, value in keys.items():
            if value is None:
                continue
            parts.append(f"{key}={self._compact_scalar(value)}")
        return "; ".join(parts)

    def _build_safety_notes(self, private: Mapping[str, Any]) -> str:
        if self.profile.role != "witch" and str(private.get("role")) != "witch":
            return ""
        return (
            "公开发言只允许基于公开事实；"
            "不得自报女巫、药水余量、夜刀目标、夜间选择、是否已用药、未公开身份链；"
            "女巫夜间决策应优先高置信公开证据，证据不足时保留资源；"
            "任何内部状态都不能被说成已公开事实。"
        )

    def _apply_role_safe_fallback(
        self,
        action: dict[str, Any],
        turn_packet: dict[str, Any],
        raw_response: object,
    ) -> dict[str, Any]:
        kind = str(action.get("kind", ""))
        if not self._is_poison_action_kind(kind):
            self._last_decision_audit = {
                "raw_response": raw_response,
                "normalized_action": action,
            }
            return action

        target_id = str(action.get("target_id") or "").strip()
        if not target_id:
            self._last_decision_audit = {
                "raw_response": raw_response,
                "normalized_action": action,
                "fallback_reason": "witch_poison_missing_target",
            }
            return action

        pass_action = self._find_pass_action(
            turn_packet["request"].get("allowed_actions", []),
            turn_packet["request"],
        )
        if pass_action is None:
            self._last_decision_audit = {
                "raw_response": raw_response,
                "normalized_action": action,
            }
            return action

        confidence_targets = self._high_confidence_targets(turn_packet)
        target_info = confidence_targets.get(target_id)
        if target_info is not None and int(target_info.get("score", 0) or 0) >= self._poison_confidence_threshold(turn_packet, None):
            self._last_decision_audit = {
                "raw_response": raw_response,
                "normalized_action": action,
                "high_confidence_targets": confidence_targets,
            }
            return action

        fallback_reason = "witch_poison_low_confidence"
        self._last_decision_audit = {
            "raw_response": raw_response,
            "normalized_action": action,
            "fallback_action": pass_action,
            "fallback_reason": fallback_reason,
            "high_confidence_targets": confidence_targets,
        }
        return pass_action

    def _plan_witch_action(
        self,
        turn_packet: Mapping[str, Any],
        current_dialogue: Any,
    ) -> dict[str, Any] | None:
        request = turn_packet.get("request") or {}
        allowed_actions = request.get("allowed_actions")
        if not isinstance(allowed_actions, list):
            return None
        if any(
            isinstance(allowed, Mapping) and self._is_speech_action_kind(str(allowed.get("kind") or ""))
            for allowed in allowed_actions
        ):
            return None

        private = turn_packet.get("private_information") or {}
        public_summary = self._build_public_state_summary(turn_packet.get("public_state"), current_dialogue)
        pass_action = self._find_pass_action(allowed_actions, request)

        if self._resource_flag(private, ("antidote_available", "save_available", "heal_available")):
            heal_action = self._find_resource_action(allowed_actions, self._is_heal_action_kind)
            if heal_action is not None:
                night_target = public_summary.get("night_kill_target")
                confirmed_good_targets = set(public_summary.get("confirmed_good_targets", []))
                if night_target and night_target in confirmed_good_targets:
                    planned = self._select_target_action(heal_action, request, str(night_target))
                    if planned is not None:
                        return planned

        if self._resource_flag(private, ("poison_available", "poison_left", "kill_available")):
            poison_action = self._find_resource_action(allowed_actions, self._is_poison_action_kind)
            if poison_action is not None:
                confidence_targets = self._high_confidence_targets(
                    turn_packet,
                    current_dialogue=current_dialogue,
                    public_summary=public_summary,
                )
                target_ids = poison_action.get("target_ids")
                if isinstance(target_ids, list) and target_ids:
                    threshold = self._poison_confidence_threshold(turn_packet, public_summary)
                    best_target: str | None = None
                    best_score = -1
                    for candidate in target_ids:
                        candidate_id = str(candidate)
                        info = confidence_targets.get(candidate_id)
                        if info is None:
                            continue
                        score = int(info.get("score", 0) or 0)
                        if score >= threshold and score > best_score:
                            best_target = candidate_id
                            best_score = score
                    if best_target is not None:
                        planned = self._select_target_action(poison_action, request, best_target)
                        if planned is not None:
                            return planned

        if pass_action is not None:
            return pass_action
        return None

    def _build_witch_state_snapshot(
        self,
        turn_packet: Mapping[str, Any],
        current_dialogue: Any,
    ) -> dict[str, Any]:
        snapshot: dict[str, Any] = {}
        if self._last_private_resource_summary:
            snapshot["resources"] = self._last_private_resource_summary
        public_summary = self._build_public_state_summary(turn_packet.get("public_state"), current_dialogue)
        if public_summary:
            snapshot["public_state"] = public_summary
        confidence_targets = self._high_confidence_targets(
            turn_packet,
            current_dialogue=current_dialogue,
            public_summary=public_summary,
        )
        if confidence_targets:
            snapshot["high_confidence_targets"] = sorted(confidence_targets)
        return snapshot

    def _build_public_state_summary(self, public_state: Any, current_dialogue: Any) -> dict[str, Any]:
        fragments = self._public_text_fragments(public_state, current_dialogue)
        summary: dict[str, Any] = {}

        alive_count = self._extract_first_scalar(
            public_state,
            ("alive_count", "living_count", "survivor_count", "remaining_players", "players_alive"),
        )
        if alive_count is not None:
            summary["alive_count"] = alive_count

        night_target = self._extract_target_ids_from_fragments(
            fragments,
            ("刀口", "夜刀", "夜杀", "被刀", "死亡", "出局", "淘汰", "night kill", "wolf kill"),
            limit=1,
        )
        if night_target:
            summary["night_kill_target"] = night_target[0]

        confirmed_good_targets = self._extract_target_ids_from_fragments(
            fragments,
            ("金水", "好人", "清白", "已验好", "验好", "确认好", "公认好", "预言家", "警长", "seer", "sheriff"),
            limit=4,
        )
        if confirmed_good_targets:
            summary["confirmed_good_targets"] = confirmed_good_targets

        public_wolf_targets = self._extract_target_ids_from_fragments(
            fragments,
            ("查杀", "强狼", "必狼", "明狼", "铁狼", "狼坑", "锁狼", "悍跳", "验出狼", "查到狼", "wolf"),
            limit=4,
        )
        if public_wolf_targets:
            summary["public_wolf_targets"] = public_wolf_targets

        vote_focus_targets = self._extract_target_ids_from_fragments(
            fragments,
            ("投票", "票型", "归票", "跟票", "票归", "vote", "ballot"),
            limit=4,
        )
        if vote_focus_targets:
            summary["recent_vote_focus"] = vote_focus_targets

        return summary

    def _public_text_fragments(self, *nodes: Any) -> list[str]:
        fragments: list[str] = []
        for node in nodes:
            if node is None:
                continue
            if isinstance(node, str):
                text = self._squash_whitespace(node)
            else:
                text = self._squash_whitespace(self._safe_json_dump(node))
            if text:
                fragments.append(text)
        fragments.extend(self._recent_public_summaries[-4:])
        if self._last_public_round_phase:
            fragments.append(self._last_public_round_phase)
        return fragments

    def _extract_target_ids_from_fragments(
        self,
        fragments: list[str],
        keywords: tuple[str, ...],
        limit: int = 5,
    ) -> list[str]:
        targets: list[str] = []
        seen: set[str] = set()
        for fragment in fragments:
            snippet = self._extract_keyword_snippet(fragment, keywords)
            if not snippet:
                continue
            for target in self._extract_player_ids(snippet):
                if target in seen:
                    continue
                seen.add(target)
                targets.append(target)
                if len(targets) >= limit:
                    return targets
        return targets

    def _score_public_targets(
        self,
        turn_packet: Mapping[str, Any],
        *,
        current_dialogue: Any = None,
        public_summary: Mapping[str, Any] | None = None,
    ) -> dict[str, dict[str, Any]]:
        fragments = self._public_text_fragments(turn_packet.get("public_state"), current_dialogue)
        if public_summary:
            fragments.append(self._safe_json_dump(public_summary))
        if not fragments:
            return {}

        strong_markers = (
            "查杀",
            "强狼",
            "必狼",
            "明狼",
            "铁狼",
            "狼坑",
            "锁狼",
            "悍跳",
            "验出狼",
            "查到狼",
            "公认狼",
            "高置信",
            "狼",
        )
        consensus_markers = ("一致", "共识", "多数", "统一", "跟票", "归票", "全票", "锁定", "确认")
        scored: dict[str, dict[str, Any]] = {}

        for fragment in fragments:
            text = self._squash_whitespace(fragment)
            if not text:
                continue
            lowered = text.lower()
            fragment_targets: set[str] = set()
            for marker in strong_markers:
                escaped = re.escape(marker)
                for match in re.finditer(rf"(?:{escaped})[\s\S]{{0,18}}(p\d+)", text, re.IGNORECASE):
                    target = match.group(1).lower()
                    entry = scored.setdefault(target, {"score": 0, "evidence": []})
                    entry["score"] += 3
                    entry["evidence"].append(self._truncate_text(match.group(0), 80))
                    fragment_targets.add(target)
                for match in re.finditer(rf"(p\d+)[\s\S]{{0,18}}(?:{escaped})", text, re.IGNORECASE):
                    target = match.group(1).lower()
                    entry = scored.setdefault(target, {"score": 0, "evidence": []})
                    entry["score"] += 3
                    entry["evidence"].append(self._truncate_text(match.group(0), 80))
                    fragment_targets.add(target)
            if len(fragment_targets) == 1 and any(marker in lowered for marker in consensus_markers):
                target = next(iter(fragment_targets))
                entry = scored.setdefault(target, {"score": 0, "evidence": []})
                entry["score"] += 1
                entry["evidence"].append(self._truncate_text(text, 80))

        if public_summary:
            for target in public_summary.get("public_wolf_targets", []):
                target_id = str(target).lower()
                if target_id not in scored:
                    continue
                scored[target_id]["score"] += 1
                scored[target_id]["evidence"].append("public_state_high_confidence")

        return scored

    def _high_confidence_targets(
        self,
        turn_packet: dict[str, Any],
        *,
        current_dialogue: Any = None,
        public_summary: Mapping[str, Any] | None = None,
    ) -> dict[str, dict[str, Any]]:
        scored = self._score_public_targets(
            turn_packet,
            current_dialogue=current_dialogue,
            public_summary=public_summary,
        )
        threshold = self._poison_confidence_threshold(turn_packet, public_summary)
        return {
            target: info
            for target, info in scored.items()
            if int(info.get("score", 0) or 0) >= threshold
        }

    def _poison_confidence_threshold(
        self,
        turn_packet: Mapping[str, Any],
        public_summary: Mapping[str, Any] | None,
    ) -> int:
        alive_count = None
        if isinstance(public_summary, Mapping):
            alive_count = self._nonnegative_int(public_summary.get("alive_count"), fallback=None)
        if alive_count is None:
            alive_count = self._nonnegative_int(
                self._extract_first_scalar(
                    turn_packet.get("public_state"),
                    ("alive_count", "living_count", "survivor_count", "remaining_players", "players_alive"),
                ),
                fallback=None,
            )
        if alive_count is not None and alive_count <= 6:
            return 5
        return 6

    def _find_resource_action(
        self,
        allowed_actions: Any,
        kind_predicate: Any,
    ) -> Mapping[str, Any] | None:
        if not isinstance(allowed_actions, list):
            return None
        for allowed in allowed_actions:
            if not isinstance(allowed, Mapping):
                continue
            kind = str(allowed.get("kind") or "")
            if kind_predicate(kind):
                return allowed
        return None

    def _resource_flag(self, private: Mapping[str, Any], key_hints: tuple[str, ...]) -> bool:
        value = self._extract_first_scalar(private, key_hints)
        if value is None:
            return False
        if isinstance(value, str):
            return value.lower() not in {"", "0", "false", "none", "no", "off", "null"}
        return bool(value)

    def _select_target_action(
        self,
        allowed: Mapping[str, Any],
        request: Mapping[str, Any],
        target_id: str,
    ) -> dict[str, Any] | None:
        allowed_target_ids = allowed.get("target_ids")
        allowed_target_id = allowed.get("target_id")
        if isinstance(allowed_target_ids, list):
            if not allowed_target_ids:
                return None
            normalized_targets = {str(item) for item in allowed_target_ids}
            if target_id not in normalized_targets:
                return None
        elif allowed_target_id is None:
            return None
        elif str(allowed_target_id) != target_id:
            return None
        kind = str(allowed.get("kind") or "")
        if not kind:
            return None
        action = {
            "request_id": request["request_id"],
            "player_id": request["player_id"],
            "kind": kind,
            "target_id": target_id,
        }
        return action

    @staticmethod
    def _is_heal_action_kind(kind: str) -> bool:
        normalized = kind.lower().strip()
        return any(token in normalized for token in ("heal", "save", "antidote", "witch_heal", "witch_save"))

    @staticmethod
    def _is_speech_action_kind(kind: str) -> bool:
        normalized = kind.lower().strip()
        return normalized in _SPEECH_ACTION_KINDS or normalized.endswith("_speak") or normalized.endswith("_speech")

    def _find_pass_action(
        self,
        allowed_actions: Any,
        request: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        if not isinstance(allowed_actions, list):
            return None
        for allowed in allowed_actions:
            if not isinstance(allowed, Mapping):
                continue
            kind = str(allowed.get("kind") or "")
            if not self._is_pass_action_kind(kind):
                continue
            action = {
                "request_id": request["request_id"],
                "player_id": request["player_id"],
                "kind": kind,
            }
            target_id = allowed.get("target_id")
            if target_id:
                action["target_id"] = str(target_id)
            text = allowed.get("text")
            if isinstance(text, str):
                action["text"] = text
            return action
        return None

    @staticmethod
    def _is_poison_action_kind(kind: str) -> bool:
        normalized = kind.lower().strip()
        return "poison" in normalized or normalized == "witch_poison"

    @staticmethod
    def _is_pass_action_kind(kind: str) -> bool:
        normalized = kind.lower().strip()
        return normalized in _PASS_ACTION_KINDS or normalized.endswith("_pass") or "pass" == normalized

    def _reset_compact_memory(self, game_id: str | None) -> None:
        self._memory_game_id = game_id
        self._recent_public_summaries = []
        self._last_public_round_phase = ""
        self._last_private_resource_summary = ""
        self._last_decision_audit = None

    def _build_public_digest(self, sync_packet: Mapping[str, Any]) -> str:
        round_tag = self._build_round_phase_tag(sync_packet)
        serialized = self._safe_json_dump(sync_packet)
        snippets: list[str] = []
        for label, keywords in (
            ("death", ("死亡", "出局", "淘汰", "死亡原因", "night kill", "wolf kill")),
            ("vote", ("投票", "票型", "归票", "vote", "ballot")),
            ("sheriff", ("警长", "上警", "警徽", "sheriff", "police")),
            ("speech", ("发言", "自称", "查杀", "金水", "悍跳", "claim", "say")),
        ):
            snippet = self._extract_keyword_snippet(serialized, keywords)
            if snippet:
                snippets.append(f"{label}:{snippet}")
        if not snippets:
            ids = self._extract_player_ids(serialized)
            if ids:
                snippets.append("players:" + ",".join(ids[:5]))
        parts = [part for part in (round_tag, *snippets) if part]
        if not parts:
            return ""
        return self._truncate_text(" | ".join(parts), 260)

    def _build_round_phase_tag(self, packet: Mapping[str, Any]) -> str:
        round_value = self._extract_first_scalar(packet, ("round", "day", "night", "turn"))
        phase_value = self._extract_first_scalar(packet, ("phase", "public_phase", "stage", "channel"))
        parts: list[str] = []
        if round_value is not None:
            parts.append(f"r{self._compact_scalar(round_value)}")
        if phase_value is not None:
            parts.append(f"phase={self._compact_scalar(phase_value)}")
        return " ".join(parts)

    def _safe_json_dump(self, value: Any) -> str:
        try:
            return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
        except TypeError:
            return self._truncate_text(str(value), 1000)

    def _extract_keyword_snippet(self, text: str, keywords: tuple[str, ...]) -> str:
        lowered = text.lower()
        best_index = -1
        best_keyword = ""
        for keyword in keywords:
            idx = lowered.find(keyword.lower())
            if idx != -1 and (best_index == -1 or idx < best_index):
                best_index = idx
                best_keyword = keyword
        if best_index == -1:
            return ""
        start = max(0, best_index - 45)
        end = min(len(text), best_index + max(60, len(best_keyword) + 45))
        snippet = text[start:end]
        return self._truncate_text(self._squash_whitespace(snippet), 140)

    def _extract_player_ids(self, text: str) -> list[str]:
        return list(dict.fromkeys(match.group(0) for match in _PLAYER_ID_PATTERN.finditer(text)))

    def _extract_first_scalar(self, node: Any, key_hints: tuple[str, ...]) -> Any:
        if isinstance(node, Mapping):
            for key, value in node.items():
                key_lower = str(key).lower()
                if any(hint in key_lower for hint in key_hints):
                    scalar = self._compact_scalar(value)
                    if scalar is not None:
                        return scalar
                nested = self._extract_first_scalar(value, key_hints)
                if nested is not None:
                    return nested
        elif isinstance(node, list):
            for item in node:
                nested = self._extract_first_scalar(item, key_hints)
                if nested is not None:
                    return nested
        return None

    def _compact_scalar(self, value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            if isinstance(value, float) and value.is_integer():
                return str(int(value))
            return str(value)
        if isinstance(value, str):
            text = self._squash_whitespace(value)
            return self._truncate_text(text, 80)
        return self._truncate_text(self._squash_whitespace(str(value)), 80)

    @staticmethod
    def _squash_whitespace(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _truncate_text(text: str, limit: int) -> str:
        if limit <= 0:
            return ""
        if len(text) <= limit:
            return text
        return text[: max(0, limit - 1)] + "…"

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

"""预言家 Task-Agent。

每次行动从一个新的模型会话开始；但 Task-Agent 实例会维护轻量的 seer 记忆，
用于跨回合保存查验、候选人、对跳与票型摘要。
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import inspect
import json
import re
from typing import Any

from ..core.errors import ModelClientError, RuleViolationError
from .llm.coordinator import ModelRequestCoordinator
from ..prompts import RoleProfile, render_prompt


_CHINESE_CHARACTER = re.compile(r"[\u3400-\u9fff]")
_PLAYER_ID_PATTERN = re.compile(r"\bp\d+\b", re.IGNORECASE)
_SEAT_REFERENCE_PATTERN = re.compile(r"(?:第\s*)?(\d{1,2})(?:\s*(?:号|位))")
_CURRENT_DIALOGUE_TEXT_LIMIT = 120
_CURRENT_DIALOGUE_ITEM_LIMIT = 6
_SUMMARY_TEXT_LIMIT = 160
_SUMMARY_LIST_LIMIT = 6
_MEMORY_RECORD_LIMIT = 10

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


@dataclass
class SeerMemory:
    game_id: str = ""
    round_no: str = ""
    phase: str = ""
    alive: list[str] = field(default_factory=list)
    dead: list[str] = field(default_factory=list)
    sheriff_id: str = ""
    sheriff_candidates: list[str] = field(default_factory=list)
    inspected: list[dict[str, Any]] = field(default_factory=list)
    # 兼容旧字段，同时用带轮次/阶段的结构化记录避免把历史声明误当当前矛盾。
    declared_claims: list[dict[str, Any]] = field(default_factory=list)
    challenged_players: list[str] = field(default_factory=list)
    inspection_map: dict[str, dict[str, Any]] = field(default_factory=dict)
    claim_timeline: list[dict[str, Any]] = field(default_factory=list)
    vote_timeline: list[dict[str, Any]] = field(default_factory=list)
    publicly_announced_inspections: list[dict[str, Any]] = field(default_factory=list)
    current_pressure_target: str = ""
    seat_to_player_id: dict[str, str] = field(default_factory=dict)
    player_to_seat: dict[str, str] = field(default_factory=dict)
    recent_vote_summary: str = ""
    recent_vote_detail: dict[str, Any] = field(default_factory=dict)
    dialogue_digest: list[str] = field(default_factory=list)

    def reset(self, game_id: str = "") -> None:
        self.game_id = game_id
        self.round_no = ""
        self.phase = ""
        self.alive.clear()
        self.dead.clear()
        self.sheriff_id = ""
        self.sheriff_candidates.clear()
        self.inspected.clear()
        self.declared_claims.clear()
        self.challenged_players.clear()
        self.inspection_map.clear()
        self.claim_timeline.clear()
        self.vote_timeline.clear()
        self.publicly_announced_inspections.clear()
        self.current_pressure_target = ""
        self.seat_to_player_id.clear()
        self.player_to_seat.clear()
        self.recent_vote_summary = ""
        self.recent_vote_detail.clear()
        self.dialogue_digest.clear()


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
        self._seer_memory = SeerMemory()

    async def observe(self, sync_packet: dict[str, Any]) -> None:
        """接收公开同步并更新轻量 seer 记忆。"""

        self._ingest_packet(sync_packet, source="observe")

    async def decide(self, turn_packet: dict[str, Any]) -> dict[str, Any]:
        if turn_packet["self"]["player_id"] != self.player_id:
            raise ValueError("行动包被发送给了错误玩家")
        private = turn_packet["private_information"]
        if str(private.get("role")) != self.profile.role:
            raise ValueError("Task-Agent 的角色档案与行动包身份不一致")

        self._ingest_packet(turn_packet, source="decide")
        mode = self._decide_mode(turn_packet)
        seer_brief = self._build_seer_brief(turn_packet, mode)
        system = self._system_prompt(private)
        prompt = {
            "game": self._compact_value(turn_packet.get("game"), max_depth=1),
            "self": self._compact_value(turn_packet.get("self"), max_depth=1),
            "request": self._compact_request(turn_packet.get("request") or {}),
            "seer_memory": self._compact_seer_memory(),
            "seer_brief": seer_brief,
            "public_state": self._compact_value(turn_packet.get("public_state"), max_depth=2),
            "private_information": self._compact_private_information(private),
            "history_policy": {
                "default_context": "seer_memory_plus_current_round_summary",
                "current_round_dialogue_tool": CURRENT_ROUND_DIALOGUE_TOOL_NAME,
                "max_tool_calls": self.max_tool_calls_per_decision,
                "max_tool_result_tokens": self.max_tool_result_tokens,
            },
        }

        tool_context = turn_packet.get("tool_context") or {}
        current_dialogue = tool_context.get("current_round_dialogue") or []
        current_dialogue_excerpt = self._summarize_dialogue(current_dialogue)
        if current_dialogue_excerpt:
            prompt["current_round_dialogue_excerpt"] = current_dialogue_excerpt

        def execute_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
            del arguments
            if name != CURRENT_ROUND_DIALOGUE_TOOL_NAME:
                return {"error": f"不支持的工具：{name}"}
            return self._current_round_dialogue_tool_result(turn_packet, current_dialogue)

        feedback = ""
        for _attempt in range(self.max_decision_retries + 1):
            # 一次玩家决策的工具预算不能因纠错重试而被放大。首个模型尝试可读取
            # 当前轮对话；所有后续纠错尝试都是没有工具的新会话。
            allow_tool = _attempt == 0 and self.max_tool_calls_per_decision > 0
            user_content: dict[str, Any] = {
                "instruction": self._turn_instruction(
                    turn_packet["request"], feedback, mode=mode, seer_brief=seer_brief
                ),
                "packet": prompt,
            }
            try:
                guarded_system, guarded_user = self._guard_prompt_budget(system, user_content)
                raw = await self._complete_json(
                    system=guarded_system,
                    messages=[
                        {
                            "role": "user",
                            "content": json.dumps(guarded_user, ensure_ascii=False),
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
            if mode == "night_inspect":
                action = self._apply_inspect_target_bias(action, turn_packet["request"], seer_brief)
            error = decision_error(action, turn_packet["request"])
            if error is None:
                error = self._strategy_error(action, turn_packet["request"], mode, seer_brief)
            if error is None:
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

    def _turn_instruction(
        self,
        request: Mapping[str, Any],
        feedback: str,
        *,
        mode: str,
        seer_brief: dict[str, Any],
    ) -> str:
        pieces = [render_action_contract(request)]
        risk_mode = seer_brief.get("risk_mode") if isinstance(seer_brief, Mapping) else None
        risk_active = bool(isinstance(risk_mode, Mapping) and risk_mode.get("active"))
        if mode == "night_inspect":
            pieces.append(
                "夜间优先验信息增益最高的目标：优先 sheriff 候选、强对跳、发言强但立场模糊者、票型关键位；"
                "严格避开自己和已验过的人。"
            )
            if seer_brief.get("preferred_targets"):
                pieces.append("建议目标：" + "、".join(seer_brief["preferred_targets"]))
        elif mode == "sheriff_election_speech":
            pieces.append(
                "竞选警长发言必须先讲已知查验事实，再讲可验证推断和带队计划；"
                "若有对跳，直接对比查验、时间线和动机，不要泛泛而谈。"
            )
        elif mode == "day_speech":
            pieces.append(
                "白天发言要先给查验事实，再给基于票型和发言的推断；先压最危险的矛盾点，再给出可执行投票方向。"
            )
        elif mode == "vote":
            pieces.append(
                "投票优先级：查验事实 > 明显对跳/矛盾 > 票型关键位 > 其他怀疑对象；若正在被集火，优先选择能自保且不背离事实的票。"
            )
        else:
            pieces.append(
                "保持 seer 视角：任何发言都要尽量围绕查验、对跳、票型和可验证推断，不要套通用狼人杀模板。"
            )
        if risk_active and mode in {"sheriff_election_speech", "day_speech", "general"}:
            pieces.append(
                "风险发言模式：只压缩推理，不压缩查验链；优先简短报出存活查杀、仍存活金水和唯一当前票口，不要漏掉已知事实。"
            )
        public_plan = seer_brief.get("public_plan") if isinstance(seer_brief, Mapping) else None
        if isinstance(public_plan, Mapping):
            pieces.append("发言固定四段：1已知事实 -> 2当前存活矛盾 -> 3唯一主票口/无硬票口 -> 4下一步验证计划。")
            if public_plan.get("must_announce"):
                pieces.append("1已知事实：" + "；".join(str(item) for item in public_plan["must_announce"]))
            if public_plan.get("current_public_targets"):
                pieces.append("2当前存活矛盾：" + "、".join(str(item) for item in public_plan["current_public_targets"]))
            if public_plan.get("vote_target_from_inspection"):
                pieces.append("3唯一主票口：" + "、".join(str(item) for item in public_plan["vote_target_from_inspection"][:1]))
            else:
                pieces.append("3当前无存活查杀，禁止把死人或历史查杀写成今日归票。")
            pieces.append("4下一步验证：围绕当前存活矛盾安排下一次查验/复盘票型，不要求已死亡玩家回应。")
            if public_plan.get("historical_dead_facts"):
                pieces.append("死亡目标仅作历史事实：" + "、".join(self._format_inspection_fact(item["round"], item["player_id"], item["result"]) for item in public_plan["historical_dead_facts"][-3:]))
        if seer_brief.get("facts"):
            facts = seer_brief["facts"]
            pieces.append("事实摘要：" + "；".join(facts))
        if seer_brief.get("pressure_points"):
            pieces.append("当前压力点：" + "；".join(seer_brief["pressure_points"]))
        if seer_brief.get("self_risk"):
            pieces.append("自保判断：" + str(seer_brief["self_risk"]))
        if risk_active and mode in {"sheriff_election_speech", "day_speech", "general"}:
            pieces.append("风险模式只压缩推理，不压缩查验链；有查杀要先报查杀，有仍存活金水要重申保护。")
        if feedback:
            pieces.append(f"上一次输出未通过校验：{feedback}")
        pieces.append("最终回答只能是符合契约的 JSON 对象。")
        return "\n".join(pieces)

    def _guard_prompt_budget(self, system: str, user_content: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """在本地保证 system + messages 不超过外部契约；先删背景，后删重复文本。"""
        user = dict(user_content)
        packet = dict(user.get("packet") or {})
        user["packet"] = packet

        def size(sys: str, value: Mapping[str, Any]) -> int:
            return len(sys) + len(json.dumps(value, ensure_ascii=False, separators=(",", ":")))

        if size(system, user) <= self.max_prompt_chars:
            return system, user
        packet.pop("current_round_dialogue_excerpt", None)
        memory = packet.get("seer_memory")
        if isinstance(memory, dict):
            memory["dialogue_digest"] = memory.get("dialogue_digest", [])[-2:]
            memory["vote_timeline"] = memory.get("vote_timeline", [])[-3:]
            memory["claim_timeline"] = memory.get("claim_timeline", [])[-4:]
            memory["declared_claims"] = memory.get("declared_claims", [])[-4:]
            memory["recent_vote_summary"] = self._truncate_text(self._stringify(memory.get("recent_vote_summary")), 80)
        if size(system, user) <= self.max_prompt_chars:
            return system, user
        packet["public_state"] = {}
        packet["game"] = {}
        if size(system, user) <= self.max_prompt_chars:
            return system, user
        # 保留 instruction、request、private_information 和 seer_brief；压缩可重复的普通记忆。
        if isinstance(packet.get("seer_memory"), dict):
            keep = packet["seer_memory"]
            for key in ("dialogue_digest", "vote_timeline", "claim_timeline", "declared_claims", "challenged_players"):
                keep.pop(key, None)
        if size(system, user) <= self.max_prompt_chars:
            return system, user
        # 极端长输入使用最小安全包，并以实际 JSON 长度再次裁剪，绝不把上限交给 Runtime。
        minimal_packet = {
            "request": packet.get("request", {}),
            "private_information": packet.get("private_information", {}),
            "seer_brief": packet.get("seer_brief", {}),
        }
        minimal = {"instruction": user.get("instruction", ""), "packet": minimal_packet}
        # 先保留契约开头；合法候选仍在 packet.request 和 render_action_contract 中。
        for _ in range(3):
            total = len(json.dumps(minimal, ensure_ascii=False, separators=(",", ":")))
            if total <= self.max_prompt_chars:
                return "", minimal
            instruction = self._stringify(minimal.get("instruction"))
            if not instruction:
                break
            minimal["instruction"] = instruction[:max(0, len(instruction) - max(1, total - self.max_prompt_chars))]
        # 处理极端的 private_information：保留其字段结构而不让请求越界。
        serialized = json.dumps(minimal, ensure_ascii=False, separators=(",", ":"))
        if len(serialized) > self.max_prompt_chars:
            minimal["instruction"] = self._truncate_text(self._stringify(minimal.get("instruction")), 200)
            minimal_packet["seer_brief"] = {"public_plan": packet.get("seer_brief", {}).get("public_plan", {})} if isinstance(packet.get("seer_brief"), dict) else {}
            serialized = json.dumps(minimal, ensure_ascii=False, separators=(",", ":"))
        if len(serialized) > self.max_prompt_chars:
            minimal_packet["private_information"] = {"role": packet.get("private_information", {}).get("role", "seer")} if isinstance(packet.get("private_information"), dict) else {}
            serialized = json.dumps(minimal, ensure_ascii=False, separators=(",", ":"))
        if len(serialized) > self.max_prompt_chars:
            # max_prompt_chars 正常远大于该最小包；此分支只为极小测试上限提供硬保证。
            minimal["instruction"] = "输出符合 packet.request 的 JSON。"
            minimal_packet["seer_brief"] = {}
        return "", minimal

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

    def _ingest_packet(self, packet: Mapping[str, Any], *, source: str) -> None:
        del source
        if not isinstance(packet, Mapping):
            return
        game_id = self._extract_game_id(packet)
        if game_id and game_id != self._seer_memory.game_id:
            self._seer_memory.reset(game_id)
        elif game_id and not self._seer_memory.game_id:
            self._seer_memory.game_id = game_id

        self._seer_memory.round_no = self._stringify(
            self._first_present(
                packet,
                (
                    ("game", "round"),
                    ("game", "day"),
                    ("round",),
                    ("day",),
                    ("public_state", "round"),
                    ("public_state", "day"),
                ),
            )
        )
        self._seer_memory.phase = self._stringify(self._extract_phase(packet))
        self._refresh_player_aliases(packet)

        alive = self._extract_player_ids(
            self._first_present(
                packet,
                (
                    ("public_state", "alive"),
                    ("public_state", "alive_players"),
                    ("public_state", "survivors"),
                    ("game", "alive"),
                    ("alive",),
                ),
            )
        )
        dead = self._extract_player_ids(
            self._first_present(
                packet,
                (
                    ("public_state", "dead"),
                    ("public_state", "dead_players"),
                    ("public_state", "eliminated"),
                    ("game", "dead"),
                    ("dead",),
                ),
            )
        )
        if alive:
            self._seer_memory.alive = alive
        if dead:
            self._seer_memory.dead = dead

        sheriff = self._extract_single_player_id(
            self._first_present(
                packet,
                (
                    ("public_state", "sheriff_id"),
                    ("public_state", "sheriff"),
                    ("public_state", "captain"),
                    ("game", "sheriff_id"),
                    ("game", "sheriff"),
                    ("sheriff_id",),
                    ("sheriff",),
                    ("captain",),
                ),
            )
        )
        if sheriff:
            self._seer_memory.sheriff_id = sheriff

        sheriff_candidates = self._extract_player_ids(
            self._first_present(
                packet,
                (
                    ("public_state", "sheriff_candidates"),
                    ("public_state", "sheriff", "candidate_ids"),
                    ("public_state", "candidate_ids"),
                    ("public_state", "candidates"),
                    ("public_state", "election_candidates"),
                    ("game", "sheriff_candidates"),
                    ("sheriff_candidates",),
                    ("candidate_ids",),
                    ("candidates",),
                ),
            )
        )
        if sheriff_candidates:
            self._seer_memory.sheriff_candidates = self._unique_stable(
                list(self._seer_memory.sheriff_candidates) + sheriff_candidates
            )

        vote_summary, vote_detail = self._summarize_vote_info(packet)
        if vote_summary:
            self._seer_memory.recent_vote_summary = vote_summary
        if vote_detail:
            self._seer_memory.recent_vote_detail = vote_detail
            self._record_vote_timeline(vote_detail)

        dialogue_digest = self._extract_dialogue_digest(packet)
        if dialogue_digest:
            self._seer_memory.dialogue_digest = dialogue_digest
            self._update_claims_from_dialogue(dialogue_digest)

        self._update_claims_and_challenges_from_packet(packet)
        self._update_inspection_records(packet)

    def _extract_game_id(self, packet: Mapping[str, Any]) -> str:
        for path in (("game", "game_id"), ("game", "id"), ("game_id",), ("match_id",), ("session_id",)):
            value = self._first_present(packet, (path,))
            game_id = self._stringify(value)
            if game_id:
                return game_id
        return ""

    def _extract_phase(self, packet: Mapping[str, Any]) -> str:
        for path in (
            ("game", "public_phase"),
            ("game", "phase"),
            ("public_state", "phase"),
            ("public_state", "public_phase"),
            ("request", "phase"),
            ("phase",),
        ):
            value = self._first_present(packet, (path,))
            phase = self._stringify(value)
            if phase:
                return phase
        return ""

    def _first_present(self, packet: Mapping[str, Any], paths: Sequence[Sequence[str]]) -> Any:
        for path in paths:
            current: Any = packet
            found = True
            for key in path:
                if not isinstance(current, Mapping) or key not in current:
                    found = False
                    break
                current = current[key]
            if found and current not in (None, ""):
                return current
        return None

    def _extract_player_ids(self, value: Any) -> list[str]:
        result: list[str] = []
        if value is None:
            return result
        if isinstance(value, str):
            result.extend(self._extract_player_ids_from_text(value))
            return self._unique_stable(result)
        if isinstance(value, Mapping):
            for key in ("player_id", "id", "target_id", "speaker", "name"):
                nested = value.get(key)
                if isinstance(nested, str) and nested:
                    result.extend(self._extract_player_ids_from_text(nested))
            for nested in value.values():
                result.extend(self._extract_player_ids(nested))
            return self._unique_stable(result)
        if isinstance(value, Sequence):
            for item in value:
                if isinstance(item, Mapping):
                    result.extend(self._extract_player_ids(item))
                elif isinstance(item, str):
                    result.extend(self._extract_player_ids_from_text(item))
                else:
                    text = self._stringify(item)
                    if text:
                        result.extend(self._extract_player_ids_from_text(text))
            return self._unique_stable(result)
        text = self._stringify(value)
        if text:
            result.extend(self._extract_player_ids_from_text(text))
        return self._unique_stable(result)

    def _extract_single_player_id(self, value: Any) -> str:
        ids = self._extract_player_ids(value)
        return ids[0] if ids else ""

    def _refresh_player_aliases(self, packet: Mapping[str, Any]) -> None:
        public_state = self._first_present(packet, (("public_state",),))
        if not isinstance(public_state, Mapping):
            return
        players = public_state.get("players")
        if players is None:
            return
        seat_to_player_id: dict[str, str] = {}
        player_to_seat: dict[str, str] = {}
        alias_to_player_id: dict[str, str] = {}

        def register_alias(alias: str, player_id: str) -> None:
            alias = self._stringify(alias).strip()
            if not alias or not player_id:
                return
            alias_to_player_id[alias.lower()] = player_id
            if alias.startswith("p"):
                alias_to_player_id[alias] = player_id

        entries: list[tuple[Any, Any]]
        if isinstance(players, Mapping):
            entries = list(players.items())
        elif isinstance(players, Sequence) and not isinstance(players, (str, bytes, bytearray)):
            entries = [(index, item) for index, item in enumerate(list(players), start=1)]
        else:
            return

        for seat_hint, item in entries:
            player_id = ""
            seat_value: Any = seat_hint
            if isinstance(item, Mapping):
                player_id = self._extract_single_player_id(
                    item.get("player_id")
                    or item.get("playerId")
                    or item.get("id")
                    or item.get("player")
                    or item.get("name")
                )
                seat_value = item.get("seat") or item.get("seat_id") or item.get("seatId") or item.get("position") or item.get("index") or seat_hint
            else:
                player_id = self._extract_single_player_id(item)
            if not player_id:
                continue
            seat_text = self._normalize_seat_text(seat_value)
            if seat_text:
                seat_to_player_id[seat_text] = player_id
                player_to_seat[player_id] = seat_text
                for alias in self._seat_alias_variants(seat_text):
                    register_alias(alias, player_id)
            register_alias(player_id, player_id)
            register_alias(player_id.lower(), player_id)

        if seat_to_player_id:
            self._seer_memory.seat_to_player_id = seat_to_player_id
        if player_to_seat:
            self._seer_memory.player_to_seat = player_to_seat
        if alias_to_player_id:
            self._seer_memory.seat_to_player_id.update(alias_to_player_id)

    def _extract_player_ids_from_text(self, text: str) -> list[str]:
        text = self._stringify(text)
        if not text:
            return []
        result = list(_PLAYER_ID_PATTERN.findall(text))
        alias_map = self._seer_memory.seat_to_player_id
        if alias_map:
            lowered = text.lower()
            matches: list[tuple[int, str]] = []
            for alias, player_id in alias_map.items():
                if not alias or not player_id:
                    continue
                pattern = self._alias_pattern(alias)
                for match in pattern.finditer(lowered):
                    matches.append((match.start(), player_id))
            if matches:
                matches.sort(key=lambda item: item[0])
                result.extend(player_id for _, player_id in matches)
        return self._unique_stable(result)

    def _alias_pattern(self, alias: str) -> re.Pattern[str]:
        escaped = re.escape(alias.lower())
        if re.fullmatch(r"p\d+", alias.lower()) or re.fullmatch(r"\d+", alias):
            return re.compile(rf"(?<!\d){escaped}(?!\d)", re.IGNORECASE)
        return re.compile(rf"(?<![A-Za-z0-9_]){escaped}(?![A-Za-z0-9_])", re.IGNORECASE)

    def _seat_alias_variants(self, seat_text: str) -> list[str]:
        normalized = self._normalize_seat_text(seat_text)
        if not normalized:
            return []
        aliases: list[str] = []
        if normalized.isdigit():
            aliases.extend([f"{normalized}号", f"{normalized}位", f"第{normalized}位", f"第{normalized}号"])
        else:
            aliases.append(normalized)
        return self._unique_stable(aliases)

    def _normalize_seat_text(self, seat_value: Any) -> str:
        text = self._stringify(seat_value).strip()
        if not text:
            return ""
        lowered = text.lower()
        if lowered.startswith("第"):
            lowered = lowered[1:]
        lowered = lowered.replace(" ", "")
        if lowered.endswith("号") or lowered.endswith("位"):
            lowered = lowered[:-1]
        if lowered.isdigit():
            return lowered
        match = _SEAT_REFERENCE_PATTERN.search(text)
        if match:
            return match.group(1)
        return text

    def _summarize_vote_info(self, packet: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
        vote_values: list[Any] = []
        for path in (
            ("public_state", "votes"),
            ("public_state", "vote"),
            ("public_state", "vote_result"),
            ("public_state", "day_vote"),
            ("public_state", "election_vote"),
            ("game", "votes"),
            ("votes",),
            ("vote_result",),
        ):
            value = self._first_present(packet, (path,))
            if value is not None:
                vote_values.append(value)
        for path in (("public_state", "events"), ("public_state", "event_log"), ("events",), ("event_log",), ("log",), ("history",)):
            value = self._first_present(packet, (path,))
            extracted = self._extract_vote_payloads(value)
            if extracted:
                vote_values.extend(extracted)
        if not vote_values:
            return "", {}
        summary_parts: list[str] = []
        detail_parts: list[Any] = []
        for value in vote_values[:_SUMMARY_LIST_LIMIT]:
            summary = self._vote_summary_text(value)
            if summary:
                summary_parts.append(summary)
            detail_parts.append(self._compact_value(value, max_depth=2))
        summary_text = "；".join(part for part in summary_parts if part)
        if not detail_parts:
            return summary_text, {}
        if len(detail_parts) == 1 and isinstance(detail_parts[0], dict):
            return summary_text, detail_parts[0]
        return summary_text, {"votes": detail_parts}

    def _vote_summary_text(self, value: Any) -> str:
        if isinstance(value, Mapping):
            voter = self._extract_single_player_id(
                value.get("voter") or value.get("from") or value.get("player_id") or value.get("source") or value.get("speaker")
            )
            target = self._extract_single_player_id(
                value.get("target") or value.get("target_id") or value.get("lynch") or value.get("eliminate") or value.get("to")
            )
            kind = self._stringify(value.get("kind") or value.get("type") or value.get("event_type")).lower()
            counts = value.get("counts") or value.get("votes") or value.get("detail")
            counts_text = self._vote_summary_text(counts)
            if voter and target:
                return f"{voter}->{target}"
            if kind and target:
                return f"{kind}:{target}"
            if target and counts_text:
                return f"{target} | {counts_text}"
            if target:
                return target
            if counts_text:
                return counts_text
            items = []
            for key, nested in value.items():
                if key in {"round", "day", "phase", "kind", "type", "event_type"}:
                    continue
                ids = self._extract_player_ids(nested)
                if ids:
                    items.append(f"{key}:{'、'.join(ids[:3])}")
                else:
                    text = self._truncate_text(self._stringify(nested), 60)
                    if text:
                        items.append(f"{key}:{text}")
            return ";".join(items[:_SUMMARY_LIST_LIMIT])
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            items = []
            for item in list(value)[:_SUMMARY_LIST_LIMIT]:
                if isinstance(item, Mapping):
                    summary = self._vote_summary_text(item)
                    if summary:
                        items.append(summary)
                        continue
                ids = self._extract_player_ids(item)
                if ids:
                    items.append("、".join(ids[:3]))
                else:
                    items.append(self._truncate_text(self._stringify(item), 60))
            return ";".join(item for item in items if item)
        return self._truncate_text(self._stringify(value), 80)

    def _extract_dialogue_digest(self, packet: Mapping[str, Any]) -> list[str]:
        dialogue = self._first_present(
            packet,
            (
                ("tool_context", "current_round_dialogue"),
                ("public_state", "current_round_dialogue"),
                ("public_state", "dialogue"),
                ("dialogue",),
                ("current_round_dialogue",),
            ),
        )
        return self._summarize_dialogue(dialogue)

    def _summarize_dialogue(self, dialogue: Any) -> list[str]:
        if not dialogue:
            return []
        if isinstance(dialogue, Mapping):
            dialogue = [dialogue]
        if not isinstance(dialogue, Sequence) or isinstance(dialogue, (str, bytes, bytearray)):
            text = self._truncate_text(self._stringify(dialogue), _CURRENT_DIALOGUE_TEXT_LIMIT)
            return [text] if text else []

        scored_items: list[tuple[int, int, str]] = []
        recent_start = max(0, len(dialogue) - 4)
        keyword_tokens = ("预言家", "查验", "查杀", "金水", "对跳", "归票", "投票", "质疑", "警长", "验过", "验了")
        for index, item in enumerate(dialogue):
            if isinstance(item, Mapping):
                speaker = self._extract_single_player_id(
                    item.get("speaker") or item.get("player_id") or item.get("id") or item.get("from")
                )
                text = self._stringify(
                    item.get("text")
                    or item.get("content")
                    or item.get("message")
                    or item.get("utterance")
                    or item
                )
            else:
                speaker = ""
                text = self._stringify(item)
            text = self._truncate_text(text.strip(), _CURRENT_DIALOGUE_TEXT_LIMIT)
            if not text:
                continue
            line = f"{speaker}:{text}" if speaker else text
            score = 0
            if speaker == self.player_id:
                score += 50
            if any(token in text for token in keyword_tokens):
                score += 30
            if index >= recent_start:
                score += 10
            if any(token in text for token in ("查杀", "金水", "对跳", "归票", "投票")):
                score += 10
            scored_items.append((score, index, line))

        if not scored_items:
            return []
        picked: list[tuple[int, str]] = []
        seen: set[str] = set()
        for score, index, line in sorted(scored_items, key=lambda item: (-item[0], item[1])):
            if line in seen:
                continue
            seen.add(line)
            picked.append((index, line))
            if len(picked) >= _CURRENT_DIALOGUE_ITEM_LIMIT:
                break
        picked.sort(key=lambda item: item[0])
        return [line for _, line in picked]

    def _current_round_dialogue_tool_result(
        self, turn_packet: Mapping[str, Any], current_dialogue: Any
    ) -> dict[str, Any]:
        return {
            "round": self._stringify(self._first_present(turn_packet, (("game", "round"), ("round",)))) ,
            "phase": self._stringify(self._extract_phase(turn_packet)),
            "count": len(current_dialogue) if isinstance(current_dialogue, Sequence) and not isinstance(current_dialogue, (str, bytes, bytearray)) else (1 if current_dialogue else 0),
            "summary": self._summarize_dialogue(current_dialogue),
        }

    def _update_claims_from_dialogue(self, dialogue_digest: list[str]) -> None:
        for line in dialogue_digest:
            if not line:
                continue
            speaker = ""
            text = line
            if ":" in line:
                speaker, text = line.split(":", 1)
                speaker = self._extract_single_player_id(speaker.strip()) or speaker.strip()
                text = text.strip()
            claim_text = self._extract_explicit_claim_text(text)
            if claim_text:
                entry_speaker = speaker or self._speaker_from_text(text)
                entry = {
                    "round": self._seer_memory.round_no,
                    "phase": self._seer_memory.phase,
                    "speaker": entry_speaker,
                    "claim": self._truncate_text(claim_text, _SUMMARY_TEXT_LIMIT),
                    "alive_at_record": self._record_alive(entry_speaker),
                }
                if entry["speaker"] or entry["claim"]:
                    self._append_bounded(self._seer_memory.declared_claims, entry)
                    self._append_bounded(self._seer_memory.claim_timeline, dict(entry))
            challenge_targets = self._extract_explicit_challenge_targets(text)
            if challenge_targets:
                if speaker:
                    challenge_targets.append(self._extract_single_player_id(speaker) or speaker)
                self._seer_memory.challenged_players = self._unique_stable(
                    list(self._seer_memory.challenged_players) + challenge_targets
                )
        self._seer_memory.declared_claims = self._dedupe_dict_list(self._seer_memory.declared_claims)
        self._seer_memory.claim_timeline = self._dedupe_timeline(self._seer_memory.claim_timeline)
        self._mark_public_inspections(dialogue_digest)

    def _mark_public_inspections(self, dialogue_digest: Sequence[str]) -> None:
        for line in dialogue_digest:
            text = self._stringify(line)
            if self.player_id not in self._extract_player_ids(text) and not text.startswith(self.player_id + ":"):
                continue
            for item in self._seer_memory.inspected:
                if not isinstance(item, Mapping):
                    continue
                target = self._stringify(item.get("target"))
                result = self._stringify(item.get("result"))
                if target and target in text and (result in text or (result == "wolf" and "查杀" in text) or (result == "villager" and "金水" in text)):
                    self._append_bounded(self._seer_memory.publicly_announced_inspections, dict(item, announced=True))

    def _extract_explicit_claim_text(self, text: str) -> str:
        text = self._stringify(text).strip()
        if not text:
            return ""
        claim_markers = ("我是", "我跳", "我自称", "自报", "我认", "报身份", "亮身份")
        role_markers = ("预言家", "女巫", "猎人", "守卫", "警长", "村民", "平民", "狼人", "好人", "坏人")
        if any(marker in text for marker in claim_markers) and any(role in text for role in role_markers):
            return text
        if any(phrase in text for phrase in ("我验", "我查", "验了", "验过", "查验")) and any(
            role in text for role in ("查杀", "金水", "银水", "狼", "好人", "坏人")
        ):
            return text
        return ""

    def _extract_explicit_challenge_targets(self, text: str) -> list[str]:
        text = self._stringify(text).strip()
        if not text:
            return []
        challenge_markers = ("对跳", "质疑", "查杀", "抗推", "冲票", "归票", "出票", "先出", "先票", "出他", "票他")
        if not any(marker in text for marker in challenge_markers):
            return []
        targets = self._extract_player_ids(text)
        if targets:
            return targets
        return []

    def _update_claims_and_challenges_from_packet(self, packet: Mapping[str, Any]) -> None:
        claims = self._first_present(
            packet,
            (
                ("public_state", "claims"),
                ("public_state", "roles_claimed"),
                ("claims",),
                ("roles_claimed",),
            ),
        )
        if isinstance(claims, Sequence) and not isinstance(claims, (str, bytes, bytearray)):
            for item in claims:
                if isinstance(item, Mapping):
                    speaker = self._extract_single_player_id(
                        item.get("speaker") or item.get("player_id") or item.get("id")
                    )
                    claim = self._truncate_text(
                        self._stringify(
                            item.get("claim")
                            or item.get("role")
                            or item.get("text")
                            or item.get("content")
                            or item
                        ),
                        _SUMMARY_TEXT_LIMIT,
                    )
                    if speaker or claim:
                        entry = {
                            "round": self._seer_memory.round_no,
                            "phase": self._seer_memory.phase,
                            "speaker": speaker,
                            "claim": claim,
                            "alive_at_record": self._record_alive(speaker),
                        }
                        self._append_bounded(self._seer_memory.declared_claims, entry)
                        self._append_bounded(self._seer_memory.claim_timeline, dict(entry))
        challenged = self._first_present(
            packet,
            (
                ("public_state", "challenged_players"),
                ("public_state", "suspects"),
                ("public_state", "challenge_targets"),
                ("challenged_players",),
                ("suspects",),
            ),
        )
        if challenged is not None:
            ids = self._extract_player_ids(challenged)
            if ids:
                self._seer_memory.challenged_players = self._unique_stable(
                    list(self._seer_memory.challenged_players) + ids
                )
        self._seer_memory.declared_claims = self._dedupe_dict_list(self._seer_memory.declared_claims)
        self._seer_memory.claim_timeline = self._dedupe_timeline(self._seer_memory.claim_timeline)
        self._seer_memory.challenged_players = self._unique_stable(self._seer_memory.challenged_players)

    def _update_inspection_records(self, packet: Mapping[str, Any]) -> None:
        private = self._first_present(
            packet,
            (
                ("private_information",),
                ("private",),
                ("self", "private_information"),
            ),
        )
        if not isinstance(private, Mapping):
            return
        target = self._extract_single_player_id(
            self._first_present(
                private,
                (
                    ("inspect_target",),
                    ("target_id",),
                    ("target",),
                    ("checked_player",),
                    ("seer_target",),
                ),
            )
        )
        result_value = self._first_present(
            private,
            (
                ("inspect_result",),
                ("check_result",),
                ("seer_result",),
                ("night_result",),
                ("result",),
            ),
        )
        if not target and not result_value:
            return
        result_text = self._inspection_result_text(result_value)
        if not result_text:
            return
        round_no = self._seer_memory.round_no
        if target:
            entry = {
                "round": round_no,
                "phase": self._seer_memory.phase,
                "target": target,
                "result": result_text,
                "alive_at_record": self._record_alive(target),
            }
            if not any(
                self._stringify(item.get("target")) == target
                and self._stringify(item.get("result")) == result_text
                for item in self._seer_memory.inspected
                if isinstance(item, Mapping)
            ):
                self._append_bounded(self._seer_memory.inspected, entry)
            self._seer_memory.inspection_map[target] = dict(entry)
            while len(self._seer_memory.inspection_map) > _MEMORY_RECORD_LIMIT:
                self._seer_memory.inspection_map.pop(next(iter(self._seer_memory.inspection_map)))

    def _inspection_result_text(self, value: Any) -> str:
        text = self._truncate_text(self._stringify(value), _SUMMARY_TEXT_LIMIT)
        if not text:
            return ""
        normalized = text.lower()
        if any(token in normalized for token in ("wolf", "werewolf", "evil", "狼", "bad")):
            return "wolf"
        if any(token in normalized for token in ("villager", "good", "human", "好人", "村民", "平民")):
            return "villager"
        return text

    def _record_alive(self, player_id: str) -> bool:
        if not player_id:
            return False
        return player_id in self._seer_memory.alive if self._seer_memory.alive else player_id not in set(self._seer_memory.dead)

    def _append_bounded(self, values: list[Any], value: Any, limit: int = _MEMORY_RECORD_LIMIT) -> None:
        values.append(value)
        if len(values) > limit:
            del values[:-limit]

    def _dedupe_timeline(self, values: list[dict[str, Any]]) -> list[dict[str, Any]]:
        seen: set[str] = set()
        result: list[dict[str, Any]] = []
        for item in values[-_MEMORY_RECORD_LIMIT:]:
            if not isinstance(item, Mapping):
                continue
            key = json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
            if key not in seen:
                seen.add(key)
                result.append(dict(item))
        return result[-_MEMORY_RECORD_LIMIT:]

    def _record_vote_timeline(self, vote_detail: Mapping[str, Any]) -> None:
        entry = {"round": self._seer_memory.round_no, "phase": self._seer_memory.phase,
                 "detail": self._compact_value(vote_detail, max_depth=2)}
        self._append_bounded(self._seer_memory.vote_timeline, entry)
        self._seer_memory.vote_timeline = self._dedupe_timeline(self._seer_memory.vote_timeline)

    def _live_inspection_facts(self, turn_packet: Mapping[str, Any] | None = None) -> list[dict[str, str]]:
        alive = set(self._seer_memory.alive)
        if not alive and turn_packet is not None:
            alive = set(self._extract_player_ids(self._first_present(
                turn_packet, (("public_state", "alive"), ("public_state", "alive_players"), ("public_state", "survivors"))
            )))
        facts: list[dict[str, str]] = []
        for item in self._seer_memory.inspected:
            if not isinstance(item, Mapping):
                continue
            target, result = self._stringify(item.get("target")), self._inspection_result_text(item.get("result"))
            if target and result in {"wolf", "villager"} and target in alive:
                facts.append({"round": self._stringify(item.get("round")), "player_id": target, "result": result})
        return facts

    def _dead_inspection_facts(self, turn_packet: Mapping[str, Any] | None = None) -> list[dict[str, str]]:
        live_ids = {item["player_id"] for item in self._live_inspection_facts(turn_packet)}
        facts: list[dict[str, str]] = []
        for item in self._seer_memory.inspected:
            if not isinstance(item, Mapping):
                continue
            target, result = self._stringify(item.get("target")), self._inspection_result_text(item.get("result"))
            if target and result in {"wolf", "villager"} and target not in live_ids:
                facts.append({"round": self._stringify(item.get("round")), "player_id": target, "result": result})
        return facts

    def _current_public_targets(self, turn_packet: Mapping[str, Any]) -> list[str]:
        del turn_packet
        alive = set(self._seer_memory.alive)
        targets: list[str] = []
        for item in (self._seer_memory.claim_timeline or self._seer_memory.declared_claims):
            if isinstance(item, Mapping):
                speaker = self._stringify(item.get("speaker"))
                if speaker in alive and speaker != self.player_id:
                    targets.append(speaker)
        targets.extend(p for p in self._seer_memory.challenged_players if p in alive and p != self.player_id)
        return self._unique_stable(targets)

    def _format_inspection_fact(self, round_no: str, target: str, result: str) -> str:
        label = "查杀" if result == "wolf" else "金水" if result == "villager" else result
        prefix = f"{round_no}轮" if round_no else "查验"
        return f"{prefix}{target}={label}"

    def _build_public_plan(
        self, turn_packet: Mapping[str, Any], mode: str, allowed_targets: Sequence[str]
    ) -> dict[str, Any]:
        allowed_target_set = set(self._unique_stable([str(target) for target in allowed_targets if str(target)]))
        live_facts = self._live_inspection_facts(turn_packet)
        dead_facts = self._dead_inspection_facts(turn_packet)
        living_known_wolves = [item["player_id"] for item in live_facts if item["result"] == "wolf"]
        living_confirmed_villagers = [item["player_id"] for item in live_facts if item["result"] == "villager"]
        vote_targets = [target for target in living_known_wolves if not allowed_target_set or target in allowed_target_set]
        public_targets = self._current_public_targets(turn_packet)
        current_pressure = vote_targets[0] if vote_targets else (public_targets[0] if public_targets else "")
        self._seer_memory.current_pressure_target = current_pressure
        return {
            "known_wolves": [item for item in live_facts if item["result"] == "wolf"],
            "historical_dead_facts": dead_facts,
            "confirmed_villagers": [item for item in live_facts if item["result"] == "villager"],
            "living_confirmed_villagers": living_confirmed_villagers,
            "must_announce": [self._format_inspection_fact(item["round"], item["player_id"], item["result"]) for item in live_facts],
            "protected_targets": living_confirmed_villagers,
            "current_public_targets": public_targets,
            "current_pressure_target": current_pressure,
            "vote_target_from_inspection": vote_targets,
            "no_living_hard_vote": not bool(vote_targets),
            "self_preservation_emergency": bool(self._self_is_under_pressure(turn_packet) and not vote_targets and mode in {"sheriff_election_speech", "day_speech", "vote", "general"}),
        }

    def _extract_vote_payloads(self, value: Any) -> list[Any]:
        payloads: list[Any] = []
        seen: set[str] = set()

        def push(item: Any) -> None:
            try:
                key = json.dumps(item, ensure_ascii=False, separators=(",", ":"), default=str)
            except TypeError:
                key = self._stringify(item)
            if key in seen:
                return
            seen.add(key)
            payloads.append(item)

        def walk(node: Any) -> None:
            if node is None:
                return
            if isinstance(node, Mapping):
                kind = self._stringify(node.get("kind") or node.get("type") or node.get("event_type")).lower()
                if kind and any(token in kind for token in ("vote", "ballot", "lynch", "elect")):
                    push(node)
                elif any(key in node for key in ("voter", "target", "target_id", "vote", "votes")):
                    push(node)
                for nested in node.values():
                    walk(nested)
                return
            if isinstance(node, Sequence) and not isinstance(node, (str, bytes, bytearray)):
                for item in node:
                    walk(item)

        walk(value)
        return payloads

    def _strategy_error(
        self,
        action: dict[str, Any],
        request: Mapping[str, Any],
        mode: str,
        seer_brief: Mapping[str, Any],
    ) -> str | None:
        action_kind = str(action.get("kind") or "").lower()
        public_plan = seer_brief.get("public_plan") if isinstance(seer_brief, Mapping) else None
        if not isinstance(public_plan, Mapping):
            public_plan = {}
        allowed_targets = self._allowed_target_ids(request.get("allowed_actions") if isinstance(request, Mapping) else None)
        if mode in {"sheriff_election_speech", "day_speech"} and action_kind == "pass":
            has_facts = bool(seer_brief.get("facts")) or bool(public_plan.get("must_announce"))
            self_is_candidate = self.player_id in self._seer_memory.sheriff_candidates or self.player_id == self._seer_memory.sheriff_id
            if has_facts or self_is_candidate:
                return "预言家有查验事实或竞选身份时不能跳过发言，必须用最短格式报验人"
        if self._kind_looks_like_vote(action_kind):
            target_id = self._stringify(action.get("target_id"))
            living_confirmed_villagers = set(self._stringify(item) for item in public_plan.get("living_confirmed_villagers", []))
            living_known_wolves = [self._stringify(item) for item in public_plan.get("vote_target_from_inspection", [])]
            if target_id and target_id in living_confirmed_villagers and not public_plan.get("self_preservation_emergency"):
                return "不能把票投给已验且仍存活的好人，除非公共计划标记为自保紧急"
            if living_known_wolves and any(target in allowed_targets for target in living_known_wolves):
                if target_id not in living_known_wolves:
                    return "已验狼在可投范围内时，投票必须指向已验狼"
        return None

    def _apply_inspect_target_bias(
        self, action: dict[str, Any], request: Mapping[str, Any], seer_brief: Mapping[str, Any]
    ) -> dict[str, Any]:
        action_kind = str(action.get("kind") or "").lower()
        if not self._kind_looks_like_inspect(action_kind):
            return action
        preferred_targets = [self._stringify(item) for item in seer_brief.get("preferred_targets", []) if self._stringify(item)]
        if not preferred_targets:
            return action
        allowed_targets = self._allowed_target_ids(request.get("allowed_actions") if isinstance(request, Mapping) else None)
        preferred_targets = [target for target in preferred_targets if not allowed_targets or target in allowed_targets]
        if not preferred_targets:
            return action
        current_target = self._stringify(action.get("target_id"))
        if current_target == preferred_targets[0]:
            return action
        reasons = [self._stringify(item) for item in seer_brief.get("target_reasons", [])]
        top_reason = reasons[0] if reasons else ""
        if any(token in top_reason for token in ("存活对跳/公共矛盾", "警长/候选", "对跳/质疑焦点", "公开身份声明", "近期票型关键位")):
            biased = dict(action)
            biased["target_id"] = preferred_targets[0]
            return biased
        return action

    def _build_seer_brief(self, turn_packet: Mapping[str, Any], mode: str) -> dict[str, Any]:
        request = turn_packet.get("request") or {}
        allowed_actions = request.get("allowed_actions") if isinstance(request, Mapping) else None
        allowed_kinds = [str(item.get("kind")) for item in allowed_actions if isinstance(item, Mapping) and item.get("kind")]
        target_candidates = self._allowed_target_ids(allowed_actions)
        preferred_targets, reasons = self._rank_inspect_targets(turn_packet, target_candidates)
        facts = self._seer_facts_summary()
        pressure_points = self._pressure_points_summary(turn_packet)
        self_risk = self._self_risk_summary(turn_packet)
        risk_mode = self._risk_mode_summary(turn_packet, mode, self_risk)
        public_plan = self._build_public_plan(turn_packet, mode, target_candidates)
        brief: dict[str, Any] = {
            "mode": mode,
            "allowed_kinds": allowed_kinds[:_SUMMARY_LIST_LIMIT],
            "facts": facts,
            "public_plan": public_plan,
            "pressure_points": pressure_points,
            "self_risk": self_risk,
            "risk_mode": risk_mode,
        }
        if preferred_targets:
            brief["preferred_targets"] = preferred_targets[:_SUMMARY_LIST_LIMIT]
            brief["target_reasons"] = reasons[:_SUMMARY_LIST_LIMIT]
        return brief

    def _seer_facts_summary(self) -> list[str]:
        facts: list[str] = []
        live = self._live_inspection_facts()
        dead = self._dead_inspection_facts()
        for item in live:
            facts.append(self._format_inspection_fact(item["round"], item["player_id"], item["result"]))
        for item in dead[-3:]:
            facts.append("历史已解决/目标已死亡：" + self._format_inspection_fact(item["round"], item["player_id"], item["result"]))
        if self._seer_memory.sheriff_id:
            facts.append(f"警长/队长位：{self._seer_memory.sheriff_id}")
        if self._seer_memory.recent_vote_summary:
            facts.append(f"最近票型：{self._truncate_text(self._seer_memory.recent_vote_summary, 100)}")
        if self._seer_memory.declared_claims:
            latest = self._seer_memory.declared_claims[-1]
            speaker = self._stringify(latest.get("speaker"))
            claim = self._stringify(latest.get("claim"))
            if speaker or claim:
                facts.append(f"最新公开声明：{speaker + ' ' if speaker else ''}{claim}".strip())
        return facts[:_MEMORY_RECORD_LIMIT]

    def _pressure_points_summary(self, turn_packet: Mapping[str, Any]) -> list[str]:
        points: list[str] = []
        if self._seer_memory.sheriff_candidates:
            points.append("警长候选：" + "、".join(self._seer_memory.sheriff_candidates[:_SUMMARY_LIST_LIMIT]))
        if self._seer_memory.challenged_players:
            points.append("已被质疑/对跳关注：" + "、".join(self._seer_memory.challenged_players[:_SUMMARY_LIST_LIMIT]))
        vote_detail = self._seer_memory.recent_vote_detail
        if isinstance(vote_detail, Mapping) and vote_detail:
            points.append("票型详情已更新")
        request = turn_packet.get("request") or {}
        allowed_targets = self._allowed_target_ids(request.get("allowed_actions") if isinstance(request, Mapping) else None)
        ranked, _ = self._rank_inspect_targets(turn_packet, allowed_targets)
        if ranked:
            points.append("夜验优先：" + "、".join(ranked[:_SUMMARY_LIST_LIMIT]))
        if self._self_is_under_pressure(turn_packet):
            points.append("当前处于高暴露/高集火风险")
        return points[:_SUMMARY_LIST_LIMIT]

    def _self_risk_summary(self, turn_packet: Mapping[str, Any]) -> str:
        if self._self_is_under_pressure(turn_packet):
            return "自保压力高：已被公开质疑/集火或处于高暴露阶段，发言宜先简后实。"
        phase = self._extract_phase(turn_packet).lower()
        if "elect" in phase or "sheriff" in phase or "警长" in phase or "竞选" in phase:
            return "身份暴露风险中等：竞选期要先立可信事实，再给投票方向。"
        if len(self._seer_memory.declared_claims) >= 3:
            return "信息竞争激烈：要把查验事实尽快转成压票点。"
        return "常规风险：优先保留信息优势，不要过度展开。"

    def _self_is_under_pressure(self, turn_packet: Mapping[str, Any]) -> bool:
        self_seen = self.player_id in self._seer_memory.challenged_players
        exposed_claims = len(self._seer_memory.declared_claims) >= 4
        return self_seen or exposed_claims

    def _risk_mode_summary(
        self, turn_packet: Mapping[str, Any], mode: str, self_risk: str
    ) -> dict[str, Any]:
        under_pressure = self._self_is_under_pressure(turn_packet)
        phase = self._extract_phase(turn_packet).lower()
        channel = str((turn_packet.get("request") or {}).get("channel") or "").lower()
        election_context = any(token in phase for token in ("elect", "sheriff", "竞选", "警长")) or any(
            token in channel for token in ("elect", "sheriff", "竞选", "警长")
        )
        if not under_pressure:
            if election_context:
                return {
                    "active": False,
                    "level": "medium",
                    "mode": mode,
                    "rule": "竞选期允许完整报首验和警徽流；只压缩推理，不压缩查验链。",
                }
            return {
                "active": False,
                "level": "normal",
                "mode": mode,
                "rule": "可正常展开事实链，但仍优先使用可回看验证的证据。",
            }
        if mode in {"sheriff_election_speech", "day_speech", "general"}:
            return {
                "active": True,
                "level": "high",
                "mode": mode,
                "rule": "只压缩推理，不压缩查验链；有查杀先报查杀，存活金水要重申保护。",
                "self_risk": self_risk,
            }
        return {
            "active": True,
            "level": "medium",
            "mode": mode,
            "rule": "保持发言克制，但不要漏报已知查杀或仍存活金水。",
            "self_risk": self_risk,
        }

    def _decide_mode(self, turn_packet: Mapping[str, Any]) -> str:
        request = turn_packet.get("request") or {}
        allowed_actions = request.get("allowed_actions") if isinstance(request, Mapping) else None
        allowed_kinds = [str(item.get("kind") or "").lower() for item in allowed_actions if isinstance(item, Mapping)]
        phase = (self._extract_phase(turn_packet) or "").lower()
        channel = str(request.get("channel") or "").lower() if isinstance(request, Mapping) else ""
        if any(self._kind_looks_like_inspect(kind) for kind in allowed_kinds):
            return "night_inspect"
        if any(kind in _SPEECH_ACTION_KINDS for kind in allowed_kinds):
            if any(token in phase for token in ("sheriff", "election", "竞选", "警长")) or any(
                token in channel for token in ("sheriff", "election", "竞选", "警长")
            ):
                return "sheriff_election_speech"
            return "day_speech"
        if any(self._kind_looks_like_vote(kind) for kind in allowed_kinds):
            return "vote"
        return "general"

    def _kind_looks_like_inspect(self, kind: str) -> bool:
        kind = kind.lower()
        return any(token in kind for token in ("inspect", "check", "seer", "查验", "验", "探查"))

    def _kind_looks_like_vote(self, kind: str) -> bool:
        kind = kind.lower()
        return any(token in kind for token in ("vote", "ballot", "lynch", "投票", "归票"))

    def _allowed_target_ids(self, allowed_actions: Any) -> list[str]:
        target_ids: list[str] = []
        if not isinstance(allowed_actions, Sequence) or isinstance(allowed_actions, (str, bytes, bytearray)):
            return target_ids
        for item in allowed_actions:
            if not isinstance(item, Mapping):
                continue
            ids = item.get("target_ids")
            if isinstance(ids, Sequence) and not isinstance(ids, (str, bytes, bytearray)):
                for target_id in ids:
                    if isinstance(target_id, str) and target_id:
                        target_ids.append(target_id)
        return self._unique_stable(target_ids)

    def _rank_inspect_targets(
        self, turn_packet: Mapping[str, Any], candidate_ids: Sequence[str]
    ) -> tuple[list[str], list[str]]:
        alive = self._seer_memory.alive or self._extract_player_ids(
            self._first_present(turn_packet, (("public_state", "alive"), ("public_state", "alive_players")))
        )
        alive_set = set(alive)
        candidates = [
            target_id
            for target_id in self._unique_stable(candidate_ids)
            if target_id and target_id != self.player_id and target_id in alive_set and target_id not in self._seen_inspection_targets()
        ]
        if not candidates:
            candidates = [
                target_id
                for target_id in alive
                if target_id and target_id != self.player_id and target_id not in self._seen_inspection_targets()
            ]
        scores: dict[str, int] = {target_id: 0 for target_id in candidates}
        reasons: dict[str, list[str]] = {target_id: [] for target_id in candidates}

        sheriff_like = self._unique_stable(
            [self._seer_memory.sheriff_id] + self._seer_memory.sheriff_candidates
        )
        unresolved_public = set(self._current_public_targets(turn_packet))
        for target_id in candidates:
            if target_id in unresolved_public:
                scores[target_id] += 70
                reasons[target_id].append("存活对跳/公共矛盾")
            if target_id in sheriff_like:
                scores[target_id] += 50
                reasons[target_id].append("警长/候选")
            if target_id in self._seer_memory.challenged_players:
                scores[target_id] += 30
                reasons[target_id].append("对跳/质疑焦点")
            if self._has_declared_claim(target_id):
                scores[target_id] += 20
                reasons[target_id].append("公开身份声明")
            if self._appears_in_recent_vote(target_id):
                scores[target_id] += 18
                reasons[target_id].append("近期票型关键位")
            if self._speaks_a_lot(target_id):
                scores[target_id] += 8
                reasons[target_id].append("发言密集")

        ranked = sorted(candidates, key=lambda item: (-scores[item], candidates.index(item)))
        ranked_reasons = [
            f"{target_id}:{'、'.join(reasons[target_id]) or '默认信息增益'}"
            for target_id in ranked
        ]
        return ranked, ranked_reasons

    def _seen_inspection_targets(self) -> set[str]:
        seen: set[str] = set()
        for item in self._seer_memory.inspected:
            if isinstance(item, Mapping):
                target = self._stringify(item.get("target"))
                if target:
                    seen.add(target)
        return seen

    def _has_declared_claim(self, target_id: str) -> bool:
        for claim in self._seer_memory.declared_claims:
            if not isinstance(claim, Mapping):
                continue
            if self._stringify(claim.get("speaker")) == target_id:
                return True
            if target_id and target_id in self._stringify(claim.get("claim")):
                return True
        return False

    def _appears_in_recent_vote(self, target_id: str) -> bool:
        vote_detail = self._seer_memory.recent_vote_detail
        if not vote_detail:
            return False
        text = json.dumps(vote_detail, ensure_ascii=False, separators=(",", ":"))
        return target_id in text

    def _speaks_a_lot(self, target_id: str) -> bool:
        count = 0
        for line in self._seer_memory.dialogue_digest:
            if target_id in self._extract_player_ids(line):
                count += 1
        return count >= 2

    def _compact_request(self, request: Mapping[str, Any]) -> dict[str, Any]:
        allowed_actions = request.get("allowed_actions")
        compact_actions: list[dict[str, Any]] = []
        if isinstance(allowed_actions, Sequence) and not isinstance(allowed_actions, (str, bytes, bytearray)):
            for allowed in allowed_actions:
                if not isinstance(allowed, Mapping):
                    continue
                item: dict[str, Any] = {"kind": self._stringify(allowed.get("kind"))}
                target_ids = allowed.get("target_ids")
                if isinstance(target_ids, Sequence) and not isinstance(target_ids, (str, bytes, bytearray)):
                    ids = [self._stringify(v) for v in list(target_ids) if self._stringify(v)]
                    if ids:
                        item["target_ids"] = ids
                if allowed.get("max_chars") is not None:
                    item["max_chars"] = allowed.get("max_chars")
                if allowed.get("require_chinese") is not None:
                    item["require_chinese"] = bool(allowed.get("require_chinese"))
                compact_actions.append(item)
        return {
            "request_id": self._stringify(request.get("request_id")),
            "phase": self._stringify(request.get("phase")),
            "channel": self._stringify(request.get("channel")),
            "allowed_actions": compact_actions,
        }

    def _compact_private_information(self, private: Mapping[str, Any]) -> dict[str, Any]:
        compact = self._compact_value(private, max_depth=2)
        if isinstance(compact, dict):
            compact["seer_memory_hint"] = self._compact_seer_memory()
        return compact

    def _compact_seer_memory(self) -> dict[str, Any]:
        priority = self._unique_stable(
            [self.player_id]
            + self._seer_memory.alive
            + self._seer_memory.dead
            + [self._stringify(item.get("target")) for item in self._seer_memory.inspected if isinstance(item, Mapping)]
            + self._seer_memory.sheriff_candidates
            + self._seer_memory.challenged_players
            + [self._stringify(item.get("speaker")) for item in self._seer_memory.claim_timeline if isinstance(item, Mapping)]
        )
        alive_set, dead_set = set(self._seer_memory.alive), set(self._seer_memory.dead)
        player_status = []
        for player_id in priority:
            if not player_id:
                continue
            player_status.append({
                "player_id": player_id,
                "alive": player_id in alive_set if alive_set else player_id not in dead_set,
                "seat": self._seer_memory.player_to_seat.get(player_id, ""),
                "known_inspection_result": self._inspection_result_text(self._seer_memory.inspection_map.get(player_id, {}).get("result")) if player_id in self._seer_memory.inspection_map else "",
                "claimed_role": next((self._truncate_text(self._stringify(item.get("claim")), 50) for item in self._seer_memory.claim_timeline if isinstance(item, Mapping) and self._stringify(item.get("speaker")) == player_id), ""),
            })
        return {
            "game_id": self._seer_memory.game_id,
            "round": self._seer_memory.round_no,
            "phase": self._seer_memory.phase,
            "alive": self._seer_memory.alive,
            "dead": self._seer_memory.dead,
            "player_status": player_status,
            "sheriff_id": self._seer_memory.sheriff_id,
            "sheriff_candidates": self._seer_memory.sheriff_candidates,
            "inspected": self._seer_memory.inspected[-_MEMORY_RECORD_LIMIT:],
            "inspection_map": {k: v for k, v in self._seer_memory.inspection_map.items()},
            "declared_claims": self._seer_memory.declared_claims[-_MEMORY_RECORD_LIMIT:],
            "claim_timeline": self._seer_memory.claim_timeline[-_MEMORY_RECORD_LIMIT:],
            "vote_timeline": self._seer_memory.vote_timeline[-_MEMORY_RECORD_LIMIT:],
            "publicly_announced_inspections": self._seer_memory.publicly_announced_inspections[-_MEMORY_RECORD_LIMIT:],
            "challenged_players": self._seer_memory.challenged_players,
            "current_pressure_target": self._seer_memory.current_pressure_target,
            "recent_vote_summary": self._truncate_text(self._seer_memory.recent_vote_summary, 120),
            "dialogue_digest": [self._truncate_text(x, 120) for x in self._seer_memory.dialogue_digest[-_SUMMARY_LIST_LIMIT:]],
        }

    def _compact_value(self, value: Any, *, max_depth: int, _depth: int = 0) -> Any:
        if value is None or isinstance(value, (bool, int, float)):
            return value
        if isinstance(value, str):
            return self._truncate_text(value, _SUMMARY_TEXT_LIMIT)
        if _depth >= max_depth:
            return self._truncate_text(self._stringify(value), _SUMMARY_TEXT_LIMIT)
        if isinstance(value, Mapping):
            items: dict[str, Any] = {}
            for idx, (key, nested) in enumerate(value.items()):
                if idx >= _SUMMARY_LIST_LIMIT:
                    items["__truncated__"] = len(value) - _SUMMARY_LIST_LIMIT
                    break
                items[self._stringify(key)] = self._compact_value(nested, max_depth=max_depth, _depth=_depth + 1)
            return items
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            items = [
                self._compact_value(item, max_depth=max_depth, _depth=_depth + 1)
                for item in list(value)[:_SUMMARY_LIST_LIMIT]
            ]
            if len(value) > _SUMMARY_LIST_LIMIT:
                items.append(f"...(+{len(value) - _SUMMARY_LIST_LIMIT})")
            return items
        return self._truncate_text(self._stringify(value), _SUMMARY_TEXT_LIMIT)

    def _unique_stable(self, values: Sequence[str]) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for value in values:
            text = self._stringify(value)
            if not text or text in seen:
                continue
            seen.add(text)
            ordered.append(text)
        return ordered

    def _dedupe_dict_list(self, values: list[dict[str, Any]]) -> list[dict[str, Any]]:
        seen: set[tuple[str, str, str, str]] = set()
        deduped: list[dict[str, Any]] = []
        for item in values:
            if not isinstance(item, Mapping):
                continue
            speaker = self._stringify(item.get("speaker"))
            claim = self._stringify(item.get("claim"))
            key = (self._stringify(item.get("round")), self._stringify(item.get("phase")), speaker, claim)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(dict(item))
        return deduped[-_MEMORY_RECORD_LIMIT:]

    def _speaker_from_text(self, text: str) -> str:
        ids = self._extract_player_ids(text)
        return ids[0] if ids else ""

    @staticmethod
    def _truncate_text(text: str, limit: int) -> str:
        text = text.strip()
        if len(text) <= limit:
            return text
        return text[: max(0, limit - 1)] + "…"

    @staticmethod
    def _stringify(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, bool):
            return "true" if value else "false"
        try:
            return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        except TypeError:
            return str(value)

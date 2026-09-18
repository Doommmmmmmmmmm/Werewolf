"""最小可运行的角色 Task-Agent。

每次行动从一个新的模型会话开始。模型默认只收到当前状态；本轮对话只能通过唯一的
受限工具按需读取。这里不包含策略进化、长期记忆、外部检索或其他 Harness。
"""

from __future__ import annotations

import asyncio
from collections import deque
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
_SIGNAL_TEXT = re.compile(r"(票|投|狼|好人|金水|查杀|对跳|冲票|带票|带节奏|怀疑|矛盾|身份|死亡|开枪|跳过|pass|vote|claim|wolf|kill|dead)", re.IGNORECASE)
_PLAYER_ID_TEXT = re.compile(r"\b[a-zA-Z]*\d+\b")

CURRENT_ROUND_DIALOGUE_TOOL_NAME = "read_current_round_dialogue"
DEFAULT_MAX_TOOL_CALLS_PER_DECISION = 5
DEFAULT_MAX_TOOL_RESULT_TOKENS = 1000
DEFAULT_MAX_PROMPT_CHARS = 12000
# 这是 Task-Agent 层的“纠错重试”次数：首次模型调用之外，最多再请求两次。
# 传输层的 HTTP/网络重试仍由 ModelClient 的 MODEL_MAX_RETRIES 单独控制。
DEFAULT_MAX_DECISION_RETRIES = 2
_SPEECH_ACTION_KINDS = frozenset({"speak", "last_words"})

_MEMORY_KEYS = {
    "round",
    "phase",
    "day",
    "night",
    "speaker",
    "speaker_id",
    "player",
    "player_id",
    "voter",
    "voter_id",
    "target",
    "target_id",
    "kind",
    "action",
    "event",
    "result",
    "status",
    "role",
    "claim",
    "text",
    "message",
    "vote",
    "votes",
    "alive",
    "dead",
    "revealed",
    "public_role",
    "reason",
    "winner",
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
        self._recent_public_events: deque[dict[str, Any]] = deque(maxlen=24)

    async def observe(self, sync_packet: dict[str, Any]) -> None:
        """接收公开同步，并压缩保存为短期证据账本。"""

        for record in self._extract_memory_records(sync_packet):
            if record:
                self._recent_public_events.append(record)

    async def decide(self, turn_packet: dict[str, Any]) -> dict[str, Any]:
        if turn_packet["self"]["player_id"] != self.player_id:
            raise ValueError("行动包被发送给了错误玩家")
        private = turn_packet["private_information"]
        if str(private.get("role")) != self.profile.role:
            raise ValueError("Task-Agent 的角色档案与行动包身份不一致")

        if self.profile.role == "hunter":
            hunter_action = self._decide_hunter_reaction_action(turn_packet)
            if hunter_action is not None:
                return hunter_action

        decision_brief = self._build_decision_brief(turn_packet)
        system = self._system_prompt(private, decision_brief=decision_brief)
        prompt = {
            "game": turn_packet["game"],
            "public_rules": turn_packet["public_rules"],
            "self": turn_packet["self"],
            "private_information": private,
            "public_state": turn_packet["public_state"],
            "request": turn_packet["request"],
            "decision_brief": decision_brief,
            "history_policy": {
                "default_context": "current_state_only",
                "current_round_dialogue_tool": CURRENT_ROUND_DIALOGUE_TOOL_NAME,
                "max_tool_calls": self.max_tool_calls_per_decision,
                "max_tool_result_tokens": self.max_tool_result_tokens,
            },
        }

        tool_context = turn_packet.get("tool_context") or {}
        current_dialogue = tool_context.get("current_round_dialogue") or []

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

    def _system_prompt(self, private: Mapping[str, Any], *, decision_brief: str = "") -> str:
        role_task = self.profile.task
        if decision_brief:
            role_task = f"{role_task}\n\n【本局决策简报】\n{decision_brief}"
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

    def _build_decision_brief(self, turn_packet: Mapping[str, Any]) -> str:
        if self.profile.role != "hunter":
            return ""
        request = turn_packet.get("request") if isinstance(turn_packet.get("request"), Mapping) else {}
        public_state = turn_packet.get("public_state") if isinstance(turn_packet.get("public_state"), Mapping) else {}
        ranking = self._rank_hunter_targets(request, public_state, list(self._recent_public_events))
        phase = str(
            turn_packet.get("game", {}).get(
                "public_phase",
                turn_packet.get("game", {}).get("phase", request.get("phase", "unknown")),
            )
        )
        lines = [f"当前阶段：{phase}"]
        legal_targets = ranking.get("legal_targets") or []
        lines.append("合法目标：" + ("、".join(legal_targets[:8]) if legal_targets else "无"))
        recent = ranking.get("recent_memory") or []
        if recent:
            lines.append("近期证据：" + " | ".join(recent[-4:]))
        ordered = ranking.get("target_ranking") or []
        if ordered:
            top_bits = []
            for item in ordered[:3]:
                reason = str(item.get("reason") or "")[:28]
                top_bits.append(f"{item.get('target_id')}({item.get('score')}:{reason})")
            lines.append("目标排序：" + " > ".join(top_bits))
        strong_evidence = ranking.get("strong_evidence") or []
        weak_evidence = ranking.get("weak_evidence") or []
        lines.append("强证据：" + ("；".join(strong_evidence[:3]) if strong_evidence else "无"))
        lines.append("弱证据：" + ("；".join(weak_evidence[:2]) if weak_evidence else "无"))
        if ranking.get("best_target"):
            lines.append(f"行动建议：优先开枪 {ranking['best_target']}")
        else:
            lines.append("行动建议：仅在确认所有候选都偏弱时跳过")
        return self._truncate_text("\n".join(lines), 900)

    def _decide_hunter_reaction_action(self, turn_packet: Mapping[str, Any]) -> dict[str, Any] | None:
        if self.profile.role != "hunter":
            return None
        request = turn_packet.get("request") if isinstance(turn_packet.get("request"), Mapping) else {}
        if not self._is_hunter_reaction_request(turn_packet, request):
            return None
        public_state = turn_packet.get("public_state") if isinstance(turn_packet.get("public_state"), Mapping) else {}
        ranking = self._rank_hunter_targets(request, public_state, list(self._recent_public_events))
        if ranking.get("best_target"):
            action = self._find_target_action(request, str(ranking["best_target"]))
            if action is not None:
                return action
            pass_action = self._find_pass_action(request)
            if pass_action is not None:
                return pass_action
        if ranking.get("pass_recommended"):
            pass_action = self._find_pass_action(request)
            if pass_action is not None:
                return pass_action
        return None

    def _is_hunter_reaction_request(
        self,
        turn_packet: Mapping[str, Any],
        request: Mapping[str, Any] | None = None,
    ) -> bool:
        request = request if isinstance(request, Mapping) else {}
        game = turn_packet.get("game") if isinstance(turn_packet.get("game"), Mapping) else {}
        haystack = " ".join(
            str(part)
            for part in (
                request.get("phase"),
                request.get("kind"),
                request.get("action"),
                request.get("channel"),
                game.get("phase"),
                game.get("public_phase"),
            )
            if part is not None
        ).lower()
        return "hunter_reaction" in haystack or "hunter reaction" in haystack or "猎人反应" in haystack

    def _find_target_action(self, request: Mapping[str, Any], target_id: str) -> dict[str, Any] | None:
        raw_actions = request.get("allowed_actions")
        if not isinstance(raw_actions, list):
            return None
        for allowed in raw_actions:
            if not isinstance(allowed, Mapping):
                continue
            raw_target_ids = allowed.get("target_ids")
            if not isinstance(raw_target_ids, list):
                continue
            normalized_targets = {str(item) for item in raw_target_ids if item is not None}
            if target_id not in normalized_targets:
                continue
            kind = str(allowed.get("kind") or "")
            if not kind:
                continue
            action: dict[str, Any] = {
                "request_id": request["request_id"],
                "player_id": request["player_id"],
                "kind": kind,
                "target_id": target_id,
            }
            return action
        return None

    def _find_pass_action(self, request: Mapping[str, Any]) -> dict[str, Any] | None:
        raw_actions = request.get("allowed_actions")
        if not isinstance(raw_actions, list):
            return None
        for allowed in raw_actions:
            if not isinstance(allowed, Mapping):
                continue
            kind = str(allowed.get("kind") or "")
            if self._is_pass_kind(kind):
                return {
                    "request_id": request["request_id"],
                    "player_id": request["player_id"],
                    "kind": kind,
                }
        return None

    @staticmethod
    def _is_pass_kind(kind: str) -> bool:
        return kind.lower() in {"pass", "skip"}

    def _rank_hunter_targets(
        self,
        request: Mapping[str, Any],
        public_state: Mapping[str, Any],
        memory_records: list[dict[str, Any]],
    ) -> dict[str, Any]:
        legal_targets = self._collect_target_ids(request)
        recent_memory = [
            self._format_memory_record(record)
            for record in memory_records[-10:]
            if isinstance(record, Mapping)
        ]
        recent_memory = [line for line in recent_memory if line]
        if not legal_targets:
            return {
                "legal_targets": [],
                "recent_memory": recent_memory[-6:],
                "target_ranking": [],
                "pass_recommended": True,
                "best_target": None,
                "strong_target": None,
                "strong_evidence": [],
                "weak_evidence": [],
            }

        try:
            public_blob_raw = json.dumps(public_state, ensure_ascii=False, separators=(",", ":"))
        except TypeError:
            public_blob_raw = repr(public_state)
        public_blob = self._truncate_text(public_blob_raw, 1600)
        target_ranking: list[dict[str, Any]] = []
        for target_id in legal_targets:
            score, reasons = self._score_hunter_target(target_id, memory_records, public_blob)
            target_ranking.append(
                {
                    "target_id": target_id,
                    "score": score,
                    "reason": "；".join(reasons) if reasons else "证据不足",
                }
            )
        target_ranking.sort(key=lambda item: (-int(item["score"]), str(item["target_id"])))
        best = target_ranking[0] if target_ranking else None
        second_score = int(target_ranking[1]["score"]) if len(target_ranking) > 1 else 0
        best_score = int(best["score"]) if best is not None else 0
        gap = best_score - second_score
        strong_target = None
        strong_evidence: list[str] = []
        weak_evidence: list[str] = []
        if best is not None:
            best_reasons = self._split_reason_text(str(best.get("reason") or ""))
            if self._is_strong_hunter_score(best_score, gap, best_reasons):
                strong_target = str(best["target_id"])
                strong_evidence = best_reasons[:4]
            else:
                weak_evidence = best_reasons[:3]
        pass_recommended = strong_target is None
        return {
            "legal_targets": legal_targets,
            "recent_memory": recent_memory[-6:],
            "target_ranking": target_ranking,
            "pass_recommended": pass_recommended,
            "best_target": strong_target,
            "strong_target": strong_target,
            "strong_evidence": strong_evidence,
            "weak_evidence": weak_evidence,
        }

    def _is_strong_hunter_score(self, best_score: int, gap: int, reasons: list[str]) -> bool:
        reason_text = "；".join(reasons)
        if best_score >= 7:
            return True
        if best_score >= 6 and gap >= 1:
            return True
        if best_score >= 5 and gap >= 2:
            return True
        if best_score >= 4 and gap >= 3 and any(token in reason_text for token in ("矛盾", "冲票", "带票", "查杀", "对跳", "不一致", "口误")):
            return True
        return False

    def _score_hunter_target(
        self,
        target_id: str,
        memory_records: list[dict[str, Any]],
        public_blob: str,
    ) -> tuple[int, list[str]]:
        score = 0
        reasons: list[str] = []
        target_token = str(target_id)
        token_regex = re.compile(rf"(?<![A-Za-z0-9]){re.escape(target_token)}(?![A-Za-z0-9])", re.IGNORECASE)
        strong_tokens = ("狼", "wolf", "查杀", "对跳", "假", "伪", "冲票", "带票", "带节奏", "矛盾", "冲突", "不一致")
        vote_tokens = ("高票", "最高票", "票王", "带票", "冲票", "带节奏", "冲锋", "连票", "集票", "站边")
        contradiction_tokens = ("矛盾", "冲突", "不一致", "前后", "口误", "说错", "人数不对", "存活人数", "自相矛盾", "打脸")
        claim_tokens = ("查杀", "对跳", "金水", "claim", "身份", "跳身份", "跳警", "跳民")
        death_tokens = ("死亡", "死", "dead", "kill", "被刀", "出局")
        target_hits = 0
        distinct_sources: set[str] = set()
        seen_reasons: set[str] = set()

        for record in memory_records[-12:]:
            if not isinstance(record, Mapping):
                continue
            text = self._format_memory_record(record)
            if not text:
                continue
            text_lower = text.lower()
            ids_in_record = self._record_player_ids(record, ("speaker_id", "speaker", "voter_id", "voter", "player_id", "player", "actor_id", "actor"))
            target_ids_in_record = self._record_player_ids(record, ("target_id", "target", "accused_id", "accused", "vote_target_id", "vote_target"))
            if not (token_regex.search(text) or target_token in ids_in_record or target_token in target_ids_in_record):
                continue
            target_hits += 1
            distinct_sources.update(ids_in_record)
            base = 1
            if any(token in text_lower for token in strong_tokens):
                base += 3
                snippet = self._truncate_text(text, 70)
                if snippet not in seen_reasons:
                    reasons.append(snippet)
                    seen_reasons.add(snippet)
            if any(token in text_lower for token in vote_tokens):
                base += 1
            if any(token in text_lower for token in claim_tokens):
                base += 1
            if any(token in text_lower for token in death_tokens):
                base += 1
            if any(token in text_lower for token in contradiction_tokens):
                base += 2
                snippet = self._truncate_text(text, 70)
                if snippet not in seen_reasons:
                    reasons.append(snippet)
                    seen_reasons.add(snippet)
            if target_token in target_ids_in_record:
                base += 1
            if any(token in text_lower for token in ("好人", "金水", "可信", "站边", "支持")):
                base -= 1
            score += max(base, 0)

        contexts = self._target_context_snippets(public_blob, target_token)
        for snippet in contexts:
            snippet_lower = snippet.lower()
            snippet_score = 1
            if any(token in snippet_lower for token in strong_tokens):
                snippet_score += 2
            if any(token in snippet_lower for token in vote_tokens):
                snippet_score += 1
            if any(token in snippet_lower for token in contradiction_tokens):
                snippet_score += 2
            if any(token in snippet_lower for token in death_tokens):
                snippet_score += 1
            if "alive" in snippet_lower and any(token in snippet_lower for token in ("dead", "死亡", "died", "kill")):
                snippet_score += 2
            if snippet not in seen_reasons:
                reasons.append(self._truncate_text(snippet, 70))
                seen_reasons.add(snippet)
            score += snippet_score

        if target_hits >= 2:
            score += 1
        if len(distinct_sources) >= 2:
            score += 1
        if len(distinct_sources) >= 3:
            score += 1

        if token_regex.search(public_blob):
            score += 1
        if any(token_regex.search(snippet) and any(word in snippet.lower() for word in contradiction_tokens) for snippet in contexts):
            score += 1

        score = max(score, 0)
        if not reasons:
            reasons.append("公开记忆里缺少强关联")
        return score, reasons[:4]

    def _split_reason_text(self, reason_text: str) -> list[str]:
        parts = [self._compact_text(part, 80) for part in re.split(r"[；;]\s*", reason_text) if part.strip()]
        return [part for part in parts if part]

    def _target_context_snippets(self, text: str, target_token: str, *, window: int = 60) -> list[str]:
        if not text:
            return []
        token_regex = re.compile(rf"(?<![A-Za-z0-9]){re.escape(target_token)}(?![A-Za-z0-9])", re.IGNORECASE)
        snippets: list[str] = []
        for match in token_regex.finditer(text):
            start = max(0, match.start() - window)
            end = min(len(text), match.end() + window)
            snippet = self._compact_text(text[start:end], 140)
            if snippet and snippet not in snippets:
                snippets.append(snippet)
            if len(snippets) >= 4:
                break
        return snippets

    def _format_memory_record(self, record: Mapping[str, Any]) -> str:
        if not isinstance(record, Mapping):
            return ""
        bits: list[str] = []
        for key in (
            "round",
            "day",
            "night",
            "phase",
            "kind",
            "speaker_id",
            "speaker",
            "voter_id",
            "voter",
            "player_id",
            "player",
            "target_id",
            "target",
            "role",
            "claim",
            "result",
            "status",
            "event",
        ):
            value = record.get(key)
            if value is None:
                continue
            compact = self._compact_value(value)
            if compact:
                bits.append(f"{key}={compact}")
        text = record.get("text")
        if isinstance(text, str) and text:
            bits.append(self._compact_text(text, 120))
        return self._compact_text(" ".join(bits), 220)

    def _record_player_ids(self, record: Mapping[str, Any], keys: tuple[str, ...]) -> list[str]:
        ids: list[str] = []
        seen: set[str] = set()
        for key in keys:
            for candidate in self._extract_player_ids(record.get(key)):
                if candidate not in seen:
                    seen.add(candidate)
                    ids.append(candidate)
        return ids

    def _extract_player_ids(self, value: Any) -> list[str]:
        ids: list[str] = []
        seen: set[str] = set()
        for candidate in self._extract_player_ids_into(value):
            if candidate not in seen:
                seen.add(candidate)
                ids.append(candidate)
        return ids

    def _extract_player_ids_into(self, value: Any) -> list[str]:
        found: list[str] = []
        if value is None:
            return found
        if isinstance(value, str):
            for match in _PLAYER_ID_TEXT.findall(value):
                if match not in found:
                    found.append(match)
            return found
        if isinstance(value, Mapping):
            for item in value.values():
                found.extend(self._extract_player_ids_into(item))
            return found
        if isinstance(value, (list, tuple, set)):
            for item in value:
                found.extend(self._extract_player_ids_into(item))
            return found
        compact = self._compact_value(value)
        if compact:
            found.extend(self._extract_player_ids_into(compact))
        return found

    def _extract_memory_records(self, value: Any, *, depth: int = 0) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        self._extract_memory_records_into(value, records, depth=depth)
        return records[:18]

    def _extract_memory_records_into(self, value: Any, records: list[dict[str, Any]], *, depth: int) -> None:
        if len(records) >= 18 or depth > 3:
            return
        if isinstance(value, Mapping):
            flat: dict[str, str] = {}
            for key, item in value.items():
                key_text = str(key)
                if key_text in _MEMORY_KEYS or key_text.endswith("_id"):
                    compact = self._compact_value(item)
                    if compact:
                        flat[key_text] = compact
                if isinstance(item, str) and self._is_signal_text(item):
                    compact = self._compact_text(item, 120)
                    if compact:
                        flat.setdefault("text", compact)
                if isinstance(item, (Mapping, list, tuple)):
                    self._extract_memory_records_into(item, records, depth=depth + 1)
                    if len(records) >= 18:
                        break
            record = self._build_memory_record(flat)
            if record:
                records.append(record)
            return
        if isinstance(value, (list, tuple)):
            for item in value[:6]:
                self._extract_memory_records_into(item, records, depth=depth + 1)
                if len(records) >= 18:
                    return
            return
        if isinstance(value, str):
            text = self._compact_text(value)
            if text and self._is_signal_text(text):
                records.append({"kind": "speech", "text": text})

    def _build_memory_record(self, flat: Mapping[str, str]) -> dict[str, Any] | None:
        if not flat:
            return None
        bits: list[str] = []
        for key in (
            "round",
            "day",
            "night",
            "phase",
            "kind",
            "speaker_id",
            "speaker",
            "voter_id",
            "voter",
            "player_id",
            "player",
            "target_id",
            "target",
            "role",
            "claim",
            "result",
            "status",
            "event",
            "reason",
        ):
            value = flat.get(key)
            if value:
                bits.append(f"{key}={value}")
        if flat.get("text"):
            bits.append(flat["text"])
        text = self._compact_text(" ".join(bits), 260)
        if not text:
            return None
        record: dict[str, Any] = {
            "kind": self._infer_memory_kind(flat, text),
            "text": text,
            "tags": sorted(self._infer_memory_tags(flat, text)),
        }
        for key in (
            "round",
            "day",
            "night",
            "phase",
            "speaker_id",
            "speaker",
            "voter_id",
            "voter",
            "player_id",
            "player",
            "target_id",
            "target",
            "role",
            "claim",
            "result",
            "status",
            "event",
            "reason",
        ):
            if key in flat:
                record[key] = flat[key]
        actor_ids = self._record_player_ids(record, ("speaker_id", "speaker", "voter_id", "voter", "player_id", "player"))
        target_ids = self._record_player_ids(record, ("target_id", "target"))
        if actor_ids:
            record["actor_ids"] = actor_ids
        if target_ids:
            record["target_ids"] = target_ids
        return record

    def _infer_memory_kind(self, flat: Mapping[str, str], text: str) -> str:
        kind = str(flat.get("kind") or flat.get("event") or "").lower()
        text_lower = text.lower()
        if any(token in text_lower for token in ("vote", "投票", "冲票", "带票", "票型", "高票")):
            return "vote"
        if any(token in text_lower for token in ("死亡", "dead", "died", "被刀", "出局", "倒下", "kill")):
            return "death"
        if any(token in text_lower for token in ("查杀", "对跳", "claim", "身份", "金水", "跳身份", "跳警", "跳民")):
            return "claim"
        if any(token in text_lower for token in ("发言", "speak", "说", "text", "message", "讨论")):
            return "speech"
        if kind:
            return kind
        return "event"

    def _infer_memory_tags(self, flat: Mapping[str, str], text: str) -> set[str]:
        tags = {self._infer_memory_kind(flat, text)}
        for key in flat:
            if key.endswith("_id"):
                tags.add("player_ref")
            if key in {"speaker", "speaker_id", "voter", "voter_id", "player", "player_id", "target", "target_id"}:
                tags.add(key)
        if any(token in text.lower() for token in ("矛盾", "冲突", "不一致", "口误", "说错", "打脸")):
            tags.add("conflict")
        return tags

    def _is_signal_text(self, text: str) -> bool:
        return bool(_SIGNAL_TEXT.search(text) or _PLAYER_ID_TEXT.search(text) or _CHINESE_CHARACTER.search(text))

    @staticmethod
    def _compact_text(text: str, limit: int = 160) -> str:
        return re.sub(r"\s+", " ", str(text)).strip()[:limit]

    def _compact_value(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, str):
            return self._compact_text(value, 80)
        try:
            dumped = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        except TypeError:
            dumped = repr(value)
        return self._compact_text(dumped, 120)

    @staticmethod
    def _truncate_text(text: str, limit: int) -> str:
        return str(text)[: max(0, int(limit))]

    @staticmethod
    def _nonnegative_int(value: object, *, fallback: int | None) -> int | None:
        if isinstance(value, bool):
            return fallback
        try:
            normalized = int(value)
        except (TypeError, ValueError):
            return fallback
        return normalized if normalized >= 0 else fallback

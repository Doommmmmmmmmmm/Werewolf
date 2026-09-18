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
_SIGNAL_TEXT = re.compile(r"(票|投|狼|好人|金水|查杀|对跳|冲票|带票|带节奏|怀疑|矛盾|身份|死亡|开枪|跳过|pass|vote|claim|wolf|kill|dead|警长|警徽|上警|竞选|验人|归票|同票|发言)", re.IGNORECASE)
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
        self._recent_public_events: deque[dict[str, Any]] = deque(maxlen=72)

    async def observe(self, sync_packet: dict[str, Any]) -> None:
        """接收公开同步，并压缩保存为结构化短记忆。"""

        for event in self._extract_memory_events(sync_packet):
            if event:
                self._recent_public_events.append(event)

    async def decide(self, turn_packet: dict[str, Any]) -> dict[str, Any]:
        if turn_packet["self"]["player_id"] != self.player_id:
            raise ValueError("行动包被发送给了错误玩家")
        private = turn_packet["private_information"]
        if str(private.get("role")) != self.profile.role:
            raise ValueError("Task-Agent 的角色档案与行动包身份不一致")

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
        memory_events = list(self._recent_public_events)
        ranking = self._rank_hunter_targets(request, public_state, memory_events)
        phase = str(
            turn_packet.get("game", {}).get(
                "public_phase",
                turn_packet.get("game", {}).get("phase", request.get("phase", "unknown")),
            )
        )
        lines = [f"阶段：{phase}"]
        alive_count = ranking.get("alive_count")
        if alive_count is not None:
            lines.append(f"存活人数：{alive_count}")
        legal_targets = ranking.get("legal_targets") or []
        if legal_targets:
            lines.append("合法目标：" + "、".join(legal_targets[:8]))
        recent = ranking.get("recent_memory") or []
        if recent:
            lines.append("最近3轮结构记忆：" + " || ".join(recent[:3]))
        ordered = ranking.get("target_ranking") or []
        if ordered:
            top_bits = []
            for item in ordered[:3]:
                reason = str(item.get("reason") or "")[:32]
                top_bits.append(f"{item.get('target_id')}({item.get('score')}:{reason})")
            lines.append("目标排序：" + " > ".join(top_bits))
        if ranking.get("late_window"):
            lines.append("窗口：猎人反应/晚局，优先兑现结构证据")
        if ranking.get("pass_recommended"):
            lines.append("倾向：仅在证据不足且无持续结构压力时跳过")
        elif ranking.get("best_target"):
            lines.append(f"倾向：优先开枪 {ranking['best_target']}")
        return self._truncate_text("\n".join(lines), 900)

    def _rank_hunter_targets(
        self,
        request: Mapping[str, Any],
        public_state: Mapping[str, Any],
        memory_events: list[dict[str, Any]],
    ) -> dict[str, Any]:
        legal_targets = self._collect_target_ids(request)
        alive_count = self._infer_alive_count(public_state)
        late_window = self._is_hunter_reaction_phase(
            request,
            public_state,
            memory_events,
            alive_count=alive_count,
            legal_targets=legal_targets,
        )
        recent_memory = self._summarize_recent_public_memory(memory_events, limit_rounds=3)
        if not legal_targets:
            return {
                "legal_targets": [],
                "recent_memory": recent_memory,
                "target_ranking": [],
                "pass_recommended": True,
                "best_target": None,
                "alive_count": alive_count,
                "late_window": late_window,
            }

        try:
            public_blob_raw = json.dumps(public_state, ensure_ascii=False, separators=(",", ":"))
        except TypeError:
            public_blob_raw = repr(public_state)
        public_blob = self._truncate_text(public_blob_raw, 1200)
        target_ranking: list[dict[str, Any]] = []
        for target_id in legal_targets:
            score, reasons, features = self._score_hunter_target(target_id, memory_events, public_blob)
            target_ranking.append(
                {
                    "target_id": target_id,
                    "score": score,
                    "reason": "；".join(reasons) if reasons else "证据不足",
                    "pressure_rounds": features["pressure_rounds"],
                    "distinct_voters": features["distinct_voters"],
                    "conflict_hits": features["conflict_hits"],
                    "trust_hits": features["trust_hits"],
                    "vote_rounds": features["vote_rounds"],
                }
            )
        target_ranking.sort(
            key=lambda item: (
                -int(item["score"]),
                -int(item.get("pressure_rounds", 0)),
                -int(item.get("distinct_voters", 0)),
                str(item["target_id"]),
            )
        )
        best = target_ranking[0] if target_ranking else None
        second_score = int(target_ranking[1]["score"]) if len(target_ranking) > 1 else 0
        best_score = int(best["score"]) if best is not None else 0
        best_pressure = int(best.get("pressure_rounds", 0)) if best is not None else 0
        best_voters = int(best.get("distinct_voters", 0)) if best is not None else 0
        strong_structure = best is not None and (
            best_pressure >= 2 or int(best.get("conflict_hits", 0)) >= 1 or best_voters >= 2
        )
        pass_cutoff = 4 if late_window else 3
        pass_recommended = best is None or best_score < pass_cutoff
        if best is not None and (strong_structure or best_score >= 6):
            pass_recommended = False
        if late_window and best is not None and best_score >= 4 and (best_pressure >= 1 or best_voters >= 2):
            pass_recommended = False
        if not late_window and best is not None and best_score - second_score < 1 and best_score < 4:
            pass_recommended = True
        return {
            "legal_targets": legal_targets,
            "recent_memory": recent_memory,
            "target_ranking": target_ranking,
            "pass_recommended": pass_recommended,
            "best_target": best["target_id"] if best and not pass_recommended else None,
            "alive_count": alive_count,
            "late_window": late_window,
        }

    def _score_hunter_target(
        self,
        target_id: str,
        memory_events: list[dict[str, Any]],
        public_blob: str,
    ) -> tuple[int, list[str], dict[str, int]]:
        target_token = str(target_id)
        latest_round = max(
            (round_no for round_no in (self._event_round_number(event) for event in memory_events) if round_no is not None),
            default=None,
        )
        vote_round_voters: dict[int, set[str]] = {}
        voter_rounds: dict[str, set[int]] = {}
        conflict_rounds: set[int] = set()
        trust_rounds: set[int] = set()
        death_rounds: set[int] = set()
        mention_rounds: set[int] = set()
        reasons: list[str] = []
        conflict_terms = ("查杀", "对跳", "假", "伪", "冲票", "带票", "带节奏", "矛盾", "狼", "wolf")
        trust_terms = ("金水", "好人", "可信", "站边", "支持", "公认好人", "公开好人")
        vote_terms = ("投", "票", "vote", "vote:", "vote=", "归票", "冲票")

        for event in memory_events:
            if not self._event_mentions_target(event, target_token):
                continue
            round_no = self._event_round_number(event)
            round_key = round_no if round_no is not None else -1
            mention_rounds.add(round_key)
            kind = self._event_kind(event)
            text = self._event_text(event)
            text_lower = text.lower()
            actor = self._event_actor_id(event)
            voter = self._event_voter_id(event)
            target = self._event_target_id(event)
            round_vote = False
            if kind == "vote" or (target == target_token and (voter or any(word in text_lower for word in vote_terms))):
                round_vote = True
                vote_round_voters.setdefault(round_key, set()).add(voter or actor or "unknown")
                if voter or actor:
                    voter_rounds.setdefault(voter or actor or "unknown", set()).add(round_key)
            if kind == "death":
                death_rounds.add(round_key)
            if kind in {"claim", "sheriff"} or any(word in text_lower for word in conflict_terms):
                if any(word in text_lower for word in conflict_terms):
                    conflict_rounds.add(round_key)
            if any(word in text_lower for word in trust_terms):
                trust_rounds.add(round_key)
            if round_vote:
                if any(word in text_lower for word in ("冲票", "带票", "带节奏", "对跳", "查杀")):
                    conflict_rounds.add(round_key)
                if any(word in text_lower for word in ("好人", "金水", "可信")):
                    trust_rounds.add(round_key)

        public_blob_lower = public_blob.lower()
        if self._contains_player_token(public_blob, target_token):
            if any(word in public_blob_lower for word in conflict_terms):
                conflict_rounds.add(-1)
            if any(word in public_blob_lower for word in trust_terms):
                trust_rounds.add(-1)
            if any(word in public_blob_lower for word in ("票王", "高票", "最高票", "冲票", "连票", "连锁", "同票")):
                conflict_rounds.add(-1)

        repeat_voter_bonus = sum(max(0, len(rounds) - 1) for rounds in voter_rounds.values())
        crowd_pressure = sum(
            max(0, len(voters) - 1) * self._recency_weight(latest_round, round_no)
            for round_no, voters in vote_round_voters.items()
        )
        vote_pressure = sum(self._recency_weight(latest_round, round_no) for round_no in vote_round_voters)
        conflict_pressure = sum(self._recency_weight(latest_round, round_no) for round_no in conflict_rounds if round_no >= 0)
        trust_pressure = sum(self._recency_weight(latest_round, round_no) for round_no in trust_rounds if round_no >= 0)
        death_pressure = sum(self._recency_weight(latest_round, round_no) for round_no in death_rounds if round_no >= 0)
        mention_pressure = len({round_no for round_no in mention_rounds if round_no >= 0})

        score = 0
        score += vote_pressure * 2
        score += crowd_pressure * 2
        score += repeat_voter_bonus * 2
        score += conflict_pressure * 3
        score += death_pressure
        score += min(3, mention_pressure)
        score -= min(2, trust_pressure)
        score = max(0, min(12, score))

        if vote_round_voters:
            richest_round, richest_voters = max(
                vote_round_voters.items(),
                key=lambda item: (len(item[1]), self._recency_weight(latest_round, item[0]), -item[0]),
            )
            reasons.append(f"R{richest_round}投票压制{len(richest_voters)}人")
        if repeat_voter_bonus:
            reasons.append(f"同一追票链重复{repeat_voter_bonus + 1}轮")
        if conflict_rounds:
            reasons.append(f"身份/对跳冲突{len({r for r in conflict_rounds if r >= 0})}轮")
        if death_rounds:
            reasons.append(f"死亡链同现{len({r for r in death_rounds if r >= 0})}轮")
        if trust_rounds:
            reasons.append(f"公开好人/金水{len({r for r in trust_rounds if r >= 0})}轮")
        if not reasons:
            reasons.append("公开记忆里缺少强结构压力")

        features = {
            "pressure_rounds": len({r for r in vote_round_voters if r >= 0} | {r for r in conflict_rounds if r >= 0}),
            "distinct_voters": len({voter for voters in vote_round_voters.values() for voter in voters if voter != "unknown"}),
            "conflict_hits": len({r for r in conflict_rounds if r >= 0}),
            "trust_hits": len({r for r in trust_rounds if r >= 0}),
            "vote_rounds": len({r for r in vote_round_voters if r >= 0}),
        }
        return score, reasons[:4], features

    def _collect_target_ids(self, request: Mapping[str, Any]) -> list[str]:
        target_ids: list[str] = []
        seen: set[str] = set()
        raw_actions = request.get("allowed_actions")
        if not isinstance(raw_actions, list):
            return target_ids
        for allowed in raw_actions:
            if not isinstance(allowed, Mapping):
                continue
            raw_target_ids = allowed.get("target_ids")
            if not isinstance(raw_target_ids, list):
                continue
            for target_id in raw_target_ids:
                if target_id is None:
                    continue
                text = str(target_id)
                if text not in seen:
                    seen.add(text)
                    target_ids.append(text)
        return target_ids

    def _extract_memory_events(self, value: Any, *, depth: int = 0, round_hint: int | None = None) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        self._extract_memory_events_into(value, events, depth=depth, round_hint=round_hint)
        return events[:24]

    def _extract_memory_events_into(
        self,
        value: Any,
        events: list[dict[str, Any]],
        *,
        depth: int,
        round_hint: int | None,
    ) -> None:
        if len(events) >= 24 or depth > 3:
            return
        if isinstance(value, Mapping):
            entry: dict[str, Any] = {}
            nested_round_hint = round_hint
            raw_round = value.get("round")
            if raw_round is not None:
                nested_round_hint = self._normalize_round_value(raw_round, fallback=round_hint)
                if nested_round_hint is not None:
                    entry["round"] = nested_round_hint
            for key, item in value.items():
                key_text = str(key)
                if key_text == "round":
                    continue
                if key_text in _MEMORY_KEYS or key_text.endswith("_id"):
                    compact = self._compact_value(item)
                    if compact:
                        entry[key_text] = compact
                if isinstance(item, (Mapping, list, tuple)):
                    self._extract_memory_events_into(item, events, depth=depth + 1, round_hint=nested_round_hint)
                elif isinstance(item, str) and self._is_signal_text(item):
                    compact = self._compact_text(item)
                    if compact:
                        entry.setdefault("text", compact)
            if self._looks_like_memory_event(entry):
                events.append(self._normalize_memory_event(entry))
            return
        if isinstance(value, (list, tuple)):
            for item in value[:8]:
                self._extract_memory_events_into(item, events, depth=depth + 1, round_hint=round_hint)
                if len(events) >= 24:
                    return
            return
        if isinstance(value, str):
            text = self._compact_text(value)
            if text and self._is_signal_text(text):
                event: dict[str, Any] = {"text": text}
                if round_hint is not None:
                    event["round"] = round_hint
                events.append(event)

    def _looks_like_memory_event(self, entry: Mapping[str, Any]) -> bool:
        if not entry:
            return False
        signal_keys = (
            "kind",
            "action",
            "event",
            "result",
            "status",
            "speaker",
            "speaker_id",
            "player",
            "player_id",
            "voter",
            "voter_id",
            "target",
            "target_id",
            "claim",
            "vote",
            "votes",
            "alive",
            "dead",
            "revealed",
            "public_role",
            "reason",
            "winner",
            "text",
            "message",
        )
        return any(key in entry for key in signal_keys)

    def _normalize_memory_event(self, entry: Mapping[str, Any]) -> dict[str, Any]:
        normalized: dict[str, Any] = {}
        for key, value in entry.items():
            if value in (None, ""):
                continue
            if key == "round":
                round_no = self._normalize_round_value(value)
                if round_no is not None:
                    normalized[key] = round_no
                continue
            if key in {"text", "message", "reason", "claim"}:
                normalized[key] = self._compact_text(value, 180)
            else:
                normalized[key] = self._compact_value(value)
        return normalized

    def _summarize_recent_public_memory(
        self,
        memory_events: list[dict[str, Any]],
        *,
        limit_rounds: int = 3,
    ) -> list[str]:
        grouped: dict[int, list[dict[str, Any]]] = {}
        loose_events: list[dict[str, Any]] = []
        for event in memory_events:
            round_no = self._event_round_number(event)
            if round_no is None:
                loose_events.append(event)
            else:
                grouped.setdefault(round_no, []).append(event)
        summaries: list[str] = []
        for round_no in sorted(grouped, reverse=True)[:limit_rounds]:
            summary = self._summarize_round_events(round_no, grouped[round_no])
            if summary:
                summaries.append(summary)
        if not summaries and loose_events:
            summary = self._summarize_round_events(None, loose_events[-6:])
            if summary:
                summaries.append(summary)
        return summaries

    def _summarize_round_events(self, round_no: int | None, events: list[dict[str, Any]]) -> str:
        vote_pairs: list[str] = []
        death_bits: list[str] = []
        claim_bits: list[str] = []
        sheriff_bits: list[str] = []
        conflict_bits: list[str] = []
        for event in events:
            kind = self._event_kind(event)
            text = self._event_text(event)
            actor = self._event_actor_id(event)
            voter = self._event_voter_id(event)
            target = self._event_target_id(event)
            text_lower = text.lower()
            if kind == "vote" or (voter and target):
                if voter and target:
                    vote_pairs.append(f"{voter}→{target}")
            elif kind == "death":
                death_bits.append(target or actor or self._compact_text(text, 18))
            elif kind == "claim":
                claim_bits.append(self._compact_text(text, 28))
            elif kind == "sheriff":
                sheriff_bits.append(self._compact_text(text, 28))
            elif any(word in text_lower for word in ("查杀", "对跳", "冲票", "带票", "带节奏", "矛盾", "反转", "换边")):
                conflict_bits.append(self._compact_text(text, 28))

        vote_pairs = list(dict.fromkeys(vote_pairs))
        death_bits = list(dict.fromkeys(bit for bit in death_bits if bit))
        claim_bits = list(dict.fromkeys(bit for bit in claim_bits if bit))
        sheriff_bits = list(dict.fromkeys(bit for bit in sheriff_bits if bit))
        conflict_bits = list(dict.fromkeys(bit for bit in conflict_bits if bit))
        parts: list[str] = []
        if round_no is not None:
            parts.append(f"R{round_no}")
        if vote_pairs:
            parts.append("投票:" + "、".join(vote_pairs[:3]))
        if death_bits:
            parts.append("死亡:" + "、".join(death_bits[:2]))
        if claim_bits:
            parts.append("claim:" + "、".join(claim_bits[:2]))
        if sheriff_bits:
            parts.append("警长:" + "、".join(sheriff_bits[:2]))
        if conflict_bits:
            parts.append("对线:" + "、".join(conflict_bits[:1]))
        if not parts and events:
            parts.append(self._compact_text(self._event_text(events[-1]), 60))
        return "；".join(parts)

    def _event_round_number(self, event: Mapping[str, Any]) -> int | None:
        return self._normalize_round_value(event.get("round"))

    def _event_kind(self, event: Mapping[str, Any]) -> str:
        kind = self._compact_text(event.get("kind") or event.get("action") or event.get("event") or "", 40).lower()
        text = self._event_text(event).lower()
        if any(word in kind or word in text for word in ("death", "dead", "kill", "被刀", "出局", "刀", "倒牌")):
            return "death"
        if any(word in kind or word in text for word in ("sheriff", "警长", "警徽", "上警", "竞选", "验人")):
            return "sheriff"
        if any(word in kind or word in text for word in ("claim", "跳身份", "自证", "金水", "查杀", "身份", "role")):
            return "claim"
        if any(word in kind or word in text for word in ("vote", "投票", "投", "票", "归票", "冲票")):
            return "vote"
        return "dialogue"

    def _event_text(self, event: Mapping[str, Any]) -> str:
        parts: list[str] = []
        for key in ("kind", "action", "event", "result", "status", "role", "claim", "text", "message", "reason", "winner", "vote", "votes", "speaker", "speaker_id", "player", "player_id", "voter", "voter_id", "target", "target_id", "public_role"):
            value = event.get(key)
            if value not in (None, ""):
                parts.append(f"{key}={self._compact_value(value, 80)}")
        return " ".join(parts)

    def _event_actor_id(self, event: Mapping[str, Any]) -> str:
        for key in ("speaker_id", "speaker", "player_id", "player", "voter_id", "voter"):
            value = event.get(key)
            if value not in (None, ""):
                return self._compact_text(value, 32)
        return ""

    def _event_voter_id(self, event: Mapping[str, Any]) -> str:
        for key in ("voter_id", "voter", "speaker_id", "speaker"):
            value = event.get(key)
            if value not in (None, ""):
                return self._compact_text(value, 32)
        return ""

    def _event_target_id(self, event: Mapping[str, Any]) -> str:
        for key in ("target_id", "target", "player_id", "player"):
            value = event.get(key)
            if value not in (None, ""):
                return self._compact_text(value, 32)
        return ""

    def _event_mentions_target(self, event: Mapping[str, Any], target_token: str) -> bool:
        if not target_token:
            return False
        for key in ("target_id", "target", "player_id", "player", "voter_id", "voter", "speaker_id", "speaker"):
            if self._value_matches_token(event.get(key), target_token):
                return True
        text = self._event_text(event)
        return self._contains_player_token(text, target_token)

    def _contains_player_token(self, text: str, token: str) -> bool:
        if not text or not token:
            return False
        return re.search(rf"(?<!\w){re.escape(token)}(?!\w)", str(text), re.IGNORECASE) is not None

    def _value_matches_token(self, value: Any, token: str) -> bool:
        if value in (None, ""):
            return False
        return self._compact_text(value, 80).lower() == token.lower()

    def _recency_weight(self, latest_round: int | None, round_no: int | None) -> int:
        if latest_round is None or round_no is None or round_no < 0:
            return 1
        delta = max(0, latest_round - round_no)
        return max(1, 4 - min(delta, 3))

    def _infer_alive_count(self, value: Any, *, depth: int = 0) -> int | None:
        if depth > 3:
            return None
        if isinstance(value, Mapping):
            for key in ("alive_count", "alive_num", "alive_players", "alive", "survivors", "living_count"):
                raw = value.get(key)
                if raw is None:
                    continue
                if isinstance(raw, (list, tuple, set)):
                    count = sum(1 for item in raw if item not in (None, ""))
                    if count:
                        return count
                else:
                    count = self._nonnegative_int(raw, fallback=None)
                    if count is not None:
                        return count
            for item in value.values():
                count = self._infer_alive_count(item, depth=depth + 1)
                if count is not None:
                    return count
            return None
        if isinstance(value, (list, tuple, set)):
            count = sum(1 for item in value if item not in (None, ""))
            return count or None
        return self._nonnegative_int(value, fallback=None)

    def _is_hunter_reaction_phase(
        self,
        request: Mapping[str, Any],
        public_state: Mapping[str, Any],
        memory_events: list[dict[str, Any]],
        *,
        alive_count: int | None,
        legal_targets: list[str],
    ) -> bool:
        phase_bits = " ".join(
            str(part)
            for part in (
                request.get("phase"),
                request.get("channel"),
                public_state.get("phase"),
                public_state.get("public_phase"),
            )
            if part not in (None, "")
        ).lower()
        if any(word in phase_bits for word in ("hunter", "猎人", "reaction", "react", "last_words", "lastword", "death_response", "死亡反应")):
            return True
        if any("猎人反应" in self._event_text(event) or "HUNTER_REACTIONS_QUEUED" in self._event_text(event) for event in memory_events[-4:]):
            return True
        if alive_count is not None and alive_count <= 7:
            return True
        if len(legal_targets) <= 3:
            return True
        return False
    def _is_signal_text(self, text: str) -> bool:
        return bool(_SIGNAL_TEXT.search(text) or _PLAYER_ID_TEXT.search(text) or _CHINESE_CHARACTER.search(text))

    @staticmethod
    def _compact_text(text: str, limit: int = 160) -> str:
        return re.sub(r"\s+", " ", str(text)).strip()[:limit]

    def _compact_value(self, value: Any, limit: int = 120) -> str:
        if value is None:
            return ""
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, str):
            return self._compact_text(value, min(80, limit))
        try:
            dumped = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        except TypeError:
            dumped = repr(value)
        return self._compact_text(dumped, limit)

    @staticmethod
    def _truncate_text(text: str, limit: int) -> str:
        return str(text)[: max(0, int(limit))]

    @staticmethod
    def _normalize_round_value(value: Any, fallback: int | None = None) -> int | None:
        if value is None:
            return fallback
        if isinstance(value, bool):
            return fallback
        if isinstance(value, (int, float)):
            normalized = int(value)
            return normalized if normalized >= 0 else fallback
        text = str(value).strip()
        if not text:
            return fallback
        match = re.search(r"\d+", text)
        if match:
            try:
                normalized = int(match.group(0))
            except ValueError:
                return fallback
            return normalized if normalized >= 0 else fallback
        return fallback

    @staticmethod
    def _nonnegative_int(value: object, *, fallback: int | None) -> int | None:
        if isinstance(value, bool):
            return fallback
        try:
            normalized = int(value)
        except (TypeError, ValueError):
            return fallback
        return normalized if normalized >= 0 else fallback

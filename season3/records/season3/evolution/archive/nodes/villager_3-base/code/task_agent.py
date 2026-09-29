"""最小可运行的角色 Task-Agent。

每次行动从一个新的模型会话开始。模型默认只收到当前状态；本轮对话只能通过唯一的
受限工具按需读取。这里不包含策略进化、长期记忆、外部检索或其他 Harness。
"""

from __future__ import annotations

import asyncio
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
_PLAYER_ID_SUFFIX = re.compile(r"(\d+)$")

CURRENT_ROUND_DIALOGUE_TOOL_NAME = "read_current_round_dialogue"
DEFAULT_MAX_TOOL_CALLS_PER_DECISION = 5
DEFAULT_MAX_TOOL_RESULT_TOKENS = 1000
DEFAULT_MAX_PROMPT_CHARS = 12000
# 这是 Task-Agent 层的“纠错重试”次数：首次模型调用之外，最多再请求两次。
# 传输层的 HTTP/网络重试仍由 ModelClient 的 MODEL_MAX_RETRIES 单独控制。
DEFAULT_MAX_DECISION_RETRIES = 2
_SPEECH_ACTION_KINDS = frozenset({"speak", "last_words"})

_PUBLIC_SNAPSHOT_LIMIT = 18
_PUBLIC_NOTE_CHAR_LIMIT = 140
_PUBLIC_BLOCK_CHAR_LIMIT = 1600
_PHASE_BLOCK_CHAR_LIMIT = 800


CURRENT_ROUND_DIALOGUE_TOOL: dict[str, Any] = {
    "name": CURRENT_ROUND_DIALOGUE_TOOL_NAME,
    "description": "读取当前昼夜轮次中、当前玩家依法可见的已发生发言。",
    "parameters": {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    },
}


def render_villager_decision_checklist(request: Mapping[str, Any]) -> str:
    """给平民用的短决策清单，只作为提示文本，不引入任何隐藏状态。"""

    phase = str(request.get("phase") or request.get("channel") or request.get("kind") or "")
    phase_lower = phase.lower()
    allowed_actions = request.get("allowed_actions")
    allowed_kinds = {
        str(item.get("kind") or "").lower()
        for item in allowed_actions
        if isinstance(item, Mapping)
    } if isinstance(allowed_actions, list) else set()
    lines = ["【平民决策清单】先分系统事实 / 玩家声明 / 自己推测；结合上一轮公共台账，不要把当前一轮聊天当成全部历史。投票前至少比较两个候选。"]
    has_vote = any(key in phase_lower for key in ("vote", "投票")) or any("vote" in kind for kind in allowed_kinds)
    has_speak = any(key in phase_lower for key in ("speak", "discussion", "发言", "白天", "day")) or any(
        kind in {"speak", "discussion", "day_speak"} for kind in allowed_kinds
    )
    has_last_words = any(key in phase_lower for key in ("last", "遗言")) or "last_words" in allowed_kinds
    has_sheriff = any(key in phase_lower for key in ("sheriff", "警长", "警徽")) or any(
        key in allowed_kinds for key in ("sheriff_vote", "sheriff")
    )
    if has_vote:
        lines.append("投票时优先保护可信金水、持续报验且查杀被票型或发言支持的预言家、强神声明者和警徽传递链；只有硬对跳、明显矛盾、查杀命中或事实错误时才推翻。")
        lines.append("优先找未被金水覆盖且存在票型异常、事实错误、被查杀或爆狼线索指向的人。")
    if has_speak:
        lines.append("发言至少点出一个关注对象和一个暂不投对象，并说明依据哪条公开信息。")
    if has_last_words:
        lines.append("遗言只留公开事实、票型、查验链和怀疑对象，不把推测说成系统事实。")
    if has_sheriff:
        lines.append("警长/警徽相关回合优先看报验链是否清晰一致；接徽时继承公开遗产但不要把它当系统确认。")
    lines.append("残局或持警徽时，额外复盘存活名单、已死名单、可信查验链和灰区名单，再决定出谁。")
    return "\n".join(lines)


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
        self._public_ledger_game_id: str | None = None
        self._public_ledger: dict[str, Any] = {}
        self._reset_public_ledger(None)

    async def observe(self, sync_packet: dict[str, Any]) -> None:
        """接收公开同步并维护一个仅含公开事实的轻量台账。"""

        if not isinstance(sync_packet, Mapping):
            return
        game_id = self._extract_game_id(sync_packet)
        if game_id is not None and game_id != self._public_ledger_game_id:
            self._reset_public_ledger(game_id)
        self._ingest_public_packet(sync_packet)

    async def decide(self, turn_packet: dict[str, Any]) -> dict[str, Any]:
        if turn_packet["self"]["player_id"] != self.player_id:
            raise ValueError("行动包被发送给了错误玩家")
        private = turn_packet["private_information"]
        if str(private.get("role")) != self.profile.role:
            raise ValueError("Task-Agent 的角色档案与行动包身份不一致")

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
                "instruction": self._turn_instruction(turn_packet, feedback),
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

    def _turn_instruction(self, turn_packet: Mapping[str, Any], feedback: str) -> str:
        request = turn_packet["request"]
        checklist = render_villager_decision_checklist(request)
        summary = self._render_turn_summary(turn_packet, request)
        feedback_text = f"上一次输出未通过校验：{feedback}" if feedback else ""
        validation_feedback = "\n".join(
            part for part in (summary, feedback_text, checklist) if part
        )
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

    @staticmethod
    def _nonnegative_int(value: object, *, fallback: int | None) -> int | None:
        if isinstance(value, bool):
            return fallback
        try:
            normalized = int(value)
        except (TypeError, ValueError):
            return fallback
        return normalized if normalized >= 0 else fallback

    def _reset_public_ledger(self, game_id: str | None) -> None:
        self._public_ledger_game_id = game_id
        self._public_ledger = {
            "alive_ids": set(),
            "dead_ids": set(),
            "known_player_ids": set(),
            "recent_events": [],
            "public_claims": [],
            "vote_snapshots": [],
            "sheriff_changes": [],
            "badge_transfers": [],
            "death_notices": [],
        }

    def _ingest_public_packet(self, payload: Any) -> None:
        notes = self._extract_public_notes(payload)
        self._merge_public_notes(notes)
        player_ids = self._extract_player_ids(payload)
        self._public_ledger["known_player_ids"].update(player_ids)

        if isinstance(payload, Mapping):
            alive = self._collect_id_collection(payload, self._ALIVE_KEYS())
            dead = self._collect_id_collection(payload, self._DEAD_KEYS())
            self._public_ledger["alive_ids"].update(alive)
            self._public_ledger["dead_ids"].update(dead)

    def _merge_public_notes(self, notes: list[str]) -> None:
        for note in notes:
            if not note:
                continue
            self._append_bounded(self._public_ledger["recent_events"], note, _PUBLIC_SNAPSHOT_LIMIT)
            lower = note.lower()
            if any(key in lower for key in ("vote", "voter=", "target=")):
                self._append_bounded(self._public_ledger["vote_snapshots"], note, _PUBLIC_SNAPSHOT_LIMIT)
            if any(key in lower for key in ("sheriff", "警长", "badge", "警徽")):
                self._append_bounded(self._public_ledger["sheriff_changes"], note, _PUBLIC_SNAPSHOT_LIMIT)
            if any(key in lower for key in ("badge", "警徽", "transfer", "转移")):
                self._append_bounded(self._public_ledger["badge_transfers"], note, _PUBLIC_SNAPSHOT_LIMIT)
            if any(key in lower for key in ("death", "dead", "出局", "死亡", "刀")):
                self._append_bounded(self._public_ledger["death_notices"], note, _PUBLIC_SNAPSHOT_LIMIT)
            if any(key in lower for key in ("claim", "自称", "预言家", "女巫", "猎人", "守卫", "金水", "查杀", "查验", "警徽")):
                self._append_bounded(self._public_ledger["public_claims"], note, _PUBLIC_SNAPSHOT_LIMIT)

    def _render_turn_summary(self, turn_packet: Mapping[str, Any], request: Mapping[str, Any]) -> str:
        public_state = turn_packet.get("public_state")
        turn_notes = self._extract_public_notes(public_state)
        ledger_alive = set(self._public_ledger["alive_ids"])
        ledger_dead = set(self._public_ledger["dead_ids"])
        ledger_known = set(self._public_ledger["known_player_ids"])
        current_alive = self._extract_player_ids(public_state, include_collections=True)
        combined_alive = self._sort_player_ids((ledger_alive | current_alive) - ledger_dead)
        combined_dead = self._sort_player_ids(ledger_dead | self._extract_dead_from_notes(turn_notes))
        combined_known = self._sort_player_ids((ledger_known | current_alive | ledger_alive | ledger_dead) - {self.player_id})
        candidate_pool = self._candidate_pool(request, combined_alive, combined_known)

        ledger_lines = ["【公共台账】"]
        ledger_lines.append("存活: " + self._join_ids(combined_alive, limit=10))
        ledger_lines.append("已死: " + self._join_ids(combined_dead, limit=10))
        claim_block = self._collect_claim_block(turn_notes)
        if claim_block:
            ledger_lines.append("查验/自述: " + claim_block)
        gray_pool = [pid for pid in candidate_pool if pid not in combined_dead]
        if gray_pool:
            ledger_lines.append("灰区: " + self._join_ids(gray_pool[:6], limit=6))

        phase = str(request.get("phase") or request.get("channel") or request.get("kind") or "")
        phase_lower = phase.lower()
        allowed_actions = request.get("allowed_actions")
        allowed_kinds = {
            str(item.get("kind") or "").lower()
            for item in allowed_actions
            if isinstance(item, Mapping)
        } if isinstance(allowed_actions, list) else set()
        is_vote = any(key in phase_lower for key in ("vote", "投票")) or any("vote" in kind for kind in allowed_kinds)
        if is_vote:
            phase_lines = self._render_vote_summary(candidate_pool, combined_alive, combined_dead, turn_notes)
        else:
            phase_lines = self._render_speak_summary(candidate_pool, combined_alive, combined_dead, turn_notes)

        parts = ["\n".join(ledger_lines), phase_lines]
        summary = "\n".join(part for part in parts if part)
        return self._trim_text(summary, _PUBLIC_BLOCK_CHAR_LIMIT + _PHASE_BLOCK_CHAR_LIMIT)

    def _render_vote_summary(
        self,
        candidate_pool: list[str],
        alive_ids: list[str],
        dead_ids: list[str],
        notes: list[str],
    ) -> str:
        ranked = self._rank_candidates(candidate_pool, notes, dead_ids)
        if len(ranked) < 2 and alive_ids:
            fallback_pool = [pid for pid in alive_ids if pid not in dead_ids and pid not in {item[0] for item in ranked}]
            ranked.extend(self._rank_candidates(fallback_pool, notes, dead_ids))
        while len(ranked) < 2 and candidate_pool:
            ranked.append((candidate_pool[0], 0, "公开信息不足"))
        ranked = ranked[:2]
        lines = ["【本轮投票参考】"]
        for index, (candidate, score, reason) in enumerate(ranked, start=1):
            lines.append(f"{index}. {candidate} | 分数={score} | {reason}")
        if len(ranked) >= 2:
            lines.append(f"优先比较：{ranked[0][0]} vs {ranked[1][0]}")
        else:
            lines.append("优先比较至少两个候选；当前候选不足时，用公开票型与查验链补足。")
        return self._trim_text("\n".join(lines), _PHASE_BLOCK_CHAR_LIMIT)

    def _render_speak_summary(
        self,
        candidate_pool: list[str],
        alive_ids: list[str],
        dead_ids: list[str],
        notes: list[str],
    ) -> str:
        ranked = self._rank_candidates(candidate_pool, notes, dead_ids)
        if not ranked and alive_ids:
            ranked = self._rank_candidates(alive_ids, notes, dead_ids)
        if not ranked:
            ranked = [("信息不足", 0, "公开信息不足")]
        focus = ranked[0]
        avoid = self._least_suspicious_candidate(candidate_pool or alive_ids, notes, dead_ids)
        lines = ["【本轮发言参考】"]
        lines.append(f"关注: {focus[0]} | {focus[2]}")
        if avoid:
            lines.append(f"暂不投: {avoid[0]} | {avoid[2]}")
        else:
            lines.append("暂不投: 先保留公开链中更稳的一人，直到票型或查验出现新变化。")
        return self._trim_text("\n".join(lines), _PHASE_BLOCK_CHAR_LIMIT)

    def _rank_candidates(
        self,
        candidate_pool: list[str],
        notes: list[str],
        dead_ids: list[str],
    ) -> list[tuple[str, int, str]]:
        dead_set = set(dead_ids)
        ranked: list[tuple[str, int, str]] = []
        for candidate in self._sort_player_ids(candidate_pool):
            if candidate in dead_set or candidate == self.player_id:
                continue
            score, reasons = self._score_candidate(candidate, notes)
            if reasons:
                reason_text = "；".join(reasons[:2])
            else:
                reason_text = "公开信息不足"
            ranked.append((candidate, score, reason_text))
        ranked.sort(key=lambda item: (-item[1], item[0]))
        return ranked

    def _least_suspicious_candidate(
        self,
        candidate_pool: list[str],
        notes: list[str],
        dead_ids: list[str],
    ) -> tuple[str, int, str] | None:
        ranked = self._rank_candidates(candidate_pool, notes, dead_ids)
        if not ranked:
            return None
        ranked.sort(key=lambda item: (item[1], item[0]))
        return ranked[0]

    def _score_candidate(self, candidate: str, notes: list[str]) -> tuple[int, list[str]]:
        lower_candidate = candidate.lower()
        score = 0
        reasons: list[str] = []
        suspicious_terms = ("查杀", "对跳", "爆狼", "矛盾", "票型异常", "反水", "不一致", "假", "冲票", "狼")
        protect_terms = ("金水", "可信", "报验链", "警徽", "预言家", "验过", "已验", "好人链", "强神")
        for note in notes:
            lower = note.lower()
            if lower_candidate not in lower:
                continue
            score += 1
            compact = self._trim_text(note, 80)
            if any(term in note for term in suspicious_terms):
                score += 2
                reasons.append(f"疑点: {compact}")
            elif any(term in note for term in protect_terms):
                score -= 2
                reasons.append(f"保护: {compact}")
            elif "vote" in lower or "票" in note:
                score += 1
                reasons.append(f"票型: {compact}")
            elif any(term in note for term in ("claim", "自称", "发言", "说", "跳")):
                reasons.append(f"发言: {compact}")
            else:
                reasons.append(compact)
        if not reasons:
            reasons.append("公开信息不足")
        return score, reasons

    def _candidate_pool(
        self,
        request: Mapping[str, Any],
        alive_ids: list[str],
        known_ids: list[str],
    ) -> list[str]:
        pool: set[str] = set(alive_ids)
        pool.update(known_ids)
        raw_actions = request.get("allowed_actions")
        if isinstance(raw_actions, list):
            for action in raw_actions:
                if not isinstance(action, Mapping):
                    continue
                target_ids = action.get("target_ids")
                if isinstance(target_ids, list):
                    for target_id in target_ids:
                        if isinstance(target_id, (str, int)) and not isinstance(target_id, bool):
                            pool.add(str(target_id))
        pool.discard(self.player_id)
        pool.difference_update(self._public_ledger["dead_ids"])
        return self._sort_player_ids(list(pool))

    def _collect_claim_block(self, notes: list[str]) -> str:
        claims = []
        for note in notes:
            if any(term in note for term in ("claim", "自称", "预言家", "女巫", "猎人", "守卫", "金水", "查杀", "查验", "警徽")):
                claims.append(self._trim_text(note, 90))
        if not claims:
            claims = [self._trim_text(item, 90) for item in self._public_ledger["public_claims"][:3]]
        return " | ".join(claims[:3])

    def _extract_public_notes(self, payload: Any) -> list[str]:
        notes: list[str] = []
        self._walk_public_payload(payload, notes, depth=0)
        return self._dedupe_preserve_order(notes)

    def _walk_public_payload(self, payload: Any, notes: list[str], depth: int) -> None:
        if depth > 4:
            return
        if isinstance(payload, Mapping):
            snapshot = self._snapshot_mapping(payload)
            if snapshot:
                notes.append(snapshot)
            for value in payload.values():
                self._walk_public_payload(value, notes, depth + 1)
            return
        if isinstance(payload, list):
            for item in payload[:25]:
                self._walk_public_payload(item, notes, depth + 1)

    def _snapshot_mapping(self, mapping: Mapping[str, Any]) -> str | None:
        lower_map = {str(key).lower(): value for key, value in mapping.items()}
        event = self._first_scalar(lower_map, ("event", "event_type", "type", "kind", "action"))
        event_lower = event.lower() if event else ""
        if not event_lower:
            event_lower = ""

        vote_like = any(
            key in lower_map
            for key in (
                "voter_id",
                "voter",
                "vote_target",
                "target_id",
                "target",
                "ballot",
                "vote",
            )
        ) or "vote" in event_lower
        claim_like = any(
            key in lower_map
            for key in (
                "speaker_id",
                "claimer_id",
                "claim",
                "role_claim",
                "speech",
                "text",
                "content",
                "message",
                "reason",
            )
        ) or any(term in event_lower for term in ("claim", "speak", "speech", "say", "发言", "自称"))
        sheriff_like = any(
            key in lower_map
            for key in (
                "sheriff_id",
                "badge_holder",
                "badge_holder_id",
                "badge_from",
                "badge_to",
                "badge",
            )
        ) or any(term in event_lower for term in ("sheriff", "badge", "警徽", "警长"))
        death_like = any(key in lower_map for key in ("dead", "death", "dead_id", "dead_player_id")) or any(
            term in event_lower for term in ("death", "dead", "kill", "out", "死亡", "出局")
        )

        if not (vote_like or claim_like or sheriff_like or death_like):
            # 仍然允许纯列表型公共事实，例如 alive / dead 的集合，会在上层单独汇总。
            return None

        round_text = self._first_scalar(lower_map, ("round", "day", "turn", "phase", "channel"))
        player = self._first_scalar(lower_map, ("player_id", "speaker_id", "voter_id", "claimer_id", "actor_id", "source_id"))
        target = self._first_scalar(lower_map, ("target_id", "target", "vote_target", "to_id", "suspect_id"))
        source = self._first_scalar(lower_map, ("from_id", "source_id", "badge_from"))
        holder = self._first_scalar(lower_map, ("badge_holder", "badge_holder_id", "sheriff_id"))
        role = self._first_scalar(lower_map, ("role", "claim_role", "claimed_role"))
        text = self._first_scalar(lower_map, ("text", "content", "message", "reason", "speech"))

        parts: list[str] = []
        if event:
            parts.append(event_lower or event.lower())
        if vote_like:
            parts.append(f"vote voter={player or self._first_scalar(lower_map, ('voter',)) or '?'} target={target or '?'}")
        elif sheriff_like:
            parts.append(
                f"sheriff holder={holder or '?'} from={source or '?'} to={target or '?'}"
            )
        elif death_like:
            parts.append(f"death player={player or self._first_scalar(lower_map, ('dead_id', 'dead_player_id')) or '?'}")
        elif claim_like:
            parts.append(f"claim speaker={player or '?'}")
            if role:
                parts.append(f"role={role}")
        if round_text:
            parts.append(f"at={round_text}")
        if text:
            cleaned = self._trim_text(text, _PUBLIC_NOTE_CHAR_LIMIT)
            parts.append(f"text={cleaned}")
        snapshot = " ".join(part for part in parts if part).strip()
        return self._trim_text(snapshot, _PUBLIC_NOTE_CHAR_LIMIT) if snapshot else None

    def _extract_player_ids(self, payload: Any, *, include_collections: bool = False) -> set[str]:
        ids: set[str] = set()

        def walk(obj: Any) -> None:
            if isinstance(obj, Mapping):
                lower_map = {str(key).lower(): value for key, value in obj.items()}
                for key, value in lower_map.items():
                    if key in self._player_id_keys() and self._is_scalar_id(value):
                        ids.add(str(value))
                    if include_collections and key in self._collection_id_keys():
                        ids.update(self._as_id_set(value))
                for value in obj.values():
                    walk(value)
            elif isinstance(obj, list):
                for item in obj[:25]:
                    walk(item)

        walk(payload)
        ids.discard(self.player_id)
        return ids

    def _collect_id_collection(self, mapping: Mapping[str, Any], allowed_keys: set[str]) -> set[str]:
        collected: set[str] = set()
        lower_map = {str(key).lower(): value for key, value in mapping.items()}
        for key, value in lower_map.items():
            if key in allowed_keys:
                collected.update(self._as_id_set(value))
        return collected

    def _extract_dead_from_notes(self, notes: list[str]) -> set[str]:
        dead: set[str] = set()
        for note in notes:
            lower = note.lower()
            if any(term in lower for term in ("death", "dead", "死亡", "出局", "淘汰")):
                dead.update(self._ids_from_text(note))
        return dead

    def _ids_from_text(self, text: str) -> set[str]:
        ids: set[str] = set()
        for token in re.split(r"[\s,，;；|/\\]+", text):
            token = token.strip("()[]{}<>：:=，。,.、")
            if token and self._looks_like_player_id(token):
                ids.add(token)
        return ids

    def _as_id_set(self, value: Any) -> set[str]:
        if isinstance(value, list):
            return {str(item) for item in value if self._is_scalar_id(item)}
        if isinstance(value, tuple):
            return {str(item) for item in value if self._is_scalar_id(item)}
        if self._is_scalar_id(value):
            return {str(value)}
        return set()

    def _merge_unique(self, items: list[str], values: list[str]) -> None:
        for value in values:
            if value and value not in items:
                items.append(value)

    def _append_bounded(self, items: list[str], value: str, limit: int) -> None:
        if not value:
            return
        if value in items:
            return
        items.append(value)
        if len(items) > limit:
            del items[: len(items) - limit]

    def _first_scalar(self, mapping: Mapping[str, Any], keys: tuple[str, ...]) -> str:
        for key in keys:
            value = mapping.get(key)
            if self._is_scalar_text(value):
                return self._trim_text(str(value), _PUBLIC_NOTE_CHAR_LIMIT)
        return ""

    @staticmethod
    def _is_scalar_text(value: Any) -> bool:
        return isinstance(value, (str, int, float)) and not isinstance(value, bool)

    def _is_scalar_id(self, value: Any) -> bool:
        if isinstance(value, bool):
            return False
        if isinstance(value, int):
            return True
        if isinstance(value, str):
            return self._looks_like_player_id(value)
        return False

    def _trim_text(self, text: str, limit: int) -> str:
        normalized = " ".join(str(text).split())
        if len(normalized) <= limit:
            return normalized
        if limit <= 3:
            return normalized[:limit]
        return normalized[: limit - 1] + "…"

    def _dedupe_preserve_order(self, items: list[str]) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for item in items:
            if item and item not in seen:
                seen.add(item)
                ordered.append(item)
        return ordered

    def _join_ids(self, ids: list[str], *, limit: int) -> str:
        if not ids:
            return "未知"
        selected = ids[:limit]
        extra = len(ids) - len(selected)
        text = "、".join(selected)
        if extra > 0:
            text += f" 等{extra}人"
        return text

    def _sort_player_ids(self, ids: list[str]) -> list[str]:
        unique = self._dedupe_preserve_order([str(item) for item in ids if str(item)])

        def sort_key(value: str) -> tuple[str, int, str]:
            match = _PLAYER_ID_SUFFIX.search(value)
            prefix = _PLAYER_ID_SUFFIX.sub("", value)
            if match:
                return (prefix, int(match.group(1)), value)
            return (prefix, 10**9, value)

        return sorted(unique, key=sort_key)

    def _looks_like_player_id(self, value: str) -> bool:
        text = str(value).strip()
        return bool(text) and len(text) <= 16 and any(ch.isdigit() for ch in text)

    @staticmethod
    def _player_id_keys() -> set[str]:
        return {
            "player_id",
            "speaker_id",
            "voter_id",
            "claimer_id",
            "actor_id",
            "source_id",
            "target_id",
            "from_id",
            "to_id",
            "badge_holder",
            "badge_holder_id",
            "sheriff_id",
            "dead_id",
            "dead_player_id",
        }

    @staticmethod
    def _collection_id_keys() -> set[str]:
        return {
            "alive",
            "alive_ids",
            "alive_player_ids",
            "dead",
            "dead_ids",
            "dead_player_ids",
            "players",
            "player_ids",
            "target_ids",
            "candidates",
            "survivors",
            "living",
            "graveyard",
            "claims",
            "votes",
        }

    @staticmethod
    def _ALIVE_KEYS() -> set[str]:
        return {
            "alive",
            "alive_ids",
            "alive_player_ids",
            "survivors",
            "living",
            "live_ids",
            "current_alive",
        }

    @staticmethod
    def _DEAD_KEYS() -> set[str]:
        return {
            "dead",
            "dead_ids",
            "dead_player_ids",
            "eliminated",
            "casualties",
            "deaths",
            "graveyard",
        }

    def _extract_game_id(self, payload: Mapping[str, Any]) -> str | None:
        for key in ("game_id", "gameid", "match_id", "matchid", "session_id"):
            value = payload.get(key)
            if self._is_scalar_text(value):
                return str(value)
        game = payload.get("game")
        if isinstance(game, Mapping):
            for key in ("game_id", "gameid", "match_id", "matchid", "session_id"):
                value = game.get(key)
                if self._is_scalar_text(value):
                    return str(value)
        return None

    def _render_phase_summary(self, turn_packet: Mapping[str, Any], request: Mapping[str, Any]) -> str:
        # 预留方法：具体内容由 _render_turn_summary 拼接。
        del turn_packet, request
        return ""

    def _collect_claim_block(self, notes: list[str]) -> str:
        claims = []
        for note in notes:
            if any(term in note for term in ("claim", "自称", "预言家", "女巫", "猎人", "守卫", "金水", "查杀", "查验", "警徽")):
                claims.append(self._trim_text(note, 90))
        if not claims:
            claims = [self._trim_text(item, 90) for item in self._public_ledger["public_claims"][:3]]
        return " | ".join(claims[:3])

    def _public_ledger_snapshot(self, turn_packet: Mapping[str, Any]) -> str:
        public_state = turn_packet.get("public_state")
        turn_notes = self._extract_public_notes(public_state)
        ledger_alive = set(self._public_ledger["alive_ids"])
        ledger_dead = set(self._public_ledger["dead_ids"])
        current_alive = self._extract_player_ids(public_state, include_collections=True)
        combined_alive = self._sort_player_ids((ledger_alive | current_alive) - ledger_dead)
        combined_dead = self._sort_player_ids(ledger_dead | self._extract_dead_from_notes(turn_notes))
        lines = ["【公共台账】"]
        lines.append("存活: " + self._join_ids(combined_alive, limit=10))
        lines.append("已死: " + self._join_ids(combined_dead, limit=10))
        claim_block = self._collect_claim_block(turn_notes)
        if claim_block:
            lines.append("查验/自述: " + claim_block)
        return "\n".join(lines)

    def _phase_summary(self, turn_packet: Mapping[str, Any], request: Mapping[str, Any]) -> str:
        public_state = turn_packet.get("public_state")
        turn_notes = self._extract_public_notes(public_state)
        combined_alive = self._sort_player_ids(
            (set(self._public_ledger["alive_ids"]) | self._extract_player_ids(public_state, include_collections=True))
            - set(self._public_ledger["dead_ids"])
        )
        combined_dead = self._sort_player_ids(
            set(self._public_ledger["dead_ids"]) | self._extract_dead_from_notes(turn_notes)
        )
        candidate_pool = self._candidate_pool(request, combined_alive, self._sort_player_ids(list(self._public_ledger["known_player_ids"])))
        phase = str(request.get("phase") or request.get("channel") or request.get("kind") or "")
        phase_lower = phase.lower()
        allowed_actions = request.get("allowed_actions")
        allowed_kinds = {
            str(item.get("kind") or "").lower()
            for item in allowed_actions
            if isinstance(item, Mapping)
        } if isinstance(allowed_actions, list) else set()
        is_vote = any(key in phase_lower for key in ("vote", "投票")) or any("vote" in kind for kind in allowed_kinds)
        if is_vote:
            return self._render_vote_summary(candidate_pool, combined_alive, combined_dead, turn_notes)
        return self._render_speak_summary(candidate_pool, combined_alive, combined_dead, turn_notes)

    def _render_turn_summary(self, turn_packet: Mapping[str, Any], request: Mapping[str, Any]) -> str:
        ledger = self._public_ledger_snapshot(turn_packet)
        phase = self._phase_summary(turn_packet, request)
        summary = "\n".join(part for part in (ledger, phase) if part)
        return self._trim_text(summary, _PUBLIC_BLOCK_CHAR_LIMIT + _PHASE_BLOCK_CHAR_LIMIT)

    def _render_vote_summary(
        self,
        candidate_pool: list[str],
        alive_ids: list[str],
        dead_ids: list[str],
        notes: list[str],
    ) -> str:
        ranked = self._rank_candidates(candidate_pool, notes, dead_ids)
        if len(ranked) < 2 and alive_ids:
            fallback_pool = [pid for pid in alive_ids if pid not in dead_ids and pid not in {item[0] for item in ranked}]
            ranked.extend(self._rank_candidates(fallback_pool, notes, dead_ids))
        while len(ranked) < 2 and candidate_pool:
            ranked.append((candidate_pool[0], 0, "公开信息不足"))
        ranked = ranked[:2]
        lines = ["【本轮投票参考】"]
        for index, (candidate, score, reason) in enumerate(ranked, start=1):
            lines.append(f"{index}. {candidate} | 分数={score} | {reason}")
        if len(ranked) >= 2:
            lines.append(f"优先比较：{ranked[0][0]} vs {ranked[1][0]}")
        else:
            lines.append("优先比较至少两个候选；当前候选不足时，用公开票型与查验链补足。")
        return self._trim_text("\n".join(lines), _PHASE_BLOCK_CHAR_LIMIT)

    def _render_speak_summary(
        self,
        candidate_pool: list[str],
        alive_ids: list[str],
        dead_ids: list[str],
        notes: list[str],
    ) -> str:
        ranked = self._rank_candidates(candidate_pool, notes, dead_ids)
        if not ranked and alive_ids:
            ranked = self._rank_candidates(alive_ids, notes, dead_ids)
        if not ranked:
            ranked = [("信息不足", 0, "公开信息不足")]
        focus = ranked[0]
        avoid = self._least_suspicious_candidate(candidate_pool or alive_ids, notes, dead_ids)
        lines = ["【本轮发言参考】"]
        lines.append(f"关注: {focus[0]} | {focus[2]}")
        if avoid:
            lines.append(f"暂不投: {avoid[0]} | {avoid[2]}")
        else:
            lines.append("暂不投: 先保留公开链中更稳的一人，直到票型或查验出现新变化。")
        return self._trim_text("\n".join(lines), _PHASE_BLOCK_CHAR_LIMIT)

    def _rank_candidates(
        self,
        candidate_pool: list[str],
        notes: list[str],
        dead_ids: list[str],
    ) -> list[tuple[str, int, str]]:
        dead_set = set(dead_ids)
        ranked: list[tuple[str, int, str]] = []
        for candidate in self._sort_player_ids(candidate_pool):
            if candidate in dead_set or candidate == self.player_id:
                continue
            score, reasons = self._score_candidate(candidate, notes)
            if reasons:
                reason_text = "；".join(reasons[:2])
            else:
                reason_text = "公开信息不足"
            ranked.append((candidate, score, reason_text))
        ranked.sort(key=lambda item: (-item[1], item[0]))
        return ranked

    def _least_suspicious_candidate(
        self,
        candidate_pool: list[str],
        notes: list[str],
        dead_ids: list[str],
    ) -> tuple[str, int, str] | None:
        ranked = self._rank_candidates(candidate_pool, notes, dead_ids)
        if not ranked:
            return None
        ranked.sort(key=lambda item: (item[1], item[0]))
        return ranked[0]

    def _score_candidate(self, candidate: str, notes: list[str]) -> tuple[int, list[str]]:
        lower_candidate = candidate.lower()
        score = 0
        reasons: list[str] = []
        suspicious_terms = ("查杀", "对跳", "爆狼", "矛盾", "票型异常", "反水", "不一致", "假", "冲票", "狼")
        protect_terms = ("金水", "可信", "报验链", "警徽", "预言家", "验过", "已验", "好人链", "强神")
        for note in notes:
            lower = note.lower()
            if lower_candidate not in lower:
                continue
            score += 1
            compact = self._trim_text(note, 80)
            if any(term in note for term in suspicious_terms):
                score += 2
                reasons.append(f"疑点: {compact}")
            elif any(term in note for term in protect_terms):
                score -= 2
                reasons.append(f"保护: {compact}")
            elif "vote" in lower or "票" in note:
                score += 1
                reasons.append(f"票型: {compact}")
            elif any(term in note for term in ("claim", "自称", "发言", "说", "跳")):
                reasons.append(f"发言: {compact}")
            else:
                reasons.append(compact)
        if not reasons:
            reasons.append("公开信息不足")
        return score, reasons

    def _candidate_pool(
        self,
        request: Mapping[str, Any],
        alive_ids: list[str],
        known_ids: list[str],
    ) -> list[str]:
        pool: set[str] = set(alive_ids)
        pool.update(known_ids)
        raw_actions = request.get("allowed_actions")
        if isinstance(raw_actions, list):
            for action in raw_actions:
                if not isinstance(action, Mapping):
                    continue
                target_ids = action.get("target_ids")
                if isinstance(target_ids, list):
                    for target_id in target_ids:
                        if isinstance(target_id, (str, int)) and not isinstance(target_id, bool):
                            pool.add(str(target_id))
        pool.discard(self.player_id)
        pool.difference_update(self._public_ledger["dead_ids"])
        return self._sort_player_ids(list(pool))

    def _extract_public_notes(self, payload: Any) -> list[str]:
        notes: list[str] = []
        self._walk_public_payload(payload, notes, depth=0)
        return self._dedupe_preserve_order(notes)

    def _walk_public_payload(self, payload: Any, notes: list[str], depth: int) -> None:
        if depth > 4:
            return
        if isinstance(payload, Mapping):
            snapshot = self._snapshot_mapping(payload)
            if snapshot:
                notes.append(snapshot)
            for value in payload.values():
                self._walk_public_payload(value, notes, depth + 1)
            return
        if isinstance(payload, list):
            for item in payload[:25]:
                self._walk_public_payload(item, notes, depth + 1)

    def _snapshot_mapping(self, mapping: Mapping[str, Any]) -> str | None:
        lower_map = {str(key).lower(): value for key, value in mapping.items()}
        event = self._first_scalar(lower_map, ("event", "event_type", "type", "kind", "action"))
        event_lower = event.lower() if event else ""

        vote_like = any(
            key in lower_map
            for key in (
                "voter_id",
                "voter",
                "vote_target",
                "target_id",
                "target",
                "ballot",
                "vote",
            )
        ) or "vote" in event_lower
        claim_like = any(
            key in lower_map
            for key in (
                "speaker_id",
                "claimer_id",
                "claim",
                "role_claim",
                "speech",
                "text",
                "content",
                "message",
                "reason",
            )
        ) or any(term in event_lower for term in ("claim", "speak", "speech", "say", "发言", "自称"))
        sheriff_like = any(
            key in lower_map
            for key in (
                "sheriff_id",
                "badge_holder",
                "badge_holder_id",
                "badge_from",
                "badge_to",
                "badge",
            )
        ) or any(term in event_lower for term in ("sheriff", "badge", "警徽", "警长"))
        death_like = any(key in lower_map for key in ("dead", "death", "dead_id", "dead_player_id")) or any(
            term in event_lower for term in ("death", "dead", "kill", "out", "死亡", "出局")
        )

        if not (vote_like or claim_like or sheriff_like or death_like):
            return None

        round_text = self._first_scalar(lower_map, ("round", "day", "turn", "phase", "channel"))
        player = self._first_scalar(lower_map, ("player_id", "speaker_id", "voter_id", "claimer_id", "actor_id", "source_id"))
        target = self._first_scalar(lower_map, ("target_id", "target", "vote_target", "to_id", "suspect_id"))
        source = self._first_scalar(lower_map, ("from_id", "source_id", "badge_from"))
        holder = self._first_scalar(lower_map, ("badge_holder", "badge_holder_id", "sheriff_id"))
        role = self._first_scalar(lower_map, ("role", "claim_role", "claimed_role"))
        text = self._first_scalar(lower_map, ("text", "content", "message", "reason", "speech"))

        parts: list[str] = []
        if event:
            parts.append(event_lower or event.lower())
        if vote_like:
            parts.append(f"vote voter={player or self._first_scalar(lower_map, ('voter',)) or '?'} target={target or '?'}")
        elif sheriff_like:
            parts.append(
                f"sheriff holder={holder or '?'} from={source or '?'} to={target or '?'}"
            )
        elif death_like:
            parts.append(f"death player={player or self._first_scalar(lower_map, ('dead_id', 'dead_player_id')) or '?'}")
        elif claim_like:
            parts.append(f"claim speaker={player or '?'}")
            if role:
                parts.append(f"role={role}"
)
        if round_text:
            parts.append(f"at={round_text}")
        if text:
            cleaned = self._trim_text(text, _PUBLIC_NOTE_CHAR_LIMIT)
            parts.append(f"text={cleaned}")
        snapshot = " ".join(part for part in parts if part).strip()
        return self._trim_text(snapshot, _PUBLIC_NOTE_CHAR_LIMIT) if snapshot else None

    def _extract_player_ids(self, payload: Any, *, include_collections: bool = False) -> set[str]:
        ids: set[str] = set()

        def walk(obj: Any) -> None:
            if isinstance(obj, Mapping):
                lower_map = {str(key).lower(): value for key, value in obj.items()}
                for key, value in lower_map.items():
                    if key in self._player_id_keys() and self._is_scalar_id(value):
                        ids.add(str(value))
                    if include_collections and key in self._collection_id_keys():
                        ids.update(self._as_id_set(value))
                for value in obj.values():
                    walk(value)
            elif isinstance(obj, list):
                for item in obj[:25]:
                    walk(item)

        walk(payload)
        ids.discard(self.player_id)
        return ids

    def _collect_id_collection(self, mapping: Mapping[str, Any], allowed_keys: set[str]) -> set[str]:
        collected: set[str] = set()
        lower_map = {str(key).lower(): value for key, value in mapping.items()}
        for key, value in lower_map.items():
            if key in allowed_keys:
                collected.update(self._as_id_set(value))
        return collected

    def _extract_dead_from_notes(self, notes: list[str]) -> set[str]:
        dead: set[str] = set()
        for note in notes:
            lower = note.lower()
            if any(term in lower for term in ("death", "dead", "死亡", "出局", "淘汰")):
                dead.update(self._ids_from_text(note))
        return dead

    def _ids_from_text(self, text: str) -> set[str]:
        ids: set[str] = set()
        for token in re.split(r"[\s,，;；|/\\]+", text):
            token = token.strip("()[]{}<>：:=，。,.、")
            if token and self._looks_like_player_id(token):
                ids.add(token)
        return ids

    def _as_id_set(self, value: Any) -> set[str]:
        if isinstance(value, list):
            return {str(item) for item in value if self._is_scalar_id(item)}
        if isinstance(value, tuple):
            return {str(item) for item in value if self._is_scalar_id(item)}
        if self._is_scalar_id(value):
            return {str(value)}
        return set()

    def _merge_unique(self, items: list[str], values: list[str]) -> None:
        for value in values:
            if value and value not in items:
                items.append(value)

    def _append_bounded(self, items: list[str], value: str, limit: int) -> None:
        if not value:
            return
        if value in items:
            return
        items.append(value)
        if len(items) > limit:
            del items[: len(items) - limit]

    def _first_scalar(self, mapping: Mapping[str, Any], keys: tuple[str, ...]) -> str:
        for key in keys:
            value = mapping.get(key)
            if self._is_scalar_text(value):
                return self._trim_text(str(value), _PUBLIC_NOTE_CHAR_LIMIT)
        return ""

    @staticmethod
    def _is_scalar_text(value: Any) -> bool:
        return isinstance(value, (str, int, float)) and not isinstance(value, bool)

    def _is_scalar_id(self, value: Any) -> bool:
        if isinstance(value, bool):
            return False
        if isinstance(value, int):
            return True
        if isinstance(value, str):
            return self._looks_like_player_id(value)
        return False

    def _trim_text(self, text: str, limit: int) -> str:
        normalized = " ".join(str(text).split())
        if len(normalized) <= limit:
            return normalized
        if limit <= 3:
            return normalized[:limit]
        return normalized[: limit - 1] + "…"

    def _dedupe_preserve_order(self, items: list[str]) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for item in items:
            if item and item not in seen:
                seen.add(item)
                ordered.append(item)
        return ordered

    def _join_ids(self, ids: list[str], *, limit: int) -> str:
        if not ids:
            return "未知"
        selected = ids[:limit]
        extra = len(ids) - len(selected)
        text = "、".join(selected)
        if extra > 0:
            text += f" 等{extra}人"
        return text

    def _sort_player_ids(self, ids: list[str]) -> list[str]:
        unique = self._dedupe_preserve_order([str(item) for item in ids if str(item)])

        def sort_key(value: str) -> tuple[str, int, str]:
            match = _PLAYER_ID_SUFFIX.search(value)
            prefix = _PLAYER_ID_SUFFIX.sub("", value)
            if match:
                return (prefix, int(match.group(1)), value)
            return (prefix, 10**9, value)

        return sorted(unique, key=sort_key)

    def _looks_like_player_id(self, value: str) -> bool:
        text = str(value).strip()
        return bool(text) and len(text) <= 16 and any(ch.isdigit() for ch in text)

    @staticmethod
    def _player_id_keys() -> set[str]:
        return {
            "player_id",
            "speaker_id",
            "voter_id",
            "claimer_id",
            "actor_id",
            "source_id",
            "target_id",
            "from_id",
            "to_id",
            "badge_holder",
            "badge_holder_id",
            "sheriff_id",
            "dead_id",
            "dead_player_id",
        }

    @staticmethod
    def _collection_id_keys() -> set[str]:
        return {
            "alive",
            "alive_ids",
            "alive_player_ids",
            "dead",
            "dead_ids",
            "dead_player_ids",
            "players",
            "player_ids",
            "target_ids",
            "candidates",
            "survivors",
            "living",
            "graveyard",
            "claims",
            "votes",
        }

    @staticmethod
    def _ALIVE_KEYS() -> set[str]:
        return {
            "alive",
            "alive_ids",
            "alive_player_ids",
            "survivors",
            "living",
            "live_ids",
            "current_alive",
        }

    @staticmethod
    def _DEAD_KEYS() -> set[str]:
        return {
            "dead",
            "dead_ids",
            "dead_player_ids",
            "eliminated",
            "casualties",
            "deaths",
            "graveyard",
        }

"""狼人杀平民 Task-Agent。

每次行动从一个新的模型会话开始；默认只提供当前行动包和一份轻量公共台账。
如需当前轮完整可见发言，模型必须显式调用唯一的受限工具。
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
DEFAULT_MAX_DECISION_RETRIES = 2
_SPEECH_ACTION_KINDS = frozenset({"speak", "last_words"})

_PUBLIC_SNAPSHOT_LIMIT = 18
_PUBLIC_NOTE_CHAR_LIMIT = 140
_PUBLIC_BLOCK_CHAR_LIMIT = 1200
_PHASE_BLOCK_CHAR_LIMIT = 700

CURRENT_ROUND_DIALOGUE_TOOL: dict[str, Any] = {
    "name": CURRENT_ROUND_DIALOGUE_TOOL_NAME,
    "description": "读取当前昼夜轮次中、当前玩家依法可见的已发生发言。",
    "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
}


def _request_is_vote(request: Mapping[str, Any]) -> bool:
    phase = str(request.get("phase") or request.get("channel") or request.get("kind") or "").lower()
    raw_actions = request.get("allowed_actions")
    kinds = {
        str(item.get("kind") or "").lower()
        for item in raw_actions
        if isinstance(item, Mapping)
    } if isinstance(raw_actions, list) else set()
    return any(key in phase for key in ("vote", "投票", "放逐")) or any("vote" in kind for kind in kinds)


def _target_actions(request: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw_actions = request.get("allowed_actions")
    if not isinstance(raw_actions, list):
        return []
    return [item for item in raw_actions if isinstance(item, Mapping) and isinstance(item.get("target_ids"), list) and item.get("target_ids")]


def render_villager_decision_checklist(request: Mapping[str, Any]) -> str:
    """给平民的短清单：压缩提示并把投票收敛作为硬策略。"""

    raw_actions = request.get("allowed_actions")
    kinds = {
        str(item.get("kind") or "").lower()
        for item in raw_actions
        if isinstance(item, Mapping)
    } if isinstance(raw_actions, list) else set()
    phase = str(request.get("phase") or request.get("channel") or request.get("kind") or "").lower()
    is_vote = _request_is_vote(request)
    is_speak = any(key in phase for key in ("speak", "discussion", "发言", "白天", "day")) or bool(
        kinds & {"speak", "discussion", "day_speak"}
    )
    is_last_words = any(key in phase for key in ("last", "遗言")) or "last_words" in kinds
    is_sheriff = any(key in phase for key in ("sheriff", "警长", "警徽")) or bool(kinds & {"sheriff", "sheriff_vote"})

    lines = [
        "【平民决策清单】区分：系统事实 > 清晰查验/警徽链 > 玩家声明 > 推测；不要把当前聊天当全部历史。",
        "优先保护可信预言家链、金水、强神、警徽链；除非硬对跳/事实矛盾/查杀坐实，不把他们作为第一票。",
        "优先找未被保护且有查杀、票型异常、事实错误、冲票抱团或反复改口的人；被质疑时不要条件反射反咬好人。",
    ]
    if is_vote:
        if _target_actions(request):
            lines.append("投票阶段禁止主动 pass/弃票/空票；必须从合法 target_id 中择一，至少比较两个候选后收敛到证据最强者。")
        else:
            lines.append("若本阶段没有 target_id，只能按 allowed_actions 合法输出，不能臆造目标。")
        lines.append("有可信归票（真预言家/警长/清晰警徽遗产）时优先跟随，避免平民互撕分票。")
    if is_speak:
        lines.append("发言要短：点一个关注对象、一个暂不投对象，并说明对应公开依据。")
    if is_last_words:
        lines.append("遗言只留公开事实、票型、查验/警徽链和灰区，不把推测说成系统事实。")
    if is_sheriff:
        lines.append("警长/警徽回合看报验链是否连续一致；接徽可继承公开遗产但不是系统确认。")
    return "\n".join(lines)


def render_action_contract(request: Mapping[str, Any]) -> str:
    """把本次 allowed_actions 转成模型容易遵守的 JSON 契约。"""

    raw_actions = request.get("allowed_actions")
    allowed_actions = [item for item in raw_actions if isinstance(item, Mapping)] if isinstance(raw_actions, list) else []
    lines = [
        "【本次行动 JSON 契约】",
        f"阶段：{request.get('phase') or 'unknown'}；频道：{request.get('channel') or 'unknown'}。",
        "最终回答只能是一个 JSON 对象；不得附加解释、Markdown 或额外字段。",
        "只能选择下列合法模板；target_id 只能取模板列出的候选值。",
    ]
    if not allowed_actions:
        lines.append("当前没有可用行动；不要臆造行动。")
        return "\n".join(lines)

    vote_has_target = _request_is_vote(request) and bool(_target_actions(request))
    for allowed in allowed_actions:
        kind = str(allowed.get("kind") or "")
        if not kind:
            continue
        target_ids = [str(pid) for pid in allowed.get("target_ids", []) if isinstance(pid, (str, int)) and not isinstance(pid, bool)]
        example: dict[str, str] = {"kind": kind}
        if kind in _SPEECH_ACTION_KINDS:
            example["text"] = "好"
        elif target_ids:
            example["target_id"] = target_ids[0]
        lines.append("- `" + json.dumps(example, ensure_ascii=False, separators=(",", ":")) + "`")
        if target_ids:
            lines.append("  target_id 候选值：" + "、".join(target_ids) + "。")
        if kind in _SPEECH_ACTION_KINDS:
            constraints = []
            if allowed.get("max_chars") is not None:
                constraints.append(f"text 最多 {allowed['max_chars']} 个字符")
            if allowed.get("require_chinese"):
                constraints.append("text 至少包含中文")
            lines.append("  " + "；".join(constraints or ["text 非空"]) + "。")
    if vote_has_target:
        lines.append("投票有合法目标时，不要选择 pass/abstain/skip/no_op；必须投给一个 listed target_id。")
    lines.append("kind/target_id 是协议字段，必须保留英文原样。")
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
    """提交引擎前的本地校验。"""

    allowed_actions = request.get("allowed_actions")
    if not isinstance(allowed_actions, list):
        return "request.allowed_actions 缺失"
    allowed = next((item for item in allowed_actions if isinstance(item, Mapping) and item.get("kind") == action.get("kind")), None)
    if allowed is None:
        return "kind 必须是 allowed_actions 中的一项"
    if _request_is_vote(request) and _target_actions(request) and str(action.get("kind", "")).lower() in {"pass", "abstain", "skip", "no_op"}:
        return "投票有合法 target_id 时不能主动 pass/弃票"
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
        """接收公开同步并维护仅含公开事实的轻量台账。"""

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

        # 当前行动包的 public_state 是状态主信源；先摄入，保证台账用最新系统事实。
        self._ingest_public_packet(turn_packet)

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
        for attempt in range(self.max_decision_retries + 1):
            allow_tool = attempt == 0 and self.max_tool_calls_per_decision > 0
            user_content = {"instruction": self._turn_instruction(turn_packet, feedback), "packet": prompt}
            try:
                raw = await self._complete_json(
                    system=system,
                    messages=[{"role": "user", "content": json.dumps(user_content, ensure_ascii=False)}],
                    tools=[CURRENT_ROUND_DIALOGUE_TOOL] if allow_tool else None,
                    tool_executor=execute_tool if allow_tool else None,
                    max_tool_calls=self.max_tool_calls_per_decision if allow_tool else 0,
                    max_tool_result_tokens=self.max_tool_result_tokens,
                    max_prompt_chars=self.max_prompt_chars,
                )
            except ModelClientError:
                if attempt >= self.max_decision_retries:
                    raise
                feedback = "上一轮模型调用没有产生可用结构化行动，请直接给出合法 JSON。"
                continue
            self._record_token_usage(raw)
            action = normalize_decision(raw, turn_packet["request"])
            error = decision_error(action, turn_packet["request"])
            if error is None:
                return action
            if attempt >= self.max_decision_retries:
                raise RuleViolationError(f"模型返回非法行动：{error}", attempted_action=action, validation_error=error)
            feedback = error
        raise RuleViolationError("模型未能产生合法行动")

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
            "task_fingerprint": hashlib.sha256(self.profile.task.encode("utf-8")).hexdigest(),
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
        parts = [self._render_turn_summary(turn_packet, request)]
        if feedback:
            parts.append(f"上一次输出未通过校验：{feedback}")
        parts.append(render_villager_decision_checklist(request))
        return render_prompt(
            "player_turn_instruction.txt",
            action_contract=render_action_contract(request),
            validation_feedback="\n".join(part for part in parts if part),
        )

    async def _complete_json(self, **kwargs: Any) -> dict[str, Any]:
        if self.request_coordinator is not None:
            return await self.request_coordinator.complete_json(self.model_client, max_tokens=self.max_tokens, **kwargs)
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
        values = {key: self._nonnegative_int(usage.get(key), fallback=None) for key in ("input_tokens", "output_tokens", "total_tokens")}
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
        self._public_ledger["known_player_ids"].update(self._extract_player_ids(payload))

        public_state = self._find_public_state(payload)
        alive, dead, known, authoritative_alive = self._extract_status_from_public_state(public_state)
        self._public_ledger["known_player_ids"].update(known)
        if authoritative_alive:
            self._public_ledger["alive_ids"] = set(alive)
        elif alive:
            self._public_ledger["alive_ids"].update(alive)
        self._public_ledger["dead_ids"].update(dead)
        # 存活/死亡严格互斥；死亡事实优先。
        self._public_ledger["alive_ids"].difference_update(self._public_ledger["dead_ids"])

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
            if any(key in lower for key in ("death", "dead", "出局", "死亡", "刀", "淘汰")):
                self._append_bounded(self._public_ledger["death_notices"], note, _PUBLIC_SNAPSHOT_LIMIT)
            if any(key in lower for key in ("claim", "自称", "预言家", "女巫", "猎人", "守卫", "金水", "查杀", "查验", "警徽")):
                self._append_bounded(self._public_ledger["public_claims"], note, _PUBLIC_SNAPSHOT_LIMIT)

    def _render_turn_summary(self, turn_packet: Mapping[str, Any], request: Mapping[str, Any]) -> str:
        public_state = turn_packet.get("public_state")
        turn_notes = self._extract_public_notes(public_state)
        alive, dead, known, authoritative_alive = self._extract_status_from_public_state(public_state)
        ledger_dead = set(self._public_ledger["dead_ids"]) | dead | self._extract_dead_from_notes(turn_notes)
        allowed_targets = self._allowed_target_ids(request)

        if authoritative_alive or alive:
            alive_set = set(alive)
        else:
            alive_set = set(self._public_ledger["alive_ids"])
        if not alive_set:
            # 回退只用于“未知时补充可行动对象”，绝不从单条发言深挖 alive 覆盖全集。
            alive_set = allowed_targets | {self.player_id}
        alive_set.difference_update(ledger_dead)

        known_set = set(self._public_ledger["known_player_ids"]) | known | alive_set | ledger_dead | allowed_targets
        combined_alive = self._sort_player_ids(alive_set)
        combined_dead = self._sort_player_ids(ledger_dead)
        combined_known = self._sort_player_ids(known_set - {self.player_id})
        candidate_pool = self._candidate_pool(request, combined_alive, combined_known)

        ledger_lines = ["【公共台账】"]
        alive_label = "存活" if authoritative_alive or alive or self._public_ledger["alive_ids"] else "存活(按合法目标补充)"
        ledger_lines.append(f"{alive_label}: " + self._join_ids(combined_alive, limit=12))
        ledger_lines.append("已死: " + self._join_ids(combined_dead, limit=12))
        claim_block = self._collect_claim_block(turn_notes)
        if claim_block:
            ledger_lines.append("查验/自述: " + claim_block)
        gray_pool = [pid for pid in candidate_pool if pid not in combined_dead]
        if gray_pool:
            ledger_lines.append("灰区/可投: " + self._join_ids(gray_pool[:8], limit=8))

        if _request_is_vote(request):
            phase_lines = self._render_vote_summary(candidate_pool, combined_alive, combined_dead, turn_notes)
        else:
            phase_lines = self._render_speak_summary(candidate_pool, combined_alive, combined_dead, turn_notes)
        return self._trim_text("\n".join(["\n".join(ledger_lines), phase_lines]), _PUBLIC_BLOCK_CHAR_LIMIT + _PHASE_BLOCK_CHAR_LIMIT)

    def _render_vote_summary(self, candidate_pool: list[str], alive_ids: list[str], dead_ids: list[str], notes: list[str]) -> str:
        ranked = self._rank_candidates(candidate_pool, notes, dead_ids)
        if len(ranked) < 2 and alive_ids:
            used = {item[0] for item in ranked}
            ranked.extend(self._rank_candidates([pid for pid in alive_ids if pid not in used], notes, dead_ids))
        ranked = ranked[:2]
        lines = ["【本轮投票参考】"]
        if not ranked:
            lines.append("候选不足：按 allowed_actions 选择唯一合法行动，不臆造目标。")
        else:
            for index, (candidate, score, reason) in enumerate(ranked, start=1):
                lines.append(f"{index}. {candidate} | 分数={score} | {reason}")
            if len(ranked) >= 2:
                lines.append(f"收敛比较：{ranked[0][0]} vs {ranked[1][0]}；若有可信归票优先跟随。")
            lines.append("有合法 target_id 时必须投票，不主动 pass。")
        return self._trim_text("\n".join(lines), _PHASE_BLOCK_CHAR_LIMIT)

    def _render_speak_summary(self, candidate_pool: list[str], alive_ids: list[str], dead_ids: list[str], notes: list[str]) -> str:
        ranked = self._rank_candidates(candidate_pool or alive_ids, notes, dead_ids)
        focus = ranked[0] if ranked else ("信息不足", 0, "公开信息不足")
        avoid = self._least_suspicious_candidate(candidate_pool or alive_ids, notes, dead_ids)
        lines = ["【本轮发言参考】", f"关注: {focus[0]} | {focus[2]}"]
        if avoid:
            lines.append(f"暂不投: {avoid[0]} | {avoid[2]}")
        else:
            lines.append("暂不投: 暂保公开链更稳者，等待票型/查验变化。")
        return self._trim_text("\n".join(lines), _PHASE_BLOCK_CHAR_LIMIT)

    def _rank_candidates(self, candidate_pool: list[str], notes: list[str], dead_ids: list[str]) -> list[tuple[str, int, str]]:
        dead_set = set(dead_ids)
        ranked: list[tuple[str, int, str]] = []
        for candidate in self._sort_player_ids(candidate_pool):
            if candidate in dead_set or candidate == self.player_id:
                continue
            score, reasons = self._score_candidate(candidate, notes)
            ranked.append((candidate, score, "；".join(reasons[:2]) if reasons else "公开信息不足"))
        ranked.sort(key=lambda item: (-item[1], item[0]))
        return ranked

    def _least_suspicious_candidate(self, candidate_pool: list[str], notes: list[str], dead_ids: list[str]) -> tuple[str, int, str] | None:
        ranked = self._rank_candidates(candidate_pool, notes, dead_ids)
        if not ranked:
            return None
        ranked.sort(key=lambda item: (item[1], item[0]))
        return ranked[0]

    def _score_candidate(self, candidate: str, notes: list[str]) -> tuple[int, list[str]]:
        """关系化打分：区分“谁说的”和“说谁”，避免关键词误伤发言者。"""

        score = 0
        reasons: list[str] = []
        vote_hits = 0
        for note in notes[-24:]:
            lower = note.lower()
            if not self._mentions_player(note, candidate):
                continue
            compact = self._trim_text(note, 78)
            speaker = self._field_value(lower, "speaker") or self._field_value(lower, "voter")
            target = self._field_value(lower, "target") or self._field_value(lower, "player")
            is_target = target == candidate.lower()
            is_speaker = speaker == candidate.lower()
            text = lower.split("text=", 1)[-1] if "text=" in lower else lower

            if "vote" in lower and is_target:
                vote_hits += 1
                score += 2
                reasons.append(f"被投/归票: {compact}")
                continue
            if is_target and any(term in note for term in ("查杀", "爆狼", "票型异常", "冲票", "矛盾")):
                score += 4
                reasons.append(f"被明确指向: {compact}")
                continue
            if is_target and any(term in note for term in ("金水", "好人", "银水", "可信")):
                score -= 4
                reasons.append(f"保护信息: {compact}")
                continue
            if is_speaker:
                if any(term in note for term in ("预言家", "女巫", "猎人", "守卫", "强神", "警长")):
                    score -= 1
                    reasons.append(f"身份/警徽声明待验: {compact}")
                if any(term in note for term in ("前后矛盾", "爆狼", "假", "反水", "不一致")):
                    score += 2
                    reasons.append(f"发言疑点: {compact}")
                if self._mentions_player(text, self.player_id) and any(term in note for term in ("狼", "查杀", "抗推", "出")):
                    score += 1
                    reasons.append(f"质疑我但低权重: {compact}")
                continue
            if self._near_player_term(note, candidate, ("金水", "好人", "银水", "可信", "保")):
                score -= 3
                reasons.append(f"近邻保护词: {compact}")
            elif self._near_player_term(note, candidate, ("查杀", "狼", "爆狼", "冲票", "矛盾", "悍跳", "抗推")):
                score += 2
                reasons.append(f"近邻疑点词: {compact}")
        if vote_hits >= 2:
            score += min(3, vote_hits - 1)
            reasons.insert(0, f"多票集中({vote_hits})，需核对是否狼队冲票")
        if not reasons:
            reasons.append("公开信息不足")
        return score, reasons

    def _candidate_pool(self, request: Mapping[str, Any], alive_ids: list[str], known_ids: list[str]) -> list[str]:
        target_ids = self._allowed_target_ids(request)
        pool: set[str] = set(target_ids or alive_ids)
        pool.update(known_ids if not target_ids else [])
        pool.discard(self.player_id)
        pool.difference_update(self._public_ledger["dead_ids"])
        return self._sort_player_ids(list(pool))

    def _allowed_target_ids(self, request: Mapping[str, Any]) -> set[str]:
        ids: set[str] = set()
        for action in request.get("allowed_actions", []) if isinstance(request.get("allowed_actions"), list) else []:
            if not isinstance(action, Mapping):
                continue
            target_ids = action.get("target_ids")
            if isinstance(target_ids, list):
                ids.update(str(pid) for pid in target_ids if self._is_scalar_id(pid))
        return ids

    def _collect_claim_block(self, notes: list[str]) -> str:
        claims = [
            self._trim_text(note, 90)
            for note in notes
            if any(term in note for term in ("claim", "自称", "预言家", "女巫", "猎人", "守卫", "金水", "查杀", "查验", "警徽"))
        ]
        if not claims:
            claims = [self._trim_text(item, 90) for item in self._public_ledger["public_claims"][-3:]]
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
        elif isinstance(payload, list):
            for item in payload[:25]:
                self._walk_public_payload(item, notes, depth + 1)

    def _snapshot_mapping(self, mapping: Mapping[str, Any]) -> str | None:
        lower_map = {str(key).lower(): value for key, value in mapping.items()}
        event = self._first_scalar(lower_map, ("event", "event_type", "type", "kind", "action"))
        event_lower = event.lower() if event else ""
        vote_like = any(key in lower_map for key in ("voter_id", "voter", "vote_target", "target_id", "target", "ballot", "vote")) or "vote" in event_lower
        claim_like = any(key in lower_map for key in ("speaker_id", "claimer_id", "claim", "role_claim", "speech", "text", "content", "message", "reason")) or any(term in event_lower for term in ("claim", "speak", "speech", "say", "发言", "自称"))
        sheriff_like = any(key in lower_map for key in ("sheriff_id", "badge_holder", "badge_holder_id", "badge_from", "badge_to", "badge")) or any(term in event_lower for term in ("sheriff", "badge", "警徽", "警长"))
        death_like = any(key in lower_map for key in ("dead", "death", "dead_id", "dead_player_id")) or any(term in event_lower for term in ("death", "dead", "kill", "out", "死亡", "出局", "淘汰"))
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
            parts.append(event_lower)
        if vote_like:
            parts.append(f"vote voter={player or self._first_scalar(lower_map, ('voter',)) or '?'} target={target or '?'}")
        elif sheriff_like:
            parts.append(f"sheriff holder={holder or '?'} from={source or '?'} to={target or '?'}")
        elif death_like:
            parts.append(f"death player={player or self._first_scalar(lower_map, ('dead_id', 'dead_player_id')) or '?'}")
        elif claim_like:
            parts.append(f"claim speaker={player or '?'}")
            if role:
                parts.append(f"role={role}")
        if round_text:
            parts.append(f"at={round_text}")
        if text:
            parts.append("text=" + self._trim_text(text, _PUBLIC_NOTE_CHAR_LIMIT))
        snapshot = " ".join(part for part in parts if part).strip()
        return self._trim_text(snapshot, _PUBLIC_NOTE_CHAR_LIMIT) if snapshot else None

    def _find_public_state(self, payload: Any) -> Any:
        if isinstance(payload, Mapping) and isinstance(payload.get("public_state"), Mapping):
            return payload["public_state"]
        return payload

    def _extract_status_from_public_state(self, public_state: Any) -> tuple[set[str], set[str], set[str], bool]:
        """从 public_state.players 优先提取存活/死亡，缺失时仅看顶层集合字段。"""

        alive: set[str] = set()
        dead: set[str] = set()
        known: set[str] = set()
        authoritative_alive = False
        if not isinstance(public_state, Mapping):
            return alive, dead, known, False

        players = public_state.get("players")
        if isinstance(players, Mapping):
            iterable = []
            for pid, info in players.items():
                if isinstance(info, Mapping):
                    merged = dict(info)
                    merged.setdefault("player_id", pid)
                    iterable.append(merged)
                else:
                    iterable.append({"player_id": pid, "status": info})
            players = iterable
        if isinstance(players, list):
            for item in players:
                if isinstance(item, Mapping):
                    pid = self._first_scalar({str(k).lower(): v for k, v in item.items()}, ("player_id", "id", "pid", "seat_id"))
                    if not self._is_scalar_id(pid):
                        continue
                    known.add(pid)
                    status = self._player_alive_status(item)
                    if status is True:
                        alive.add(pid)
                        authoritative_alive = True
                    elif status is False:
                        dead.add(pid)
                        authoritative_alive = True
                elif self._is_scalar_id(item):
                    known.add(str(item))
        if authoritative_alive:
            return alive - dead, dead, known, True

        # 兼容没有 players 的 mock/非标状态：只接受 public_state 顶层全量字段，避免深层单条事件误覆盖。
        lower_map = {str(key).lower(): value for key, value in public_state.items()}
        for key in ("alive", "alive_ids", "alive_player_ids", "survivors", "living", "live_ids", "current_alive"):
            alive.update(self._as_id_set(lower_map.get(key)))
        for key in ("dead", "dead_ids", "dead_player_ids", "eliminated", "casualties", "deaths", "graveyard"):
            dead.update(self._as_id_set(lower_map.get(key)))
        known.update(alive | dead)
        return alive - dead, dead, known, bool(alive)

    def _player_alive_status(self, player: Mapping[str, Any]) -> bool | None:
        lower = {str(key).lower(): value for key, value in player.items()}
        for key in ("alive", "is_alive", "living", "is_living"):
            if isinstance(lower.get(key), bool):
                return bool(lower[key])
        for key in ("dead", "is_dead", "eliminated", "out"):
            if isinstance(lower.get(key), bool):
                return not bool(lower[key])
        for key in ("status", "state", "life_state"):
            value = lower.get(key)
            if self._is_scalar_text(value):
                text = str(value).lower()
                if any(term in text for term in ("alive", "living", "存活", "在场", "生存")):
                    return True
                if any(term in text for term in ("dead", "death", "out", "eliminated", "死亡", "出局", "淘汰")):
                    return False
        return None

    def _extract_player_ids(self, payload: Any) -> set[str]:
        ids: set[str] = set()

        def walk(obj: Any, depth: int) -> None:
            if depth > 4:
                return
            if isinstance(obj, Mapping):
                lower_map = {str(key).lower(): value for key, value in obj.items()}
                for key, value in lower_map.items():
                    if key in self._player_id_keys() and self._is_scalar_id(value):
                        ids.add(str(value))
                for value in obj.values():
                    walk(value, depth + 1)
            elif isinstance(obj, list):
                for item in obj[:25]:
                    walk(item, depth + 1)

        walk(payload, 0)
        ids.discard(self.player_id)
        return ids

    def _extract_dead_from_notes(self, notes: list[str]) -> set[str]:
        dead: set[str] = set()
        for note in notes:
            lower = note.lower()
            if any(term in lower for term in ("death", "dead", "死亡", "出局", "淘汰")):
                for pid in self._ids_from_text(note):
                    if pid != self.player_id:
                        dead.add(pid)
        return dead

    def _ids_from_text(self, text: str) -> set[str]:
        ids: set[str] = set()
        for token in re.split(r"[\s,，;；|/\\]+", text):
            token = token.strip("()[]{}<>：:=，。,.、")
            if token and self._looks_like_player_id(token):
                ids.add(token)
        return ids

    def _as_id_set(self, value: Any) -> set[str]:
        if isinstance(value, Mapping):
            return {str(key) for key in value if self._is_scalar_id(key)}
        if isinstance(value, (list, tuple, set)):
            ids: set[str] = set()
            for item in value:
                if self._is_scalar_id(item):
                    ids.add(str(item))
                elif isinstance(item, Mapping):
                    pid = self._first_scalar({str(k).lower(): v for k, v in item.items()}, ("player_id", "id", "pid", "seat_id"))
                    if self._is_scalar_id(pid):
                        ids.add(pid)
            return ids
        if self._is_scalar_id(value):
            return {str(value)}
        return set()

    def _field_value(self, text: str, field: str) -> str:
        match = re.search(rf"(?:^|\s){re.escape(field)}=([^\s|;；,，]+)", text)
        return match.group(1).strip().lower() if match else ""

    def _mentions_player(self, text: str, player_id: str) -> bool:
        pid = re.escape(str(player_id).lower())
        return re.search(rf"(?<![A-Za-z0-9_]){pid}(?![A-Za-z0-9_])", str(text).lower()) is not None

    def _near_player_term(self, text: str, player_id: str, terms: tuple[str, ...]) -> bool:
        lower = str(text).lower()
        pid = str(player_id).lower()
        index = lower.find(pid)
        while index >= 0:
            window = lower[max(0, index - 16): index + len(pid) + 16]
            if any(term.lower() in window for term in terms):
                return True
            index = lower.find(pid, index + len(pid))
        return False

    def _append_bounded(self, items: list[str], value: str, limit: int) -> None:
        if value and value not in items:
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
        return normalized[: max(0, limit - 1)] + "…" if limit > 1 else normalized[:limit]

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
        text = "、".join(selected)
        if len(ids) > len(selected):
            text += f" 等{len(ids) - len(selected)}人"
        return text

    def _sort_player_ids(self, ids: Any) -> list[str]:
        unique = self._dedupe_preserve_order([str(item) for item in ids if str(item)])

        def sort_key(value: str) -> tuple[str, int, str]:
            match = _PLAYER_ID_SUFFIX.search(value)
            prefix = _PLAYER_ID_SUFFIX.sub("", value)
            return (prefix, int(match.group(1)) if match else 10**9, value)

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
            "id",
            "pid",
            "seat_id",
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

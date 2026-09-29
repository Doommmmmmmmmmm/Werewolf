"""狼人杀平民 Task-Agent（villager_3）。

目标：保持规则与工具预算，使用轻量公共台账和短提示，避免超时弃票。
本文件不访问网络、文件系统、环境变量、进程或隐藏状态；只处理引擎传入的公开/私有行动包。
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
_MAX_PROMPT_CHARS_HARD_CAP = 15000
_MAX_MODEL_OUTPUT_TOKENS = 900
_SPEECH_ACTION_KINDS = frozenset({"speak", "last_words"})

_PUBLIC_SNAPSHOT_LIMIT = 12
_PUBLIC_NOTE_CHAR_LIMIT = 120
_SUMMARY_CHAR_LIMIT = 1100
_FEEDBACK_CHAR_LIMIT = 500

CURRENT_ROUND_DIALOGUE_TOOL: dict[str, Any] = {
    "name": CURRENT_ROUND_DIALOGUE_TOOL_NAME,
    "description": "读取当前昼夜轮次中、当前玩家依法可见的已发生发言。",
    "parameters": {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    },
}


def _allowed_action_kinds(request: Mapping[str, Any]) -> set[str]:
    raw_actions = request.get("allowed_actions")
    if not isinstance(raw_actions, list):
        return set()
    return {
        str(item.get("kind") or "").lower()
        for item in raw_actions
        if isinstance(item, Mapping)
    }


def _is_vote_request(request: Mapping[str, Any]) -> bool:
    phase = str(request.get("phase") or request.get("channel") or request.get("kind") or "").lower()
    kinds = _allowed_action_kinds(request)
    return any(key in phase for key in ("vote", "投票")) or any("vote" in kind for kind in kinds)


def render_villager_decision_checklist(request: Mapping[str, Any]) -> str:
    """短清单：少字、强约束，避免模型长篇推理导致超时。"""

    kinds = _allowed_action_kinds(request)
    is_vote = _is_vote_request(request)
    has_speak = bool(kinds & _SPEECH_ACTION_KINDS) or any(
        key in str(request.get("phase") or "").lower() for key in ("speak", "discussion", "发言", "day")
    )
    lines = ["【平民原则】公开事实优先；查杀链/警徽链和票型高于印象；发言简短可复盘。"]
    if is_vote:
        lines.append("【投票硬要求】除非没有合法 target_id，必须出票，禁止主动 pass；优先投公开查杀、硬对跳矛盾点、冲票/票型异常者。")
    if has_speak:
        lines.append("【发言】点明1个关注对象、1个暂不投对象和对应公开依据，不编造系统事实。")
    return "\n".join(lines)


def render_action_contract(request: Mapping[str, Any]) -> str:
    """把 allowed_actions 转成当前可执行 JSON 模板。"""

    raw_actions = request.get("allowed_actions")
    allowed_actions = [item for item in raw_actions if isinstance(item, Mapping)] if isinstance(raw_actions, list) else []
    phase = str(request.get("phase") or "unknown")
    channel = str(request.get("channel") or "unknown")
    lines = [
        "【最终 JSON 契约】",
        f"阶段:{phase}；频道:{channel}。",
        "最终回答只能是一个 JSON 对象，不能有 Markdown、解释或多余字段。",
        "只能从下列模板选一项；target_id 只能取列出的候选值。",
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
        valid_targets = [str(target_id) for target_id in target_ids] if isinstance(target_ids, list) and target_ids else []
        if kind in _SPEECH_ACTION_KINDS:
            example["text"] = "好"
        elif valid_targets:
            example["target_id"] = valid_targets[0]
        lines.append("- `" + json.dumps(example, ensure_ascii=False, separators=(",", ":")) + "`")
        if valid_targets:
            lines.append("  target_id 候选:" + "、".join(valid_targets))
        if kind in _SPEECH_ACTION_KINDS:
            max_chars = allowed.get("max_chars")
            extra = []
            if max_chars is not None:
                extra.append(f"text≤{max_chars}字")
            if allowed.get("require_chinese"):
                extra.append("至少含中文")
            if extra:
                lines.append("  " + "；".join(extra))
    if _is_vote_request(request):
        lines.append("投票阶段若存在带 target_id 的投票模板，必须选择它并给出明确 target_id，不要 pass。")
    lines.append("kind/target_id 是协议字段，必须保留英文。")
    return "\n".join(lines)


def normalize_decision(raw: dict[str, Any] | None, request: dict[str, Any]) -> dict[str, Any]:
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
    allowed = next((item for item in request["allowed_actions"] if item["kind"] == action["kind"]), None)
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
    """平民任务求解器：轻量台账 + 短 prompt + 投票阶段快速出票。"""

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
        self.max_tokens = min(max(1, int(max_tokens)), _MAX_MODEL_OUTPUT_TOKENS)
        self.max_decision_retries = max(0, int(max_decision_retries))
        self.max_tool_calls_per_decision = min(max(0, int(max_tool_calls_per_decision)), DEFAULT_MAX_TOOL_CALLS_PER_DECISION)
        self.max_tool_result_tokens = min(max(1, int(max_tool_result_tokens)), DEFAULT_MAX_TOOL_RESULT_TOKENS)
        self.max_prompt_chars = min(max(1, int(max_prompt_chars)), _MAX_PROMPT_CHARS_HARD_CAP)
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

        # 白天投票采用公开信息确定性快速出票，避免模型超时导致 pass。
        fast_vote = self._fast_vote_action(turn_packet)
        if fast_vote is not None:
            return fast_vote

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
            user_content = {
                "instruction": self._turn_instruction(turn_packet, feedback),
                "packet": prompt,
            }
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
                feedback = "上一轮模型调用失败，请直接输出合法 JSON。"
                continue
            self._record_token_usage(raw)
            action = normalize_decision(raw, turn_packet["request"])
            action = self._repair_pass_vote(action, turn_packet)
            error = decision_error(action, turn_packet["request"])
            if error is None:
                return action
            if attempt >= self.max_decision_retries:
                raise RuleViolationError(
                    f"模型返回非法行动：{error}",
                    attempted_action=action,
                    validation_error=error,
                )
            feedback = self._trim_text(f"上一次行动未通过校验：{error}", _FEEDBACK_CHAR_LIMIT)

        raise RuleViolationError("模型未产生合法行动", attempted_action={}, validation_error="empty")

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
            parts.append(self._trim_text(f"校验反馈:{feedback}", _FEEDBACK_CHAR_LIMIT))
        parts.append(render_villager_decision_checklist(request))
        validation_feedback = self._trim_text("\n".join(part for part in parts if part), _SUMMARY_CHAR_LIMIT + _FEEDBACK_CHAR_LIMIT)
        return render_prompt(
            "player_turn_instruction.txt",
            action_contract=render_action_contract(request),
            validation_feedback=validation_feedback,
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
            "death_notices": [],
        }

    def _ingest_public_packet(self, payload: Any) -> None:
        notes = self._extract_public_notes(payload)
        self._merge_public_notes(notes)
        ids = self._extract_player_ids(payload, include_collections=True)
        self._public_ledger["known_player_ids"].update(ids)
        alive = self._collect_ids_by_keys(payload, self._ALIVE_KEYS(), max_depth=2)
        dead = self._collect_ids_by_keys(payload, self._DEAD_KEYS(), max_depth=2) | self._extract_dead_from_notes(notes)
        self._public_ledger["alive_ids"].update(alive)
        self._public_ledger["dead_ids"].update(dead)
        self._public_ledger["alive_ids"].difference_update(dead)

    def _merge_public_notes(self, notes: list[str]) -> None:
        for note in notes:
            if not note:
                continue
            self._append_bounded(self._public_ledger["recent_events"], note, _PUBLIC_SNAPSHOT_LIMIT)
            lower = note.lower()
            if "vote" in lower or "票" in note:
                self._append_bounded(self._public_ledger["vote_snapshots"], note, _PUBLIC_SNAPSHOT_LIMIT)
            if any(key in lower for key in ("sheriff", "badge")) or any(key in note for key in ("警长", "警徽")):
                self._append_bounded(self._public_ledger["sheriff_changes"], note, _PUBLIC_SNAPSHOT_LIMIT)
            if any(key in lower for key in ("death", "dead", "kill")) or any(key in note for key in ("死亡", "出局", "刀")):
                self._append_bounded(self._public_ledger["death_notices"], note, _PUBLIC_SNAPSHOT_LIMIT)
            if any(key in note for key in ("查杀", "金水", "查验", "预言家", "女巫", "猎人", "守卫", "对跳")):
                self._append_bounded(self._public_ledger["public_claims"], note, _PUBLIC_SNAPSHOT_LIMIT)

    def _render_turn_summary(self, turn_packet: Mapping[str, Any], request: Mapping[str, Any]) -> str:
        public_state = turn_packet.get("public_state")
        notes = self._extract_public_notes(public_state)
        dead_ids = self._sort_player_ids(list(set(self._public_ledger["dead_ids"]) | self._extract_dead_from_notes(notes)))
        alive_ids = self._sort_player_ids(
            list((set(self._public_ledger["alive_ids"]) | self._extract_player_ids(public_state, include_collections=True)) - set(dead_ids))
        )
        known_ids = self._sort_player_ids(list(set(self._public_ledger["known_player_ids"]) | set(alive_ids) | set(dead_ids)))
        candidates = self._candidate_pool(request, alive_ids, known_ids)
        lines = ["【轻量公共台账】", f"存活:{self._join_ids(alive_ids, limit=10)}", f"已死:{self._join_ids(dead_ids, limit=8)}"]
        claims = self._select_notes(notes, ("查杀", "金水", "查验", "预言家", "对跳", "警徽"), limit=3)
        if not claims:
            claims = list(self._public_ledger["public_claims"][-3:])
        if claims:
            lines.append("查验/警徽:" + " | ".join(self._trim_text(item, 70) for item in claims[:3]))
        vote_notes = self._select_notes(notes, ("vote", "投票", "票", "冲票"), limit=2)
        if vote_notes:
            lines.append("票型:" + " | ".join(self._trim_text(item, 65) for item in vote_notes[:2]))
        ranked = self._rank_candidates(candidates, notes + list(self._public_ledger["recent_events"][-6:]), dead_ids)
        if _is_vote_request(request):
            if ranked:
                top = "；".join(f"{pid}:{reason}" for pid, _score, reason in ranked[:3])
                lines.append("出票优先:" + top)
            lines.append("投票结论:必须从合法 target_id 中选一人，不要弃票。")
        else:
            focus = ranked[0] if ranked else ("信息不足", 0, "先听发言")
            protect = self._least_suspicious_candidate(candidates, notes, dead_ids)
            lines.append(f"发言关注:{focus[0]}({focus[2]})")
            if protect:
                lines.append(f"暂不投:{protect[0]}({protect[2]})")
        return self._trim_text("\n".join(lines), _SUMMARY_CHAR_LIMIT)

    def _fast_vote_action(self, turn_packet: Mapping[str, Any]) -> dict[str, Any] | None:
        request = turn_packet["request"]
        if not _is_vote_request(request):
            return None
        vote_action = self._target_action(request, prefer_vote=True)
        if vote_action is None:
            return None
        kind, target_ids = vote_action
        if not target_ids:
            return None
        target = self._choose_target(target_ids, turn_packet)
        return {
            "request_id": request["request_id"],
            "player_id": request["player_id"],
            "kind": kind,
            "target_id": target,
        }

    def _repair_pass_vote(self, action: dict[str, Any], turn_packet: Mapping[str, Any]) -> dict[str, Any]:
        if not _is_vote_request(turn_packet["request"]):
            return action
        has_target = any(isinstance(item, Mapping) and item.get("target_ids") for item in turn_packet["request"].get("allowed_actions", []))
        if has_target and (not action.get("target_id") or action.get("kind", "").lower() == "pass"):
            fixed = self._fast_vote_action(turn_packet)
            if fixed is not None:
                return fixed
        return action

    def _target_action(self, request: Mapping[str, Any], *, prefer_vote: bool) -> tuple[str, list[str]] | None:
        raw_actions = request.get("allowed_actions")
        if not isinstance(raw_actions, list):
            return None
        candidates: list[tuple[str, list[str]]] = []
        for item in raw_actions:
            if not isinstance(item, Mapping):
                continue
            kind = str(item.get("kind") or "")
            target_ids = item.get("target_ids")
            if kind and isinstance(target_ids, list) and target_ids:
                candidates.append((kind, [str(target_id) for target_id in target_ids]))
        if not candidates:
            return None
        if prefer_vote:
            for kind, target_ids in candidates:
                if "vote" in kind.lower() or "投票" in kind:
                    return kind, target_ids
        return candidates[0]

    def _choose_target(self, target_ids: list[str], turn_packet: Mapping[str, Any]) -> str:
        notes = self._extract_public_notes(turn_packet.get("public_state")) + list(self._public_ledger["recent_events"][-8:])
        dead_ids = self._sort_player_ids(list(self._public_ledger["dead_ids"]))
        legal = [pid for pid in self._sort_player_ids(target_ids) if pid != self.player_id and pid not in set(dead_ids)]
        if not legal:
            legal = [pid for pid in self._sort_player_ids(target_ids) if pid != self.player_id] or self._sort_player_ids(target_ids)
        ranked = self._rank_candidates(legal, notes, dead_ids)
        return ranked[0][0] if ranked else legal[0]

    def _rank_candidates(self, candidate_pool: list[str], notes: list[str], dead_ids: list[str]) -> list[tuple[str, int, str]]:
        dead = set(dead_ids)
        ranked: list[tuple[str, int, str]] = []
        for pid in self._sort_player_ids(candidate_pool):
            if pid == self.player_id or pid in dead:
                continue
            score, reason = self._score_candidate(pid, notes)
            ranked.append((pid, score, reason))
        ranked.sort(key=lambda item: (-item[1], item[0]))
        return ranked

    def _least_suspicious_candidate(self, candidate_pool: list[str], notes: list[str], dead_ids: list[str]) -> tuple[str, int, str] | None:
        ranked = self._rank_candidates(candidate_pool, notes, dead_ids)
        if not ranked:
            return None
        ranked.sort(key=lambda item: (item[1], item[0]))
        return ranked[0]

    def _score_candidate(self, candidate: str, notes: list[str]) -> tuple[int, str]:
        score = 0
        best_reason = "公开信息不足"
        lower_candidate = candidate.lower()
        for note in notes[-14:]:
            lower = note.lower()
            if lower_candidate not in lower:
                continue
            compact = self._trim_text(note, 58)
            if "查杀" in note or "被验狼" in note:
                return 100, "公开查杀/验狼:" + compact
            if any(term in note for term in ("金水", "好人", "银水", "强神", "守卫", "女巫", "猎人")):
                score -= 30
                best_reason = "保护链:" + compact
            elif any(term in note for term in ("对跳", "矛盾", "爆狼", "冲票", "票型异常", "反水", "悍跳")):
                score += 40
                best_reason = "矛盾/票型:" + compact
            elif "vote" in lower or "票" in note:
                score += 10
                best_reason = "票型相关:" + compact
            elif any(term in note for term in ("怀疑", "狼", "假", "踩")):
                score += 15
                best_reason = "发言疑点:" + compact
        return score, best_reason

    def _candidate_pool(self, request: Mapping[str, Any], alive_ids: list[str], known_ids: list[str]) -> list[str]:
        pool = set(alive_ids) | set(known_ids)
        raw_actions = request.get("allowed_actions")
        if isinstance(raw_actions, list):
            for action in raw_actions:
                if isinstance(action, Mapping) and isinstance(action.get("target_ids"), list):
                    pool.update(str(target_id) for target_id in action["target_ids"] if self._is_scalar_id(target_id))
        pool.discard(self.player_id)
        pool.difference_update(self._public_ledger["dead_ids"])
        return self._sort_player_ids(list(pool))

    def _extract_public_notes(self, payload: Any) -> list[str]:
        notes: list[str] = []
        self._walk_public_payload(payload, notes, depth=0)
        return self._dedupe_preserve_order(notes)[:_PUBLIC_SNAPSHOT_LIMIT]

    def _walk_public_payload(self, payload: Any, notes: list[str], depth: int) -> None:
        if depth > 2 or len(notes) >= _PUBLIC_SNAPSHOT_LIMIT:
            return
        if isinstance(payload, Mapping):
            snapshot = self._snapshot_mapping(payload)
            if snapshot:
                notes.append(snapshot)
            for value in list(payload.values())[:16]:
                self._walk_public_payload(value, notes, depth + 1)
                if len(notes) >= _PUBLIC_SNAPSHOT_LIMIT:
                    break
        elif isinstance(payload, list):
            for item in payload[:12]:
                self._walk_public_payload(item, notes, depth + 1)
                if len(notes) >= _PUBLIC_SNAPSHOT_LIMIT:
                    break

    def _snapshot_mapping(self, mapping: Mapping[str, Any]) -> str | None:
        lower = {str(key).lower(): value for key, value in mapping.items()}
        event = self._first_scalar(lower, ("event", "event_type", "type", "kind", "action"))
        event_l = event.lower()
        text = self._first_scalar(lower, ("text", "content", "message", "speech", "reason"))
        has_core_text = any(term in text for term in ("查杀", "金水", "查验", "预言家", "对跳", "警徽", "投票", "冲票"))
        vote_like = "vote" in event_l or any(key in lower for key in ("voter_id", "vote_target", "ballot"))
        death_like = any(key in lower for key in ("dead_id", "dead_player_id", "death", "dead")) or any(term in event_l for term in ("death", "dead", "kill", "out"))
        sheriff_like = "sheriff" in event_l or "badge" in event_l or any(key in lower for key in ("sheriff_id", "badge_holder", "badge_to"))
        claim_like = has_core_text or any(key in lower for key in ("claim", "role_claim", "claimed_role"))
        if not (vote_like or death_like or sheriff_like or claim_like):
            return None

        player = self._first_scalar(lower, ("player_id", "speaker_id", "voter_id", "claimer_id", "actor_id", "source_id"))
        target = self._first_scalar(lower, ("target_id", "target", "vote_target", "to_id", "suspect_id"))
        role = self._first_scalar(lower, ("role", "claim_role", "claimed_role"))
        parts: list[str] = []
        if event:
            parts.append(event_l)
        if vote_like:
            parts.append(f"vote voter={player or '?'} target={target or '?'}")
        elif death_like:
            dead = self._first_scalar(lower, ("dead_id", "dead_player_id")) or player or target
            parts.append(f"death player={dead or '?'}")
        elif sheriff_like:
            holder = self._first_scalar(lower, ("badge_holder", "badge_holder_id", "sheriff_id"))
            parts.append(f"sheriff holder={holder or '?'} to={target or '?'}")
        elif claim_like:
            parts.append(f"claim speaker={player or '?'}")
            if role:
                parts.append(f"role={role}")
        if text:
            parts.append("text=" + self._trim_text(text, 70))
        return self._trim_text(" ".join(parts), _PUBLIC_NOTE_CHAR_LIMIT)

    def _extract_player_ids(self, payload: Any, *, include_collections: bool = False) -> set[str]:
        ids: set[str] = set()

        def walk(obj: Any, depth: int) -> None:
            if depth > 2:
                return
            if isinstance(obj, Mapping):
                lower = {str(key).lower(): value for key, value in obj.items()}
                for key, value in lower.items():
                    if key in self._PLAYER_ID_KEYS() and self._is_scalar_id(value):
                        ids.add(str(value))
                    if include_collections and key in self._COLLECTION_ID_KEYS():
                        ids.update(self._as_id_set(value))
                for value in list(obj.values())[:16]:
                    walk(value, depth + 1)
            elif isinstance(obj, list):
                for item in obj[:12]:
                    walk(item, depth + 1)

        walk(payload, 0)
        ids.discard(self.player_id)
        return ids

    def _collect_ids_by_keys(self, payload: Any, allowed_keys: set[str], *, max_depth: int) -> set[str]:
        found: set[str] = set()

        def walk(obj: Any, depth: int) -> None:
            if depth > max_depth:
                return
            if isinstance(obj, Mapping):
                for key, value in obj.items():
                    lower_key = str(key).lower()
                    if lower_key in allowed_keys:
                        found.update(self._as_id_set(value))
                    if isinstance(value, (Mapping, list)):
                        walk(value, depth + 1)
            elif isinstance(obj, list):
                for item in obj[:12]:
                    walk(item, depth + 1)

        walk(payload, 0)
        return found

    def _extract_dead_from_notes(self, notes: list[str]) -> set[str]:
        dead: set[str] = set()
        for note in notes:
            lower = note.lower()
            if any(term in lower for term in ("death", "dead", "kill")) or any(term in note for term in ("死亡", "出局", "淘汰")):
                dead.update(self._ids_from_text(note))
        return dead

    def _ids_from_text(self, text: str) -> set[str]:
        ids: set[str] = set()
        for token in re.split(r"[\s,，;；|/\\()\[\]{}<>：:=。,.、]+", str(text)):
            if token and self._looks_like_player_id(token):
                ids.add(token)
        return ids

    def _select_notes(self, notes: list[str], keywords: tuple[str, ...], *, limit: int) -> list[str]:
        selected: list[str] = []
        for note in notes:
            lower = note.lower()
            if any(keyword in note or keyword in lower for keyword in keywords):
                selected.append(note)
                if len(selected) >= limit:
                    break
        return selected

    def _as_id_set(self, value: Any) -> set[str]:
        if isinstance(value, (list, tuple, set)):
            return {str(item) for item in value if self._is_scalar_id(item)}
        if self._is_scalar_id(value):
            return {str(value)}
        return set()

    def _append_bounded(self, items: list[str], value: str, limit: int) -> None:
        if not value or value in items:
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
        return text + (f" 等{extra}人" if extra > 0 else "")

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

    @staticmethod
    def _PLAYER_ID_KEYS() -> set[str]:
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
    def _COLLECTION_ID_KEYS() -> set[str]:
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
        }

    @staticmethod
    def _ALIVE_KEYS() -> set[str]:
        return {"alive", "alive_ids", "alive_player_ids", "survivors", "living", "live_ids", "current_alive"}

    @staticmethod
    def _DEAD_KEYS() -> set[str]:
        return {"dead", "dead_ids", "dead_player_ids", "eliminated", "casualties", "deaths", "graveyard"}

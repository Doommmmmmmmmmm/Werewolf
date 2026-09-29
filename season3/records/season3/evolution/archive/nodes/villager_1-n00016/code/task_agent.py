"""Villager Task-Agent: compact, public-information-only decision helper."""

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
_NOTE_LIMIT = 100
_LEDGER_ITEMS = 10

CURRENT_ROUND_DIALOGUE_TOOL: dict[str, Any] = {
    "name": CURRENT_ROUND_DIALOGUE_TOOL_NAME,
    "description": "读取当前昼夜轮次中、当前玩家依法可见的已发生发言。",
    "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
}


def render_villager_decision_checklist(request: Mapping[str, Any]) -> str:
    """Short checklist; no hidden info, no history stuffing."""

    phase = str(request.get("phase") or request.get("channel") or request.get("kind") or "").lower()
    actions = request.get("allowed_actions")
    kinds = {
        str(item.get("kind") or "").lower()
        for item in actions
        if isinstance(item, Mapping)
    } if isinstance(actions, list) else set()
    is_vote = "投票" in phase or "vote" in phase or any("vote" in kind for kind in kinds)
    is_speak = any(x in phase for x in ("speak", "discussion", "发言", "day", "白天")) or bool(kinds & {"speak", "discussion", "day_speak"})
    lines = ["【平民短清单】只用公开事实/玩家声明/自己推测三类信息；必要历史让模型调用当前轮工具，不拼完整历史。"]
    if is_vote:
        lines.append("投票先比至少两人：优先出未被金水/强神/可信警徽链保护，且有查杀、矛盾、冲票或事实错误者。")
        lines.append("严禁惯性 pass；你持警徽或残局/决选时，必须在合法候选中投最高疑点者。")
    if is_speak:
        lines.append("发言给出一个关注对象、一个暂不投对象，并说明依据的公开信息。")
    return "\n".join(lines)


def render_action_contract(request: Mapping[str, Any]) -> str:
    """Render only the legal JSON templates for this turn."""

    raw = request.get("allowed_actions")
    allowed = [item for item in raw if isinstance(item, Mapping)] if isinstance(raw, list) else []
    lines = [
        "【最终 JSON 契约】只能输出一个 JSON 对象，无 Markdown/解释/额外字段。",
        f"阶段={request.get('phase') or 'unknown'}；频道={request.get('channel') or 'unknown'}。",
    ]
    if not allowed:
        return "\n".join(lines + ["当前没有合法行动；不要臆造。"])
    for item in allowed:
        kind = str(item.get("kind") or "")
        if not kind:
            continue
        example: dict[str, str] = {"kind": kind}
        targets = [str(x) for x in item.get("target_ids", [])] if isinstance(item.get("target_ids"), list) else []
        if kind in _SPEECH_ACTION_KINDS:
            example["text"] = "好"
        elif targets:
            example["target_id"] = targets[0]
        lines.append("- " + json.dumps(example, ensure_ascii=False, separators=(",", ":")))
        if targets:
            lines.append("  target_id 只能取：" + "、".join(targets))
        if kind in _SPEECH_ACTION_KINDS:
            constraints = []
            if item.get("max_chars") is not None:
                constraints.append(f"text≤{item.get('max_chars')}字符")
            if item.get("require_chinese"):
                constraints.append("含中文")
            lines.append("  " + "；".join(constraints or ["非空发言"]))
    lines.append("kind/target_id 是协议字段，必须保留英文原值。")
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
    """A minimal villager solver. It stores only compact public facts."""

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
        self._model_token_usage = {
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
        current_dialogue = (turn_packet.get("tool_context") or {}).get("current_round_dialogue") or []

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
                feedback = "上一轮模型调用没有产生可用结构化行动，请直接重新作答。"
                continue
            self._record_token_usage(raw)
            action = normalize_decision(raw, turn_packet["request"])
            action = self._avoid_critical_pass(action, turn_packet)
            error = decision_error(action, turn_packet["request"])
            if error is None:
                return action
            if attempt >= self.max_decision_retries:
                raise RuleViolationError(f"模型返回非法行动：{error}", attempted_action=action, validation_error=error)
            feedback = f"上一次行动未通过校验：{error}"
        raise RuleViolationError("模型未返回可用行动")

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
            parts.append("校验反馈：" + feedback)
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
        }

    def _ingest_public_packet(self, payload: Any) -> None:
        notes = self._extract_public_notes(payload)
        for note in notes:
            self._add_note(note)
        self._public_ledger["known_player_ids"].update(self._extract_player_ids(payload))
        if isinstance(payload, Mapping):
            self._public_ledger["alive_ids"].update(self._collect_ids_by_keys(payload, self._ALIVE_KEYS()))
            self._public_ledger["dead_ids"].update(self._collect_ids_by_keys(payload, self._DEAD_KEYS()))

    def _add_note(self, note: str) -> None:
        note = self._trim_text(note, _NOTE_LIMIT)
        self._append_bounded(self._public_ledger["recent_events"], note, _LEDGER_ITEMS)
        lower = note.lower()
        if "vote" in lower or "票" in note:
            self._append_bounded(self._public_ledger["vote_snapshots"], note, _LEDGER_ITEMS)
        if any(x in lower for x in ("sheriff", "badge")) or any(x in note for x in ("警长", "警徽")):
            self._append_bounded(self._public_ledger["sheriff_changes"], note, _LEDGER_ITEMS)
        if any(x in note for x in ("claim", "自称", "预言家", "女巫", "猎人", "守卫", "金水", "查杀", "查验")):
            self._append_bounded(self._public_ledger["public_claims"], note, _LEDGER_ITEMS)

    def _render_turn_summary(self, turn_packet: Mapping[str, Any], request: Mapping[str, Any]) -> str:
        public_state = turn_packet.get("public_state")
        notes = self._extract_public_notes(public_state)
        alive = self._sort_player_ids((set(self._public_ledger["alive_ids"]) | self._extract_player_ids(public_state, include_collections=True)) - set(self._public_ledger["dead_ids"]))
        dead = self._sort_player_ids(set(self._public_ledger["dead_ids"]) | self._dead_from_notes(notes))
        known = self._sort_player_ids((set(self._public_ledger["known_player_ids"]) | set(alive)) - {self.player_id})
        pool = self._candidate_pool(request, alive, known)
        claim_text = " | ".join((notes + self._public_ledger["public_claims"])[:3])
        lines = [
            "【公共台账摘要】",
            "存活: " + self._join_ids(alive, 10),
            "已死: " + self._join_ids(dead, 10),
        ]
        if claim_text:
            lines.append("链/声明: " + self._trim_text(claim_text, 240))
        if self._is_vote_request(request):
            ranked = self._rank_candidates(pool or alive, notes + self._public_ledger["recent_events"], dead)[:2]
            if ranked:
                lines.append("投票参考: " + "；".join(f"{pid}({score},{reason})" for pid, score, reason in ranked))
            lines.append("关键原则: 持警徽/残局/决选不 pass，投疑点最高的合法候选。")
        else:
            ranked = self._rank_candidates(pool or alive, notes + self._public_ledger["recent_events"], dead)
            if ranked:
                avoid = self._least_suspicious_candidate(pool or alive, notes, dead)
                text = f"发言参考: 关注 {ranked[0][0]}；"
                if avoid:
                    text += f"暂不投 {avoid[0]}。"
                lines.append(text)
        return self._trim_text("\n".join(lines), 1400)

    def _avoid_critical_pass(self, action: dict[str, Any], turn_packet: Mapping[str, Any]) -> dict[str, Any]:
        request = turn_packet["request"]
        if not self._is_vote_request(request) or action.get("kind", "").lower() not in {"pass", "abstain", "skip"}:
            return action
        vote_action = self._vote_action(request)
        if vote_action is None:
            return action
        targets = [str(x) for x in vote_action.get("target_ids", []) if str(x) != self.player_id]
        if not targets:
            return action
        alive_count = len(self._extract_player_ids(turn_packet.get("public_state"), include_collections=True) or self._public_ledger["alive_ids"])
        critical = self._is_self_sheriff(turn_packet) or alive_count <= 5 or len(targets) <= 3
        if not critical:
            return action
        target = self._best_vote_target(turn_packet, targets)
        return {"request_id": request["request_id"], "player_id": request["player_id"], "kind": str(vote_action["kind"]), "target_id": target}

    def _best_vote_target(self, turn_packet: Mapping[str, Any], targets: list[str]) -> str:
        notes = self._extract_public_notes(turn_packet.get("public_state")) + self._public_ledger["recent_events"]
        dead = self._sort_player_ids(self._public_ledger["dead_ids"] | self._dead_from_notes(notes))
        ranked = self._rank_candidates(targets, notes, dead)
        return ranked[0][0] if ranked else self._sort_player_ids(targets)[0]

    def _rank_candidates(self, candidates: list[str], notes: list[str], dead_ids: list[str]) -> list[tuple[str, int, str]]:
        ranked = []
        for candidate in self._sort_player_ids(candidates):
            if candidate == self.player_id or candidate in dead_ids:
                continue
            score, reasons = self._score_candidate(candidate, notes)
            ranked.append((candidate, score, "；".join(reasons[:2]) if reasons else "公开信息不足"))
        ranked.sort(key=lambda item: (-item[1], item[0]))
        return ranked

    def _least_suspicious_candidate(self, candidates: list[str], notes: list[str], dead_ids: list[str]) -> tuple[str, int, str] | None:
        ranked = self._rank_candidates(candidates, notes, dead_ids)
        if not ranked:
            return None
        ranked.sort(key=lambda item: (item[1], item[0]))
        return ranked[0]

    def _score_candidate(self, candidate: str, notes: list[str]) -> tuple[int, list[str]]:
        score = 0
        reasons: list[str] = []
        suspicious = ("查杀", "对跳", "爆狼", "矛盾", "票型异常", "反水", "不一致", "冲票", "狼")
        protected = ("金水", "可信", "报验链", "警徽链", "验过", "已验", "好人链", "强神")
        for note in notes[:24]:
            if candidate.lower() not in note.lower():
                continue
            compact = self._trim_text(note, 64)
            score += 1
            if any(term in note for term in suspicious):
                score += 3
                reasons.append("疑点:" + compact)
            elif any(term in note for term in protected):
                score -= 3
                reasons.append("保护:" + compact)
            elif "vote" in note.lower() or "票" in note:
                score += 1
                reasons.append("票型:" + compact)
            else:
                reasons.append(compact)
        return (score, reasons or ["公开信息不足"])

    def _candidate_pool(self, request: Mapping[str, Any], alive_ids: list[str], known_ids: list[str]) -> list[str]:
        pool = set(alive_ids) | set(known_ids)
        raw = request.get("allowed_actions")
        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, Mapping) and isinstance(item.get("target_ids"), list):
                    pool.update(str(x) for x in item["target_ids"] if self._is_scalar_id(x))
        pool.discard(self.player_id)
        pool.difference_update(self._public_ledger["dead_ids"])
        return self._sort_player_ids(pool)

    def _is_vote_request(self, request: Mapping[str, Any]) -> bool:
        phase = str(request.get("phase") or request.get("channel") or request.get("kind") or "").lower()
        if "vote" in phase or "投票" in phase:
            return True
        raw = request.get("allowed_actions")
        return isinstance(raw, list) and any(isinstance(a, Mapping) and "vote" in str(a.get("kind", "")).lower() for a in raw)

    def _vote_action(self, request: Mapping[str, Any]) -> Mapping[str, Any] | None:
        raw = request.get("allowed_actions")
        if not isinstance(raw, list):
            return None
        for item in raw:
            if isinstance(item, Mapping) and "vote" in str(item.get("kind", "")).lower() and isinstance(item.get("target_ids"), list) and item.get("target_ids"):
                return item
        return None

    def _is_self_sheriff(self, turn_packet: Mapping[str, Any]) -> bool:
        self_id = self.player_id
        for obj in (turn_packet.get("self"), turn_packet.get("public_state"), turn_packet.get("game")):
            for value in self._find_values(obj, {"sheriff_id", "badge_holder", "badge_holder_id", "current_sheriff", "警长", "警徽持有者"}):
                if str(value) == self_id:
                    return True
        return False

    def _extract_public_notes(self, payload: Any) -> list[str]:
        notes: list[str] = []
        self._walk_public_payload(payload, notes, 0)
        return self._dedupe(notes)[:24]

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
            for item in payload[:20]:
                self._walk_public_payload(item, notes, depth + 1)

    def _snapshot_mapping(self, mapping: Mapping[str, Any]) -> str | None:
        lower = {str(k).lower(): v for k, v in mapping.items()}
        event = self._first_scalar(lower, ("event", "event_type", "type", "kind", "action"))
        event_l = event.lower()
        vote = any(k in lower for k in ("voter_id", "vote_target", "target_id", "ballot", "vote")) or "vote" in event_l
        claim = any(k in lower for k in ("speaker_id", "claimer_id", "claim", "role_claim", "speech", "text", "content", "message")) or any(t in event_l for t in ("claim", "speak", "speech", "发言", "自称"))
        sheriff = any(k in lower for k in ("sheriff_id", "badge_holder", "badge_holder_id", "badge_from", "badge_to")) or any(t in event_l for t in ("sheriff", "badge", "警徽", "警长"))
        death = any(k in lower for k in ("dead", "death", "dead_id", "dead_player_id")) or any(t in event_l for t in ("death", "dead", "kill", "死亡", "出局"))
        if not (vote or claim or sheriff or death):
            return None
        player = self._first_scalar(lower, ("player_id", "speaker_id", "voter_id", "claimer_id", "actor_id", "source_id"))
        target = self._first_scalar(lower, ("target_id", "target", "vote_target", "to_id", "suspect_id"))
        role = self._first_scalar(lower, ("role", "claim_role", "claimed_role"))
        text = self._first_scalar(lower, ("text", "content", "message", "reason", "speech"))
        parts = [event_l] if event_l else []
        if vote:
            parts.append(f"vote voter={player or '?'} target={target or '?'}")
        elif sheriff:
            holder = self._first_scalar(lower, ("badge_holder", "badge_holder_id", "sheriff_id"))
            parts.append(f"sheriff holder={holder or '?'} target={target or '?'}")
        elif death:
            parts.append(f"death player={player or self._first_scalar(lower, ('dead_id', 'dead_player_id')) or '?'}")
        elif claim:
            parts.append(f"claim speaker={player or '?'}")
            if role:
                parts.append(f"role={role}")
        if text:
            parts.append("text=" + self._trim_text(text, _NOTE_LIMIT))
        return self._trim_text(" ".join(parts), _NOTE_LIMIT)

    def _extract_player_ids(self, payload: Any, *, include_collections: bool = False) -> set[str]:
        ids: set[str] = set()
        player_keys = self._PLAYER_ID_KEYS()
        collection_keys = self._COLLECTION_ID_KEYS()

        def walk(obj: Any) -> None:
            if isinstance(obj, Mapping):
                for key, value in obj.items():
                    key_l = str(key).lower()
                    if key_l in player_keys and self._is_scalar_id(value):
                        ids.add(str(value))
                    if include_collections and key_l in collection_keys:
                        ids.update(self._as_id_set(value))
                for value in obj.values():
                    walk(value)
            elif isinstance(obj, list):
                for item in obj[:20]:
                    walk(item)

        walk(payload)
        ids.discard(self.player_id)
        return ids

    def _collect_ids_by_keys(self, mapping: Mapping[str, Any], keys: set[str]) -> set[str]:
        found: set[str] = set()
        for key, value in mapping.items():
            if str(key).lower() in keys:
                found.update(self._as_id_set(value))
        return found

    def _dead_from_notes(self, notes: list[str]) -> set[str]:
        dead: set[str] = set()
        for note in notes:
            if any(term in note.lower() for term in ("death", "dead", "死亡", "出局", "淘汰")):
                dead.update(self._ids_from_text(note))
        return dead

    def _ids_from_text(self, text: str) -> set[str]:
        ids: set[str] = set()
        for token in re.split(r"[\s,，;；|/\\]+", text):
            token = token.strip("()[]{}<>：:=，。,.、")
            if self._looks_like_player_id(token):
                ids.add(token)
        return ids

    def _find_values(self, payload: Any, keys: set[str]) -> list[Any]:
        values: list[Any] = []
        if isinstance(payload, Mapping):
            for key, value in payload.items():
                if str(key).lower() in keys or str(key) in keys:
                    values.append(value)
                values.extend(self._find_values(value, keys))
        elif isinstance(payload, list):
            for item in payload[:20]:
                values.extend(self._find_values(item, keys))
        return values

    def _as_id_set(self, value: Any) -> set[str]:
        if isinstance(value, (list, tuple, set)):
            return {str(item) for item in value if self._is_scalar_id(item)}
        return {str(value)} if self._is_scalar_id(value) else set()

    def _first_scalar(self, mapping: Mapping[str, Any], keys: tuple[str, ...]) -> str:
        for key in keys:
            value = mapping.get(key)
            if self._is_scalar_text(value):
                return self._trim_text(str(value), _NOTE_LIMIT)
        return ""

    @staticmethod
    def _is_scalar_text(value: Any) -> bool:
        return isinstance(value, (str, int, float)) and not isinstance(value, bool)

    def _is_scalar_id(self, value: Any) -> bool:
        if isinstance(value, bool):
            return False
        if isinstance(value, int):
            return True
        return isinstance(value, str) and self._looks_like_player_id(value)

    def _trim_text(self, text: str, limit: int) -> str:
        normalized = " ".join(str(text).split())
        return normalized if len(normalized) <= limit else normalized[: max(1, limit - 1)] + "…"

    def _append_bounded(self, items: list[str], value: str, limit: int) -> None:
        if value and value not in items:
            items.append(value)
            if len(items) > limit:
                del items[: len(items) - limit]

    def _dedupe(self, items: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for item in items:
            if item and item not in seen:
                seen.add(item)
                out.append(item)
        return out

    def _join_ids(self, ids: list[str], limit: int) -> str:
        if not ids:
            return "未知"
        selected = ids[:limit]
        return "、".join(selected) + (f" 等{len(ids) - len(selected)}人" if len(ids) > len(selected) else "")

    def _sort_player_ids(self, ids: Any) -> list[str]:
        unique = self._dedupe([str(item) for item in ids if str(item)])

        def sort_key(value: str) -> tuple[str, int, str]:
            match = _PLAYER_ID_SUFFIX.search(value)
            prefix = _PLAYER_ID_SUFFIX.sub("", value)
            return (prefix, int(match.group(1)) if match else 10**9, value)

        return sorted(unique, key=sort_key)

    @staticmethod
    def _looks_like_player_id(value: str) -> bool:
        text = str(value).strip()
        return bool(text) and len(text) <= 16 and any(ch.isdigit() for ch in text)

    @staticmethod
    def _PLAYER_ID_KEYS() -> set[str]:
        return {"player_id", "speaker_id", "voter_id", "claimer_id", "actor_id", "source_id", "target_id", "from_id", "to_id", "badge_holder", "badge_holder_id", "sheriff_id", "dead_id", "dead_player_id"}

    @staticmethod
    def _COLLECTION_ID_KEYS() -> set[str]:
        return {"alive", "alive_ids", "alive_player_ids", "dead", "dead_ids", "dead_player_ids", "players", "player_ids", "target_ids", "candidates", "survivors", "living", "graveyard", "claims", "votes"}

    @staticmethod
    def _ALIVE_KEYS() -> set[str]:
        return {"alive", "alive_ids", "alive_player_ids", "survivors", "living", "live_ids", "current_alive"}

    @staticmethod
    def _DEAD_KEYS() -> set[str]:
        return {"dead", "dead_ids", "dead_player_ids", "eliminated", "casualties", "deaths", "graveyard"}

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

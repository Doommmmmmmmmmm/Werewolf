"""狼人杀平民 Task-Agent。

只维护从同步包中可见的轻量公共台账；每次决策仍是新模型会话，历史只能通过
有界台账摘要和当前轮受限工具进入提示，避免拼接完整历史绕过预算。
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
_ID_IN_TEXT = re.compile(r"(?:^|[^A-Za-z0-9])(?:p|P)?(\d{1,2})(?:号|號)?")

CURRENT_ROUND_DIALOGUE_TOOL_NAME = "read_current_round_dialogue"
DEFAULT_MAX_TOOL_CALLS_PER_DECISION = 5
DEFAULT_MAX_TOOL_RESULT_TOKENS = 1000
DEFAULT_MAX_PROMPT_CHARS = 12000
DEFAULT_MAX_DECISION_RETRIES = 2
_SPEECH_ACTION_KINDS = frozenset({"speak", "last_words"})

_PUBLIC_SNAPSHOT_LIMIT = 18
_PUBLIC_NOTE_CHAR_LIMIT = 140
_PUBLIC_BLOCK_CHAR_LIMIT = 1600
_PHASE_BLOCK_CHAR_LIMIT = 900

CURRENT_ROUND_DIALOGUE_TOOL: dict[str, Any] = {
    "name": CURRENT_ROUND_DIALOGUE_TOOL_NAME,
    "description": "读取当前昼夜轮次中、当前玩家依法可见的已发生发言。",
    "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
}


def render_villager_decision_checklist(request: Mapping[str, Any]) -> str:
    phase = str(request.get("phase") or request.get("channel") or request.get("kind") or "")
    phase_lower = phase.lower()
    allowed_actions = request.get("allowed_actions")
    allowed_kinds = {
        str(item.get("kind") or "").lower()
        for item in allowed_actions
        if isinstance(item, Mapping)
    } if isinstance(allowed_actions, list) else set()
    has_vote = any(key in phase_lower for key in ("vote", "投票")) or any("vote" in kind for kind in allowed_kinds)
    has_speak = any(key in phase_lower for key in ("speak", "discussion", "发言", "白天", "day")) or any(
        kind in {"speak", "discussion", "day_speak"} for kind in allowed_kinds
    )
    has_last_words = any(key in phase_lower for key in ("last", "遗言")) or "last_words" in allowed_kinds
    has_sheriff = any(key in phase_lower for key in ("sheriff", "警长", "警徽")) or any(
        key in allowed_kinds for key in ("sheriff_vote", "sheriff")
    )
    lines = [
        "【平民决策清单】先分系统事实 / 玩家声明 / 自己推测；不要把当前一轮聊天当成全部历史。",
        "查验链要看主语和宾语：谁报验、谁被查杀/金水；硬对跳未决时保留弹性，但不要无视首验查杀和警徽链。",
    ]
    if has_vote:
        lines.append("投票优先级：明确查杀且未被强力推翻 > 警长/持徽好人归票 > 票型冲票异常 > 普通发言疑点；避免散票。")
        lines.append("默认保护警长、已公开金水、可信报验者、强神声明者；除非硬矛盾或事实错误，不优先投他们。")
    if has_speak:
        lines.append("发言至少点出一个关注对象和一个暂不投对象，并说明依据哪条公开信息。")
    if has_last_words:
        lines.append("遗言只留公开事实、票型、查验链和怀疑对象，不把推测说成系统事实。")
    if has_sheriff:
        lines.append("警长/警徽回合优先看报验链是否清晰一致；接徽时继承公开遗产但不要当系统确认。")
    lines.append("残局或持警徽时，先复盘存活、已死、可信查验链、灰区，再决定出谁。")
    return "\n".join(lines)


def render_action_contract(request: Mapping[str, Any]) -> str:
    raw_actions = request.get("allowed_actions")
    allowed_actions = [item for item in raw_actions if isinstance(item, Mapping)] if isinstance(raw_actions, list) else []
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
        valid_targets = [str(target_id) for target_id in target_ids] if isinstance(target_ids, list) and target_ids else []
        if kind in _SPEECH_ACTION_KINDS:
            example["text"] = "好"
        elif valid_targets:
            example["target_id"] = valid_targets[0]
        lines.append("- `" + json.dumps(example, ensure_ascii=False, separators=(",", ":")) + "`")
        if valid_targets:
            lines.append("  target_id 候选值：" + "、".join(valid_targets) + "。")
        if kind in _SPEECH_ACTION_KINDS:
            constraints: list[str] = []
            if allowed.get("max_chars") is not None:
                constraints.append(f"text 最多 {allowed['max_chars']} 个字符")
            if allowed.get("require_chinese"):
                constraints.append("text 至少包含一个中文字符")
            constraints.append("text 可以包含英文、数字和玩家编号")
            lines.append("  " + "；".join(constraints) + "。")
    lines.append("kind 和 target_id 是协议字段，使用英文是合法且必须原样保留；不要把它们翻译成中文。")
    return "\n".join(lines)


def normalize_decision(raw: dict[str, Any] | None, request: dict[str, Any]) -> dict[str, Any]:
    source = raw.get("decision", raw) if isinstance(raw, dict) else {}
    action = {
        "request_id": request["request_id"],
        "player_id": request["player_id"],
        "kind": str(source.get("kind", source.get("action", ""))),
    }
    target_id = source.get("target_id", source.get("targetId"))
    if target_id is not None and str(target_id):
        action["target_id"] = str(target_id)
    if isinstance(source.get("text"), str):
        action["text"] = source["text"]
    return action


def decision_error(action: dict[str, Any], request: dict[str, Any]) -> str | None:
    allowed_actions = request.get("allowed_actions") or []
    allowed = next((item for item in allowed_actions if isinstance(item, Mapping) and item.get("kind") == action["kind"]), None)
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
        if str(action.get("target_id")) not in {str(target_id) for target_id in target_ids}:
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
        self.max_tokens = min(900, max(1, int(max_tokens)))
        self.max_decision_retries = max(0, int(max_decision_retries))
        self.max_tool_calls_per_decision = min(5, max(0, int(max_tool_calls_per_decision)))
        self.max_tool_result_tokens = min(1000, max(1, int(max_tool_result_tokens)))
        self.max_prompt_chars = min(15000, max(1, int(max_prompt_chars)))
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

        system = self._system_prompt(private)
        prompt = {
            "game": turn_packet["game"],
            "public_rules": turn_packet["public_rules"],
            "self": turn_packet["self"],
            "private_information": private,
            "public_state": turn_packet["public_state"],
            "request": turn_packet["request"],
            "history_policy": {
                "default_context": "current_state_only_plus_bounded_public_ledger",
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
                feedback = "上一轮模型调用没有产生可用的结构化行动，请直接重新作答。"
                continue
            self._record_token_usage(raw)
            action = normalize_decision(raw, turn_packet["request"])
            error = decision_error(action, turn_packet["request"])
            if error is None:
                return action
            if attempt >= self.max_decision_retries:
                raise RuleViolationError(f"模型返回非法行动：{error}", attempted_action=action, validation_error=error)
            feedback = f"上一次行动未通过校验：{error}"
        raise RuleViolationError("模型未能返回合法行动")

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
            "seer_claimers": set(),
            "inspection_claims": [],
            "sheriff_id": "",
            "sheriff_calls": [],
        }

    def _ingest_public_packet(self, payload: Any) -> None:
        notes = self._extract_public_notes(payload)
        player_ids = self._extract_player_ids(payload, include_collections=True)
        self._public_ledger["known_player_ids"].update(player_ids)
        if isinstance(payload, Mapping):
            self._public_ledger["alive_ids"].update(self._collect_id_collection(payload, self._ALIVE_KEYS()))
            self._public_ledger["dead_ids"].update(self._collect_id_collection(payload, self._DEAD_KEYS()))
        self._merge_public_notes(notes)

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
                holder = self._extract_field(note, "holder") or self._extract_field(note, "to")
                if holder and holder != "?":
                    self._public_ledger["sheriff_id"] = holder
            if any(key in lower for key in ("badge", "警徽", "transfer", "转移", "移交")):
                self._append_bounded(self._public_ledger["badge_transfers"], note, _PUBLIC_SNAPSHOT_LIMIT)
            if any(key in lower for key in ("death", "dead", "出局", "死亡", "刀")):
                self._append_bounded(self._public_ledger["death_notices"], note, _PUBLIC_SNAPSHOT_LIMIT)
                self._public_ledger["dead_ids"].update(self._ids_from_text(note))
            if any(key in lower for key in ("claim", "自称", "预言家", "女巫", "猎人", "守卫", "金水", "查杀", "查验", "警徽", "归票")):
                self._append_bounded(self._public_ledger["public_claims"], note, _PUBLIC_SNAPSHOT_LIMIT)
            self._ingest_claim_semantics(note)

    def _ingest_claim_semantics(self, note: str) -> None:
        speaker = self._extract_field(note, "speaker") or self._extract_field(note, "player")
        text = self._extract_text_field(note) or note
        if speaker and ("预言家" in text or "role=预言家" in note or "seer" in note.lower()):
            self._public_ledger["seer_claimers"].add(speaker)
        for source, target, result in self._extract_inspections(note, speaker):
            event = {"source": source, "target": target, "result": result, "note": self._trim_text(note, 100)}
            if event not in self._public_ledger["inspection_claims"]:
                self._append_bounded(self._public_ledger["inspection_claims"], event, _PUBLIC_SNAPSHOT_LIMIT)
        sheriff_id = self._public_ledger.get("sheriff_id") or ""
        if speaker and sheriff_id and self._same_player(speaker, sheriff_id) and any(term in text for term in ("归票", "出", "投", "放逐")):
            targets = [pid for pid in self._ids_from_text(text) if not self._same_player(pid, speaker)]
            if targets:
                self._append_bounded(self._public_ledger["sheriff_calls"], {"source": speaker, "target": targets[-1], "note": self._trim_text(note, 100)}, 6)

    def _render_turn_summary(self, turn_packet: Mapping[str, Any], request: Mapping[str, Any]) -> str:
        public_state = turn_packet.get("public_state")
        turn_notes = self._extract_public_notes(public_state)
        local_events = self._build_local_events(turn_notes)
        ledger_alive = set(self._public_ledger["alive_ids"])
        ledger_dead = set(self._public_ledger["dead_ids"])
        current_alive = self._extract_player_ids(public_state, include_collections=True)
        combined_alive = self._sort_player_ids((ledger_alive | current_alive) - ledger_dead)
        combined_dead = self._sort_player_ids(ledger_dead | self._extract_dead_from_notes(turn_notes))
        combined_known = self._sort_player_ids((set(self._public_ledger["known_player_ids"]) | current_alive | ledger_alive | ledger_dead) - {self.player_id})
        candidate_pool = self._candidate_pool(request, combined_alive, combined_known)
        ledger_lines = ["【公共台账】"]
        ledger_lines.append("存活: " + self._join_ids(combined_alive, limit=10))
        ledger_lines.append("已死: " + self._join_ids(combined_dead, limit=10))
        sheriff_id = self._current_sheriff(local_events)
        if sheriff_id:
            ledger_lines.append(f"警长/警徽: {sheriff_id}（归票优先参考，但仍核对查验链）")
        claim_block = self._collect_claim_block(turn_notes, local_events)
        if claim_block:
            ledger_lines.append("查验/自述: " + claim_block)
        gray_pool = [pid for pid in candidate_pool if pid not in set(combined_dead)]
        if gray_pool:
            ledger_lines.append("灰区: " + self._join_ids(gray_pool[:6], limit=6))
        phase_lines = self._render_phase_summary(request, candidate_pool, combined_alive, combined_dead, turn_notes, local_events)
        return self._trim_text("\n".join(part for part in ("\n".join(ledger_lines), phase_lines) if part), _PUBLIC_BLOCK_CHAR_LIMIT + _PHASE_BLOCK_CHAR_LIMIT)

    def _render_phase_summary(
        self,
        request: Mapping[str, Any],
        candidate_pool: list[str],
        alive_ids: list[str],
        dead_ids: list[str],
        notes: list[str],
        events: dict[str, Any],
    ) -> str:
        phase = str(request.get("phase") or request.get("channel") or request.get("kind") or "")
        allowed_actions = request.get("allowed_actions")
        allowed_kinds = {
            str(item.get("kind") or "").lower()
            for item in allowed_actions
            if isinstance(item, Mapping)
        } if isinstance(allowed_actions, list) else set()
        is_vote = any(key in phase.lower() for key in ("vote", "投票")) or any("vote" in kind for kind in allowed_kinds)
        if is_vote:
            return self._render_vote_summary(candidate_pool, alive_ids, dead_ids, notes, events)
        return self._render_speak_summary(candidate_pool, alive_ids, dead_ids, notes, events)

    def _render_vote_summary(
        self,
        candidate_pool: list[str],
        alive_ids: list[str],
        dead_ids: list[str],
        notes: list[str],
        events: dict[str, Any],
    ) -> str:
        ranked = self._rank_candidates(candidate_pool, notes, dead_ids, events)
        if len(ranked) < 2 and alive_ids:
            used = {item[0] for item in ranked}
            ranked.extend(self._rank_candidates([pid for pid in alive_ids if pid not in used], notes, dead_ids, events))
        while len(ranked) < 2 and candidate_pool:
            candidate = next((pid for pid in candidate_pool if pid not in {item[0] for item in ranked}), candidate_pool[0])
            ranked.append((candidate, 0, "公开信息不足"))
        ranked = ranked[:2]
        sheriff_call = self._latest_sheriff_call(events)
        top_inspection = self._top_wolf_target(candidate_pool, dead_ids, events)
        lines = ["【本轮投票参考】"]
        if top_inspection:
            lines.append(f"查杀优先: {top_inspection[0]} | {top_inspection[1]}。若未出现硬对跳坐实，优先集中票力。")
        if sheriff_call:
            lines.append(f"警长归票: {sheriff_call['target']} | {sheriff_call['note']}。无更高权重查杀时尽量对齐，防止分票。")
        for index, (candidate, score, reason) in enumerate(ranked, start=1):
            lines.append(f"{index}. {candidate} | 分数={score} | {reason}")
        if ranked:
            lines.append("建议：优先跟随清晰查杀/警长归票集中到第一候选；只有对跳矛盾明确时才改投第二候选。")
        else:
            lines.append("优先比较至少两个候选；当前候选不足时，用公开票型与查验链补足。")
        return self._trim_text("\n".join(lines), _PHASE_BLOCK_CHAR_LIMIT)

    def _render_speak_summary(
        self,
        candidate_pool: list[str],
        alive_ids: list[str],
        dead_ids: list[str],
        notes: list[str],
        events: dict[str, Any],
    ) -> str:
        ranked = self._rank_candidates(candidate_pool, notes, dead_ids, events) or self._rank_candidates(alive_ids, notes, dead_ids, events)
        if not ranked:
            ranked = [("信息不足", 0, "公开信息不足")]
        avoid = self._least_suspicious_candidate(candidate_pool or alive_ids, notes, dead_ids, events)
        lines = ["【本轮发言参考】", f"关注: {ranked[0][0]} | {ranked[0][2]}"]
        if avoid:
            lines.append(f"暂不投: {avoid[0]} | {avoid[2]}")
        else:
            lines.append("暂不投: 先保留警长、金水或公开链更稳的一人。")
        sheriff_call = self._latest_sheriff_call(events)
        if sheriff_call:
            lines.append(f"可表态是否对齐警长归票 {sheriff_call['target']}，并说明是否有更高权重查杀。")
        return self._trim_text("\n".join(lines), _PHASE_BLOCK_CHAR_LIMIT)

    def _rank_candidates(
        self,
        candidate_pool: list[str],
        notes: list[str],
        dead_ids: list[str],
        events: dict[str, Any],
    ) -> list[tuple[str, int, str]]:
        dead_set = set(dead_ids)
        ranked: list[tuple[str, int, str]] = []
        for candidate in self._sort_player_ids(candidate_pool):
            if candidate in dead_set or candidate == self.player_id:
                continue
            score, reasons = self._score_candidate(candidate, notes, events)
            ranked.append((candidate, score, "；".join(reasons[:3]) if reasons else "公开信息不足"))
        ranked.sort(key=lambda item: (-item[1], item[0]))
        return ranked

    def _least_suspicious_candidate(
        self,
        candidate_pool: list[str],
        notes: list[str],
        dead_ids: list[str],
        events: dict[str, Any],
    ) -> tuple[str, int, str] | None:
        ranked = self._rank_candidates(candidate_pool, notes, dead_ids, events)
        if not ranked:
            return None
        ranked.sort(key=lambda item: (item[1], item[0]))
        return ranked[0]

    def _score_candidate(self, candidate: str, notes: list[str], events: dict[str, Any]) -> tuple[int, list[str]]:
        score = 0
        reasons: list[str] = []
        sheriff_id = events.get("sheriff_id") or ""
        if sheriff_id and self._same_player(candidate, sheriff_id):
            score -= 6
            reasons.append("警长/持徽者，默认保护")
        seer_claimers = events.get("seer_claimers", set())
        if any(self._same_player(candidate, seer) for seer in seer_claimers):
            score -= 3
            reasons.append("预言家声明者，未被推翻前不优先出")
        for claim in events.get("inspection_claims", []):
            target = str(claim.get("target") or "")
            source = str(claim.get("source") or "")
            result = str(claim.get("result") or "")
            if not target or not self._same_player(candidate, target):
                continue
            trusted_bonus = 3 if source and (source in events.get("dead_ids", set()) or any(self._same_player(source, seer) for seer in seer_claimers)) else 0
            if result == "wolf":
                score += 9 + trusted_bonus
                reasons.append(f"被 {source or '公开链'} 查杀")
            elif result == "gold":
                score -= 7 + trusted_bonus
                reasons.append(f"{source or '公开链'} 报金水/好人")
        sheriff_call = self._latest_sheriff_call(events)
        if sheriff_call and self._same_player(candidate, str(sheriff_call.get("target") or "")):
            score += 5
            reasons.append("警长归票目标，需优先对齐防分票")
        for note in notes[-10:] + list(self._public_ledger["vote_snapshots"])[-8:]:
            if not self._mentions_player(note, candidate):
                continue
            compact = self._trim_text(note, 70)
            if any(term in note for term in ("冲票", "票型异常", "爆狼", "矛盾", "反水", "假")):
                score += 3
                reasons.append(f"疑点: {compact}")
            elif "vote" in note.lower() or "票" in note:
                score += 1
                reasons.append(f"票型: {compact}")
            elif any(term in note for term in ("金水", "可信", "强神", "守卫", "女巫", "猎人")):
                score -= 2
                reasons.append(f"保护: {compact}")
        if not reasons:
            reasons.append("公开信息不足")
        return score, reasons

    def _candidate_pool(self, request: Mapping[str, Any], alive_ids: list[str], known_ids: list[str]) -> list[str]:
        pool: set[str] = set(alive_ids) | set(known_ids)
        raw_actions = request.get("allowed_actions")
        if isinstance(raw_actions, list):
            for action in raw_actions:
                if isinstance(action, Mapping) and isinstance(action.get("target_ids"), list):
                    pool.update(str(target_id) for target_id in action["target_ids"] if self._is_scalar_id(target_id))
        pool.discard(self.player_id)
        pool.difference_update(self._public_ledger["dead_ids"])
        return self._sort_player_ids(list(pool))

    def _collect_claim_block(self, notes: list[str], events: dict[str, Any]) -> str:
        pieces: list[str] = []
        for claim in events.get("inspection_claims", [])[-6:]:
            result = "查杀" if claim.get("result") == "wolf" else "金水"
            pieces.append(f"{claim.get('source') or '?'}->{claim.get('target') or '?'} {result}")
        if events.get("seer_claimers"):
            pieces.append("预言家声明: " + self._join_ids(self._sort_player_ids(list(events["seer_claimers"])), limit=4))
        if not pieces:
            for note in notes + self._public_ledger["public_claims"][:3]:
                if any(term in note for term in ("claim", "自称", "预言家", "女巫", "猎人", "守卫", "金水", "查杀", "查验", "警徽")):
                    pieces.append(self._trim_text(note, 90))
                    if len(pieces) >= 3:
                        break
        return " | ".join(pieces[:6])

    def _build_local_events(self, notes: list[str]) -> dict[str, Any]:
        events = {
            "seer_claimers": set(self._public_ledger["seer_claimers"]),
            "inspection_claims": list(self._public_ledger["inspection_claims"]),
            "sheriff_id": self._public_ledger.get("sheriff_id") or "",
            "sheriff_calls": list(self._public_ledger["sheriff_calls"]),
            "dead_ids": set(self._public_ledger["dead_ids"]),
        }
        for note in notes:
            self._add_note_semantics_to(events, note)
        return events

    def _add_note_semantics_to(self, events: dict[str, Any], note: str) -> None:
        speaker = self._extract_field(note, "speaker") or self._extract_field(note, "player")
        text = self._extract_text_field(note) or note
        if speaker and ("预言家" in text or "role=预言家" in note or "seer" in note.lower()):
            events["seer_claimers"].add(speaker)
        for source, target, result in self._extract_inspections(note, speaker):
            event = {"source": source, "target": target, "result": result, "note": self._trim_text(note, 100)}
            if event not in events["inspection_claims"]:
                events["inspection_claims"].append(event)
        if any(term in note.lower() for term in ("death", "dead", "死亡", "出局", "淘汰")):
            events["dead_ids"].update(self._ids_from_text(note))
        if any(term in note.lower() for term in ("sheriff", "badge", "警长", "警徽")):
            holder = self._extract_field(note, "holder") or self._extract_field(note, "to")
            if holder and holder != "?":
                events["sheriff_id"] = holder
        if speaker and events.get("sheriff_id") and self._same_player(speaker, events["sheriff_id"]) and any(term in text for term in ("归票", "出", "投", "放逐")):
            targets = [pid for pid in self._ids_from_text(text) if not self._same_player(pid, speaker)]
            if targets:
                events["sheriff_calls"].append({"source": speaker, "target": targets[-1], "note": self._trim_text(note, 100)})

    def _current_sheriff(self, events: dict[str, Any]) -> str:
        return str(events.get("sheriff_id") or "")

    def _latest_sheriff_call(self, events: dict[str, Any]) -> dict[str, str] | None:
        calls = events.get("sheriff_calls") or []
        return calls[-1] if calls else None

    def _top_wolf_target(self, candidate_pool: list[str], dead_ids: list[str], events: dict[str, Any]) -> tuple[str, str] | None:
        dead_set = set(dead_ids)
        best: tuple[str, int, str] | None = None
        for candidate in candidate_pool:
            if candidate in dead_set or candidate == self.player_id:
                continue
            for claim in events.get("inspection_claims", []):
                if claim.get("result") == "wolf" and self._same_player(candidate, str(claim.get("target") or "")):
                    reason = f"{claim.get('source') or '?'} 查杀，{self._trim_text(str(claim.get('note') or ''), 60)}"
                    score = 2 if str(claim.get("source") or "") in events.get("dead_ids", set()) else 1
                    if best is None or score > best[1]:
                        best = (candidate, score, reason)
        return (best[0], best[2]) if best else None

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
        death_like = any(key in lower_map for key in ("dead", "death", "dead_id", "dead_player_id")) or any(term in event_lower for term in ("death", "dead", "kill", "out", "死亡", "出局"))
        if not (vote_like or claim_like or sheriff_like or death_like):
            return None
        round_text = self._first_scalar(lower_map, ("round", "day", "turn", "phase", "channel"))
        player = self._first_scalar(lower_map, ("player_id", "speaker_id", "voter_id", "claimer_id", "actor_id", "source_id"))
        target = self._first_scalar(lower_map, ("target_id", "target", "vote_target", "to_id", "suspect_id", "badge_to"))
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
            parts.append(f"text={self._trim_text(text, _PUBLIC_NOTE_CHAR_LIMIT)}")
        snapshot = " ".join(part for part in parts if part).strip()
        return self._trim_text(snapshot, _PUBLIC_NOTE_CHAR_LIMIT) if snapshot else None

    def _extract_inspections(self, note: str, speaker: str = "") -> list[tuple[str, str, str]]:
        text = self._extract_text_field(note) or note
        source = speaker or self._extract_field(note, "source") or self._extract_field(note, "speaker")
        ids = self._ids_from_text(text)
        events: list[tuple[str, str, str]] = []
        for result, terms in (("wolf", ("查杀", "验出狼", "是狼", "狼人")), ("gold", ("金水", "好人", "验好", "查验为好"))):
            if not any(term in text for term in terms):
                continue
            target = self._target_near_terms(text, terms, source, ids)
            if target:
                events.append((source, target, result))
        return events

    def _target_near_terms(self, text: str, terms: tuple[str, ...], source: str, ids: set[str]) -> str:
        if not ids:
            return ""
        for term in terms:
            idx = text.find(term)
            if idx < 0:
                continue
            suffix_ids = self._ids_from_text(text[idx: idx + 16])
            suffix_ids = {pid for pid in suffix_ids if not self._same_player(pid, source)}
            if suffix_ids:
                return self._sort_player_ids(list(suffix_ids))[0]
            prefix_ids = self._ids_from_text(text[max(0, idx - 10): idx])
            prefix_ids = {pid for pid in prefix_ids if not self._same_player(pid, source)}
            if prefix_ids:
                return self._sort_player_ids(list(prefix_ids))[-1]
        candidates = [pid for pid in ids if not self._same_player(pid, source)]
        return self._sort_player_ids(candidates)[-1] if candidates else ""

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
            if any(term in note.lower() for term in ("death", "dead", "死亡", "出局", "淘汰")):
                dead.update(self._ids_from_text(note))
        return dead

    def _ids_from_text(self, text: str) -> set[str]:
        ids: set[str] = set()
        for match in _ID_IN_TEXT.finditer(str(text)):
            ids.add("p" + match.group(1))
        for token in re.split(r"[\s,，;；|/\\]+", str(text)):
            token = token.strip("()[]{}<>：:=，。,.、")
            if self._looks_like_player_id(token):
                ids.add(str(token))
        return ids

    def _extract_field(self, note: str, field: str) -> str:
        match = re.search(rf"(?:^|\s){re.escape(field)}=([^\s]+)", note)
        if not match:
            return ""
        value = match.group(1).strip("，。,.；;| ")
        return value if self._looks_like_player_id(value) or value == "?" else ""

    def _extract_text_field(self, note: str) -> str:
        marker = "text="
        idx = note.find(marker)
        return note[idx + len(marker):] if idx >= 0 else ""

    def _mentions_player(self, text: str, player_id: str) -> bool:
        if player_id and player_id.lower() in text.lower():
            return True
        return any(self._same_player(found, player_id) for found in self._ids_from_text(text))

    def _same_player(self, left: str, right: str) -> bool:
        if not left or not right:
            return False
        if str(left) == str(right):
            return True
        return self._player_number(left) != "" and self._player_number(left) == self._player_number(right)

    def _player_number(self, player_id: str) -> str:
        match = _PLAYER_ID_SUFFIX.search(str(player_id).strip())
        return match.group(1) if match else ""

    def _as_id_set(self, value: Any) -> set[str]:
        if isinstance(value, (list, tuple, set)):
            return {str(item) for item in value if self._is_scalar_id(item)}
        if self._is_scalar_id(value):
            return {str(value)}
        return set()

    def _append_bounded(self, items: list[Any], value: Any, limit: int) -> None:
        if value in (None, "") or value in items:
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
        return isinstance(value, str) and self._looks_like_player_id(value)

    def _trim_text(self, text: str, limit: int) -> str:
        normalized = " ".join(str(text).split())
        if len(normalized) <= limit:
            return normalized
        return normalized[: max(0, limit - 1)] + "…"

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
        suffix = f" 等{len(ids) - len(selected)}人" if len(ids) > len(selected) else ""
        return "、".join(selected) + suffix

    def _sort_player_ids(self, ids: list[str] | set[str]) -> list[str]:
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
            "player_id", "speaker_id", "voter_id", "claimer_id", "actor_id", "source_id",
            "target_id", "from_id", "to_id", "badge_holder", "badge_holder_id", "sheriff_id",
            "dead_id", "dead_player_id",
        }

    @staticmethod
    def _collection_id_keys() -> set[str]:
        return {
            "alive", "alive_ids", "alive_player_ids", "dead", "dead_ids", "dead_player_ids",
            "players", "player_ids", "target_ids", "candidates", "survivors", "living",
            "graveyard", "claims", "votes",
        }

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

"""villager_2 轻量 Task-Agent：压缩上下文、限制工具预算、强调残局不弃票。"""
from __future__ import annotations
import asyncio
from collections.abc import Mapping
import hashlib, inspect, json, re
from typing import Any
from ..core.errors import ModelClientError, RuleViolationError
from .llm.coordinator import ModelRequestCoordinator
from ..prompts import RoleProfile, render_prompt

_CHINESE_CHARACTER = re.compile(r"[\u3400-\u9fff]")
_PLAYER_ID_SUFFIX = re.compile(r"(\d+)$")
_ID_IN_TEXT = re.compile(r"\b[pP]?\d+\b")
CURRENT_ROUND_DIALOGUE_TOOL_NAME = "read_current_round_dialogue"
DEFAULT_MAX_TOOL_CALLS_PER_DECISION = 5
DEFAULT_MAX_TOOL_RESULT_TOKENS = 1000
DEFAULT_MAX_PROMPT_CHARS = 12000
DEFAULT_MAX_DECISION_RETRIES = 2
_SPEECH_ACTION_KINDS = frozenset({"speak", "last_words"})
_LEDGER_LIMIT = 12
_NOTE_LIMIT = 120
_HINT_LIMIT = 400
_TOOL_DIALOGUE_CHAR_LIMIT = 1000
PLAYER_ID_KEYS = {"player_id", "speaker_id", "voter_id", "claimer_id", "actor_id", "source_id", "target_id", "from_id", "to_id", "badge_holder", "badge_holder_id", "sheriff_id", "dead_id", "dead_player_id"}
COLLECTION_ID_KEYS = {"alive", "alive_ids", "alive_player_ids", "players", "player_ids", "target_ids", "candidates", "survivors", "living", "current_alive", "dead", "dead_ids", "dead_player_ids", "graveyard"}
ALIVE_KEYS = {"alive", "alive_ids", "alive_player_ids", "survivors", "living", "live_ids", "current_alive"}
DEAD_KEYS = {"dead", "dead_ids", "dead_player_ids", "eliminated", "casualties", "deaths", "graveyard"}
CURRENT_ROUND_DIALOGUE_TOOL: dict[str, Any] = {"name": CURRENT_ROUND_DIALOGUE_TOOL_NAME, "description": "读取当前昼夜轮次中、当前玩家依法可见的已发生发言。", "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}

def _phase_flags(request: Mapping[str, Any]) -> tuple[bool, bool, bool, bool]:
    phase = str(request.get("phase") or request.get("channel") or request.get("kind") or "").lower()
    raw = request.get("allowed_actions")
    kinds = {str(a.get("kind") or "").lower() for a in raw if isinstance(a, Mapping)} if isinstance(raw, list) else set()
    is_vote = any(k in phase for k in ("vote", "投票")) or any("vote" in k for k in kinds)
    is_speak = any(k in phase for k in ("speak", "discussion", "发言", "白天", "day")) or bool(kinds & {"speak", "discussion", "day_speak"})
    is_last = any(k in phase for k in ("last", "遗言")) or "last_words" in kinds
    is_sheriff = any(k in phase for k in ("sheriff", "警长", "警徽")) or bool(kinds & {"sheriff_vote", "sheriff"})
    return is_vote, is_speak, is_last, is_sheriff

def render_villager_decision_checklist(request: Mapping[str, Any]) -> str:
    is_vote, is_speak, is_last, is_sheriff = _phase_flags(request)
    lines = ["信息层级：系统事实>可信查验链>玩家声明>推测。"]
    if is_vote:
        lines.append("投票比较至少两人；<=5人或已有查杀/异常票型时不要pass，归到证据最强者。")
    if is_speak:
        lines.append("发言点一个关注对象和一个暂缓对象。")
    if is_last:
        lines.append("遗言只留公开事实、票型、查验链和怀疑。")
    if is_sheriff:
        lines.append("警徽回合继承清晰报验链，但玩家声明不是系统确认。")
    return "\n".join(lines)

def render_action_contract(request: Mapping[str, Any]) -> str:
    raw = request.get("allowed_actions")
    actions = [a for a in raw if isinstance(a, Mapping)] if isinstance(raw, list) else []
    lines = ["【最终 JSON 契约】", f"阶段：{request.get('phase') or 'unknown'}；频道：{request.get('channel') or 'unknown'}。", "只能输出一个 JSON 对象；不得有 Markdown、解释或额外字段。", "只能选当前合法模板；target_id 只能取列出的候选值。"]
    if not actions:
        return "\n".join(lines + ["当前没有可用行动；不要臆造行动。"])
    for allowed in actions:
        kind = str(allowed.get("kind") or "")
        if not kind:
            continue
        targets = [str(t) for t in allowed.get("target_ids", [])] if isinstance(allowed.get("target_ids"), list) else []
        example: dict[str, str] = {"kind": kind}
        if kind in _SPEECH_ACTION_KINDS:
            example["text"] = "好"
        elif targets:
            example["target_id"] = targets[0]
        lines.append("- `" + json.dumps(example, ensure_ascii=False, separators=(",", ":")) + "`")
        if targets:
            lines.append("  target_id 候选值：" + "、".join(targets) + "。")
        if kind in _SPEECH_ACTION_KINDS:
            detail = []
            if allowed.get("max_chars") is not None:
                detail.append(f"text最多{allowed['max_chars']}字")
            if allowed.get("require_chinese"):
                detail.append("text至少含中文")
            lines.append("  " + "；".join(detail + ["字段名保持英文"]) + "。")
    return "\n".join(lines + ["kind 和 target_id 是协议字段，必须英文原样保留。"])

def normalize_decision(raw: dict[str, Any] | None, request: dict[str, Any]) -> dict[str, Any]:
    src = raw.get("decision", raw) if isinstance(raw, dict) else {}
    action = {"request_id": request["request_id"], "player_id": request["player_id"], "kind": str(src.get("kind", src.get("action", "")))}
    target_id = src.get("target_id", src.get("targetId"))
    if target_id:
        action["target_id"] = str(target_id)
    if isinstance(src.get("text"), str):
        action["text"] = src["text"]
    return action

def decision_error(action: dict[str, Any], request: dict[str, Any]) -> str | None:
    allowed = next((a for a in request["allowed_actions"] if a["kind"] == action["kind"]), None)
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
    targets = allowed.get("target_ids")
    if targets and action.get("target_id") not in targets:
        return "target_id 必须是 target_ids 中的一项"
    if not targets and action.get("target_id"):
        return "该行动不能提供 target_id"
    return None

class TaskAgent:
    """平民任务求解器；保留外部构造接口。"""
    def __init__(self, *, player_id: str, profile: RoleProfile, model_client: Any, persona: str = "", request_coordinator: ModelRequestCoordinator | None = None, max_tokens: int = 900, max_decision_retries: int = DEFAULT_MAX_DECISION_RETRIES, max_tool_calls_per_decision: int = DEFAULT_MAX_TOOL_CALLS_PER_DECISION, max_tool_result_tokens: int = DEFAULT_MAX_TOOL_RESULT_TOKENS, max_prompt_chars: int = DEFAULT_MAX_PROMPT_CHARS) -> None:
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
        self._model_token_usage = {"successful_response_count": 0, "api_attempt_count": 0, "reported_usage_response_count": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
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
        self._ingest_public_packet(turn_packet.get("public_state"))
        request = turn_packet["request"]
        prompt = self._compact_packet(turn_packet)
        dialogue = self._compact_dialogue((turn_packet.get("tool_context") or {}).get("current_round_dialogue") or [])
        def execute_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
            del arguments
            if name != CURRENT_ROUND_DIALOGUE_TOOL_NAME:
                return {"error": f"不支持的工具：{name}"}
            return {"round": turn_packet["game"].get("round"), "phase": turn_packet["game"].get("public_phase", turn_packet["game"].get("phase")), "dialogue": dialogue}
        feedback = ""
        system = self._system_prompt(private)
        for attempt in range(self.max_decision_retries + 1):
            allow_tool = attempt == 0 and self.max_tool_calls_per_decision > 0
            user_content = {"instruction": self._turn_instruction(turn_packet, feedback), "packet": prompt}
            try:
                raw = await self._complete_json(system=system, messages=[{"role": "user", "content": json.dumps(user_content, ensure_ascii=False)}], tools=[CURRENT_ROUND_DIALOGUE_TOOL] if allow_tool else None, tool_executor=execute_tool if allow_tool else None, max_tool_calls=self.max_tool_calls_per_decision if allow_tool else 0, max_tool_result_tokens=self.max_tool_result_tokens, max_prompt_chars=self.max_prompt_chars)
            except ModelClientError:
                if attempt >= self.max_decision_retries:
                    raise
                feedback = "上一轮模型调用无可用结构化行动，请直接按合法 JSON 作答。"
                continue
            self._record_token_usage(raw)
            action = normalize_decision(raw, request)
            error = decision_error(action, request)
            if error is None:
                return action
            if attempt >= self.max_decision_retries:
                raise RuleViolationError(f"模型返回非法行动：{error}", attempted_action=action, validation_error=error)
            feedback = error
        raise RuleViolationError("模型没有返回合法行动", attempted_action={}, validation_error="empty")
    def agent_manifest(self) -> dict[str, Any]:
        return {"agent_type": "task_agent", "player_id": self.player_id, "role": self.profile.role, "conversation_mode": "new_session_per_decision", "tool_name": CURRENT_ROUND_DIALOGUE_TOOL_NAME, "max_tool_calls_per_decision": self.max_tool_calls_per_decision, "max_tool_result_tokens": self.max_tool_result_tokens, "max_prompt_chars": self.max_prompt_chars, "max_decision_retries": self.max_decision_retries, "task_fingerprint": hashlib.sha256(self.profile.task.encode("utf-8")).hexdigest()}
    def model_token_usage_snapshot(self) -> dict[str, int]:
        return dict(self._model_token_usage)
    def _system_prompt(self, private: Mapping[str, Any]) -> str:
        return render_prompt("player_system.txt", player_id=self.player_id, role=private["role"], team=private["team"], persona=self.persona, role_base=self.profile.base, role_task=self.profile.task, tool_name=CURRENT_ROUND_DIALOGUE_TOOL_NAME, max_tool_calls=self.max_tool_calls_per_decision, max_tool_result_tokens=self.max_tool_result_tokens, max_prompt_chars=self.max_prompt_chars)
    def _turn_instruction(self, turn_packet: Mapping[str, Any], feedback: str) -> str:
        request = turn_packet["request"]
        brief = "\n".join(p for p in (self._render_turn_summary(turn_packet, request), render_villager_decision_checklist(request)) if p)
        if feedback:
            brief += f"\n校验反馈：{feedback}。"
        return render_prompt("player_turn_instruction.txt", action_contract=render_action_contract(request), validation_feedback=self._trim_text(brief, _HINT_LIMIT))
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
        values = {k: self._nonnegative_int(usage.get(k), fallback=None) for k in ("input_tokens", "output_tokens", "total_tokens")}
        if all(v is None for v in values.values()):
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
            n = int(value)
        except (TypeError, ValueError):
            return fallback
        return n if n >= 0 else fallback
    def _reset_public_ledger(self, game_id: str | None) -> None:
        self._public_ledger_game_id = game_id
        self._public_ledger = {"alive_ids": set(), "dead_ids": set(), "known_player_ids": set(), "claims": [], "votes": [], "deaths": [], "events": []}
    def _compact_packet(self, turn_packet: Mapping[str, Any]) -> dict[str, Any]:
        req = turn_packet["request"]
        return {"game": self._compact_mapping(turn_packet.get("game"), ("round", "phase", "public_phase", "day", "night")), "self": self._compact_mapping(turn_packet.get("self"), ("player_id", "seat", "status")), "private_information": self._compact_mapping(turn_packet.get("private_information"), ("role", "team")), "request": req, "public_snapshot": self._public_snapshot(turn_packet, req), "history_policy": {"default_context": "compact_public_snapshot_only", "current_round_dialogue_tool": CURRENT_ROUND_DIALOGUE_TOOL_NAME, "max_tool_calls": self.max_tool_calls_per_decision, "max_tool_result_tokens": self.max_tool_result_tokens}}
    @staticmethod
    def _compact_mapping(value: Any, keys: tuple[str, ...]) -> dict[str, Any]:
        return {k: value[k] for k in keys if isinstance(value, Mapping) and k in value}
    def _ingest_public_packet(self, payload: Any) -> None:
        if not isinstance(payload, (Mapping, list)):
            return
        notes = self._extract_public_notes(payload)
        self._public_ledger["known_player_ids"].update(self._extract_player_ids(payload))
        if isinstance(payload, Mapping):
            self._public_ledger["alive_ids"].update(self._collect_id_collection(payload, ALIVE_KEYS))
            self._public_ledger["dead_ids"].update(self._collect_id_collection(payload, DEAD_KEYS))
        for note in notes:
            self._append_bounded(self._public_ledger["events"], note, _LEDGER_LIMIT)
            lower = note.lower()
            if "vote" in lower or "票" in note:
                self._append_bounded(self._public_ledger["votes"], note, _LEDGER_LIMIT)
            if any(t in lower for t in ("death", "dead", "kill")) or any(t in note for t in ("死亡", "出局", "淘汰")):
                self._append_bounded(self._public_ledger["deaths"], note, _LEDGER_LIMIT)
                self._public_ledger["dead_ids"].update(self._ids_from_text(note))
            if any(t in note for t in ("预言家", "查杀", "金水", "查验", "警徽", "自称", "女巫", "猎人", "守卫")):
                self._append_bounded(self._public_ledger["claims"], note, _LEDGER_LIMIT)
    def _render_turn_summary(self, turn_packet: Mapping[str, Any], request: Mapping[str, Any]) -> str:
        snap = self._public_snapshot(turn_packet, request)
        lines = [f"公共: 存活={self._join_ids(snap['alive'], 8)}；已死={self._join_ids(snap['dead'], 6)}。"]
        if snap["claims"]:
            lines.append("查验/自述=" + " | ".join(snap["claims"][:2]))
        if snap["focus"]:
            lines.append("焦点=" + "；".join(f"{p}:{r}" for p, r in snap["focus"][:2]))
        if snap["endgame"] and snap["is_vote"]:
            lines.append("关键残局：不要流票，归到证据最强者。")
        return self._trim_text("\n".join(lines), 360)
    def _public_snapshot(self, turn_packet: Mapping[str, Any], request: Mapping[str, Any]) -> dict[str, Any]:
        public_state = turn_packet.get("public_state")
        notes = self._extract_public_notes(public_state)
        alive = set(self._public_ledger["alive_ids"]) | self._extract_player_ids(public_state, include_collections=True)
        dead = set(self._public_ledger["dead_ids"]) | self._extract_dead_from_notes(notes)
        alive -= dead
        known = (set(self._public_ledger["known_player_ids"]) | alive | dead) - {self.player_id}
        candidates = self._candidate_pool(request, self._sort_player_ids(list(alive)), self._sort_player_ids(list(known)))
        ranked = self._rank_candidates(candidates, notes, self._sort_player_ids(list(dead)))
        claims = self._dedupe([self._trim_text(x, 80) for x in (self._ledger_recent("claims") + self._claim_notes(notes))])[:3]
        is_vote = _phase_flags(request)[0]
        return {"alive": self._sort_player_ids(list(alive)), "dead": self._sort_player_ids(list(dead)), "claims": claims, "focus": [(p, r) for p, _s, r in ranked[:3]], "is_vote": is_vote, "endgame": 0 < len(alive) <= 5}
    def _rank_candidates(self, candidate_pool: list[str], notes: list[str], dead_ids: list[str]) -> list[tuple[str, int, str]]:
        ranked = []
        for cand in self._sort_player_ids(candidate_pool):
            if cand == self.player_id or cand in dead_ids:
                continue
            score, reason = self._candidate_score(cand, notes)
            ranked.append((cand, score, reason))
        return sorted(ranked, key=lambda x: (-x[1], x[0]))
    def _candidate_score(self, cand: str, notes: list[str]) -> tuple[int, str]:
        related = [n for n in (notes + self._ledger_recent("claims") + self._ledger_recent("votes")) if cand.lower() in n.lower()]
        if not related:
            return 1, "灰区未表态"
        score, reasons = 1, []
        for note in related[-6:]:
            lower, text = note.lower(), self._trim_text(note, 55)
            if "查杀" in note or "wolf" in lower or "werewolf" in lower:
                score += 8; reasons.append("查杀链:" + text)
            elif any(t in note for t in ("票型异常", "冲票", "改票", "跟票", "归票")) or "vote" in lower:
                score += 4; reasons.append("票型:" + text)
            elif any(t in note for t in ("事实错误", "矛盾", "不一致", "对跳", "反水", "爆狼")):
                score += 5; reasons.append("矛盾:" + text)
            elif any(t in note for t in ("金水", "银水", "强神", "猎人", "女巫", "守卫")):
                score -= 4; reasons.append("暂保护:" + text)
            elif any(t in note for t in ("预言家", "查验", "警徽")):
                score -= 1; reasons.append("链上信息:" + text)
        return score, (reasons[0] if reasons else "灰区未表态")
    def _candidate_pool(self, request: Mapping[str, Any], alive_ids: list[str], known_ids: list[str]) -> list[str]:
        pool = set(alive_ids) | set(known_ids)
        raw = request.get("allowed_actions")
        for act in raw if isinstance(raw, list) else []:
            if isinstance(act, Mapping) and isinstance(act.get("target_ids"), list):
                pool.update(str(t) for t in act["target_ids"] if self._is_scalar_id(t))
        pool.discard(self.player_id)
        pool.difference_update(self._public_ledger["dead_ids"])
        return self._sort_player_ids(list(pool))
    def _extract_public_notes(self, payload: Any) -> list[str]:
        notes: list[str] = []
        def walk(obj: Any, depth: int) -> None:
            if depth > 3 or len(notes) >= 24:
                return
            if isinstance(obj, Mapping):
                snap = self._snapshot_mapping(obj)
                if snap:
                    notes.append(snap)
                for val in obj.values():
                    walk(val, depth + 1)
            elif isinstance(obj, list):
                for item in obj[:16]:
                    walk(item, depth + 1)
        walk(payload, 0)
        return self._dedupe(notes)
    def _snapshot_mapping(self, mapping: Mapping[str, Any]) -> str | None:
        lower = {str(k).lower(): v for k, v in mapping.items()}
        event = self._first_scalar(lower, ("event", "event_type", "type", "kind", "action"))
        vote = "vote" in event.lower() or any(k in lower for k in ("voter_id", "vote_target", "ballot"))
        death = any(k in lower for k in ("dead", "dead_id", "dead_player_id", "death")) or any(t in event.lower() for t in ("death", "dead", "kill", "out"))
        claim = any(k in lower for k in ("speaker_id", "claim", "role_claim", "speech", "text", "content", "message", "reason"))
        if not (vote or death or claim or any(t in event for t in ("警徽", "警长"))):
            return None
        player = self._first_scalar(lower, ("player_id", "speaker_id", "voter_id", "actor_id", "source_id", "dead_id", "dead_player_id")) or "?"
        target = self._first_scalar(lower, ("target_id", "target", "vote_target", "to_id", "suspect_id"))
        role = self._first_scalar(lower, ("role", "claim_role", "claimed_role"))
        text = self._first_scalar(lower, ("text", "content", "message", "reason", "speech"))
        parts = [event.lower() or "event", f"player={player}"]
        if target: parts.append(f"target={target}")
        if role: parts.append(f"role={role}")
        if text: parts.append("text=" + self._trim_text(text, 70))
        return self._trim_text(" ".join(parts), _NOTE_LIMIT)
    def _extract_player_ids(self, payload: Any, *, include_collections: bool = False) -> set[str]:
        ids: set[str] = set()
        def walk(obj: Any, depth: int) -> None:
            if depth > 4:
                return
            if isinstance(obj, Mapping):
                for k, v in ((str(k).lower(), v) for k, v in obj.items()):
                    if k in PLAYER_ID_KEYS and self._is_scalar_id(v): ids.add(str(v))
                    if include_collections and k in COLLECTION_ID_KEYS: ids.update(self._as_id_set(v))
                for v in obj.values(): walk(v, depth + 1)
            elif isinstance(obj, list):
                for item in obj[:20]: walk(item, depth + 1)
        walk(payload, 0)
        ids.discard(self.player_id)
        return ids
    def _collect_id_collection(self, mapping: Mapping[str, Any], keys: set[str]) -> set[str]:
        found: set[str] = set()
        for key, value in ((str(k).lower(), v) for k, v in mapping.items()):
            if key in keys:
                found.update(self._as_id_set(value))
        return found
    def _extract_dead_from_notes(self, notes: list[str]) -> set[str]:
        dead: set[str] = set()
        for note in notes:
            lower = note.lower()
            if any(t in lower for t in ("death", "dead", "kill", "out")) or any(t in note for t in ("死亡", "出局", "淘汰")):
                dead.update(self._ids_from_text(note))
        return dead
    def _claim_notes(self, notes: list[str]) -> list[str]:
        return [n for n in notes if any(t in n for t in ("预言家", "查杀", "金水", "查验", "警徽", "自称", "女巫", "猎人", "守卫"))]
    def _ledger_recent(self, key: str) -> list[str]:
        v = self._public_ledger.get(key, [])
        return list(v) if isinstance(v, list) else []
    def _compact_dialogue(self, dialogue: Any) -> list[Any]:
        if not isinstance(dialogue, list):
            return []
        result, used = [], 0
        for item in dialogue[-12:]:
            compact = self._compact_mapping(item, ("speaker_id", "player_id", "kind", "text", "content", "message")) if isinstance(item, Mapping) else item
            text = self._trim_text(json.dumps(compact, ensure_ascii=False, default=str), 180)
            if used + len(text) > _TOOL_DIALOGUE_CHAR_LIMIT:
                break
            used += len(text); result.append(compact)
        return result
    def _ids_from_text(self, text: str) -> set[str]:
        return {m.group(0) for m in _ID_IN_TEXT.finditer(str(text)) if self._looks_like_player_id(m.group(0))}
    def _as_id_set(self, value: Any) -> set[str]:
        if isinstance(value, (list, tuple, set)):
            return {str(x) for x in value if self._is_scalar_id(x)}
        if isinstance(value, Mapping):
            return {str(x) for x in value.keys() if self._is_scalar_id(x)}
        return {str(value)} if self._is_scalar_id(value) else set()
    @staticmethod
    def _append_bounded(items: list[str], value: str, limit: int) -> None:
        if value and value not in items:
            items.append(value)
        if len(items) > limit:
            del items[: len(items) - limit]
    def _first_scalar(self, mapping: Mapping[str, Any], keys: tuple[str, ...]) -> str:
        for key in keys:
            value = mapping.get(key)
            if isinstance(value, (str, int, float)) and not isinstance(value, bool):
                return self._trim_text(str(value), _NOTE_LIMIT)
        return ""
    def _is_scalar_id(self, value: Any) -> bool:
        return (isinstance(value, int) and not isinstance(value, bool)) or (isinstance(value, str) and self._looks_like_player_id(value))
    @staticmethod
    def _trim_text(text: str, limit: int) -> str:
        normalized = " ".join(str(text).split())
        return normalized if len(normalized) <= limit else normalized[: max(0, limit - 1)] + "…"
    @staticmethod
    def _dedupe(items: list[str]) -> list[str]:
        seen, ordered = set(), []
        for item in items:
            if item and item not in seen:
                seen.add(item); ordered.append(item)
        return ordered
    def _join_ids(self, ids: list[str], limit: int) -> str:
        if not ids:
            return "未知"
        selected = ids[:limit]
        return "、".join(selected) + (f"等{len(ids) - len(selected)}人" if len(ids) > len(selected) else "")
    def _sort_player_ids(self, ids: list[str]) -> list[str]:
        def key(value: str) -> tuple[str, int, str]:
            m = _PLAYER_ID_SUFFIX.search(value)
            return (_PLAYER_ID_SUFFIX.sub("", value), int(m.group(1)) if m else 10**9, value)
        return sorted(self._dedupe([str(x) for x in ids if str(x)]), key=key)
    @staticmethod
    def _looks_like_player_id(value: str) -> bool:
        text = str(value).strip()
        return bool(text) and len(text) <= 16 and any(ch.isdigit() for ch in text)
    def _extract_game_id(self, payload: Mapping[str, Any]) -> str | None:
        for key in ("game_id", "gameid", "match_id", "matchid", "session_id"):
            val = payload.get(key)
            if isinstance(val, (str, int, float)) and not isinstance(val, bool):
                return str(val)
        game = payload.get("game")
        if isinstance(game, Mapping):
            for key in ("game_id", "gameid", "match_id", "matchid", "session_id"):
                val = game.get(key)
                if isinstance(val, (str, int, float)) and not isinstance(val, bool):
                    return str(val)
        return None

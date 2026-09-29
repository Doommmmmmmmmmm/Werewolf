from __future__ import annotations

import asyncio, hashlib, inspect, json, re
from collections.abc import Mapping
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
_PUBLIC_BLOCK_CHAR_LIMIT = 1600
_PHASE_BLOCK_CHAR_LIMIT = 800

CURRENT_ROUND_DIALOGUE_TOOL: dict[str, Any] = {
    "name": CURRENT_ROUND_DIALOGUE_TOOL_NAME,
    "description": "读取当前昼夜轮次中、当前玩家依法可见的已发生发言。",
    "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
}


def _phase_and_kinds(request):
    phase = str(request.get("phase") or request.get("channel") or request.get("kind") or "").lower()
    raw = request.get("allowed_actions")
    kinds = {str(x.get("kind") or "").lower() for x in raw if isinstance(x, Mapping)} if isinstance(raw, list) else set()
    return phase, kinds


def render_villager_decision_checklist(request):
    phase, kinds = _phase_and_kinds(request)
    vote = any(k in phase for k in ("vote", "投票")) or any("vote" in k for k in kinds)
    speak = any(k in phase for k in ("speak", "discussion", "发言", "白天", "day")) or bool(kinds & {"speak", "discussion", "day_speak"})
    lines = ["【平民清单】分清事实/声明/推测；延续台账。"]
    if vote:
        lines += [
            "投票先比两人；若持续预言家/警徽链无硬对跳并给出查杀，优先跟票查杀。",
            "勿因被点名或私怨反踩；仅硬对跳/事实矛盾/验链断裂才推翻查杀优先。",
            "优先未被金水覆盖且有票型异常/事实错误/查杀/爆狼线索者。",
        ]
    if speak:
        lines.append("发言给具体逻辑：查杀/对跳、矛盾/票型、倾向与疑问；拒绝复读。")
    if any(k in phase for k in ("last", "遗言")) or "last_words" in kinds:
        lines.append("遗言只留公开事实、票型、查验链和怀疑对象。")
    if any(k in phase for k in ("sheriff", "警长", "警徽")) or bool(kinds & {"sheriff_vote", "sheriff"}):
        lines.append("警长/警徽回合看报验链是否一致。")
    lines.append("残局/持徽先复盘存活、死亡、查验链、灰区。")
    return "\n".join(lines)


def render_action_contract(request):
    actions = [x for x in request.get("allowed_actions") if isinstance(x, Mapping)] if isinstance(request.get("allowed_actions"), list) else []
    lines = ["【行动JSON】", f"阶段:{request.get('phase') or 'unknown'}；频道:{request.get('channel') or 'unknown'}。", "只输出一个JSON对象，无解释/Markdown。", "只能选合法模板；target_id用候选值。"]
    if not actions:
        return "\n".join(lines + ["无可用行动，勿臆造。"])
    for a in actions:
        kind = str(a.get("kind") or "")
        if not kind:
            continue
        targets = [str(t) for t in a.get("target_ids")] if isinstance(a.get("target_ids"), list) and a.get("target_ids") else []
        ex = {"kind": kind}
        if kind in _SPEECH_ACTION_KINDS:
            ex["text"] = "好"
        elif targets:
            ex["target_id"] = targets[0]
        lines.append("- `" + json.dumps(ex, ensure_ascii=False, separators=(",", ":")) + "`")
        if targets:
            lines.append("  target_id:" + "、".join(targets) + "。")
        if kind in _SPEECH_ACTION_KINDS:
            c = []
            if a.get("max_chars") is not None:
                c.append(f"text最多{a['max_chars']}字")
            if a.get("require_chinese"):
                c.append("至少含中文")
            lines.append("  " + "；".join(c + ["可含英文、数字和玩家编号"]) + "。")
    lines.append("kind/target_id须英文原样保留。")
    return "\n".join(lines)


def normalize_decision(raw, request: dict[str, Any]):
    s = raw.get("decision", raw) if isinstance(raw, dict) else {}
    a = {"request_id": request["request_id"], "player_id": request["player_id"], "kind": str(s.get("kind", s.get("action", "")))}
    if s.get("target_id", s.get("targetId")):
        a["target_id"] = str(s.get("target_id", s.get("targetId")))
    if isinstance(s.get("text"), str):
        a["text"] = s["text"]
    return a


def decision_error(action, request: dict[str, Any]):
    allowed = next((x for x in request["allowed_actions"] if x["kind"] == action["kind"]), None)
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
    tids = allowed.get("target_ids")
    if tids and action.get("target_id") not in tids:
        return "target_id 必须是 target_ids 中的一项"
    if not tids and action.get("target_id"):
        return "该行动不能提供 target_id"
    return None


class TaskAgent:
    def __init__(self, *, player_id, profile, model_client, persona="", request_coordinator=None, max_tokens=900, max_decision_retries=DEFAULT_MAX_DECISION_RETRIES, max_tool_calls_per_decision=DEFAULT_MAX_TOOL_CALLS_PER_DECISION, max_tool_result_tokens=DEFAULT_MAX_TOOL_RESULT_TOKENS, max_prompt_chars=DEFAULT_MAX_PROMPT_CHARS):
        if not hasattr(model_client, "complete_json"):
            raise ValueError("TaskAgent 需要具有 complete_json 的模型客户端")
        self.player_id, self.profile, self.model_client, self.persona = str(player_id), profile, model_client, persona
        self.request_coordinator = request_coordinator
        self.max_tokens = max_tokens
        self.max_decision_retries = max(0, int(max_decision_retries))
        self.max_tool_calls_per_decision = max(0, int(max_tool_calls_per_decision))
        self.max_tool_result_tokens = max(1, int(max_tool_result_tokens))
        self.max_prompt_chars = max(1, int(max_prompt_chars))
        self._model_token_usage = {k: 0 for k in ("successful_response_count", "api_attempt_count", "reported_usage_response_count", "input_tokens", "output_tokens", "total_tokens")}
        self._public_ledger_game_id = None
        self._reset_public_ledger(None)

    async def observe(self, sync_packet):
        if isinstance(sync_packet, Mapping):
            gid = self._extract_game_id(sync_packet)
            if gid is not None and gid != self._public_ledger_game_id:
                self._reset_public_ledger(gid)
            self._ingest_public_packet(sync_packet)

    async def decide(self, turn_packet):
        if turn_packet["self"]["player_id"] != self.player_id:
            raise ValueError("行动包玩家错误")
        private = turn_packet["private_information"]
        if str(private.get("role")) != self.profile.role:
            raise ValueError("角色档案与行动包不一致")
        prompt = {k: turn_packet[k] for k in ("game", "public_rules", "self", "private_information", "public_state", "request")}
        prompt["history_policy"] = {"default_context": "current_state_only", "current_round_dialogue_tool": CURRENT_ROUND_DIALOGUE_TOOL_NAME, "max_tool_calls": self.max_tool_calls_per_decision, "max_tool_result_tokens": self.max_tool_result_tokens}
        dialogue = (turn_packet.get("tool_context") or {}).get("current_round_dialogue") or []

        def execute_tool(name: str, arguments):
            del arguments
            if name != CURRENT_ROUND_DIALOGUE_TOOL_NAME:
                return {"error": f"不支持的工具：{name}"}
            return {"round": turn_packet["game"].get("round"), "phase": turn_packet["game"].get("public_phase", turn_packet["game"].get("phase")), "dialogue": dialogue}

        feedback = ""
        for attempt in range(self.max_decision_retries + 1):
            allow_tool = attempt == 0 and self.max_tool_calls_per_decision > 0
            content = {"instruction": self._turn_instruction(turn_packet, feedback), "packet": prompt}
            try:
                raw = await self._complete_json(system=self._system_prompt(private), messages=[{"role": "user", "content": json.dumps(content, ensure_ascii=False)}], tools=[CURRENT_ROUND_DIALOGUE_TOOL] if allow_tool else None, tool_executor=execute_tool if allow_tool else None, max_tool_calls=self.max_tool_calls_per_decision if allow_tool else 0, max_tool_result_tokens=self.max_tool_result_tokens, max_prompt_chars=self.max_prompt_chars)
            except ModelClientError:
                if attempt >= self.max_decision_retries:
                    raise
                feedback = "上次无可用JSON，请重答。"
                continue
            self._record_token_usage(raw)
            action = normalize_decision(raw, turn_packet["request"])
            err = decision_error(action, turn_packet["request"])
            if err is None:
                return action
            if attempt >= self.max_decision_retries:
                raise RuleViolationError(f"非法行动：{err}", attempted_action=action, validation_error=err)
            feedback = f"上一次行动未通过校验：{err}"
        raise RuleViolationError("模型未返回合法行动")

    def agent_manifest(self):
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

    def model_token_usage_snapshot(self):
        return dict(self._model_token_usage)

    def _system_prompt(self, private):
        return render_prompt("player_system.txt", player_id=self.player_id, role=private["role"], team=private["team"], persona=self.persona, role_base=self.profile.base, role_task=self.profile.task, tool_name=CURRENT_ROUND_DIALOGUE_TOOL_NAME, max_tool_calls=self.max_tool_calls_per_decision, max_tool_result_tokens=self.max_tool_result_tokens, max_prompt_chars=self.max_prompt_chars)

    def _turn_instruction(self, turn_packet: Mapping[str, Any], feedback):
        req = turn_packet["request"]
        bits = [self._render_turn_summary(turn_packet, req), f"上一次输出未通过校验：{feedback}" if feedback else "", render_villager_decision_checklist(req)]
        return render_prompt("player_turn_instruction.txt", action_contract=render_action_contract(req), validation_feedback="\n".join(x for x in bits if x))

    async def _complete_json(self, **kw):
        if self.request_coordinator is not None:
            return await self.request_coordinator.complete_json(self.model_client, max_tokens=self.max_tokens, **kw)
        fn = self.model_client.complete_json
        return await fn(max_tokens=self.max_tokens, **kw) if inspect.iscoroutinefunction(fn) else await asyncio.to_thread(fn, max_tokens=self.max_tokens, **kw)

    def _record_token_usage(self, response):
        self._model_token_usage["successful_response_count"] += 1
        self._model_token_usage["api_attempt_count"] += max(1, self._nonnegative_int(getattr(response, "api_attempts", 1), fallback=1) or 1)
        usage = getattr(response, "token_usage", None)
        if isinstance(usage, dict):
            vals = {k: self._nonnegative_int(usage.get(k), fallback=None) for k in ("input_tokens", "output_tokens", "total_tokens")}
            if any(v is not None for v in vals.values()):
                self._model_token_usage["reported_usage_response_count"] += 1
                for k, v in vals.items():
                    if v is not None:
                        self._model_token_usage[k] += v

    @staticmethod
    def _nonnegative_int(value, *, fallback):
        try:
            v = int(value) if not isinstance(value, bool) else -1
        except (TypeError, ValueError):
            return fallback
        return v if v >= 0 else fallback

    def _reset_public_ledger(self, game_id):
        self._public_ledger_game_id = game_id
        self._public_ledger = {"alive_ids": set(), "dead_ids": set(), "known_player_ids": set(), "recent_events": [], "public_claims": []}

    def _ingest_public_packet(self, payload):
        notes = self._extract_public_notes(payload)
        for n in notes:
            self._append_bounded(self._public_ledger["recent_events"], n, _PUBLIC_SNAPSHOT_LIMIT)
            if any(k in n for k in ("claim", "自称", "预言家", "女巫", "猎人", "守卫", "金水", "查杀", "查验", "警徽")):
                self._append_bounded(self._public_ledger["public_claims"], n, _PUBLIC_SNAPSHOT_LIMIT)
        self._public_ledger["known_player_ids"].update(self._extract_player_ids(payload))
        if isinstance(payload, Mapping):
            self._public_ledger["alive_ids"].update(self._collect_id_collection(payload, self._ALIVE_KEYS()))
            self._public_ledger["dead_ids"].update(self._collect_id_collection(payload, self._DEAD_KEYS()))

    def _render_turn_summary(self, turn_packet: Mapping[str, Any], request):
        state = turn_packet.get("public_state")
        notes = self._extract_public_notes(state)
        alive = self._sort_player_ids((self._public_ledger["alive_ids"] | self._extract_player_ids(state, include_collections=True)) - self._public_ledger["dead_ids"])
        dead = self._sort_player_ids(self._public_ledger["dead_ids"] | self._extract_dead_from_notes(notes))
        known = self._sort_player_ids((self._public_ledger["known_player_ids"] | set(alive) | set(dead)) - {self.player_id})
        pool = self._candidate_pool(request, alive, known)
        base = ["【台账】", "存活: " + self._join_ids(alive, 10), "已死: " + self._join_ids(dead, 10)]
        claims = self._collect_claim_block(notes)
        if claims:
            base.append("验/述:" + claims)
        if pool:
            base.append("灰区: " + self._join_ids([p for p in pool if p not in dead][:6], 6))
        phase, kinds = _phase_and_kinds(request)
        vote = any(k in phase for k in ("vote", "投票")) or any("vote" in k for k in kinds)
        detail = self._render_vote_summary(pool, alive, dead, notes) if vote else self._render_speak_summary(pool, alive, dead, notes)
        return self._trim_text("\n".join(["\n".join(base), detail]), _PUBLIC_BLOCK_CHAR_LIMIT + _PHASE_BLOCK_CHAR_LIMIT)

    def _render_vote_summary(self, pool, alive, dead, notes):
        pri = self._priority_check_targets(pool or alive, notes, dead)
        ranked = self._unique_ranked(pri + [x for x in self._rank_candidates(pool, notes, dead) if x[0] not in {p[0] for p in pri}])[:3]
        if len(ranked) < 2:
            ranked += self._rank_candidates([p for p in alive if p not in {x[0] for x in ranked}], notes, dead)[: 2 - len(ranked)]
        lines = ["【投票参考】"]
        if pri:
            lines.append("查杀纪律:可信验链优先，勿被点名就反踩分票。")
        lines += [f"{i}. {p} | 分数={s} | {r}" for i, (p, s, r) in enumerate(ranked, 1)]
        lines.append(f"比较：{ranked[0][0]} vs {ranked[1][0]}；1号可信查杀且无硬反证则归票。" if len(ranked) >= 2 else "候选不足用票型/验链补足，少弃票。")
        return self._trim_text("\n".join(lines), _PHASE_BLOCK_CHAR_LIMIT)

    def _render_speak_summary(self, pool, alive, dead, notes):
        ranked = self._rank_candidates(pool or alive, notes, dead)
        pri = self._priority_check_targets(pool or alive, notes, dead)
        lines = ["【发言参考】"]
        if pri:
            lines.append(f"回应查杀:{pri[0][0]} | {pri[0][2]}")
        lines.append("质疑点:" + (self._trim_text("；".join(f"{p}({r})" for p, _s, r in ranked[:2]), 220) if ranked else "信息不足则引用具体发言/票型。"))
        lines.append("结构:事实→查杀/对跳/票型→倾向与疑问。")
        lines.append("禁复读模板。")
        return self._trim_text("\n".join(lines), _PHASE_BLOCK_CHAR_LIMIT)

    def _rank_candidates(self, pool, notes, dead):
        pri = {x[0] for x in self._priority_check_targets(pool, notes, dead)}
        out = []
        for p in self._sort_player_ids(pool):
            if p in set(dead) or p == self.player_id:
                continue
            score, reasons = self._score_candidate(p, notes)
            if p in pri:
                score += 30; reasons.insert(0, "可信查杀优先于互踩")
            out.append((p, score, "；".join(reasons[:2]) if reasons else "信息不足"))
        return sorted(out, key=lambda x: (-x[1], x[0]))

    def _score_candidate(self, cand, notes):
        score, reasons = 0, []
        for n in notes:
            c = self._trim_text(n, 80); speaker = self._note_actor(n, "speaker") or self._note_actor(n, "voter"); target = self._note_actor(n, "target")
            if self._is_explicit_protection(cand, n):
                score -= 8
                reasons.append(f"保护:{c}")
                continue
            if self._is_explicit_accusation(cand, n):
                bonus = 12 if self._looks_like_check_claim(n) else 6
                score += bonus
                reasons.append(("查杀: " if bonus == 12 else "被点名疑点: ") + c)
                continue
            if target == cand and ("vote" in n.lower() or "票" in n):
                score += 2
                reasons.append(f"票型:{c}")
                continue
            if speaker == cand:
                if any(t in n for t in ("自称", "claim", "预言家", "女巫", "猎人", "守卫")):
                    reasons.append(f"自述:{c}")
                continue
            if self._mentions_player(cand, n) and any(t in n for t in ("矛盾", "票型异常", "冲票", "爆狼")):
                score += 1
                reasons.append(f"弱疑:{c}")
        return score, reasons or ["信息不足"]

    def _priority_check_targets(self, pool, notes, dead):
        cands = set(pool) - set(dead) - {self.player_id}; found: dict[str, tuple[int, str]] = {}
        for n in notes:
            if not self._looks_like_check_claim(n) or self._claim_is_discredited(n):
                continue
            for t in self._extract_accused_targets(n, cands):
                s = 100 + (20 if self._claim_has_authority(n) else 0); r = "可信查杀:" + self._trim_text(n, 90)
                if t not in found or s > found[t][0]:
                    found[t] = (s, r)
        return sorted([(p, s, r) for p, (s, r) in found.items()], key=lambda x: (-x[1], x[0]))

    def _looks_like_check_claim(self, n):
        return "查杀" in n or bool(re.search(r"验.{0,8}(?:狼|狼人)", n))

    def _claim_has_authority(self, n):
        return any(t in n for t in ("预言家", "警长", "警徽", "报验", "查验", "第二验", "昨晚验", "今晚验"))

    def _claim_is_discredited(self, n):
        return any(t in n for t in ("假预言家", "悍跳", "不信", "编的", "造谣", "反咬")) and not any(t in n for t in ("警长", "警徽"))

    def _is_explicit_accusation(self, cand, n):
        return self._mentions_player(cand, n) and (cand in self._extract_accused_targets(n, {cand}) or (self._note_actor(n, "target") == cand and any(t in n for t in ("查杀", "狼", "矛盾", "票型异常", "冲票", "爆狼"))))

    def _is_explicit_protection(self, cand, n):
        if not self._mentions_player(cand, n):
            return False
        if self._note_actor(n, "target") == cand and any(t in n for t in ("金水", "好人", "可信", "银水", "守过", "强神", "验过")):
            return True
        return any(re.search(p, n, re.I) for p in (rf"{re.escape(cand)}.{{0,6}}(?:金水|好人|银水|可信)", rf"(?:金水|好人|银水|可信).{{0,6}}{re.escape(cand)}", rf"验.{{0,6}}{re.escape(cand)}.{{0,6}}(?:好|金水)"))

    def _extract_accused_targets(self, n, cands):
        out = set(); explicit = self._note_actor(n, "target")
        if explicit in cands and any(t in n for t in ("查杀", "狼", "出", "归票")):
            out.add(explicit)
        for c in cands:
            e = re.escape(c)
            if any(re.search(p, n, re.I) for p in (rf"查杀\s*{e}", rf"{e}\s*(?:查杀|是狼|狼人|狼坑|出局|归票)", rf"验.{{0,8}}{e}.{{0,8}}(?:狼|狼人)", rf"(?:出|票|归票)\s*{e}")):
                out.add(c)
        return out

    def _mentions_player(self, cand, n):
        return re.search(rf"(?<![A-Za-z0-9_]){re.escape(cand)}(?![A-Za-z0-9_])", n, re.I) is not None

    def _note_actor(self, n, field):
        m = re.search(rf"\b{re.escape(field)}=([^\s|，,；;。]+)", n, re.I)
        return m.group(1) if m else ""

    def _candidate_pool(self, request, alive, known):
        pool = set(alive) | set(known)
        for a in request.get("allowed_actions", []) if isinstance(request.get("allowed_actions"), list) else []:
            tids = a.get("target_ids") if isinstance(a, Mapping) else None
            if isinstance(tids, list):
                pool.update(str(t) for t in tids if self._is_scalar_id(t))
        pool.discard(self.player_id); pool.difference_update(self._public_ledger["dead_ids"])
        return self._sort_player_ids(pool)

    def _collect_claim_block(self, notes):
        keys = ("claim", "自称", "预言家", "女巫", "猎人", "守卫", "金水", "查杀", "查验", "警徽")
        claims = [self._trim_text(n, 90) for n in notes if any(k in n for k in keys)] or [self._trim_text(x, 90) for x in self._public_ledger["public_claims"][:3]]
        return " | ".join(claims[:3])

    def _extract_public_notes(self, payload):
        notes = []
        self._walk_public_payload(payload, notes, 0)
        return self._dedupe_preserve_order(notes)

    def _walk_public_payload(self, obj, notes, depth: int):
        if depth > 4:
            return
        if isinstance(obj, Mapping):
            snap = self._snapshot_mapping(obj)
            if snap:
                notes.append(snap)
            for v in obj.values():
                self._walk_public_payload(v, notes, depth + 1)
        elif isinstance(obj, list):
            for x in obj[:25]:
                self._walk_public_payload(x, notes, depth + 1)

    def _snapshot_mapping(self, mp):
        m = {str(k).lower(): v for k, v in mp.items()}; ev = self._first_scalar(m, ("event", "event_type", "type", "kind", "action")).lower()
        vote = any(k in m for k in ("voter_id", "voter", "vote_target", "target_id", "target", "ballot", "vote")) or "vote" in ev
        claim = any(k in m for k in ("speaker_id", "claimer_id", "claim", "role_claim", "speech", "text", "content", "message", "reason")) or any(t in ev for t in ("claim", "speak", "speech", "say", "发言", "自称"))
        sheriff = any(k in m for k in ("sheriff_id", "badge_holder", "badge_holder_id", "badge_from", "badge_to", "badge")) or any(t in ev for t in ("sheriff", "badge", "警徽", "警长"))
        death = any(k in m for k in ("dead", "death", "dead_id", "dead_player_id")) or any(t in ev for t in ("death", "dead", "kill", "out", "死亡", "出局"))
        if not (vote or claim or sheriff or death):
            return None
        player = self._first_scalar(m, ("player_id", "speaker_id", "voter_id", "claimer_id", "actor_id", "source_id")); target = self._first_scalar(m, ("target_id", "target", "vote_target", "to_id", "suspect_id"))
        parts = [ev] if ev else []
        if vote:
            parts.append(f"vote voter={player or self._first_scalar(m, ('voter',)) or '?'} target={target or '?'}")
        elif sheriff:
            parts.append(f"sheriff holder={self._first_scalar(m, ('badge_holder', 'badge_holder_id', 'sheriff_id')) or '?'} from={self._first_scalar(m, ('from_id', 'source_id', 'badge_from')) or '?'} to={target or '?'}")
        elif death:
            parts.append(f"death player={player or self._first_scalar(m, ('dead_id', 'dead_player_id')) or '?'}")
        elif claim:
            parts.append(f"claim speaker={player or '?'}"); role = self._first_scalar(m, ("role", "claim_role", "claimed_role"))
            if role: parts.append(f"role={role}")
        for label, keys in (("at", ("round", "day", "turn", "phase", "channel")), ("text", ("text", "content", "message", "reason", "speech"))):
            val = self._first_scalar(m, keys)
            if val: parts.append(f"{label}={self._trim_text(val, _PUBLIC_NOTE_CHAR_LIMIT)}")
        return self._trim_text(" ".join(parts), _PUBLIC_NOTE_CHAR_LIMIT)

    def _extract_player_ids(self, payload, *, include_collections: bool = False):
        ids = set()
        def walk(o: Any):
            if isinstance(o, Mapping):
                lm = {str(k).lower(): v for k, v in o.items()}
                for k, v in lm.items():
                    if k in self._player_id_keys() and self._is_scalar_id(v): ids.add(str(v))
                    if include_collections and k in self._collection_id_keys(): ids.update(self._as_id_set(v))
                for v in o.values(): walk(v)
            elif isinstance(o, list):
                for x in o[:25]: walk(x)
        walk(payload); ids.discard(self.player_id); return ids

    def _collect_id_collection(self, mapping, keys):
        out = set()
        for k, v in {str(k).lower(): v for k, v in mapping.items()}.items():
            if k in keys: out.update(self._as_id_set(v))
        return out

    def _extract_dead_from_notes(self, notes):
        out = set()
        for n in notes:
            if any(t in n.lower() for t in ("death", "dead", "死亡", "出局", "淘汰")): out.update(self._ids_from_text(n))
        return out

    def _ids_from_text(self, text):
        return {t for t in (x.strip("()[]{}<>：:=，。,.、") for x in re.split(r"[\s,，;；|/\\]+", text)) if t and self._looks_like_player_id(t)}

    def _as_id_set(self, v):
        return {str(x) for x in v if self._is_scalar_id(x)} if isinstance(v, (list, tuple)) else ({str(v)} if self._is_scalar_id(v) else set())

    def _append_bounded(self, items, value, limit):
        if value and value not in items:
            items.append(value); del items[: max(0, len(items) - limit)]

    def _first_scalar(self, mapping, keys: tuple[str, ...]):
        for k in keys:
            v = mapping.get(k)
            if isinstance(v, (str, int, float)) and not isinstance(v, bool): return self._trim_text(str(v), _PUBLIC_NOTE_CHAR_LIMIT)
        return ""

    def _trim_text(self, text, limit):
        s = " ".join(str(text).split())
        return s if len(s) <= limit else s[: max(0, limit - 1)] + "…"

    def _dedupe_preserve_order(self, items):
        seen = set(); out = []
        for x in items:
            if x and x not in seen: seen.add(x); out.append(x)
        return out

    def _unique_ranked(self, items: list[tuple[str, int, str]]):
        seen = set(); out = []
        for x in items:
            if x[0] not in seen: seen.add(x[0]); out.append(x)
        return out

    def _join_ids(self, ids, limit):
        return "未知" if not ids else "、".join(ids[:limit]) + (f" 等{len(ids)-limit}人" if len(ids) > limit else "")

    def _sort_player_ids(self, ids: Any):
        def key(v: str):
            m = _PLAYER_ID_SUFFIX.search(v); return (_PLAYER_ID_SUFFIX.sub("", v), int(m.group(1)) if m else 10**9, v)
        return sorted(self._dedupe_preserve_order([str(x) for x in ids if str(x)]), key=key)

    def _looks_like_player_id(self, value):
        s = str(value).strip(); return bool(s) and len(s) <= 16 and any(ch.isdigit() for ch in s)

    def _is_scalar_id(self, v):
        return (isinstance(v, int) and not isinstance(v, bool)) or (isinstance(v, str) and self._looks_like_player_id(v))

    @staticmethod
    def _player_id_keys():
        return {"player_id", "speaker_id", "voter_id", "claimer_id", "actor_id", "source_id", "target_id", "from_id", "to_id", "badge_holder", "badge_holder_id", "sheriff_id", "dead_id", "dead_player_id"}

    @staticmethod
    def _collection_id_keys():
        return {"alive", "alive_ids", "alive_player_ids", "dead", "dead_ids", "dead_player_ids", "players", "player_ids", "target_ids", "candidates", "survivors", "living", "graveyard", "claims", "votes"}

    @staticmethod
    def _ALIVE_KEYS():
        return {"alive", "alive_ids", "alive_player_ids", "survivors", "living", "live_ids", "current_alive"}

    @staticmethod
    def _DEAD_KEYS():
        return {"dead", "dead_ids", "dead_player_ids", "eliminated", "casualties", "deaths", "graveyard"}

    def _extract_game_id(self, payload: Mapping[str, Any]):
        for src in (payload, payload.get("game") if isinstance(payload.get("game"), Mapping) else {}):
            for k in ("game_id", "gameid", "match_id", "matchid", "session_id"):
                v = src.get(k)
                if isinstance(v, (str, int, float)) and not isinstance(v, bool): return str(v)
        return None

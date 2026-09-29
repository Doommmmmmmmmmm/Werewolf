"""狼人杀平民 Task-Agent：轻量公共台账 + 结构化票口提示。

本文件只使用行动包与公开同步中依法可见的信息；不访问网络、文件系统、环境变量、
进程或隐藏状态。每次决策仍从新的模型会话开始，历史需要时只能通过受限工具读取当前
轮已公开对话，且工具预算不因纠错重试而扩大。
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
_RAW_NUM = re.compile(r"(?<![A-Za-z0-9])(\d{1,2})(?![A-Za-z0-9])")
CURRENT_ROUND_DIALOGUE_TOOL_NAME = "read_current_round_dialogue"
DEFAULT_MAX_TOOL_CALLS_PER_DECISION = 5
DEFAULT_MAX_TOOL_RESULT_TOKENS = 1000
DEFAULT_MAX_PROMPT_CHARS = 8000
DEFAULT_MAX_DECISION_RETRIES = 1
_SPEECH_ACTION_KINDS = frozenset({"speak", "last_words"})
_PUBLIC_SNAPSHOT_LIMIT = 12
_PUBLIC_NOTE_CHAR_LIMIT = 120
_PUBLIC_BLOCK_CHAR_LIMIT = 1200
_PHASE_BLOCK_CHAR_LIMIT = 650


CURRENT_ROUND_DIALOGUE_TOOL: dict[str, Any] = {
    "name": CURRENT_ROUND_DIALOGUE_TOOL_NAME,
    "description": "读取当前昼夜轮次中、当前玩家依法可见的已发生发言。",
    "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
}


def render_villager_decision_checklist(request: Mapping[str, Any]) -> str:
    """短清单：强调查杀链优先、强神与金水保护，避免长推理空转。"""

    phase = str(request.get("phase") or request.get("channel") or request.get("kind") or "").lower()
    raw_actions = request.get("allowed_actions")
    allowed_kinds = {
        str(item.get("kind") or "").lower()
        for item in raw_actions
        if isinstance(item, Mapping)
    } if isinstance(raw_actions, list) else set()
    is_vote = any(key in phase for key in ("vote", "投票")) or any("vote" in kind for kind in allowed_kinds)
    is_speak = any(key in phase for key in ("speak", "discussion", "发言", "day", "白天")) or bool(
        allowed_kinds & {"speak", "discussion", "day_speak"}
    )
    lines = [
        "【平民快检】先看系统生死/投票，再看可信查验链，最后才看普通发言。",
        "唯一带徽且无硬对跳的预言家报出存活查杀时：投票必须优先跟该查杀；发言必须明确该票口。",
        "无硬对跳或事实矛盾时，不把金水、带徽预言家、守卫/猎人/女巫等自保神职列为第一出局位。",
    ]
    if is_vote:
        lines.append("投票只在合法 target_id 中选；少弃票；至少比较首推票口和一个备选灰位。")
    if is_speak:
        lines.append("发言给出一个关注对象、一个暂不投对象，并引用台账中的公开依据。")
    return "\n".join(lines)


def render_action_contract(request: Mapping[str, Any]) -> str:
    """把本次 allowed_actions 压缩为最终 JSON 契约。"""

    actions = [item for item in request.get("allowed_actions", []) if isinstance(item, Mapping)]
    lines = [
        "【最终 JSON 契约】只能输出一个 JSON 对象，不要 Markdown/解释/额外字段。",
        f"phase={request.get('phase') or 'unknown'} channel={request.get('channel') or 'unknown'}",
    ]
    if not actions:
        return "\n".join(lines + ["当前没有可用行动；不要臆造行动。"])
    for allowed in actions:
        kind = str(allowed.get("kind") or "")
        if not kind:
            continue
        example: dict[str, str] = {"kind": kind}
        targets = [str(x) for x in allowed.get("target_ids", [])] if isinstance(allowed.get("target_ids"), list) else []
        if kind in _SPEECH_ACTION_KINDS:
            example["text"] = "我跟公开查验链投票，先保神职和金水。"
        elif targets:
            example["target_id"] = targets[0]
        lines.append(json.dumps(example, ensure_ascii=False, separators=(",", ":")))
        if targets:
            lines.append("target_id 只能取：" + "、".join(targets))
        if kind in _SPEECH_ACTION_KINDS:
            bits = []
            if allowed.get("max_chars") is not None:
                bits.append(f"text≤{allowed.get('max_chars')}字")
            if allowed.get("require_chinese"):
                bits.append("含中文")
            lines.append("发言约束：" + "；".join(bits))
    lines.append("协议字段 kind/target_id 必须保留英文原样。")
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
        self.max_tool_calls_per_decision = min(DEFAULT_MAX_TOOL_CALLS_PER_DECISION, max(0, int(max_tool_calls_per_decision)))
        self.max_tool_result_tokens = min(DEFAULT_MAX_TOOL_RESULT_TOKENS, max(1, int(max_tool_result_tokens)))
        self.max_prompt_chars = min(DEFAULT_MAX_PROMPT_CHARS, max(1, int(max_prompt_chars)))
        self._model_token_usage: dict[str, int] = {
            "successful_response_count": 0,
            "api_attempt_count": 0,
            "reported_usage_response_count": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }
        self._public_ledger_game_id: str | None = None
        self._reset_public_ledger(None)

    async def observe(self, sync_packet: dict[str, Any]) -> None:
        """接收公开同步并维护轻量台账。"""

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
        prompt_packet = self._compact_prompt_packet(turn_packet)
        dialogue = self._bounded_tool_dialogue((turn_packet.get("tool_context") or {}).get("current_round_dialogue") or [])

        def execute_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
            del arguments
            if name != CURRENT_ROUND_DIALOGUE_TOOL_NAME:
                return {"error": f"不支持的工具：{name}"}
            return {
                "round": turn_packet.get("game", {}).get("round"),
                "phase": turn_packet.get("game", {}).get("public_phase", turn_packet.get("game", {}).get("phase")),
                "dialogue": dialogue,
            }

        system = self._system_prompt(private)
        feedback = ""
        for attempt in range(self.max_decision_retries + 1):
            allow_tool = attempt == 0 and self.max_tool_calls_per_decision > 0
            user_content = {"instruction": self._turn_instruction(turn_packet, feedback), "packet": prompt_packet}
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
                feedback = "上一轮模型调用没有产生可用 JSON，请直接按契约作答。"
                continue
            self._record_token_usage(raw)
            action = normalize_decision(raw, request)
            error = decision_error(action, request)
            if error is None:
                return action
            if attempt >= self.max_decision_retries:
                raise RuleViolationError(f"模型返回非法行动：{error}", attempted_action=action, validation_error=error)
            feedback = error
        raise RuleViolationError("模型未返回合法行动", attempted_action={}, validation_error="empty")

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
            parts.append("上一次输出未通过校验：" + feedback)
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

    def _compact_prompt_packet(self, turn_packet: Mapping[str, Any]) -> dict[str, Any]:
        request = turn_packet["request"]
        game = turn_packet.get("game", {}) if isinstance(turn_packet.get("game"), Mapping) else {}
        private = turn_packet.get("private_information", {}) if isinstance(turn_packet.get("private_information"), Mapping) else {}
        return {
            "game": {k: game.get(k) for k in ("round", "phase", "public_phase") if k in game},
            "self": turn_packet.get("self"),
            "private_information": {k: private.get(k) for k in ("role", "team") if k in private},
            "request": request,
            "public_summary": self._render_turn_summary(turn_packet, request),
            "history_policy": {
                "default_context": "compact_public_ledger_only",
                "current_round_dialogue_tool": CURRENT_ROUND_DIALOGUE_TOOL_NAME,
                "max_tool_calls": self.max_tool_calls_per_decision,
                "max_tool_result_tokens": self.max_tool_result_tokens,
            },
        }

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
            "sheriff_holders": set(),
            "death_notices": [],
        }

    def _ingest_public_packet(self, payload: Any) -> None:
        notes = self._extract_public_notes(payload)
        self._merge_public_notes(notes)
        ids = self._extract_player_ids(payload)
        self._public_ledger["known_player_ids"].update(ids)
        if isinstance(payload, Mapping):
            self._public_ledger["alive_ids"].update(self._collect_id_collection(payload, self._ALIVE_KEYS()))
            self._public_ledger["dead_ids"].update(self._collect_id_collection(payload, self._DEAD_KEYS()))

    def _merge_public_notes(self, notes: list[str]) -> None:
        for note in notes:
            if not note:
                continue
            lower = note.lower()
            self._append_bounded(self._public_ledger["recent_events"], note, _PUBLIC_SNAPSHOT_LIMIT)
            if "vote" in lower or "票" in note:
                self._append_bounded(self._public_ledger["vote_snapshots"], note, _PUBLIC_SNAPSHOT_LIMIT)
            if any(term in lower for term in ("death", "dead")) or any(term in note for term in ("死亡", "出局", "淘汰", "刀")):
                self._append_bounded(self._public_ledger["death_notices"], note, _PUBLIC_SNAPSHOT_LIMIT)
            if any(term in lower for term in ("claim", "sheriff", "badge")) or any(
                term in note for term in ("预言家", "女巫", "猎人", "守卫", "金水", "查杀", "查验", "警徽", "警长")
            ):
                self._append_bounded(self._public_ledger["public_claims"], note, _PUBLIC_SNAPSHOT_LIMIT)
            if any(term in lower for term in ("sheriff", "badge")) or any(term in note for term in ("警徽", "警长")):
                self._public_ledger["sheriff_holders"].update(self._ids_after_labels(note, ("holder=", "sheriff_id=", "警长", "警徽")))

    def _render_turn_summary(self, turn_packet: Mapping[str, Any], request: Mapping[str, Any]) -> str:
        public_state = turn_packet.get("public_state")
        notes = self._dedupe_preserve_order(self._extract_public_notes(public_state) + list(self._public_ledger["recent_events"]))
        alive = self._sort_player_ids((set(self._public_ledger["alive_ids"]) | self._extract_player_ids(public_state, include_collections=True)) - set(self._public_ledger["dead_ids"]))
        dead = self._sort_player_ids(set(self._public_ledger["dead_ids"]) | self._extract_dead_from_notes(notes))
        known = self._sort_player_ids(set(self._public_ledger["known_player_ids"]) | set(alive) | set(dead))
        candidate_pool = self._candidate_pool(request, alive, known)
        analysis = self._analyze_public_claims(notes, alive, dead)

        lines = ["【压缩公共台账】"]
        lines.append("存活=" + self._join_ids(alive, limit=10) + "；已死=" + self._join_ids(dead, limit=8))
        if analysis["credible_prophet"]:
            lines.append("可信预言家=" + analysis["credible_prophet"])
        if analysis["live_kills"]:
            lines.append("存活查杀=" + "、".join(analysis["live_kills"][:4]))
        if analysis["goldens"]:
            lines.append("金水/保护=" + "、".join(analysis["goldens"][:5]))
        if analysis["role_claims"]:
            role_bits = [f"{pid}:{'/'.join(sorted(roles))}" for pid, roles in sorted(analysis["role_claims"].items())[:5]]
            lines.append("神职自称=" + "、".join(role_bits))
        phase = str(request.get("phase") or request.get("channel") or request.get("kind") or "").lower()
        allowed = request.get("allowed_actions")
        kinds = {str(a.get("kind") or "").lower() for a in allowed if isinstance(a, Mapping)} if isinstance(allowed, list) else set()
        is_vote = "vote" in phase or "投票" in phase or any("vote" in kind for kind in kinds)
        lines.append(self._render_vote_summary(candidate_pool, alive, dead, notes, analysis) if is_vote else self._render_speak_summary(candidate_pool, alive, dead, notes, analysis))
        return self._trim_text("\n".join(line for line in lines if line), _PUBLIC_BLOCK_CHAR_LIMIT + _PHASE_BLOCK_CHAR_LIMIT)

    def _render_vote_summary(self, candidate_pool: list[str], alive_ids: list[str], dead_ids: list[str], notes: list[str], analysis: dict[str, Any]) -> str:
        ranked = self._rank_candidates(candidate_pool, notes, dead_ids, analysis)
        if len(ranked) < 2:
            existing = {item[0] for item in ranked}
            ranked.extend(self._rank_candidates([pid for pid in alive_ids if pid not in existing], notes, dead_ids, analysis))
        if not ranked and candidate_pool:
            ranked = [(candidate_pool[0], 0, "公开信息不足")]
        lines = ["【投票参考】"]
        for i, (pid, score, reason) in enumerate(ranked[:3], 1):
            lines.append(f"{i}. {pid} 分={score} {reason}")
        if ranked:
            if ranked[0][0] in analysis["live_kills"] and analysis["credible_prophet"]:
                lines.append(f"结论：优先投 {ranked[0][0]}（唯一带徽/无硬对跳预言家查杀）。")
            else:
                alt = ranked[1][0] if len(ranked) > 1 else "无"
                lines.append(f"结论：比较 {ranked[0][0]} vs {alt}，不优先投保护位。")
        return self._trim_text("\n".join(lines), _PHASE_BLOCK_CHAR_LIMIT)

    def _render_speak_summary(self, candidate_pool: list[str], alive_ids: list[str], dead_ids: list[str], notes: list[str], analysis: dict[str, Any]) -> str:
        ranked = self._rank_candidates(candidate_pool or alive_ids, notes, dead_ids, analysis)
        focus = ranked[0] if ranked else ("信息不足", 0, "公开信息不足")
        avoid = self._least_suspicious_candidate(candidate_pool or alive_ids, notes, dead_ids, analysis)
        lines = ["【发言参考】", f"关注: {focus[0]} | {focus[2]}"]
        if avoid:
            lines.append(f"暂不投: {avoid[0]} | {avoid[2]}")
        else:
            lines.append("暂不投: 金水、带徽预言家、未对跳强神。")
        return self._trim_text("\n".join(lines), _PHASE_BLOCK_CHAR_LIMIT)

    def _rank_candidates(self, candidate_pool: list[str], notes: list[str], dead_ids: list[str], analysis: dict[str, Any]) -> list[tuple[str, int, str]]:
        dead = set(dead_ids)
        protected = set(analysis["goldens"]) | set(analysis["protected_roles"])
        ranked: list[tuple[str, int, str]] = []
        for pid in self._sort_player_ids(candidate_pool):
            if pid == self.player_id or pid in dead:
                continue
            score, reason = self._score_candidate(pid, notes, analysis, protected)
            ranked.append((pid, score, reason))
        ranked.sort(key=lambda x: (-x[1], self._pid_num(x[0]), x[0]))
        return ranked

    def _least_suspicious_candidate(self, candidate_pool: list[str], notes: list[str], dead_ids: list[str], analysis: dict[str, Any]) -> tuple[str, int, str] | None:
        ranked = self._rank_candidates(candidate_pool, notes, dead_ids, analysis)
        if not ranked:
            return None
        ranked.sort(key=lambda x: (x[1], self._pid_num(x[0]), x[0]))
        return ranked[0]

    def _score_candidate(self, pid: str, notes: list[str], analysis: dict[str, Any], protected: set[str]) -> tuple[int, str]:
        if pid in analysis["live_kills"] and analysis["credible_prophet"]:
            return 10000, f"可信预言家 {analysis['credible_prophet']} 的存活查杀"
        if pid == analysis["credible_prophet"]:
            return -9000, "带徽/唯一预言家，除非硬对跳或事实冲突否则保护"
        if pid in analysis["goldens"]:
            return -7000, "公开金水/被验好，非首推"
        if pid in analysis["protected_roles"]:
            roles = "/".join(sorted(analysis["role_claims"].get(pid, {"神职"})))
            return -6000, f"未被可信查杀的{roles}自称，先不票强神"

        score = 0
        reasons: list[str] = []
        if pid in analysis["contested_claimers"]:
            score += 80
            reasons.append("存在对跳/身份冲突")
        if pid in analysis["any_kills"]:
            score += 60
            reasons.append("被公开报过查杀")
        related = [n for n in notes if self._mentions_player(n, pid)]
        if any(any(term in n for term in ("爆狼", "矛盾", "票型异常", "冲票", "反水", "悍跳")) for n in related):
            score += 25
            reasons.append("发言/票型有明确矛盾词")
        if any("vote" in n.lower() or "票" in n for n in related):
            score += 8
            reasons.append("进入公开票型讨论")
        if not reasons:
            reasons.append("灰区：未被金水或神职保护")
        return score, "；".join(reasons[:2])

    def _analyze_public_claims(self, notes: list[str], alive_ids: list[str], dead_ids: list[str]) -> dict[str, Any]:
        alive, dead = set(alive_ids), set(dead_ids)
        prophet_claimers: set[str] = set()
        role_claims: dict[str, set[str]] = {}
        kills_by: dict[str, set[str]] = {}
        goldens_by: dict[str, set[str]] = {}
        badge_holders = set(self._public_ledger["sheriff_holders"])
        hard_contest = False

        for note in notes:
            speaker = self._speaker_from_note(note)
            text = note
            if any(term in note for term in ("警徽", "警长")) or any(term in note.lower() for term in ("sheriff", "badge")):
                badge_holders.update(self._ids_after_labels(note, ("holder=", "to=", "sheriff_id=", "警长", "警徽")))
            if speaker and any(term in text for term in ("预言家", "先知", "验人链", "报验", "查验")):
                prophet_claimers.add(speaker)
                role_claims.setdefault(speaker, set()).add("预言家")
            if speaker:
                for role in ("守卫", "猎人", "女巫", "白痴", "预言家"):
                    if role in text and any(term in text for term in ("自称", "claim", "跳", "我是", "身份")):
                        role_claims.setdefault(speaker, set()).add(role)
                kills = {pid for pid in self._ids_near_terms(text, ("查杀", "狼", "主票")) if pid != speaker}
                if "查杀" in text or "主票" in text:
                    kills_by.setdefault(speaker, set()).update(kills)
                golds = {pid for pid in self._ids_near_terms(text, ("金水", "好人", "验好")) if pid != speaker}
                if any(term in text for term in ("金水", "验好")):
                    goldens_by.setdefault(speaker, set()).update(golds)
            if any(term in text for term in ("对跳", "硬跳", "双预", "两个预言家")):
                hard_contest = True

        if len(prophet_claimers) > 1:
            hard_contest = True
        credible = ""
        if len(prophet_claimers) == 1 and not hard_contest:
            only = next(iter(prophet_claimers))
            if only in badge_holders or any(self._mentions_player(n, only) and ("警徽" in n or "badge" in n.lower()) for n in notes):
                credible = only
        elif not prophet_claimers and len(badge_holders) == 1:
            holder = next(iter(badge_holders))
            if any(self._mentions_player(n, holder) and any(t in n for t in ("查杀", "金水", "验人", "查验", "主票")) for n in notes):
                credible = holder

        live_kills = self._sort_player_ids([pid for pid in kills_by.get(credible, set()) if pid in alive and pid not in dead]) if credible else []
        goldens = self._sort_player_ids([pid for pid in goldens_by.get(credible, set()) if pid not in dead]) if credible else []
        any_kills = self._sort_player_ids([pid for targets in kills_by.values() for pid in targets if pid not in dead])
        protected_roles = self._sort_player_ids([pid for pid, roles in role_claims.items() if pid not in dead and pid not in live_kills and roles & {"守卫", "猎人", "女巫", "白痴", "预言家"}])
        contested_claimers = self._sort_player_ids([pid for pid, roles in role_claims.items() if "预言家" in roles and hard_contest])
        return {
            "credible_prophet": credible,
            "live_kills": live_kills,
            "goldens": goldens,
            "any_kills": any_kills,
            "role_claims": role_claims,
            "protected_roles": protected_roles,
            "contested_claimers": contested_claimers,
        }

    def _candidate_pool(self, request: Mapping[str, Any], alive_ids: list[str], known_ids: list[str]) -> list[str]:
        pool: set[str] = set(alive_ids) | set(known_ids)
        for action in request.get("allowed_actions", []) if isinstance(request.get("allowed_actions"), list) else []:
            if not isinstance(action, Mapping):
                continue
            targets = action.get("target_ids")
            if isinstance(targets, list):
                pool.update(str(t) for t in targets if self._is_scalar_id(t))
        pool.discard(self.player_id)
        pool.difference_update(self._public_ledger["dead_ids"])
        return self._sort_player_ids(list(pool))

    def _extract_public_notes(self, payload: Any) -> list[str]:
        notes: list[str] = []
        self._walk_public_payload(payload, notes, 0)
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
            for item in payload[:20]:
                self._walk_public_payload(item, notes, depth + 1)

    def _snapshot_mapping(self, mapping: Mapping[str, Any]) -> str | None:
        lower = {str(k).lower(): v for k, v in mapping.items()}
        event = self._first_scalar(lower, ("event", "event_type", "type", "kind", "action"))
        event_l = event.lower() if event else ""
        vote_like = any(k in lower for k in ("voter_id", "vote_target", "target_id", "target", "vote")) or "vote" in event_l
        claim_like = any(k in lower for k in ("speaker_id", "claimer_id", "claim", "role_claim", "speech", "text", "content", "message", "reason")) or any(t in event_l for t in ("claim", "speak", "speech", "say"))
        sheriff_like = any(k in lower for k in ("sheriff_id", "badge_holder", "badge_holder_id", "badge_from", "badge_to", "badge")) or any(t in event_l for t in ("sheriff", "badge"))
        death_like = any(k in lower for k in ("dead", "death", "dead_id", "dead_player_id")) or any(t in event_l for t in ("death", "dead", "kill", "out"))
        if not (vote_like or claim_like or sheriff_like or death_like):
            return None
        player = self._first_scalar(lower, ("player_id", "speaker_id", "voter_id", "claimer_id", "actor_id", "source_id"))
        target = self._first_scalar(lower, ("target_id", "target", "vote_target", "to_id", "suspect_id"))
        holder = self._first_scalar(lower, ("badge_holder", "badge_holder_id", "sheriff_id"))
        role = self._first_scalar(lower, ("role", "claim_role", "claimed_role"))
        text = self._first_scalar(lower, ("text", "content", "message", "reason", "speech"))
        parts: list[str] = []
        if event:
            parts.append(event_l)
        if vote_like:
            parts.append(f"vote voter={player or '?'} target={target or '?'}")
        elif sheriff_like:
            parts.append(f"sheriff holder={holder or target or '?'}")
        elif death_like:
            parts.append(f"death player={player or self._first_scalar(lower, ('dead_id', 'dead_player_id')) or '?'}")
        elif claim_like:
            parts.append(f"claim speaker={player or '?'}")
            if role:
                parts.append(f"role={role}")
        if text:
            parts.append("text=" + self._trim_text(text, _PUBLIC_NOTE_CHAR_LIMIT))
        return self._trim_text(" ".join(parts), _PUBLIC_NOTE_CHAR_LIMIT)

    def _bounded_tool_dialogue(self, dialogue: Any) -> list[Any]:
        if not isinstance(dialogue, list):
            return []
        result: list[Any] = []
        used = 0
        for item in dialogue[-12:]:
            compact = self._trim_text(json.dumps(item, ensure_ascii=False, default=str), 180)
            if used + len(compact) > self.max_tool_result_tokens:
                break
            result.append(compact)
            used += len(compact)
        return result

    def _extract_player_ids(self, payload: Any, *, include_collections: bool = False) -> set[str]:
        ids: set[str] = set()

        def walk(obj: Any) -> None:
            if isinstance(obj, Mapping):
                lower = {str(k).lower(): v for k, v in obj.items()}
                for key, value in lower.items():
                    if key in self._player_id_keys() and self._is_scalar_id(value):
                        ids.add(str(value))
                    if include_collections and key in self._collection_id_keys():
                        ids.update(self._as_id_set(value))
                for value in obj.values():
                    walk(value)
            elif isinstance(obj, list):
                for item in obj[:20]:
                    walk(item)
        walk(payload)
        ids.discard(self.player_id)
        return ids

    def _collect_id_collection(self, mapping: Mapping[str, Any], keys: set[str]) -> set[str]:
        lower = {str(k).lower(): v for k, v in mapping.items()}
        out: set[str] = set()
        for key, value in lower.items():
            if key in keys:
                out.update(self._as_id_set(value))
        return out

    def _extract_dead_from_notes(self, notes: list[str]) -> set[str]:
        dead: set[str] = set()
        for note in notes:
            if any(t in note.lower() for t in ("death", "dead")) or any(t in note for t in ("死亡", "出局", "淘汰")):
                dead.update(self._ids_from_text(note))
        return dead

    def _ids_from_text(self, text: str) -> set[str]:
        ids = {m.group(0) for m in re.finditer(r"p\d+", text, re.IGNORECASE)}
        ids.update("p" + m.group(1) for m in _RAW_NUM.finditer(text))
        return {pid.lower() for pid in ids if self._looks_like_player_id(pid)}

    def _ids_near_terms(self, text: str, terms: tuple[str, ...]) -> set[str]:
        if not any(term in text for term in terms):
            return set()
        found: set[str] = set()
        for sentence in re.split(r"[。！？!?；;\n]", text):
            if any(term in sentence for term in terms):
                found.update(self._ids_from_text(sentence))
        return found or self._ids_from_text(text)

    def _ids_after_labels(self, text: str, labels: tuple[str, ...]) -> set[str]:
        ids: set[str] = set()
        lower = text.lower()
        for label in labels:
            pos = lower.find(label.lower())
            if pos >= 0:
                ids.update(self._ids_from_text(text[pos : pos + 40]))
        return ids

    def _speaker_from_note(self, note: str) -> str:
        for pattern in (r"speaker=([A-Za-z]*\d+)", r"player_id=([A-Za-z]*\d+)", r"\b(p\d+)\b"):
            m = re.search(pattern, note, re.IGNORECASE)
            if m:
                pid = m.group(1).lower()
                return pid if pid.startswith("p") else "p" + pid
        return ""

    def _mentions_player(self, text: str, pid: str) -> bool:
        return pid.lower() in self._ids_from_text(text) or pid.lower() in text.lower()

    def _as_id_set(self, value: Any) -> set[str]:
        if isinstance(value, (list, tuple, set)):
            return {str(item).lower() for item in value if self._is_scalar_id(item)}
        if self._is_scalar_id(value):
            return {str(value).lower()}
        return set()

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
        return isinstance(value, str) and self._looks_like_player_id(value)

    def _trim_text(self, text: str, limit: int) -> str:
        normalized = " ".join(str(text).split())
        return normalized if len(normalized) <= limit else normalized[: max(0, limit - 1)] + "…"

    def _dedupe_preserve_order(self, items: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for item in items:
            if item and item not in seen:
                seen.add(item)
                out.append(item)
        return out

    def _join_ids(self, ids: list[str], *, limit: int) -> str:
        if not ids:
            return "未知"
        shown = ids[:limit]
        suffix = f"等{len(ids) - len(shown)}人" if len(ids) > len(shown) else ""
        return "、".join(shown) + suffix

    def _sort_player_ids(self, ids: list[str] | set[str]) -> list[str]:
        unique = self._dedupe_preserve_order([str(x).lower() for x in ids if str(x)])
        return sorted(unique, key=lambda value: (re.sub(r"\d+$", "", value), self._pid_num(value), value))

    def _pid_num(self, value: str) -> int:
        m = _PLAYER_ID_SUFFIX.search(str(value))
        return int(m.group(1)) if m else 10**9

    def _looks_like_player_id(self, value: str) -> bool:
        text = str(value).strip().lower()
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
        return {"alive", "alive_ids", "alive_player_ids", "dead", "dead_ids", "dead_player_ids", "players", "player_ids", "target_ids", "candidates", "survivors", "living", "graveyard"}

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

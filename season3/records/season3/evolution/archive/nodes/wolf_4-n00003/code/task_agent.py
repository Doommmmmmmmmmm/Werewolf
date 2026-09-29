"""精简版狼人 Task-Agent。

只使用本次行动包、受限工具返回和少量内存摘要；不访问外部文件/网络/环境。
夜间目标选择采用硬收敛：存活狼队友默认从刀口候选中过滤；主刀合法时只推荐/执行主刀。
"""

from __future__ import annotations

import asyncio
from collections import Counter
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
CURRENT_ROUND_DIALOGUE_TOOL_NAME = "read_current_round_dialogue"
DEFAULT_MAX_TOOL_CALLS_PER_DECISION = 5
DEFAULT_MAX_TOOL_RESULT_TOKENS = 1000
DEFAULT_MAX_PROMPT_CHARS = 12000
DEFAULT_MAX_DECISION_RETRIES = 2
_SPEECH_ACTION_KINDS = frozenset({"speak", "last_words"})
_MAX_HISTORY = 6
_SNIP = 90

CURRENT_ROUND_DIALOGUE_TOOL: dict[str, Any] = {
    "name": CURRENT_ROUND_DIALOGUE_TOOL_NAME,
    "description": "读取当前昼夜轮次中、当前玩家依法可见的已发生发言。",
    "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
}


def render_action_contract(request: Mapping[str, Any]) -> str:
    actions = [a for a in request.get("allowed_actions", []) if isinstance(a, Mapping)]
    lines = [
        "【最终 JSON 契约】只能输出一个 JSON 对象，不要 Markdown/解释。",
        f"phase={request.get('phase','unknown')} channel={request.get('channel','unknown')}",
    ]
    for action in actions:
        kind = str(action.get("kind") or "")
        if not kind:
            continue
        example: dict[str, Any] = {"kind": kind}
        targets = [str(x) for x in action.get("target_ids") or []]
        if kind in _SPEECH_ACTION_KINDS:
            example["text"] = "好"
        elif targets:
            example["target_id"] = targets[0]
        lines.append("- " + json.dumps(example, ensure_ascii=False, separators=(",", ":")))
        if targets:
            lines.append("  target_id 只能是：" + "、".join(targets))
        if kind in _SPEECH_ACTION_KINDS:
            limit = action.get("max_chars")
            req = "；需含中文" if action.get("require_chinese") else ""
            lines.append(f"  text 最多 {limit} 字{req}")
    return "\n".join(lines)


def normalize_decision(raw: dict[str, Any] | None, request: dict[str, Any]) -> dict[str, Any]:
    src = raw.get("decision", raw) if isinstance(raw, dict) else {}
    action = {
        "request_id": request["request_id"],
        "player_id": request["player_id"],
        "kind": str(src.get("kind", src.get("action", ""))),
    }
    target = src.get("target_id", src.get("targetId"))
    if target is not None and str(target) != "":
        action["target_id"] = str(target)
    if isinstance(src.get("text"), str):
        action["text"] = src["text"]
    return action


def decision_error(action: dict[str, Any], request: dict[str, Any]) -> str | None:
    allowed = next((a for a in request.get("allowed_actions", []) if a.get("kind") == action.get("kind")), None)
    if allowed is None:
        return "kind 必须来自 allowed_actions"
    targets = [str(x) for x in allowed.get("target_ids") or []]
    if targets and str(action.get("target_id")) not in set(targets):
        return "target_id 必须来自 target_ids"
    if not targets and action.get("target_id"):
        return "该行动不能包含 target_id"
    if action["kind"] in _SPEECH_ACTION_KINDS:
        text = str(action.get("text", "")).strip()
        if not text:
            return "发言不能为空"
        max_chars = int(allowed.get("max_chars") or 10**9)
        if len(text) > max_chars:
            return f"发言超过 max_chars={max_chars}"
        if allowed.get("require_chinese") and not _CHINESE_CHARACTER.search(text):
            return "发言必须包含中文"
    return None


class TaskAgent:
    """狼人角色任务求解器，保留原有构造接口。"""

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
        self._model_token_usage = {"successful_response_count": 0, "api_attempt_count": 0, "reported_usage_response_count": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        self._memory: dict[str, Any] = {"round": None, "phase": None, "alive_players": [], "dead_players": [], "sheriff_id": None, "recent_votes": [], "night_plan": None}
        self._ledger: dict[str, list[dict[str, Any]]] = {"claims": [], "votes": [], "deaths": [], "sheriff": []}
        self._recent_public_texts: list[str] = []
        self._recent_self_texts: list[str] = []

    async def observe(self, sync_packet: dict[str, Any]) -> None:
        summary = self._summarize_sync(sync_packet)
        if not summary:
            return
        for key in ("round", "phase", "sheriff_id"):
            if summary.get(key) is not None:
                self._memory[key] = summary[key]
        for key in ("alive_players", "dead_players", "recent_votes"):
            if summary.get(key):
                self._memory[key] = summary[key][-_MAX_HISTORY:]
        self._recent_public_texts.extend(summary.get("texts") or [])
        self._recent_public_texts = self._recent_public_texts[-_MAX_HISTORY:]
        self._update_ledger(summary)

    async def decide(self, turn_packet: dict[str, Any]) -> dict[str, Any]:
        if turn_packet["self"]["player_id"] != self.player_id:
            raise ValueError("行动包被发送给了错误玩家")
        private = turn_packet["private_information"]
        if str(private.get("role")) != self.profile.role:
            raise ValueError("Task-Agent 的角色档案与行动包身份不一致")

        memory = self._memory_snapshot()
        dialogue = self._extract_current_round_dialogue(turn_packet)
        prompt = self._build_compact_turn_prompt(turn_packet, memory, dialogue)
        strategy = prompt["decision_state"].get("wolf_strategy", {})

        # 夜间投票是高风险低文本动作：如果主刀合法，直接执行唯一主刀，防止模型把票分到备刀或狼队友。
        forced = self._forced_target_action(turn_packet["request"], strategy)
        if forced is not None and self._is_night(turn_packet) and strategy.get("recommended_target_id"):
            self._record_self_action(turn_packet, forced)
            return forced

        tool_context = turn_packet.get("tool_context") or {}
        current_dialogue_tool = tool_context.get("current_round_dialogue") or []

        def execute_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
            del arguments
            if name != CURRENT_ROUND_DIALOGUE_TOOL_NAME:
                return {"error": f"不支持的工具：{name}"}
            return {"round": self._safe_get(turn_packet.get("game"), "round"), "phase": self._safe_get(turn_packet.get("game"), "phase"), "dialogue": current_dialogue_tool}

        system = self._system_prompt(private)
        feedback = ""
        for attempt in range(self.max_decision_retries + 1):
            allow_tool = attempt == 0 and self.max_tool_calls_per_decision > 0
            content = {"instruction": self._turn_instruction(turn_packet["request"], feedback), "packet": prompt}
            try:
                raw = await self._complete_json(
                    system=system,
                    messages=[{"role": "user", "content": json.dumps(content, ensure_ascii=False, separators=(",", ":"))}],
                    tools=[CURRENT_ROUND_DIALOGUE_TOOL] if allow_tool else None,
                    tool_executor=execute_tool if allow_tool else None,
                    max_tool_calls=self.max_tool_calls_per_decision if allow_tool else 0,
                    max_tool_result_tokens=self.max_tool_result_tokens,
                    max_prompt_chars=self.max_prompt_chars,
                )
            except ModelClientError:
                if attempt >= self.max_decision_retries:
                    raise
                feedback = "上一轮模型调用失败，请只按契约输出合法 JSON。"
                continue
            self._record_token_usage(raw)
            action = normalize_decision(raw, turn_packet["request"])
            # 任何目标类动作都优先收敛到代码给出的合法推荐，夜间尤其禁止偏离主刀。
            repaired = self._repair_target_action(action, turn_packet["request"], strategy)
            if repaired is not None:
                action = repaired
            error = decision_error(action, turn_packet["request"])
            if error is None:
                self._record_self_action(turn_packet, action)
                return action
            if attempt >= self.max_decision_retries:
                raise RuleViolationError(f"模型返回非法行动：{error}", attempted_action=action, validation_error=error)
            feedback = error
        raise RuleViolationError("未产生合法行动")

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

    @staticmethod
    def _turn_instruction(request: Mapping[str, Any], feedback: str) -> str:
        return render_prompt("player_turn_instruction.txt", action_contract=render_action_contract(request), validation_feedback=(f"上一次输出未通过校验：{feedback}" if feedback else ""))

    async def _complete_json(self, **kwargs: Any) -> dict[str, Any]:
        if self.request_coordinator is not None:
            return await self.request_coordinator.complete_json(self.model_client, max_tokens=self.max_tokens, **kwargs)
        complete_json = self.model_client.complete_json
        if inspect.iscoroutinefunction(complete_json):
            return await complete_json(max_tokens=self.max_tokens, **kwargs)
        return await asyncio.to_thread(complete_json, max_tokens=self.max_tokens, **kwargs)

    def _record_token_usage(self, response: object) -> None:
        self._model_token_usage["successful_response_count"] += 1
        attempts = self._nonnegative_int(getattr(response, "api_attempts", 1), fallback=1) or 1
        self._model_token_usage["api_attempt_count"] += max(1, attempts)
        usage = getattr(response, "token_usage", None)
        if isinstance(usage, dict):
            self._model_token_usage["reported_usage_response_count"] += 1
            for key in ("input_tokens", "output_tokens", "total_tokens"):
                value = self._nonnegative_int(usage.get(key), fallback=None)
                if value is not None:
                    self._model_token_usage[key] += value

    def _memory_snapshot(self) -> dict[str, Any]:
        plan = self._memory.get("night_plan") if isinstance(self._memory.get("night_plan"), Mapping) else None
        if plan and plan.get("round") != self._memory.get("round"):
            plan = None
        return {"game": {k: self._memory.get(k) for k in ("round", "phase", "alive_players", "dead_players", "sheriff_id")}, "ledger": {k: v[-3:] for k, v in self._ledger.items()}, "recent_votes": self._memory.get("recent_votes", [])[-3:], "recent_public_texts": self._recent_public_texts[-3:], "recent_self_texts": self._recent_self_texts[-2:], "night_plan": plan}

    def _build_compact_turn_prompt(self, turn_packet: Mapping[str, Any], memory: Mapping[str, Any], dialogue: list[dict[str, Any]]) -> dict[str, Any]:
        digest = self._current_public_state_digest(turn_packet, memory)
        strategy = self._build_wolf_strategy(turn_packet, memory, dialogue, digest)
        request = turn_packet.get("request") or {}
        packet = {
            "game": self._pick(turn_packet.get("game"), ("round", "phase", "public_phase")),
            "request": self._pick(request, ("request_id", "phase", "channel", "allowed_actions")),
            "self": self._pick(turn_packet.get("self"), ("player_id", "seat", "alive", "status")),
            "private_information": self._pick(turn_packet.get("private_information"), ("role", "team", "wolf_ids", "teammates")),
            "state_digest": digest,
            "recent_dialogue": dialogue[-3:],
            "wolf_strategy": strategy,
        }
        if strategy.get("stage") == "night":
            packet["night_contract"] = "严禁刀存活狼队友；recommended_target_id 是唯一主刀，合法时必须投它，不得改投备刀。backup 仅在主刀不在 target_ids/已死时启用。"
        return {"decision_state": packet}

    def _build_wolf_strategy(self, turn_packet: Mapping[str, Any], memory: Mapping[str, Any], dialogue: list[dict[str, Any]], digest: Mapping[str, Any] | None = None) -> dict[str, Any]:
        request = turn_packet.get("request") or {}
        actions = [a for a in request.get("allowed_actions", []) if isinstance(a, Mapping)]
        target_actions = [a for a in actions if a.get("target_ids")]
        digest = digest or self._current_public_state_digest(turn_packet, memory)
        if self._is_night(turn_packet) and target_actions:
            ranked = self._rank_targets(turn_packet, memory, target_actions, dialogue, digest)
            primary = ranked[0] if ranked else None
            backup = ranked[1] if len(ranked) > 1 else None
            plan = {
                "stage": "night",
                "discipline": "主刀合法则唯一执行主刀；禁止投存活狼队友；禁止因备刀分票。",
                "recommended_kind": target_actions[0].get("kind"),
                "recommended_target_id": primary.get("player_id") if primary else None,
                "primary_target": primary,
                "backup_target": backup,
                "avoid_target_ids": self._alive_wolf_ids(turn_packet.get("private_information"), digest),
                "candidate_order": ranked[:3],
            }
            self._memory["night_plan"] = {"round": digest.get("round"), "target_id": plan["recommended_target_id"]}
            return plan
        if target_actions:
            return self._day_vote_strategy(turn_packet, memory, target_actions, dialogue, digest)
        if any(a.get("kind") in _SPEECH_ACTION_KINDS for a in actions):
            return self._speech_strategy(turn_packet, memory, dialogue, digest)
        return {"stage": "other", "objective": "只选择当前 allowed_actions 中的合法动作"}

    def _rank_targets(self, turn_packet: Mapping[str, Any], memory: Mapping[str, Any], target_actions: list[Mapping[str, Any]], dialogue: list[dict[str, Any]], digest: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
        digest = digest or self._current_public_state_digest(turn_packet, memory)
        records = self._extract_player_records(turn_packet.get("public_state"), turn_packet.get("game"), memory)
        candidates = self._collect_target_ids(target_actions)
        alive = set(digest.get("alive_player_ids") or [])
        dead = set(digest.get("dead_player_ids") or [])
        alive_known = bool(alive)
        alive_wolves = set(self._alive_wolf_ids(turn_packet.get("private_information"), digest))
        signal = self._parse_wolf_discussion(dialogue, [pid for pid in candidates if pid not in alive_wolves])
        locked = signal.get("consensus_target") or signal.get("primary_target") or (self._memory.get("night_plan") or {}).get("target_id")
        scored: list[dict[str, Any]] = []
        for pid in candidates:
            if pid == self.player_id or pid in dead or (alive_known and pid not in alive) or pid in alive_wolves:
                continue
            base, reason = self._score_target(pid, records.get(pid, {}), memory, alive_wolves=alive_wolves)
            if pid == locked:
                base += 100
                reason = "狼队锁定主刀," + reason
            elif pid == signal.get("backup_target"):
                base += 5
                reason = "备刀仅兜底," + reason
            scored.append({"player_id": pid, "score": base, "reason": reason, "seat": self._seat_sort_value(records.get(pid, {}), pid)})
        scored.sort(key=lambda x: (-int(x["score"]), x["seat"], x["player_id"]))
        # 主刀合法时降维：把它置顶且 prompt 只称它为 recommended，备刀不参与投票。
        return [{"player_id": x["player_id"], "reason": x["reason"]} for x in scored[:3]]

    def _score_target(self, target_id: str, record: Mapping[str, Any], memory: Mapping[str, Any], *, alive_wolves: set[str] | None = None) -> tuple[int, str]:
        if target_id == self.player_id:
            return -1000, "禁止自刀"
        if alive_wolves and target_id in alive_wolves:
            return -1000, "禁止刀存活狼队友"
        score = 0
        reasons: list[str] = []
        if self._is_alive_record(record) is False:
            return -500, "已死亡"
        if record.get("is_sheriff") or record.get("sheriff") or target_id == (memory.get("game") or {}).get("sheriff_id"):
            score += 8; reasons.append("警徽/带队位")
        claim = self._public_claim_text(record)
        if claim:
            score += 6; reasons.append("公开身份链")
        votes = self._nonnegative_int(record.get("vote_count", record.get("votes")), fallback=0) or 0
        if votes:
            score += min(5, votes); reasons.append("票型焦点")
        mentions = self._recent_mention_count(target_id)
        if mentions:
            score += min(4, mentions); reasons.append("近期被提及")
        ledger = memory.get("ledger") or memory.get("public_ledger") or {}
        if any(e.get("player_id") == target_id for e in ledger.get("claims", [])[-3:]):
            score += 3; reasons.append("近期报身份")
        if any(e.get("to") == target_id or e.get("from") == target_id for e in ledger.get("sheriff", [])[-2:]):
            score += 3; reasons.append("警徽链")
        return score, ",".join(reasons[:3]) or "默认非狼合法目标"

    def _parse_wolf_discussion(self, dialogue: list[dict[str, Any]], candidate_ids: list[str]) -> dict[str, Any]:
        ids = [str(x) for x in candidate_ids]
        mentions: Counter[str] = Counter()
        primary: str | None = None
        backup: str | None = None
        consensus: str | None = None
        lines: list[str] = []
        for item in dialogue[-6:]:
            text = str(item.get("text") or item.get("content") or item.get("message") or "")
            hits = [pid for pid in ids if pid in text]
            if not hits:
                continue
            lines.append(self._clip_text(text, 50))
            for pid in hits:
                mentions[pid] += 1
            if any(w in text for w in ("主刀", "先刀", "第一刀", "主目标", "优先刀", "就刀")):
                primary = hits[0]
            if any(w in text for w in ("备刀", "次刀", "备选", "兜底", "替补")):
                backup = hits[0]
            if any(w in text for w in ("定", "锁", "一致", "统一", "拍板", "最后")):
                consensus = hits[0]
        if primary is None and mentions:
            primary = mentions.most_common(1)[0][0]
        if consensus is None:
            consensus = primary
        if backup is None:
            backup = next((pid for pid, _ in mentions.most_common() if pid != primary), None)
        order = [pid for pid, _ in mentions.most_common()]
        return {"candidate_targets": order[:5], "mention_counts": dict(mentions), "primary_target": primary, "backup_target": backup, "consensus_target": consensus, "discussion_lines": lines[-3:], "has_disagreement": len(mentions) > 1}

    def _day_vote_strategy(self, turn_packet: Mapping[str, Any], memory: Mapping[str, Any], target_actions: list[Mapping[str, Any]], dialogue: list[dict[str, Any]], digest: Mapping[str, Any]) -> dict[str, Any]:
        del dialogue
        action = target_actions[0]
        candidates = self._collect_target_ids([action])
        alive = set(digest.get("alive_player_ids") or [])
        dead = set(digest.get("dead_player_ids") or [])
        wolves = set(self._alive_wolf_ids(turn_packet.get("private_information"), digest))
        records = self._extract_player_records(turn_packet.get("public_state"), turn_packet.get("game"), memory)
        scored = []
        for pid in candidates:
            if pid == self.player_id or pid in dead or (alive and pid not in alive):
                continue
            score, reason = self._score_target(pid, records.get(pid, {}), memory, alive_wolves=set())
            if pid in wolves:
                score -= 6; reason = "队友慎投," + reason
            scored.append((score, self._seat_sort_value(records.get(pid, {}), pid), pid, reason))
        scored.sort(key=lambda x: (-x[0], x[1], x[2]))
        non_wolf = next((x for x in scored if x[2] not in wolves), None)
        chosen = non_wolf or (scored[0] if scored else None)
        return {"stage": "day_vote", "recommended_kind": action.get("kind"), "recommended_target_id": chosen[2] if chosen else None, "avoid_target_ids": sorted(wolves), "reason": chosen[3] if chosen else "无合法目标"}

    def _speech_strategy(self, turn_packet: Mapping[str, Any], memory: Mapping[str, Any], dialogue: list[dict[str, Any]], digest: Mapping[str, Any]) -> dict[str, Any]:
        records = self._extract_player_records(turn_packet.get("public_state"), turn_packet.get("game"), memory)
        wolves = set(self._alive_wolf_ids(turn_packet.get("private_information"), digest))
        dead = set(digest.get("dead_player_ids") or [])
        scored = []
        for pid, record in records.items():
            if pid == self.player_id or pid in dead or pid in wolves:
                continue
            score, reason = self._score_target(pid, record, memory, alive_wolves=set())
            scored.append((score, self._seat_sort_value(record, pid), pid, reason))
        scored.sort(key=lambda x: (-x[0], x[1], x[2]))
        target = scored[0] if scored else None
        last_line = next((f"{d.get('speaker')}:{d.get('text')}" for d in reversed(dialogue[-3:]) if d.get("text")), "当前发言/票型")
        return {"stage": "day", "objective": "公开发言要像好人推理：绑定对象、证据、下一步动作，避免暴露狼队。", "focus_player": target[2] if target else None, "evidence": target[3] if target else self._clip_text(last_line, 60), "avoid_target_ids": sorted(wolves), "anti_repeat": self._recent_self_texts[-1:]}

    def _forced_target_action(self, request: Mapping[str, Any], strategy: Mapping[str, Any]) -> dict[str, Any] | None:
        target = strategy.get("recommended_target_id")
        kind = strategy.get("recommended_kind")
        if not target or not kind:
            return None
        for action in request.get("allowed_actions", []) or []:
            if not isinstance(action, Mapping) or action.get("kind") != kind:
                continue
            if str(target) in {str(x) for x in action.get("target_ids") or []}:
                return {"request_id": request["request_id"], "player_id": request["player_id"], "kind": str(kind), "target_id": str(target)}
        return None

    def _repair_target_action(self, action: Mapping[str, Any], request: Mapping[str, Any], strategy: Mapping[str, Any]) -> dict[str, Any] | None:
        if action.get("kind") in _SPEECH_ACTION_KINDS:
            return None
        forced = self._forced_target_action(request, strategy)
        if forced is not None:
            return forced
        return None

    def _record_self_action(self, turn_packet: Mapping[str, Any], action: Mapping[str, Any]) -> None:
        item = {"round": self._safe_get(turn_packet.get("game"), "round"), "phase": self._safe_get(turn_packet.get("request"), "phase"), "kind": action.get("kind")}
        if action.get("target_id") is not None:
            item["target_id"] = str(action["target_id"])
        if isinstance(action.get("text"), str) and action["text"].strip():
            item["text"] = self._clip_text(action["text"], _SNIP)
            self._recent_self_texts.append(item["text"])
            self._recent_self_texts = self._recent_self_texts[-_MAX_HISTORY:]

    def _summarize_sync(self, packet: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(packet, Mapping):
            return {}
        game = packet.get("game") if isinstance(packet.get("game"), Mapping) else {}
        public = packet.get("public_state") if isinstance(packet.get("public_state"), Mapping) else {}
        digest = self._current_alive_dead(public, game, self._memory_snapshot())
        summary = {"round": self._safe_get(game, "round") or self._safe_get(public, "round"), "phase": self._safe_get(game, "public_phase") or self._safe_get(game, "phase") or self._safe_get(public, "phase"), "alive_players": digest["alive_player_ids"], "dead_players": digest["dead_player_ids"]}
        sheriff = self._current_sheriff_id(public, game, self._memory_snapshot())
        if sheriff is not None:
            summary["sheriff_id"] = sheriff
        votes = self._summarize_votes(public.get("votes") or game.get("votes") or public.get("vote_history") or game.get("vote_history"))
        if votes:
            summary["recent_votes"] = votes
        texts = self._extract_text_snippets(packet)
        if texts:
            summary["texts"] = texts[-_MAX_HISTORY:]
        return summary

    def _update_ledger(self, summary: Mapping[str, Any]) -> None:
        r = summary.get("round")
        for pid in summary.get("dead_players") or []:
            self._append_unique("deaths", {"round": r, "player_id": str(pid)})
        for vote in summary.get("recent_votes") or []:
            self._append_unique("votes", {"round": r, "vote": str(vote)})
        if summary.get("sheriff_id"):
            self._append_unique("sheriff", {"round": r, "to": str(summary["sheriff_id"])})

    def _append_unique(self, bucket: str, item: dict[str, Any]) -> None:
        values = self._ledger[bucket]
        if item not in values:
            values.append(item)
            del values[:-_MAX_HISTORY]

    def _current_public_state_digest(self, turn_packet: Mapping[str, Any], memory: Mapping[str, Any]) -> dict[str, Any]:
        public = turn_packet.get("public_state") if isinstance(turn_packet.get("public_state"), Mapping) else {}
        game = turn_packet.get("game") if isinstance(turn_packet.get("game"), Mapping) else {}
        status = self._current_alive_dead(public, game, memory)
        status.update({"round": self._safe_get(game, "round") or self._safe_get(public, "round"), "phase": self._safe_get(game, "public_phase") or self._safe_get(game, "phase") or self._safe_get(public, "phase"), "current_sheriff_id": self._current_sheriff_id(public, game, memory)})
        status["sheriff_id"] = status["current_sheriff_id"]
        return status

    def _current_alive_dead(self, public: Any, game: Any, memory: Mapping[str, Any] | None = None) -> dict[str, Any]:
        alive: list[str] = []
        dead: list[str] = []
        known: dict[str, bool] = {}

        def add(raw: Any, is_alive: bool) -> None:
            ids = self._ids_from(raw)
            for pid in ids:
                known[pid] = is_alive
                bucket = alive if is_alive else dead
                if pid not in bucket:
                    bucket.append(pid)

        def inspect(raw: Any) -> None:
            if isinstance(raw, Mapping):
                iterable = raw.items()
            elif isinstance(raw, list):
                iterable = [(self._first_value(x, ("player_id", "id", "seat", "seat_id")) if isinstance(x, Mapping) else x, x) for x in raw]
            else:
                iterable = []
            for pid, info in iterable:
                if pid is None:
                    continue
                state = self._is_alive_record(info if isinstance(info, Mapping) else {})
                if state is not None:
                    add([pid], state)

        for src in (public, game):
            if not isinstance(src, Mapping):
                continue
            inspect(src.get("players"))
            add(src.get("alive_players") or src.get("living_players") or src.get("survivors"), True)
            add(src.get("dead_players") or src.get("eliminated_players") or src.get("casualties"), False)
        if memory:
            mg = memory.get("game") or memory
            if not alive:
                add(mg.get("alive_players"), True)
            if not dead:
                add(mg.get("dead_players"), False)
        dead_set = set(dead)
        alive = [pid for pid in alive if pid not in dead_set]
        return {"alive_player_ids": alive, "dead_player_ids": dead, "players_alive_map": {pid: v for pid, v in sorted(known.items())}}

    def _extract_player_records(self, public: Any, game: Any, memory: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
        records: dict[str, dict[str, Any]] = {}

        def rec(pid: Any) -> dict[str, Any]:
            text = str(pid)
            return records.setdefault(text, {"player_id": text})

        def ingest(raw: Any, alive: bool | None = None) -> None:
            if isinstance(raw, Mapping):
                items = raw.items()
            elif isinstance(raw, list):
                items = []
                for x in raw:
                    pid = self._first_value(x, ("player_id", "id", "seat", "seat_id")) if isinstance(x, Mapping) else x
                    items.append((pid, x))
            else:
                return
            for pid, info in items:
                if pid is None:
                    continue
                r = rec(pid)
                if isinstance(info, Mapping):
                    for k in ("seat", "seat_id", "alive", "status", "claim", "claimed_role", "public_role", "vote_count", "votes", "speak_count", "speech_count", "sheriff", "is_sheriff"):
                        if k in info:
                            r[k] = self._clip_value(info[k])
                if alive is not None:
                    r["alive"] = alive

        for src in (public, game):
            if not isinstance(src, Mapping):
                continue
            ingest(src.get("players"))
            ingest(src.get("alive_players") or src.get("living_players") or src.get("survivors"), True)
            ingest(src.get("dead_players") or src.get("eliminated_players") or src.get("casualties"), False)
            sheriff = self._current_sheriff_id(src, {}, None)
            if sheriff is not None:
                rec(sheriff)["is_sheriff"] = True
            votes = src.get("votes")
            if isinstance(votes, Mapping):
                for voter, target in votes.items():
                    tid = self._first_value(target, ("player_id", "id", "target_id")) if isinstance(target, Mapping) else target
                    rec(voter)["vote_target"] = tid
                    if tid is not None:
                        tr = rec(tid)
                        tr["vote_count"] = (self._nonnegative_int(tr.get("vote_count"), fallback=0) or 0) + 1
        mg = memory.get("game") or {}
        for pid in mg.get("alive_players") or []:
            rec(pid).setdefault("alive", True)
        for pid in mg.get("dead_players") or []:
            rec(pid)["alive"] = False
        if mg.get("sheriff_id"):
            rec(mg["sheriff_id"])["is_sheriff"] = True
        return records

    def _wolf_team_ids(self, private: Any) -> list[str]:
        if not isinstance(private, Mapping):
            return []
        found: list[str] = []
        for key in ("wolf_ids", "wolves", "teammates", "known_wolves", "team_members"):
            for pid in self._ids_from(private.get(key)):
                if pid not in found:
                    found.append(pid)
        return found

    def _alive_wolf_ids(self, private: Any, digest: Mapping[str, Any]) -> list[str]:
        wolves = self._wolf_team_ids(private)
        alive = {str(pid) for pid in digest.get("alive_player_ids") or []}
        dead = {str(pid) for pid in digest.get("dead_player_ids") or []}
        if alive:
            return [pid for pid in wolves if pid in alive and pid not in dead]
        return [pid for pid in wolves if pid not in dead]

    def _extract_current_round_dialogue(self, turn_packet: Mapping[str, Any]) -> list[dict[str, Any]]:
        return self._summarize_dialogue((turn_packet.get("tool_context") or {}).get("current_round_dialogue") or [])

    def _summarize_dialogue(self, raw: Any) -> list[dict[str, Any]]:
        items = raw if isinstance(raw, list) else ([raw] if raw else [])
        out: list[dict[str, Any]] = []
        for item in items[-_MAX_HISTORY:]:
            if isinstance(item, Mapping):
                d: dict[str, Any] = {}
                speaker = self._first_value(item, ("speaker", "speaker_id", "player_id", "from", "by"))
                text = self._first_value(item, ("text", "content", "utterance", "speech", "message"))
                if speaker is not None:
                    d["speaker"] = str(speaker)
                if text is not None:
                    d["text"] = self._clip_text(str(text), _SNIP)
                if d:
                    out.append(d)
            elif item is not None:
                out.append({"text": self._clip_text(str(item), _SNIP)})
        return out

    def _extract_text_snippets(self, value: Any) -> list[str]:
        result: list[str] = []
        if isinstance(value, Mapping):
            for k in ("text", "content", "message", "utterance"):
                if isinstance(value.get(k), str):
                    result.append(self._clip_text(value[k], _SNIP))
            for k in ("dialogue", "public_dialogue", "events", "history"):
                result.extend(self._extract_text_snippets(value.get(k)))
        elif isinstance(value, list):
            for x in value[-_MAX_HISTORY:]:
                result.extend(self._extract_text_snippets(x))
        elif isinstance(value, str):
            result.append(self._clip_text(value, _SNIP))
        return result[-_MAX_HISTORY:]

    def _summarize_votes(self, raw: Any) -> list[str]:
        out: list[str] = []
        if isinstance(raw, Mapping):
            for voter, target in raw.items():
                tid = self._first_value(target, ("player_id", "id", "target_id")) if isinstance(target, Mapping) else target
                out.append(f"{voter}->{tid}")
        elif isinstance(raw, list):
            for item in raw[-_MAX_HISTORY:]:
                if isinstance(item, Mapping):
                    voter = self._first_value(item, ("voter", "from", "player_id"))
                    target = self._first_value(item, ("target", "target_id", "to"))
                    out.append(f"{voter}->{target}")
                else:
                    out.append(self._clip_text(str(item), 40))
        return out[-_MAX_HISTORY:]

    def _current_sheriff_id(self, public: Any, game: Any, memory: Mapping[str, Any] | None = None) -> str | None:
        for src in (public, game):
            if not isinstance(src, Mapping):
                continue
            value = self._first_value(src, ("sheriff_id", "chairman_id"))
            if value is None:
                for key in ("sheriff", "chairman"):
                    if isinstance(src.get(key), Mapping):
                        value = self._first_value(src[key], ("player_id", "id", "seat", "seat_id"))
                        if value is not None:
                            break
            if value is not None:
                return str(value)
        if memory:
            value = (memory.get("game") or {}).get("sheriff_id")
            if value is not None:
                return str(value)
        return None

    def _public_claim_text(self, record: Mapping[str, Any]) -> str | None:
        for key in ("public_role", "claimed_role", "claim", "role"):
            value = record.get(key)
            if value is not None:
                return self._clip_text(str(value), 32)
        return None

    def _recent_mention_count(self, target_id: str) -> int:
        text = " ".join(self._recent_public_texts[-_MAX_HISTORY:] + self._recent_self_texts[-3:])
        return text.count(str(target_id)) if target_id and text else 0

    def _is_night(self, turn_packet: Mapping[str, Any]) -> bool:
        req = turn_packet.get("request") or {}
        game = turn_packet.get("game") or {}
        text = " ".join(str(x) for x in (req.get("phase"), req.get("channel"), game.get("phase"), game.get("public_phase"))).lower()
        return any(token in text for token in ("night", "wolf", "夜"))

    @staticmethod
    def _ids_from(raw: Any) -> list[str]:
        if raw is None:
            return []
        if isinstance(raw, Mapping):
            if any(k in raw for k in ("player_id", "id", "seat", "seat_id")):
                for k in ("player_id", "id", "seat", "seat_id"):
                    if raw.get(k) is not None:
                        return [str(raw[k])]
            return [str(k) for k in raw.keys() if k is not None]
        if isinstance(raw, (list, tuple, set)):
            out: list[str] = []
            for x in raw:
                if isinstance(x, Mapping):
                    for k in ("player_id", "id", "seat", "seat_id"):
                        if x.get(k) is not None:
                            out.append(str(x[k])); break
                elif x is not None:
                    out.append(str(x))
            return out
        return [str(raw)]

    @staticmethod
    def _pick(value: Any, keys: tuple[str, ...]) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            return {}
        return {k: value[k] for k in keys if k in value}

    def _clip_value(self, value: Any) -> Any:
        if isinstance(value, str):
            return self._clip_text(value, 80)
        if isinstance(value, (int, float, bool)) or value is None:
            return value
        if isinstance(value, Mapping):
            return {str(k): self._clip_value(v) for k, v in list(value.items())[:4]}
        if isinstance(value, list):
            return [self._clip_value(x) for x in value[:4]]
        return self._clip_text(str(value), 80)

    @staticmethod
    def _clip_text(text: str, limit: int) -> str:
        text = str(text)
        return text if len(text) <= limit else text[: max(0, limit - 1)] + "…"

    @staticmethod
    def _safe_get(value: Any, key: str) -> Any:
        return value.get(key) if isinstance(value, Mapping) else None

    @staticmethod
    def _first_value(value: Any, keys: tuple[str, ...]) -> Any:
        if not isinstance(value, Mapping):
            return None
        for key in keys:
            if value.get(key) is not None:
                return value[key]
        return None

    @staticmethod
    def _nonnegative_int(value: object, *, fallback: int | None) -> int | None:
        if isinstance(value, bool):
            return fallback
        try:
            n = int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return fallback
        return n if n >= 0 else fallback

    @staticmethod
    def _is_alive_record(record: Mapping[str, Any]) -> bool | None:
        if not record:
            return None
        if record.get("alive") is True or record.get("is_alive") is True:
            return True
        if record.get("alive") is False or record.get("is_alive") is False or record.get("is_dead") is True:
            return False
        status = str(record.get("status") or "").lower()
        if status in {"dead", "eliminated", "out"}:
            return False
        if status in {"alive", "living"}:
            return True
        return None

    @staticmethod
    def _seat_sort_value(record: Mapping[str, Any], target_id: str) -> tuple[int, str]:
        seat = record.get("seat") or record.get("seat_id") if isinstance(record, Mapping) else None
        for value in (seat, target_id):
            m = re.search(r"\d+", str(value)) if value is not None else None
            if m:
                return int(m.group()), str(target_id)
        return 10**9, str(target_id)

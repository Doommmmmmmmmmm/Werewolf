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
        lines.append("投票时优先保护系统事实支持的金水、持续报验且查杀被票型或发言支持的预言家、已公开强神；警徽只是投票权，不是身份认证。")
        lines.append("持徽者/接徽者必须重新审计查验链、翻牌和最近票型；无持续查验链时不得自动保护，至少与另一候选并列比较。")
        lines.append("优先找未被公开链覆盖且存在票型异常、事实错误、被查杀或爆狼线索指向的人。")
    if has_speak:
        lines.append("发言至少点出一个关注对象和一个暂不投对象，并说明依据哪条公开信息。")
    if has_last_words:
        lines.append("遗言只留公开事实、票型、查验链和怀疑对象，不把推测说成系统事实。")
    if has_sheriff:
        lines.append("警长/警徽相关回合优先看报验链是否清晰一致；接徽时只继承投票权和公开遗产，不继承身份可信度。")
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
        self, *, player_id: str, profile: RoleProfile, model_client: Any,
        persona: str = "", request_coordinator: ModelRequestCoordinator | None = None,
        max_tokens: int = 900, max_decision_retries: int = DEFAULT_MAX_DECISION_RETRIES,
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
            "successful_response_count": 0, "api_attempt_count": 0,
            "reported_usage_response_count": 0, "input_tokens": 0,
            "output_tokens": 0, "total_tokens": 0,
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
            "game": turn_packet["game"], "public_rules": turn_packet["public_rules"],
            "self": turn_packet["self"], "private_information": private,
            "public_state": turn_packet["public_state"], "request": turn_packet["request"],
            "history_policy": {"default_context": "current_state_only",
                "current_round_dialogue_tool": CURRENT_ROUND_DIALOGUE_TOOL_NAME,
                "max_tool_calls": self.max_tool_calls_per_decision,
                "max_tool_result_tokens": self.max_tool_result_tokens},
        }
        tool_context = turn_packet.get("tool_context") or {}
        current_dialogue = tool_context.get("current_round_dialogue") or []

        def execute_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
            del arguments
            if name != CURRENT_ROUND_DIALOGUE_TOOL_NAME:
                return {"error": f"不支持的工具：{name}"}
            return {"round": turn_packet["game"].get("round"),
                    "phase": turn_packet["game"].get("public_phase", turn_packet["game"].get("phase")),
                    "dialogue": current_dialogue}

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
        raise RuleViolationError("模型没有产生行动")

    def agent_manifest(self) -> dict[str, Any]:
        return {"agent_type": "task_agent", "player_id": self.player_id,
                "role": self.profile.role, "conversation_mode": "new_session_per_decision",
                "tool_name": CURRENT_ROUND_DIALOGUE_TOOL_NAME,
                "max_tool_calls_per_decision": self.max_tool_calls_per_decision,
                "max_tool_result_tokens": self.max_tool_result_tokens,
                "max_prompt_chars": self.max_prompt_chars,
                "max_decision_retries": self.max_decision_retries,
                "task_fingerprint": hashlib.sha256(self.profile.task.encode("utf-8")).hexdigest()}

    def model_token_usage_snapshot(self) -> dict[str, int]:
        return dict(self._model_token_usage)

    def _system_prompt(self, private: Mapping[str, Any]) -> str:
        return render_prompt("player_system.txt", player_id=self.player_id, role=private["role"],
            team=private["team"], persona=self.persona, role_base=self.profile.base,
            role_task=self.profile.task, tool_name=CURRENT_ROUND_DIALOGUE_TOOL_NAME,
            max_tool_calls=self.max_tool_calls_per_decision,
            max_tool_result_tokens=self.max_tool_result_tokens, max_prompt_chars=self.max_prompt_chars)

    def _turn_instruction(self, turn_packet: Mapping[str, Any], feedback: str) -> str:
        request = turn_packet["request"]
        summary = self._render_turn_summary(turn_packet, request)
        checklist = render_villager_decision_checklist(request)
        feedback_text = f"上一次输出未通过校验：{feedback}" if feedback else ""
        return render_prompt("player_turn_instruction.txt",
            action_contract=render_action_contract(request),
            validation_feedback="\n".join(x for x in (summary, feedback_text, checklist) if x))

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
        values = {k: self._nonnegative_int(usage.get(k), fallback=None)
                  for k in ("input_tokens", "output_tokens", "total_tokens")}
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
            result = int(value)
        except (TypeError, ValueError):
            return fallback
        return result if result >= 0 else fallback

    def _reset_public_ledger(self, game_id: str | None) -> None:
        self._public_ledger_game_id = game_id
        self._public_ledger = {
            "game_id": game_id, "round": None, "phase": "",
            "known_player_ids": set(), "alive_ids": set(), "dead_ids": set(),
            "public_reveals": [], "claim_records": [], "vote_records": [],
            "badge_records": [], "death_records": [], "recent_speeches": [],
            "recent_events": [],
        }

    # ---- Public-fact layer -------------------------------------------------
    # This parser deliberately does not turn an arbitrary target_id into a vote.
    def _ingest_public_packet(self, payload: Any) -> None:
        facts = self._parse_public_snapshot(payload)
        if facts.get("game_id") is not None and facts["game_id"] != self._public_ledger_game_id:
            self._reset_public_ledger(facts["game_id"])
        for key in ("round", "phase"):
            if facts.get(key) not in (None, ""):
                self._public_ledger[key] = facts[key]
        known = facts["known_player_ids"]
        self._public_ledger["known_player_ids"].update(known)
        if facts["authoritative_players"]:
            self._public_ledger["alive_ids"] = set(facts["alive_ids"])
            self._public_ledger["dead_ids"] = set(facts["dead_ids"])
        else:
            if facts["explicit_alive"]:
                self._public_ledger["alive_ids"] = set(facts["alive_ids"])
            self._public_ledger["dead_ids"].update(facts["dead_ids"])
            self._public_ledger["alive_ids"].difference_update(self._public_ledger["dead_ids"])
        for key in ("public_reveals", "claim_records", "vote_records", "badge_records",
                    "death_records", "recent_speeches", "recent_events"):
            for record in facts[key]:
                self._append_record(self._public_ledger[key], record)
        # A death event is authoritative even when the surrounding snapshot is sparse.
        self._public_ledger["dead_ids"].update(
            r["player_id"] for r in facts["death_records"] if r.get("player_id")
        )
        self._public_ledger["alive_ids"].difference_update(self._public_ledger["dead_ids"])

    def _parse_public_snapshot(self, payload: Any) -> dict[str, Any]:
        facts: dict[str, Any] = {"game_id": None, "round": None, "phase": "",
            "known_player_ids": set(), "alive_ids": set(), "dead_ids": set(),
            "explicit_alive": False, "authoritative_players": False,
            "public_reveals": [], "claim_records": [], "vote_records": [],
            "badge_records": [], "death_records": [], "recent_speeches": [], "recent_events": []}
        if not isinstance(payload, (Mapping, list)):
            return facts
        if isinstance(payload, Mapping):
            facts["game_id"] = self._extract_game_id(payload)
        def visit(obj: Any, inherited_round: Any = None, inherited_phase: str = "", context: str = "") -> None:
            if isinstance(obj, list):
                for item in obj[:30]:
                    visit(item, inherited_round, inherited_phase, context)
                return
            if not isinstance(obj, Mapping):
                return
            m = {str(k).lower(): v for k, v in obj.items()}
            current_round = self._first_scalar(m, ("round", "day", "turn")) or inherited_round
            current_phase = self._first_scalar(m, ("phase", "channel", "stage")) or inherited_phase
            event = self._first_scalar(m, ("event", "event_type", "type", "kind", "action"))
            # A collection name is useful context, but never overrides an explicit
            # event/type.  This keeps e.g. a badge transfer out of vote_records.
            if not event and context:
                event = context
            event_l = event.lower()
            if current_round is not None and facts["round"] is None:
                facts["round"] = current_round
            if current_phase and not facts["phase"]:
                facts["phase"] = current_phase
            # players[].alive is stronger than appearances or inferred collections.
            players = m.get("players")
            if isinstance(m.get("player_id"), (str, int)) and isinstance(m.get("alive"), bool):
                pid = self._first_id(m, ("player_id",))
                if pid:
                    facts["known_player_ids"].add(pid)
                    facts["authoritative_players"] = True
                    (facts["alive_ids"] if m["alive"] else facts["dead_ids"]).add(pid)
            if isinstance(players, list):
                for player in players[:30]:
                    if not isinstance(player, Mapping):
                        continue
                    pm = {str(k).lower(): v for k, v in player.items()}
                    pid = self._first_id(pm, ("player_id", "id", "uid"))
                    if not pid:
                        continue
                    facts["known_player_ids"].add(pid)
                    if isinstance(pm.get("alive"), bool):
                        facts["authoritative_players"] = True
                        (facts["alive_ids"] if pm["alive"] else facts["dead_ids"]).add(pid)
                    elif isinstance(pm.get("is_alive"), bool):
                        facts["authoritative_players"] = True
                        (facts["alive_ids"] if pm["is_alive"] else facts["dead_ids"]).add(pid)
            for key in self._ALIVE_KEYS():
                if key in m:
                    ids = self._as_id_set(m[key])
                    if ids:
                        facts["explicit_alive"] = True
                        facts["alive_ids"].update(ids); facts["known_player_ids"].update(ids)
            for key in self._DEAD_KEYS():
                if key in m:
                    ids = self._as_id_set(m[key])
                    facts["dead_ids"].update(ids); facts["known_player_ids"].update(ids)
                    # Explicit dead/death collections are system facts even when
                    # they are not wrapped in a typed event object.
                    if key in {"death_order", "deaths", "dead_player_ids", "eliminated", "casualties"}:
                        for pid in ids:
                            facts["death_records"].append(
                                self._record("death", current_round, current_phase, pid, None, m, "system_public")
                            )
            for key in ("publicly_revealed_roles", "revealed_roles", "flipped_roles"):
                value = m.get(key)
                if isinstance(value, Mapping):
                    for pid, role in list(value.items())[:20]:
                        if self._is_scalar_id(pid):
                            facts["public_reveals"].append({"kind": "reveal", "round": str(current_round or "?"), "phase": current_phase, "player_id": str(pid), "target_id": "?", "role": self._trim_text(str(role), 40), "source": "system_public"})
                            facts["known_player_ids"].add(str(pid))
                elif isinstance(value, list):
                    for item in value[:20]:
                        if isinstance(item, Mapping):
                            im = {str(k).lower(): v for k, v in item.items()}
                            pid = self._first_id(im, ("player_id", "id"))
                            role = self._first_scalar(im, ("role", "revealed_role"))
                            if pid:
                                facts["public_reveals"].append({"kind": "reveal", "round": str(current_round or "?"), "phase": current_phase, "player_id": pid, "target_id": "?", "role": role, "source": "system_public"})
                                facts["known_player_ids"].add(pid)
            votes_value = m.get("votes")
            if isinstance(votes_value, Mapping):
                for voter, target in list(votes_value.items())[:30]:
                    if self._is_scalar_id(voter) and self._is_scalar_id(target):
                        facts["vote_records"].append({"kind": "vote", "round": str(current_round or "?"), "phase": current_phase, "player_id": str(voter), "target_id": str(target), "source": "system_public"})
                        facts["known_player_ids"].update({str(voter), str(target)})
            record = self._record_from_mapping(m, event, current_round, current_phase)
            if record:
                kind, value = record
                facts[kind].append(value)
                facts["recent_events"].append(value)
                for field in ("player_id", "actor", "target_id", "from_player_id", "to_player_id"):
                    if value.get(field) and self._looks_like_player_id(value[field]):
                        facts["known_player_ids"].add(value[field])
            for key, value in obj.items():
                if isinstance(value, (Mapping, list)) and value is not players:
                    visit(value, current_round, current_phase, str(key).lower())
        visit(payload)
        facts["known_player_ids"].discard(self.player_id)
        for key in ("public_reveals", "claim_records", "vote_records", "badge_records",
                    "death_records", "recent_speeches", "recent_events"):
            facts[key] = self._dedupe_records(facts[key])
        return facts

    def _record_from_mapping(self, m: Mapping[str, Any], event: str, round_value: Any, phase: str) -> tuple[str, dict[str, Any]] | None:
        e = event.lower()
        def has(*keys: str) -> bool: return any(k in m for k in keys)
        # Explicit event names win.  Field heuristics are deliberately narrow:
        # target_id alone is not evidence of a ballot or a claim.
        death = any(x in e for x in ("death", "dead", "kill", "eliminat", "out", "死亡", "出局", "淘汰")) or has("dead_player_id", "dead_id", "victim_id")
        badge = any(x in e for x in ("sheriff", "badge", "警徽", "警长")) or has("badge_from", "badge_to", "badge_holder_id", "badge_holder", "current_sheriff", "sheriff_id", "sheriff")
        vote = "vote" in e or "ballot" in e or has("voter_id", "voter", "vote_target", "ballot")
        reveal = any(x in e for x in ("reveal", "flip", "翻牌", "公开身份")) or has("publicly_revealed_role", "revealed_role", "role_reveal")
        speech = "speak" in e or "speech" in e or "发言" in e or has("speaker_id", "speech", "text", "content", "message")
        claim = "claim" in e or "自称" in e or has("claimer_id", "claim", "role_claim", "claimed_role", "check_result", "查验结果")
        if death:
            pid = self._first_id(m, ("dead_player_id", "dead_id", "player_id", "victim_id", "target_id"))
            if not pid: return None
            return "death_records", self._record("death", round_value, phase, pid, None, m, "system_public")
        if badge:
            holder = self._first_id(m, ("to_player_id", "badge_to", "badge_holder_id", "badge_holder", "current_sheriff", "sheriff_id", "sheriff"))
            source = self._first_id(m, ("from_player_id", "badge_from", "source_id"))
            if not holder and not source: return None
            record = self._record("badge", round_value, phase, holder, None, m, "system_public")
            record["from_player_id"] = source or "?"
            record["to_player_id"] = holder or "?"
            return "badge_records", record
        if reveal:
            pid = self._first_id(m, ("player_id", "target_id", "revealed_player_id"))
            role = self._first_scalar(m, ("publicly_revealed_role", "revealed_role", "role_reveal", "role"))
            if not pid: return None
            return "public_reveals", self._record("reveal", round_value, phase, pid, None, m, "system_public", role=role)
        if vote and (has("voter_id", "voter") or "vote" in e or "ballot" in e) and has("vote_target", "target_id", "target", "ballot"):
            actor = self._first_id(m, ("voter_id", "voter", "player_id", "actor_id"))
            target = self._first_id(m, ("vote_target", "target_id", "target", "ballot"))
            if actor and target:
                return "vote_records", self._record("vote", round_value, phase, actor, target, m, "system_public")
        text = self._first_scalar(m, ("text", "content", "message", "speech", "statement"))
        if claim and not text:
            result = self._first_scalar(m, ("check_result", "result", "outcome", "claim_result", "查验结果"))
            role = self._first_scalar(m, ("claim_role", "claimed_role", "role"))
            if result or role:
                text = self._trim_text("宣称 " + (role + " " if role else "") + result, 100)
        if (speech or claim) and text:
            actor = self._first_id(m, ("speaker_id", "claimer_id", "player_id", "actor_id"))
            target = self._first_id(m, ("checked_player_id", "claimed_player_id", "target_id", "target"))
            if actor:
                kind = "claim_records" if claim else "recent_speeches"
                return kind, self._record("claim" if claim else "speech", round_value, phase, actor, target, m, "player_statement", text=text, role=self._first_scalar(m, ("claim_role", "claimed_role", "role")))
        return None

    def _record(self, kind: str, round_value: Any, phase: str, actor: str | None, target: str | None, m: Mapping[str, Any], source: str, **extra: Any) -> dict[str, Any]:
        record = {"kind": kind, "round": self._trim_text(str(round_value), 24) if round_value is not None else "?",
                  "phase": self._trim_text(phase, 24), "player_id": actor or "?",
                  "target_id": target or "?", "source": source}
        text = self._first_scalar(m, ("text", "content", "message", "speech", "reason"))
        if text: record["text"] = self._trim_text(text, 100)
        record.update({k: v for k, v in extra.items() if v not in (None, "")})
        if "to_id" in record: record["to_player_id"] = record.pop("to_id")
        return record

    def _append_record(self, records: list[dict[str, Any]], record: dict[str, Any], limit: int = _PUBLIC_SNAPSHOT_LIMIT) -> None:
        key = json.dumps(record, ensure_ascii=False, sort_keys=True, default=str)
        if not any(json.dumps(x, ensure_ascii=False, sort_keys=True, default=str) == key for x in records):
            records.append(record)
        if len(records) > limit:
            del records[:-limit]

    def _dedupe_records(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for record in records:
            self._append_record(result, record)
        return result

    # ---- Compact, auditable model context ---------------------------------
    def _render_turn_summary(self, turn_packet: Mapping[str, Any], request: Mapping[str, Any]) -> str:
        public_state = turn_packet.get("public_state")
        if isinstance(public_state, (Mapping, list)):
            self._ingest_public_packet(public_state)
        alive = self._sort_player_ids(set(self._public_ledger["alive_ids"]) - set(self._public_ledger["dead_ids"]))
        dead = self._sort_player_ids(self._public_ledger["dead_ids"])
        allowed = self._allowed_target_ids(request)
        candidates = self._candidate_pool(request, alive, list(self._public_ledger["known_player_ids"]))
        lines = ["【证据审计】", f"轮次={self._public_ledger.get('round') or '?'} 阶段={self._public_ledger.get('phase') or request.get('phase') or '?'}",
                 "存活: " + self._join_ids(alive, limit=12), "已死: " + self._join_ids(dead, limit=12),
                 "公开翻牌: " + self._render_records("public_reveals", 4),
                 "最近两轮票型: " + self._render_recent_votes(8),
                 "警徽来源/持有者: " + self._render_badges()]
        holder = self._badge_holder()
        if holder:
            chain = self._has_supported_claim_chain(holder)
            lines.append(f"持徽者={holder}；持续查验链={'有' if chain else '无/未验证'}；警徽不是身份认证，必须重新审计")
        uncovered = [p for p in alive if p not in self._claim_covered_players() and p != self.player_id]
        lines.append("未被公开链覆盖: " + self._join_ids(uncovered, limit=8))
        vote_phase = self._is_vote_request(request)
        ranked = self._rank_candidates(candidates, [], dead)
        if vote_phase:
            lines.append(self._render_vote_summary(candidates, alive, dead, []))
        else:
            lines.append(self._render_speak_summary(candidates, alive, dead, []))
        if len(alive) <= 6 or vote_phase and len(alive) <= 8:
            lines.append("【残局模式】人数少，先重审翻牌、票型和警徽继承；今天投错可能直接达到狼人胜利，勿因继承警徽或前任支持自动保护。")
        return self._trim_text("\n".join(lines), _PUBLIC_BLOCK_CHAR_LIMIT + _PHASE_BLOCK_CHAR_LIMIT)

    def _render_vote_summary(self, candidate_pool: list[str], alive_ids: list[str], dead_ids: list[str], notes: list[str]) -> str:
        del alive_ids, notes
        ranked = self._rank_candidates(candidate_pool, [], dead_ids)[:2]
        lines = ["【本轮投票参考】"]
        for i, (pid, score, reason) in enumerate(ranked, 1):
            lines.append(f"{i}. {pid} | 可疑度={score} | {reason}")
        if len(ranked) >= 2:
            lines.append(f"必须比较：{ranked[0][0]} vs {ranked[1][0]}；逐项看系统事实/声明/票型。警徽继承、前任支持、暂无对跳均不自动保护")
        else:
            lines.append("候选不足；只按当前允许目标投票，不把未知当成事实。")
        return self._trim_text("\n".join(lines), _PHASE_BLOCK_CHAR_LIMIT)

    def _render_recent_votes(self, limit: int) -> str:
        records = self._public_ledger.get("vote_records", [])
        if not records:
            return "无"
        current = self._public_ledger.get("round")
        try:
            current_number = int(str(current))
        except (TypeError, ValueError):
            current_number = None
        if current_number is not None:
            selected = []
            for record in records:
                try:
                    number = int(str(record.get("round", "")))
                except (TypeError, ValueError):
                    number = None
                if number is not None and number >= current_number - 1:
                    selected.append(record)
            if selected:
                records = selected
        return " | ".join(self._record_text(r) for r in records[-limit:]) or "无"

    def _render_speak_summary(self, candidate_pool: list[str], alive_ids: list[str], dead_ids: list[str], notes: list[str]) -> str:
        del notes
        ranked = self._rank_candidates(candidate_pool or alive_ids, [], dead_ids)
        if not ranked: return "【本轮发言参考】公开证据不足；区分事实、声明和推测。"
        focus = ranked[0]
        avoid = ranked[-1]
        return self._trim_text(f"【本轮发言参考】关注={focus[0]}（{focus[2]}）；暂不投={avoid[0]}（{avoid[2]}）。", _PHASE_BLOCK_CHAR_LIMIT)

    def _rank_candidates(self, candidate_pool: list[str], notes: list[str], dead_ids: list[str]) -> list[tuple[str, int, str]]:
        del notes
        dead = set(dead_ids)
        result = []
        for pid in self._sort_player_ids(candidate_pool):
            if pid == self.player_id or pid in dead: continue
            score, reasons = self._score_candidate(pid, [])
            result.append((pid, score, "；".join(reasons[:2]) if reasons else "公开信息不足"))
        result.sort(key=lambda x: (-x[1], x[0]))
        return result

    def _least_suspicious_candidate(self, candidate_pool: list[str], notes: list[str], dead_ids: list[str]) -> tuple[str, int, str] | None:
        del notes
        ranked = self._rank_candidates(candidate_pool, [], dead_ids)
        return min(ranked, key=lambda x: (x[1], x[0])) if ranked else None

    def _score_candidate(self, candidate: str, notes: list[str]) -> tuple[int, list[str]]:
        del notes
        score = 0
        reasons: list[str] = []
        ledger = self._public_ledger
        reveals = [r for r in ledger["public_reveals"] if r.get("player_id") == candidate]
        deaths = [r for r in ledger["death_records"] if r.get("player_id") == candidate]
        votes = [r for r in ledger["vote_records"] if r.get("player_id") == candidate]
        claims = [r for r in ledger["claim_records"] if r.get("player_id") == candidate or r.get("target_id") == candidate]
        if deaths or candidate in ledger.get("dead_ids", set()):
            return -100, ["系统事实：已公开死亡"]
        for r in reveals:
            role = str(r.get("role", ""))
            if any(x in role.lower() for x in ("wolf", "狼人", "werewolf")):
                score += 20; reasons.append(f"系统翻牌：{candidate}={role}")
            else:
                score -= 10; reasons.append(f"系统翻牌：{candidate}={role or '已公开'}")
        # A ballot is a system fact, but its meaning depends on the publicly
        # revealed target; do not reward or punish a player merely for voting.
        reveal_roles = {str(r.get("player_id")): str(r.get("role", "")) for r in ledger["public_reveals"]}
        for r in votes[-3:]:
            target = str(r.get("target_id", "?"))
            target_role = reveal_roles.get(target, "")
            if any(x in target_role.lower() for x in ("wolf", "狼人", "werewolf")):
                score -= 2
            elif target in reveal_roles:
                score += 2
        if votes:
            reasons.append("系统票型：" + ",".join(f"第{r.get('round','?')}轮投{r.get('target_id','?')}" for r in votes[-2:]))
        for r in claims:
            text = str(r.get("text", ""))
            if r.get("target_id") == candidate and any(x in text.lower() for x in ("查杀", "狼人", "狼", "wolf", "bad", "坏人")):
                score += 4; reasons.append("玩家声明：被报查杀（声明，须与系统票型核对）")
            elif r.get("target_id") == candidate and any(x in text.lower() for x in ("金水", "好人", "good", "villager")):
                score -= 2; reasons.append("玩家声明：被报好人（声明，非系统事实）")
            elif r.get("player_id") == candidate:
                reasons.append("玩家声明：自称/查验，未自动认证")
        holder = self._badge_holder()
        if holder == candidate:
            if not self._has_supported_claim_chain(candidate):
                score += 1; reasons.append("系统警徽事实：继承者无持续查验链，必须重新比较")
            else:
                reasons.append("系统警徽事实：持徽但警徽不增加好人分")
        return score, reasons[:2]

    def _candidate_pool(self, request: Mapping[str, Any], alive_ids: list[str], known_ids: list[str]) -> list[str]:
        allowed = self._allowed_target_ids(request)
        pool = set(allowed) if allowed else set(alive_ids or known_ids)
        pool.discard(self.player_id)
        pool.difference_update(self._public_ledger["dead_ids"])
        return self._sort_player_ids(list(pool))

    def _allowed_target_ids(self, request: Mapping[str, Any]) -> list[str]:
        result: list[str] = []
        actions = request.get("allowed_actions")
        if isinstance(actions, list):
            for action in actions:
                if not isinstance(action, Mapping): continue
                targets = action.get("target_ids")
                if isinstance(targets, list):
                    for target in targets:
                        if self._is_scalar_id(target) and str(target) != self.player_id:
                            self._merge_unique(result, [str(target)])
        return result

    def _collect_claim_block(self, notes: list[str]) -> str:
        del notes
        return self._render_records("claim_records", 3)

    def _extract_public_notes(self, payload: Any) -> list[str]:
        facts = self._parse_public_snapshot(payload)
        notes: list[str] = []
        for key in ("recent_events", "claim_records", "vote_records", "badge_records", "death_records"):
            for record in facts[key]:
                notes.append(self._record_text(record))
        return self._dedupe_preserve_order(notes)

    def _snapshot_mapping(self, mapping: Mapping[str, Any]) -> str | None:
        facts = self._parse_public_snapshot(mapping)
        for key in ("vote_records", "badge_records", "death_records", "public_reveals", "claim_records", "recent_speeches"):
            if facts[key]: return self._record_text(facts[key][0])
        return None

    def _record_text(self, r: Mapping[str, Any]) -> str:
        bits = [str(r.get("kind", "event")), f"round={r.get('round','?')}"]
        for key in ("player_id", "target_id", "from_player_id", "to_player_id", "role", "text", "source"):
            if r.get(key) not in (None, "", "?"):
                bits.append(f"{key}={self._trim_text(str(r[key]), 80)}")
        return self._trim_text(" ".join(bits), _PUBLIC_NOTE_CHAR_LIMIT)

    def _render_records(self, key: str, limit: int) -> str:
        records = self._public_ledger.get(key, [])[-limit:]
        return " | ".join(self._record_text(r) for r in records) or "无"

    def _render_badges(self) -> str:
        return self._render_records("badge_records", 4)

    def _badge_holder(self) -> str | None:
        records = self._public_ledger.get("badge_records", [])
        if not records: return None
        record = records[-1]
        holder = record.get("to_player_id") or record.get("player_id")
        return str(holder) if holder and holder != "?" else None

    def _claim_covered_players(self) -> set[str]:
        covered: set[str] = set()
        for r in self._public_ledger["claim_records"]:
            if r.get("target_id") not in (None, "?"): covered.add(str(r["target_id"]))
        return covered

    def _has_supported_claim_chain(self, player_id: str) -> bool:
        # 持徽不等于预言家。把“持续”定义为至少两个不同轮次的明确查验
        # 声明；一次自称或一次金水不能构成自动保护。
        records = [r for r in self._public_ledger["claim_records"]
                   if r.get("player_id") == player_id
                   and any(x in str(r.get("text", "")) for x in ("查验", "验人", "金水", "查杀", "check"))]
        rounds = {str(r.get("round", "?")) for r in records}
        rounds.discard("?")
        return len(rounds) >= 2

    def _is_vote_request(self, request: Mapping[str, Any]) -> bool:
        phase = str(request.get("phase") or request.get("channel") or request.get("kind") or "").lower()
        actions = request.get("allowed_actions")
        kinds = [str(a.get("kind", "")).lower() for a in actions if isinstance(a, Mapping)] if isinstance(actions, list) else []
        return any(x in phase for x in ("vote", "投票")) or any("vote" in x for x in kinds)

    # ---- Small parsing helpers --------------------------------------------
    def _extract_game_id(self, payload: Mapping[str, Any]) -> str | None:
        for key in ("game_id", "gameid", "match_id", "matchid", "session_id"):
            if self._is_scalar_text(payload.get(key)): return str(payload[key])
        game = payload.get("game")
        if isinstance(game, Mapping): return self._extract_game_id(game)
        return None

    def _first_id(self, mapping: Mapping[str, Any], keys: tuple[str, ...]) -> str | None:
        for key in keys:
            value = mapping.get(key)
            if self._is_scalar_id(value): return str(value)
        return None

    def _first_scalar(self, mapping: Mapping[str, Any], keys: tuple[str, ...]) -> str:
        for key in keys:
            value = mapping.get(key)
            if self._is_scalar_text(value): return self._trim_text(str(value), _PUBLIC_NOTE_CHAR_LIMIT)
        return ""

    def _collect_id_collection(self, mapping: Mapping[str, Any], allowed_keys: set[str]) -> set[str]:
        result: set[str] = set()
        lower = {str(k).lower(): v for k, v in mapping.items()}
        for key, value in lower.items():
            if key in allowed_keys: result.update(self._as_id_set(value))
        return result

    def _extract_player_ids(self, payload: Any, *, include_collections: bool = False) -> set[str]:
        facts = self._parse_public_snapshot(payload)
        # If the packet states who is alive, return only that authoritative set.
        # Otherwise retain only observed, non-dead IDs; this helper must never make
        # a dead player alive merely because they occur in a historical event.
        if facts["authoritative_players"] or facts["explicit_alive"]:
            ids = set(facts["alive_ids"])
        else:
            ids = set(facts["known_player_ids"])
        if include_collections:
            ids.update(facts["alive_ids"])
        ids.difference_update(facts["dead_ids"])
        ids.discard(self.player_id)
        return ids

    def _extract_dead_from_notes(self, notes: list[str]) -> set[str]:
        return {pid for note in notes if any(x in note.lower() for x in ("death", "dead", "死亡", "出局")) for pid in self._ids_from_text(note)}

    def _ids_from_text(self, text: str) -> set[str]:
        return {token.strip("()[]{}<>：:=，。,.、") for token in re.split(r"[\s,，;；|/\\]+", text) if self._looks_like_player_id(token.strip("()[]{}<>：:=，。,.、"))}

    def _as_id_set(self, value: Any) -> set[str]:
        if isinstance(value, (list, tuple, set)):
            return {str(x) for x in value if self._is_scalar_id(x)}
        if isinstance(value, Mapping):
            return set()
        return {str(value)} if self._is_scalar_id(value) else set()

    def _merge_unique(self, items: list[str], values: list[str]) -> None:
        for value in values:
            if value and value not in items: items.append(value)

    def _append_bounded(self, items: list[Any], value: Any, limit: int) -> None:
        if value not in items: items.append(value)
        if len(items) > limit: del items[:-limit]

    def _trim_text(self, text: str, limit: int) -> str:
        normalized = " ".join(str(text).split())
        return normalized if len(normalized) <= limit else (normalized[:max(0, limit - 1)] + "…" if limit > 1 else normalized[:limit])

    def _dedupe_preserve_order(self, items: list[str]) -> list[str]:
        result: list[str] = []
        for item in items:
            if item and item not in result: result.append(item)
        return result

    def _join_ids(self, ids: list[str], *, limit: int) -> str:
        if not ids: return "未知"
        selected = ids[:limit]
        extra = len(ids) - len(selected)
        return "、".join(selected) + (f" 等{extra}人" if extra else "")

    def _sort_player_ids(self, ids: list[str]) -> list[str]:
        unique = self._dedupe_preserve_order([str(x) for x in ids if str(x)])
        def key(value: str) -> tuple[str, int, str]:
            match = _PLAYER_ID_SUFFIX.search(value)
            return (_PLAYER_ID_SUFFIX.sub("", value), int(match.group(1)) if match else 10**9, value)
        return sorted(unique, key=key)

    def _looks_like_player_id(self, value: str) -> bool:
        text = str(value).strip()
        return bool(text) and len(text) <= 16 and any(ch.isdigit() for ch in text)

    @staticmethod
    def _is_scalar_text(value: Any) -> bool:
        return isinstance(value, (str, int, float)) and not isinstance(value, bool)

    def _is_scalar_id(self, value: Any) -> bool:
        return (isinstance(value, int) and not isinstance(value, bool)) or (isinstance(value, str) and self._looks_like_player_id(value))

    @staticmethod
    def _player_id_keys() -> set[str]:
        return {"player_id", "speaker_id", "voter_id", "claimer_id", "actor_id", "target_id", "from_id", "to_id", "badge_holder_id", "sheriff_id", "dead_id", "dead_player_id"}

    @staticmethod
    def _collection_id_keys() -> set[str]:
        return {"alive_ids", "dead_ids", "dead_player_ids", "players", "player_ids", "target_ids", "candidates", "survivors", "living", "graveyard"}

    @staticmethod
    def _ALIVE_KEYS() -> set[str]:
        return {"alive", "alive_ids", "alive_player_ids", "survivors", "living", "live_ids", "current_alive"}

    @staticmethod
    def _DEAD_KEYS() -> set[str]:
        return {"dead", "dead_ids", "dead_player_ids", "death_order", "eliminated", "casualties", "deaths", "graveyard"}

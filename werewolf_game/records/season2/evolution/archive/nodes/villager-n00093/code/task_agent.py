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
        lines.append("投票时把查验、金水、预言家、警徽继承和他人支持都先标为玩家声明；目标死亡或击杀对跳者不验证查验。比较硬矛盾、对跳、票型和弱协调线，不要仅凭持续报验保护。")
        lines.append("有两个以上合法目标且任一有明确矛盾、未验证查杀指向或强协调线时不要 pass；只有候选几乎等价或会明确误伤独立事实保护对象才考虑 pass。")
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
            # These mappings are derived only from the public players list.  A seat
            # is never treated as a player id unless the mapping is unique.
            "player_seats": {},
            "seat_to_player_id": {},
            "events": [],
            "verified_public_facts": [],
            "claims": [],
            "role_claims": [],
            "inspection_claims": [],
            "votes": [],
            "sheriff_votes": [],
            "deaths": [],
            "supports": [],
            "oppositions": [],
            "badges": [],
        }

    # The ledger deliberately stores only public, bounded records.  In particular, a
    # target_id is not evidence of a vote: the event type/key names must say so.
    def _ingest_public_packet(self, payload: Any) -> None:
        if not isinstance(payload, Mapping):
            return
        self._update_player_mapping(payload)
        self._public_ledger["known_player_ids"].update(
            self._normalize_id_set(self._extract_player_ids(payload, include_collections=True))
        )
        self._public_ledger["alive_ids"].update(self._normalize_id_set(self._collect_id_collection(payload, self._ALIVE_KEYS())))
        self._public_ledger["dead_ids"].update(self._normalize_id_set(self._collect_id_collection(payload, self._DEAD_KEYS())))
        for record in self._extract_public_records(payload):
            self._store_record(record)

    def _store_record(self, record: dict[str, Any]) -> None:
        """Store a de-duplicated, public record in a typed long-lived ledger."""
        record = dict(record)
        for field in ("round", "speaker", "voter", "target", "text", "source_kind"):
            record.setdefault(field, "")
        for field in ("speaker", "voter", "target", "source"):
            if record.get(field):
                record[field] = self._normalize_player_ref(record[field])
        record["text"] = self._trim_text(record.get("text", ""), _PUBLIC_NOTE_CHAR_LIMIT)
        event_type = str(record.get("event_type", "unknown"))
        if event_type == "vote":
            bucket = "votes"
        elif event_type == "sheriff_vote":
            bucket = "sheriff_votes"
        elif event_type == "death":
            bucket = "deaths"
            target = record.get("target") or record.get("speaker")
            if target:
                self._public_ledger["dead_ids"].add(str(target))
        elif event_type in {"role_claim", "inspection_claim"}:
            bucket = "claims"
        elif event_type in {"support", "opposition"}:
            bucket = event_type + "s"
        elif event_type in {"badge", "hunter_shot"}:
            bucket = "badges" if event_type == "badge" else "verified_public_facts"
        elif record.get("source_kind") == "system":
            bucket = "verified_public_facts"
        else:
            bucket = "events"
        key = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        existing = self._public_ledger[bucket]
        if not any(json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":")) == key for item in existing):
            # Keep history by category rather than a recent-24 global window.  The
            # renderer still applies a strict character budget before model input.
            self._append_bounded(existing, record, 500 if bucket in {"events", "verified_public_facts"} else 240)
        if event_type in {"role_claim", "inspection_claim"}:
            self._append_bounded(self._public_ledger[event_type + "s"], record, 240)
        if record.get("source_kind") == "system":
            self._append_bounded(self._public_ledger["verified_public_facts"], record, 500)
        self._append_bounded(self._public_ledger["events"], record, 500)
        for value in (record.get("speaker"), record.get("voter"), record.get("target"), record.get("source")):
            if value and self._looks_like_player_id(str(value)):
                self._public_ledger["known_player_ids"].add(str(value))

    def _extract_public_records(self, payload: Any) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        self._walk_records(payload, records, 0)
        unique: list[dict[str, Any]] = []
        seen: set[str] = set()
        for record in records:
            key = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            if key not in seen:
                seen.add(key)
                unique.append(record)
        return unique

    def _walk_records(self, payload: Any, records: list[dict[str, Any]], depth: int) -> None:
        if depth > 5:
            return
        if isinstance(payload, Mapping):
            record = self._record_from_mapping(payload)
            if record:
                records.append(record)
            for value in payload.values():
                self._walk_records(value, records, depth + 1)
        elif isinstance(payload, list):
            for item in payload[:30]:
                self._walk_records(item, records, depth + 1)

    def _record_from_mapping(self, mapping: Mapping[str, Any]) -> dict[str, Any] | None:
        data = {str(k).lower(): v for k, v in mapping.items()}
        raw_event = self._first_scalar(data, ("event_type", "event", "type", "action", "kind"))
        event = raw_event.lower()
        text = self._first_scalar(data, ("text", "content", "message", "speech", "reason"))
        speaker = self._first_scalar(data, ("speaker_id", "claimer_id", "player_id", "actor_id", "source_id"))
        voter = self._first_scalar(data, ("voter_id", "voter", "ballot_by"))
        target = self._first_scalar(data, ("vote_target", "target_id", "target", "suspect_id", "to_id"))
        if not target and text:
            target = self._infer_target_from_text(text)
        round_text = self._first_scalar(data, ("round", "day", "turn"))
        lower_event = event.lower()
        # Explicit engine event names take precedence over keyword heuristics.
        no_death = "no_one_died" in lower_event or "no_death" in lower_event or "无人死亡" in text
        explicit_sheriff_vote = "sheriff_vote_cast" in lower_event or lower_event in {"sheriff_vote", "警长投票"}
        explicit_vote = "vote_cast" in lower_event or lower_event in {"vote", "day_vote", "放逐投票"}
        explicit_death = any(x in lower_event for x in ("player_died", "player_eliminated", "hunter_shot_fired"))
        explicit_badge = any(x in lower_event for x in ("sheriff_elected", "badge_transferred", "badge_passed"))
        if explicit_sheriff_vote:
            return {"round": round_text, "event_type": "sheriff_vote", "voter": voter or speaker, "target": target, "text": text, "source_kind": "system"}
        if explicit_vote:
            return {"round": round_text, "event_type": "vote", "voter": voter or speaker, "target": target, "text": text, "source_kind": "system"}
        if explicit_death and not no_death:
            dead = self._first_scalar(data, ("dead_id", "dead_player_id", "player_id", "target_id", "target"))
            return {"round": round_text, "event_type": "death", "speaker": dead, "target": dead, "text": text, "source_kind": "system"}
        if explicit_badge:
            return {"round": round_text, "event_type": "badge", "speaker": speaker, "target": target, "source": self._first_scalar(data, ("from_id", "badge_from")), "text": text, "source_kind": "system"}
        has_vote = not no_death and ("vote" in lower_event or any(k in data for k in ("voter_id", "voter", "vote_target", "ballot_by", "ballot")))
        has_death = not no_death and (any(t in lower_event for t in ("death", "dead", "kill", "eliminat", "out", "死亡", "出局")) or any(k in data for k in ("dead_id", "dead_player_id")))
        has_badge = any(t in lower_event for t in ("sheriff", "badge", "警徽", "警长")) or any(k in data for k in ("badge_holder", "badge_holder_id", "badge_from", "badge_to"))
        has_speech = bool(text) and (speaker or any(t in lower_event for t in ("speak", "speech", "claim", "say", "发言", "声明")))
        if has_vote:
            return {"round": round_text, "event_type": "vote", "voter": voter or speaker, "target": target, "text": text, "source_kind": "system"}
        if has_death:
            dead = self._first_scalar(data, ("dead_id", "dead_player_id", "player_id", "target_id", "target"))
            return {"round": round_text, "event_type": "death", "speaker": dead, "target": dead, "text": text, "source_kind": "system"}
        if has_badge:
            return {"round": round_text, "event_type": "badge", "speaker": speaker, "target": target, "source": self._first_scalar(data, ("from_id", "badge_from")), "text": text, "source_kind": "system"}
        if not has_speech:
            return None
        claim_type, claimed_role, claimed_alignment = self._classify_claim(text, data)
        if claim_type is None:
            claim_type = "support" if any(t in text for t in ("支持", "相信", "跟", "同意", "保他", "保她")) else "opposition" if any(t in text for t in ("反对", "不信", "怀疑", "投", "狼")) else "speech"
        if claim_type == "speech":
            return {"round": round_text, "event_type": "events", "speaker": speaker, "target": target, "text": text, "source_kind": "player_statement"}
        return {"round": round_text, "event_type": claim_type, "speaker": speaker, "target": target, "claimed_role": claimed_role, "claimed_alignment": claimed_alignment, "text": text, "source_kind": "player_statement", "independent_verification": False}

    def _infer_target_from_text(self, text: str) -> str:
        match = re.search(r"(?:查杀|查验|验人|金水|银水|怀疑|反对|支持|投)\s*(?:玩家|选手)?\s*([pP]?\d+|\d+号)", str(text), re.IGNORECASE)
        if not match:
            return ""
        return self._normalize_player_ref(match.group(1))

    def _classify_claim(self, text: str, data: Mapping[str, Any]) -> tuple[str | None, str, str]:
        role = self._first_scalar(data, ("claim_role", "claimed_role", "role_claim"))
        if not role:
            for candidate in ("预言家", "女巫", "猎人", "守卫", "警长", "村民"):
                if candidate in text:
                    role = candidate
                    break
        inspection = any(t in text for t in ("查杀", "查验", "验人", "金水", "银水")) or any(k in data for k in ("inspection", "inspection_result", "checked_alignment"))
        if inspection:
            alignment = "wolf" if "查杀" in text or "狼人" in text else "good" if any(t in text for t in ("金水", "银水", "好人")) else self._first_scalar(data, ("checked_alignment", "alignment"))
            return "inspection_claim", role, alignment
        if role:
            return "role_claim", role, ""
        if any(t in text for t in ("支持", "相信", "跟票", "保他", "保她")):
            return "support", "", ""
        if any(t in text for t in ("反对", "不信", "怀疑", "投他", "投她")):
            return "opposition", "", ""
        return None, "", ""

    def _records_for_turn(self, turn_packet: Mapping[str, Any]) -> list[dict[str, Any]]:
        # Ingest the visible snapshot as well; deduplication makes repeated snapshots
        # harmless.  Do not discard early votes/claims: the renderer aggregates them.
        self._update_player_mapping(turn_packet.get("public_state"))
        for record in self._extract_public_records(turn_packet.get("public_state")):
            self._store_record(record)
        return list(self._public_ledger["events"])

    def _render_turn_summary(self, turn_packet: Mapping[str, Any], request: Mapping[str, Any]) -> str:
        public_state = turn_packet.get("public_state")
        records = self._records_for_turn(turn_packet)
        current_ids = self._normalize_id_set(self._extract_player_ids(public_state, include_collections=True))
        alive = self._sort_player_ids((set(self._public_ledger["alive_ids"]) | current_ids) - set(self._public_ledger["dead_ids"]))
        dead = self._sort_player_ids(set(self._public_ledger["dead_ids"]))
        candidate_pool = self._candidate_pool(request, alive, self._sort_player_ids(list(self._public_ledger["known_player_ids"])))
        phase = str(request.get("phase") or request.get("channel") or request.get("kind") or "")
        allowed = request.get("allowed_actions")
        kinds = {str(a.get("kind", "")).lower() for a in allowed if isinstance(a, Mapping)} if isinstance(allowed, list) else set()
        is_vote = "vote" in phase.lower() or "投票" in phase or any("vote" in k for k in kinds)
        audit = self._decision_audit(records, alive, dead, candidate_pool)
        phase_summary = self._render_vote_summary(candidate_pool, alive, dead, records) if is_vote else self._render_speak_summary(candidate_pool, alive, dead, records)
        return self._trim_text(audit + "\n" + phase_summary, _PUBLIC_BLOCK_CHAR_LIMIT + _PHASE_BLOCK_CHAR_LIMIT)

    def _decision_audit(self, records: list[dict[str, Any]], alive: list[str], dead: list[str], candidates: list[str]) -> str:
        facts: list[str] = []
        claims: list[str] = []
        for record in records:
            kind = record.get("event_type")
            if record.get("source_kind") == "system" and kind in {"vote", "sheriff_vote", "death", "badge", "hunter_shot"}:
                who = record.get("voter") or record.get("speaker") or "?"
                target = record.get("target") or "?"
                facts.append(f"{kind} {who}->{target} r={record.get('round') or '?'}")
            elif kind in {"role_claim", "inspection_claim", "support", "opposition"}:
                if kind == "inspection_claim":
                    status = "未被独立系统验证"
                    claims.append(f"{record.get('speaker') or '?'} 声称查验 {record.get('target') or '?'}={record.get('claimed_alignment') or '?'} r={record.get('round') or '?'}（{status}）")
                else:
                    claims.append(f"{kind} {record.get('speaker') or '?'}->{record.get('target') or '?'}：{self._trim_text(record.get('text',''), 55)}")
        conflicts = self._claim_conflicts(records)
        coordination = self._coordination_players(records)
        lines = ["【公开证据审计】", "1）系统事实：存活=" + self._join_ids(alive, limit=10) + "；死亡=" + self._join_ids(dead, limit=10)]
        lines.append("事件：" + (" | ".join(facts[-5:]) if facts else "无明确公开投票/死亡/警徽事件"))
        vote_table = self._render_vote_table(records)
        lines.append("公开票型表（按轮次，日票/警长票分开）：" + (vote_table or "无"))
        lines.append("2）玩家声明：" + (" | ".join(claims[-6:]) if claims else "无结构化声明"))
        lines.append("3）声明冲突/对跳：" + (" | ".join(conflicts[:4]) if conflicts else "未识别；仍须核对原话"))
        lines.append("4）票型与死亡：死亡只证明死亡，不验证查验；对跳者被击杀也不验证存活者。" + (" 票型=" + " | ".join(facts[-3:]) if facts else ""))
        if coordination:
            lines.append("重复发言线索（弱协调，不是可信度加分）：" + "、".join(coordination[:5]))
        lines.append("5）当前至少两个候选：" + ("、".join(candidates[:4]) if candidates else "无合法候选"))
        lines.append("硬规则：目标死亡不验证查验；击杀对跳者不验证存活者；警徽/票型支持只是支持证据；两个未验证预言家不能仅凭持续报验定性。先用verified facts和票表，再用claims，最后才看当前对话与推测。")
        return self._trim_text("\n".join(lines), _PUBLIC_BLOCK_CHAR_LIMIT)

    def _render_vote_table(self, records: list[dict[str, Any]]) -> str:
        rows: dict[str, dict[str, list[str]]] = {}
        for record in records:
            kind = record.get("event_type")
            if kind not in {"vote", "sheriff_vote"} or not record.get("voter") or not record.get("target"):
                continue
            round_key = str(record.get("round") or "?")
            label = "日票" if kind == "vote" else "警长票"
            rows.setdefault(round_key, {}).setdefault(label, []).append(f"{record['voter']}->{record['target']}")
        rendered: list[str] = []
        for round_key in sorted(rows, key=lambda x: (int(x) if x.isdigit() else 10**9, x)):
            groups = rows[round_key]
            rendered.append(f"r{round_key} " + ";".join(f"{label}:{','.join(values[-8:])}" for label, values in groups.items()))
        return " | ".join(rendered[-10:])

    def _claim_conflicts(self, records: list[dict[str, Any]]) -> list[str]:
        """Report actual, typed conflicts; multiple reports alone are not a conflict."""
        claims = [r for r in records if r.get("event_type") == "inspection_claim"]
        result: list[str] = []
        by_target: dict[str, list[dict[str, Any]]] = {}
        by_role: dict[str, list[dict[str, Any]]] = {}
        for claim in claims:
            target = str(claim.get("target") or "未解析目标")
            by_target.setdefault(target, []).append(claim)
            role = str(claim.get("claimed_role") or "预言家")
            by_role.setdefault(role, []).append(claim)
        for target, group in by_target.items():
            alignments = {str(x.get("claimed_alignment") or "unknown") for x in group}
            if len(alignments) > 1:
                speakers = self._join_ids(self._sort_player_ids([str(x.get("speaker")) for x in group if x.get("speaker")]), limit=4)
                result.append(f"同目标{target}的相反查验：{speakers}，结论={','.join(sorted(alignments))}（均未验证）")
        for role, group in by_role.items():
            speakers = self._sort_player_ids([str(x.get("speaker")) for x in group if x.get("speaker")])
            if len(speakers) > 1:
                first_rounds = [(str(x.get("round") or "?"), str(x.get("speaker") or "?"), str(x.get("target") or "未解析")) for x in group]
                detail = ",".join(f"{s}@r{r}->{t}" for r, s, t in first_rounds[:4])
                result.append(f"{role}多来源声明需对比首条/顺序：{detail}（非自动对跳）")
        for speaker in self._dedupe_preserve_order([str(x.get("speaker")) for x in claims if x.get("speaker")]):
            own = [x for x in claims if str(x.get("speaker")) == speaker]
            seen: dict[str, str] = {}
            for claim in own:
                target = str(claim.get("target") or "未解析")
                alignment = str(claim.get("claimed_alignment") or "unknown")
                if target in seen and seen[target] != alignment:
                    result.append(f"{speaker} 对{target}前后结论矛盾：{seen[target]}->{alignment}")
                seen[target] = alignment
        return self._dedupe_preserve_order(result)[:8]

    def _coordination_players(self, records: list[dict[str, Any]]) -> list[str]:
        speech = [r for r in records if r.get("text") and r.get("speaker")]
        result: set[str] = set()
        for i, left in enumerate(speech):
            normalized_left = self._normalize_speech(left["text"])
            if len(normalized_left) < 8:
                continue
            for right in speech[i + 1:]:
                if left["speaker"] == right["speaker"]:
                    continue
                normalized_right = self._normalize_speech(right["text"])
                if normalized_left == normalized_right or normalized_left in normalized_right or normalized_right in normalized_left:
                    result.update((str(left["speaker"]), str(right["speaker"])))
        return self._sort_player_ids(list(result))

    @staticmethod
    def _normalize_speech(text: str) -> str:
        return re.sub(r"[\s，。！？、,.!?：:；;]+", "", str(text)).lower()

    def _render_vote_summary(self, candidate_pool: list[str], alive_ids: list[str], dead_ids: list[str], records: list[dict[str, Any]]) -> str:
        ranked = self._rank_candidates(candidate_pool, records, dead_ids)
        legal = [pid for pid in self._sort_player_ids(alive_ids) if pid in candidate_pool and pid not in dead_ids and pid != self.player_id]
        for pid in legal:
            if pid not in {item[0] for item in ranked}:
                ranked.append((pid, 0, "组件均无明显信号", "声明和票型均未独立验证"))
        ranked = ranked[:2]
        lines = ["【本轮逐候选票型审计】（仅比较当前合法 target_id）"]
        if len(ranked) >= 2:
            for label, item in zip(("候选A", "候选B"), ranked):
                lines.append(f"{label}={item[0]}：组件依据={item[2]}；反证/未验证={item[3]}")
            lines.append("为什么A优于B：只在事实矛盾/未验证查杀/多轮票型较强时拉开差距；警徽仅是票权，复制发言只作微小tie-breaker。最终必须选择一个合法目标。")
            lines.append("不要pass：有两个以上合法目标且存在可解释差异时投票；只有证据近乎等价或会误伤独立系统事实保护对象才考虑pass。")
        elif ranked:
            lines.append(f"唯一候选={ranked[0][0]}：{ranked[0][2]}；{ranked[0][3]}。按合法行动决定，不臆造第二目标。")
        else:
            lines.append("没有可用合法投票目标；只能遵守 allowed_actions，不臆造目标。")
        return self._trim_text("\n".join(lines), _PHASE_BLOCK_CHAR_LIMIT)

    def _render_speak_summary(self, candidate_pool: list[str], alive_ids: list[str], dead_ids: list[str], records: list[dict[str, Any]]) -> str:
        ranked = self._rank_candidates(candidate_pool or alive_ids, records, dead_ids)
        focus = ranked[0] if ranked else ("信息不足", 0, "无结构化公开依据", "所有声明均未独立验证")
        avoid = ranked[-1] if ranked else None
        lines = ["【本轮发言参考】", f"关注：{focus[0]}；依据：{focus[2]}；未验证：{focus[3]}"]
        lines.append(f"暂不投：{avoid[0] if avoid else '无'}；说明一个系统事实和一个仍未验证的玩家声明。")
        return self._trim_text("\n".join(lines), _PHASE_BLOCK_CHAR_LIMIT)

    def _coerce_records(self, records: list[Any]) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for item in records:
            if isinstance(item, Mapping):
                normalized.append(dict(item))
            elif isinstance(item, str) and item.strip():
                normalized.append({"event_type": "events", "text": self._trim_text(item, _PUBLIC_NOTE_CHAR_LIMIT), "source_kind": "player_statement"})
        return normalized

    def _rank_candidates(self, candidate_pool: list[str], records: list[Any], dead_ids: list[str]) -> list[tuple[str, int, str, str]]:
        records = self._coerce_records(records)
        ranked: list[tuple[str, int, str, str]] = []
        for candidate in self._sort_player_ids(candidate_pool):
            if candidate in set(dead_ids) or candidate == self.player_id:
                continue
            score, reasons, unknown = self._score_candidate(candidate, records)
            ranked.append((candidate, score, "；".join(reasons[:2]) or "公开信息不足", "；".join(unknown[:2]) or "暂无独立反证"))
        ranked.sort(key=lambda item: (-item[1], item[0]))
        return ranked

    def _score_candidate(self, candidate: str, records: list[Any]) -> tuple[int, list[str], list[str]]:
        """Use public, componentized evidence; declarations never become facts."""
        records = self._coerce_records(records)
        score = 0
        reasons: list[str] = []
        unknown: list[str] = []
        claims = [r for r in records if r.get("event_type") in {"role_claim", "inspection_claim", "support", "opposition"}]
        inspections = [r for r in claims if r.get("event_type") == "inspection_claim"]
        for record in claims:
            speaker, target = str(record.get("speaker") or ""), str(record.get("target") or "")
            text = self._trim_text(record.get("text", ""), 70)
            if record.get("event_type") == "inspection_claim" and target == candidate and record.get("claimed_alignment") == "wolf":
                score += 2
                reasons.append(f"claim+2：{speaker or '?'}在r{record.get('round') or '?'}报未验证查杀")
                unknown.append("该查验仍是玩家声明，未被系统独立验证")
            if speaker == candidate and record.get("event_type") in {"role_claim", "inspection_claim"}:
                unknown.append("本人神职/查验声明未被系统确认")
            if target == candidate and record.get("event_type") == "support":
                score -= 1
                reasons.append(f"claim-1：{speaker or '?'}公开支持（仅弱保护）")
            if speaker == candidate and record.get("event_type") == "opposition":
                score += 1
                reasons.append(f"claim+1：公开反对/投向声明：{text}")

        day_votes = [r for r in records if r.get("event_type") == "vote" and r.get("voter") and r.get("target")]
        received = [r for r in day_votes if str(r.get("target")) == candidate]
        if received:
            amount = min(3, len(received))
            score += amount
            detail = ",".join(f"{r.get('voter')}@r{r.get('round') or '?'}" for r in received[-4:])
            reasons.append(f"vote+{amount}：获得公开日票({detail})")
        # Day votes and sheriff votes are separate.  A public badge holder gives
        # voting power, not identity protection; matching them is only moderate.
        holders = [str(r.get("target") or r.get("speaker")) for r in records if r.get("event_type") == "badge" and (r.get("target") or r.get("speaker"))]
        holder = holders[-1] if holders else ""
        if holder:
            together = [r for r in day_votes if str(r.get("voter")) == candidate and any(
                str(h.get("voter")) == holder and str(h.get("round")) == str(r.get("round")) and str(h.get("target")) == str(r.get("target")) for h in day_votes
            )]
            if together:
                amount = min(2, len(together))
                score += amount
                detail = ",".join(f"r{r.get('round') or '?'}->{r.get('target')}" for r in together[-3:])
                reasons.append(f"vote+{amount}：与警徽持有者{holder}同票({detail})，仅中等线索")
        claimant_ids = {str(r.get("speaker")) for r in inspections if r.get("speaker")}
        against_claimant = [r for r in day_votes if str(r.get("voter")) == candidate and str(r.get("target")) in claimant_ids]
        if against_claimant:
            score += 1
            detail = ",".join(f"r{r.get('round') or '?'}->{r.get('target')}" for r in against_claimant[-3:])
            reasons.append(f"vote+1：公开日票投向查验声明者({detail})，需核对首条声明")
        if candidate in {str(r.get("target")) for r in records if r.get("event_type") == "death"}:
            unknown.append("候选死亡/出局只证明死亡，不能验证任何查验")

        coordination = self._coordination_players(records)
        if candidate in coordination:
            reasons.append("coord+微弱：与他人高度重复发言")
            unknown.append("复制发言只能作微弱参考，不能单独定票")
        if not reasons:
            reasons.append("组件均无硬矛盾；公开信息有限")
        if not unknown:
            unknown.append("没有独立系统保护事实；玩家声明仍未验证")
        return score, reasons, unknown

    def _candidate_pool(self, request: Mapping[str, Any], alive_ids: list[str], known_ids: list[str]) -> list[str]:
        pool: set[str] = set(alive_ids) | set(known_ids)
        raw_actions = request.get("allowed_actions")
        vote_targets: set[str] = set()
        if isinstance(raw_actions, list):
            for action in raw_actions:
                if not isinstance(action, Mapping):
                    continue
                if str(action.get("kind", "")).lower() in {"day_vote", "vote", "sheriff_vote"}:
                    targets = action.get("target_ids")
                    if isinstance(targets, list):
                        vote_targets.update(str(x) for x in targets if self._is_scalar_id(x))
        # During a vote, the engine's target list is the only legal candidate set;
        # do not make the model compare a known but currently unselectable player.
        if vote_targets:
            pool = vote_targets
        pool.discard(self.player_id)
        pool.difference_update(self._public_ledger["dead_ids"])
        return self._sort_player_ids(list(pool))

    def _extract_public_notes(self, payload: Any) -> list[str]:
        return [self._format_record(r) for r in self._extract_public_records(payload)]

    def _snapshot_mapping(self, mapping: Mapping[str, Any]) -> str | None:
        """兼容旧的内部快照调用，但返回结构化记录的有界文本。"""
        record = self._record_from_mapping(mapping)
        return self._format_record(record) if record else None

    def _extract_dead_from_notes(self, notes: list[str]) -> set[str]:
        dead: set[str] = set()
        for note in notes:
            if any(word in note.lower() for word in ("death", "dead", "死亡", "出局")):
                dead.update(self._ids_from_text(note))
        return dead

    def _ids_from_text(self, text: str) -> set[str]:
        return {token.strip("()[]{}<>：:=，。,.、") for token in re.split(r"[\\s,，;；|/\\\\]+", str(text)) if self._looks_like_player_id(token.strip("()[]{}<>：:=，。,.、"))}

    def _merge_unique(self, items: list[str], values: list[str]) -> None:
        for value in values:
            if value and value not in items:
                items.append(value)

    def _least_suspicious_candidate(self, candidate_pool: list[str], records: list[dict[str, Any]], dead_ids: list[str]) -> tuple[str, int, str, str] | None:
        ranked = self._rank_candidates(candidate_pool, records, dead_ids)
        return min(ranked, key=lambda item: (item[1], item[0])) if ranked else None

    def _format_record(self, record: Mapping[str, Any]) -> str:
        return self._trim_text(json.dumps(dict(record), ensure_ascii=False, separators=(",", ":")), _PUBLIC_NOTE_CHAR_LIMIT)

    def _update_player_mapping(self, payload: Any) -> None:
        """Read only public ``players`` entries and retain unambiguous seat mappings."""
        if not isinstance(payload, Mapping):
            return
        players = payload.get("players")
        if not isinstance(players, list):
            state = payload.get("public_state")
            if isinstance(state, Mapping):
                players = state.get("players")
        if not isinstance(players, list):
            return
        candidates: dict[str, set[str]] = {}
        for item in players:
            if not isinstance(item, Mapping):
                continue
            lower = {str(k).lower(): v for k, v in item.items()}
            pid = self._first_scalar(lower, ("player_id", "id"))
            seat = self._first_scalar(lower, ("seat", "seat_id", "number", "player_number", "index"))
            if pid and seat:
                seat_key = self._seat_number(seat)
                if seat_key:
                    candidates.setdefault(seat_key, set()).add(self._normalize_known_id(pid))
        for seat, ids in candidates.items():
            if len(ids) != 1:
                continue
            pid = next(iter(ids))
            old = self._public_ledger["seat_to_player_id"].get(seat)
            if old and old != pid:
                # A changing/non-unique public mapping must not lead to guessing.
                self._public_ledger["seat_to_player_id"].pop(seat, None)
                self._public_ledger["player_seats"].pop(old, None)
                continue
            self._public_ledger["seat_to_player_id"][seat] = pid
            self._public_ledger["player_seats"][pid] = seat
            self._public_ledger["known_player_ids"].add(pid)

    @staticmethod
    def _seat_number(value: Any) -> str:
        match = re.fullmatch(r"\s*(?:seat\s*)?(\d+)(?:号)?\s*", str(value), re.IGNORECASE)
        return match.group(1) if match else ""

    def _normalize_known_id(self, value: Any) -> str:
        text = str(value).strip()
        if re.fullmatch(r"[pP]\d+", text):
            return "p" + text[1:]
        return text

    def _normalize_player_ref(self, value: Any) -> str:
        text = str(value).strip().strip("()[]{}<>，。,:：")
        if not text:
            return ""
        if re.fullmatch(r"[pP]\d+", text):
            return "p" + text[1:]
        seat = ""
        match = re.fullmatch(r"(?:玩家|选手)?\s*(\d+)号?", text, re.IGNORECASE)
        if match:
            seat = match.group(1)
        elif text.isdigit():
            seat = text
        if seat:
            mapped = self._public_ledger["seat_to_player_id"].get(seat)
            return mapped or text
        return text

    def _normalize_id_set(self, values: set[str]) -> set[str]:
        return {self._normalize_player_ref(value) for value in values if value}

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
                for value in obj[:30]:
                    walk(value)
        walk(payload)
        ids.discard(self.player_id)
        return ids

    def _collect_id_collection(self, mapping: Mapping[str, Any], allowed_keys: set[str]) -> set[str]:
        result: set[str] = set()
        for key, value in mapping.items():
            if str(key).lower() in allowed_keys:
                result.update(self._as_id_set(value))
        return result

    def _as_id_set(self, value: Any) -> set[str]:
        if isinstance(value, (list, tuple, set)):
            return {str(x) for x in value if self._is_scalar_id(x)}
        return {str(value)} if self._is_scalar_id(value) else set()

    def _append_bounded(self, items: list[Any], value: Any, limit: int) -> None:
        if value in items:
            return
        items.append(value)
        if len(items) > limit:
            del items[:len(items) - limit]

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
        return not isinstance(value, bool) and (isinstance(value, int) or (isinstance(value, str) and self._looks_like_player_id(value)))

    def _trim_text(self, text: str, limit: int) -> str:
        normalized = " ".join(str(text).split())
        return normalized if len(normalized) <= limit else normalized[:max(0, limit - 1)] + "…"

    def _dedupe_preserve_order(self, items: list[str]) -> list[str]:
        return list(dict.fromkeys(item for item in items if item))

    def _join_ids(self, ids: list[str], *, limit: int) -> str:
        if not ids:
            return "未知"
        selected = ids[:limit]
        extra = len(ids) - len(selected)
        return "、".join(selected) + (f" 等{extra}人" if extra > 0 else "")

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
    def _player_id_keys() -> set[str]:
        return {"player_id", "speaker_id", "voter_id", "claimer_id", "actor_id", "source_id", "target_id", "from_id", "to_id", "badge_holder", "badge_holder_id", "sheriff_id", "dead_id", "dead_player_id"}

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

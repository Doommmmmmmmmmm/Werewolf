"""最小可运行的角色 Task-Agent。

每次行动从一个新的模型会话开始。模型默认只收到当前状态；本轮对话只能通过唯一的
受限工具按需读取。这里不包含策略进化、长期记忆、外部检索或其他 Harness。
"""

from __future__ import annotations

import asyncio
from collections import Counter, deque
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
_PLAYER_ID_PATTERN = re.compile(r"\b[a-zA-Z]{0,4}\d+\b")

CURRENT_ROUND_DIALOGUE_TOOL_NAME = "read_current_round_dialogue"
DEFAULT_MAX_TOOL_CALLS_PER_DECISION = 5
DEFAULT_MAX_TOOL_RESULT_TOKENS = 1000
DEFAULT_MAX_PROMPT_CHARS = 12000
# 这是 Task-Agent 层的“纠错重试”次数：首次模型调用之外，最多再请求两次。
# 传输层的 HTTP/网络重试仍由 ModelClient 的 MODEL_MAX_RETRIES 单独控制。
DEFAULT_MAX_DECISION_RETRIES = 2
_SPEECH_ACTION_KINDS = frozenset({"speak", "last_words"})
_PASS_LIKE_ACTION_KINDS = frozenset({"pass", "skip", "wait", "idle", "noop", "none"})
_IDENTITY_ROLE_TERMS = ("预言家", "女巫", "守卫", "猎人", "白痴", "平民", "狼人", "狼")
_SELF_CLAIM_MARKERS = ("我是", "我就是", "我跳", "我报", "我认", "我自称", "我来跳", "我来报", "我这把", "我这轮", "我牌是", "claim")
_IDENTITY_CONTRADICTION_MARKERS = ("改口", "矛盾", "对跳", "反水", "卖队", "假", "伪", "反口")
_IDENTITY_CLEAR_MARKERS = ("金水", "查验好人", "已验好", "验好", "确认好人", "查到好人", "验到好人")
DEFAULT_PUBLIC_MEMORY_SIZE = 24


CURRENT_ROUND_DIALOGUE_TOOL: dict[str, Any] = {
    "name": CURRENT_ROUND_DIALOGUE_TOOL_NAME,
    "description": "读取当前昼夜轮次中、当前玩家依法可见的已发生发言。",
    "parameters": {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    },
}


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
        # 外部契约上限在 Agent 边界强制执行，调用方不能通过构造参数放大预算。
        self.max_tokens = min(900, max(1, int(max_tokens)))
        self.max_decision_retries = max(0, int(max_decision_retries))
        self.max_tool_calls_per_decision = min(5, max(0, int(max_tool_calls_per_decision)))
        self.max_tool_result_tokens = min(1000, max(1, int(max_tool_result_tokens)))
        self.max_prompt_chars = min(12000, max(1, int(max_prompt_chars)))
        self._public_memory: deque[str] = deque(maxlen=DEFAULT_PUBLIC_MEMORY_SIZE)
        self._player_notes: dict[str, dict[str, Any]] = {}
        self._round_memory: deque[str] = deque(maxlen=10)
        self._last_hunter_action_summary = ""
        self._model_token_usage: dict[str, int] = {
            "successful_response_count": 0,
            "api_attempt_count": 0,
            "reported_usage_response_count": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }

    async def observe(self, sync_packet: dict[str, Any]) -> None:
        """接收公开同步，并压缩成有限长的短记忆。"""

        if not isinstance(sync_packet, Mapping):
            return
        for event in self._extract_public_events(sync_packet):
            if event:
                self._public_memory.append(event)

    def _recent_memory(self, limit: int = 8) -> list[str]:
        if limit <= 0:
            return []
        combined = list(self._public_memory)
        for item in self._round_memory:
            if item not in combined:
                combined.append(item)
        return combined[-limit:]

    def _decision_brief(
        self,
        turn_packet: Mapping[str, Any],
        current_dialogue: list[Any],
    ) -> dict[str, Any]:
        request = turn_packet["request"]
        allowed_actions = [
            item for item in request.get("allowed_actions", []) if isinstance(item, Mapping)
        ]
        phase = self._request_phase_label(turn_packet, request, allowed_actions)
        recent_memory = self._recent_memory(5)
        # 只摘取少量当前轮事实；需要完整当前轮发言时由模型主动调用受限工具。
        current_records = self._current_decision_records(
            turn_packet,
            current_dialogue,
            round_value=(turn_packet.get("game") or {}).get("round"),
            phase=phase,
        )
        evidence_snapshot = self._evidence_snapshot(recent_memory, extra_records=current_records)
        mode = self._request_mode(allowed_actions)
        brief: dict[str, Any] = {
            "mode": mode,
            "phase": phase,
            "focus": "猎人活着时隐藏身份、稳健站好人位；死亡反应时按硬清避让、冲票/对跳优先。",
            "recent": recent_memory,
            "evidence": evidence_snapshot,
            "current_preview": self._compact_dialogue(current_dialogue, limit=2),
            "current_facts": self._compact_records(current_records, limit=4),
        }
        if mode == "hunter_reaction":
            ranking = self._hunter_target_ranking(
                allowed_actions, recent_memory, turn_packet, current_records=current_records
            )
            brief.update(
                {
                    "shoot_plan": ranking,
                    "rule": "若有推荐目标且目标非引擎硬清，优先开枪；只有全目标证据单薄或均硬清才考虑跳过。",
                }
            )
        elif mode == "last_words":
            brief.update(
                {
                    "actual_hunter_action": self._last_hunter_action_summary or "无上一动作记录；不要声称已开枪。",
                    "say_style": "简短交代票型/对跳依据和好人视野，不虚构。",
                }
            )
        elif mode == "speak":
            brief.update(
                {
                    "say_style": "不要跳猎人或挑衅全场；低调给出票型观察、保留坑位，避免把自己做成焦点。",
                    "current_issue": self._key_contradiction(recent_memory),
                }
            )
        return brief

    @staticmethod
    def _request_mode(allowed_actions: list[Mapping[str, Any]]) -> str:
        kinds = [str(item.get("kind") or "") for item in allowed_actions if item.get("kind")]
        if "hunter_reaction" in kinds:
            return "hunter_reaction"
        if "last_words" in kinds:
            return "last_words"
        if "speak" in kinds:
            return "speak"
        if any(item.get("target_ids") for item in allowed_actions):
            return "hunter_reaction"
        return kinds[0] if kinds else "unknown"

    def _request_phase_label(
        self,
        turn_packet: Mapping[str, Any],
        request: Mapping[str, Any],
        allowed_actions: list[Mapping[str, Any]],
    ) -> str:
        game = turn_packet.get("game") or {}
        phase = game.get("public_phase") or game.get("phase")
        if phase:
            return str(phase)
        mode = self._request_mode(allowed_actions)
        request_phase = request.get("phase")
        if request_phase:
            return str(request_phase)
        return mode

    @staticmethod
    def _compact_text(value: object, limit: int = 48) -> str:
        text = str(value).strip()
        if len(text) <= limit:
            return text
        return text[: max(0, limit - 1)].rstrip() + "…"

    def _compact_dialogue(self, current_dialogue: list[Any], limit: int = 3) -> list[str]:
        if not isinstance(current_dialogue, list) or limit <= 0:
            return []
        preview: list[str] = []
        for item in current_dialogue[-limit:]:
            if isinstance(item, Mapping):
                speaker = str(
                    item.get("speaker")
                    or item.get("player_id")
                    or item.get("from_player_id")
                    or item.get("from")
                    or item.get("name")
                    or "?"
                )
                text = item.get("text", item.get("content", item.get("message", "")))
                compact = self._compact_text(text, limit=48)
                if compact:
                    preview.append(f"{speaker}: {compact}")
            elif isinstance(item, str):
                compact = self._compact_text(item, limit=48)
                if compact:
                    preview.append(compact)
        return preview

    def _recent_controversy(self, recent_memory: list[str]) -> str:
        if not recent_memory:
            return "暂无足够公开记忆"
        return "；".join(recent_memory[-3:])

    def _key_contradiction(self, recent_memory: list[str]) -> str:
        for line in reversed(recent_memory):
            if any(keyword in line for keyword in ("改口", "矛盾", "对跳", "反水", "投票", "死亡", "警长", "冲票")):
                return line
        for player_id in reversed(list(self._player_notes.keys())):
            note = self._player_notes.get(player_id) or {}
            flag = str(note.get("flag") or "").strip()
            if flag:
                return f"{player_id}:{flag}"
        return recent_memory[-1] if recent_memory else "暂无明确矛盾"

    def _player_note(self, player_id: str) -> dict[str, Any]:
        note = self._player_notes.get(player_id)
        if note is None:
            note = {
                "status": "unknown",
                "claim": "",
                "last_vote": "",
                "clear_level": "",  # hard / soft / ""
                "flag": "",  # 最近公开疑点：改口、对跳、查杀、冲票等
            }
            self._player_notes[player_id] = note
        return note

    @staticmethod
    def _unique_strings(values: object) -> list[str]:
        if not isinstance(values, list):
            return []
        seen: set[str] = set()
        result: list[str] = []
        for value in values:
            text = str(value).strip()
            if text and text not in seen:
                seen.add(text)
                result.append(text)
        return result

    @staticmethod
    def _role_looks_clear(role: str) -> bool:
        text = str(role).strip()
        if not text:
            return False
        if any(keyword in text for keyword in ("狼人", "狼", "悍跳", "假", "伪")):
            return False
        return any(keyword in text for keyword in ("预言家", "女巫", "猎人", "守卫", "白痴", "好人", "平民", "民"))

    @staticmethod
    def _player_ids_from_text(text: str) -> list[str]:
        if not text:
            return []
        seen: set[str] = set()
        result: list[str] = []
        for match in _PLAYER_ID_PATTERN.findall(text):
            player_id = str(match).strip()
            if player_id and player_id not in seen:
                seen.add(player_id)
                result.append(player_id)
        return result

    @staticmethod
    def _extract_roles_from_text(text: str) -> list[str]:
        if not text:
            return []
        ordered_terms = _IDENTITY_ROLE_TERMS
        result: list[str] = []
        seen: set[str] = set()
        for term in ordered_terms:
            if term in text and term not in seen:
                if term == "狼" and "狼人" in seen:
                    continue
                seen.add(term)
                result.append(term)
        if "民" in text and "平民" not in seen and any(marker in text for marker in _SELF_CLAIM_MARKERS):
            seen.add("平民")
            result.append("平民")
        return result

    @classmethod
    def _parse_identity_signals(
        cls,
        *,
        actor: str,
        target: str,
        kind: str,
        text: str,
        role: str,
    ) -> dict[str, Any]:
        clean_text = str(text or "").strip()
        lower_kind = str(kind or "").strip().lower()
        mentions = cls._player_ids_from_text(clean_text)
        roles = cls._extract_roles_from_text(clean_text)
        claim_context = lower_kind in {"claim", "reveal", "identity", "open_role", "last_words"}
        if not claim_context and any(marker in clean_text for marker in _SELF_CLAIM_MARKERS):
            claim_context = True
        negated = any(marker in clean_text for marker in ("不是", "没", "无", "不跳", "不认", "不报"))
        if role and cls._role_looks_clear(role):
            roles.append(role)
        roles = cls._unique_strings(roles)

        claims: list[str] = []
        contradictions: list[str] = []
        role_links: list[str] = []
        hard_clear = False
        soft_clear = False

        if claim_context and roles and not negated:
            claims.extend(roles)
            if any(marker in clean_text for marker in ("对跳", "伪", "假", "反水", "矛盾", "改口")):
                contradictions.append(cls._compact_text(clean_text, limit=60))

        if negated and roles:
            contradictions.append(cls._compact_text(clean_text, limit=60))
        if any(marker in clean_text for marker in _IDENTITY_CONTRADICTION_MARKERS):
            contradictions.append(cls._compact_text(clean_text, limit=60))

        if any(marker in clean_text for marker in _IDENTITY_CLEAR_MARKERS):
            soft_clear = True

        if lower_kind in {"reveal", "identity", "open_role"}:
            if role and cls._role_looks_clear(role):
                hard_clear = True
            if clean_text and any(marker in clean_text for marker in _IDENTITY_CLEAR_MARKERS):
                hard_clear = True
            if roles and any(marker in clean_text for marker in _IDENTITY_CLEAR_MARKERS):
                hard_clear = True

        if lower_kind in {"death", "died", "eliminate", "eliminated", "kill", "killed"}:
            if roles and any(marker in clean_text for marker in ("遗言", "最后发言", "最后一句", "临死", "出局")):
                soft_clear = True

        relation_markers = ("查杀", "金水", "对跳", "验到", "验出", "验中", "验好", "认狼", "认好")
        relation_hit = any(marker in clean_text for marker in relation_markers)
        for mentioned_id in mentions:
            if mentioned_id == actor:
                continue
            for found_role in roles:
                if not negated and (relation_hit or claim_context or any(marker in clean_text for marker in ("是", "就是", "认", "报", "跳"))):
                    role_links.append(f"{mentioned_id}:{found_role}")
        if target and target != actor and relation_hit and not negated:
            relation = "查杀" if "查杀" in clean_text or "认狼" in clean_text else "金水" if "金水" in clean_text else "对跳"
            role_links.append(f"{target}:{relation}")

        return {
            "claims": cls._unique_strings(claims),
            "contradictions": cls._unique_strings(contradictions),
            "role_links": cls._unique_strings(role_links),
            "hard_clear": hard_clear,
            "soft_clear": soft_clear,
        }

    def _player_label(self, player_id: str) -> str:
        note = self._player_notes.get(player_id)
        if not note:
            return player_id
        tags: list[str] = []
        if note.get("status") in {"dead", "out"}:
            tags.append("死")
        clear_level = str(note.get("clear_level") or "")
        if clear_level == "hard":
            tags.append("硬清")
        elif clear_level == "soft":
            tags.append("软清")
        claim = str(note.get("claim") or "").strip()
        if claim:
            tags.append(claim)
        last_vote = str(note.get("last_vote") or "").strip()
        if last_vote:
            tags.append(f"票{last_vote}")
        return f"{player_id}({'/'.join(tags)})" if tags else player_id

    def _note_public_record(self, record: Mapping[str, Any]) -> None:
        actor = str(record.get("actor") or "").strip()
        target = str(record.get("target") or "").strip()
        kind = str(record.get("kind") or "").strip().lower()
        text = str(record.get("text") or "").strip()
        role = str(record.get("role") or "").strip()
        summary = self._compact_text(record.get("summary") or text or role, limit=50)
        is_vote = kind in {"vote", "voting"} or bool(record.get("vote"))
        is_death = kind in {"death", "died", "eliminate", "eliminated", "kill", "killed"} or bool(record.get("death"))
        is_reveal = kind in {"reveal", "identity", "open_role"}
        parsed = self._parse_identity_signals(actor=actor, target=target, kind=kind, text=text, role=role)

        if actor:
            actor_note = self._player_note(actor)
            if is_vote and target:
                actor_note["last_vote"] = target
            claim = role or (parsed["claims"][0] if parsed["claims"] else "")
            if claim:
                claim = self._compact_text(claim, limit=18)
                if actor_note.get("claim") and actor_note.get("claim") != claim:
                    actor_note["flag"] = "改口/多身份"
                actor_note["claim"] = claim
            if parsed["contradictions"]:
                actor_note["flag"] = self._compact_text(parsed["contradictions"][-1], limit=36)
            if is_reveal and (role and self._role_looks_clear(role) or parsed["hard_clear"]):
                actor_note["clear_level"] = "hard"
            elif parsed["soft_clear"] and actor_note.get("clear_level") != "hard":
                actor_note["clear_level"] = "soft"
            if text and any(word in text for word in ("冲票", "带节奏", "查杀", "对跳", "假", "悍跳", "狼人", "狼")):
                actor_note["flag"] = summary
            if kind == "sheriff":
                actor_note["flag"] = "警长"

        if target and kind == "badge":
            self._player_note(target)["flag"] = f"接警徽←{actor or '?'}"
        if target and text and any(word in text for word in ("金水", "查验好人", "验好", "确认好人")):
            note = self._player_note(target)
            if note.get("clear_level") != "hard":
                note["clear_level"] = "soft"
        if target and text and any(word in text for word in ("查杀", "对跳", "假", "悍跳", "冲票", "带走", "狼人", "狼")):
            self._player_note(target)["flag"] = summary
        for link in parsed["role_links"][:2]:
            linked_target = link.split(":", 1)[0].strip()
            if linked_target:
                linked_note = self._player_note(linked_target)
                if any(word in link for word in ("金水", "好人")) and linked_note.get("clear_level") != "hard":
                    linked_note["clear_level"] = "soft"
                elif any(word in link for word in ("查杀", "对跳", "狼", "假", "伪")):
                    linked_note["flag"] = self._compact_text(link, limit=36)
        if is_death:
            dead_ids = self._unique_strings(
                [value for value in [*list(record.get("dead_player_ids") or []), target, actor] if value]
            )
            for player_id in dead_ids:
                self._player_note(player_id)["status"] = "dead"

    def _current_decision_records(
        self,
        turn_packet: Mapping[str, Any],
        current_dialogue: list[Any],
        *,
        round_value: object = None,
        phase: str = "",
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        candidates: list[Any] = list(current_dialogue) if isinstance(current_dialogue, list) else []
        request = turn_packet.get("request") or {}
        public_state = turn_packet.get("public_state") or {}
        for container in (request, public_state):
            if not isinstance(container, Mapping):
                continue
            for key in ("events", "public_events", "recent_events", "last_event", "resolution", "vote_results"):
                value = container.get(key)
                if isinstance(value, list):
                    candidates.extend(value[-8:])
                elif isinstance(value, Mapping):
                    candidates.append(value)
        for item in candidates[-20:]:
            for expanded in self._expand_public_record_items(item):
                record = self._normalize_public_record(
                    expanded, default_round=round_value, default_phase=phase
                )
                if record and record.get("summary") and record["summary"] not in {
                    existing.get("summary") for existing in records
                }:
                    records.append(record)
        return records[-12:]

    def _current_dialogue_records(
        self,
        current_dialogue: list[Any],
        *,
        round_value: object = None,
        phase: str = "",
    ) -> list[dict[str, Any]]:
        # 保留一个小的兼容辅助方法，便于旧调用者只提供对白。
        records: list[dict[str, Any]] = []
        for item in current_dialogue if isinstance(current_dialogue, list) else []:
            for expanded in self._expand_public_record_items(item):
                record = self._normalize_public_record(
                    expanded, default_round=round_value, default_phase=phase
                )
                if record and record.get("summary"):
                    records.append(record)
        return records[-12:]

    def _compact_records(self, records: list[Mapping[str, Any]], limit: int = 8) -> list[str]:
        result: list[str] = []
        for record in records[-max(0, limit):]:
            summary = str(record.get("summary") or "").strip()
            if summary and summary not in result:
                result.append(summary)
        return result

    def _evidence_snapshot(
        self,
        recent_memory: list[str],
        *,
        extra_records: list[Mapping[str, Any]] | None = None,
    ) -> dict[str, list[str]]:
        notes: dict[str, dict[str, Any]] = {pid: dict(note) for pid, note in self._player_notes.items()}
        if extra_records:
            # 本轮证据只进入本次快照，避免深拷贝整段历史和跨轮重复膨胀。
            original_notes = self._player_notes
            self._player_notes = notes
            try:
                for record in extra_records[-8:]:
                    self._note_public_record(record)
            finally:
                notes = self._player_notes
                self._player_notes = original_notes
        vote_pressure: Counter[str] = Counter()
        claims: list[str] = []
        flags: list[str] = []
        hard_clears: list[str] = []
        soft_clears: list[str] = []
        for player_id, note in notes.items():
            last_vote = str(note.get("last_vote") or "").strip()
            if last_vote:
                vote_pressure[last_vote] += 1
            claim = str(note.get("claim") or "").strip()
            if claim:
                claims.append(f"{player_id}:{claim}")
            flag = str(note.get("flag") or "").strip()
            if flag:
                flags.append(f"{player_id}:{self._compact_text(flag, limit=30)}")
            if note.get("clear_level") == "hard":
                hard_clears.append(player_id)
            elif note.get("clear_level") == "soft":
                soft_clears.append(player_id)
        return {
            "vote_pressure": [f"{player_id}×{count}" for player_id, count in vote_pressure.most_common(3)],
            "claims": claims[-5:],
            "flags": flags[-5:],
            "hard_clears": hard_clears[:5],
            "soft_clears": soft_clears[:5],
            "recent": recent_memory[-3:],
        }

    def _compact_public_state(self, public_state: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(public_state, Mapping):
            return {}
        compact: dict[str, Any] = {}
        for key in ("phase", "public_phase", "round", "sheriff_id", "leader_id"):
            if key in public_state:
                compact[key] = public_state[key]
        players = public_state.get("players")
        items = players.values() if isinstance(players, Mapping) else players if isinstance(players, list) else []
        compact_players: list[dict[str, Any]] = []
        for item in items:
            if not isinstance(item, Mapping):
                continue
            player_id = str(item.get("player_id") or item.get("id") or item.get("name") or "").strip()
            if not player_id:
                continue
            player_info: dict[str, Any] = {"id": player_id}
            for source_key, out_key in (
                ("alive", "alive"), ("status", "status"), ("revealed_role", "role"),
                ("role_revealed", "role"), ("identity", "role"),
            ):
                value = item.get(source_key)
                if value not in (None, "") and out_key not in player_info:
                    player_info[out_key] = value
            compact_players.append(player_info)
        if compact_players:
            compact["players"] = compact_players
        return compact

    def _public_player_facts(self, turn_packet: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
        facts: dict[str, dict[str, Any]] = {}
        public_state = turn_packet.get("public_state") or {}
        players = public_state.get("players") if isinstance(public_state, Mapping) else None
        items = players.values() if isinstance(players, Mapping) else players if isinstance(players, list) else []
        for item in items:
            if not isinstance(item, Mapping):
                continue
            player_id = str(item.get("player_id") or item.get("id") or item.get("name") or "").strip()
            if not player_id:
                continue
            role = str(item.get("revealed_role") or item.get("role_revealed") or item.get("identity") or "").strip()
            raw_alive = item.get("alive")
            if raw_alive is None:
                raw_alive = item.get("status") not in {"dead", "out", "eliminated"}
            alive = not (
                raw_alive is False
                or str(raw_alive).strip().lower() in {"false", "dead", "out", "eliminated"}
            )
            facts[player_id] = {
                "alive": alive,
                "revealed_role": role,
                "hard_clear": bool(role and self._role_looks_clear(role)),
            }
        return facts

    def _hunter_target_ranking(
        self,
        allowed_actions: list[Mapping[str, Any]],
        recent_memory: list[str],
        turn_packet: Mapping[str, Any],
        *,
        current_records: list[Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        target_ids: list[str] = []
        for action in allowed_actions:
            raw_targets = action.get("target_ids")
            if isinstance(raw_targets, list):
                for target in raw_targets:
                    target_id = str(target).strip()
                    if target_id and target_id not in target_ids:
                        target_ids.append(target_id)
        facts = self._public_player_facts(turn_packet)
        target_ids = [target for target in target_ids if facts.get(target, {}).get("alive", True)]
        skip_option = self._pass_like_action_kind(allowed_actions)

        votes_on_me: Counter[str] = Counter()
        votes_by_target: Counter[str] = Counter()
        attacks_by_target: Counter[str] = Counter()
        recent_text = "\n".join(recent_memory[-5:])
        for player_id, note in self._player_notes.items():
            last_vote = str(note.get("last_vote") or "").strip()
            if last_vote == self.player_id:
                votes_on_me[player_id] += 1
            if last_vote:
                votes_by_target[last_vote] += 1
            flag = str(note.get("flag") or "")
            if flag and any(word in flag for word in ("冲票", "查杀", "对跳", "悍跳", "假", "狼")):
                attacks_by_target[player_id] += 1
        for record in current_records or []:
            actor = str(record.get("actor") or "").strip()
            target = str(record.get("target") or "").strip()
            kind = str(record.get("kind") or "").lower()
            text = str(record.get("text") or "")
            if kind in {"vote", "voting"} and actor:
                if target == self.player_id:
                    votes_on_me[actor] += 1
                if target:
                    votes_by_target[target] += 1
            if actor and any(word in text for word in ("冲票", "归票", "出猎人", "投猎人", "查杀", "对跳", "悍跳", "带走")):
                attacks_by_target[actor] += 1

        claim_counts = Counter(
            str(note.get("claim") or "").strip()
            for note in self._player_notes.values()
            if str(note.get("claim") or "").strip()
        )
        ranking: list[dict[str, Any]] = []
        for target_id in target_ids:
            note = self._player_notes.get(target_id) or {}
            hard_clear = bool(facts.get(target_id, {}).get("hard_clear")) or note.get("clear_level") == "hard"
            soft_clear = note.get("clear_level") == "soft"
            claim = str(note.get("claim") or "").strip()
            flag = str(note.get("flag") or "").strip()
            score = 0.0
            reasons: list[str] = []
            if hard_clear:
                score -= 99
                reasons.append("硬清避让")
            if soft_clear:
                score -= 2
                reasons.append("金水/软清")
            if votes_on_me[target_id]:
                score += 4
                reasons.append("投/冲猎人")
            pressure = votes_by_target[target_id]
            if pressure >= 2:
                score += 2 + min(pressure, 4) * 0.5
                reasons.append(f"票型焦点×{pressure}")
            if attacks_by_target[target_id]:
                score += 3
                reasons.append("带头归票/对跳攻击")
            if claim and claim_counts[claim] >= 2:
                score += 2.5
                reasons.append(f"对跳{claim}")
            if flag:
                score += 1.2
                reasons.append(self._compact_text(flag, limit=28))
            if target_id in recent_text and any(word in recent_text for word in ("冲票", "归票", "对跳", "查杀", "狼", "假")):
                score += 0.8
                reasons.append("近期争议命中")
            ranking.append({"target_id": target_id, "score": round(score, 2), "reasons": reasons[:3] or ["证据单薄"]})
        ranking.sort(key=lambda item: item["score"], reverse=True)
        recommended = ranking[0]["target_id"] if ranking and ranking[0]["score"] >= 2.5 else None
        return {
            "target_ranking": ranking[:3],
            "shoot_recommended_target": recommended,
            "forced_recommendation": recommended,
            "skip_option": skip_option,
            "skip_recommended": bool(skip_option and not recommended),
            "explicit_instruction": (
                f"优先开枪 {recommended}；该目标符合冲票/对跳/归票核心证据且非硬清。"
                if recommended else
                "没有非硬清的核心冲票/对跳目标；可谨慎跳过或选最高分合法目标。"
            ),
        }

    @staticmethod
    def _pass_like_action_kind(allowed_actions: list[Mapping[str, Any]]) -> str:
        for action in allowed_actions:
            kind = str(action.get("kind") or "").lower()
            if kind in _PASS_LIKE_ACTION_KINDS:
                return kind
        return ""

    @staticmethod
    def _canonical_event_kind(value: object) -> str:
        raw = str(value or "").strip()
        upper = raw.upper()
        mapped = {
            "PLAYER_SPOKE": "speak",
            "PLAYER_PASSED": "pass",
            "VOTE_CAST": "vote",
            "SHERIFF_VOTE_CAST": "vote",
            "DAY_VOTE_RESOLVED": "vote",
            "PLAYER_ELIMINATED": "death",
            "PLAYER_DIED": "death",
            "PLAYER_DEATH": "death",
            "PLAYER_LAST_WORDS": "last_words",
            "SHERIFF_BADGE_TRANSFERRED": "badge",
            "SHERIFF_ELECTED": "sheriff",
        }
        return mapped.get(upper, raw.lower())

    def _expand_public_record_items(self, item: object) -> list[object]:
        """将批量投票事件拆成可进入票账的单票公开记录。"""
        if not isinstance(item, Mapping):
            return [item]
        payload = item.get("payload")
        source = payload if isinstance(payload, Mapping) else item
        raw_votes = source.get("votes") or source.get("vote_records") or source.get("vote_results")
        if not raw_votes:
            return [item]
        vote_items: list[tuple[object, object]] = []
        if isinstance(raw_votes, Mapping):
            for voter, vote in raw_votes.items():
                if isinstance(vote, Mapping):
                    target = (
                        vote.get("target_id") or vote.get("voted_player_id")
                        or vote.get("votee") or vote.get("target") or vote.get("vote")
                    )
                else:
                    target = vote
                vote_items.append((voter, target))
        elif isinstance(raw_votes, list):
            for vote in raw_votes:
                if not isinstance(vote, Mapping):
                    continue
                voter = (
                    vote.get("voter_id") or vote.get("player_id")
                    or vote.get("actor") or vote.get("from_player_id")
                )
                target = (
                    vote.get("target_id") or vote.get("voted_player_id")
                    or vote.get("votee") or vote.get("target") or vote.get("vote")
                )
                if voter or target:
                    vote_items.append((voter, target))
        if not vote_items:
            return [item]
        expanded: list[object] = []
        for voter, target in vote_items:
            record = dict(item)
            merged_payload = dict(payload) if isinstance(payload, Mapping) else {}
            merged_payload.update({"voter_id": voter, "target_id": target, "vote": target})
            record["payload"] = merged_payload
            record["type"] = "VOTE_CAST"
            expanded.append(record)
        return expanded

    def _normalize_public_record(
        self,
        item: object,
        *,
        default_round: object = None,
        default_phase: object = None,
    ) -> dict[str, Any] | None:
        if isinstance(item, str):
            text = item.strip()
            if not text:
                return None
            return {
                "round": default_round, "phase": default_phase, "actor": "", "target": "",
                "kind": "text", "text": text, "role": "", "vote": None, "death": None,
                "dead_player_ids": [], "summary": self._compact_text(text, limit=72),
            }
        if not isinstance(item, Mapping):
            return None
        payload = item.get("payload")
        merged: dict[str, Any] = dict(item)
        if isinstance(payload, Mapping):
            # payload 字段优先，包装层的 type/round/phase/channel 作为上下文保留。
            merged.update(payload)
        round_value = merged.get("round", item.get("round", default_round))
        phase_value = merged.get("phase", merged.get("public_phase", item.get("phase", default_phase)))
        outer_type = str(item.get("type") or "").strip().upper()
        payload_type = str(payload.get("type") or "").strip() if isinstance(payload, Mapping) else ""
        event_type = (
            payload_type
            if outer_type in {"EVENT", "PUBLIC_EVENT"} and payload_type
            else item.get("type") or merged.get("kind") or merged.get("action") or merged.get("type")
        )
        kind = self._canonical_event_kind(event_type)
        actor = str(
            merged.get("speaker") or merged.get("player_id") or merged.get("voter_id")
            or merged.get("from_player_id") or merged.get("from") or merged.get("actor")
            or merged.get("source") or ""
        ).strip()
        target = str(
            merged.get("target_id") or merged.get("target") or merged.get("votee")
            or merged.get("voted_player_id") or merged.get("to_player_id")
            or merged.get("recipient_player_id") or merged.get("receiver_id") or ""
        ).strip()
        dead_ids = (
            merged.get("dead_player_ids") or merged.get("deceased_player_ids")
            or merged.get("dead_player_id") or (merged.get("player_ids") if kind == "death" else [])
            or []
        )
        if isinstance(dead_ids, str):
            dead_ids = [dead_ids]
        dead_ids = self._unique_strings(dead_ids) if isinstance(dead_ids, list) else []
        if not target and dead_ids:
            target = dead_ids[0]
        text = str(merged.get("text") or merged.get("content") or merged.get("message") or "").strip()
        role = str(
            merged.get("role") or merged.get("revealed_role") or merged.get("role_revealed")
            or merged.get("identity") or ""
        ).strip()
        vote = merged.get("vote")
        if not kind:
            kind = "speak" if text else "vote" if vote is not None else ""
        if not target and vote is not None and kind in {"vote", "voting"}:
            target = str(vote).strip()
        normalized = dict(merged)
        normalized.update({
            "round": round_value, "phase": phase_value, "actor": actor, "target": target,
            "kind": kind, "text": text, "role": role, "vote": vote,
            "death": merged.get("death") or kind == "death", "dead_player_ids": dead_ids,
        })
        summary = self._summarize_public_event(normalized, default_round=round_value, default_phase=phase_value)
        if not summary:
            return None
        normalized["summary"] = summary
        return normalized

    def _extract_public_events(self, sync_packet: Mapping[str, Any]) -> list[str]:
        events: list[str] = []
        seen: set[str] = set()
        round_groups: dict[str, list[dict[str, Any]]] = {}
        default_round = sync_packet.get("round")
        default_phase = sync_packet.get("phase") or sync_packet.get("public_phase")

        def build_record(item: object) -> dict[str, Any] | None:
            return self._normalize_public_record(
                item, default_round=default_round, default_phase=default_phase
            )

        def add_event(item: object) -> None:
            for expanded in self._expand_public_record_items(item):
                record = build_record(expanded)
                if not record:
                    continue
                event = record["summary"]
                if not event or event in seen:
                    continue
                seen.add(event)
                self._note_public_record(record)
                events.append(event)
                group_key = f"r{record['round']}" if record.get("round") is not None else "r?"
                phase_key = str(record.get("phase") or "?")
                round_groups.setdefault(f"{group_key}/{phase_key}", []).append(record)
                if len(events) >= 10:
                    break

        for key in ("events", "public_events", "log", "logs", "history", "records", "timeline"):
            value = sync_packet.get(key)
            if isinstance(value, list):
                for item in value[-10:]:
                    add_event(item)
                    if len(events) >= 10:
                        break
            if len(events) >= 10:
                break
        if len(events) < 10:
            for key in ("public_state", "state", "game"):
                value = sync_packet.get(key)
                if value is not None:
                    add_event(value)
                    if len(events) >= 10:
                        break
        if len(events) < 10:
            def walk(item: object, depth: int) -> None:
                if len(events) >= 10 or depth > 2:
                    return
                if isinstance(item, Mapping):
                    if self._looks_like_public_event(item):
                        add_event(item)
                    else:
                        for value in item.values():
                            if isinstance(value, (Mapping, list, tuple)):
                                walk(value, depth + 1)
                    return
                if isinstance(item, (list, tuple)):
                    for value in item:
                        walk(value, depth + 1)
                        if len(events) >= 10:
                            return

            walk(sync_packet, 0)
        for group_key, group in round_groups.items():
            summary = self._summarize_round_group(group_key, group)
            if summary and summary not in seen:
                seen.add(summary)
                events.append(summary)
                self._round_memory.append(summary)
            if len(events) >= 12:
                break
        return events

    @staticmethod
    def _looks_like_public_event(item: Mapping[str, Any]) -> bool:
        return any(
            key in item
            for key in (
                "speaker",
                "player_id",
                "from_player_id",
                "voter_id",
                "voted_player_id",
                "to_player_id",
                "target_id",
                "payload",
                "vote",
                "votes",
                "kind",
                "action",
                "type",
                "text",
                "content",
                "message",
                "role",
                "death",
                "died",
                "alive",
                "status",
            )
        )

    @staticmethod
    def _summarize_public_event(
        item: object,
        *,
        default_round: object = None,
        default_phase: object = None,
    ) -> str | None:
        if isinstance(item, str):
            text = item.strip()
            if not text:
                return None
            return TaskAgent._compact_text(text, limit=72)
        if not isinstance(item, Mapping):
            return None

        round_value = item.get("round", default_round)
        phase_value = item.get("phase", item.get("public_phase", default_phase))
        prefix_parts: list[str] = []
        if round_value is not None:
            prefix_parts.append(f"r{round_value}")
        if phase_value is not None:
            prefix_parts.append(str(phase_value))
        prefix = "[" + "/".join(prefix_parts) + "] " if prefix_parts else ""

        actor = str(
            item.get("speaker")
            or item.get("player_id")
            or item.get("from_player_id")
            or item.get("from")
            or item.get("actor")
            or item.get("source")
            or ""
        ).strip()
        target = str(item.get("target_id") or item.get("target") or item.get("votee") or "").strip()
        kind = str(item.get("kind") or item.get("action") or item.get("type") or "").strip().lower()
        text = str(item.get("text") or item.get("content") or item.get("message") or "").strip()
        role = str(item.get("role") or item.get("revealed_role") or item.get("identity") or "").strip()

        parts: list[str] = []
        if actor:
            parts.append(actor)
        if kind in {"speak", "speech", "say", "dialogue", "statement"} or (text and kind not in {"last_words", "vote", "death"}):
            if text:
                parts.append(f"发言:{TaskAgent._compact_text(text, limit=36)}")
        elif kind == "last_words":
            if text:
                parts.append(f"遗言:{TaskAgent._compact_text(text, limit=36)}")
        elif kind in {"vote", "voting"} or item.get("vote"):
            if target:
                parts.append(f"投票→{target}")
            elif item.get("vote"):
                parts.append(f"投票→{TaskAgent._compact_text(item.get('vote'), limit=24)}")
        elif kind in {"death", "died", "eliminate", "eliminated", "kill", "killed"} or item.get("death"):
            if target:
                parts.append(f"死亡→{target}")
            else:
                parts.append("发生死亡")
        elif kind in {"reveal", "identity", "open_role", "claim"} or role:
            claim = role or text or "身份公开"
            parts.append(f"身份:{TaskAgent._compact_text(claim, limit=28)}")
        elif kind == "badge":
            parts.append(f"警徽→{target or '?'}")
        elif kind == "sheriff":
            parts.append("公开当选警长")
        elif kind == "pass":
            parts.append("跳过")
        else:
            for key in ("vote", "votes", "result", "status", "event"):
                value = item.get(key)
                if value is not None:
                    parts.append(f"{key}:{TaskAgent._compact_text(value, limit=28)}")
                    break
            if not parts and text:
                parts.append(TaskAgent._compact_text(text, limit=36))

        if not parts:
            return None
        if prefix:
            return prefix + " ".join(parts)
        return " ".join(parts)

    def _summarize_round_group(self, group_key: str, group: list[dict[str, Any]]) -> str | None:
        pieces: list[str] = []
        for record in group:
            actor = str(record.get("actor") or "").strip()
            target = str(record.get("target") or "").strip()
            kind = str(record.get("kind") or "").strip().lower()
            text = str(record.get("text") or "").strip()
            role = str(record.get("role") or "").strip()
            if kind in {"vote", "voting"} or record.get("vote"):
                if actor or target:
                    pieces.append(f"{actor or '?'}→{target or str(record.get('vote') or '?')}")
            elif kind in {"death", "died", "eliminate", "eliminated", "kill", "killed"} or record.get("death"):
                pieces.append(f"{target or actor or '?'}死亡")
            elif kind in {"reveal", "identity", "open_role", "claim"} or role:
                pieces.append(f"{actor or '?'}:{self._compact_text(role or text or '身份', limit=18)}")
            elif text and any(keyword in text for keyword in ("改口", "矛盾", "对跳", "反水", "查杀", "金水", "站边", "投票")):
                pieces.append(f"{actor or '?'}:{self._compact_text(text, limit=22)}")
            if len(pieces) >= 3:
                break
        pieces = self._unique_strings(pieces)
        if not pieces:
            return None
        return f"[{group_key}] " + "；".join(pieces[:3])

    async def decide(self, turn_packet: dict[str, Any]) -> dict[str, Any]:
        if turn_packet["self"]["player_id"] != self.player_id:
            raise ValueError("行动包被发送给了错误玩家")
        private = turn_packet["private_information"]
        if str(private.get("role")) != self.profile.role:
            raise ValueError("Task-Agent 的角色档案与行动包身份不一致")

        system = self._system_prompt(private)

        tool_context = turn_packet.get("tool_context") or {}
        current_dialogue = tool_context.get("current_round_dialogue") or []
        prompt = {
            "game": turn_packet["game"],
            "self": turn_packet["self"],
            "private_information": private,
            "public_state": self._compact_public_state(turn_packet.get("public_state") or {}),
            "request": turn_packet["request"],
            "decision_brief": self._decision_brief(turn_packet, current_dialogue),
            "history_policy": {
                "default_context": "current_state_only",
                "current_round_dialogue_tool": CURRENT_ROUND_DIALOGUE_TOOL_NAME,
                "max_tool_calls": self.max_tool_calls_per_decision,
                "max_tool_result_tokens": self.max_tool_result_tokens,
            },
        }

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
                "instruction": self._turn_instruction(turn_packet["request"], feedback),
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
                request_mode = self._request_mode([
                    item for item in turn_packet["request"].get("allowed_actions", [])
                    if isinstance(item, Mapping)
                ])
                if request_mode == "hunter_reaction" and str(action.get("kind") or "").lower() not in {
                    "vote", "sheriff_vote", "sheriff_vote_cast"
                }:
                    self._record_hunter_action(action)
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

    def _record_hunter_action(self, action: Mapping[str, Any]) -> None:
        kind = str(action.get("kind") or "").strip().lower()
        target = str(action.get("target_id") or "").strip()
        if kind in _PASS_LIKE_ACTION_KINDS or not target:
            self._last_hunter_action_summary = f"实际猎人动作：{kind or 'pass'}（没有带走目标）。"
        else:
            self._last_hunter_action_summary = f"实际猎人动作：{kind}→{target}（已提交目标）。"

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

    @staticmethod
    def _turn_instruction(request: Mapping[str, Any], feedback: str) -> str:
        return render_prompt(
            "player_turn_instruction.txt",
            action_contract=render_action_contract(request),
            validation_feedback=(f"上一次输出未通过校验：{feedback}" if feedback else ""),
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

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
        self.max_tokens = max_tokens
        self.max_decision_retries = max(0, int(max_decision_retries))
        self.max_tool_calls_per_decision = max(0, int(max_tool_calls_per_decision))
        self.max_tool_result_tokens = max(1, int(max_tool_result_tokens))
        self.max_prompt_chars = max(1, int(max_prompt_chars))
        self._public_memory: deque[str] = deque(maxlen=DEFAULT_PUBLIC_MEMORY_SIZE)
        self._player_notes: dict[str, dict[str, Any]] = {}
        self._round_book: dict[str, dict[str, deque[str]]] = {}
        self._round_memory: deque[str] = deque(maxlen=10)
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
        recent_memory = self._recent_memory(8)
        evidence_snapshot = self._evidence_snapshot(recent_memory)
        top_suspects = self._top_suspects(recent_memory, turn_packet, limit=4)
        round_book = self._round_book_preview(limit=3)
        brief: dict[str, Any] = {
            "phase": phase,
            "recent_public_signals": recent_memory,
            "recent_round_book": round_book,
            "evidence_snapshot": evidence_snapshot,
            "current_round_dialogue_preview": self._compact_dialogue(current_dialogue, limit=3),
            "top_suspects": top_suspects,
        }
        mode = self._request_mode(allowed_actions)
        if mode == "hunter_reaction":
            ranking = self._hunter_target_ranking(allowed_actions, recent_memory, turn_packet)
            brief.update(
                {
                    "decision_mode": "hunter_reaction",
                    "focus": "先比票型、改口、身份声明和死亡链条；只有多源证据都很弱时才考虑跳过。",
                    "required_output": "在合法 target 中优先选多源证据最强者；只有全部都弱才允许 skip。",
                    "target_ranking": ranking["target_ranking"],
                    "skip_option": ranking["skip_option"],
                    "skip_condition": ranking["skip_condition"],
                    "top_vote_pressure": evidence_snapshot["top_vote_pressure"],
                    "top_repeat_vote_pressure": evidence_snapshot["top_repeat_vote_pressure"],
                    "top_contradictions": evidence_snapshot["top_contradictions"],
                    "claimed_roles": evidence_snapshot["claimed_roles"],
                    "hard_clears": evidence_snapshot["hard_clears"],
                }
            )
        elif mode == "last_words":
            brief.update(
                {
                    "decision_mode": "last_words",
                    "focus": "收束到已公开的票型、身份冲突、改口与死亡顺序，先列证据再下结论。",
                    "required_output": "必须点名一个具体怀疑对象，并给出一条具体证据链。",
                    "current_controversy": self._recent_controversy(recent_memory),
                    "key_contradiction": self._key_contradiction(recent_memory),
                    "top_vote_pressure": evidence_snapshot["top_vote_pressure"],
                    "top_repeat_vote_pressure": evidence_snapshot["top_repeat_vote_pressure"],
                    "top_contradictions": evidence_snapshot["top_contradictions"],
                    "claimed_roles": evidence_snapshot["claimed_roles"],
                    "hard_clears": evidence_snapshot["hard_clears"],
                }
            )
        elif mode == "speak":
            brief.update(
                {
                    "decision_mode": "speak",
                    "focus": "回应当前争议，优先点出票型变化、身份声明矛盾和谁在保谁。",
                    "required_output": "必须点名一个具体怀疑对象，并给出一条具体证据链。",
                    "current_controversy": self._recent_controversy(recent_memory),
                    "key_contradiction": self._key_contradiction(recent_memory),
                    "top_vote_pressure": evidence_snapshot["top_vote_pressure"],
                    "top_repeat_vote_pressure": evidence_snapshot["top_repeat_vote_pressure"],
                    "top_contradictions": evidence_snapshot["top_contradictions"],
                    "claimed_roles": evidence_snapshot["claimed_roles"],
                    "hard_clears": evidence_snapshot["hard_clears"],
                }
            )
        else:
            brief.update(
                {
                    "decision_mode": mode,
                    "focus": "遵守当前阶段的合法行动范围，并尽量保持公开信息一致。",
                    "evidence_snapshot": evidence_snapshot,
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
            if any(keyword in line for keyword in ("改口", "矛盾", "对跳", "反水", "自相矛盾", "投票", "死亡", "警长")):
                return line
        for player_id in reversed(list(self._player_notes.keys())):
            note = self._player_notes.get(player_id) or {}
            contradictions = list(note.get("contradictions") or [])
            if contradictions:
                return f"{player_id}:{contradictions[-1]}"
        return recent_memory[-1] if recent_memory else "暂无明确矛盾"

    def _round_key(self, round_value: object, phase_value: object) -> str:
        parts: list[str] = []
        if round_value is not None:
            parts.append(f"r{round_value}")
        if phase_value is not None:
            parts.append(str(phase_value))
        return "/".join(parts) if parts else "r?"

    def _round_bucket(self, round_value: object, phase_value: object) -> dict[str, deque[str]]:
        key = self._round_key(round_value, phase_value)
        bucket = self._round_book.get(key)
        if bucket is None:
            bucket = {
                "votes": deque(maxlen=4),
                "claims": deque(maxlen=4),
                "reveals": deque(maxlen=4),
                "deaths": deque(maxlen=4),
                "supports": deque(maxlen=4),
                "attacks": deque(maxlen=4),
                "contradictions": deque(maxlen=4),
                "summary": deque(maxlen=4),
            }
            self._round_book[key] = bucket
        return bucket

    def _round_book_preview(self, limit: int = 3) -> list[str]:
        if limit <= 0:
            return []
        items = list(self._round_book.items())[-limit:]
        preview: list[str] = []
        for key, bucket in items:
            pieces: list[str] = []
            for field in ("votes", "claims", "reveals", "deaths", "supports", "attacks", "contradictions"):
                values = list(bucket.get(field) or [])
                if values:
                    pieces.append(f"{field}:{self._compact_text(values[-1], limit=28)}")
            summary = list(bucket.get("summary") or [])
            if summary:
                pieces.insert(0, self._compact_text(summary[-1], limit=36))
            if pieces:
                preview.append(f"[{key}] " + "；".join(pieces[:4]))
        return preview

    def _top_suspects(
        self,
        recent_memory: list[str],
        turn_packet: Mapping[str, Any],
        limit: int = 4,
    ) -> list[str]:
        del turn_packet
        candidates: list[dict[str, Any]] = []
        for player_id in self._player_notes:
            if player_id == self.player_id:
                continue
            note = self._player_notes.get(player_id) or {}
            if note.get("status") in {"dead", "out"}:
                continue
            score_info = self._score_hunter_target(player_id, recent_memory, {})
            candidates.append(
                {
                    "player_id": player_id,
                    "score": round(score_info["score"], 2),
                    "reasons": score_info["reasons"],
                }
            )
        candidates.sort(key=lambda item: item["score"], reverse=True)
        return [
            f"{self._player_label(item['player_id'])}:{item['score']:.2f}({'/'.join(item['reasons'][:2])})"
            for item in candidates[:limit]
        ]

    def _player_note(self, player_id: str) -> dict[str, Any]:
        note = self._player_notes.get(player_id)
        if note is None:
            note = {
                "speeches": deque(maxlen=4),
                "claims": deque(maxlen=4),
                "revealed_roles": deque(maxlen=3),
                "vote_history": deque(maxlen=8),
                "attack_lines": deque(maxlen=4),
                "support_lines": deque(maxlen=4),
                "contradictions": deque(maxlen=4),
                "death_links": deque(maxlen=4),
                "attack_targets": deque(maxlen=6),
                "support_targets": deque(maxlen=6),
                "claim_history": deque(maxlen=6),
                "reveal_history": deque(maxlen=6),
                "latest_vote": "",
                "status": "unknown",
                "hard_clear": False,
                "soft_clear": False,
                "last_round": None,
                "last_phase": None,
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

    def _player_label(self, player_id: str) -> str:
        note = self._player_notes.get(player_id)
        if not note:
            return player_id
        tags: list[str] = []
        if note.get("status") in {"dead", "out"}:
            tags.append("死")
        if note.get("hard_clear"):
            tags.append("硬清")
        elif note.get("soft_clear"):
            tags.append("软清")
        claims = self._unique_strings(list(note.get("claims") or []) + list(note.get("revealed_roles") or []))
        if claims:
            tags.append("/".join(claims[:2]))
        latest_vote = str(note.get("latest_vote") or "").strip()
        if latest_vote:
            tags.append(f"票{latest_vote}")
        return f"{player_id}({'/'.join(tags)})" if tags else player_id

    def _note_public_record(self, record: Mapping[str, Any]) -> None:
        actor = str(record.get("actor") or "").strip()
        target = str(record.get("target") or "").strip()
        kind = str(record.get("kind") or "").strip().lower()
        text = str(record.get("text") or "").strip()
        role = str(record.get("role") or "").strip()
        round_value = record.get("round")
        phase_value = record.get("phase")
        summary = str(record.get("summary") or "").strip()
        mentioned_ids = self._player_ids_from_text(text)
        related_ids = self._unique_strings([value for value in [actor, target, *mentioned_ids] if value])
        speech_kind = kind in {"speak", "speech", "say", "dialogue", "statement", "text"}
        is_vote = kind in {"vote", "voting"} or record.get("vote") is not None
        is_death = kind in {"death", "died", "eliminate", "eliminated", "kill", "killed"} or record.get("death") is not None
        is_claim = kind == "claim"
        is_reveal = kind in {"reveal", "identity", "open_role"}
        negative_words = ("改口", "矛盾", "对跳", "反水", "卖队", "带节奏", "假", "骗", "冲票", "踩", "查杀", "自相矛盾")
        positive_words = ("可信", "一致", "解释清楚", "对得上", "稳定", "金水", "好人", "站对", "硬清")
        contradiction_words = ("改口", "矛盾", "对跳", "反水", "自相矛盾")
        round_bucket = self._round_bucket(round_value, phase_value)
        for player_id in related_ids:
            note = self._player_note(player_id)
            note["last_round"] = round_value
            note["last_phase"] = phase_value
        if actor:
            actor_note = self._player_note(actor)
            if speech_kind and text:
                actor_note["speeches"].append(self._compact_text(text, limit=56))
            if is_vote:
                if target:
                    actor_note["latest_vote"] = target
                    actor_note["vote_history"].append(target)
                    round_bucket["votes"].append(self._compact_text(summary or f"{actor}→{target}", limit=48))
            if is_claim:
                previous_claims = self._unique_strings(list(actor_note["claims"]))
                if role:
                    if previous_claims and role not in previous_claims:
                        contradiction = self._compact_text(summary or text or role, limit=60)
                        actor_note["contradictions"].append(contradiction)
                        round_bucket["contradictions"].append(contradiction)
                    claim_text = self._compact_text(role, limit=24)
                    actor_note["claims"].append(claim_text)
                    actor_note["claim_history"].append(claim_text)
                    round_bucket["claims"].append(self._compact_text(summary or f"{actor}:{claim_text}", limit=48))
                if text and text != role:
                    claim_text = self._compact_text(text, limit=32)
                    actor_note["claims"].append(claim_text)
                    actor_note["claim_history"].append(claim_text)
            if is_reveal:
                previous_roles = self._unique_strings(list(actor_note["revealed_roles"]))
                if role:
                    if previous_roles and role not in previous_roles:
                        contradiction = self._compact_text(summary or text or role, limit=60)
                        actor_note["contradictions"].append(contradiction)
                        round_bucket["contradictions"].append(contradiction)
                    reveal_text = self._compact_text(role, limit=24)
                    actor_note["revealed_roles"].append(reveal_text)
                    actor_note["reveal_history"].append(reveal_text)
                    round_bucket["reveals"].append(self._compact_text(summary or f"{actor}:{reveal_text}", limit=48))
                    if self._role_looks_clear(role):
                        actor_note["hard_clear"] = True
                if text and any(keyword in text for keyword in ("金水", "查验好人", "已验好", "确认好人")):
                    actor_note["hard_clear"] = True
            if text and any(keyword in text for keyword in contradiction_words):
                contradiction = self._compact_text(summary or text, limit=60)
                actor_note["contradictions"].append(contradiction)
                round_bucket["contradictions"].append(contradiction)
            if text and any(keyword in text for keyword in ("金水", "可信", "好人", "站边", "支持")):
                actor_note["soft_clear"] = True
        if is_death:
            dead_ids = self._unique_strings([value for value in [target, actor, *mentioned_ids] if value])
            for player_id in dead_ids:
                note = self._player_note(player_id)
                note["status"] = "dead"
                death_text = self._compact_text(summary or text or f"{player_id}死亡", limit=60)
                note["death_links"].append(death_text)
                round_bucket["deaths"].append(death_text)
        if text:
            summary_text = self._compact_text(summary or text, limit=64)
            if any(keyword in text for keyword in negative_words) and target:
                self._player_note(target)["attack_lines"].append(summary_text)
                self._player_note(target)["attack_targets"].append(target)
                round_bucket["attacks"].append(summary_text)
            if any(keyword in text for keyword in positive_words) and target:
                self._player_note(target)["support_lines"].append(summary_text)
                self._player_note(target)["support_targets"].append(target)
                round_bucket["supports"].append(summary_text)
            for player_id in mentioned_ids:
                if player_id == actor:
                    continue
                mention_note = self._player_note(player_id)
                if any(keyword in text for keyword in negative_words):
                    mention_note["attack_lines"].append(summary_text)
                    mention_note["attack_targets"].append(player_id)
                if any(keyword in text for keyword in positive_words):
                    mention_note["support_lines"].append(summary_text)
                    mention_note["support_targets"].append(player_id)
                mention_note["last_round"] = round_value
                mention_note["last_phase"] = phase_value
            round_bucket["summary"].append(summary_text)
        if target and is_vote:
            vote_text = self._compact_text(summary or f"{actor}投{target}", limit=60)
            self._player_note(target)["attack_lines"].append(vote_text)
            self._player_note(target)["attack_targets"].append(target)
        if target and is_death:
            self._player_note(target)["death_links"].append(self._compact_text(summary or text or f"{target}死亡", limit=60))

    def _evidence_snapshot(self, recent_memory: list[str]) -> dict[str, list[str]]:
        vote_pressure: Counter[str] = Counter()
        repeat_vote_pressure: Counter[str] = Counter()
        role_claims: dict[str, list[str]] = {}
        top_contradictions: list[str] = []
        hard_clears: list[str] = []
        attack_pressure: Counter[str] = Counter()
        support_pressure: Counter[str] = Counter()
        for player_id, note in self._player_notes.items():
            latest_vote = str(note.get("latest_vote") or "").strip()
            if latest_vote:
                vote_pressure[latest_vote] += 1
            for vote_target in list(note.get("vote_history") or []):
                target = str(vote_target or "").strip()
                if target:
                    repeat_vote_pressure[target] += 1
            for attack_target in list(note.get("attack_targets") or []):
                target = str(attack_target or "").strip()
                if target:
                    attack_pressure[target] += 1
            for support_target in list(note.get("support_targets") or []):
                target = str(support_target or "").strip()
                if target:
                    support_pressure[target] += 1
            contradictions = list(note.get("contradictions") or [])
            if contradictions:
                top_contradictions.append(
                    f"{self._player_label(player_id)}:{self._compact_text(contradictions[-1], limit=40)}"
                )
            claims = self._unique_strings(list(note.get("claims") or []) + list(note.get("revealed_roles") or []))
            for role in claims:
                role_claims.setdefault(role, []).append(player_id)
            if note.get("hard_clear"):
                hard_clears.append(self._player_label(player_id))
        claimed_roles: list[str] = []
        for role, players in sorted(role_claims.items(), key=lambda item: (-len(item[1]), item[0])):
            claimed_roles.append(f"{role}:{'/'.join(players[:3])}")
            if len(claimed_roles) >= 4:
                break
        return {
            "top_vote_pressure": [f"{player_id}×{count}" for player_id, count in vote_pressure.most_common(3)],
            "top_repeat_vote_pressure": [f"{player_id}×{count}" for player_id, count in repeat_vote_pressure.most_common(3)],
            "top_attack_pressure": [f"{player_id}×{count}" for player_id, count in attack_pressure.most_common(3)],
            "top_support_pressure": [f"{player_id}×{count}" for player_id, count in support_pressure.most_common(3)],
            "top_contradictions": top_contradictions[:3],
            "claimed_roles": claimed_roles[:4],
            "hard_clears": hard_clears[:4],
            "recent_signals": recent_memory[-4:],
        }

    def _hunter_target_ranking(
        self,
        allowed_actions: list[Mapping[str, Any]],
        recent_memory: list[str],
        turn_packet: Mapping[str, Any],
    ) -> dict[str, Any]:
        target_ids: list[str] = []
        for action in allowed_actions:
            raw_targets = action.get("target_ids")
            if not isinstance(raw_targets, list):
                continue
            for target in raw_targets:
                target_id = str(target)
                if target_id and target_id not in target_ids:
                    target_ids.append(target_id)
        ranking: list[dict[str, Any]] = []
        for target_id in target_ids:
            score_info = self._score_hunter_target(target_id, recent_memory, turn_packet)
            ranking.append(
                {
                    "target_id": target_id,
                    "score": round(score_info["score"], 2),
                    "reasons": score_info["reasons"],
                    "signals": score_info["signals"],
                }
            )
        ranking.sort(key=lambda item: item["score"], reverse=True)
        skip_option = self._pass_like_action_kind(allowed_actions)
        best_score = ranking[0]["score"] if ranking else 0.0
        strongest_signals = max(
            (
                sum(1 for value in item.get("signals", {}).values() if isinstance(value, int) and value > 0)
                for item in ranking
            ),
            default=0,
        )
        if skip_option:
            if strongest_signals >= 2 or best_score >= 1.8:
                skip_condition = "只有当所有合法目标都只剩单一薄弱线索、且没有改口/对跳/死亡链条冲突时才可跳过。"
            else:
                skip_condition = "只要出现票压、身份冲突、死亡链或持续站边中的任意两类，就优先出手而不是跳过。"
        else:
            skip_condition = "没有跳过动作时，必须从合法目标里选风险最低且证据最弱的那个。"
        return {
            "target_ranking": ranking[:4],
            "skip_option": skip_option,
            "skip_condition": skip_condition,
        }

    def _score_hunter_target(
        self,
        target_id: str,
        recent_memory: list[str],
        turn_packet: Mapping[str, Any],
    ) -> dict[str, Any]:
        pattern = re.compile(rf"(?<![A-Za-z0-9]){re.escape(target_id)}(?![A-Za-z0-9])")
        score = 0.0
        reasons: list[str] = []
        signals = {
            "vote_pressure": 0,
            "repeat_vote_pressure": 0,
            "ally_vote_pressure": 0,
            "contradictions": 0,
            "claim_conflict": 0,
            "death_links": 0,
            "clear_hits": 0,
            "recent_mentions": 0,
            "support_pressure": 0,
            "attack_pressure": 0,
        }
        vote_pressure = 0
        repeat_vote_pressure = 0
        support_pressure = 0
        attack_pressure = 0
        for player_id, note in self._player_notes.items():
            if str(note.get("latest_vote") or "").strip() == target_id:
                vote_pressure += 1
            vote_history = Counter(str(vote or "").strip() for vote in list(note.get("vote_history") or []))
            repeat_vote_pressure += max(0, vote_history.get(target_id, 0) - 1)
            if player_id == target_id:
                continue
            if target_id in {str(item or "").strip() for item in list(note.get("support_targets") or [])}:
                support_pressure += 1
            if target_id in {str(item or "").strip() for item in list(note.get("attack_targets") or [])}:
                attack_pressure += 1
            for line in list(note.get("attack_lines") or [])[-2:]:
                if pattern.search(line):
                    signals["recent_mentions"] += 1
            for line in list(note.get("support_lines") or [])[-1:]:
                if pattern.search(line):
                    signals["recent_mentions"] += 1
        if vote_pressure:
            signals["vote_pressure"] = vote_pressure
            score += 0.55 + min(vote_pressure, 4) * 0.5
            reasons.append(f"票压{vote_pressure}")
        if repeat_vote_pressure:
            signals["repeat_vote_pressure"] = repeat_vote_pressure
            score += 0.25 * min(repeat_vote_pressure, 4)
            reasons.append(f"重复站边{repeat_vote_pressure}")
        if support_pressure:
            signals["ally_vote_pressure"] = support_pressure
            signals["support_pressure"] = support_pressure
            score += 0.2 * min(support_pressure, 4)
            reasons.append(f"持续同盟{support_pressure}")
        if attack_pressure:
            signals["attack_pressure"] = attack_pressure
            score += 0.3 * min(attack_pressure, 4)
            reasons.append(f"持续攻击{attack_pressure}")
        target_note = self._player_notes.get(target_id) or {}
        contradictions = list(target_note.get("contradictions") or [])
        if contradictions:
            signals["contradictions"] = len(contradictions)
            score += 0.9 + 0.4 * min(len(contradictions), 3)
            reasons.append("改口/对跳:" + self._compact_text(contradictions[-1], limit=48))
        death_links = list(target_note.get("death_links") or [])
        if death_links:
            signals["death_links"] = len(death_links)
            score += 0.8 + 0.3 * min(len(death_links), 3)
            reasons.append("死亡链:" + self._compact_text(death_links[-1], limit=48))
        target_claims = self._unique_strings(list(target_note.get("claims") or []) + list(target_note.get("revealed_roles") or []))
        if len(target_claims) >= 2:
            signals["claim_conflict"] = len(target_claims)
            score += 1.0
            reasons.append("多次身份声明:" + "/".join(target_claims[:2]))
        if target_claims:
            role_claim_counts: dict[str, list[str]] = {}
            for player_id, note in self._player_notes.items():
                claims = self._unique_strings(list(note.get("claims") or []) + list(note.get("revealed_roles") or []))
                for role in claims:
                    role_claim_counts.setdefault(role, []).append(player_id)
            for role in target_claims:
                claimers = role_claim_counts.get(role, [])
                if len(claimers) >= 2:
                    signals["claim_conflict"] = max(signals["claim_conflict"], len(claimers))
                    score += 0.6 + 0.2 * min(len(claimers), 4)
                    reasons.append(f"对跳{role}:{'/'.join(claimers[:3])}")
                    break
        recent_hits = 0
        for line in recent_memory:
            if not pattern.search(line):
                continue
            recent_hits += 1
            if any(keyword in line for keyword in ("改口", "矛盾", "对跳", "反水", "卖队", "带节奏", "假", "骗", "冲票", "踩", "查杀", "自相矛盾")):
                score += 0.75
                reasons.append(self._compact_text(line, limit=60))
            elif any(keyword in line for keyword in ("投票", "票型", "站边", "出局", "死亡", "身份", "警长", "发言", "表态")):
                score += 0.18
            elif any(keyword in line for keyword in ("可信", "一致", "解释清楚", "对得上", "稳定", "金水", "好人")):
                score -= 0.45
        signals["recent_mentions"] += recent_hits
        if recent_hits and not reasons:
            reasons.append(self._compact_text(f"最近记忆命中{recent_hits}次", limit=60))
        if target_note.get("hard_clear"):
            signals["clear_hits"] += 1
            score -= 2.2
            reasons.append("公开硬清")
        if target_note.get("soft_clear"):
            signals["clear_hits"] += 1
            score -= 1.0
            reasons.append("软清/正向背书")
        if target_note.get("status") in {"dead", "out"}:
            score -= 0.8
            reasons.append("已出局")
        if not reasons:
            reasons.append("公开证据单薄")
        if signals["vote_pressure"] == 0 and signals["contradictions"] == 0 and signals["claim_conflict"] == 0 and signals["death_links"] == 0:
            score = max(0.0, score - 0.8)
        if signals["attack_pressure"] >= 2:
            score += 0.2
        if signals["support_pressure"] >= 2:
            score -= 0.2
        return {"score": score, "reasons": reasons[:4], "signals": signals}

    @staticmethod
    def _pass_like_action_kind(allowed_actions: list[Mapping[str, Any]]) -> str:
        for action in allowed_actions:
            kind = str(action.get("kind") or "").lower()
            if kind in _PASS_LIKE_ACTION_KINDS:
                return kind
        return ""

    def _extract_public_events(self, sync_packet: Mapping[str, Any]) -> list[str]:
        events: list[str] = []
        seen: set[str] = set()
        round_groups: dict[str, list[dict[str, Any]]] = {}
        default_round = sync_packet.get("round")
        default_phase = sync_packet.get("phase") or sync_packet.get("public_phase")

        def build_record(item: object) -> dict[str, Any] | None:
            if isinstance(item, str):
                text = item.strip()
                if not text:
                    return None
                summary = self._compact_text(text, limit=72)
                return {
                    "round": default_round,
                    "phase": default_phase,
                    "actor": "",
                    "target": "",
                    "kind": "text",
                    "text": text,
                    "role": "",
                    "vote": None,
                    "death": None,
                    "summary": summary,
                }
            if not isinstance(item, Mapping):
                return None
            round_value = item.get("round", default_round)
            phase_value = item.get("phase", item.get("public_phase", default_phase))
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
            vote = item.get("vote")
            if not target and vote is not None and kind in {"vote", "voting"}:
                target = str(vote).strip()
            summary = self._summarize_public_event(item, default_round=default_round, default_phase=default_phase)
            if not summary:
                return None
            return {
                "round": round_value,
                "phase": phase_value,
                "actor": actor,
                "target": target,
                "kind": kind,
                "text": text,
                "role": role,
                "vote": vote,
                "death": item.get("death"),
                "summary": summary,
            }

        def add_event(item: object) -> None:
            record = build_record(item)
            if not record:
                return
            event = record["summary"]
            if not event or event in seen:
                return
            seen.add(event)
            self._note_public_record(record)
            events.append(event)
            group_key = f"r{record['round']}" if record.get("round") is not None else "r?"
            phase_key = str(record.get("phase") or "?")
            round_groups.setdefault(f"{group_key}/{phase_key}", []).append(record)

        for key in ("events", "public_events", "log", "logs", "history", "records", "timeline"):
            value = sync_packet.get(key)
            if isinstance(value, list):
                for item in value:
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
                "target_id",
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
        if kind in {"speak", "speech", "say", "dialogue", "statement"} or text:
            if text:
                parts.append(f"发言:{TaskAgent._compact_text(text, limit=36)}")
        elif kind in {"vote", "voting"} or item.get("vote") is not None:
            if target:
                parts.append(f"投票→{target}")
            elif item.get("vote") is not None:
                parts.append(f"投票→{TaskAgent._compact_text(item.get('vote'), limit=24)}")
        elif kind in {"death", "died", "eliminate", "eliminated", "kill", "killed"} or item.get("death") is not None:
            if target:
                parts.append(f"死亡→{target}")
            else:
                parts.append("发生死亡")
        elif kind in {"reveal", "identity", "open_role", "claim"} or role:
            claim = role or text or "身份公开"
            parts.append(f"身份:{TaskAgent._compact_text(claim, limit=28)}")
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
            if kind in {"vote", "voting"} or record.get("vote") is not None:
                if actor or target:
                    pieces.append(f"{actor or '?'}→{target or str(record.get('vote') or '?')}")
            elif kind in {"death", "died", "eliminate", "eliminated", "kill", "killed"} or record.get("death") is not None:
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
            "public_rules": turn_packet["public_rules"],
            "self": turn_packet["self"],
            "private_information": private,
            "public_state": turn_packet["public_state"],
            "request": turn_packet["request"],
            "memory_brief": self._recent_memory(8),
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

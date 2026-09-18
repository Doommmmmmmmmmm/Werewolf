"""最小可运行的角色 Task-Agent。

每次行动从一个新的模型会话开始。模型默认只收到当前状态；本轮对话只能通过唯一的
受限工具按需读取。这里不包含策略进化、长期记忆、外部检索或其他 Harness。
"""

from __future__ import annotations

import asyncio
from collections import deque
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
# 这是 Task-Agent 层的“纠错重试”次数：首次模型调用之外，最多再请求两次。
# 传输层的 HTTP/网络重试仍由 ModelClient 的 MODEL_MAX_RETRIES 单独控制。
DEFAULT_MAX_DECISION_RETRIES = 2
_SPEECH_ACTION_KINDS = frozenset({"speak", "last_words"})
_PASS_LIKE_ACTION_KINDS = frozenset({"pass", "skip", "wait", "idle", "noop", "none"})
DEFAULT_PUBLIC_MEMORY_SIZE = 12
DEFAULT_MAX_TRACKED_PLAYERS = 12
DEFAULT_DOSSIER_MAX_ENTRIES = 3
PLAYER_ID_PATTERN = re.compile(r"(?<![A-Za-z0-9])(p\d+)(?![A-Za-z0-9])")


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
        self._player_dossiers: dict[str, dict[str, Any]] = {}
        self._dossier_seq = 0
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
        self._prune_player_dossiers()

    def _recent_memory(self, limit: int = 5) -> list[str]:
        if limit <= 0:
            return []
        return list(self._public_memory)[-limit:]

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
        evidence_snapshot = self._evidence_snapshot(recent_memory)
        brief: dict[str, Any] = {
            "phase": phase,
            "recent_public_signals": recent_memory,
            "current_round_dialogue_preview": self._compact_dialogue(current_dialogue, limit=3),
            "evidence_snapshot": evidence_snapshot,
        }
        mode = self._request_mode(allowed_actions)
        if mode == "hunter_reaction":
            ranking = self._hunter_target_ranking(allowed_actions, recent_memory)
            brief.update(
                {
                    "decision_mode": "hunter_reaction",
                    "focus": "先看 evidence_snapshot 里的票压、改口、身份冲突和硬清；证据不够就跳过。",
                    "target_ranking": ranking["target_ranking"],
                    "skip_option": ranking["skip_option"],
                    "skip_condition": ranking["skip_condition"],
                    "suspicion_focus": ranking.get("suspicion_focus", {}),
                }
            )
        elif mode == "last_words":
            brief.update(
                {
                    "decision_mode": "last_words",
                    "focus": "收束到已公开的票型、身份声明和死亡链条，只点一个具体怀疑对象和一个依据。",
                    "current_controversy": self._recent_controversy(recent_memory),
                    "key_contradiction": self._key_contradiction(recent_memory),
                    "suspicion_focus": self._top_suspicion_focus(recent_memory),
                }
            )
        elif mode == "speak":
            brief.update(
                {
                    "decision_mode": "speak",
                    "focus": "回应当前争议，必须点名一个具体怀疑对象和一个具体依据，避免空话。",
                    "current_controversy": self._recent_controversy(recent_memory),
                    "key_contradiction": self._key_contradiction(recent_memory),
                    "suspicion_focus": self._top_suspicion_focus(recent_memory),
                }
            )
        else:
            brief.update(
                {
                    "decision_mode": mode,
                    "focus": "遵守当前阶段的合法行动范围，并尽量保持公开信息一致。",
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
        snapshot = self._evidence_snapshot(recent_memory, limit=2)
        parts: list[str] = []
        if snapshot["top_contradictions"]:
            parts.extend(item["summary"] for item in snapshot["top_contradictions"][:2])
        if not parts and recent_memory:
            parts.extend(recent_memory[-2:])
        if not parts:
            return "暂无足够公开记忆"
        return "；".join(parts[:2])

    def _key_contradiction(self, recent_memory: list[str]) -> str:
        snapshot = self._evidence_snapshot(recent_memory, limit=1)
        if snapshot["top_contradictions"]:
            return snapshot["top_contradictions"][0]["summary"]
        for line in reversed(recent_memory):
            if any(keyword in line for keyword in ("改口", "矛盾", "对跳", "反水", "自相矛盾", "投票", "死亡", "警长")):
                return line
        return recent_memory[-1] if recent_memory else "暂无明确矛盾"

    def _top_suspicion_focus(self, recent_memory: list[str]) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        for player_id, dossier in self._player_dossiers.items():
            if player_id == self.player_id:
                continue
            rows.append(self._player_dossier_snapshot(player_id, dossier))
        if rows:
            rows.sort(
                key=lambda item: (
                    item["score"],
                    item["vote_pressure"],
                    item["contradiction_score"],
                    item["mention_count"],
                ),
                reverse=True,
            )
            top = rows[0]
            return {
                "player_id": top["player_id"],
                "score": top["score"],
                "reasons": [top["summary"]][:2],
                "evidence": {
                    "summary": top["summary"],
                    "vote_pressure": top["vote_pressure"],
                    "contradiction_score": top["contradiction_score"],
                    "claims": top["claims"],
                    "hard_clear": top["hard_clear"],
                    "soft_clear": top["soft_clear"],
                },
            }
        ranking = self._hunter_target_ranking([], recent_memory)
        if ranking["target_ranking"]:
            top = ranking["target_ranking"][0]
            return {
                "player_id": top["target_id"],
                "score": top["score"],
                "reasons": top["reasons"][:2],
                "evidence": top["evidence"],
            }
        for line in reversed(recent_memory):
            if line:
                return {"player_id": "", "score": 0.0, "reasons": [line[:60]]}
        return {}

    def _ensure_player_dossier(self, player_id: str) -> dict[str, Any]:
        dossier = self._player_dossiers.get(player_id)
        if dossier is None:
            dossier = {
                "last_seen_seq": 0,
                "last_round": "",
                "mention_count": 0,
                "vote_switches": 0,
                "contradictions": 0,
                "recent_speeches": deque(maxlen=DEFAULT_DOSSIER_MAX_ENTRIES),
                "claimed_roles": deque(maxlen=DEFAULT_DOSSIER_MAX_ENTRIES),
                "vote_out": deque(maxlen=DEFAULT_DOSSIER_MAX_ENTRIES + 1),
                "vote_in": deque(maxlen=DEFAULT_DOSSIER_MAX_ENTRIES + 1),
                "support_lines": deque(maxlen=DEFAULT_DOSSIER_MAX_ENTRIES),
                "attack_lines": deque(maxlen=DEFAULT_DOSSIER_MAX_ENTRIES),
                "support_mentions": deque(maxlen=DEFAULT_DOSSIER_MAX_ENTRIES),
                "attack_mentions": deque(maxlen=DEFAULT_DOSSIER_MAX_ENTRIES),
                "hard_clears": deque(maxlen=DEFAULT_DOSSIER_MAX_ENTRIES),
                "soft_clears": deque(maxlen=DEFAULT_DOSSIER_MAX_ENTRIES),
                "death_links": deque(maxlen=DEFAULT_DOSSIER_MAX_ENTRIES),
            }
            self._player_dossiers[player_id] = dossier
        return dossier

    def _touch_player_dossier(self, player_id: str, round_value: object = None) -> dict[str, Any]:
        dossier = self._ensure_player_dossier(player_id)
        self._dossier_seq += 1
        dossier["last_seen_seq"] = self._dossier_seq
        if round_value is not None:
            dossier["last_round"] = self._compact_text(round_value, limit=12)
        return dossier

    def _prune_player_dossiers(self) -> None:
        excess = len(self._player_dossiers) - DEFAULT_MAX_TRACKED_PLAYERS
        if excess <= 0:
            return
        removable = sorted(
            self._player_dossiers.items(),
            key=lambda item: (
                int(item[1].get("last_seen_seq", 0)),
                int(item[1].get("mention_count", 0))
                + int(item[1].get("vote_switches", 0))
                + int(item[1].get("contradictions", 0)),
            ),
        )
        for player_id, _ in removable[:excess]:
            self._player_dossiers.pop(player_id, None)

    @staticmethod
    def _extract_player_ids(text: str) -> list[str]:
        if not text:
            return []
        seen: list[str] = []
        for match in PLAYER_ID_PATTERN.findall(text):
            if match not in seen:
                seen.append(match)
        return seen

    @staticmethod
    def _extract_claim_from_text(text: str) -> str:
        if not text:
            return ""
        claim_markers = ("我是", "我跳", "我认", "我报", "自称", "claim", "报身份", "身份是")
        if not any(marker in text for marker in claim_markers):
            return ""
        for role in ("预言家", "女巫", "猎人", "守卫", "白痴", "狼人", "村民", "平民", "警长", "好人"):
            if role in text:
                return role
        return ""

    @staticmethod
    def _public_text_signals(text: str) -> dict[str, bool]:
        support = any(keyword in text for keyword in ("站边", "保", "保一下", "保护", "认好", "支持", "偏好", "信他", "放过"))
        attack = any(keyword in text for keyword in ("踩", "打", "冲票", "查杀", "怀疑", "可疑", "卖队", "反水", "矛盾", "对跳", "假身份", "骗", "抗推"))
        hard_clear = any(keyword in text for keyword in ("金水", "硬清", "铁好", "查验好人", "验好", "确定好人"))
        soft_clear = any(keyword in text for keyword in ("像好人", "暂放", "先放", "不急出", "先不出", "偏好", "大概率好", "好人面", "暂时放"))
        contradiction = any(keyword in text for keyword in ("改口", "矛盾", "对跳", "反水", "前后不一", "自相矛盾", "打脸", "变票"))
        death = any(keyword in text for keyword in ("死亡", "出局", "被刀", "被杀", "倒牌", "翻牌"))
        return {
            "support": support and not attack,
            "attack": attack and not support,
            "hard_clear": hard_clear,
            "soft_clear": soft_clear and not hard_clear,
            "contradiction": contradiction,
            "death": death,
        }

    def _update_player_dossiers_from_public_item(
        self,
        item: object,
        *,
        default_round: object = None,
        default_phase: object = None,
    ) -> None:
        if not isinstance(item, Mapping):
            return

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
        line = self._summarize_public_event(item, default_round=default_round, default_phase=default_phase) or ""
        mentioned_ids = self._extract_player_ids(text)
        signals = self._public_text_signals(text)

        tracked_ids: list[str] = []
        for player_id in (actor, target, *mentioned_ids):
            if player_id and player_id not in tracked_ids:
                tracked_ids.append(player_id)
        for player_id in tracked_ids:
            self._touch_player_dossier(player_id, round_value=round_value)

        if actor:
            actor_dossier = self._touch_player_dossier(actor, round_value=round_value)
            if text and kind in {"speak", "speech", "say", "dialogue", "statement", "talk"}:
                actor_dossier["recent_speeches"].append(self._compact_text(text, limit=56))
            if role:
                role_text = self._compact_text(role, limit=20)
                if role_text and role_text not in actor_dossier["claimed_roles"]:
                    actor_dossier["claimed_roles"].append(role_text)
            claim = self._extract_claim_from_text(text)
            if claim and claim not in actor_dossier["claimed_roles"]:
                actor_dossier["claimed_roles"].append(claim)
            if signals["contradiction"]:
                actor_dossier["contradictions"] += 1
            if signals["support"] and line:
                actor_dossier["support_lines"].append(self._compact_text(line, limit=56))
            if signals["attack"] and line:
                actor_dossier["attack_lines"].append(self._compact_text(line, limit=56))
            if signals["death"] and line:
                actor_dossier["death_links"].append(self._compact_text(line, limit=48))

        if kind in {"vote", "voting"} and actor and target:
            actor_dossier = self._touch_player_dossier(actor, round_value=round_value)
            target_dossier = self._touch_player_dossier(target, round_value=round_value)
            if actor_dossier["vote_out"] and actor_dossier["vote_out"][-1] != target:
                actor_dossier["vote_switches"] += 1
                actor_dossier["contradictions"] += 1
            actor_dossier["vote_out"].append(target)
            target_dossier["vote_in"].append(actor)

        if target and line:
            target_dossier = self._touch_player_dossier(target, round_value=round_value)
            if signals["hard_clear"]:
                target_dossier["hard_clears"].append(self._compact_text(line, limit=56))
            elif signals["soft_clear"]:
                target_dossier["soft_clears"].append(self._compact_text(line, limit=56))
            if signals["support"]:
                target_dossier["support_mentions"].append(self._compact_text(line, limit=56))
            if signals["attack"]:
                target_dossier["attack_mentions"].append(self._compact_text(line, limit=56))
            if signals["death"]:
                target_dossier["death_links"].append(self._compact_text(line, limit=48))
            if text:
                target_dossier["mention_count"] += 1

        if not line:
            return
        for player_id in mentioned_ids:
            dossier = self._touch_player_dossier(player_id, round_value=round_value)
            dossier["mention_count"] += 1
            if signals["hard_clear"]:
                dossier["hard_clears"].append(self._compact_text(line, limit=56))
            elif signals["soft_clear"]:
                dossier["soft_clears"].append(self._compact_text(line, limit=56))
            if signals["support"]:
                dossier["support_mentions"].append(self._compact_text(line, limit=56))
            if signals["attack"]:
                dossier["attack_mentions"].append(self._compact_text(line, limit=56))
            if signals["death"]:
                dossier["death_links"].append(self._compact_text(line, limit=48))
            if phase_value is not None:
                dossier["last_round"] = self._compact_text(f"{round_value}/{phase_value}", limit=12)

    def _player_dossier_snapshot(self, player_id: str, dossier: Mapping[str, Any]) -> dict[str, Any]:
        claimed_roles = self._unique_tail(list(dossier.get("claimed_roles", [])), limit=2)
        vote_out = self._unique_tail(list(dossier.get("vote_out", [])), limit=3)
        vote_in = self._unique_tail(list(dossier.get("vote_in", [])), limit=3)
        support_mentions = list(dossier.get("support_mentions", []))[-2:]
        attack_mentions = list(dossier.get("attack_mentions", []))[-2:]
        support_lines = list(dossier.get("support_lines", []))[-2:]
        attack_lines = list(dossier.get("attack_lines", []))[-2:]
        hard_clears = list(dossier.get("hard_clears", []))[-2:]
        soft_clears = list(dossier.get("soft_clears", []))[-2:]
        death_links = list(dossier.get("death_links", []))[-2:]
        claim_conflict = max(0, len(claimed_roles) - 1)
        vote_pressure = len(vote_in) + len(attack_mentions)
        contradiction_score = int(dossier.get("contradictions", 0)) + int(dossier.get("vote_switches", 0)) + claim_conflict
        hard_clear = bool(hard_clears)
        soft_clear = bool(soft_clears)
        support_pressure = len(support_mentions) + len(support_lines)
        attack_pressure = len(attack_mentions) + len(attack_lines)
        risk_score = vote_pressure + contradiction_score + attack_pressure - support_pressure
        if hard_clear:
            risk_score -= 3
        elif soft_clear:
            risk_score -= 1
        summary_bits: list[str] = []
        if vote_pressure:
            summary_bits.append(f"{vote_pressure}票压")
        if contradiction_score:
            summary_bits.append(f"{contradiction_score}次冲突")
        if claimed_roles:
            summary_bits.append("身份:" + "/".join(claimed_roles))
        if hard_clear:
            summary_bits.append("硬清")
        elif soft_clear:
            summary_bits.append("软清")
        if death_links:
            summary_bits.append("死亡相关")
        summary = "；".join(summary_bits[:3]) if summary_bits else "公开证据稀薄"
        return {
            "player_id": player_id,
            "summary": summary,
            "score": round(float(risk_score), 2),
            "vote_pressure": vote_pressure,
            "contradiction_score": contradiction_score,
            "claims": claimed_roles,
            "vote_out": vote_out,
            "vote_in": vote_in,
            "support_mentions": support_mentions,
            "attack_mentions": attack_mentions,
            "support_lines": support_lines,
            "attack_lines": attack_lines,
            "hard_clear": hard_clear,
            "soft_clear": soft_clear,
            "death_links": death_links,
            "last_round": str(dossier.get("last_round") or ""),
            "mention_count": int(dossier.get("mention_count", 0)),
        }

    def _evidence_snapshot(self, recent_memory: list[str], limit: int = 3) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        for player_id, dossier in self._player_dossiers.items():
            rows.append(self._player_dossier_snapshot(player_id, dossier))
        if not rows:
            fallback = [
                {"player_id": "", "summary": line, "score": 0.0}
                for line in recent_memory[-limit:]
                if line
            ]
            return {
                "top_contradictions": fallback[:limit],
                "top_vote_pressure": [],
                "claimed_roles": [],
                "hard_clears": [],
                "soft_clears": [],
            }

        contradictions = sorted(
            [item for item in rows if item["contradiction_score"] > 0],
            key=lambda item: (item["contradiction_score"], item["vote_pressure"], item["score"]),
            reverse=True,
        )[:limit]
        vote_pressure = sorted(
            [item for item in rows if item["vote_pressure"] > 0],
            key=lambda item: (item["vote_pressure"], item["contradiction_score"], item["score"]),
            reverse=True,
        )[:limit]
        claimed_roles = [
            {
                "player_id": item["player_id"],
                "claims": item["claims"],
                "summary": item["summary"],
            }
            for item in sorted(rows, key=lambda item: (len(item["claims"]), item["score"]), reverse=True)
            if item["claims"]
        ][:limit]
        hard_clears = [
            {"player_id": item["player_id"], "summary": item["summary"]}
            for item in rows
            if item["hard_clear"]
        ][:limit]
        soft_clears = [
            {"player_id": item["player_id"], "summary": item["summary"]}
            for item in rows
            if item["soft_clear"] and not item["hard_clear"]
        ][:limit]
        return {
            "top_contradictions": [
                {"player_id": item["player_id"], "summary": item["summary"], "score": item["score"]}
                for item in contradictions
            ],
            "top_vote_pressure": [
                {"player_id": item["player_id"], "summary": item["summary"], "score": item["score"]}
                for item in vote_pressure
            ],
            "claimed_roles": claimed_roles,
            "hard_clears": hard_clears,
            "soft_clears": soft_clears,
        }

    @staticmethod
    def _unique_tail(values: list[str], limit: int) -> list[str]:
        if limit <= 0:
            return []
        seen: list[str] = []
        for value in values[-(limit * 2) :]:
            if value and value not in seen:
                seen.append(value)
        return seen[-limit:]

    def _hunter_target_ranking(
        self,
        allowed_actions: list[Mapping[str, Any]],
        recent_memory: list[str],
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
            score, reasons = self._score_hunter_target(target_id, recent_memory)
            dossier = self._player_dossiers.get(target_id)
            evidence = self._player_dossier_snapshot(target_id, dossier) if dossier else {
                "player_id": target_id,
                "summary": "公开证据稀薄",
                "score": 0.0,
                "vote_pressure": 0,
                "contradiction_score": 0,
                "claims": [],
                "hard_clear": False,
                "soft_clear": False,
            }
            ranking.append(
                {
                    "target_id": target_id,
                    "score": round(score, 2),
                    "reasons": reasons,
                    "evidence": {
                        "summary": evidence["summary"],
                        "vote_pressure": evidence.get("vote_pressure", 0),
                        "contradiction_score": evidence.get("contradiction_score", 0),
                        "claims": evidence.get("claims", []),
                        "hard_clear": evidence.get("hard_clear", False),
                        "soft_clear": evidence.get("soft_clear", False),
                    },
                }
            )
        ranking.sort(key=lambda item: item["score"], reverse=True)
        skip_option = self._pass_like_action_kind(allowed_actions)
        best_score = ranking[0]["score"] if ranking else 0.0
        second_score = ranking[1]["score"] if len(ranking) > 1 else None
        if skip_option:
            if best_score < 1.8:
                skip_condition = "证据仍偏软：优先跳过，等票压、改口和身份冲突更硬再出手。"
            elif second_score is not None and best_score - second_score < 0.5 and best_score < 2.6:
                skip_condition = "候选之间差距不够，优先跳过以降低误伤。"
            else:
                skip_condition = "只有当某个目标同时具备票压、改口和身份冲突时才出手。"
        else:
            skip_condition = "没有跳过动作时，优先选证据最硬且硬清风险最低的目标。"
        suspicion_focus = {}
        if ranking:
            top = ranking[0]
            suspicion_focus = {
                "player_id": top["target_id"],
                "score": top["score"],
                "reasons": top["reasons"][:2],
                "evidence": top["evidence"],
            }
        return {
            "target_ranking": ranking[:4],
            "skip_option": skip_option,
            "skip_condition": skip_condition,
            "suspicion_focus": suspicion_focus,
        }

    def _score_hunter_target(self, target_id: str, recent_memory: list[str]) -> tuple[float, list[str]]:
        dossier = self._player_dossiers.get(target_id)
        score = 0.0
        reasons: list[str] = []
        if dossier is not None:
            snapshot = self._player_dossier_snapshot(target_id, dossier)
            if snapshot["hard_clear"]:
                score -= 4.0
                reasons.append("已有硬清/强保")
            if snapshot["soft_clear"]:
                score -= 1.5
                reasons.append("存在软清")
            if snapshot["vote_pressure"]:
                vote_bonus = min(2.4, 0.6 * snapshot["vote_pressure"])
                score += vote_bonus
                reasons.append(f"{snapshot['vote_pressure']}点投票/攻击压力")
            if snapshot["contradiction_score"]:
                contradiction_bonus = min(3.0, 0.8 * snapshot["contradiction_score"])
                score += contradiction_bonus
                reasons.append(f"{snapshot['contradiction_score']}点改口/冲突")
            if snapshot["attack_mentions"] or snapshot["attack_lines"]:
                attack_bonus = min(1.6, 0.4 * (len(snapshot["attack_mentions"]) + len(snapshot["attack_lines"])))
                score += attack_bonus
                reasons.append("存在攻击线")
            if snapshot["support_mentions"] or snapshot["support_lines"]:
                support_bonus = min(1.2, 0.35 * (len(snapshot["support_mentions"]) + len(snapshot["support_lines"])))
                score -= support_bonus
                reasons.append("也有站边/保护")
            if len(snapshot["claims"]) > 1:
                claim_bonus = min(1.2, 0.4 * (len(snapshot["claims"]) - 1))
                score += claim_bonus
                reasons.append("身份说法不止一种")
            if snapshot["death_links"]:
                death_bonus = min(0.8, 0.25 * len(snapshot["death_links"]))
                score += death_bonus
                reasons.append("与死亡链条有关")
            if snapshot["mention_count"]:
                score += min(0.6, 0.12 * snapshot["mention_count"])
        pattern = re.compile(rf"(?<![A-Za-z0-9]){re.escape(target_id)}(?![A-Za-z0-9])")
        fallback_hits = 0
        for line in recent_memory[-5:]:
            if not pattern.search(line):
                continue
            fallback_hits += 1
            if any(keyword in line for keyword in ("改口", "矛盾", "对跳", "反水", "带节奏", "卖队", "假", "骗", "冲票", "踩", "查杀", "自相矛盾")):
                score += 0.5
                if len(reasons) < 3:
                    reasons.append(self._compact_text(line, limit=60))
            elif any(keyword in line for keyword in ("投票", "票型", "站边", "出局", "死亡", "身份", "警长", "发言", "表态")):
                score += 0.2
            elif any(keyword in line for keyword in ("可信", "一致", "解释清楚", "对得上", "稳定", "金水")):
                score -= 0.2
        if not reasons:
            if fallback_hits:
                reasons.append("最近公开记忆里有少量相关线索，但结构化证据仍弱")
            else:
                reasons.append("结构化证据很少，只能低置信度判断")
        return score, reasons[:3]

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
        default_round = sync_packet.get("round")
        default_phase = sync_packet.get("phase") or sync_packet.get("public_phase")

        def add_event(item: object) -> None:
            self._update_player_dossiers_from_public_item(
                item,
                default_round=default_round,
                default_phase=default_phase,
            )
            event = self._summarize_public_event(item, default_round=default_round, default_phase=default_phase)
            if not event or event in seen:
                return
            seen.add(event)
            events.append(event)

        for key in ("events", "public_events", "log", "logs", "history", "records", "timeline"):
            value = sync_packet.get(key)
            if isinstance(value, list):
                for item in value:
                    add_event(item)
                    if len(events) >= 8:
                        return events
        for key in ("public_state", "state", "game"):
            value = sync_packet.get(key)
            if value is not None:
                add_event(value)
                if len(events) >= 8:
                    return events

        def walk(item: object, depth: int) -> None:
            if len(events) >= 8 or depth > 2:
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
                    if len(events) >= 8:
                        return

        walk(sync_packet, 0)
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
            "memory_brief": self._recent_memory(5),
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

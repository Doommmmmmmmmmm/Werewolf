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
DEFAULT_PUBLIC_MEMORY_SIZE = 18
DEFAULT_STRUCTURED_ROUND_MEMORY_SIZE = 3
DEFAULT_STRUCTURED_CATEGORY_MEMORY_SIZE = 6


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
        self._vote_memory: deque[str] = deque(maxlen=DEFAULT_STRUCTURED_CATEGORY_MEMORY_SIZE)
        self._claim_memory: deque[str] = deque(maxlen=DEFAULT_STRUCTURED_CATEGORY_MEMORY_SIZE)
        self._death_memory: deque[str] = deque(maxlen=DEFAULT_STRUCTURED_CATEGORY_MEMORY_SIZE)
        self._conflict_memory: deque[str] = deque(maxlen=DEFAULT_STRUCTURED_CATEGORY_MEMORY_SIZE)
        self._round_summary_by_round: dict[str, deque[str]] = {}
        self._round_order: deque[str] = deque()
        self._player_signals: dict[str, dict[str, Any]] = {}
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
                self._record_public_summary(event)

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
        recent_memory = self._recent_memory(6)
        layered_memory = self._structured_memory_snapshot()
        brief: dict[str, Any] = {
            "phase": phase,
            "recent_public_signals": recent_memory,
            "layered_public_memory": layered_memory,
            "current_round_dialogue_preview": self._compact_dialogue(current_dialogue, limit=3),
        }
        mode = self._request_mode(allowed_actions)
        if mode == "hunter_reaction":
            ranking = self._hunter_target_ranking(allowed_actions)
            brief.update(
                {
                    "decision_mode": "hunter_reaction",
                    "focus": "先逐一比较合法 target_ids 的票型、身份声明、改口和死亡链；若最高分目标具备多类公开证据且分差明显，优先开枪，只有所有候选都只是单类弱证据且误伤风险高时才允许 pass。",
                    "target_ranking": ranking["target_ranking"],
                    "best_target": ranking["best_target"],
                    "best_score": ranking["best_score"],
                    "second_score": ranking["second_score"],
                    "shooting_rule": ranking["shooting_rule"],
                    "skip_option": ranking["skip_option"],
                    "skip_condition": ranking["skip_condition"],
                }
            )
        elif mode == "last_words":
            brief.update(
                {
                    "decision_mode": "last_words",
                    "focus": "收束到已公开的票型、身份矛盾和死亡顺序，不要泛化找狼。",
                    "current_controversy": self._recent_controversy(recent_memory),
                    "key_contradiction": self._key_contradiction(recent_memory),
                }
            )
        elif mode == "speak":
            brief.update(
                {
                    "decision_mode": "speak",
                    "focus": "回应当前争议，优先点出最近票型变化与身份声明矛盾。",
                    "current_controversy": self._recent_controversy(recent_memory),
                    "key_contradiction": self._key_contradiction(recent_memory),
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
        pieces = list(self._conflict_memory)[-2:]
        pieces.extend(line for line in recent_memory[-2:] if line)
        if not pieces:
            return "暂无足够公开记忆"
        return "；".join(pieces[-3:])

    def _key_contradiction(self, recent_memory: list[str]) -> str:
        for line in reversed(list(self._conflict_memory) + recent_memory):
            if any(
                keyword in line
                for keyword in (
                    "改口",
                    "矛盾",
                    "对跳",
                    "反水",
                    "自相矛盾",
                    "投票",
                    "死亡",
                    "警长",
                )
            ):
                return line
        return recent_memory[-1] if recent_memory else "暂无明确矛盾"

    def _structured_memory_snapshot(self) -> dict[str, list[str]]:
        return {
            "recent_public_signals": self._recent_memory(6),
            "vote_summaries": list(self._vote_memory)[-3:],
            "identity_summaries": list(self._claim_memory)[-3:],
            "death_summaries": list(self._death_memory)[-3:],
            "conflict_summaries": list(self._conflict_memory)[-3:],
            "round_summaries": self._round_summary_snapshot(),
        }

    def _round_summary_snapshot(self) -> list[str]:
        snapshot: list[str] = []
        for round_key in list(self._round_order)[-DEFAULT_STRUCTURED_ROUND_MEMORY_SIZE:]:
            notes = list(self._round_summary_by_round.get(round_key, []))
            if notes:
                snapshot.append(f"{round_key}: " + "；".join(notes[-3:]))
        return snapshot

    def _record_public_summary(self, summary: str) -> None:
        categories = self._summary_categories(summary)
        if "vote" in categories:
            self._vote_memory.append(summary)
        if "claim" in categories:
            self._claim_memory.append(summary)
        if "death" in categories:
            self._death_memory.append(summary)
        if "conflict" in categories:
            self._conflict_memory.append(summary)
        self._update_player_signals(summary, categories)
        round_key = self._extract_round_key(summary)
        if round_key:
            if round_key not in self._round_summary_by_round:
                self._round_summary_by_round[round_key] = deque(maxlen=DEFAULT_STRUCTURED_CATEGORY_MEMORY_SIZE)
                self._round_order.append(round_key)
                while len(self._round_order) > DEFAULT_STRUCTURED_ROUND_MEMORY_SIZE:
                    oldest = self._round_order.popleft()
                    self._round_summary_by_round.pop(oldest, None)
            bucket = self._round_summary_by_round[round_key]
            if summary not in bucket:
                bucket.append(summary)

    def _update_player_signals(self, summary: str, categories: list[str]) -> None:
        player_ids = re.findall(r"\bp\d+\b", summary)
        if not player_ids:
            return
        lower = summary.lower()
        for player_id in dict.fromkeys(player_ids):
            bucket = self._player_signals.setdefault(
                player_id,
                {
                    "vote": 0,
                    "claim": 0,
                    "death": 0,
                    "conflict": 0,
                    "support": 0,
                    "accusation": 0,
                    "notes": deque(maxlen=4),
                },
            )
            for category in categories:
                if category in bucket:
                    bucket[category] = int(bucket[category]) + 1
            if any(token in summary for token in ("好人", "金水", "可信", "稳定", "对得上")):
                bucket["support"] = int(bucket["support"]) + 1
            if any(token in summary for token in ("狼人", "查杀", "冲票", "带狼")):
                bucket["accusation"] = int(bucket["accusation"]) + 1
            if any(token in lower for token in ("support", "trusted", "good")):
                bucket["support"] = int(bucket["support"]) + 1
            notes = bucket["notes"]
            if isinstance(notes, deque) and summary not in notes:
                notes.append(summary)

    @staticmethod
    def _summary_categories(summary: str) -> list[str]:
        categories: list[str] = []
        lower = summary.lower()
        if any(token in summary for token in ("票型", "投票→", "投票")) or "vote" in lower:
            categories.append("vote")
        if any(token in summary for token in ("身份声明", "身份:", "身份公开", "claim")):
            categories.append("claim")
        if any(token in summary for token in ("死亡链", "死亡→", "被刀", "出局", "被票", "发生死亡")) or any(
            token in lower for token in ("death", "killed", "eliminated")
        ):
            categories.append("death")
        if any(
            keyword in summary
            for keyword in (
                "改口",
                "矛盾",
                "对跳",
                "反水",
                "自相矛盾",
                "带节奏",
                "卖队",
                "冲票",
                "踩",
                "查杀",
            )
        ):
            categories.append("conflict")
        if not categories:
            categories.append("speech")
        return categories

    @staticmethod
    def _extract_round_key(summary: str) -> str:
        match = re.search(r"\[(r\d+)", summary)
        if match:
            return match.group(1)
        match = re.search(r"\br\d+\b", summary)
        return match.group(0) if match else ""

    def _target_evidence_snapshot(self, target_id: str) -> dict[str, Any]:
        pattern = re.compile(rf"(?<![A-Za-z0-9]){re.escape(target_id)}(?![A-Za-z0-9])")
        buckets: dict[str, list[str]] = {
            "vote": [],
            "claim": [],
            "death": [],
            "conflict": [],
            "support": [],
            "accusation": [],
        }
        seen: set[str] = set()
        sources = (
            ("vote", self._vote_memory),
            ("claim", self._claim_memory),
            ("death", self._death_memory),
            ("conflict", self._conflict_memory),
            ("recent", self._public_memory),
        )
        for source_category, source in sources:
            for line in reversed(source):
                if line in seen or not pattern.search(line):
                    continue
                seen.add(line)
                lower = line.lower()
                if source_category == "vote" or any(token in line for token in ("票型", "投票→", "投票")) or "vote" in lower:
                    buckets["vote"].append(line)
                if source_category == "claim" or any(token in line for token in ("身份声明", "身份:", "身份公开")) or "claim" in lower:
                    buckets["claim"].append(line)
                if source_category == "death" or any(token in line for token in ("死亡链", "死亡→", "被刀", "出局", "被票", "发生死亡")) or any(
                    token in lower for token in ("death", "killed", "eliminated")
                ):
                    buckets["death"].append(line)
                if source_category == "conflict" or any(
                    keyword in line
                    for keyword in (
                        "改口",
                        "矛盾",
                        "对跳",
                        "反水",
                        "自相矛盾",
                        "带节奏",
                        "卖队",
                        "冲票",
                        "踩",
                        "查杀",
                    )
                ):
                    buckets["conflict"].append(line)
                if any(token in line for token in ("好人", "金水", "可信", "稳定", "对得上")):
                    buckets["support"].append(line)
                if any(token in line for token in ("狼人", "查杀", "冲票", "带狼")):
                    buckets["accusation"].append(line)
                if all(len(items) >= 2 for items in buckets.values()):
                    break
        counts = {name: len(items) for name, items in buckets.items()}
        player_bucket = self._player_signals.get(target_id)
        if isinstance(player_bucket, Mapping):
            for name in ("vote", "claim", "death", "conflict", "support", "accusation"):
                try:
                    counts[name] += int(player_bucket.get(name, 0))
                except (TypeError, ValueError):
                    continue
            bucket_notes = player_bucket.get("notes")
            if isinstance(bucket_notes, deque):
                for note in reversed(bucket_notes):
                    if note not in seen and pattern.search(note):
                        lower_note = note.lower()
                        if any(token in note for token in ("票型", "投票→", "投票")) or "vote" in lower_note:
                            buckets["vote"].append(note)
                        if any(token in note for token in ("身份声明", "身份:", "身份公开")) or "claim" in lower_note:
                            buckets["claim"].append(note)
                        if any(token in note for token in ("死亡链", "死亡→", "被刀", "出局", "被票", "发生死亡")) or any(
                            token in lower_note for token in ("death", "killed", "eliminated")
                        ):
                            buckets["death"].append(note)
                        if any(
                            keyword in note
                            for keyword in (
                                "改口",
                                "矛盾",
                                "对跳",
                                "反水",
                                "自相矛盾",
                                "带节奏",
                                "卖队",
                                "冲票",
                                "踩",
                                "查杀",
                            )
                        ):
                            buckets["conflict"].append(note)
                        if any(token in note for token in ("好人", "金水", "可信", "稳定", "对得上")):
                            buckets["support"].append(note)
                        if any(token in note for token in ("狼人", "查杀", "冲票", "带狼")):
                            buckets["accusation"].append(note)
        channels = sum(1 for name in ("vote", "claim", "death", "conflict") if buckets[name])
        return {
            "counts": counts,
            "evidence_channels": channels,
            "notes": {name: items[:2] for name, items in buckets.items() if items},
        }

    def _hunter_target_ranking(
        self,
        allowed_actions: list[Mapping[str, Any]],
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
            evidence = self._target_evidence_snapshot(target_id)
            score, reasons = self._score_hunter_target(target_id, evidence)
            ranking.append(
                {
                    "target_id": target_id,
                    "score": round(score, 2),
                    "evidence_channels": evidence["evidence_channels"],
                    "reasons": reasons,
                }
            )
        ranking.sort(key=lambda item: (item["score"], item["evidence_channels"]), reverse=True)
        skip_option = self._pass_like_action_kind(allowed_actions)
        best = ranking[0] if ranking else {"target_id": "", "score": 0.0, "evidence_channels": 0}
        second_score = ranking[1]["score"] if len(ranking) > 1 else 0.0
        best_score = float(best["score"])
        best_channels = int(best.get("evidence_channels", 0))
        if skip_option:
            shooting_rule = "最高分目标若同时拥有≥2类公开证据且分差明显，优先开枪。"
            if best_score >= 2.5 and best_channels >= 2:
                skip_condition = "当前最高分目标已有多类公开证据，不要因为没有完美定论而跳过。"
            else:
                skip_condition = "只有当所有候选都只是单类弱证据，且最高分<2.0或与次高分差<0.6时才可跳过。"
        else:
            shooting_rule = "没有跳过动作时，直接选择最高分且证据渠道更多的目标。"
            skip_condition = "没有跳过动作时，只选分数最高且风险最低的目标。"
        return {
            "target_ranking": ranking[:4],
            "best_target": best.get("target_id", ""),
            "best_score": best_score,
            "second_score": second_score,
            "shooting_rule": shooting_rule,
            "skip_option": skip_option,
            "skip_condition": skip_condition,
        }

    def _score_hunter_target(self, target_id: str, evidence: Mapping[str, Any]) -> tuple[float, list[str]]:
        counts = evidence.get("counts") if isinstance(evidence, Mapping) else {}
        notes = evidence.get("notes") if isinstance(evidence, Mapping) else {}
        score = 0.0
        reasons: list[str] = []

        def note_excerpt(category: str) -> str:
            category_notes = notes.get(category) if isinstance(notes, Mapping) else None
            if isinstance(category_notes, list) and category_notes:
                return self._compact_text(category_notes[0], limit=52)
            return ""

        vote_count = int(counts.get("vote", 0)) if isinstance(counts, Mapping) else 0
        claim_count = int(counts.get("claim", 0)) if isinstance(counts, Mapping) else 0
        death_count = int(counts.get("death", 0)) if isinstance(counts, Mapping) else 0
        conflict_count = int(counts.get("conflict", 0)) if isinstance(counts, Mapping) else 0
        support_count = int(counts.get("support", 0)) if isinstance(counts, Mapping) else 0
        accusation_count = int(counts.get("accusation", 0)) if isinstance(counts, Mapping) else 0

        if vote_count:
            score += min(3.0, vote_count * 0.9)
            reasons.append(f"票型×{vote_count}" + (f" {note_excerpt('vote')}" if note_excerpt('vote') else ""))
        if conflict_count:
            score += min(4.0, conflict_count * 1.25)
            reasons.append(f"改口/对跳×{conflict_count}" + (f" {note_excerpt('conflict')}" if note_excerpt('conflict') else ""))
        if claim_count:
            score += min(1.8, claim_count * 0.45)
            reasons.append(f"身份声明×{claim_count}" + (f" {note_excerpt('claim')}" if note_excerpt('claim') else ""))
        if death_count:
            score += min(2.0, death_count * 0.8)
            reasons.append(f"死亡链×{death_count}" + (f" {note_excerpt('death')}" if note_excerpt('death') else ""))
        if accusation_count:
            score += min(1.8, accusation_count * 0.55)
            reasons.append(f"查杀/狼人×{accusation_count}" + (f" {note_excerpt('accusation')}" if note_excerpt('accusation') else ""))
        if support_count:
            score -= min(2.0, support_count * 0.7)
            reasons.append(f"好人/金水×{support_count}" + (f" {note_excerpt('support')}" if note_excerpt('support') else ""))

        if conflict_count and vote_count:
            score += 0.4
        if claim_count and conflict_count:
            score += 0.6
        if death_count and (vote_count or conflict_count):
            score += 0.3
        if support_count and not (vote_count or conflict_count or death_count):
            score -= 0.4

        if not reasons:
            reasons.append("公开证据里还没有稳定的结构化命中")
        return score, reasons[:4]

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
            event = self._summarize_public_event(
                item, default_round=default_round, default_phase=default_phase
            )
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

        conflict_keywords = (
            "改口",
            "矛盾",
            "对跳",
            "反水",
            "自相矛盾",
            "带节奏",
            "卖队",
            "冲票",
            "踩",
            "查杀",
        )

        category = "发言"
        detail = ""
        if kind in {"vote", "voting"} or item.get("vote") is not None or "投票" in text:
            category = "票型"
            voter = actor or target or "?"
            if target:
                detail = f"{voter}→{target}"
            else:
                detail = f"{voter}:{TaskAgent._compact_text(text or item.get('vote'), limit=24)}"
        elif kind in {"death", "died", "eliminate", "eliminated", "kill", "killed"} or item.get("death") is not None or any(
            token in text for token in ("死亡", "被刀", "出局", "被票")
        ):
            category = "死亡链"
            who = target or actor or "?"
            detail = who
            if text:
                detail += f" {TaskAgent._compact_text(text, limit=28)}"
        elif kind in {"reveal", "identity", "open_role", "claim"} or role or any(
            token in text for token in ("身份", "预言家", "守卫", "猎人", "狼人", "好人", "金水", "查杀")
        ):
            category = "身份声明"
            claimer = actor or target or "?"
            claim = role or text or "身份公开"
            detail = f"{claimer}: {TaskAgent._compact_text(claim, limit=28)}"
        elif any(keyword in text for keyword in conflict_keywords):
            category = "矛盾"
            subject = actor or target or "?"
            detail = f"{subject}: {TaskAgent._compact_text(text, limit=28)}"
        else:
            subject = actor or target
            if subject and text:
                detail = f"{subject}: {TaskAgent._compact_text(text, limit=28)}"
            elif text:
                detail = TaskAgent._compact_text(text, limit=40)
            elif subject:
                detail = subject

        if not detail:
            return None
        return f"{prefix}{category} {detail}"

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
            "memory_brief": self._recent_memory(6),
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

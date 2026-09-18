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

CURRENT_ROUND_DIALOGUE_TOOL_NAME = "read_current_round_dialogue"
DEFAULT_MAX_TOOL_CALLS_PER_DECISION = 5
DEFAULT_MAX_TOOL_RESULT_TOKENS = 1000
DEFAULT_MAX_PROMPT_CHARS = 12000
# 这是 Task-Agent 层的“纠错重试”次数：首次模型调用之外，最多再请求两次。
# 传输层的 HTTP/网络重试仍由 ModelClient 的 MODEL_MAX_RETRIES 单独控制。
DEFAULT_MAX_DECISION_RETRIES = 2
_SPEECH_ACTION_KINDS = frozenset({"speak", "last_words"})
_DAY_VOTE_PASS_KINDS = frozenset({"pass", "skip", "noop", "idle"})
_PUBLIC_ID_KEYS = ("player_id", "playerId", "id", "target_id", "targetId")


CURRENT_ROUND_DIALOGUE_TOOL: dict[str, Any] = {
    "name": CURRENT_ROUND_DIALOGUE_TOOL_NAME,
    "description": "读取当前昼夜轮次中、当前玩家依法可见的已发生发言。",
    "parameters": {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    },
}


def _shorten_text(text: str, limit: int) -> str:
    text = str(text).strip()
    if limit <= 0:
        return ""
    return text if len(text) <= limit else text[: max(0, limit - 1)] + "…"


def _unique_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        normalized = str(value).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


def _coerce_player_id(value: Any) -> str | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, Mapping):
        for key in _PUBLIC_ID_KEYS:
            nested = value.get(key)
            coerced = _coerce_player_id(nested)
            if coerced:
                return coerced
        return None
    if isinstance(value, (list, tuple, set)):
        for item in value:
            coerced = _coerce_player_id(item)
            if coerced:
                return coerced
        return None
    text = str(value).strip()
    return text or None


def _collect_values_by_keys(node: Any, key_names: set[str], results: list[Any], seen: set[int]) -> None:
    if node is None:
        return
    if isinstance(node, (str, bytes, int, float, bool)):
        return
    node_id = id(node)
    if node_id in seen:
        return
    seen.add(node_id)
    if isinstance(node, Mapping):
        for key, value in node.items():
            if str(key) in key_names:
                results.append(value)
            _collect_values_by_keys(value, key_names, results, seen)
    elif isinstance(node, (list, tuple, set)):
        for item in node:
            _collect_values_by_keys(item, key_names, results, seen)


def _extract_id_list(node: Mapping[str, Any], key_names: tuple[str, ...]) -> list[str]:
    candidates: list[Any] = []
    _collect_values_by_keys(node, set(key_names), candidates, set())
    ids: list[str] = []
    for candidate in candidates:
        if isinstance(candidate, (list, tuple, set)):
            for item in candidate:
                coerced = _coerce_player_id(item)
                if coerced:
                    ids.append(coerced)
        else:
            coerced = _coerce_player_id(candidate)
            if coerced:
                ids.append(coerced)
    return _unique_preserve_order(ids)


_DIALOGUE_TEXT_KEYS = ("text", "content", "message", "speech", "utterance", "dialogue")


def _append_bounded(values: list[Any], value: Any, limit: int) -> None:
    if value is None or limit <= 0 or value in values:
        return
    values.append(value)
    if len(values) > limit:
        del values[:-limit]


def _direct_values(mapping: Mapping[str, Any], keys: tuple[str, ...]) -> list[Any]:
    return [mapping[key] for key in keys if key in mapping and mapping[key] is not None]


def _extract_players_alive_dead(public_state: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    alive: list[str] = []
    dead: list[str] = []
    for value in _direct_values(public_state, ("alive_ids", "alive_players", "living_ids", "survivor_ids", "living")):
        alive.extend(_extract_id_list({"value": value}, ("value",)))
    for value in _direct_values(public_state, ("dead_ids", "dead_players", "deceased_ids", "eliminated_ids", "dead")):
        dead.extend(_extract_id_list({"value": value}, ("value",)))
    return _unique_preserve_order(alive), _unique_preserve_order(dead)


def _extract_sheriff(public_state: Mapping[str, Any]) -> str | None:
    for value in _direct_values(public_state, ("sheriff_id", "sheriff_player_id", "sheriff")):
        player_id = _coerce_player_id(value)
        if player_id:
            return player_id
    return None


def _extract_dialogue_texts(current_dialogue: Any) -> list[str]:
    texts: list[str] = []
    if isinstance(current_dialogue, str):
        text = current_dialogue.strip()
        if text:
            texts.append(text)
        return texts
    if isinstance(current_dialogue, Mapping):
        for key in _DIALOGUE_TEXT_KEYS:
            value = current_dialogue.get(key)
            if isinstance(value, str) and value.strip():
                texts.append(value.strip())
        return texts
    if isinstance(current_dialogue, (list, tuple, set)):
        for item in current_dialogue:
            texts.extend(_extract_dialogue_texts(item))
    return texts


def _extract_day_vote_targets(request: Mapping[str, Any]) -> list[str]:
    phase = str(request.get("phase") or request.get("channel") or request.get("kind") or "").lower()
    allowed_actions = request.get("allowed_actions")
    if not isinstance(allowed_actions, list):
        return []
    targets: list[str] = []
    for item in allowed_actions:
        if not isinstance(item, Mapping):
            continue
        kind = str(item.get("kind") or "").lower()
        if "vote" not in kind or "sheriff" in kind:
            continue
        if "day" not in kind and "day" not in phase and kind != "vote":
            continue
        for target_id in item.get("target_ids") or []:
            coerced = _coerce_player_id(target_id)
            if coerced:
                targets.append(coerced)
    return _unique_preserve_order(targets)


class PublicLedger:
    """有界、只读公开事实摘要；不保存私有角色信息或完整历史。"""

    def __init__(self) -> None:
        self.alive_ids: list[str] = []
        self.dead_ids: list[str] = []
        self.revealed_roles: list[str] = []
        self.current_sheriff_id: str | None = None
        self.sheriff_alive: bool | None = None
        self.badge_transfers: list[str] = []
        self.death_log: list[str] = []
        self.day_eliminations: list[str] = []
        self.night_deaths: list[str] = []
        self.last_vote_counts: list[str] = []
        self.public_claims: list[str] = []

    def ingest(self, packet: Mapping[str, Any] | None, dialogue: Any = None) -> None:
        if not isinstance(packet, Mapping):
            return
        state = packet.get("public_state")
        if not isinstance(state, Mapping):
            state = packet
        alive, dead = _extract_players_alive_dead(state)
        if alive:
            self.alive_ids = alive[:24]
        if dead:
            self.dead_ids = dead[:24]
        sheriff = _extract_sheriff(state)
        if sheriff:
            self.current_sheriff_id = sheriff
            self.sheriff_alive = sheriff in self.alive_ids if self.alive_ids else None
        elif self.current_sheriff_id and self.alive_ids:
            self.sheriff_alive = self.current_sheriff_id in self.alive_ids
        for value in _direct_values(state, ("revealed_roles", "revealed_role_map", "public_roles")):
            if isinstance(value, Mapping):
                for player, role in value.items():
                    player_id = _coerce_player_id(player)
                    if player_id and role:
                        _append_bounded(self.revealed_roles, f"{player_id}:{_shorten_text(str(role), 24)}", 8)
        for value in _direct_values(state, ("badge_transfers", "sheriff_transfers", "public_badge_transfers")):
            if isinstance(value, list):
                for transfer in value[-6:]:
                    if isinstance(transfer, Mapping):
                        old_id = _coerce_player_id(transfer.get("from_id") or transfer.get("old_sheriff_id"))
                        new_id = _coerce_player_id(transfer.get("to_id") or transfer.get("new_sheriff_id"))
                        if old_id and new_id:
                            _append_bounded(self.badge_transfers, f"{old_id}→{new_id}", 6)
        for value in _direct_values(state, ("last_vote_counts", "recent_vote_counts", "vote_counts")):
            if value is not None:
                _append_bounded(self.last_vote_counts, _shorten_text(str(value), 120), 4)
        for value in _direct_values(state, ("day_eliminations", "day_eliminated_ids", "public_day_eliminations")):
            for player_id in _extract_id_list({"value": value}, ("value",)):
                note = f"日放逐:{player_id}"
                _append_bounded(self.day_eliminations, note, 6)
                _append_bounded(self.death_log, note, 8)
        for value in _direct_values(state, ("night_deaths", "night_dead_ids", "public_night_deaths")):
            for player_id in _extract_id_list({"value": value}, ("value",)):
                note = f"夜死:{player_id}"
                _append_bounded(self.night_deaths, note, 6)
                _append_bounded(self.death_log, note, 8)

        # 只读取明确标为公开的事件容器，避免递归扫到 private/audit 字段。
        records: list[Any] = []
        for key in ("public_events", "public_event_log", "recent_events", "events"):
            value = packet.get(key)
            if value is None and isinstance(state, Mapping):
                value = state.get(key)
            if isinstance(value, list):
                records.extend(value[-16:])
        for record in records:
            self._ingest_public_record(record)
        for text in _extract_dialogue_texts(dialogue):
            claim = _shorten_text(" ".join(text.split()), 96)
            if claim:
                _append_bounded(self.public_claims, claim, 8)

    def _ingest_public_record(self, record: Any) -> None:
        if not isinstance(record, Mapping):
            return
        kind = str(record.get("kind") or record.get("type") or record.get("event") or "").lower()
        target = _coerce_player_id(record.get("player_id") or record.get("target_id") or record.get("victim_id"))
        actor = _coerce_player_id(record.get("from_id") or record.get("source_id") or record.get("old_sheriff_id"))
        recipient = _coerce_player_id(record.get("to_id") or record.get("new_sheriff_id"))
        if target and any(word in kind for word in ("night", "wolf", "kill", "death", "死亡", "夜")):
            note = f"夜死:{target}"
            _append_bounded(self.night_deaths, note, 6)
            _append_bounded(self.death_log, note, 8)
        elif target and any(word in kind for word in ("day", "eliminat", "放逐", "出局")):
            note = f"日放逐:{target}"
            _append_bounded(self.day_eliminations, note, 6)
            _append_bounded(self.death_log, note, 8)
        if actor and recipient and any(word in kind for word in ("badge", "sheriff", "警徽", "警长")):
            _append_bounded(self.badge_transfers, f"{actor}→{recipient}", 6)
            self.current_sheriff_id = recipient
        role = record.get("revealed_role") or record.get("role_reveal")
        if target and role:
            _append_bounded(self.revealed_roles, f"{target}:{_shorten_text(str(role), 24)}", 8)
        votes = record.get("vote_counts") or record.get("counts")
        if votes is not None:
            _append_bounded(self.last_vote_counts, _shorten_text(str(votes), 120), 4)
        claim = record.get("claim") or record.get("text") or record.get("message")
        if isinstance(claim, str) and claim.strip():
            _append_bounded(self.public_claims, _shorten_text(" ".join(claim.split()), 96), 8)

    def summary(self, limit: int = 1700) -> str:
        sections: list[str] = []
        if self.alive_ids:
            sections.append("存活：" + "、".join(self.alive_ids))
        if self.dead_ids:
            sections.append("已死：" + "、".join(self.dead_ids))
        if self.death_log:
            sections.append("死亡方式：" + "；".join(self.death_log[-6:]))
        if self.current_sheriff_id:
            status = "存活" if self.sheriff_alive is not False else "已死"
            sections.append(f"警长：{self.current_sheriff_id}（{status}）")
        if self.badge_transfers:
            sections.append("警徽传递：" + "；".join(self.badge_transfers[-4:]))
        if self.revealed_roles:
            sections.append("公开身份：" + "；".join(self.revealed_roles[-5:]))
        if self.day_eliminations:
            sections.append("最近日放逐：" + "；".join(self.day_eliminations[-2:]))
        if self.night_deaths:
            sections.append("最近夜死：" + "；".join(self.night_deaths[-2:]))
        if self.last_vote_counts:
            sections.append("最近票型：" + "；".join(self.last_vote_counts[-2:]))
        if self.public_claims:
            sections.append("最近公开声明：" + "；".join(self.public_claims[-6:]))
        return _shorten_text("\n".join(sections), limit)


_PUBLIC_CLAIM_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("seer_claimers", re.compile(r"(?P<claimer>p\d+).{0,8}?(?:我是|我就是|自称).{0,8}?(?:预言家|先知)")),
    ("wolf_check", re.compile(r"(?:验了|查了|昨晚验了|昨晚查了|查验).{0,10}?(?P<target>p\d+).{0,12}?(?:狼|狼人|查杀)")),
    ("wolf_check", re.compile(r"(?:查杀|验出狼|验到狼).{0,5}?(?P<target>p\d+)")),
    ("wolf_check", re.compile(r"(?P<target>p\d+).{0,6}?(?:是|为)?(?:狼|狼人|查杀)")),
    ("rally", re.compile(r"(?:归票|票出|投票给|建议投|归票给).{0,5}?(?P<target>p\d+)")),
    ("gold", re.compile(r"(?:金水|好人).{0,5}?(?P<target>p\d+)")),
    ("gold", re.compile(r"(?P<target>p\d+).{0,4}?(?:是|为)?金水")),
    ("opposition", re.compile(r"对跳|跳预言家|双预言家|硬反对")),
)


def _extract_public_claim_hint(current_dialogue: Any) -> dict[str, Any]:
    texts = _extract_dialogue_texts(current_dialogue)
    buckets: dict[str, list[str]] = {"wolf_check_targets": [], "rally_targets": [], "gold_targets": [], "seer_claimers": [], "strong_role_claimers": []}
    snippets: list[str] = []
    contested = False
    for raw_text in texts:
        text = " ".join(str(raw_text).split())
        if not text:
            continue
        matched = False
        for label, pattern in _PUBLIC_CLAIM_PATTERNS:
            for match in pattern.finditer(text):
                matched = True
                if label == "opposition":
                    contested = True
                    snippets.append(_shorten_text(text, 64))
                    continue
                target = match.groupdict().get("target") or match.groupdict().get("claimer")
                if target:
                    buckets[label].append(target)
                    snippets.append(f"{label}:{target}")
        if matched and len(snippets) >= 8:
            break
    for key in buckets:
        buckets[key] = _unique_preserve_order(buckets[key])[:6]
    hard_targets = _unique_preserve_order(buckets["wolf_check_targets"] + buckets["rally_targets"])
    if len(hard_targets) > 1:
        contested = True
    hint: dict[str, Any] = {key: value for key, value in buckets.items() if value}
    if snippets:
        hint["claim_snippets"] = _unique_preserve_order(snippets)[:6]
    if contested:
        hint["contested"] = True
    if len(hard_targets) == 1 and not contested:
        hint["clear_target"] = hard_targets[0]
        hint["clear_target_confident"] = True
    return hint


def build_villager_public_hint(turn_packet: Mapping[str, Any], current_dialogue: Any) -> dict[str, Any]:
    public_state = turn_packet.get("public_state")
    request = turn_packet.get("request") if isinstance(turn_packet.get("request"), Mapping) else {}
    hint: dict[str, Any] = {}

    if isinstance(public_state, Mapping):
        alive_ids = _extract_id_list(
            public_state,
            ("alive_ids", "alive_players", "living_ids", "survivor_ids", "living"),
        )
        dead_ids = _extract_id_list(
            public_state,
            ("dead_ids", "dead_players", "deceased_ids", "eliminated_ids", "dead"),
        )
        sheriff_ids = _extract_id_list(
            public_state,
            ("sheriff_id", "sheriff_player_id", "sheriff", "警长", "警徽"),
        )
        if alive_ids:
            hint["alive_ids"] = alive_ids[:16]
        if dead_ids:
            hint["dead_ids"] = dead_ids[:16]
        if sheriff_ids:
            hint["sheriff_id"] = sheriff_ids[0]
            if alive_ids:
                hint["sheriff_alive"] = sheriff_ids[0] in alive_ids

    day_vote_targets = _extract_day_vote_targets(request) if request else []
    if day_vote_targets:
        hint["day_vote_targets"] = day_vote_targets[:16]

    claim_hint = _extract_public_claim_hint(current_dialogue)
    hint.update(claim_hint)
    return hint


def villager_quality_error(action: dict[str, Any], request: Mapping[str, Any], hint: Mapping[str, Any] | None) -> str | None:
    phase = str(request.get("phase") or request.get("channel") or request.get("kind") or "").lower()
    allowed_actions = request.get("allowed_actions")
    if not isinstance(allowed_actions, list):
        return None
    if "vote" not in phase and not any("vote" in str(item.get("kind") or "").lower() for item in allowed_actions if isinstance(item, Mapping)):
        return None
    vote_targets = _extract_day_vote_targets(request)
    if not vote_targets:
        return None
    action_kind = str(action.get("kind") or "").lower()
    if not isinstance(hint, Mapping):
        hint = {}
    if not hint.get("contested"):
        clear_target = hint.get("clear_target")
        if isinstance(clear_target, str) and clear_target in vote_targets:
            selected_target = str(action.get("target_id") or "")
            if action_kind in _DAY_VOTE_PASS_KINDS or selected_target != clear_target:
                return f"公开查杀/警长归票目标为{clear_target}；请说明硬反证。无硬反证时应投{clear_target}。"
    if action_kind in _DAY_VOTE_PASS_KINDS:
        return "当前有可投目标，平民除非会明显误伤强神或金水，否则不要弃票；请比较两个候选后投证据最强者。"

    return None


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
    lines = ["【平民决策清单】先分系统事实 / 玩家声明 / 自己推测；投票前至少比较两个候选。"]
    has_sheriff_vote = any(key in phase_lower for key in ("sheriff_vote", "警长投票", "警徽投票")) or any(
        kind in {"sheriff_vote", "sheriff"} or "sheriff_vote" in kind for kind in allowed_kinds
    )
    has_sheriff_candidacy = any(key in phase_lower for key in ("sheriff_candidacy", "警长竞选", "竞选", "竞警")) or any(
        "candidate" in kind or "candid" in kind or "sheriff_candidacy" in kind for kind in allowed_kinds
    )
    has_day_vote = (
        (any(key in phase_lower for key in ("vote", "投票")) or any("vote" in kind for kind in allowed_kinds))
        and not has_sheriff_vote
        and not has_sheriff_candidacy
    )
    has_speak = (
        any(key in phase_lower for key in ("speak", "discussion", "发言", "白天"))
        or any(kind in {"speak", "discussion", "day_speak"} for kind in allowed_kinds)
    ) and not has_day_vote and not has_sheriff_vote and not has_sheriff_candidacy
    has_last_words = any(key in phase_lower for key in ("last", "遗言")) or "last_words" in allowed_kinds
    has_sheriff = any(key in phase_lower for key in ("sheriff", "警长", "警徽")) or any(
        key in allowed_kinds for key in ("sheriff_vote", "sheriff")
    )
    if has_day_vote:
        lines.append("投票时优先保护可信金水、持续报验且查杀被支持的预言家、强神声明者和警徽传递链；只有硬对跳、明显矛盾、查杀命中或事实错误时才推翻。")
        lines.append("若当前轮或此前公开链中只有一名持续报验的预言家/警长明确查杀或归票，且没有硬对跳、事实矛盾、被公开查杀等强反证，day_vote 默认投被查杀/归票目标；其他嫌疑只能做备选，不要压过查杀目标，更不要轻易 pass。")
        lines.append("优先找未被金水覆盖且存在票型异常、事实错误、被查杀或爆狼线索指向的人。")
    if has_sheriff_vote:
        lines.append("警长投票时优先看谁的公开报验链更清晰、更稳定；不要把空泛自证或低权重印象压过查杀和归票信息。")
    if has_sheriff_candidacy:
        lines.append("警长竞选时平民默认不上警；只有需要挡刀、保警徽传递或扰乱狼视角时才考虑上警。")
    if has_speak:
        lines.append("发言前先核对存活名单、已死名单和警长是否仍存活；不要把白天放逐说成夜死，也不要把已死玩家当成可继续追责的当前存活对象。")
        lines.append("若你是平民但持有警徽，只能说继承/采信某人的公开查验链，不能说自己昨晚验了谁；日放逐叫出局、夜晚死亡叫夜死。平安夜不能推出具体刀口、救人或守人对象。")
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
    lines.append(
        "最终 JSON 只能包含实际行动字段；比较、复盘和理由先在内部完成，除非是 speak / last_words 才把简短依据写进 text。"
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
        self.public_ledger = PublicLedger()
        self._model_token_usage: dict[str, int] = {
            "successful_response_count": 0,
            "api_attempt_count": 0,
            "reported_usage_response_count": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }

    async def observe(self, sync_packet: dict[str, Any]) -> None:
        """只把同步包中的公开事实压缩进有界台账。"""

        self.public_ledger.ingest(sync_packet)

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
        self.public_ledger.ingest(turn_packet, current_dialogue)
        ledger_summary = self.public_ledger.summary()
        villager_public_hint = build_villager_public_hint(turn_packet, current_dialogue)

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
                "instruction": self._turn_instruction(turn_packet["request"], feedback, villager_public_hint, ledger_summary),
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
                quality_error = villager_quality_error(action, turn_packet["request"], villager_public_hint)
                if quality_error is None:
                    return action
                # 策略质量提示不是 schema 违规：重试耗尽后优先安全跟随高置信公开目标，
                # 否则提交已经通过 schema 的行动，而不是把分歧升级成 fallback。
                error = quality_error
                if _attempt >= self.max_decision_retries:
                    clear_target = villager_public_hint.get("clear_target") if isinstance(villager_public_hint, Mapping) else None
                    vote_targets = _extract_day_vote_targets(turn_packet["request"])
                    if (
                        isinstance(clear_target, str)
                        and villager_public_hint.get("clear_target_confident")
                        and clear_target in vote_targets
                        and not villager_public_hint.get("contested")
                    ):
                        return {
                            "request_id": turn_packet["request"]["request_id"],
                            "player_id": turn_packet["request"]["player_id"],
                            "kind": "day_vote",
                            "target_id": clear_target,
                        }
                    return action
            if _attempt >= self.max_decision_retries:
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
    def _turn_instruction(
        request: Mapping[str, Any],
        feedback: str,
        public_hint: Mapping[str, Any] | None,
        ledger_summary: str = "",
    ) -> str:
        checklist = render_villager_decision_checklist(request)
        feedback_text = f"上一次输出未通过校验：{feedback}" if feedback else ""
        ledger_text = f"【公开事实台账】\n{_shorten_text(ledger_summary, 1700)}" if ledger_summary else ""
        hint_text = ""
        if isinstance(public_hint, Mapping) and public_hint:
            hint_text = "【公开信息提示】" + json.dumps(public_hint, ensure_ascii=False, separators=(",", ":"))
        validation_feedback = "\n".join(
            part for part in (feedback_text, ledger_text, hint_text, checklist) if part
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

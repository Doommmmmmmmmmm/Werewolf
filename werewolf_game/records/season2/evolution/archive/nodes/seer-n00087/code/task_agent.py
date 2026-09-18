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
_SEER_ROLE_CONTEXT_TAG = "seer_v1"


def _as_str_id(value: Any) -> str | None:
    if isinstance(value, Mapping):
        for key in (
            "player_id",
            "target_id",
            "id",
            "name",
            "player",
            "target",
        ):
            nested = value.get(key)
            if nested is not None:
                return _as_str_id(nested)
        return None
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_alignment(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    if any(token in text for token in ("wolf", "werewolf", "狼人", "悍跳")):
        return "wolf"
    if any(token in text for token in ("vill", "good", "civil", "human", "村民", "好人", "平民")):
        return "village"
    return text


def _normalize_life_status(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    if text in {"alive", "live", "living", "true", "1", "yes", "y"}:
        return "alive"
    if text in {"dead", "died", "false", "0", "no"}:
        return "dead"
    if any(token in text for token in ("alive", "存活", "生存", "在场", "未死")):
        return "alive"
    if any(token in text for token in ("dead", "死亡", "阵亡", "出局", "淘汰", "放逐")):
        return "dead"
    return None


def _extract_player_status(item: Any) -> str | None:
    if not isinstance(item, Mapping):
        return None
    for key in ("alive", "is_alive", "isAlive", "status", "life_state", "state"):
        status = _normalize_life_status(item.get(key))
        if status is not None:
            return status
    return None


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


def _extract_id_list(value: Any) -> list[str]:
    ids: list[str] = []
    if isinstance(value, list):
        items = value
    elif isinstance(value, tuple):
        items = list(value)
    elif isinstance(value, set):
        items = list(value)
    elif isinstance(value, Mapping):
        items = []
        for key, nested in value.items():
            if isinstance(nested, bool):
                if nested:
                    pid = _as_str_id(key)
                    if pid and pid not in ids:
                        ids.append(pid)
                continue
            if isinstance(nested, (list, tuple, set, Mapping)):
                items.extend(_extract_id_list(nested))
                continue
            pid = _as_str_id(nested)
            if pid and pid not in ids:
                ids.append(pid)
            elif isinstance(key, str) and key and (key[0].lower() == "p" or key[0].isdigit()):
                key_id = _as_str_id(key)
                if key_id and key_id not in ids:
                    ids.append(key_id)
        return ids
    else:
        items = [value]

    for item in items:
        pid = _as_str_id(item)
        if pid and pid not in ids:
            ids.append(pid)
    return ids


def _extract_alive_player_ids(public_state: Mapping[str, Any]) -> list[str]:
    if not isinstance(public_state, Mapping):
        return []
    for key in (
        "alive_players",
        "alive_player_ids",
        "living_players",
        "live_players",
        "survivors",
    ):
        ids = _extract_id_list(public_state.get(key))
        if ids:
            return ids

    players = public_state.get("players")
    alive_ids: list[str] = []
    if isinstance(players, list):
        iterable = players
        for item in iterable:
            pid = _as_str_id(item)
            if not pid:
                continue
            status = _extract_player_status(item)
            if status == "alive":
                alive_ids.append(pid)
            elif status is None and not isinstance(item, Mapping):
                alive_ids.append(pid)
    elif isinstance(players, Mapping):
        for key, item in players.items():
            pid = _as_str_id(item) or _as_str_id(key)
            if not pid:
                continue
            status = _extract_player_status(item)
            if status is None and not isinstance(item, Mapping):
                status = _normalize_life_status(item)
            if status == "alive":
                alive_ids.append(pid)
            elif status is None and not isinstance(item, Mapping):
                alive_ids.append(pid)

    if alive_ids:
        return list(dict.fromkeys(alive_ids))

    for key in ("players_alive", "alive", "current_players"):
        ids = _extract_id_list(public_state.get(key))
        if ids:
            return ids
    return []


def _extract_sheriff_id(turn_packet: Mapping[str, Any]) -> str | None:
    candidates = (
        turn_packet.get("public_state"),
        turn_packet.get("private_information"),
        turn_packet.get("game"),
        turn_packet.get("self"),
    )
    for source in candidates:
        if not isinstance(source, Mapping):
            continue
        for key in (
            "sheriff_id",
            "police_chief_id",
            "chief_id",
            "badge_owner_id",
            "current_sheriff_id",
            "sheriff",
            "police_chief",
            "badge_owner",
        ):
            pid = _as_str_id(source.get(key))
            if pid:
                return pid
    return None


def _extract_inspection_records(private_information: Mapping[str, Any], alive_ids: set[str]) -> list[dict[str, Any]]:
    if not isinstance(private_information, Mapping):
        return []

    candidate_sources: list[Mapping[str, Any]] = [private_information]
    for key in ("role_state", "seer_state", "state"):
        nested = private_information.get(key)
        if isinstance(nested, Mapping):
            candidate_sources.append(nested)
            for subkey in ("seer", "role", "state"):
                subnested = nested.get(subkey)
                if isinstance(subnested, Mapping):
                    candidate_sources.append(subnested)

    records: list[dict[str, Any]] = []
    seen_targets: set[str] = set()
    candidate_fields = (
        "inspections",
        "inspection",
        "inspection_results",
        "inspection_history",
        "checked_players",
        "checks",
        "last_inspection",
        "latest_inspection",
        "night_checks",
        "night_inspections",
        "查验",
        "查验记录",
        "查验结果",
    )

    def add_entry(entry: Any) -> None:
        target_id: str | None = None
        alignment: str | None = None
        status: str | None = None
        public_flag: bool | None = None

        if isinstance(entry, Mapping):
            target_id = _as_str_id(
                entry.get("target_id")
                or entry.get("player_id")
                or entry.get("target")
                or entry.get("player")
                or entry.get("id")
            )
            alignment = _normalize_alignment(
                entry.get("result")
                or entry.get("alignment")
                or entry.get("faction")
                or entry.get("team")
                or entry.get("role")
            )
            status = _normalize_life_status(
                entry.get("status")
                or entry.get("alive")
                or entry.get("is_alive")
                or entry.get("life_state")
            )
            if entry.get("public") is not None:
                public_flag = bool(entry.get("public"))
        elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
            target_id = _as_str_id(entry[0])
            alignment = _normalize_alignment(entry[1])
            if len(entry) >= 3:
                status = _normalize_life_status(entry[2])
        elif isinstance(entry, str):
            parts = re.split(r"[,:，；;|\s]+", entry.strip(), maxsplit=2)
            if len(parts) >= 2:
                target_id = _as_str_id(parts[0])
                alignment = _normalize_alignment(parts[1])

        if not target_id or target_id in seen_targets:
            return
        if status is None:
            status = "alive" if target_id in alive_ids else ("dead" if alive_ids else None)
        record: dict[str, Any] = {"target_id": target_id}
        if alignment is not None:
            record["alignment"] = alignment
        if status is not None:
            record["status"] = status
        if public_flag is not None:
            record["public"] = public_flag
        records.append(record)
        seen_targets.add(target_id)

    for source in candidate_sources:
        for field in candidate_fields:
            value = source.get(field)
            if value is None:
                continue
            if isinstance(value, list):
                for entry in value:
                    add_entry(entry)
            elif isinstance(value, Mapping):
                for key, nested in value.items():
                    if isinstance(nested, Mapping):
                        payload = dict(nested)
                        payload.setdefault("target_id", key)
                        add_entry(payload)
                    else:
                        add_entry((key, nested))
            else:
                add_entry(value)

    return records


def _public_player_ids(packet: Mapping[str, Any]) -> list[str]:
    """Return IDs exposed by public state only; never inspect private role data."""
    state = packet.get("public_state") if isinstance(packet, Mapping) else None
    ids: list[str] = []
    if isinstance(state, Mapping):
        players = state.get("players")
        if isinstance(players, list):
            for item in players:
                pid = _as_str_id(item)
                if pid and pid not in ids:
                    ids.append(pid)
        elif isinstance(players, Mapping):
            for key, item in players.items():
                pid = _as_str_id(item) or _as_str_id(key)
                if pid and pid not in ids:
                    ids.append(pid)
    return ids


def _public_dialogue(packet: Mapping[str, Any]) -> list[dict[str, str]]:
    """Extract already-public speech records without reconstructing hidden history."""
    records: list[dict[str, str]] = []
    sources: list[Any] = []
    for key in ("public_dialogue", "dialogue", "speeches", "public_speeches", "public_events", "events"):
        value = packet.get(key) if isinstance(packet, Mapping) else None
        if isinstance(value, list):
            sources.extend(value)
    state = packet.get("public_state") if isinstance(packet, Mapping) else None
    if isinstance(state, Mapping):
        for key in ("public_dialogue", "dialogue", "speeches", "public_speeches", "public_events", "events"):
            value = state.get(key)
            if isinstance(value, list):
                sources.extend(value)
    for item in sources:
        if not isinstance(item, Mapping):
            continue
        speaker = _as_str_id(item.get("player_id") or item.get("speaker_id") or item.get("speaker") or item.get("from"))
        text = item.get("text") or item.get("content") or item.get("speech") or item.get("message")
        if speaker and isinstance(text, str) and text.strip():
            records.append({"speaker_id": speaker, "text": text.strip()[:500]})
    return records


def _public_votes(packet: Mapping[str, Any]) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    sources: list[Any] = []
    for container in (packet, packet.get("public_state") if isinstance(packet, Mapping) else None):
        if not isinstance(container, Mapping):
            continue
        for key in ("votes", "vote_history", "ballots", "public_votes", "elimination_votes"):
            value = container.get(key)
            if isinstance(value, list):
                sources.extend(value)
            elif isinstance(value, Mapping):
                for voter, target in value.items():
                    records.append({"voter_id": str(voter), "target_id": _as_str_id(target) or str(target)})
    for item in sources:
        if not isinstance(item, Mapping):
            continue
        voter = _as_str_id(item.get("voter_id") or item.get("voter") or item.get("player_id") or item.get("from"))
        target = _as_str_id(item.get("target_id") or item.get("target") or item.get("vote") or item.get("to"))
        if voter and target:
            records.append({"voter_id": voter, "target_id": target})
    return records


def _mentioned_ids(text: str, known_ids: list[str]) -> list[str]:
    found: list[str] = []
    for pid in known_ids:
        if pid and pid in text and pid not in found:
            found.append(pid)
    for token in re.findall(r"(?<![A-Za-z0-9_])p\d+|(?<![A-Za-z0-9_])\d+号", text, flags=re.IGNORECASE):
        pid = token[:-1] if token.endswith("号") else token
        if token.endswith("号") and ("p" + pid) in known_ids:
            pid = "p" + pid
        if pid not in found:
            found.append(pid)
    return found


def _update_seer_memory(memory: dict[str, Any], packet: Mapping[str, Any]) -> None:
    """Keep a bounded, public-only ledger. Duplicate sync packets are harmless."""
    if not isinstance(packet, Mapping):
        return
    known_ids = _public_player_ids(packet)
    for pid in _extract_alive_player_ids(packet.get("public_state", {})):
        if pid not in known_ids:
            known_ids.append(pid)
    claims = memory.setdefault("public_claims_by_player", {})
    claimants = memory.setdefault("seer_claimants", [])
    gold = memory.setdefault("public_gold_targets", {})
    for record in _public_dialogue(packet):
        speaker, text = record["speaker_id"], record["text"]
        if any(word in text for word in ("预言家", "验人", "查验", "金水", "查杀")):
            entries = claims.setdefault(speaker, [])
            if text not in entries:
                entries.append(text)
                del entries[:-4]
        if "预言家" in text and any(word in text for word in ("我是", "我跳", "自称", "对跳", "真预言家")):
            if speaker not in claimants:
                claimants.append(speaker)
        mentioned = _mentioned_ids(text, known_ids)
        if "金水" in text:
            for target in mentioned:
                values = gold.setdefault(target, [])
                if speaker not in values:
                    values.append(speaker)
        if any(word in text for word in ("验了", "查验", "查杀", "金水")) and mentioned:
            public_inspections = memory.setdefault("inspection_claims_public", [])
            claim = {"speaker_id": speaker, "targets": mentioned[:6], "text": text[:240]}
            if claim not in public_inspections:
                public_inspections.append(claim)
                del public_inspections[:-40]
    votes = memory.setdefault("vote_history", [])
    for vote in _public_votes(packet):
        if vote not in votes:
            votes.append(vote)
    del votes[:-80]

    state = packet.get("public_state")
    if isinstance(state, Mapping):
        dead: list[str] = []
        players = state.get("players")
        items = players.values() if isinstance(players, Mapping) else players if isinstance(players, list) else []
        for item in items:
            pid = _as_str_id(item)
            if pid and _extract_player_status(item) == "dead":
                dead.append(pid)
        for key in ("dead_players", "dead_player_ids", "eliminated_ids", "deaths"):
            dead.extend(_extract_id_list(state.get(key)))
        eliminated = memory.setdefault("eliminated_ids", [])
        for pid in dead:
            if pid not in eliminated:
                eliminated.append(pid)
        memory["dawn_dead_ids"] = _extract_id_list(state.get("dawn_dead_ids") or state.get("night_deaths"))[-20:]
        sheriff = _extract_sheriff_id({"public_state": state})
        if sheriff:
            memory["last_sheriff_id"] = sheriff


def _extract_public_signals(
    turn_packet: Mapping[str, Any], memory: Mapping[str, Any], seer_context: Mapping[str, Any]
) -> dict[str, Any]:
    alive = set(str(pid) for pid in seer_context.get("alive_player_ids", []))
    known_gold = set(str(pid) for pid in seer_context.get("known_live_villagers", []))
    public_gold = set(str(pid) for pid in memory.get("public_gold_targets", {}).keys()) & alive
    claimants = [str(pid) for pid in memory.get("seer_claimants", [])]
    direct = [pid for pid in claimants if pid in alive and pid != str(turn_packet.get("self", {}).get("player_id", ""))]
    pushes: list[str] = []
    for speaker, texts in memory.get("public_claims_by_player", {}).items():
        if any(any(word in text for word in ("推", "出", "投", "归票", "狼坑")) for text in texts):
            if any(target in "".join(texts) for target in known_gold):
                pushes.append(str(speaker))
    voting_gold = [
        str(v["voter_id"]) for v in memory.get("vote_history", [])
        if str(v.get("target_id")) in known_gold and str(v.get("voter_id")) not in pushes
    ]
    self_id = str(turn_packet.get("self", {}).get("player_id", ""))
    voting_self_or_gold = [
        str(v["voter_id"]) for v in memory.get("vote_history", [])
        if str(v.get("target_id")) == self_id or str(v.get("target_id")) in known_gold
    ]
    recent = [str(v["voter_id"]) for v in memory.get("vote_history", []) if str(v.get("target_id")) in set(memory.get("eliminated_ids", []))]
    pressured = list(dict.fromkeys(direct + pushes + voting_gold + recent))
    uninspected = set(str(pid) for pid in seer_context.get("uninspected_alive_ids", []))
    suggested_inspect = [pid for pid in dict.fromkeys(direct + pushes + list(seer_context.get("uninspected_alive_ids", []))) if pid in alive and pid in uninspected]
    suggested_vote = list(dict.fromkeys(direct + pushes + voting_gold))
    return {
        "direct_seer_counterclaims": direct,
        "public_gold_targets": sorted(public_gold),

        "players_pushing_known_villagers": list(dict.fromkeys(pushes)),
        "players_voting_known_villagers": list(dict.fromkeys(voting_gold)),
        "players_voting_self_or_gold": list(dict.fromkeys(voting_self_or_gold)),
        "recent_elimination_voters": list(dict.fromkeys(recent))[-20:],
        "unresolved_pressure_targets": pressured,
        "suggested_inspect_targets": suggested_inspect,
        "suggested_vote_targets": suggested_vote,
    }


def _rank_seer_inspection_targets(
    request: Mapping[str, Any], seer_context: Mapping[str, Any], public_signals: Mapping[str, Any]
) -> list[str]:
    allowed: list[str] = []
    for item in request.get("allowed_actions", []):
        if isinstance(item, Mapping) and "inspect" in str(item.get("kind", "")).lower():
            allowed.extend(str(pid) for pid in item.get("target_ids", []) if pid is not None)
    alive = set(str(pid) for pid in seer_context.get("alive_player_ids", []))
    uninspected = set(str(pid) for pid in seer_context.get("uninspected_alive_ids", []))
    allowed = [pid for pid in dict.fromkeys(allowed) if pid in alive and pid in uninspected]
    ranked: list[str] = []
    for group in (
        public_signals.get("direct_seer_counterclaims", []),
        public_signals.get("players_pushing_known_villagers", []),
        public_signals.get("unresolved_pressure_targets", []),
        seer_context.get("uninspected_alive_ids", []),
    ):
        for pid in group:
            pid = str(pid)
            if pid in allowed and pid not in ranked:
                ranked.append(pid)
    return ranked


def _seer_context(turn_packet: Mapping[str, Any]) -> dict[str, Any]:
    private_information = turn_packet.get("private_information")
    public_state = turn_packet.get("public_state")
    self_state = turn_packet.get("self")
    game = turn_packet.get("game")
    request = turn_packet.get("request")

    alive_ids = set(_extract_alive_player_ids(public_state) if isinstance(public_state, Mapping) else [])
    inspections = _extract_inspection_records(private_information, alive_ids) if isinstance(private_information, Mapping) else []
    inspected_targets = {str(item["target_id"]) for item in inspections if item.get("target_id")}

    known_live_wolves = [
        item["target_id"]
        for item in inspections
        if item.get("alignment") == "wolf" and item.get("status") == "alive"
    ]
    known_dead_wolves = [
        item["target_id"]
        for item in inspections
        if item.get("alignment") == "wolf" and item.get("status") == "dead"
    ]
    known_live_villagers = [
        item["target_id"]
        for item in inspections
        if item.get("alignment") == "village" and item.get("status") == "alive"
    ]
    known_dead_villagers = [
        item["target_id"]
        for item in inspections
        if item.get("alignment") == "village" and item.get("status") == "dead"
    ]

    alive_order: list[str] = []
    if isinstance(public_state, Mapping):
        players = public_state.get("players")
        if isinstance(players, list):
            for item in players:
                pid = _as_str_id(item)
                if pid and pid not in alive_order:
                    alive_order.append(pid)
        elif isinstance(players, Mapping):
            for key, item in players.items():
                pid = _as_str_id(item) or _as_str_id(key)
                if pid and pid not in alive_order:
                    alive_order.append(pid)
    if not alive_order:
        alive_order = list(alive_ids)
    if not alive_ids and alive_order and isinstance(public_state, Mapping):
        dead_ids = set()
        for key in ("dead_players", "dead_player_ids", "eliminated_ids", "deaths"):
            dead_ids.update(_extract_id_list(public_state.get(key)))
        alive_ids.update(pid for pid in alive_order if pid not in dead_ids)
    uninspected_alive_ids = [pid for pid in alive_order if pid in alive_ids and pid not in inspected_targets]

    self_alive = True
    for source in (self_state, private_information):
        if isinstance(source, Mapping):
            status = _extract_player_status(source)
            if status is not None:
                self_alive = status == "alive"
                break
            if source.get("is_dead") is not None:
                self_alive = not bool(source.get("is_dead"))
                break

    phase = None
    round_no = None
    if isinstance(game, Mapping):
        phase = game.get("public_phase", game.get("phase"))
        round_no = game.get("round")
    if phase is None and isinstance(request, Mapping):
        phase = request.get("phase")

    sheriff_id = _extract_sheriff_id(turn_packet)

    return {
        "role": "seer",
        "round": round_no,
        "phase": phase,
        "self_alive": self_alive,
        "sheriff_id": sheriff_id,
        "inspections": inspections,
        "known_live_wolves": known_live_wolves,
        "known_dead_wolves": known_dead_wolves,
        "known_live_villagers": known_live_villagers,
        "known_dead_villagers": known_dead_villagers,
        "uninspected_alive_ids": uninspected_alive_ids,
        "alive_player_ids": [pid for pid in alive_order if pid in alive_ids],
        "player_order": alive_order,
    }


def _seer_instruction_text(request: Mapping[str, Any], seer_context: Mapping[str, Any]) -> str:
    raw_allowed_actions = request.get("allowed_actions") if isinstance(request, Mapping) else None
    allowed_actions = [item for item in raw_allowed_actions if isinstance(item, Mapping)] if isinstance(raw_allowed_actions, list) else []
    speech_allowed = any(str(item.get("kind") or "") in _SPEECH_ACTION_KINDS for item in allowed_actions)
    vote_allowed = any("vote" in str(item.get("kind") or "").lower() for item in allowed_actions)
    inspect_allowed = any("inspect" in str(item.get("kind") or "").lower() for item in allowed_actions)
    transfer_allowed = any("transfer" in str(item.get("kind") or "").lower() and "badge" in str(item.get("kind") or "").lower() for item in allowed_actions)

    def _join(ids: Any) -> str:
        values = _extract_id_list(ids)
        return "、".join(values) if values else "无"

    signals = seer_context.get("public_signals", {})
    lines = [
        "【预言家专用状态机】先核对 seer_context 和 public_signals，再输出唯一合法 JSON。",
        f"查验账本：活狼={_join(seer_context.get('known_live_wolves'))}；死狼={_join(seer_context.get('known_dead_wolves'))}；活金水={_join(seer_context.get('known_live_villagers'))}；未验活人={_join(seer_context.get('uninspected_alive_ids'))}。",
        f"公开信号：对跳={_join(signals.get('direct_seer_counterclaims'))}；推/票金水={_join(signals.get('suggested_vote_targets'))}；近期关键票手={_join(signals.get('recent_elimination_voters'))}。",
        "查验事实与票型推测必须分开；不能把死人、已验目标或对跳狼给出的低信息金水作为优先目标。",
        "已跳预言家或持警徽后，不能长期说‘先听一圈/不归票’，除非没有对跳且没有金水受压。",
    ]
    if inspect_allowed:
        lines.append("夜验排序（仅限合法 target_ids）：1直接对跳预言家；2推/票已知金水者；3关键保踩、带票、分票位；4普通未验活人。优先见 suggested_inspect_targets。")
    if speech_allowed:
        lines.extend([
            "【白天/竞选发言固定短骨架，最多200字】查验事实：N1 x=结果；N2 y=结果。公开矛盾：对跳/推金水/异常票型。今天主票：A；备票：B（均为推测）。今晚主验：C；备验：D。若我夜死且无遗言：警徽给存活金水E；优先盘A/C及其票型链。",
            "必须给单一主票口、备票口、今晚主验和警徽流遗产；没有查杀也要形成票口。",
        ])
    if vote_allowed:
        lines.append("投票优先级：合法活查杀 > 直接对跳预言家 > 推/票已知金水者 > 造成金水出局的集中票型核心；不得投已知活金水或无关分散目标。")
    if transfer_allowed:
        lines.append("警徽移交只优先存活金水，其次站边清晰且未保狼的存活好人；不要移交死人。")
    return "\n".join(lines)


def _seer_action_group(kind: str) -> str | None:
    normalized = kind.strip().lower()
    if normalized in {"day_vote", "vote"} or normalized.endswith("_vote"):
        return "vote"
    if normalized in {"sheriff_badge_transfer", "badge_transfer", "transfer_badge", "sheriff_transfer"}:
        return "badge_transfer"
    if "inspect" in normalized or "check" in normalized or "seer" in normalized and "target" in normalized:
        return "inspect"
    if "speech_order" in normalized or "speak_order" in normalized or "sheriff_order" in normalized:
        return "speech_order"
    if "vote" in normalized and "speech" not in normalized:
        return "vote"
    if "badge" in normalized and "transfer" in normalized:
        return "badge_transfer"
    return None


def _apply_seer_safety_overrides(
    action: dict[str, Any],
    request: Mapping[str, Any],
    seer_context: Mapping[str, Any],
    public_signals: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(action, dict):
        return action
    group = _seer_action_group(str(action.get("kind") or ""))
    if group is None:
        return action
    allowed_actions = request.get("allowed_actions") if isinstance(request, Mapping) else None
    if not isinstance(allowed_actions, list):
        return action
    allowed = next(
        (
            item
            for item in allowed_actions
            if isinstance(item, Mapping) and str(item.get("kind") or "").strip() == str(action.get("kind") or "").strip()
        ),
        None,
    )
    if not isinstance(allowed, Mapping):
        return action
    target_ids = allowed.get("target_ids")
    if not isinstance(target_ids, list) or not target_ids:
        return action
    target_id_set = {str(target_id) for target_id in target_ids}
    signals = public_signals or seer_context.get("public_signals", {})
    if group == "vote":
        known_wolves = [str(pid) for pid in seer_context.get("known_live_wolves", []) if str(pid) in target_id_set]
        pressure = [str(pid) for pid in signals.get("suggested_vote_targets", []) if str(pid) in target_id_set]
        known_gold = {str(pid) for pid in seer_context.get("known_live_villagers", [])}
        current = str(action.get("target_id") or "")
        # 查杀最高优先；无查杀日若模型投金水或无关目标，收束到公开压力链。
        preferred_targets = known_wolves or (
            [pid for pid in pressure if pid not in known_gold]
            if current in known_gold or current not in pressure
            else []
        )
    elif group == "badge_transfer":
        preferred_targets = [str(pid) for pid in seer_context.get("known_live_villagers", []) if str(pid) in target_id_set]
    elif group == "inspect":
        ranked = _rank_seer_inspection_targets(request, seer_context, signals)
        preferred_targets = [] if ranked and str(action.get("target_id") or "") == ranked[0] else ranked
    else:
        preferred_targets = []
        if str(seer_context.get("sheriff_id")) == str(request.get("player_id")) and signals.get("suggested_vote_targets"):
            order = [str(pid) for pid in seer_context.get("player_order", [])]
            me = str(request.get("player_id"))
            candidates = [str(pid) for pid in target_ids if str(pid) in order and str(pid) != me]
            if candidates and me in order:
                start = order.index(me)
                preferred_targets = sorted(candidates, key=lambda pid: (order.index(pid) - start) % len(order), reverse=True)
    if not preferred_targets:
        return action
    overridden = dict(action)
    overridden["target_id"] = preferred_targets[0]
    return overridden


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
        self._seer_memory: dict[str, Any] = {
            "public_claims_by_player": {},
            "seer_claimants": [],
            "public_gold_targets": {},
            "inspection_claims_public": [],
            "vote_history": [],
            "eliminated_ids": [],
            "dawn_dead_ids": [],
            "last_sheriff_id": None,
            "last_pressure_targets": [],
        }
        self._model_token_usage: dict[str, int] = {
            "successful_response_count": 0,
            "api_attempt_count": 0,
            "reported_usage_response_count": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }

    async def observe(self, sync_packet: dict[str, Any]) -> None:
        """只把同步包中的公开发言、票型、死亡和警徽变化记入局内账本。"""

        if self.profile.role == "seer":
            _update_seer_memory(self._seer_memory, sync_packet)

    async def decide(self, turn_packet: dict[str, Any]) -> dict[str, Any]:
        if turn_packet["self"]["player_id"] != self.player_id:
            raise ValueError("行动包被发送给了错误玩家")
        private = turn_packet["private_information"]
        if str(private.get("role")) != self.profile.role:
            raise ValueError("Task-Agent 的角色档案与行动包身份不一致")

        if self.profile.role == "seer":
            _update_seer_memory(self._seer_memory, turn_packet)
            tool_context = turn_packet.get("tool_context") or {}
            dialogue_packet = {"dialogue": tool_context.get("current_round_dialogue", [])}
            _update_seer_memory(self._seer_memory, dialogue_packet)
        system = self._system_prompt(private)
        seer_context = _seer_context(turn_packet) if self.profile.role == "seer" else {}
        if self.profile.role == "seer":
            public_signals = _extract_public_signals(turn_packet, self._seer_memory, seer_context)
            seer_context["public_signals"] = public_signals
            self._seer_memory["last_pressure_targets"] = list(public_signals.get("suggested_vote_targets", []))
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
        if self.profile.role == "seer":
            prompt["role_context"] = {"seer": seer_context}

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
                "instruction": self._turn_instruction(
                    turn_packet["request"],
                    feedback,
                    role=self.profile.role,
                    seer_context=seer_context,
                ),
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
            if self.profile.role == "seer":
                action = _apply_seer_safety_overrides(
                    action,
                    turn_packet["request"],
                    seer_context,
                    seer_context.get("public_signals", {}),
                )
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
            "role_context": _SEER_ROLE_CONTEXT_TAG if self.profile.role == "seer" else "base_v1",
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
        *,
        role: str = "",
        seer_context: Mapping[str, Any] | None = None,
    ) -> str:
        instruction = render_prompt(
            "player_turn_instruction.txt",
            action_contract=render_action_contract(request),
            validation_feedback=(f"上一次输出未通过校验：{feedback}" if feedback else ""),
        )
        if role == "seer":
            instruction = instruction + "\n" + _seer_instruction_text(request, seer_context or {})
        return instruction

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

"""最小可运行的角色 Task-Agent。

每次行动从一个新的模型会话开始。模型默认只收到当前状态；本轮对话只能通过唯一的
受限工具按需读取。这里只保留预言家的有界结构化账本，不包含策略进化、外部检索或其他 Harness。
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
_SPEECH_ACTION_KINDS = frozenset({
    "speak", "last_words", "day_discussion", "sheriff_election_speech",
})
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


def _current_round_dialogue_summary(turn_packet: Mapping[str, Any]) -> dict[str, Any]:
    """Return bounded freshness metadata, never the dialogue itself."""
    tool_context = turn_packet.get("tool_context")
    raw_dialogue = tool_context.get("current_round_dialogue") if isinstance(tool_context, Mapping) else []
    if isinstance(raw_dialogue, (list, tuple)):
        entries = list(raw_dialogue)
    elif isinstance(raw_dialogue, Mapping):
        # A single dialogue record is also commonly represented as a mapping.
        has_record_fields = any(
            key in raw_dialogue for key in ("text", "content", "speaker_id", "player_id", "speaker")
        )
        entries = [raw_dialogue] if has_record_fields else list(raw_dialogue.values())
    elif raw_dialogue:
        entries = [raw_dialogue]
    else:
        entries = []

    speaker_ids: list[str] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        speaker = _as_str_id(
            entry.get("speaker_id")
            or entry.get("player_id")
            or entry.get("speaker")
            or entry.get("author")
            or entry.get("actor")
            or entry.get("from")
        )
        if speaker and speaker not in speaker_ids:
            speaker_ids.append(speaker)
    game = turn_packet.get("game")
    game = game if isinstance(game, Mapping) else {}
    phase = game.get("public_phase", game.get("phase"))
    request = turn_packet.get("request")
    if phase is None and isinstance(request, Mapping):
        phase = request.get("phase")
    return {
        "current_round_dialogue_count": len(entries),
        "current_round_speaker_ids": speaker_ids[:32],
        "current_sheriff_id": _extract_sheriff_id({"public_state": turn_packet.get("public_state")}),
        "current_phase": phase,
    }


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


def _extract_dead_player_ids(public_state: Mapping[str, Any]) -> list[str]:
    """从公开状态提取死亡名单；不读取或推断任何隐藏阵营字段。"""
    if not isinstance(public_state, Mapping):
        return []
    for key in ("dead_players", "dead_player_ids", "deceased_players", "eliminated_players"):
        ids = _extract_id_list(public_state.get(key))
        if ids:
            return ids
    players = public_state.get("players")
    dead: list[str] = []
    if isinstance(players, list):
        items = players
    elif isinstance(players, Mapping):
        items = []
        for key, item in players.items():
            if isinstance(item, Mapping):
                payload = dict(item)
                payload.setdefault("player_id", key)
                items.append(payload)
    else:
        items = []
    for item in items:
        pid = _as_str_id(item)
        if pid and _extract_player_status(item) == "dead" and pid not in dead:
            dead.append(pid)
    return dead


def _game_id(turn_packet: Mapping[str, Any]) -> str:
    for source in (turn_packet.get("game"), turn_packet.get("public_state")):
        if isinstance(source, Mapping):
            for key in ("game_id", "id", "session_id", "match_id"):
                value = source.get(key)
                if value is not None:
                    return str(value)
    # 没有显式 game_id 的协议仍应在最小状态变化时避免跨局污染；request_id
    # 通常含有局标识，但这里只把它作为隔离键，不把它发送给模型。
    request = turn_packet.get("request")
    if isinstance(request, Mapping) and request.get("game_id") is not None:
        return str(request["game_id"])
    return "unknown"


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
    freshness = _current_round_dialogue_summary(turn_packet)
    dead_ids = _extract_dead_player_ids(public_state) if isinstance(public_state, Mapping) else []
    inspection_map = {
        str(item["target_id"]): {
            key: value for key, value in item.items() if key != "target_id"
        }
        for item in inspections
        if item.get("target_id")
    }

    return {
        "role": "seer",
        "round": round_no,
        "phase": phase,
        "self_alive": self_alive,
        "sheriff_id": sheriff_id,
        "alive_player_ids": alive_order,
        "dead_player_ids": dead_ids,
        "inspection_map": inspection_map,
        "inspections": inspections,
        "known_live_wolves": known_live_wolves,
        "known_dead_wolves": known_dead_wolves,
        "known_live_villagers": known_live_villagers,
        "known_dead_villagers": known_dead_villagers,
        "uninspected_alive_ids": uninspected_alive_ids,
        "current_round_dialogue_count": freshness["current_round_dialogue_count"],
        "current_round_speaker_ids": freshness["current_round_speaker_ids"],
        "current_sheriff_id": freshness["current_sheriff_id"],
        "current_phase": freshness["current_phase"],
    }


def _public_id_values(value: Any) -> list[str]:
    """只在公开状态中收集玩家 id，供目标排序使用。"""
    if isinstance(value, Mapping):
        result: list[str] = []
        for nested in value.values():
            result.extend(_public_id_values(nested))
        return list(dict.fromkeys(result))
    if isinstance(value, (list, tuple, set)):
        result: list[str] = []
        for nested in value:
            result.extend(_public_id_values(nested))
        return list(dict.fromkeys(result))
    pid = _as_str_id(value)
    return [pid] if pid else []


def _rank_seer_inspect_targets(
    request: Mapping[str, Any],
    seer_context: Mapping[str, Any],
    public_state: Mapping[str, Any],
) -> list[str]:
    """按公开证据给合法夜验候选排序；绝不产生候选集之外的 id。"""
    allowed_actions = request.get("allowed_actions") if isinstance(request, Mapping) else None
    inspect_action = next(
        (item for item in allowed_actions or []
         if isinstance(item, Mapping) and str(item.get("kind") or "") == "seer_inspect"),
        None,
    )
    raw_targets = inspect_action.get("target_ids") if isinstance(inspect_action, Mapping) else []
    candidates = [str(pid) for pid in raw_targets or [] if str(pid) != str(request.get("player_id") or "")]
    candidates = list(dict.fromkeys(candidates))
    if not candidates:
        return []
    alive = {str(pid) for pid in seer_context.get("alive_player_ids", [])}
    dead = {str(pid) for pid in seer_context.get("dead_player_ids", [])}
    inspected = {str(pid) for pid in seer_context.get("inspection_map", {})}
    gold = {str(pid) for pid in seer_context.get("known_live_villagers", [])}
    order = {pid: index for index, pid in enumerate(candidates)}
    scores = {pid: 0 for pid in candidates}

    # 这些分数只来自 public_state 的字段名和值，不从自然语言猜阵营。
    def walk(value: Any, key_hint: str = "") -> None:
        hint = key_hint.lower()
        ids = [pid for pid in _public_id_values(value) if pid in scores]
        if any(token in hint for token in ("claim", "claimed", "对跳", "role_claim")):
            for pid in ids:
                scores[pid] += 4
        elif any(token in hint for token in ("vote", "ballot", "投票", "归票")):
            for pid in ids:
                scores[pid] += 2
        elif any(token in hint for token in ("sheriff", "badge", "警长", "警徽")):
            for pid in ids:
                scores[pid] += 1
        if isinstance(value, Mapping):
            for key, nested in value.items():
                walk(nested, str(key))
        elif isinstance(value, (list, tuple, set)):
            for nested in value:
                walk(nested, hint)

    if isinstance(public_state, Mapping):
        walk(public_state)
    for pid in candidates:
        if pid in alive and pid not in dead:
            scores[pid] += 2
        if pid in inspected:
            scores[pid] -= 8
        if pid in gold:
            scores[pid] -= 5
        if pid in dead:
            scores[pid] -= 10
    return sorted(candidates, key=lambda pid: (-scores[pid], order[pid]))


def _seer_instruction_text(
    request: Mapping[str, Any],
    seer_context: Mapping[str, Any],
    public_state: Mapping[str, Any] | None = None,
) -> str:
    raw_allowed_actions = request.get("allowed_actions") if isinstance(request, Mapping) else None
    allowed_actions = [item for item in raw_allowed_actions if isinstance(item, Mapping)] if isinstance(raw_allowed_actions, list) else []
    kinds = {str(item.get("kind") or "") for item in allowed_actions}
    speech_allowed = bool(kinds & _SPEECH_ACTION_KINDS)
    vote_allowed = any("vote" in kind.lower() for kind in kinds)
    inspect_allowed = "seer_inspect" in kinds
    transfer_allowed = any("transfer" in kind.lower() and "badge" in kind.lower() for kind in kinds)

    def _join(ids: Any) -> str:
        values = _extract_id_list(ids)
        return "、".join(values) if values else "无"

    ranked = _rank_seer_inspect_targets(request, seer_context, public_state or {}) if inspect_allowed else []
    live_wolves = _join(seer_context.get("known_live_wolves"))
    live_gold = _join(seer_context.get("known_live_villagers"))
    lines = [
        "【预言家专用：短、事实优先】",
        f"查验事实账本：活查杀={live_wolves}；死查杀={_join(seer_context.get('known_dead_wolves'))}；活金水={live_gold}；死金水={_join(seer_context.get('known_dead_villagers'))}。",
        "只有 seer_context.inspections 中的记录才是查验事实；不得补写、改写或把推测当查验。死亡目标不能作为今天归票。",
        "票型、发言和站边只能写‘我推测/票型显示/需要验证’，禁止用‘确定、明显帮狼、狼队协作’把有限关联说成身份事实。",
        "当前轮新鲜度摘要："
        f"current_round_dialogue_count={seer_context.get('current_round_dialogue_count', 0)}；"
        f"current_round_speaker_ids={_join(seer_context.get('current_round_speaker_ids'))}；"
        f"current_sheriff_id={seer_context.get('current_sheriff_id') or '无'}；"
        f"current_phase={seer_context.get('current_phase') or request.get('phase') or 'unknown'}。",
    ]
    if inspect_allowed:
        lines.append("本次合法夜验候选（已排序，仅可从 allowed_actions.target_ids 选择）：" + (_join(ranked[:5]) if ranked else "无") + "。")
    if speech_allowed:
        lines.extend([
            "每次发言前必须先调用 read_current_round_dialogue；以工具返回的当前轮内容和当前 public_state 覆盖任何旧摘要。不得把完整历史自动拼入上下文。",
            "每次发言必须重新核对当前 phase 和当前 sheriff_id；当前轮已有发言不得声称‘无人发言’，已有警长不得声称‘警徽空缺’。不得复制上一阶段或上一轮的整段话术。",
            "发言严格按四段且每段短句：【查验事实】；【公开事实】（存活、公开身份声明、公开票型/发言）；【推测】；【行动】。",
            "【行动】只给一个主投目标、一个条件性备选、一个下一验目标；不要列多个并列主线。主投必须是存活者。",
            "有活查杀时第一句先公开‘查杀目标+其存活状态’；若有未公开活金水且自己可能夜死，同时公开金水。",
            "无活查杀时不归票死人，只讨论该查杀的保人、带票、分票、强踩链，并把结论保持为推测。高压阶段优先短句，禁止 pass/空泛发言。",
        ])
        if "day_discussion" in kinds:
            lines.append("day_discussion 短模式：先报活查杀，再报一个主票和一个条件备选；最多五个要点。")
        if "sheriff_election_speech" in kinds:
            lines.append("sheriff_election_speech 短模式：公开查验链、唯一主线、警徽流和下一验；不把竞选票型说成身份事实。")
        if "last_words" in kinds:
            lines.append("last_words 短模式：优先留下活查杀、未公开金水、唯一主票和下一验；不要复述长历史。")
        if str(seer_context.get("sheriff_id") or "") == str(request.get("player_id") or ""):
            lines.append("你已有警徽：行动段只保留一个明确主票，并给出条件性备选。")
    if vote_allowed:
        lines.append("投票顺序：存活已验狼人 > 单一推测主线；不要把多个目标同时写成主票。")
    if transfer_allowed:
        lines.append("警徽移交优先存活金水；没有明确金水时选择公开站边清晰且未保狼的存活好人。")
    if not speech_allowed and not inspect_allowed and not vote_allowed:
        lines.append("本次只执行当前 allowed_actions，不臆造其他行动。")
    return "\n".join(lines)


def _seer_action_group(kind: str) -> str | None:
    normalized = kind.strip().lower()
    if normalized in {"day_vote", "vote"} or normalized.endswith("_vote"):
        return "vote"
    if normalized in {"sheriff_badge_transfer", "badge_transfer", "transfer_badge", "sheriff_transfer"}:
        return "badge_transfer"
    if "vote" in normalized and "speech" not in normalized:
        return "vote"
    if "badge" in normalized and "transfer" in normalized:
        return "badge_transfer"
    return None


def _apply_seer_safety_overrides(
    action: dict[str, Any],
    request: Mapping[str, Any],
    seer_context: Mapping[str, Any],
    public_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(action, dict):
        return action
    kind = str(action.get("kind") or "")
    if kind == "seer_inspect":
        ranked = _rank_seer_inspect_targets(request, seer_context, public_state or {})
        allowed = next(
            (item for item in request.get("allowed_actions", [])
             if isinstance(item, Mapping) and str(item.get("kind") or "") == kind),
            None,
        )
        target_ids = {str(pid) for pid in (allowed.get("target_ids", []) if isinstance(allowed, Mapping) else [])}
        alive = {str(pid) for pid in seer_context.get("alive_player_ids", [])}
        dead = {str(pid) for pid in seer_context.get("dead_player_ids", [])}
        inspected = {str(pid) for pid in seer_context.get("inspection_map", {})}
        gold = {str(pid) for pid in seer_context.get("known_live_villagers", [])}
        # 只有公开状态有可排序证据时才做有限纠偏；否则保留模型的合法选择。
        public_keys = {str(key).lower() for key in (public_state or {}).keys()}
        has_signal = any(any(token in key for token in ("vote", "claim", "sheriff", "badge", "警")) for key in public_keys)
        preferred = next(
            (pid for pid in ranked[:3] if pid in target_ids and pid in alive and pid not in dead
             and pid not in inspected and pid not in gold),
            None,
        )
        current = str(action.get("target_id"))
        current_is_invalid_or_repeated = (
            current not in target_ids or current == str(request.get("player_id") or "")
            or current in dead or current in inspected
        )
        if not has_signal and not current_is_invalid_or_repeated:
            preferred = None
        if preferred is not None and current != preferred:
            overridden = dict(action)
            overridden["target_id"] = preferred
            return overridden
        return action
    group = _seer_action_group(kind)
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
    if group == "vote":
        preferred_targets = [str(pid) for pid in seer_context.get("known_live_wolves", []) if str(pid) in target_id_set]
    else:
        preferred_targets = [str(pid) for pid in seer_context.get("known_live_villagers", []) if str(pid) in target_id_set]
    if not preferred_targets:
        return action
    overridden = dict(action)
    overridden["target_id"] = preferred_targets[0]
    return overridden


def _compact_seer_packet(
    turn_packet: Mapping[str, Any],
    seer_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """删去角色无关元数据，保留当前行动契约和预言家查验记录。"""
    context = seer_context or _seer_context(turn_packet)
    game = turn_packet.get("game") if isinstance(turn_packet.get("game"), Mapping) else {}
    public_state = turn_packet.get("public_state") if isinstance(turn_packet.get("public_state"), Mapping) else {}
    private = turn_packet.get("private_information") if isinstance(turn_packet.get("private_information"), Mapping) else {}
    self_state = turn_packet.get("self") if isinstance(turn_packet.get("self"), Mapping) else {}
    request = turn_packet.get("request") if isinstance(turn_packet.get("request"), Mapping) else {}

    def pick(source: Mapping[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
        return {key: source[key] for key in keys if key in source}

    compact_players: list[dict[str, Any]] = []
    players = public_state.get("players")
    if isinstance(players, list):
        for item in players:
            pid = _as_str_id(item)
            if pid:
                row: dict[str, Any] = {"player_id": pid}
                status = _extract_player_status(item)
                if status:
                    row["status"] = status
                compact_players.append(row)
    elif isinstance(players, Mapping):
        for key, item in players.items():
            pid = _as_str_id(item) or _as_str_id(key)
            if pid:
                row = {"player_id": pid}
                status = _extract_player_status(item)
                if status:
                    row["status"] = status
                compact_players.append(row)

    public_keys = (
        "alive_players", "alive_player_ids", "dead_players", "dead_player_ids",
        "sheriff_id", "current_sheriff_id", "badge_owner_id", "players",
        "votes", "vote_history", "sheriff_votes", "claims", "role_claims",
        "public_claims", "announcements",
    )
    compact_public = pick(public_state, public_keys)
    if compact_players:
        compact_public["players"] = compact_players
    compact_public.update({
        "alive_player_ids": context.get("alive_player_ids", []),
        "dead_player_ids": context.get("dead_player_ids", []),
        "sheriff_id": context.get("sheriff_id"),
    })
    compact_private = pick(private, ("role", "team", "role_state", "seer_state"))
    # role_state 可能带有与预言家无关的字段，只复制查验相关字段。
    for state_key in ("role_state", "seer_state"):
        state = compact_private.get(state_key)
        if isinstance(state, Mapping):
            compact_private[state_key] = pick(state, (
                "inspections", "inspection", "inspection_results", "inspection_history",
                "checked_players", "checks", "last_inspection", "latest_inspection",
            ))
    compact_request = pick(request, ("request_id", "player_id", "phase", "channel"))
    raw_allowed = request.get("allowed_actions")
    compact_request["allowed_actions"] = [
        pick(item, ("kind", "target_ids", "max_chars", "require_chinese"))
        for item in raw_allowed if isinstance(item, Mapping)
    ] if isinstance(raw_allowed, list) else []
    return {
        "game": pick(game, ("game_id", "id", "round", "phase", "public_phase", "alive_player_ids", "dead_player_ids")),
        "public_rules": pick(turn_packet.get("public_rules", {}) if isinstance(turn_packet.get("public_rules"), Mapping) else {},
                              ("phases", "actions", "visibility", "win_conditions")),
        "self": pick(self_state, ("player_id", "alive", "is_alive", "status", "is_sheriff")),
        "private_information": compact_private,
        "public_state": compact_public,
        "request": compact_request,
        "role_context": context,
        "current_round_freshness": {
            "current_round_dialogue_count": context.get("current_round_dialogue_count", 0),
            "current_round_speaker_ids": list(context.get("current_round_speaker_ids", []))[:32],
            "current_sheriff_id": context.get("current_sheriff_id"),
            "current_phase": context.get("current_phase"),
        },
    }


def _seer_speech_semantic_error(
    action: Mapping[str, Any],
    request: Mapping[str, Any],
    seer_context: Mapping[str, Any],
    public_state: Mapping[str, Any] | None = None,
    *,
    previous_speech: str | None = None,
    previous_speech_digest: str | None = None,
    previous_round: Any = None,
) -> str | None:
    """Reject only public, deterministic stale-fact speech patterns."""
    if str(action.get("kind") or "") not in _SPEECH_ACTION_KINDS:
        return None
    text = str(action.get("text") or "").strip().lower()
    if not text:
        return None
    try:
        dialogue_count = int(seer_context.get("current_round_dialogue_count") or 0)
    except (TypeError, ValueError):
        dialogue_count = 0
    if dialogue_count > 0 and any(
        phrase in text for phrase in ("无人发言", "没有人发言", "没人发言")
    ):
        return "当前公开事实与文本冲突，请重新按当前轮状态作答"
    public_sheriff_id = _extract_sheriff_id({"public_state": public_state or {}})
    if public_sheriff_id and any(
        phrase in text for phrase in ("警徽空缺", "没有警长", "无警长", "没有警徽")
    ):
        return "当前公开事实与文本冲突，请重新按当前轮状态作答"
    current_round = seer_context.get("round")
    if (
        (previous_speech or previous_speech_digest)
        and previous_round is not None
        and current_round is not None
        and previous_round == current_round
        and (
            (previous_speech is not None and text == previous_speech.strip().lower())
            or (
                previous_speech_digest is not None
                and hashlib.sha256(text.encode("utf-8")).hexdigest() == previous_speech_digest
            )
        )
    ):
        return "当前公开事实与文本冲突，请重新按当前轮状态作答"
    return None


def _run_seer_self_checks() -> bool:
    """轻量回归检查；只验证本文件的解析、压缩和合法目标边界。"""
    packet: dict[str, Any] = {
        "game": {"game_id": "self-check", "round": 2, "phase": "night"},
        "public_rules": {},
        "self": {"player_id": "p0", "alive": True},
        "private_information": {
            "role": "seer", "team": "village",
            "role_state": {"inspections": {"p1": "wolf", "p2": "village"}},
        },
        "public_state": {
            "players": [{"player_id": "p0", "alive": True}, {"player_id": "p1", "alive": False},
                        {"player_id": "p2", "alive": True}, {"player_id": "p3", "alive": True}],
            "dead_player_ids": ["p1"], "claims": [{"player_id": "p3", "role": "seer"}],
        },
        "request": {"request_id": "r", "player_id": "p0", "allowed_actions": [
            {"kind": "seer_inspect", "target_ids": ["p0", "p1", "p2", "p3"]},
            {"kind": "day_discussion", "max_chars": 100, "require_chinese": True},
        ]},
    }
    context = _seer_context(packet)
    if context["inspection_map"].get("p1", {}).get("alignment") != "wolf":
        return False
    if "p1" in context["known_live_wolves"]:
        return False
    ranked = _rank_seer_inspect_targets(packet["request"], context, packet["public_state"])
    if not ranked or ranked[0] not in {"p2", "p3"} or "p0" in ranked:
        return False
    compact = _compact_seer_packet(packet, context)
    if "allowed_actions" not in compact["request"] or "role_state" not in compact["private_information"]:
        return False
    speech = _seer_instruction_text(packet["request"], context, packet["public_state"])
    if "【推测】" not in speech or "【查验事实】" not in speech:
        return False

    speech_request = {"allowed_actions": [{"kind": "day_discussion", "max_chars": 200}]}
    current_context = dict(context)
    current_context.update({
        "round": 2,
        "current_round_dialogue_count": 1,
        "current_round_speaker_ids": ["p2"],
        "current_sheriff_id": "p3",
    })
    stale_text = "目前无人发言，警徽空缺。"
    if _seer_speech_semantic_error(
        {"kind": "day_discussion", "text": stale_text},
        speech_request,
        current_context,
        {"sheriff_id": "p3"},
        previous_speech=stale_text,
        previous_round=2,
    ) is None:
        return False
    election_context = dict(context)
    election_context.update({"round": 1, "current_round_dialogue_count": 0})
    if _seer_speech_semantic_error(
        {"kind": "sheriff_election_speech", "text": stale_text},
        speech_request,
        election_context,
        {},
    ) is not None:
        return False
    if decision_error(
        {"kind": "seer_inspect", "target_id": "p3"},
        {"allowed_actions": [{"kind": "seer_inspect", "target_ids": ["p2", "p3"]}]},
    ) is not None:
        return False
    if decision_error(
        {"kind": "wolf_kill", "target_id": "p3"},
        {"allowed_actions": [{"kind": "wolf_kill", "target_ids": ["p2", "p3"]}]},
    ) is not None:
        return False
    corrected = _apply_seer_safety_overrides(
        {"kind": "seer_inspect", "target_id": "p2"}, packet["request"], context, packet["public_state"]
    )
    return corrected.get("target_id") in {"p2", "p3"}


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
        if self.profile.role == "seer" and not _run_seer_self_checks():
            raise RuntimeError("预言家 Task-Agent 本地自检失败")
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
        # 有界、可验证的跨行动账本；只保存结构化承诺，不保存公开话术。
        self._seer_ledger: dict[str, Any] = {
            "game_id": None,
            "inspection_map": {},
            "alive_ids": [],
            "dead_ids": [],
            "last_public_round": None,
            "last_action_kind": None,
            "declared_inspection_targets": [],
            "declared_main_vote": None,
        }
        # 仅供本地防止同轮逐字复制；只保存摘要，永不注入模型 prompt。
        self._last_seer_speech_digest: str | None = None
        self._last_seer_speech_round: Any = None

    async def observe(self, sync_packet: dict[str, Any]) -> None:
        """接收公开同步；详细历史仍不跨行动保存。"""

        # 公开同步不进入账本；下一次 decide 会从当前 packet 重建可验证字段。
        del sync_packet

    async def decide(self, turn_packet: dict[str, Any]) -> dict[str, Any]:
        if turn_packet["self"]["player_id"] != self.player_id:
            raise ValueError("行动包被发送给了错误玩家")
        private = turn_packet["private_information"]
        if str(private.get("role")) != self.profile.role:
            raise ValueError("Task-Agent 的角色档案与行动包身份不一致")

        system = self._system_prompt(private)
        seer_context = _seer_context(turn_packet) if self.profile.role == "seer" else {}
        if self.profile.role == "seer":
            self._refresh_seer_ledger(turn_packet, seer_context)
        prompt = (
            _compact_seer_packet(turn_packet, seer_context)
            if self.profile.role == "seer"
            else {
                "game": turn_packet["game"],
                "public_rules": turn_packet["public_rules"],
                "self": turn_packet["self"],
                "private_information": private,
                "public_state": turn_packet["public_state"],
                "request": turn_packet["request"],
            }
        )
        if self.profile.role != "seer":
            prompt["history_policy"] = {
                "default_context": "current_state_only",
                "current_round_dialogue_tool": CURRENT_ROUND_DIALOGUE_TOOL_NAME,
                "max_tool_calls": self.max_tool_calls_per_decision,
                "max_tool_result_tokens": self.max_tool_result_tokens,
            }
        elif isinstance(prompt, dict):
            prompt["history_policy"] = {
                "default_context": "current_state_only",
                "current_round_dialogue_tool": CURRENT_ROUND_DIALOGUE_TOOL_NAME,
                "max_tool_calls": self.max_tool_calls_per_decision,
                "max_tool_result_tokens": self.max_tool_result_tokens,
            }
        if self.profile.role == "seer":
            # 账本只含短结构化字段；尤其不放入上一段公开发言。
            prompt["seer_ledger"] = dict(self._seer_ledger)
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
                    public_state=turn_packet.get("public_state") if isinstance(turn_packet.get("public_state"), Mapping) else {},
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
                    action, turn_packet["request"], seer_context,
                    turn_packet.get("public_state") if isinstance(turn_packet.get("public_state"), Mapping) else {},
                )
            error = decision_error(action, turn_packet["request"])
            if error is None and self.profile.role == "seer":
                error = _seer_speech_semantic_error(
                    action,
                    turn_packet["request"],
                    seer_context,
                    turn_packet.get("public_state") if isinstance(turn_packet.get("public_state"), Mapping) else {},
                    previous_speech_digest=self._last_seer_speech_digest,
                    previous_round=self._last_seer_speech_round,
                )
            if error is None:
                if self.profile.role == "seer":
                    self._record_seer_action(action, seer_context)
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

    def _refresh_seer_ledger(
        self,
        turn_packet: Mapping[str, Any],
        seer_context: Mapping[str, Any],
    ) -> None:
        current_game = _game_id(turn_packet)
        if self._seer_ledger.get("game_id") != current_game:
            self._seer_ledger = {
                "game_id": current_game,
                "inspection_map": {},
                "alive_ids": [],
                "dead_ids": [],
                "last_public_round": None,
                "last_action_kind": None,
                "declared_inspection_targets": [],
                "declared_main_vote": None,
            }
            self._last_seer_speech_digest = None
            self._last_seer_speech_round = None
        self._seer_ledger["inspection_map"] = dict(seer_context.get("inspection_map", {}))
        self._seer_ledger["alive_ids"] = list(seer_context.get("alive_player_ids", []))
        self._seer_ledger["dead_ids"] = list(seer_context.get("dead_player_ids", []))

    def _record_seer_action(
        self,
        action: Mapping[str, Any],
        seer_context: Mapping[str, Any],
    ) -> None:
        kind = str(action.get("kind") or "")
        target = action.get("target_id")
        if kind == "seer_inspect" and target is not None:
            targets = list(self._seer_ledger.get("declared_inspection_targets", []))
            target_text = str(target)
            if target_text not in targets:
                targets.append(target_text)
            self._seer_ledger["declared_inspection_targets"] = targets[-8:]
        if kind == "day_vote" and target is not None:
            self._seer_ledger["declared_main_vote"] = str(target)
        self._seer_ledger["last_action_kind"] = kind
        if kind in _SPEECH_ACTION_KINDS and isinstance(action.get("text"), str):
            # 只保存文本摘要用于本地防复制检查，不进入 seer_ledger 或模型 prompt。
            self._last_seer_speech_digest = hashlib.sha256(
                action["text"].strip().lower().encode("utf-8")
            ).hexdigest()
            self._last_seer_speech_round = seer_context.get("round")
            self._seer_ledger["last_public_round"] = seer_context.get("round")

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
        public_state: Mapping[str, Any] | None = None,
    ) -> str:
        instruction = render_prompt(
            "player_turn_instruction.txt",
            action_contract=render_action_contract(request),
            validation_feedback=(f"上一次输出未通过校验：{feedback}" if feedback else ""),
        )
        if role == "seer":
            instruction = instruction + "\n" + _seer_instruction_text(request, seer_context or {}, public_state or {})
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

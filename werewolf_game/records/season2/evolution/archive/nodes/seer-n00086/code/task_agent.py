"""最小可运行的角色 Task-Agent。

每次行动从一个新的模型会话开始。模型默认只收到当前状态；本轮对话只能通过唯一的
受限工具按需读取。预言家仅保留最近三轮的结构化公开摘要，不保存完整历史、隐藏状态或外部记忆。
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


def _short_text(value: Any, limit: int = 72) -> str:
    text = str(value or "").replace("\n", " ").strip()
    return text[:limit]


def _game_id(packet: Mapping[str, Any]) -> str:
    game = packet.get("game") if isinstance(packet, Mapping) else None
    if isinstance(game, Mapping):
        for key in ("game_id", "id", "match_id", "session_id"):
            value = game.get(key)
            if value is not None:
                return str(value)
    for key in ("game_id", "match_id"):
        value = packet.get(key) if isinstance(packet, Mapping) else None
        if value is not None:
            return str(value)
    return "unknown"


def _round_number(packet: Mapping[str, Any]) -> int | str | None:
    game = packet.get("game") if isinstance(packet, Mapping) else None
    request = packet.get("request") if isinstance(packet, Mapping) else None
    value = game.get("round") if isinstance(game, Mapping) else None
    if value is None and isinstance(request, Mapping):
        value = request.get("round")
    return value


def _all_visible_player_ids(packet: Mapping[str, Any]) -> list[str]:
    ids: list[str] = []
    public = packet.get("public_state") if isinstance(packet, Mapping) else None
    if isinstance(public, Mapping):
        ids.extend(_extract_alive_player_ids(public))
        players = public.get("players")
        if isinstance(players, list):
            ids.extend(pid for pid in (_as_str_id(item) for item in players) if pid)
        elif isinstance(players, Mapping):
            ids.extend(
                pid for key, item in players.items()
                for pid in [_as_str_id(item) or _as_str_id(key)] if pid
            )
    private = packet.get("private_information") if isinstance(packet, Mapping) else None
    if isinstance(private, Mapping):
        for item in _extract_inspection_records(private, set()):
            if item.get("target_id"):
                ids.append(str(item["target_id"]))
    return list(dict.fromkeys(ids))


def _dialogue_entry(entry: Any) -> tuple[str | None, str]:
    if isinstance(entry, Mapping):
        speaker = _as_str_id(
            entry.get("speaker_id") or entry.get("player_id") or entry.get("speaker")
            or entry.get("player") or entry.get("author")
        )
        text = entry.get("text") or entry.get("content") or entry.get("message") or entry.get("speech") or ""
        if isinstance(text, Mapping):
            text = text.get("text") or text.get("content") or ""
        return speaker, _short_text(text)
    return None, _short_text(entry)


def _ids_in_text(text: str, known_ids: set[str]) -> list[str]:
    found: list[str] = []
    for pid in sorted(known_ids):
        if re.search(r"(?<![A-Za-z0-9_])" + re.escape(pid) + r"(?![A-Za-z0-9_])", text, re.IGNORECASE):
            found.append(pid)
    # Common packets use p6/p11. This fallback is only a label extractor, not a source of facts.
    for pid in re.findall(r"(?<![A-Za-z0-9_])p\d+(?![A-Za-z0-9_])", text, re.IGNORECASE):
        if pid not in found:
            found.append(pid)
    return found[:6]


def _public_dialogue_summary(dialogue: Any, known_ids: set[str]) -> dict[str, Any]:
    claims: list[dict[str, str]] = []
    inspections: list[dict[str, str]] = []
    pressure: dict[str, set[str]] = {}
    votes: list[str] = []
    snippets: list[dict[str, str]] = []
    entries = dialogue if isinstance(dialogue, list) else []
    for raw in entries[-8:]:
        speaker, text = _dialogue_entry(raw)
        if not text:
            continue
        ids = _ids_in_text(text, known_ids)
        lower = text.lower()
        role = None
        if re.search(r"(我是|我跳|自称|认领).{0,8}(预言家|seer)", text, re.I) or re.search(r"\bseer\b", lower):
            role = "seer"
        elif re.search(r"(我是|我跳|自称|认领).{0,8}女巫", text):
            role = "witch"
        elif re.search(r"(我是|我跳|自称|认领).{0,8}(猎人|guard|守卫)", text, re.I):
            role = "other_claim"
        if role and speaker:
            claims.append({"player_id": speaker, "role": role})
        result = None
        if re.search(r"查杀|验出.{0,12}(狼人|狼|wolf)|验到.{0,12}(狼人|狼|wolf)|查到.{0,12}(狼人|狼|wolf)", text, re.I):
            result = "wolf"
        elif re.search(r"金水|银水|验出.{0,12}(好人|村民|平民|villager)|验到.{0,12}(好人|村民|平民|villager)|查到.{0,12}(好人|村民|平民|villager)", text, re.I):
            result = "village"
        if result and ids:
            inspections.append({"target_id": ids[0], "alignment": result})
        if re.search(r"(归票|投票|投|出|放逐|票型)", text):
            vote_ids = _ids_in_text(text, known_ids)
            if vote_ids:
                votes.append(vote_ids[-1])
        for pid in ids:
            labels: set[str] = pressure.setdefault(pid, set())
            if re.search(r"(强保|保|站边|认好|捞)", text) and pid != speaker:
                labels.add("protect")
            if re.search(r"(强踩|踩|打|怀疑|狼面|优先出)", text) and pid != speaker:
                labels.add("attack")
            if re.search(r"(归票|投|放逐|出)", text):
                labels.add("vote_link")
        if speaker or text:
            snippets.append({"speaker": speaker or "?", "text": text[:64]})
    unique_claims: list[dict[str, str]] = []
    for item in claims:
        if item not in unique_claims:
            unique_claims.append(item)
    claim_players = list(dict.fromkeys(item["player_id"] for item in unique_claims))
    return {
        "claims": unique_claims[:8],
        "claim_players": claim_players[:8],
        "inspections": inspections[:8],
        "pressure": {pid: sorted(labels) for pid, labels in list(pressure.items())[:12] if labels},
        "vote_targets": list(dict.fromkeys(votes))[:8],
        "snippets": snippets[-6:],
    }


class SeerDecisionMemory:
    """只保留最近三轮的结构化公开摘要；不保存完整对话或隐藏状态。"""

    def __init__(self) -> None:
        self.game_id = ""
        self.rounds: list[dict[str, Any]] = []

    def reset_if_needed(self, game_id: str) -> None:
        if self.game_id != game_id:
            self.game_id = game_id
            self.rounds = []

    def update(self, packet: Mapping[str, Any], dialogue: Any = None) -> dict[str, Any]:
        self.reset_if_needed(_game_id(packet))
        known_ids = set(_all_visible_player_ids(packet))
        summary = _public_dialogue_summary(dialogue, known_ids)
        summary["round"] = _round_number(packet)
        public = packet.get("public_state") if isinstance(packet, Mapping) else None
        summary["sheriff_id"] = _extract_sheriff_id(packet)
        summary["dead_ids"] = self._dead_ids(public)
        summary["public_votes"] = self._public_votes(public, known_ids)
        if isinstance(public, Mapping):
            public_claims: list[Any] = []
            for key in ("claims", "role_claims", "declarations", "public_declarations"):
                value = public.get(key)
                if isinstance(value, list):
                    public_claims.extend(value[-8:])
            if public_claims:
                extracted = _public_dialogue_summary(public_claims, known_ids)
                for key in ("claims", "claim_players", "inspections", "vote_targets", "snippets"):
                    summary[key] = list(dict.fromkeys(summary.get(key, []) + extracted.get(key, [])))[:12]
                merged = dict(summary.get("pressure", {}))
                for pid, tags in (extracted.get("pressure", {}) or {}).items():
                    merged[pid] = sorted(set(merged.get(pid, [])) | set(tags))
                summary["pressure"] = dict(list(merged.items())[:12])
        # Merge repeated decisions in one round (day/night packets may expose different slices),
        # while keeping every field bounded and avoiding duplicate entries.
        old = next((item for item in self.rounds if item.get("round") == summary.get("round")), None)
        if old is not None:
            for key in ("claims", "inspections", "vote_targets", "snippets", "dead_ids", "public_votes"):
                combined = list(old.get(key, [])) + list(summary.get(key, []))
                unique: list[Any] = []
                for value in combined:
                    if value not in unique:
                        unique.append(value)
                summary[key] = unique[-12:]
            merged_pressure = dict(old.get("pressure", {}))
            for pid, tags in (summary.get("pressure", {}) or {}).items():
                merged_pressure[pid] = sorted(set(merged_pressure.get(pid, [])) | set(tags))
            summary["pressure"] = dict(list(merged_pressure.items())[:12])
            summary["claim_players"] = list(dict.fromkeys(
                list(old.get("claim_players", [])) + list(summary.get("claim_players", []))
            ))[:8]
            if not summary.get("sheriff_id"):
                summary["sheriff_id"] = old.get("sheriff_id")
        self.rounds = [item for item in self.rounds if item.get("round") != summary.get("round")]
        self.rounds.append(summary)
        self.rounds = self.rounds[-3:]
        return summary

    @staticmethod
    def _dead_ids(public: Any) -> list[str]:
        if not isinstance(public, Mapping):
            return []
        dead: list[str] = []
        players = public.get("players")
        if isinstance(players, list):
            items = players
        elif isinstance(players, Mapping):
            items = list(players.values())
        else:
            items = []
        for item in items:
            pid = _as_str_id(item)
            if pid and _extract_player_status(item) == "dead":
                dead.append(pid)
        for key in ("dead_players", "dead_player_ids", "eliminated_players", "deceased"):
            dead.extend(_extract_id_list(public.get(key)))
        return list(dict.fromkeys(dead))[:24]

    @staticmethod
    def _public_votes(public: Any, known_ids: set[str]) -> list[dict[str, str]]:
        if not isinstance(public, Mapping):
            return []
        result: list[dict[str, str]] = []
        for key in ("votes", "vote_records", "voting", "last_votes", "ballots"):
            value = public.get(key)
            if isinstance(value, list):
                for item in value[-12:]:
                    if isinstance(item, Mapping):
                        voter = _as_str_id(item.get("voter") or item.get("player_id") or item.get("from"))
                        target = _as_str_id(item.get("target") or item.get("target_id") or item.get("vote"))
                        if voter and target and target in known_ids:
                            result.append({"voter": voter, "target": target})
        return result[-12:]

    def compact(self) -> list[dict[str, Any]]:
        return self.rounds[-3:]


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

    return {
        "role": "seer",
        "self_id": _as_str_id(self_state) or _as_str_id(turn_packet.get("self")),
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
        "alive_player_ids": alive_order,
    }


def _inspection_action_allowed(request: Mapping[str, Any]) -> Mapping[str, Any] | None:
    actions = request.get("allowed_actions") if isinstance(request, Mapping) else None
    if not isinstance(actions, list):
        return None
    for item in actions:
        if isinstance(item, Mapping) and "inspect" in str(item.get("kind") or "").lower():
            return item
    return None


def _seer_priority_order(
    request: Mapping[str, Any],
    seer_context: Mapping[str, Any],
    memory: SeerDecisionMemory | None = None,
) -> list[str]:
    allowed = _inspection_action_allowed(request)
    targets = [str(item) for item in (allowed.get("target_ids", []) if allowed else [])]
    if not targets:
        return []
    dead = {str(item) for item in seer_context.get("known_dead_wolves", [])}
    dead.update(str(item) for item in seer_context.get("known_dead_villagers", []))
    dead.update(str(item) for item in (memory.rounds[-1].get("dead_ids", []) if memory and memory.rounds else []))
    inspected = {str(item.get("target_id")) for item in seer_context.get("inspections", []) if item.get("target_id")}
    confirmed_good = {str(item) for item in seer_context.get("known_live_villagers", [])}
    excluded = dead | inspected | confirmed_good | {str(seer_context.get("self_id") or "")}
    candidates = [pid for pid in targets if pid not in excluded]
    if not candidates:
        candidates = [pid for pid in targets if pid not in dead and pid != str(seer_context.get("self_id") or "")]
    if not candidates:
        return targets

    score: dict[str, int] = {pid: 0 for pid in candidates}
    labels: dict[str, set[str]] = {pid: set() for pid in candidates}
    recent = memory.rounds[-3:] if memory else []
    claimants: set[str] = set()
    for summary in recent:
        claimants.update(str(item) for item in summary.get("claim_players", []))
        for pid, tag_list in (summary.get("pressure", {}) or {}).items():
            if str(pid) in labels:
                labels[str(pid)].update(str(tag) for tag in tag_list)
        for item in summary.get("inspections", []):
            target = str(item.get("target_id") or "")
            if target in labels:
                labels[target].add("public_inspection_claim")
        for item in summary.get("public_votes", []):
            target = str(item.get("target") or "")
            if target in labels:
                labels[target].add("vote_link")
    for pid in claimants:
        if pid in labels:
            labels[pid].add("unresolved_claim")
    sheriff = str(seer_context.get("sheriff_id") or "")
    for pid in candidates:
        tags = labels[pid]
        if "unresolved_claim" in tags:
            score[pid] += 90
        if "attack" in tags:
            score[pid] += 35
        if "protect" in tags:
            score[pid] += 30
        if "vote_link" in tags:
            score[pid] += 25
        if "public_inspection_claim" in tags:
            score[pid] += 20
        if pid == sheriff:
            score[pid] += 12
    # Stable allowed_actions order is the tie-break, never an invented player id.
    return sorted(candidates, key=lambda pid: (-score[pid], targets.index(pid)))


def _seer_instruction_text(request: Mapping[str, Any], seer_context: Mapping[str, Any]) -> str:
    raw_allowed_actions = request.get("allowed_actions") if isinstance(request, Mapping) else None
    allowed_actions = [item for item in raw_allowed_actions if isinstance(item, Mapping)] if isinstance(raw_allowed_actions, list) else []
    speech_allowed = any(str(item.get("kind") or "") in _SPEECH_ACTION_KINDS for item in allowed_actions)
    vote_allowed = any("vote" in str(item.get("kind") or "").lower() for item in allowed_actions)
    transfer_allowed = any(
        "transfer" in str(item.get("kind") or "").lower() and "badge" in str(item.get("kind") or "").lower()
        for item in allowed_actions
    )

    def _join(ids: Any) -> str:
        values = _extract_id_list(ids)
        return "、".join(values) if values else "无"

    lines = [
        "【预言家专用自检】",
        "每次决策前先核对 seer_context，再写最终行动。",
        f"当前账本：活狼={_join(seer_context.get('known_live_wolves'))}；死狼={_join(seer_context.get('known_dead_wolves'))}；活金水={_join(seer_context.get('known_live_villagers'))}；死金水={_join(seer_context.get('known_dead_villagers'))}；未验活人={_join(seer_context.get('uninspected_alive_ids'))}。",
        "发言和归票前必须确认：我说到的查验必须存在于 seer_context.inspections，且今天的主投目标必须是存活目标。",
        "不要把已死亡或已出局的查杀重复当作今天的归票对象；查杀出局后，下一天主线转为找其狼同伴、强保/强踩链和票型冲突位。",
        "若你已跳预言家、持警徽，或已成功推出查杀：白天应公开所有仍会影响站边的未公开查验，并给出下一验和警徽流。",
        "夜间优先查验存活且未查验的人；优先查验查杀链上的强保、强踩、带票、分票或持续站边冲突位，尽量避免重复验证已死对象或低信息量金水。",
        "警徽移交优先给存活金水，其次给站边清晰且未保狼的存活好人。",
        "不要长期隐藏金水，尤其当自己可能夜死且没有遗言时，查验遗产需要提前公开。",
    ]
    if speech_allowed:
        lines.extend(
            [
                "【发言前自检清单】",
                "1. 我声称的每个查验是否确实存在于 seer_context.inspections。",
                "2. 我要求今天放逐的人是否仍然存活。",
                "3. 我是否把查验事实和推测分开说。",
                "4. 如果我已跳预言家或持警徽，是否说明了警徽流和下一验目标。",
                "5. 不要说“昨晚查 p6”如果这其实是更早的查验，或 p6 已经死亡。",
            ]
        )
    priority = _join(seer_context.get("priority_inspect_ids"))
    if priority != "无":
        lines.append(f"代码按合法 target_ids 排出的下一验顺序：{priority}；只能从该列表对应的合法目标中选。")
    public_summary = seer_context.get("current_public_summary")
    if isinstance(public_summary, Mapping):
        claims = _join(public_summary.get("claim_players"))
        pressure = ",".join(
            f"{pid}:{'/'.join(tags)}" for pid, tags in list((public_summary.get("pressure") or {}).items())[:8]
        ) or "无"
        votes = _join(public_summary.get("vote_targets"))
        lines.append(f"本轮公开摘要：对跳/身份声明={claims}；冲突标签={pressure}；公开归票={votes}。")
    if speech_allowed:
        lines.extend([
            "【白天固定顺序】发言/警上必须按：查验事实；公开事实与对跳矛盾；唯一主压或主票；下一验；警徽流。事实与推测分开，不能提出互斥的多个主票。",
            "每个查验只能来自 seer_context.inspections；主压目标必须存活。若自己可能夜死，当前发言留下完整、可执行的查验遗产。",
        ])
    if vote_allowed:
        lines.append("投票执行已经公开的唯一主票；若没有主票，从当前高信息冲突目标中选一名合法存活目标，不要无理由 pass。已知活狼优先。")
    if transfer_allowed:
        lines.append("警徽移交优先落到存活金水，其次是站边清晰且未被查验为狼的存活好人。")
    if any("inspect" in str(item.get("kind") or "").lower() for item in allowed_actions):
        lines.append("夜间查验只在代码排序的合法目标中选择；排除自己、死亡、已验和已公开确认的普通金水。")
    return "\n".join(lines)


def _seer_action_group(kind: str) -> str | None:
    normalized = kind.strip().lower()
    if "inspect" in normalized:
        return "inspect"
    if normalized in {"day_vote", "vote"} or normalized.endswith("_vote"):
        return "vote"
    if normalized in {"sheriff_badge_transfer", "badge_transfer", "transfer_badge", "sheriff_transfer"}:
        return "badge_transfer"
    if "vote" in normalized and "speech" not in normalized:
        return "vote"
    if "badge" in normalized and "transfer" in normalized:
        return "badge_transfer"
    return None


def _apply_seer_safety_overrides(action: dict[str, Any], request: Mapping[str, Any], seer_context: Mapping[str, Any]) -> dict[str, Any]:
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
    if group == "inspect":
        preferred_targets = [str(pid) for pid in seer_context.get("priority_inspect_ids", []) if str(pid) in target_id_set]
        if not preferred_targets:
            return action
        chosen = str(action.get("target_id") or "")
        inspected = {str(item.get("target_id")) for item in seer_context.get("inspections", []) if item.get("target_id")}
        dead = {str(pid) for pid in seer_context.get("known_dead_wolves", [])} | {str(pid) for pid in seer_context.get("known_dead_villagers", [])}
        low_value = chosen not in target_id_set or chosen in inspected or chosen in dead or chosen in {str(pid) for pid in seer_context.get("known_live_villagers", [])}
        if low_value or chosen != preferred_targets[0]:
            overridden = dict(action)
            overridden["target_id"] = preferred_targets[0]
            return overridden
        return action
    if group == "vote":
        preferred_targets = [str(pid) for pid in seer_context.get("known_live_wolves", []) if str(pid) in target_id_set]
        pass_like = str(action.get("target_id") or "").lower() in {"", "pass", "abstain", "skip", "弃票", "过"}
        if not preferred_targets and pass_like:
            preferred_targets = [str(pid) for pid in seer_context.get("priority_inspect_ids", []) if str(pid) in target_id_set]
    else:
        preferred_targets = [str(pid) for pid in seer_context.get("known_live_villagers", []) if str(pid) in target_id_set]
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
        self._seer_memory = SeerDecisionMemory() if self.profile.role == "seer" else None
        self._model_token_usage: dict[str, int] = {
            "successful_response_count": 0,
            "api_attempt_count": 0,
            "reported_usage_response_count": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }

    async def observe(self, sync_packet: dict[str, Any]) -> None:
        """接收公开同步；预言家只用它检测新局，不保存完整历史。"""

        # 预言家只按 game_id 重置轻量公开摘要；不在 observe 中拼接完整历史。
        if self._seer_memory is not None and isinstance(sync_packet, Mapping):
            self._seer_memory.reset_if_needed(_game_id(sync_packet))
        del sync_packet

    async def decide(self, turn_packet: dict[str, Any]) -> dict[str, Any]:
        if turn_packet["self"]["player_id"] != self.player_id:
            raise ValueError("行动包被发送给了错误玩家")
        private = turn_packet["private_information"]
        if str(private.get("role")) != self.profile.role:
            raise ValueError("Task-Agent 的角色档案与行动包身份不一致")

        system = self._system_prompt(private)
        seer_context = _seer_context(turn_packet) if self.profile.role == "seer" else {}
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
        if self._seer_memory is not None:
            current_summary = self._seer_memory.update(turn_packet, current_dialogue)
            seer_context["current_public_summary"] = current_summary
            seer_context["recent_public_summaries"] = self._seer_memory.compact()
            seer_context["priority_inspect_ids"] = _seer_priority_order(
                turn_packet["request"], seer_context, self._seer_memory
            )

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
                action = _apply_seer_safety_overrides(action, turn_packet["request"], seer_context)
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

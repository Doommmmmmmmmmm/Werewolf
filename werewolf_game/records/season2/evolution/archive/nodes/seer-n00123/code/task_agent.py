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
        self._model_token_usage: dict[str, int] = {
            "successful_response_count": 0,
            "api_attempt_count": 0,
            "reported_usage_response_count": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }
        # 这不是完整历史，只是从公开同步和自己的已提交行动中提炼的事实锚点。
        self._seer_state: dict[str, list[Any]] = {
            "own_public_claims": [],
            "announced_inspections": [],
            "announced_vote_targets": [],
            "announced_badge_flows": [],
            # 只保存发言中明确给出的有限票口，不保存原始发言或完整历史。
            "published_vote_targets": [],
            "published_backup_targets": [],
            "last_day_vote_counts": [],
            "death_events": [],
            "sheriff_history": [],
        }

    async def observe(self, sync_packet: dict[str, Any]) -> None:
        """保守读取公开事件；未知同步格式一律忽略。"""

        if not isinstance(sync_packet, Mapping):
            return
        for event in self._public_events(sync_packet):
            self._remember_public_event(event)

    @staticmethod
    def _public_events(packet: Mapping[str, Any]) -> list[Mapping[str, Any]]:
        events: list[Mapping[str, Any]] = []
        seen: set[int] = set()
        event_keys = {"event", "event_type", "type", "kind", "name"}

        def walk(value: object) -> None:
            if isinstance(value, Mapping):
                marker = any(key in value for key in event_keys)
                if marker and id(value) not in seen:
                    seen.add(id(value))
                    events.append(value)
                for key, nested in value.items():
                    # 只沿着事件容器和同步包继续走，避免把 public_state 的普通字段
                    # 当成审计事件；字段缺失时静默兼容不同 runner。
                    if key in {"events", "event", "event_list", "event_log", "audit_events", "public_events", "sync", "data", "payload"}:
                        walk(nested)
            elif isinstance(value, list):
                for item in value:
                    walk(item)

        walk(packet)
        return events

    def _remember_public_event(self, event: Mapping[str, Any]) -> None:
        # 有些 runner 把事件名放在外层、事件字段放在 data/payload；合并仅用于
        # 读取公开字段，不把该结构或原始文本保存进状态。
        flattened: dict[str, Any] = dict(event)
        for container_key in ("data", "payload", "details"):
            nested = event.get(container_key)
            if isinstance(nested, Mapping):
                for key, value in nested.items():
                    flattened.setdefault(key, value)
        kind = str(flattened.get("event_type", flattened.get("event", flattened.get("type", flattened.get("kind", flattened.get("name", "")))))).upper()
        round_value = flattened.get("round", flattened.get("round_number"))
        stage_value = flattened.get("stage", flattened.get("phase"))
        actor = self._event_player_id(flattened)
        text = self._event_text(flattened)
        if kind in {"PLAYER_SPOKE", "PLAYER_LAST_WORDS"} and actor == self.player_id:
            if text:
                if re.search(r"预言家|先知|\bseer\b", text, re.I):
                    self._remember("own_public_claims", "公开声称身份：预言家")
                for target, team in self._spoken_inspections(text):
                    self._remember("announced_inspections", {"target_id": target, "team": team})
                main_targets, backup_targets = self._spoken_vote_commitments(text)
                for target in self._spoken_targets(text, r"(?:主票|归票|投票|票口|压|出)" ):
                    self._remember("announced_vote_targets", target)
                for target in main_targets:
                    self._remember("published_vote_targets", self._commitment_record(target, "main", round_value, stage_value))
                for target in backup_targets:
                    self._remember("published_backup_targets", self._commitment_record(target, "backup", round_value, stage_value))
                for target in self._spoken_badge_targets(text):
                    self._remember("announced_badge_flows", target)
        elif kind in {"VOTE_CAST", "DAY_VOTE_CAST"} and actor == self.player_id:
            target = self._event_target(flattened)
            if target:
                self._remember("announced_vote_targets", target)
        elif kind in {"DAY_VOTE_RESOLVED", "VOTE_RESOLVED"}:
            counts = flattened.get("vote_counts", flattened.get("counts", flattened.get("results")))
            if isinstance(counts, Mapping):
                self._remember("last_day_vote_counts", {str(k): str(v) for k, v in counts.items()})
        elif kind in {"PLAYER_ELIMINATED", "PLAYER_DIED", "PLAYER_KILLED"}:
            target = self._event_target(flattened) or actor
            reason = flattened.get("death_reason", flattened.get("reason", flattened.get("cause", "未知")))
            if target:
                self._remember("death_events", {"player_id": str(target), "reason": str(reason)[:24]})
        elif kind in {"SHERIFF_ELECTED", "SHERIFF_BADGE_TRANSFERRED", "SHERIFF_BADGE_TRANSFER"}:
            target = self._event_target(flattened)
            if target:
                self._remember("sheriff_history", {"event": kind, "player_id": target})

    @staticmethod
    def _event_player_id(event: Mapping[str, Any]) -> str:
        for key in ("player_id", "actor_id", "speaker_id", "voter_id", "speaker", "player", "actor"):
            value = event.get(key)
            if isinstance(value, Mapping):
                value = value.get("player_id", value.get("id"))
            if value is not None:
                return str(value)
        return ""

    @staticmethod
    def _event_target(event: Mapping[str, Any]) -> str:
        for key in ("target_id", "eliminated_player_id", "recipient_id", "to_player_id", "elected_player_id", "target", "to", "player_id"):
            value = event.get(key)
            if isinstance(value, Mapping):
                value = value.get("player_id", value.get("id"))
            if value is not None:
                return str(value)
        return ""

    @staticmethod
    def _event_text(event: Mapping[str, Any]) -> str:
        for key in ("text", "speech", "content", "message", "last_words"):
            if isinstance(event.get(key), str):
                return event[key]
        return ""

    def _remember(self, category: str, value: Any, limit: int = 12) -> None:
        values = self._seer_state[category]
        if value in values:
            return
        values.append(value)
        del values[:-limit]

    @staticmethod
    def _commitment_record(
        target: str, priority: str, round_value: object = None, stage_value: object = None
    ) -> dict[str, str]:
        record = {"target_id": str(target), "priority": priority}
        if round_value is not None:
            record["round"] = str(round_value)
        if stage_value is not None:
            record["stage"] = str(stage_value)
        return record

    @staticmethod
    def _spoken_inspections(text: str) -> list[tuple[str, str]]:
        found: list[tuple[str, str]] = []
        target_re = r"(?:p\d+|\d+号)"
        for match in re.finditer(target_re + r"[^。；\n]{0,12}(查杀|金水|狼人|狼|好人|村民|wolf|village)", text, re.I):
            target_match = re.search(target_re, match.group(0), re.I)
            if target_match is None:
                continue
            target = TaskAgent._normalize_public_target(target_match.group(0))
            result = match.group(1).lower()
            team = "wolf" if result in {"查杀", "狼人", "狼", "wolf"} else "village"
            found.append((target, team))
        return found

    @staticmethod
    def _spoken_targets(text: str, prefix: str) -> list[str]:
        return [
            TaskAgent._normalize_public_target(m.group(1))
            for m in re.finditer(prefix + r"[^。；\n]{0,10}?((?:p\d+|\d+号))", text, re.I)
        ]

    @staticmethod
    def _spoken_badge_targets(text: str) -> list[str]:
        match = re.search(r"警徽流\s*[：:]?\s*([^。；\n]+)", text)
        if not match:
            return []
        flow_text = re.split(r"[，,]\s*(?:主票|归票|当日)", match.group(1), maxsplit=1)[0]
        return [
            TaskAgent._normalize_public_target(x)
            for x in re.findall(r"(?:p\d+|\d+号)", flow_text, re.I)
        ]

    @staticmethod
    def _normalize_public_target(text: str) -> str:
        text = str(text).strip()
        if text.endswith("号") and text[:-1].isdigit():
            return "p" + text[:-1]
        return text

    @staticmethod
    def _spoken_vote_commitments(text: str) -> tuple[list[str], list[str]]:
        """提取短而明确的主票/备选；不把普通怀疑当成承诺。"""
        target_re = r"((?:p\d+|\d+号))"
        main = [
            TaskAgent._normalize_public_target(match.group(1))
            for match in re.finditer(r"(?:主票|归票|唯一票口|第一票)\s*[：:]?[^。；\n]{0,12}?" + target_re, text, re.I)
        ]
        backup = [
            TaskAgent._normalize_public_target(match.group(1))
            for match in re.finditer(r"(?:备选|副票|次票|第二票|转票)\s*[：:]?[^。；\n]{0,12}?" + target_re, text, re.I)
        ]
        return TaskAgent._unique_targets(main), TaskAgent._unique_targets(backup)

    @staticmethod
    def _unique_targets(values: object) -> list[str]:
        if not isinstance(values, list):
            return []
        result: list[str] = []
        seen: set[str] = set()
        for value in values:
            target = str(value).strip()
            if target and target not in seen:
                seen.add(target)
                result.append(target)
        return result

    @staticmethod
    def _allowed_target_ids(request: Mapping[str, Any], kind: str) -> list[str]:
        if not isinstance(request, Mapping):
            return []
        raw_actions = request.get("allowed_actions")
        if not isinstance(raw_actions, list):
            return []
        for item in raw_actions:
            if not isinstance(item, Mapping) or str(item.get("kind") or "") != kind:
                continue
            targets = item.get("target_ids")
            if not isinstance(targets, list):
                return []
            return TaskAgent._unique_targets(targets)
        return []

    @staticmethod
    def _seer_target_ids(request: Mapping[str, Any], kind: str = "seer_inspect") -> list[str]:
        """返回行动包中指定 kind 的候选；不推断或扩展引擎候选集合。"""
        return TaskAgent._allowed_target_ids(request, kind)

    @staticmethod
    def _allowed_kinds(request: Mapping[str, Any]) -> set[str]:
        if not isinstance(request, Mapping):
            return set()
        raw_actions = request.get("allowed_actions")
        if not isinstance(raw_actions, list):
            return set()
        return {
            str(item.get("kind") or "")
            for item in raw_actions
            if isinstance(item, Mapping) and item.get("kind")
        }

    @staticmethod
    def _known_target_set(private: Mapping[str, Any]) -> set[str]:
        if not isinstance(private, Mapping):
            return set()
        return {
            str(item.get("target_id"))
            for item in TaskAgent._extract_seer_known_results(private)
            if isinstance(item, Mapping) and item.get("target_id") is not None
        }

    @staticmethod
    def _dead_target_set(public_state: object, seer_state: Mapping[str, list[Any]] | None = None) -> set[str]:
        """仅从角色可见 public_state/已记录公开死亡事件提取死亡目标。"""
        dead: set[str] = set()
        id_keys = {
            "player_id", "target_id", "eliminated_player_id", "dead_player_id",
            "killed_player_id", "victim_id", "id",
        }
        dead_keys = {
            "dead_players", "dead_player_ids", "deceased_players", "deceased_player_ids",
            "eliminated_players", "eliminated_player_ids", "killed_players", "killed_player_ids",
            "deaths", "death_events",
        }

        def add_ids(value: object) -> None:
            if isinstance(value, str) or isinstance(value, (int, float)):
                dead.add(str(value))
            elif isinstance(value, Mapping):
                found_id = False
                for key in id_keys:
                    if key in value and value.get(key) is not None:
                        found_id = True
                        item = value.get(key)
                        if isinstance(item, Mapping):
                            item = item.get("player_id", item.get("id"))
                        if item is not None:
                            dead.add(str(item))
                # 兼容 {"p7": "dead"} 这类公开状态映射，但不把普通字段当玩家。
                if not found_id:
                    for key, item in value.items():
                        if isinstance(key, (str, int)) and isinstance(item, str):
                            if item.lower() in {"dead", "dead_player", "eliminated", "deceased", "死亡", "出局"}:
                                dead.add(str(key))
            elif isinstance(value, list):
                for item in value:
                    add_ids(item)

        def walk(value: object) -> None:
            if isinstance(value, Mapping):
                for key, nested in value.items():
                    key_text = str(key).lower()
                    if key_text in dead_keys or "dead" in key_text or "eliminat" in key_text or "deceas" in key_text:
                        add_ids(nested)
                    elif isinstance(nested, (Mapping, list)):
                        walk(nested)
                status = str(value.get("status", value.get("state", ""))).lower()
                if value.get("alive") is False or value.get("is_alive") is False or status in {"dead", "dead_player", "eliminated", "deceased", "死亡", "出局"}:
                    add_ids(value)
            elif isinstance(value, list):
                for item in value:
                    walk(item)

        walk(public_state)
        if isinstance(seer_state, Mapping):
            add_ids(seer_state.get("death_events", []))
        return dead

    @staticmethod
    def _state_targets(seer_state: Mapping[str, list[Any]] | None, key: str) -> list[str]:
        if not isinstance(seer_state, Mapping):
            return []
        values = seer_state.get(key, [])
        if not isinstance(values, list):
            return []
        result: list[str] = []
        for value in values:
            if isinstance(value, Mapping):
                value = value.get("target_id", value.get("player_id"))
            if value is not None:
                result.append(str(value))
        return TaskAgent._unique_targets(result)

    @classmethod
    def _published_targets(cls, seer_state: Mapping[str, list[Any]] | None, backup: bool = False) -> list[str]:
        key = "published_backup_targets" if backup else "published_vote_targets"
        targets = cls._state_targets(seer_state, key)
        if targets:
            return targets
        # 兼容旧状态：旧字段没有主/备区分，只能作为主票候选。
        return cls._state_targets(seer_state, "announced_vote_targets") if not backup else []

    def _repair_seer_action(
        self, action: Mapping[str, Any], request: Mapping[str, Any],
        private: Mapping[str, Any], public_state: object = None,
    ) -> dict[str, Any]:
        """只为 seer 的硬事实/合法性冲突做保守兜底。"""
        repaired = dict(action) if isinstance(action, Mapping) else {}
        if not isinstance(private, Mapping) or not isinstance(request, Mapping) or str(private.get("role")) != "seer":
            return repaired
        kind = str(repaired.get("kind") or "")
        known = self._known_target_set(private)
        if kind == "seer_inspect":
            legal = self._allowed_target_ids(request, kind)
            if repaired.get("target_id") in legal and repaired.get("target_id") not in known:
                return repaired
            candidates = [target for target in legal if target not in known]
            if candidates:
                preferred = (
                    self._published_targets(self._seer_state)
                    + self._published_targets(self._seer_state, backup=True)
                    + self._state_targets(self._seer_state, "announced_vote_targets")
                    + self._state_targets(self._seer_state, "announced_badge_flows")
                )
                ordered = [target for target in preferred if target in candidates]
                ordered.extend(target for target in candidates if target not in ordered)
                repaired["target_id"] = ordered[0]
                return repaired
            kinds = self._allowed_kinds(request)
            pass_kind = next((item for item in ("pass", "seer_pass") if item in kinds), None)
            if pass_kind is not None:
                repaired.pop("target_id", None)
                repaired["kind"] = pass_kind
        elif kind in {"day_vote"}:
            legal = self._allowed_target_ids(request, kind)
            live_wolves = [
                item["target_id"] for item in self._extract_seer_known_results(private)
                if item.get("team") == "wolf" and item.get("target_id") in legal
                and item.get("target_id") not in self._dead_target_set(public_state, self._seer_state)
            ]
            if live_wolves:
                repaired["target_id"] = live_wolves[0]
            elif repaired.get("target_id") in {
                item["target_id"] for item in self._extract_seer_known_results(private)
                if item.get("team") == "village"
            }:
                main = next((target for target in self._published_targets(self._seer_state) if target in legal), None)
                backup = next((target for target in self._published_targets(self._seer_state, backup=True) if target in legal), None)
                if main is not None:
                    repaired["target_id"] = main
                elif backup is not None:
                    repaired["target_id"] = backup
        return repaired

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
        seer_known_results = self._extract_seer_known_results(private)
        if seer_known_results:
            prompt["seer_known_results"] = seer_known_results
        if str(private.get("role")) == "seer":
            prompt["seer_state_brief"] = self._seer_state_brief(
                private, turn_packet["public_state"], turn_packet["request"]
            )

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
                    turn_packet["request"], private, feedback, self._seer_state, self.player_id
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
            # 只在 seer 自己的硬事实冲突处兜底；其他角色和其他行动保持模型原样。
            action = self._repair_seer_action(
                action,
                turn_packet["request"],
                private,
                turn_packet.get("public_state"),
            )
            error = decision_error(action, turn_packet["request"])
            if error is None:
                self._remember_own_action(
                    action,
                    turn_packet.get("game"),
                    turn_packet.get("request"),
                )
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

    def _remember_own_action(
        self, action: Mapping[str, Any], game: Mapping[str, Any] | None = None,
        request: Mapping[str, Any] | None = None,
    ) -> None:
        # game/request 仅提供本次公开发言的短元数据，不保存原始文本。
        round_value = game.get("round", game.get("round_number")) if isinstance(game, Mapping) else None
        stage_value = request.get("phase", request.get("stage")) if isinstance(request, Mapping) else None
        if action.get("kind") in _SPEECH_ACTION_KINDS:
            text = action.get("text")
            if isinstance(text, str):
                if re.search(r"预言家|先知|\bseer\b", text, re.I):
                    self._remember("own_public_claims", "公开声称身份：预言家")
                for target, team in self._spoken_inspections(text):
                    self._remember("announced_inspections", {"target_id": target, "team": team})
                main_targets, backup_targets = self._spoken_vote_commitments(text)
                for target in self._spoken_targets(text, r"(?:主票|归票|投票|票口|压|出)"):
                    self._remember("announced_vote_targets", target)
                for target in main_targets:
                    self._remember("published_vote_targets", self._commitment_record(target, "main", round_value, stage_value))
                for target in backup_targets:
                    self._remember("published_backup_targets", self._commitment_record(target, "backup", round_value, stage_value))
                for target in self._spoken_badge_targets(text):
                    self._remember("announced_badge_flows", target)
        elif action.get("kind") == "day_vote" and action.get("target_id"):
            self._remember("announced_vote_targets", str(action["target_id"]))

    def _seer_decision_summary(
        self, private: Mapping[str, Any], public_state: object, request: Mapping[str, Any]
    ) -> dict[str, Any]:
        known = self._extract_seer_known_results(private) if isinstance(private, Mapping) else []
        known_map = {item["target_id"]: item["team"] for item in known}
        dead = self._dead_target_set(public_state, self._seer_state)
        legal_inspections = self._allowed_target_ids(request, "seer_inspect")
        legal_votes = self._allowed_target_ids(request, "day_vote")
        live_known_wolves = [
            item["target_id"] for item in known
            if item.get("team") == "wolf"
            and item.get("target_id") in legal_votes
            and item.get("target_id") not in dead
        ]
        return {
            "known_inspection_map": known_map,
            "known_dead_targets": sorted(dead),
            "legal_inspection_targets": legal_inspections,
            "live_known_wolves": live_known_wolves,
            "published_vote_target": next(
                (target for target in self._published_targets(self._seer_state) if target in legal_votes), None
            ),
            "published_backup_target": next(
                (target for target in self._published_targets(self._seer_state, backup=True) if target in legal_votes), None
            ),
        }

    def _seer_state_brief(
        self, private: Mapping[str, Any], public_state: object, request: Mapping[str, Any]
    ) -> str:
        summary = self._seer_decision_summary(private, public_state, request)
        brief = {
            **summary,
            "known_inspections_in_order": self._extract_seer_known_results(private)[:6],
            "published_inspections_in_order": self._seer_state.get("announced_inspections", [])[-8:],
            "published_badge_flow": self._state_targets(self._seer_state, "announced_badge_flows")[-6:],
            "recent_vote_counts": self._seer_state.get("last_day_vote_counts", [])[-3:],
            "death_events": self._seer_state.get("death_events", [])[-6:],
            "sheriff_history": self._seer_state.get("sheriff_history", [])[-6:],
        }
        # 结构化摘要而非原话；截断是最后一道预算保护，避免异常 ID/状态撑大 prompt。
        return json.dumps(brief, ensure_ascii=False, separators=(",", ":"))[:2200]

    @staticmethod
    def _turn_instruction(request: Mapping[str, Any], private: Mapping[str, Any], feedback: str,
                          seer_state: Mapping[str, list[Any]] | None = None,
                          player_id: str | None = None) -> str:
        instruction = render_prompt(
            "player_turn_instruction.txt",
            action_contract=render_action_contract(request),
            validation_feedback=(f"上一次输出未通过校验：{feedback}" if feedback else ""),
        )
        seer_hint = TaskAgent._seer_phase_hint(request, private, seer_state, player_id)
        if seer_hint:
            instruction = instruction + "\n\n" + seer_hint
        return instruction

    @staticmethod
    def _seer_phase_hint(
        request: Mapping[str, Any], private: Mapping[str, Any],
        seer_state: Mapping[str, list[Any]] | None = None,
        player_id: str | None = None,
    ) -> str:
        if not isinstance(request, Mapping) or not isinstance(private, Mapping) or str(private.get("role")) != "seer":
            return ""
        kinds = TaskAgent._allowed_kinds(request)
        known = TaskAgent._extract_seer_known_results(private)
        known_map = {item["target_id"]: item["team"] for item in known}
        legal_inspections = TaskAgent._allowed_target_ids(request, "seer_inspect")
        legal_votes = TaskAgent._allowed_target_ids(request, "day_vote")
        dead = TaskAgent._dead_target_set(None, seer_state)
        live_wolves = [
            item["target_id"] for item in known
            if item.get("team") == "wolf" and item.get("target_id") in legal_votes
            and item.get("target_id") not in dead
        ]
        live_villages = [
            item["target_id"] for item in known
            if item.get("team") == "village" and item.get("target_id") in legal_votes
            and item.get("target_id") not in dead
        ]
        published = TaskAgent._published_targets(seer_state)
        backup = TaskAgent._published_targets(seer_state, backup=True)
        hints: list[str] = [
            "预言家结构化事实：known_inspection_map=" + json.dumps(known_map, ensure_ascii=False, separators=(",", ":"))
            + "；known_dead_targets=" + json.dumps(sorted(dead), ensure_ascii=False)
            + "；legal_inspection_targets=" + json.dumps(legal_inspections, ensure_ascii=False)
            + "；live_known_wolves=" + json.dumps(live_wolves, ensure_ascii=False)
            + "；published_vote_target=" + json.dumps(next((x for x in published if x in legal_votes), None), ensure_ascii=False)
            + "；published_backup_target=" + json.dumps(next((x for x in backup if x in legal_votes), None), ensure_ascii=False),
        ]
        # 固定顺序：硬事实 -> 唯一主行动 -> 备选 -> 普通推理。
        if "seer_inspect" in kinds:
            uninspected = [target for target in legal_inspections if target not in known_map]
            hints.append("唯一夜验优先级：提交必须从 legal_inspection_targets 选，排除 known_inspection_map 和 known_dead_targets；模型选重复/缺失时只会从合法未验候选稳定兜底。可选未验=" + json.dumps(uninspected, ensure_ascii=False))
        if "day_vote" in kinds:
            if live_wolves:
                hints.append("唯一白天主行动：存活已验狼=" + "、".join(live_wolves) + "，day_vote 先投其中排序第一者。")
            else:
                main = next((x for x in published if x in legal_votes), None)
                alt = next((x for x in backup if x in legal_votes), None)
                if main:
                    hints.append(f"唯一白天主行动：公开主票 {main} 仍合法，优先执行；主票失效才用公开备选 {alt or '无'}，之后才普通推理。")
                elif alt:
                    hints.append(f"唯一白天主行动：公开主票已失效，执行公开备选 {alt}；之后才普通推理。")
                else:
                    hints.append("唯一白天主行动：没有有效公开票口；保护存活已验村民，不投 known_inspection_map 中的 village，才使用普通推理。")
        sheriff_kinds = {"sheriff_vote", "sheriff_election_vote"}
        if kinds & sheriff_kinds:
            sheriff_targets = []
            for sheriff_kind in sheriff_kinds:
                sheriff_targets.extend(TaskAgent._allowed_target_ids(request, sheriff_kind))
            sheriff_targets = TaskAgent._unique_targets(sheriff_targets)
            if live_wolves and any(target in sheriff_targets for target in live_wolves):
                hints.append("警长竞选唯一优先级：候选中有已验狼时不得无理由 pass，优先阻止该已验狼。")
            elif (player_id or str(private.get("player_id", ""))) in sheriff_targets:
                hints.append("警长竞选唯一优先级：若自己是合法候选且未验狼，优先自投；不要无理由 pass。")
            else:
                hints.append("警长竞选唯一优先级：自己不在候选时，投公开可信的预言家或已公开金水候选；不要无理由 pass。")
        if kinds & {"sheriff_candidate", "sheriff_election_speech", "speak", "last_words"}:
            hints.append("公开发言骨架：按【查验事实及顺序】【对跳/票型矛盾】【唯一主票与备选】【下一验和金水保护】输出；调整承诺先说明新的硬事实。")
        if seer_state and seer_state.get("own_public_claims"):
            hints.append("公开一致性：已跳预言家不得否认；查验复述严格按收到顺序，不颠倒时间线。")
        if "last_words" in kinds:
            death_events = (seer_state or {}).get("death_events", [])
            own_death = [x for x in death_events if isinstance(x, Mapping)]
            if own_death:
                reason = own_death[-1].get("reason", "未知")
                hints.append(f"遗言事实约束：公开记录显示你的死亡原因是“{reason}”；不要说成其他方式。")
            else:
                hints.append("遗言事实约束：死亡方式无法确认时使用‘我死亡后’等中性说法。")
        if kinds & {"sheriff_badge_transfer", "destroy"}:
            hints.append("警徽处置：优先传给已公开金水或最可信带队好人；没有可信接徽者再撕徽。")
        return " ".join(hints)

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
    def _extract_seer_known_results(private: Mapping[str, Any]) -> list[dict[str, str]]:
        if str(private.get("role")) != "seer":
            return []

        results: list[dict[str, str]] = []
        seen_targets: set[str] = set()

        def normalize_team(value: object) -> str | None:
            if value is None:
                return None
            text = str(value).strip()
            if not text:
                return None
            lowered = text.lower()
            aliases = {
                "village": "village",
                "villager": "village",
                "good": "village",
                "goodish": "village",
                "town": "village",
                "wolf": "wolf",
                "werewolf": "wolf",
                "evil": "wolf",
                "mafia": "wolf",
            }
            return aliases.get(lowered, text)

        def add_entry(target: object, team: object) -> None:
            if target is None:
                return
            target_id = str(target).strip()
            team_name = normalize_team(team)
            if not target_id or not team_name or target_id in seen_targets:
                return
            seen_targets.add(target_id)
            results.append({"target_id": target_id, "team": team_name})

        def walk(value: object) -> None:
            if isinstance(value, Mapping):
                target = None
                for key in ("target_id", "target", "player_id", "player", "subject", "checked_player"):
                    if key in value:
                        target = value.get(key)
                        break
                team = None
                for key in ("team", "result", "alignment", "side", "check_result", "reveal"):
                    if key in value:
                        team = value.get(key)
                        break
                if target is not None and team is not None:
                    add_entry(target, team)
                elif target is None and team is None and value and all(
                    isinstance(key, (str, int)) and isinstance(item, str)
                    for key, item in value.items()
                ):
                    # 常见内核格式：inspections={target_id: team}。Mapping 保持
                    # 插入顺序，add_entry 负责稳定去重。
                    for target_id, team_name in value.items():
                        add_entry(target_id, team_name)
                for nested_key in (
                    "inspection_results",
                    "inspections",
                    "checks",
                    "check_results",
                    "results",
                    "history",
                    "data",
                ):
                    nested = value.get(nested_key)
                    if nested is not None:
                        walk(nested)
            elif isinstance(value, list):
                for item in value:
                    walk(item)

        for key in (
            "inspection_results",
            "inspections",
            "checks",
            "check_results",
            "seer_inspections",
            "seer_checks",
            "known_results",
            "memory",
            "role_state",
            "seer_state",
        ):
            value = private.get(key)
            if value is not None:
                walk(value)

        return results[:6]

    @staticmethod
    def _nonnegative_int(value: object, *, fallback: int | None) -> int | None:
        if isinstance(value, bool):
            return fallback
        try:
            normalized = int(value)
        except (TypeError, ValueError):
            return fallback
        return normalized if normalized >= 0 else fallback

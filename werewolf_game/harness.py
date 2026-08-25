"""Task-Agent Harness 的受限描述、运行时和基础档案。

Season 2 不允许模型生成可执行代码。模型只能生成这里定义的 JSON 数据：
上下文筛选策略、信念板字段、战术卡和记忆上限。运行时由 Python 解释这些数据，
因此候选 Harness 即使包含恶意文本，也不能获得文件、网络、shell 或引擎权限。
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import math
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping

from .constants import ALL_ROLES, public_phase_for
from .prompts import RoleProfile


HARNESS_SCHEMA_VERSION = 1
HARNESS_MODULES = (
    "context_policy",
    "belief_board",
    "tactic_router",
    "planning_policy",
    "memory_policy",
    "coordination_policy",
    "output_policy",
)
HARNESS_SOURCE_TYPES = frozenset(
    {
        "incumbent",
        "replay_mutation",
        "counterfactual",
        "research_prior",
        "red_team",
        "recombined",
        "baseline",
    }
)
_SAFE_ID = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.:-]{0,127}$")
_EXECUTABLE_KEY = re.compile(
    r"(?:^|_)(?:code|exec|execute|script|shell|command|python|javascript|tool_call)(?:_|$)",
    re.IGNORECASE,
)


def _text(value: object, *, limit: int = 1000) -> str:
    normalized = " ".join(str(value or "").strip().split())
    return normalized[:limit]


def _text_list(value: object, *, limit: int = 12, item_limit: int = 1000) -> tuple[str, ...]:
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, (list, tuple)):
        values = list(value)
    else:
        values = []
    result: list[str] = []
    for item in values:
        normalized = _text(item, limit=item_limit)
        if normalized and normalized not in result:
            result.append(normalized)
        if len(result) >= limit:
            break
    return tuple(result)


def _json_safe(value: object, *, depth: int = 0) -> Any:
    """复制候选配置，同时拒绝不可审计的可执行结构。"""

    if depth > 6:
        raise ValueError("Harness 配置嵌套层级不能超过 6 层")
    if value is None or isinstance(value, (int, bool)):
        return value
    if isinstance(value, str):
        return value[:4000]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Harness 配置不能包含 NaN 或无穷浮点数")
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for raw_key, raw_value in value.items():
            key = _text(raw_key, limit=100)
            if not key:
                continue
            if _EXECUTABLE_KEY.search(key):
                raise ValueError(f"Harness 配置包含禁止字段：{key}")
            result[key] = _json_safe(raw_value, depth=depth + 1)
        return result
    if isinstance(value, (list, tuple)):
        return [_json_safe(item, depth=depth + 1) for item in value[:64]]
    raise ValueError(f"Harness 配置包含不可序列化值：{type(value).__name__}")


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class TacticalCard:
    """一张可被运行时选择的战术卡。

    卡片只描述适用条件和可观察信号，不包含 Python 表达式或动作执行代码。
    """

    card_id: str
    title: str
    trigger: str
    action_tendency: str
    counterexamples: tuple[str, ...]
    exit_conditions: tuple[str, ...]
    observable_signals: tuple[str, ...]
    confidence: float = 0.5
    tags: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], *, index: int = 0) -> "TacticalCard":
        card_id = _text(value.get("card_id", value.get("id", f"card-{index}")), limit=128)
        if not _SAFE_ID.fullmatch(card_id):
            card_id = f"card-{index}"
        try:
            confidence = float(value.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5
        confidence = max(0.0, min(1.0, confidence))
        card = cls(
            card_id=card_id,
            title=_text(value.get("title", "未命名战术卡"), limit=200),
            trigger=_text(value.get("trigger", ""), limit=800),
            action_tendency=_text(
                value.get("action_tendency", value.get("action", "")), limit=1200
            ),
            counterexamples=_text_list(value.get("counterexamples"), item_limit=600),
            exit_conditions=_text_list(
                value.get("exit_conditions", value.get("exit")), item_limit=600
            ),
            observable_signals=_text_list(
                value.get("observable_signals", value.get("signals")), item_limit=600
            ),
            confidence=confidence,
            tags=_text_list(value.get("tags"), limit=10, item_limit=80),
        )
        card.validate()
        return card

    def validate(self) -> None:
        if not _SAFE_ID.fullmatch(self.card_id):
            raise ValueError(f"非法战术卡 ID：{self.card_id}")
        if len(self.title) < 2:
            raise ValueError(f"战术卡 {self.card_id} 缺少标题")
        if not self.action_tendency:
            raise ValueError(f"战术卡 {self.card_id} 缺少行动倾向")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"战术卡 {self.card_id} 的 confidence 必须在 0–1 之间")
        if len(self.counterexamples) > 12 or len(self.exit_conditions) > 12:
            raise ValueError(f"战术卡 {self.card_id} 的反例或退出条件过多")

    def as_dict(self) -> dict[str, Any]:
        return {
            "card_id": self.card_id,
            "title": self.title,
            "trigger": self.trigger,
            "action_tendency": self.action_tendency,
            "counterexamples": list(self.counterexamples),
            "exit_conditions": list(self.exit_conditions),
            "observable_signals": list(self.observable_signals),
            "confidence": self.confidence,
            "tags": list(self.tags),
        }

    def markdown(self) -> str:
        lines = [
            f"# {self.card_id} · {self.title}",
            "",
            f"- 适用条件：{self.trigger or '由当前阶段和公开信息共同判断。'}",
            f"- 行动倾向：{self.action_tendency}",
            f"- 置信度：{self.confidence:.2f}",
            "",
            "## 可观察信号",
            "",
            *(f"- {item}" for item in (self.observable_signals or ("暂无明确先验。",))),
            "",
            "## 反例",
            "",
            *(f"- {item}" for item in (self.counterexamples or ("暂无反例；下一次复盘必须验证。",))),
            "",
            "## 退出条件",
            "",
            *(f"- {item}" for item in (self.exit_conditions or ("出现与假设冲突的硬信息时退出。",))),
            "",
        ]
        return "\n".join(lines)


def _default_modules(role: str) -> dict[str, dict[str, Any]]:
    return {
        "context_policy": {
            "max_visible_events": 28,
            "max_public_events": 36,
            "max_event_text_chars": 260,
            "max_private_notes": 10,
            "strategy_mode": "bounded",
            "strategy_char_limit": 2600,
            "include": ["game", "public_rules", "self", "private_information", "public_state", "visible_events", "public_events", "request"],
        },
        "belief_board": {
            "columns": ["confirmed_facts", "player_claims", "hypotheses", "unresolved_questions"],
            "separate_claims_from_facts": True,
            "require_confidence": True,
            "max_items": 12,
        },
        "tactic_router": {
            "max_cards_per_turn": 3,
            "reselect_on_phase_change": True,
            "reselect_on_new_public_event": True,
            "default_posture": "谨慎收集证据后再行动",
            "role": role,
        },
        "planning_policy": {
            "critical_phases": ["day_discussion", "day_vote", "night", "sheriff_election", "reaction"],
            "compare_alternatives": True,
            "counterfactual_question": "如果当前目标是好人，谁会从这个行动中获益？",
            "max_plan_steps": 4,
        },
        "memory_policy": {
            "max_notes": 12,
            "max_note_chars": 240,
            "retain": ["可验证承诺", "票型转折", "未解决矛盾", "队友约定"],
            "discard": ["完整推理过程", "重复的情绪描述", "未经证实的绝对结论"],
        },
        "coordination_policy": {
            "agenda": ["当前硬信息", "分工", "主方案", "备选方案", "撤退条件"],
            "confirm_before_action": True,
            "share_only_with_allowed_channel": True,
        },
        "output_policy": {
            "validate_facts_before_speaking": True,
            "recheck_allowed_actions": True,
            "language": "中文",
            "never_emit_hidden_reasoning": True,
        },
    }


@dataclass(frozen=True)
class HarnessSpec:
    """可版本化的 Task-Agent Harness 描述。"""

    harness_id: str
    role: str
    source_type: str
    version: int
    parent_id: str | None
    context_policy: dict[str, Any]
    belief_board: dict[str, Any]
    tactic_router: dict[str, Any]
    planning_policy: dict[str, Any]
    memory_policy: dict[str, Any]
    coordination_policy: dict[str, Any]
    output_policy: dict[str, Any]
    cards: tuple[TacticalCard, ...] = ()
    rationale: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    schema_version: int = HARNESS_SCHEMA_VERSION

    @classmethod
    def baseline(cls, role: str, *, strategy: str = "") -> "HarnessSpec":
        if role not in ALL_ROLES:
            raise ValueError(f"不支持的角色：{role}")
        modules = _default_modules(role)
        card = TacticalCard(
            card_id="baseline-evidence-first",
            title="证据优先与可撤退",
            trigger="所有阶段均可使用；出现硬信息或新公开事件时重新判断。",
            action_tendency="先区分系统事实、玩家声明和自己的假设，再选择合法行动；保留一个可撤退的备选方案。",
            counterexamples=("单局异常不等于稳定规律。", "没有足够信息时不要伪装成已确认事实。"),
            exit_conditions=("出现与当前假设冲突的系统事实。", "新的票型或技能结果改变收益比较。"),
            observable_signals=("新死讯", "新发言", "投票结果", "警徽或遗言事件", "技能查验结果"),
            confidence=0.5,
            tags=("baseline", "replan"),
        )
        rationale = ("这是没有历史证据时的保守基线；后续只能由可审计回放验证和修改。",)
        if strategy:
            rationale = rationale + ("角色 strategy.md 仅作为历史背景，不自动转化为硬规则。",)
        return cls(
            harness_id=f"baseline-{role}-v0",
            role=role,
            source_type="baseline",
            version=0,
            parent_id=None,
            **modules,
            cards=(card,),
            rationale=rationale,
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], *, role: str | None = None) -> "HarnessSpec":
        if isinstance(value, HarnessSpec):
            if role is not None and value.role != role:
                raise ValueError(f"Harness 角色 {value.role} 与请求角色 {role} 不一致")
            value.validate()
            return value
        if not isinstance(value, Mapping):
            raise ValueError("HarnessSpec 必须是 JSON 对象")
        resolved_role = str(value.get("role") or role or "").strip()
        if resolved_role not in ALL_ROLES:
            raise ValueError(f"HarnessSpec 的 role 不受支持：{resolved_role}")
        defaults = _default_modules(resolved_role)
        modules: dict[str, dict[str, Any]] = {}
        for module in HARNESS_MODULES:
            raw = value.get(module, defaults[module])
            if not isinstance(raw, Mapping):
                raise ValueError(f"HarnessSpec.{module} 必须是对象")
            modules[module] = _json_safe(raw)
        cards_raw = value.get("cards", [])
        if not isinstance(cards_raw, (list, tuple)):
            raise ValueError("HarnessSpec.cards 必须是数组")
        cards = tuple(
            TacticalCard.from_mapping(item, index=index)
            for index, item in enumerate(cards_raw[:12])
            if isinstance(item, Mapping)
        )
        if not cards:
            cards = HarnessSpec.baseline(resolved_role).cards
        try:
            version = int(value.get("version", 0))
        except (TypeError, ValueError) as error:
            raise ValueError("HarnessSpec.version 必须是整数") from error
        if version < 0:
            raise ValueError("HarnessSpec.version 不能为负数")
        source_type = _text(value.get("source_type", "replay_mutation"), limit=40)
        if source_type not in HARNESS_SOURCE_TYPES:
            raise ValueError(f"不支持的 Harness source_type：{source_type}")
        raw_id = _text(value.get("harness_id", value.get("id", "")), limit=128)
        if not raw_id or not _SAFE_ID.fullmatch(raw_id):
            raw_id = f"candidate-{resolved_role}-v{version}"
        parent_id = _text(value.get("parent_id"), limit=128) or None
        spec = cls(
            harness_id=raw_id,
            role=resolved_role,
            source_type=source_type,
            version=version,
            parent_id=parent_id,
            **modules,
            cards=cards,
            rationale=_text_list(value.get("rationale"), limit=20, item_limit=1000),
            evidence_refs=_text_list(value.get("evidence_refs"), limit=30, item_limit=300),
            schema_version=int(value.get("schema_version", HARNESS_SCHEMA_VERSION)),
        )
        spec.validate()
        return spec

    def validate(self) -> None:
        if self.schema_version != HARNESS_SCHEMA_VERSION:
            raise ValueError(f"不支持的 Harness schema_version：{self.schema_version}")
        if self.role not in ALL_ROLES:
            raise ValueError(f"不支持的 Harness 角色：{self.role}")
        if not _SAFE_ID.fullmatch(self.harness_id):
            raise ValueError(f"非法 Harness ID：{self.harness_id}")
        if self.source_type not in HARNESS_SOURCE_TYPES:
            raise ValueError(f"不支持的 Harness source_type：{self.source_type}")
        if self.version < 0:
            raise ValueError("Harness version 不能为负数")
        for module in HARNESS_MODULES:
            value = getattr(self, module)
            if not isinstance(value, dict):
                raise ValueError(f"Harness.{module} 必须是对象")
            _json_safe(value)
        if len(self.cards) > 12:
            raise ValueError("单个 Harness 最多包含 12 张战术卡")
        ids: set[str] = set()
        for card in self.cards:
            card.validate()
            if card.card_id in ids:
                raise ValueError(f"战术卡 ID 重复：{card.card_id}")
            ids.add(card.card_id)
        serialized_size = len(_canonical_json(self._content_dict()))
        if serialized_size > 60000:
            raise ValueError("Harness 配置过大，不能退化为全量提示词")

    @property
    def fingerprint(self) -> str:
        payload = self._content_dict()
        payload.pop("harness_id", None)
        return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()

    def _content_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "harness_id": self.harness_id,
            "role": self.role,
            "source_type": self.source_type,
            "version": self.version,
            "parent_id": self.parent_id,
            "context_policy": deepcopy(self.context_policy),
            "belief_board": deepcopy(self.belief_board),
            "tactic_router": deepcopy(self.tactic_router),
            "planning_policy": deepcopy(self.planning_policy),
            "memory_policy": deepcopy(self.memory_policy),
            "coordination_policy": deepcopy(self.coordination_policy),
            "output_policy": deepcopy(self.output_policy),
            "cards": [card.as_dict() for card in self.cards],
            "rationale": list(self.rationale),
            "evidence_refs": list(self.evidence_refs),
        }

    def as_dict(self) -> dict[str, Any]:
        value = self._content_dict()
        value["fingerprint"] = self.fingerprint
        return value

    def manifest(self) -> dict[str, Any]:
        """返回可放入对局记录的精简版本信息。"""

        return {
            "role": self.role,
            "harness_id": self.harness_id,
            "harness_fingerprint": self.fingerprint,
            "version": self.version,
            "parent_id": self.parent_id,
            "source_type": self.source_type,
            "card_ids": [card.card_id for card in self.cards],
        }


class HarnessRuntime:
    """在不执行任意模型代码的前提下解释 HarnessSpec。"""

    def __init__(self, spec: HarnessSpec, profile: RoleProfile) -> None:
        spec.validate()
        if spec.role != profile.role:
            raise ValueError(f"Harness 角色 {spec.role} 与档案角色 {profile.role} 不一致")
        self.spec = spec
        self.profile = profile
        self._last_signature: tuple[Any, ...] | None = None
        self._last_event_seq = 0
        self._belief_board: dict[str, list[str]] = {
            str(column): []
            for column in self._belief_columns()
        }

    def build_context(
        self,
        turn_packet: Mapping[str, Any],
        *,
        private_notes: list[str] | tuple[str, ...] = (),
        latest_public_sync: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """筛选当前行动真正需要的少量上下文。"""

        policy = self.spec.context_policy
        raw_include = policy.get("include")
        include = {
            str(item).strip().lower()
            for item in raw_include
            if str(item).strip()
        } if isinstance(raw_include, (list, tuple, set, frozenset)) else set()
        if not include:
            include = {
                "game",
                "public_rules",
                "self",
                "private_information",
                "public_state",
                "request",
            }
        max_visible = self._positive_int(policy.get("max_visible_events"), 28, 100)
        max_public = self._positive_int(policy.get("max_public_events"), 36, 120)
        max_event_chars = self._positive_int(policy.get("max_event_text_chars"), 260, 1200)
        selected_visible = self._clip_events(turn_packet.get("visible_events"), max_visible, max_event_chars)
        selected_sync_events = self._clip_events(
            (latest_public_sync or {}).get("public_events"), max_public, max_event_chars
        )
        max_notes = self._positive_int(
            self.spec.memory_policy.get("max_notes", policy.get("max_private_notes")),
            10,
            40,
        )
        notes = [_text(item, limit=self._positive_int(self.spec.memory_policy.get("max_note_chars"), 240, 1000)) for item in private_notes]
        notes = [item for item in notes if item][-max_notes:]
        selected_cards = self._select_cards(turn_packet, selected_visible, selected_sync_events)
        replan, replan_reason = self._replan_state(turn_packet, selected_visible, selected_sync_events)
        phase = str(turn_packet.get("game", {}).get("phase") or "")
        public_phase = public_phase_for(phase) if phase else phase
        raw_strategy = self.profile.strategy
        strategy_mode = str(policy.get("strategy_mode") or "bounded").lower()
        if strategy_mode == "none":
            strategy = ""
        else:
            limit = self._positive_int(policy.get("strategy_char_limit"), 2600, 6000)
            strategy = raw_strategy[:limit]
        context = {
            "game": deepcopy(turn_packet.get("game", {})) if "game" in include else {},
            "public_rules": deepcopy(turn_packet.get("public_rules", {})),
            "self": deepcopy(turn_packet.get("self", {})),
            "private_information": deepcopy(turn_packet.get("private_information", {})),
            "public_state": deepcopy(turn_packet.get("public_state", {})) if "public_state" in include else {},
            "visible_events": selected_visible if include.intersection({"visible_events", "events", "recent_events"}) else [],
            "synchronized_public_events": selected_sync_events if include.intersection({"public_events", "synchronized_public_events", "events"}) else [],
            "request": deepcopy(turn_packet.get("request", {})),
            "task_agent_context": {
                "harness_id": self.spec.harness_id,
                "harness_fingerprint": self.spec.fingerprint,
                "public_phase": public_phase,
                "replan_required": replan,
                "replan_reason": replan_reason,
                "selected_cards": [card.as_dict() for card in selected_cards],
                "belief_board_template": deepcopy(self.spec.belief_board),
                "belief_board": deepcopy(self._belief_board),
                "planning_policy": deepcopy(self.spec.planning_policy),
                "memory_notes": notes,
                "bounded_role_strategy": strategy,
            },
        }
        return context

    def update_belief_board(self, update: object) -> dict[str, list[str]]:
        """应用模型返回的受限信念更新，不保存完整推理过程。"""

        if not isinstance(update, Mapping):
            return deepcopy(self._belief_board)
        max_items = self._positive_int(self.spec.belief_board.get("max_items"), 12, 40)
        max_chars = self._positive_int(
            self.spec.memory_policy.get("max_note_chars"), 240, 1000
        )
        columns = set(self._belief_board)
        for raw_column, raw_values in update.items():
            column = str(raw_column).strip()
            if column not in columns:
                continue
            if isinstance(raw_values, str):
                values = [raw_values]
            elif isinstance(raw_values, (list, tuple)):
                values = list(raw_values)
            else:
                continue
            for raw_value in values:
                if isinstance(raw_value, Mapping):
                    value = raw_value.get("text", raw_value.get("claim", ""))
                else:
                    value = raw_value
                normalized = _text(value, limit=max_chars)
                if not normalized:
                    continue
                current = self._belief_board[column]
                if normalized in current:
                    current.remove(normalized)
                current.append(normalized)
                self._belief_board[column] = current[-max_items:]
        return deepcopy(self._belief_board)

    def belief_board_snapshot(self) -> dict[str, list[str]]:
        """返回当前信念板的副本，供审计或测试使用。"""

        return deepcopy(self._belief_board)

    def _belief_columns(self) -> tuple[str, ...]:
        raw_columns = self.spec.belief_board.get("columns")
        if isinstance(raw_columns, (list, tuple)):
            columns = tuple(
                dict.fromkeys(
                    str(item).strip()
                    for item in raw_columns
                    if str(item).strip()
                )
            )
            if columns:
                return columns
        return ("confirmed_facts", "player_claims", "hypotheses", "unresolved_questions")

    def remember(self, notes: list[str], note: object) -> list[str]:
        """按 Harness 的记忆策略保存短期笔记，不保存完整思维链。"""

        if not isinstance(note, str):
            return list(notes)
        limit = self._positive_int(self.spec.memory_policy.get("max_note_chars"), 240, 1000)
        normalized = _text(note, limit=limit)
        if not normalized:
            return list(notes)[-self._positive_int(self.spec.memory_policy.get("max_notes"), 10, 40) :]
        result = [item for item in notes if isinstance(item, str) and item.strip()]
        if normalized in result:
            result.remove(normalized)
        result.append(normalized)
        max_notes = self._positive_int(self.spec.memory_policy.get("max_notes"), 10, 40)
        return result[-max_notes:]

    def strategy_for_prompt(self) -> str:
        policy = self.spec.context_policy
        if str(policy.get("strategy_mode") or "bounded").lower() == "none":
            return ""
        limit = self._positive_int(policy.get("strategy_char_limit"), 2600, 6000)
        return self.profile.strategy[:limit]

    def manifest(self) -> dict[str, Any]:
        return {
            "agent_type": "task_agent_harness",
            "role": self.spec.role,
            "harness_id": self.spec.harness_id,
            "harness_fingerprint": self.spec.fingerprint,
            "version": self.spec.version,
            "parent_id": self.spec.parent_id,
            "source_type": self.spec.source_type,
            "card_ids": [card.card_id for card in self.spec.cards],
        }

    def _replan_state(
        self,
        turn_packet: Mapping[str, Any],
        visible_events: list[dict[str, Any]],
        sync_events: list[dict[str, Any]],
    ) -> tuple[bool, str]:
        game = turn_packet.get("game") if isinstance(turn_packet.get("game"), Mapping) else {}
        phase = str(game.get("phase") or "")
        round_number = game.get("round")
        latest_seq = max(
            [self._event_seq(item) for item in visible_events + sync_events] + [self._last_event_seq]
        )
        signature = (phase, round_number, latest_seq)
        if self._last_signature is None:
            reason = "首次行动，需要建立信念板"
            changed = True
        elif phase != self._last_signature[0]:
            reason = "阶段发生变化"
            changed = True
        elif round_number != self._last_signature[1]:
            reason = "游戏轮次发生变化"
            changed = True
        elif latest_seq > self._last_event_seq:
            reason = "出现新的可见事件"
            changed = True
        else:
            reason = "没有检测到触发重规划的新事件"
            changed = False
        self._last_signature = signature
        self._last_event_seq = latest_seq
        return changed, reason

    def _select_cards(
        self,
        turn_packet: Mapping[str, Any],
        visible_events: list[dict[str, Any]],
        sync_events: list[dict[str, Any]],
    ) -> list[TacticalCard]:
        router = self.spec.tactic_router
        maximum = self._positive_int(router.get("max_cards_per_turn"), 3, 6)
        game = turn_packet.get("game") if isinstance(turn_packet.get("game"), Mapping) else {}
        phase_text = f"{game.get('phase', '')} {public_phase_for(str(game.get('phase') or ''))}"
        evidence = _canonical_json(visible_events + sync_events).lower()
        scored: list[tuple[float, TacticalCard]] = []
        for card in self.spec.cards:
            trigger = card.trigger.lower()
            score = card.confidence
            if not trigger:
                score += 0.2
            for term in self._terms(trigger):
                if term in phase_text.lower() or term in evidence:
                    score += 0.4
            for tag in card.tags:
                if tag.lower() in phase_text.lower():
                    score += 0.25
            scored.append((score, card))
        scored.sort(key=lambda item: (-item[0], item[1].card_id))
        return [card for _, card in scored[:maximum]]

    @staticmethod
    def _terms(value: str) -> tuple[str, ...]:
        terms = re.findall(r"[\u3400-\u9fff]{2,}|[a-zA-Z_]{3,}", value.lower())
        return tuple(dict.fromkeys(terms))

    @classmethod
    def _clip_events(cls, value: object, maximum: int, text_limit: int) -> list[dict[str, Any]]:
        if not isinstance(value, (list, tuple)):
            return []
        result: list[dict[str, Any]] = []
        for raw in list(value)[-maximum:]:
            if not isinstance(raw, Mapping):
                continue
            item = deepcopy(dict(raw))
            cls._truncate_strings(item, text_limit)
            result.append(item)
        return result

    @classmethod
    def _truncate_strings(cls, value: Any, limit: int) -> None:
        if isinstance(value, dict):
            for key, item in list(value.items()):
                if isinstance(item, str):
                    value[key] = item[:limit]
                else:
                    cls._truncate_strings(item, limit)
        elif isinstance(value, list):
            for item in value:
                cls._truncate_strings(item, limit)

    @staticmethod
    def _event_seq(event: Mapping[str, Any]) -> int:
        try:
            return max(0, int(event.get("seq", 0)))
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _positive_int(value: object, fallback: int, maximum: int) -> int:
        try:
            normalized = int(value)
        except (TypeError, ValueError):
            normalized = fallback
        return max(1, min(maximum, normalized))


@dataclass(frozen=True)
class HarnessStaticEvaluation:
    """不调用模型的 Harness 质量闸门结果。

    Season 2 的候选不能只凭模型红队意见晋升。这个评估器用固定的合成行动包
    解释候选，检查上下文、记忆和重规划边界是否真的被运行时执行。它不是胜率
    评估，也不替代冻结对手池；它负责先挡住结构上不合格的候选。
    """

    passed: bool
    score: float
    checks: dict[str, bool] = field(default_factory=dict)
    metrics: dict[str, int | float | bool] = field(default_factory=dict)
    violations: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "score": self.score,
            "checks": dict(self.checks),
            "metrics": dict(self.metrics),
            "violations": list(self.violations),
        }


def evaluate_harness(spec: HarnessSpec, profile: RoleProfile) -> HarnessStaticEvaluation:
    """用确定性的合成行动包检查一个候选 Harness。

    这里刻意不调用外部模型、不读取文件，也不访问 GameEngine。所有检查只验证
    ``HarnessRuntime`` 是否遵守候选声明的上限和公开事件触发规则，因此可在每个
    Meta-Agent 候选上重复执行，并可在没有 API 配额时运行测试。
    """

    checks: dict[str, bool] = {}
    metrics: dict[str, int | float | bool] = {}
    violations: list[str] = []
    try:
        spec.validate()
        checks["schema_valid"] = True
    except (TypeError, ValueError) as error:
        checks["schema_valid"] = False
        violations.append(f"schema 校验失败：{type(error).__name__}")
        return HarnessStaticEvaluation(
            passed=False,
            score=0.0,
            checks=checks,
            metrics=metrics,
            violations=tuple(violations),
        )

    try:
        runtime = HarnessRuntime(spec, profile)
        packet: dict[str, Any] = {
            "game": {"phase": "day_discussion", "round": 1},
            "public_rules": {"language": "中文"},
            "self": {"player_id": "synthetic-player"},
            "private_information": {
                "role": profile.role,
                "team": "wolf" if profile.role == "wolf" else "village",
            },
            "public_state": {"alive_player_ids": ["synthetic-player", "p2"]},
            "visible_events": [
                {
                    "seq": index,
                    "type": "PLAYER_SPOKE",
                    "payload": {"text": "合成公开事件" * 200},
                }
                for index in range(80)
            ],
            "request": {
                "kind": "discussion",
                "allowed_actions": [{"kind": "speak", "max_chars": 200}],
            },
        }
        context = runtime.build_context(
            packet,
            private_notes=[f"合成记忆 {index}" for index in range(40)],
        )
        task_context = context.get("task_agent_context", {})
        if not isinstance(task_context, Mapping):
            task_context = {}

        visible_events = context.get("visible_events", [])
        memory_notes = task_context.get("memory_notes", [])
        checks["context_bounded"] = isinstance(visible_events, list) and len(visible_events) <= 120
        checks["memory_bounded"] = isinstance(memory_notes, list) and len(memory_notes) <= 40
        checks["event_text_bounded"] = _nested_string_lengths_bounded(visible_events, 1200)
        checks["replanning_on_first_turn"] = bool(task_context.get("replan_required"))
        checks["cards_have_exit_conditions"] = all(
            bool(card.exit_conditions) for card in spec.cards
        )
        checks["cards_have_observable_signals"] = all(
            bool(card.observable_signals) for card in spec.cards
        )
        belief_board = task_context.get("belief_board")
        max_belief_items = HarnessRuntime._positive_int(
            spec.belief_board.get("max_items"), 12, 40
        )
        checks["belief_board_bounded"] = isinstance(belief_board, dict) and all(
            isinstance(values, list) and len(values) <= max_belief_items
            for values in belief_board.values()
        )
        checks["task_context_is_audit_safe"] = not any(
            key in task_context for key in {"prompt", "system_prompt", "api_key", "code"}
        )
        context_size = len(_canonical_json(context))
        checks["context_size_bounded"] = context_size <= 40000
        metrics.update(
            {
                "visible_event_count": len(visible_events) if isinstance(visible_events, list) else 0,
                "memory_note_count": len(memory_notes) if isinstance(memory_notes, list) else 0,
                "card_count": len(spec.cards),
                "selected_card_count": len(task_context.get("selected_cards", []))
                if isinstance(task_context.get("selected_cards"), list)
                else 0,
                "belief_column_count": len(belief_board)
                if isinstance(belief_board, dict)
                else 0,
                "context_chars": context_size,
            }
        )
        for name, passed in checks.items():
            if not passed:
                violations.append(f"静态检查未通过：{name}")
    except (TypeError, ValueError, KeyError, AttributeError) as error:
        checks["runtime_interpretable"] = False
        violations.append(f"运行时解释失败：{type(error).__name__}")
    else:
        checks["runtime_interpretable"] = True

    passed_count = sum(checks.values())
    score = round(100.0 * passed_count / max(1, len(checks)), 2)
    return HarnessStaticEvaluation(
        passed=not violations,
        score=score,
        checks=checks,
        metrics=metrics,
        violations=tuple(violations),
    )


def _nested_string_lengths_bounded(value: object, limit: int) -> bool:
    """检查运行时裁剪后的结构中没有超长字符串。"""

    if isinstance(value, str):
        return len(value) <= limit
    if isinstance(value, Mapping):
        return all(_nested_string_lengths_bounded(item, limit) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_nested_string_lengths_bounded(item, limit) for item in value)
    return True


def default_harness_for_profile(profile: RoleProfile) -> HarnessSpec:
    return HarnessSpec.baseline(profile.role, strategy=profile.strategy)


class HarnessFileStore:
    """读取和写入 active Harness；archive 的写入由 meta_agent 负责。"""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory).resolve()
        self.active_directory = self.directory / "active"

    def active_path(self, role: str) -> Path:
        if role not in ALL_ROLES:
            raise ValueError(f"不支持的角色：{role}")
        return self.active_directory / f"{role}.json"

    def load_active(self, role: str, *, profile: RoleProfile | None = None) -> HarnessSpec:
        path = self.active_path(role)
        if not path.exists():
            return default_harness_for_profile(profile or RoleProfile(role, "", ""))
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"active Harness JSON 无法读取：{path}") from error
        return HarnessSpec.from_mapping(value, role=role)

    def save_active(self, spec: HarnessSpec) -> Path:
        spec.validate()
        self.active_directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        path = self.active_path(spec.role)
        temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        temporary.write_text(json.dumps(spec.as_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        try:
            temporary.chmod(0o600)
        except PermissionError:
            pass
        temporary.replace(path)
        try:
            path.chmod(0o600)
        except PermissionError:
            pass
        return path


# 语义别名：文档中把 HarnessSpec 称为 Task-Agent Harness 的“基因”。
TaskAgentHarness = HarnessSpec

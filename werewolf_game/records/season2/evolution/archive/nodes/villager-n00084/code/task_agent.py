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
_PLAYER_ID_SUFFIX = re.compile(r"(\d+)$")

CURRENT_ROUND_DIALOGUE_TOOL_NAME = "read_current_round_dialogue"
DEFAULT_MAX_TOOL_CALLS_PER_DECISION = 5
DEFAULT_MAX_TOOL_RESULT_TOKENS = 1000
DEFAULT_MAX_PROMPT_CHARS = 12000
# 这是 Task-Agent 层的“纠错重试”次数：首次模型调用之外，最多再请求两次。
# 传输层的 HTTP/网络重试仍由 ModelClient 的 MODEL_MAX_RETRIES 单独控制。
DEFAULT_MAX_DECISION_RETRIES = 2
_SPEECH_ACTION_KINDS = frozenset({"speak", "last_words"})

_PUBLIC_SNAPSHOT_LIMIT = 18
_PUBLIC_NOTE_CHAR_LIMIT = 140
_PUBLIC_BLOCK_CHAR_LIMIT = 1600
_PHASE_BLOCK_CHAR_LIMIT = 800


CURRENT_ROUND_DIALOGUE_TOOL: dict[str, Any] = {
    "name": CURRENT_ROUND_DIALOGUE_TOOL_NAME,
    "description": "读取当前昼夜轮次中、当前玩家依法可见的已发生发言。",
    "parameters": {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    },
}


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
    lines = ["【平民决策清单】先读 verified_public_facts / vote_records / death_records，再读 player_claims；最后才做推测。玩家说法（包括金水、预言家、警徽链、遗言）不是系统确认；投票前必须比较两个当前合法候选。"]
    has_vote = any(key in phase_lower for key in ("vote", "投票")) or any("vote" in kind for kind in allowed_kinds)
    has_speak = any(key in phase_lower for key in ("speak", "discussion", "发言", "白天", "day")) or any(
        kind in {"speak", "discussion", "day_speak"} for kind in allowed_kinds
    )
    has_last_words = any(key in phase_lower for key in ("last", "遗言")) or "last_words" in allowed_kinds
    has_sheriff = any(key in phase_lower for key in ("sheriff", "警长", "警徽")) or any(
        key in allowed_kinds for key in ("sheriff_vote", "sheriff")
    )
    if has_vote:
        lines.append("投票时把系统公开死亡/存活/票型放在最前；玩家金水、预言家、强神和警徽链只算声明或公开行为，不能自动升级为确认。硬对跳、事实错误、查验断裂或来源不明的夜间信息优先于重复站边。")
        lines.append("只在当前 allowed_actions 的 target_ids 中选人；优先找未被独立公开事实保护且有可追溯矛盾或成组票型异常的人。存在明确矛盾时不要 pass。")
    if has_speak:
        lines.append("发言至少点出一个关注对象和一个暂不投对象，并说明依据哪条公开信息。")
    if has_last_words:
        lines.append("遗言只留公开事实、票型、查验链和怀疑对象，不把推测说成系统事实。")
    if has_sheriff:
        lines.append("警长/警徽相关回合优先看报验链是否清晰一致；接徽时继承公开遗产但不要把它当系统确认。")
    lines.append("残局（存活不超过 6 人）或持警徽时，依次复盘存活/死亡、verified_public_facts、断裂查验/警徽链、历史票型和来源不明声明；不要用沉默或重复发言替代证据。")
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
        self._public_ledger_game_id: str | None = None
        self._public_ledger: dict[str, Any] = {}
        self._reset_public_ledger(None)

    async def observe(self, sync_packet: dict[str, Any]) -> None:
        """接收公开同步并维护一个仅含公开事实的轻量台账。"""

        if not isinstance(sync_packet, Mapping):
            return
        game_id = self._extract_game_id(sync_packet)
        if game_id is not None and game_id != self._public_ledger_game_id:
            self._reset_public_ledger(game_id)
        self._ingest_public_packet(sync_packet)

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
                "instruction": self._turn_instruction(turn_packet, feedback),
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

    def _turn_instruction(self, turn_packet: Mapping[str, Any], feedback: str) -> str:
        request = turn_packet["request"]
        checklist = render_villager_decision_checklist(request)
        summary = self._render_turn_summary(turn_packet, request)
        feedback_text = f"上一次输出未通过校验：{feedback}" if feedback else ""
        validation_feedback = "\n".join(
            part for part in (summary, feedback_text, checklist) if part
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

    def _reset_public_ledger(self, game_id: str | None) -> None:
        self._public_ledger_game_id = game_id
        self._public_ledger = {
            "alive_ids": set(),
            "dead_ids": set(),
            "known_player_ids": set(),
            "records": [],
            "verified_public_facts": [],
            "player_claims": [],
            "votes": [],
            "deaths": [],
            "badges": [],
            "recent_events": [],
        }

    def _ingest_public_packet(self, payload: Any) -> None:
        if not isinstance(payload, Mapping):
            return
        player_ids = self._extract_player_ids(payload, include_collections=True)
        self._public_ledger["known_player_ids"].update(player_ids)
        self._public_ledger["alive_ids"].update(
            self._collect_id_collection(payload, self._ALIVE_KEYS())
        )
        self._public_ledger["dead_ids"].update(
            self._collect_id_collection(payload, self._DEAD_KEYS())
        )
        for record in self._extract_public_records(payload):
            self._merge_record(record)
            for key in ("speaker", "voter", "target"):
                value = record.get(key)
                if value and self._looks_like_player_id(value):
                    self._public_ledger["known_player_ids"].add(value)

    def _merge_record(self, record: dict[str, str]) -> None:
        key = tuple(record.get(field, "") for field in (
            "round", "event_type", "speaker", "voter", "target", "text"
        ))
        records = self._public_ledger["records"]
        if any(tuple(item.get(field, "") for field in (
            "round", "event_type", "speaker", "voter", "target", "text"
        )) == key for item in records):
            return
        self._append_bounded(records, record, _PUBLIC_SNAPSHOT_LIMIT * 3)
        event_type = record["event_type"]
        if event_type in {"vote", "death", "badge"}:
            self._append_bounded(
                self._public_ledger["verified_public_facts"], record, _PUBLIC_SNAPSHOT_LIMIT
            )
        if event_type in {"role_claim", "inspection_claim", "support", "opposition", "event"}:
            self._append_bounded(self._public_ledger["player_claims"], record, _PUBLIC_SNAPSHOT_LIMIT)
        if event_type == "vote":
            self._append_bounded(self._public_ledger["votes"], record, _PUBLIC_SNAPSHOT_LIMIT)
        elif event_type == "death":
            self._append_bounded(self._public_ledger["deaths"], record, _PUBLIC_SNAPSHOT_LIMIT)
            if record.get("target"):
                self._public_ledger["dead_ids"].add(record["target"])
        elif event_type == "badge":
            self._append_bounded(self._public_ledger["badges"], record, _PUBLIC_SNAPSHOT_LIMIT)
        self._append_bounded(self._public_ledger["recent_events"], record, _PUBLIC_SNAPSHOT_LIMIT)

    def _records_for_payload(self, payload: Any) -> list[dict[str, str]]:
        records = list(self._public_ledger.get("records", []))
        seen = {tuple(item.get(field, "") for field in (
            "round", "event_type", "speaker", "voter", "target", "text"
        )) for item in records}
        for record in self._extract_public_records(payload):
            key = tuple(record.get(field, "") for field in (
                "round", "event_type", "speaker", "voter", "target", "text"
            ))
            if key not in seen:
                records.append(record)
                seen.add(key)
        return records[-_PUBLIC_SNAPSHOT_LIMIT * 3:]

    def _extract_public_records(self, payload: Any) -> list[dict[str, str]]:
        found: list[dict[str, str]] = []
        seen: set[tuple[str, ...]] = set()

        def walk(value: Any, depth: int) -> None:
            if depth > 5:
                return
            if isinstance(value, Mapping):
                record = self._snapshot_mapping(value)
                if record is not None:
                    key = tuple(record.get(field, "") for field in (
                        "round", "event_type", "speaker", "voter", "target", "text"
                    ))
                    if key not in seen:
                        seen.add(key)
                        found.append(record)
                for child in value.values():
                    walk(child, depth + 1)
            elif isinstance(value, list):
                for child in value[:25]:
                    walk(child, depth + 1)

        walk(payload, 0)
        return found[-_PUBLIC_SNAPSHOT_LIMIT * 2:]

    def _snapshot_mapping(self, mapping: Mapping[str, Any]) -> dict[str, str] | None:
        """将一个明确的公开事件转换为 bounded record。

        关键约束：target_id 本身没有事件语义，不能因此生成 vote。未知结构只进
        event 桶，且 source_kind 仍为 player_claim，避免把玩家文本升级为系统事实。
        """
        values = {str(key).lower(): value for key, value in mapping.items()}
        event_raw = self._first_scalar(values, ("event_type", "event", "type", "kind", "action"))
        event = event_raw.lower().replace("-", "_").replace(" ", "_")
        text = self._first_scalar(values, ("text", "content", "message", "speech", "reason"))

        def has(*terms: str) -> bool:
            return any(term in event for term in terms)

        explicit_vote = (
            ("voter_id" in values or "voter" in values or "actor_id" in values)
            and ("vote_target" in values or "target_id" in values or "target" in values)
        )
        if has("vote", "ballot", "投票") and explicit_vote:
            event_type = "vote"
        elif has("death", "died", "dead", "kill", "eliminat", "out", "死亡", "出局") or any(
            key in values for key in ("dead_id", "dead_player_id", "death_id")
        ):
            event_type = "death"
        elif has("badge", "sheriff", "警徽", "警长") or any(
            key in values for key in ("badge_from", "badge_to", "badge_holder", "badge_holder_id")
        ):
            event_type = "badge"
        elif has("inspection", "inspect", "check", "seer", "查验", "验人", "查杀", "金水"):
            event_type = "inspection_claim"
        elif has("role_claim", "claim", "self_claim", "自称", "身份"):
            event_type = "role_claim"
        elif has("support", "agree", "back", "站边", "支持"):
            event_type = "support"
        elif has("opposition", "oppose", "disagree", "对跳", "反对"):
            event_type = "opposition"
        elif text and ("speaker_id" in values or "claimer_id" in values or "actor_id" in values):
            event_type = "event"
        elif event_raw:
            # 未知事件保留为普通 event；绝不凭 target_id 推断票型或系统事实。
            event_type = "event"
        elif explicit_vote and ("vote" in values or "ballot" in values):
            event_type = "vote"
        else:
            return None

        round_text = self._first_scalar(values, ("round", "day", "turn", "phase"))
        speaker = self._first_scalar(values, ("speaker_id", "claimer_id", "actor_id", "player_id"))
        voter = self._first_scalar(values, ("voter_id", "voter", "actor_id", "player_id"))
        target = self._first_scalar(values, (
            "vote_target", "target_id", "target", "suspect_id", "to_id", "badge_to", "dead_id", "dead_player_id", "dead", "death"
        ))
        role = self._first_scalar(values, ("claim_role", "claimed_role", "role"))
        if event_type == "vote":
            speaker = ""
        elif event_type == "death":
            speaker = ""
            voter = ""
        source_kind = "system_fact" if event_type in {"vote", "death", "badge"} else "player_claim"
        if not text and role:
            text = f"claimed_role={role}"
        if event_type == "death" and not target:
            target = self._first_scalar(values, ("player_id", "actor_id"))
        if not any((round_text, speaker, voter, target, text)):
            return None
        return {
            "round": self._trim_text(round_text, 24),
            "event_type": event_type,
            "speaker": self._trim_text(speaker, 24),
            "voter": self._trim_text(voter, 24),
            "target": self._trim_text(target, 24),
            "source_kind": source_kind,
            "text": self._trim_text(text, _PUBLIC_NOTE_CHAR_LIMIT),
        }

    def _render_turn_summary(self, turn_packet: Mapping[str, Any], request: Mapping[str, Any]) -> str:
        public_state = turn_packet.get("public_state")
        records = self._records_for_payload(public_state)
        ledger_alive = set(self._public_ledger["alive_ids"])
        ledger_dead = set(self._public_ledger["dead_ids"])
        current_alive = self._extract_player_ids(public_state, include_collections=True)
        current_dead = self._collect_id_collection(public_state, self._DEAD_KEYS()) if isinstance(public_state, Mapping) else set()
        dead_ids = self._sort_player_ids(ledger_dead | current_dead | {
            record["target"] for record in records if record["event_type"] == "death" and record.get("target")
        })
        alive_ids = self._sort_player_ids((ledger_alive | current_alive) - set(dead_ids))
        known_ids = self._sort_player_ids(
            set(self._public_ledger["known_player_ids"]) | set(alive_ids) | set(dead_ids)
        )
        candidates = self._candidate_pool(request, alive_ids, known_ids, dead_ids)
        verified = [r for r in records if r["source_kind"] == "system_fact"]
        claims = [r for r in records if r["source_kind"] == "player_claim"]
        lines = ["【结构化公共台账】"]
        lines.append("存活: " + self._join_ids(alive_ids, limit=12))
        lines.append("已死: " + self._join_ids(dead_ids, limit=12))
        lines.append("verified_public_facts: " + self._render_records(verified, 6))
        lines.append("player_claims: " + self._render_records(claims, 8))
        lines.append("vote_records: " + self._render_records([r for r in records if r["event_type"] == "vote"], 8))
        lines.append("death_records: " + self._render_records([r for r in records if r["event_type"] == "death"], 5))
        lines.append("badge_records: " + self._render_records([r for r in records if r["event_type"] == "badge"], 5))
        if len(alive_ids) <= 6:
            lines.append("【残局协议】先复盘存活/死亡、有效系统事实、断裂查验或警徽链、历史票型和来源不明声明；不要用沉默或重复发言替代复盘。")
        if self._is_vote_request(request):
            phase_lines = self._render_vote_summary(candidates, alive_ids, dead_ids, records)
        else:
            phase_lines = self._render_speak_summary(candidates, alive_ids, dead_ids, records)
        return self._trim_text("\n".join(lines + [phase_lines]), _PUBLIC_BLOCK_CHAR_LIMIT + _PHASE_BLOCK_CHAR_LIMIT)

    def _render_records(self, records: list[dict[str, str]], limit: int) -> str:
        if not records:
            return "无"
        rendered: list[str] = []
        for record in records[-limit:]:
            fields = [record.get("round", "") or "?", record.get("event_type", "event")]
            for key in ("speaker", "voter", "target"):
                if record.get(key):
                    fields.append(f"{key}={record[key]}")
            if record.get("text"):
                fields.append("text=" + self._trim_text(record["text"], 70))
            rendered.append("{" + ",".join(fields) + "}")
        return " | ".join(rendered)

    def _render_vote_summary(
        self, candidate_pool: list[str], alive_ids: list[str], dead_ids: list[str], records: list[dict[str, str]]
    ) -> str:
        ranked = self._rank_candidates(candidate_pool, records, dead_ids)
        lines = ["【本轮投票：仅比较当前合法候选】"]
        if not ranked:
            return "\n".join(lines + ["当前没有合法目标；不要臆造 target_id。"])
        for index, (candidate, score, evidence, counter) in enumerate(ranked[:2], 1):
            lines.append(f"候选{index}: {candidate} | 结构化分={score} | 证据: {evidence or '无'} | 反证/不确定: {counter or '无'}")
        if len(ranked) >= 2:
            lines.append(f"必须双候选比较: {ranked[0][0]} vs {ranked[1][0]}。以可追溯公开记录为先，不因沉默或多数重复就集火。")
        else:
            lines.append("合法候选不足两个：明确说明信息不足，不加入历史或已死玩家。")
        hard = any(any(word in (r.get("text", "") + r.get("event_type", "")) for word in ("查杀", "矛盾", "越权", "来源不明", "对跳")) for r in records)
        vote_groups = sum(1 for r in records if r["event_type"] == "vote" and r.get("target"))
        if len(candidate_pool) > 1 and (hard or vote_groups >= 3):
            lines.append("存在硬矛盾、来源不明声明或成组票型时，不要 pass；只有近似等价或继续投票明显误伤独立硬事实保护对象才考虑 pass。")
        return self._trim_text("\n".join(lines), _PHASE_BLOCK_CHAR_LIMIT)

    def _render_speak_summary(
        self, candidate_pool: list[str], alive_ids: list[str], dead_ids: list[str], records: list[dict[str, str]]
    ) -> str:
        ranked = self._rank_candidates(candidate_pool, records, dead_ids)
        lines = ["【本轮发言参考】"]
        if ranked:
            lines.append(f"关注: {ranked[0][0]} | {ranked[0][2] or '公开信息不足'}")
        else:
            lines.append("关注: 信息不足")
        avoid = self._least_suspicious_candidate(candidate_pool, records, dead_ids)
        lines.append(f"暂不投: {avoid[0]} | {avoid[2]}" if avoid else "暂不投: 信息不足，说明不确定性")
        lines.append("发言需区分系统事实和玩家声明，并点出依据；未知夜间信息必须标为来源不明。")
        return self._trim_text("\n".join(lines), _PHASE_BLOCK_CHAR_LIMIT)

    def _rank_candidates(
        self, candidate_pool: list[str], records: list[dict[str, str]], dead_ids: list[str]
    ) -> list[tuple[str, int, str, str]]:
        ranked: list[tuple[str, int, str, str]] = []
        for candidate in self._sort_player_ids(candidate_pool):
            if candidate == self.player_id or candidate in set(dead_ids):
                continue
            score, evidence, counter = self._score_candidate(candidate, records)
            ranked.append((candidate, score, "；".join(evidence[:3]), "；".join(counter[:2])))
        ranked.sort(key=lambda item: (-item[1], item[0]))
        return ranked

    def _least_suspicious_candidate(
        self, candidate_pool: list[str], records: list[dict[str, str]], dead_ids: list[str]
    ) -> tuple[str, int, str, str] | None:
        ranked = self._rank_candidates(candidate_pool, records, dead_ids)
        return min(ranked, key=lambda item: (item[1], item[0])) if ranked else None

    def _score_candidate(
        self, candidate: str, records: list[dict[str, str]]
    ) -> tuple[int, list[str], list[str]]:
        score = 0
        evidence: list[str] = []
        counter: list[str] = []
        for record in records:
            event_type = record["event_type"]
            text = record.get("text", "")
            short = self._trim_text(self._render_records([record], 1), 110)
            if event_type == "inspection_claim" and record.get("target") == candidate:
                if any(term in text for term in ("查杀", "狼", "坏")):
                    score += 3
                    evidence.append("查验声明指向该玩家: " + short)
                elif any(term in text for term in ("金水", "好人", "验过")):
                    score -= 2
                    counter.append("查验声明保护该玩家: " + short)
            elif event_type in {"opposition", "support", "role_claim", "event"}:
                involved = candidate in {record.get("speaker", ""), record.get("target", "")}
                if involved and any(term in text for term in ("矛盾", "对跳", "越权", "来源不明", "假", "狼")):
                    score += 2
                    evidence.append("玩家声明存在冲突/来源问题: " + short)
                elif involved and any(term in text for term in ("金水", "好人", "可信")):
                    score -= 1
                    counter.append("玩家声明提供保护但非系统确认: " + short)
            elif event_type == "vote" and record.get("target") == candidate:
                score += 1
                evidence.append("公开票型记录: " + short)
        if not evidence and not counter:
            counter.append("暂无可追溯结构化证据")
        return score, evidence, counter

    def _candidate_pool(
        self, request: Mapping[str, Any], alive_ids: list[str], known_ids: list[str], dead_ids: list[str] | None = None
    ) -> list[str]:
        dead = set(dead_ids or ()) | set(self._public_ledger.get("dead_ids", set()))
        raw_actions = request.get("allowed_actions")
        actions = [item for item in raw_actions if isinstance(item, Mapping)] if isinstance(raw_actions, list) else []
        if self._is_vote_request(request):
            pool: set[str] = set()
            for action in actions:
                kind = str(action.get("kind") or "").lower()
                if "vote" not in kind:
                    continue
                targets = action.get("target_ids")
                if isinstance(targets, list):
                    pool.update(str(value) for value in targets if self._is_scalar_id(value))
        else:
            pool = set(alive_ids) | set(known_ids)
        pool.discard(self.player_id)
        pool.difference_update(dead)
        return self._sort_player_ids(list(pool))

    def _is_vote_request(self, request: Mapping[str, Any]) -> bool:
        phase = str(request.get("phase") or request.get("channel") or request.get("kind") or "").lower()
        actions = request.get("allowed_actions")
        kinds = {str(item.get("kind") or "").lower() for item in actions if isinstance(item, Mapping)} if isinstance(actions, list) else set()
        return "vote" in phase or "投票" in phase or any("vote" in kind for kind in kinds)

    # 这些小包装保持旧的内部调用面，但都返回结构化/有界数据，不再做关键词归类。
    def _extract_public_notes(self, payload: Any) -> list[str]:
        return [self._render_records([record], 1) for record in self._extract_public_records(payload)]

    def _walk_public_payload(self, payload: Any, notes: list[str], depth: int) -> None:
        if depth > 5:
            return
        for record in self._extract_public_records(payload):
            rendered = self._render_records([record], 1)
            if rendered not in notes:
                notes.append(rendered)

    def _merge_public_notes(self, notes: list[str]) -> None:
        # 仅兼容旧调用；裸字符串没有公开事件来源，故只能作为普通声明。
        for note in notes[:_PUBLIC_SNAPSHOT_LIMIT]:
            if isinstance(note, str) and note.strip():
                self._merge_record({
                    "round": "", "event_type": "event", "speaker": "", "voter": "",
                    "target": "", "source_kind": "player_claim",
                    "text": self._trim_text(note, _PUBLIC_NOTE_CHAR_LIMIT),
                })

    def _collect_claim_block(self, notes: list[str]) -> str:
        return " | ".join(self._trim_text(note, 100) for note in notes[:3]) or "无"

    def _public_ledger_snapshot(self, turn_packet: Mapping[str, Any]) -> str:
        public_state = turn_packet.get("public_state")
        records = self._records_for_payload(public_state)
        alive = self._sort_player_ids(set(self._public_ledger["alive_ids"]) | self._extract_player_ids(public_state, include_collections=True))
        dead = self._sort_player_ids(set(self._public_ledger["dead_ids"]))
        return "\n".join((
            "【结构化公共台账】",
            "存活: " + self._join_ids(alive, limit=12),
            "已死: " + self._join_ids(dead, limit=12),
            "verified_public_facts: " + self._render_records([r for r in records if r["source_kind"] == "system_fact"], 5),
            "player_claims: " + self._render_records([r for r in records if r["source_kind"] == "player_claim"], 5),
        ))

    def _phase_summary(self, turn_packet: Mapping[str, Any], request: Mapping[str, Any]) -> str:
        return self._render_turn_summary(turn_packet, request)

    def _render_phase_summary(self, turn_packet: Mapping[str, Any], request: Mapping[str, Any]) -> str:
        return self._phase_summary(turn_packet, request)

    def _extract_dead_from_notes(self, notes: list[str]) -> set[str]:
        return {record["target"] for record in self._public_ledger.get("deaths", []) if record.get("target")} | {
            token for note in notes for token in self._ids_from_text(note)
        }

    def _ids_from_text(self, text: str) -> set[str]:
        return {token.strip("()[]{}<>：:=，。,.、") for token in re.split(r"[\\s,，;；|/\\\\=]+", str(text)) if self._looks_like_player_id(token.strip("()[]{}<>：:=，。,.、"))}

    def _dedupe_preserve_order(self, items: list[Any]) -> list[Any]:
        return list(dict.fromkeys(item for item in items if item))

    def _merge_unique(self, items: list[str], values: list[str]) -> None:
        for value in values:
            if value and value not in items:
                items.append(value)

    def _extract_player_ids(self, payload: Any, *, include_collections: bool = False) -> set[str]:
        ids: set[str] = set()
        def walk(obj: Any) -> None:
            if isinstance(obj, Mapping):
                lower = {str(key).lower(): value for key, value in obj.items()}
                for key, value in lower.items():
                    if key in self._player_id_keys() and self._is_scalar_id(value):
                        ids.add(str(value))
                    if include_collections and key in self._collection_id_keys():
                        ids.update(self._as_id_set(value))
                for value in obj.values():
                    walk(value)
            elif isinstance(obj, list):
                for item in obj[:25]:
                    walk(item)
        walk(payload)
        ids.discard(self.player_id)
        return ids

    def _collect_id_collection(self, mapping: Mapping[str, Any], allowed_keys: set[str]) -> set[str]:
        result: set[str] = set()
        lower = {str(key).lower(): value for key, value in mapping.items()}
        for key, value in lower.items():
            if key in allowed_keys:
                result.update(self._as_id_set(value))
        return result

    def _as_id_set(self, value: Any) -> set[str]:
        if isinstance(value, (list, tuple, set)):
            return {str(item) for item in value if self._is_scalar_id(item)}
        return {str(value)} if self._is_scalar_id(value) else set()

    def _append_bounded(self, items: list[Any], value: Any, limit: int) -> None:
        if value in items:
            return
        items.append(value)
        if len(items) > limit:
            del items[:len(items) - limit]

    def _first_scalar(self, mapping: Mapping[str, Any], keys: tuple[str, ...]) -> str:
        for key in keys:
            value = mapping.get(key)
            if self._is_scalar_text(value):
                return self._trim_text(str(value), _PUBLIC_NOTE_CHAR_LIMIT)
        return ""

    @staticmethod
    def _is_scalar_text(value: Any) -> bool:
        return isinstance(value, (str, int, float)) and not isinstance(value, bool)

    def _is_scalar_id(self, value: Any) -> bool:
        if isinstance(value, bool):
            return False
        if isinstance(value, int):
            return True
        return isinstance(value, str) and self._looks_like_player_id(value)

    def _trim_text(self, text: str, limit: int) -> str:
        normalized = " ".join(str(text).split())
        if len(normalized) <= limit:
            return normalized
        return normalized[:max(0, limit - 1)] + "…" if limit > 1 else normalized[:limit]

    def _join_ids(self, ids: list[str], *, limit: int) -> str:
        if not ids:
            return "未知"
        selected = ids[:limit]
        extra = len(ids) - len(selected)
        return "、".join(selected) + (f" 等{extra}人" if extra else "")

    def _sort_player_ids(self, ids: list[str]) -> list[str]:
        unique = list(dict.fromkeys(str(item) for item in ids if str(item)))
        def key(value: str) -> tuple[str, int, str]:
            match = _PLAYER_ID_SUFFIX.search(value)
            return (_PLAYER_ID_SUFFIX.sub("", value), int(match.group(1)) if match else 10**9, value)
        return sorted(unique, key=key)

    def _looks_like_player_id(self, value: str) -> bool:
        text = str(value).strip()
        return bool(text) and len(text) <= 16 and any(ch.isdigit() for ch in text)

    @staticmethod
    def _player_id_keys() -> set[str]:
        return {"player_id", "speaker_id", "voter_id", "claimer_id", "actor_id", "source_id", "target_id", "from_id", "to_id", "badge_holder", "badge_holder_id", "sheriff_id", "dead_id", "dead_player_id"}

    @staticmethod
    def _collection_id_keys() -> set[str]:
        return {"alive", "alive_ids", "alive_player_ids", "dead", "dead_ids", "dead_player_ids", "players", "player_ids", "target_ids", "candidates", "survivors", "living", "graveyard"}

    @staticmethod
    def _ALIVE_KEYS() -> set[str]:
        return {"alive", "alive_ids", "alive_player_ids", "survivors", "living", "live_ids", "current_alive"}

    @staticmethod
    def _DEAD_KEYS() -> set[str]:
        return {"dead", "dead_ids", "dead_player_ids", "eliminated", "casualties", "deaths", "graveyard"}

    def _extract_game_id(self, payload: Mapping[str, Any]) -> str | None:
        for key in ("game_id", "gameid", "match_id", "matchid", "session_id"):
            value = payload.get(key)
            if self._is_scalar_text(value):
                return str(value)
        game = payload.get("game")
        if isinstance(game, Mapping):
            for key in ("game_id", "gameid", "match_id", "matchid", "session_id"):
                value = game.get(key)
                if self._is_scalar_text(value):
                    return str(value)
        return None

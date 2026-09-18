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
_SIGNAL_TEXT = re.compile(r"(票|投|狼|好人|金水|查杀|对跳|冲票|带票|带节奏|怀疑|矛盾|身份|死亡|开枪|跳过|pass|vote|claim|wolf|kill|dead)", re.IGNORECASE)
_PLAYER_ID_TEXT = re.compile(r"\b[a-zA-Z]*\d+\b")

CURRENT_ROUND_DIALOGUE_TOOL_NAME = "read_current_round_dialogue"
DEFAULT_MAX_TOOL_CALLS_PER_DECISION = 5
DEFAULT_MAX_TOOL_RESULT_TOKENS = 1000
DEFAULT_MAX_PROMPT_CHARS = 12000
# 这是 Task-Agent 层的“纠错重试”次数：首次模型调用之外，最多再请求两次。
# 传输层的 HTTP/网络重试仍由 ModelClient 的 MODEL_MAX_RETRIES 单独控制。
DEFAULT_MAX_DECISION_RETRIES = 2
_SPEECH_ACTION_KINDS = frozenset({"speak", "last_words"})
_PASS_KIND_ALIASES = frozenset({"pass", "skip", "wait", "noop", "none", "hold"})
_PUBLIC_MEMORY_MAX_ROUNDS = 8
_PUBLIC_MEMORY_MAX_RECORDS_PER_ENTRY = 6
_PUBLIC_MEMORY_MAX_SCAN_DEPTH = 3

_MEMORY_KEYS = {
    "round",
    "phase",
    "day",
    "night",
    "speaker",
    "speaker_id",
    "player",
    "player_id",
    "voter",
    "voter_id",
    "target",
    "target_id",
    "kind",
    "action",
    "event",
    "result",
    "status",
    "role",
    "claim",
    "text",
    "message",
    "vote",
    "votes",
    "alive",
    "dead",
    "revealed",
    "public_role",
    "reason",
    "winner",
}


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
        "kind": _normalize_action_kind(str(source.get("kind", source.get("action", ""))), request),
    }
    target_id = source.get("target_id", source.get("targetId"))
    if target_id:
        action["target_id"] = str(target_id)
    if isinstance(source.get("text"), str):
        action["text"] = source["text"]
    return action


def _normalize_action_kind(kind: str, request: Mapping[str, Any]) -> str:
    normalized = str(kind or "").strip()
    if not normalized:
        return ""
    lower = normalized.lower()
    raw_actions = request.get("allowed_actions")
    if not isinstance(raw_actions, list):
        return normalized

    allowed_kinds: list[str] = []
    targetless_kinds: list[str] = []
    for allowed in raw_actions:
        if not isinstance(allowed, Mapping):
            continue
        allowed_kind = str(allowed.get("kind") or "").strip()
        if not allowed_kind:
            continue
        allowed_kinds.append(allowed_kind)
        if not allowed.get("target_ids") and allowed_kind.lower() not in _SPEECH_ACTION_KINDS:
            targetless_kinds.append(allowed_kind)

    if lower in _PASS_KIND_ALIASES:
        for allowed_kind in allowed_kinds:
            if allowed_kind.lower() in _PASS_KIND_ALIASES:
                return allowed_kind
        if targetless_kinds:
            return targetless_kinds[0]
        if len(allowed_kinds) == 1 and allowed_kinds[0].lower() in _PASS_KIND_ALIASES:
            return allowed_kinds[0]
    return normalized


def _text_mentions_token(text: Any, token: Any) -> bool:
    text_str = str(text or "")
    token_str = str(token or "").strip()
    if not text_str or not token_str:
        return False
    pattern = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(token_str)}(?![A-Za-z0-9_])", re.IGNORECASE)
    return bool(pattern.search(text_str))


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
        self._recent_public_events: deque[dict[str, Any]] = deque(maxlen=_PUBLIC_MEMORY_MAX_ROUNDS)

    async def observe(self, sync_packet: dict[str, Any]) -> None:
        """接收公开同步，并压缩保存为按轮次组织的短记忆。"""

        entry = self._build_public_memory_entry(sync_packet)
        if entry is None:
            return
        if self._recent_public_events:
            last = self._recent_public_events[-1]
            if self._public_memory_key(last) == self._public_memory_key(entry):
                self._merge_public_memory_entry(last, entry)
                return
        self._recent_public_events.append(entry)

    async def decide(self, turn_packet: dict[str, Any]) -> dict[str, Any]:
        if turn_packet["self"]["player_id"] != self.player_id:
            raise ValueError("行动包被发送给了错误玩家")
        private = turn_packet["private_information"]
        if str(private.get("role")) != self.profile.role:
            raise ValueError("Task-Agent 的角色档案与行动包身份不一致")

        tool_context = turn_packet.get("tool_context") or {}
        current_dialogue = tool_context.get("current_round_dialogue") or []
        # 当前轮对话是本玩家在该决策点依法可见的信息。先压缩入局内记忆，
        # 这样夜间的猎人反应仍能看到上一白天的争议，而不会把完整历史注入模型。
        self._observe_decision_context(turn_packet)
        decision_brief = self._build_decision_brief(turn_packet)
        system = self._system_prompt(private, decision_brief=decision_brief)
        prompt = {
            "game": turn_packet["game"],
            "public_rules": turn_packet["public_rules"],
            "self": turn_packet["self"],
            "private_information": private,
            "public_state": turn_packet["public_state"],
            "request": turn_packet["request"],
            "decision_brief": decision_brief,
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

    def _system_prompt(self, private: Mapping[str, Any], *, decision_brief: str = "") -> str:
        role_task = self.profile.task
        if decision_brief:
            role_task = f"{role_task}\n\n【本局决策简报】\n{decision_brief}"
        return render_prompt(
            "player_system.txt",
            player_id=self.player_id,
            role=private["role"],
            team=private["team"],
            persona=self.persona,
            role_base=self.profile.base,
            role_task=role_task,
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

    def _observe_decision_context(self, turn_packet: Mapping[str, Any]) -> None:
        """保存本次决策点已经可见的少量公开信息。

        这里只接收 turn_packet 内的 dialogue 和公开状态；不读取工具之外的历史，
        也不把原始对话无限累积。发言及其身份/查验说法另标为 claim，不能冒充事实。
        """
        if self.profile.role != "hunter":
            return
        game = turn_packet.get("game") if isinstance(turn_packet.get("game"), Mapping) else {}
        request = turn_packet.get("request") if isinstance(turn_packet.get("request"), Mapping) else {}
        public_state = turn_packet.get("public_state") if isinstance(turn_packet.get("public_state"), Mapping) else {}
        tool_context = turn_packet.get("tool_context") if isinstance(turn_packet.get("tool_context"), Mapping) else {}
        dialogue = tool_context.get("current_round_dialogue")
        dialogue_records = self._dialogue_to_public_records(
            dialogue if isinstance(dialogue, list) else [],
            self._first_compact(game, ("round", "day", "turn")),
            self._first_compact(request, ("phase",)) or self._first_compact(game, ("public_phase", "phase")),
        )
        state_records: list[dict[str, Any]] = []
        self._extract_public_records(
            {
                "round": self._first_compact(game, ("round", "day", "turn")),
                "phase": self._first_compact(request, ("phase",)) or self._first_compact(game, ("public_phase", "phase")),
                "public_state": public_state,
            },
            state_records,
            depth=0,
            seen=set(),
        )
        # 给状态事实留出位置；发言最多六条，但不让六条发言挤掉死亡/警徽/投票摘要。
        records = dialogue_records[-5:] + state_records[:1]
        if not records:
            return
        round_part = self._first_compact(game, ("round", "day", "turn"))
        phase_part = self._first_compact(request, ("phase",)) or self._first_compact(game, ("public_phase", "phase"))
        entry = {
            "round": round_part,
            "phase": phase_part,
            "records": records[:_PUBLIC_MEMORY_MAX_RECORDS_PER_ENTRY],
            "summary": self._format_public_memory_entry(round_part, phase_part, records, public_state),
        }
        if not entry["summary"]:
            return
        if self._recent_public_events and self._public_memory_key(self._recent_public_events[-1]) == (round_part, phase_part):
            self._merge_public_memory_entry(self._recent_public_events[-1], entry)
        else:
            self._recent_public_events.append(entry)

    def _dialogue_to_public_records(
        self, dialogue: list[Any], round_part: str, phase_part: str
    ) -> list[dict[str, Any]]:
        """把可见发言压成最多六条，保留原话但明确其只是玩家声称。"""
        records: list[dict[str, Any]] = []
        seen: set[tuple[str, ...]] = set()
        for item in dialogue:
            if isinstance(item, Mapping):
                speaker = self._first_compact(item, ("speaker", "speaker_id", "player", "player_id", "name"))
                text = self._first_compact(item, ("text", "message", "content", "reason"))
            elif isinstance(item, str):
                speaker, text = "", self._compact_text(item, 120)
            else:
                continue
            if not text:
                continue
            text = self._compact_text(text, 120)
            signature = (round_part, phase_part, speaker, text[:48])
            if signature in seen:
                continue
            seen.add(signature)
            record = {
                "speaker": speaker,
                "text": text,
                "source": "claim",
                "evidence_type": "player_statement",
                "round": round_part,
                "phase": phase_part,
            }
            records.append(record)
            for signal in self._extract_claim_signals(text, speaker):
                signal.update({"round": round_part, "phase": phase_part, "source": "claim"})
                signal_signature = tuple(str(signal.get(key, "")) for key in ("speaker", "target", "claim", "text"))
                if signal_signature not in seen:
                    seen.add(signal_signature)
                    records.append(signal)
        # 后来的发言更接近当前决策点；单条记录仍已截断到约 120 字。
        return records[-_PUBLIC_MEMORY_MAX_RECORDS_PER_ENTRY:]

    def _extract_claim_signals(self, text: str, speaker: str) -> list[dict[str, Any]]:
        """轻量提取身份、查验、归票和死亡说法；结果只供简报启发，不是事实。"""
        signals: list[dict[str, Any]] = []
        player = r"([A-Za-z]*\d+)"
        identity = r"(预言家|猎人|女巫|守卫|白痴|摄梦人|神职)"
        for role in re.findall(r"(?:我是|自称|跳|认)" + identity, text):
            signals.append({"speaker": speaker, "claim": role, "kind": "identity_claim", "text": self._compact_text(text, 120)})
        for label, pattern in (("查杀", r"(?:查杀|验出狼|验到狼)[：: ]*" + player), ("金水", r"金水[：: ]*" + player), ("验人", r"(?:验|查验|查了)[：: ]*" + player)):
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                signals.append({"speaker": speaker, "target": match.group(1), "claim": label, "kind": "investigation_claim", "text": self._compact_text(text, 120)})
        vote_match = re.search(r"(?:归票|投票|投|出|上)\s*" + player, text, re.IGNORECASE)
        if vote_match and not re.search(r"(?:别投|不要投|别出|不要出)\s*" + player, text, re.IGNORECASE):
            signals.append({"speaker": speaker, "target": vote_match.group(1), "claim": "vote_direction", "kind": "vote_claim", "text": self._compact_text(text, 120)})
        death_match = re.search(r"(?:平安夜|死了|死亡|被杀|出局).*?" + player, text, re.IGNORECASE)
        if death_match:
            signals.append({"speaker": speaker, "target": death_match.group(1), "claim": "death_claim", "kind": "death_claim", "text": self._compact_text(text, 120)})
        return signals[:3]

    def _build_decision_brief(self, turn_packet: Mapping[str, Any]) -> str:
        if self.profile.role != "hunter":
            return ""
        request = turn_packet.get("request") if isinstance(turn_packet.get("request"), Mapping) else {}
        public_state = turn_packet.get("public_state") if isinstance(turn_packet.get("public_state"), Mapping) else {}
        memory_entries = list(self._recent_public_events)
        ranking = self._rank_hunter_targets(request, public_state, memory_entries)
        phase = str(
            turn_packet.get("game", {}).get(
                "public_phase",
                turn_packet.get("game", {}).get("phase", request.get("phase", "unknown")),
            )
        )
        mode = self._infer_hunter_decision_mode(request, phase)
        pressure = self._rank_public_pressure(memory_entries, public_state)
        dialogue_preview = self._summarize_current_dialogue(turn_packet.get("tool_context"))

        lines = [f"模式：{mode}；阶段：{phase}"]
        recent_memory = ranking.get("recent_memory") or []
        if recent_memory:
            lines.append("最近公开记忆：" + " | ".join(recent_memory[-3:]))
        if pressure.get("conflicts"):
            lines.append("最近争议：" + "；".join(pressure["conflicts"][:2]))
        if pressure.get("vote_chain"):
            lines.append("票链：" + pressure["vote_chain"])
        if pressure.get("top_evidence"):
            lines.append("证据：" + "；".join(pressure["top_evidence"][:2]))
        layers = self._evidence_layer_summary(memory_entries)
        for label in ("引擎事实", "玩家声称", "推断"):
            if layers.get(label):
                lines.append(label + "：" + "；".join(layers[label][:2]))
        if dialogue_preview:
            lines.append("当前对话：" + dialogue_preview)

        ordered = ranking.get("target_ranking") or []
        if mode == "hunter_reaction":
            legal_targets = ranking.get("legal_targets") or []
            if legal_targets:
                lines.append("合法目标：" + "、".join(legal_targets[:6]))
            if ordered:
                top_bits = []
                for item in ordered[:3]:
                    reason = str(item.get("reason") or "")[:28]
                    top_bits.append(f"{item.get('target_id')}({item.get('score')}:{reason})")
                lines.append("目标排序：" + " > ".join(top_bits))
            if ranking.get("target_ranking"):
                lines.append("目标证据来源数：" + "；".join(
                    f"{item.get('target_id')}={item.get('evidence_count', 0)}类/{item.get('score', 0)}分"
                    for item in ranking["target_ranking"][:3]
                ))
            if ranking.get("pass_recommended"):
                lines.append("证据不足或只有单一来源时跳过；不要把玩家声称当引擎事实")
            elif ranking.get("best_target"):
                lines.append(f"若开枪优先：{ranking['best_target']}；只有无法确认多来源证据时才pass")
        elif mode == "day_vote":
            if pressure.get("top_names"):
                lines.append("重点怀疑：" + "、".join(pressure["top_names"]))
            if ordered:
                top_bits = []
                for item in ordered[:2]:
                    reason = str(item.get("reason") or "")[:24]
                    top_bits.append(f"{item.get('target_id')}({item.get('score')}:{reason})")
                lines.append("若需跟票优先看：" + " > ".join(top_bits))
            lines.append("发言时要点出当前票型与具体冲突，不要只说泛化站边。")
        else:
            if pressure.get("top_names"):
                lines.append("重点人名：" + "、".join(pressure["top_names"]))
            lines.append("发言要求：必须引用最近1-2条公开证据，点名1个具体玩家，并给出1个具体理由；优先说票型、对跳、金水/查杀冲突，不要空泛套话。")
        return self._truncate_text("\n".join(lines), 900)

    def _infer_hunter_decision_mode(self, request: Mapping[str, Any], phase: str) -> str:
        raw_actions = request.get("allowed_actions") if isinstance(request.get("allowed_actions"), list) else []
        kinds = [str(item.get("kind") or "").lower() for item in raw_actions if isinstance(item, Mapping)]
        if any(kind in _SPEECH_ACTION_KINDS for kind in kinds):
            return "speak"
        phase_text = f"{phase} {request.get('phase', '')}".lower()
        if any(token in phase_text for token in ("hunter", "last_words", "遗言", "反应", "开枪", "被击杀", "被刀", "death")):
            return "hunter_reaction"
        return "day_vote"

    def _summarize_current_dialogue(self, tool_context: Any) -> str:
        if not isinstance(tool_context, Mapping):
            return ""
        dialogue = tool_context.get("current_round_dialogue")
        if not isinstance(dialogue, list):
            return ""
        bits: list[str] = []
        for item in dialogue[-3:]:
            if isinstance(item, Mapping):
                speaker = self._first_compact(item, ("speaker", "speaker_id", "player", "player_id", "name"))
                text = self._first_compact(item, ("text", "message", "content", "reason"))
                if speaker and text:
                    bits.append(self._truncate_text(f"{speaker}:{text}", 72))
                elif text:
                    bits.append(self._truncate_text(text, 72))
                elif speaker:
                    bits.append(self._truncate_text(speaker, 24))
            elif isinstance(item, str):
                compact = self._truncate_text(self._compact_text(item, 72), 72)
                if compact:
                    bits.append(compact)
        return " | ".join(bits[:3])

    def _evidence_layer_summary(self, memory_entries: list[dict[str, Any]]) -> dict[str, list[str]]:
        layers = {"引擎事实": [], "玩家声称": [], "推断": []}
        for entry in memory_entries[-6:]:
            label = self._public_memory_label(entry)
            records = entry.get("records") if isinstance(entry.get("records"), list) else []
            for record in records[:_PUBLIC_MEMORY_MAX_RECORDS_PER_ENTRY]:
                text = self._public_record_text(record)
                if not text:
                    continue
                source = str(record.get("source") or "") if isinstance(record, Mapping) else ""
                bucket = "玩家声称" if source in {"claim", "dialogue"} else "引擎事实" if source == "fact" else "推断"
                if len(layers[bucket]) < 4:
                    layers[bucket].append(self._truncate_text(f"{label} {text}", 78))
        return layers

    def _rank_public_pressure(self, memory_entries: list[dict[str, Any]], public_state: Mapping[str, Any]) -> dict[str, Any]:
        candidate_scores: dict[str, dict[str, Any]] = {}
        conflicts: list[str] = []
        vote_chain: list[str] = []
        for entry in memory_entries[-6:]:
            entry_label = self._public_memory_label(entry)
            records = entry.get("records") if isinstance(entry.get("records"), list) else []
            for record in records[:_PUBLIC_MEMORY_MAX_RECORDS_PER_ENTRY]:
                record_text = self._public_record_text(record)
                lower = record_text.lower()
                speaker = self._first_compact(record, ("speaker", "voter", "player"))
                target = self._first_compact(record, ("target", "vote"))
                claim = self._first_compact(record, ("claim",))
                if speaker and target:
                    vote_chain.append(self._truncate_text(f"{speaker}->{target}", 28))
                if record_text and any(word in lower for word in ("查杀", "对跳", "假", "伪", "冲票", "带票", "带节奏", "矛盾", "反水", "狼", "wolf")):
                    conflicts.append(self._truncate_text(f"{entry_label} {record_text}", 86))
                for role, name in (("speaker", speaker), ("target", target), ("claim", claim)):
                    if not name:
                        continue
                    delta = 0
                    if role == "target":
                        if any(word in lower for word in ("查杀", "狼", "wolf", "对跳", "假", "伪", "冲票", "带票", "矛盾", "反水")):
                            delta += 3
                        elif any(word in lower for word in ("票", "vote", "投", "怀疑", "死亡", "身份")):
                            delta += 1
                        if any(word in lower for word in ("金水", "可信", "好人", "站边", "支持", "清白")):
                            delta -= 2
                    elif role == "speaker":
                        if any(word in lower for word in ("查杀", "对跳", "冲票", "带票", "矛盾", "狼", "wolf")):
                            delta += 1
                        if any(word in lower for word in ("金水", "可信", "好人", "站边")):
                            delta -= 1
                    elif role == "claim" and any(word in lower for word in ("对跳", "假", "伪", "查杀", "狼")):
                        delta += 1
                    if not delta:
                        continue
                    bucket = candidate_scores.setdefault(name, {"score": 0, "evidence": []})
                    bucket["score"] += delta
                    if len(bucket["evidence"]) < 2:
                        bucket["evidence"].append(self._truncate_text(f"{entry_label} {record_text}", 78))
        top = sorted(candidate_scores.items(), key=lambda item: (-int(item[1]["score"]), str(item[0])))
        return {
            "conflicts": self._dedupe_preserve_order(conflicts)[:2],
            "vote_chain": "；".join(self._dedupe_preserve_order(vote_chain)[:3]),
            "top_names": [name for name, bucket in top[:2] if int(bucket["score"]) > 0],
            "top_evidence": [f"{name}:{' / '.join(bucket['evidence'][:2])}" for name, bucket in top[:2] if int(bucket["score"]) > 0],
        }

    def _rank_hunter_targets(
        self,
        request: Mapping[str, Any],
        public_state: Mapping[str, Any],
        memory_entries: list[dict[str, Any]],
    ) -> dict[str, Any]:
        legal_targets = self._collect_target_ids(request)
        recent_memory = [
            self._truncate_text(str(entry.get("summary") or self._public_memory_label(entry)), 140)
            for entry in memory_entries[-4:]
        ]
        if not legal_targets:
            return {"legal_targets": [], "recent_memory": recent_memory, "target_ranking": [], "pass_recommended": True, "best_target": None}
        target_ranking: list[dict[str, Any]] = []
        for target_id in legal_targets:
            score, reasons, evidence = self._score_hunter_target_with_evidence(target_id, memory_entries, public_state)
            target_ranking.append({
                "target_id": target_id, "score": score, "reason": "；".join(reasons) if reasons else "证据不足",
                "evidence_count": len(evidence), "evidence_types": sorted(evidence),
            })
        target_ranking.sort(key=lambda item: (-int(item["score"]), str(item["target_id"])))
        best = target_ranking[0] if target_ranking else None
        second = target_ranking[1] if len(target_ranking) > 1 else None
        best_score = int(best["score"]) if best else 0
        second_score = int(second["score"]) if second else 0
        alive_count = self._alive_count(public_state)
        round_text = self._first_compact(public_state, ("round", "day"))
        if not round_text:
            round_text = self._first_compact(memory_entries[-1], ("round",)) if memory_entries else ""
        try:
            late_game = int(re.search(r"\d+", round_text).group()) >= 3 if re.search(r"\d+", round_text) else False
        except (AttributeError, ValueError):
            late_game = False
        pressure = alive_count is not None and alive_count <= 8 or late_game
        strong_multi_source = bool(best and best_score >= 5 and int(best.get("evidence_count", 0)) >= 2 and best_score - second_score >= 1)
        if pressure and strong_multi_source:
            pass_recommended = False
        elif best_score >= 7 and best and int(best.get("evidence_count", 0)) >= 2:
            pass_recommended = False
        else:
            pass_recommended = best is None or best_score < 4 or (best_score - second_score < 2 and best_score < 6)
        return {
            "legal_targets": legal_targets, "recent_memory": recent_memory, "target_ranking": target_ranking,
            "pass_recommended": pass_recommended, "best_target": best["target_id"] if best and not pass_recommended else None,
            "alive_count": alive_count, "late_game": late_game,
        }

    def _score_hunter_target(self, target_id: str, memory_entries: list[dict[str, Any]], public_state: Mapping[str, Any]) -> tuple[int, list[str]]:
        score, reasons, _evidence = self._score_hunter_target_with_evidence(target_id, memory_entries, public_state)
        return score, reasons

    def _score_hunter_target_with_evidence(self, target_id: str, memory_entries: list[dict[str, Any]], public_state: Mapping[str, Any]) -> tuple[int, list[str], set[str]]:
        target_token = self._compact_text(target_id, 32)
        if not target_token:
            return 0, ["证据不足"], set()
        target_lower = target_token.lower()
        score = 0
        reasons: list[str] = []
        evidence: set[str] = set()
        seen_rounds: set[str] = set()
        for entry in memory_entries[-8:]:
            entry_label = self._public_memory_label(entry)
            records = entry.get("records") if isinstance(entry.get("records"), list) else []
            matched_this_entry = False
            for record in records[:_PUBLIC_MEMORY_MAX_RECORDS_PER_ENTRY]:
                record_text = self._public_record_text(record)
                if not record_text:
                    continue
                lower = record_text.lower()
                speaker = self._first_compact(record, ("speaker", "voter", "player"))
                target = self._first_compact(record, ("target", "vote"))
                claim = self._first_compact(record, ("claim",))
                hit = _text_mentions_token(record_text, target_token) or any(
                    _text_mentions_token(candidate, target_token) or _text_mentions_token(target_token, candidate)
                    for candidate in (speaker, target, claim) if candidate
                )
                if not hit:
                    continue
                matched_this_entry = True
                delta = 0
                if target and _text_mentions_token(target, target_token):
                    if str(record.get("source") or "") == "fact" and any(word in lower for word in ("wolf", "狼人", "狼")):
                        delta += 5
                        evidence.add("引擎公开身份")
                    elif any(word in lower for word in ("查杀", "验出狼", "狼", "对跳", "假", "伪", "冲票", "带票", "矛盾", "反水", "带节奏")):
                        delta += 4
                        evidence.add("身份/查验冲突")
                    elif any(word in lower for word in ("票", "vote", "投", "归票", "怀疑", "出局")):
                        delta += 2
                        evidence.add("票型")
                    if any(word in lower for word in ("金水", "可信", "好人", "站边", "支持", "清白")):
                        delta -= 3
                        evidence.add("好人链保护")
                if speaker and _text_mentions_token(speaker, target_token) and any(word in lower for word in ("预言家", "猎人", "女巫", "守卫")):
                    if any(word in lower for word in ("对跳", "假", "伪", "冲突", "放弃", "不认")):
                        delta += 3
                        evidence.add("身份冲突")
                    else:
                        delta += 1
                        evidence.add("身份声称")
                if claim and _text_mentions_token(target, target_token) and claim in ("查杀", "death_claim"):
                    delta += 2 if claim == "查杀" else 1
                    evidence.add("玩家查验/死亡声称")
                if target_token in lower and any(word in lower for word in ("平安夜", "死了", "死亡", "被杀")):
                    delta += 2
                    evidence.add("死亡事实争议")
                if speaker and _text_mentions_token(speaker, target_token) and str(record.get("source") or "") == "fact" and any(word in lower for word in ("好人", "村民", "神职")):
                    delta -= 3
                    evidence.add("引擎好人身份")
                if not delta:
                    continue
                score += delta
                if len(reasons) < 3:
                    reasons.append(self._truncate_text(f"{entry_label} {record_text}", 78))
            if matched_this_entry and entry_label not in seen_rounds:
                seen_rounds.add(entry_label)
                score += 1
        score = max(score, 0)
        if not reasons:
            reasons.append("公开记忆里缺少强关联")
        return score, reasons[:3], evidence

    def _alive_count(self, public_state: Mapping[str, Any]) -> int | None:
        players = public_state.get("players")
        if not isinstance(players, (list, tuple, Mapping)):
            players = public_state.get("alive_players")
        if isinstance(players, Mapping):
            players = list(players.values())
        if not isinstance(players, (list, tuple)):
            return None
        count = 0
        found = False
        for player in players:
            if not isinstance(player, Mapping):
                continue
            found = True
            alive = player.get("alive")
            if alive is True or str(player.get("status", "")).lower() in {"alive", "存活"}:
                count += 1
        return count if found else None

    def _build_public_memory_entry(self, value: Any) -> dict[str, Any] | None:
        if not isinstance(value, Mapping):
            return None
        records: list[dict[str, Any]] = []
        self._extract_public_records(value, records, depth=0, seen=set())
        round_part, phase_part = self._public_memory_key(value)
        summary = self._format_public_memory_entry(round_part, phase_part, records, value)
        if not summary:
            return None
        return {
            "round": round_part,
            "phase": phase_part,
            "records": records[:_PUBLIC_MEMORY_MAX_RECORDS_PER_ENTRY],
            "summary": summary,
        }

    def _merge_public_memory_entry(self, base: dict[str, Any], incoming: dict[str, Any]) -> None:
        base_records = base.setdefault("records", [])
        incoming_records = incoming.get("records") if isinstance(incoming.get("records"), list) else []
        base_records.extend(incoming_records)
        deduped: list[dict[str, Any]] = []
        seen: set[tuple[str, ...]] = set()
        for record in base_records:
            if not isinstance(record, Mapping):
                continue
            signature = tuple(
                self._first_compact(record, (key,)) for key in ("speaker", "voter", "player", "target", "vote", "claim", "death", "kind", "text", "sheriff")
            )
            if signature in seen:
                continue
            seen.add(signature)
            deduped.append(dict(record))
        base["records"] = deduped[:_PUBLIC_MEMORY_MAX_RECORDS_PER_ENTRY]
        base["summary"] = self._format_public_memory_entry(
            base.get("round", ""),
            base.get("phase", ""),
            base["records"],
            base,
        )

    def _public_memory_key(self, value: Mapping[str, Any]) -> tuple[str, str]:
        round_part = self._first_compact(value, ("round", "day", "night", "turn"))
        phase_part = self._first_compact(value, ("phase", "public_phase"))
        return round_part, phase_part

    def _public_memory_label(self, entry: Mapping[str, Any]) -> str:
        if not isinstance(entry, Mapping):
            return self._truncate_text(str(entry), 80)
        round_part = self._compact_text(str(entry.get("round") or ""), 12)
        phase_part = self._compact_text(str(entry.get("phase") or ""), 16)
        bits: list[str] = []
        if round_part:
            bits.append(f"R{round_part}" if round_part.isdigit() else round_part)
        if phase_part:
            bits.append(phase_part)
        if bits:
            return " ".join(bits)
        summary = self._compact_text(str(entry.get("summary") or ""), 80)
        return summary or "sync"

    def _format_public_memory_entry(
        self,
        round_part: str,
        phase_part: str,
        records: list[dict[str, Any]],
        value: Any,
    ) -> str:
        header_bits: list[str] = []
        if round_part:
            header_bits.append(f"R{round_part}" if round_part.isdigit() else round_part)
        if phase_part:
            header_bits.append(phase_part)
        header = " ".join(header_bits) or "sync"
        segments: list[str] = []
        for record in records[:_PUBLIC_MEMORY_MAX_RECORDS_PER_ENTRY]:
            text = self._public_record_text(record)
            if text:
                segments.append(text)
        if not segments:
            fallback = self._public_record_text(value)
            if fallback:
                segments.append(fallback)
        if not segments:
            return ""
        return self._truncate_text(f"{header}: " + " | ".join(segments), 220)

    def _extract_public_records(self, value: Any, records: list[dict[str, Any]], *, depth: int, seen: set[tuple[str, ...]]) -> None:
        if len(records) >= 12 or depth > _PUBLIC_MEMORY_MAX_SCAN_DEPTH:
            return
        if isinstance(value, Mapping):
            record = self._mapping_to_public_record(value)
            if record:
                signature = tuple(
                    record.get(key, "") for key in ("speaker", "voter", "player", "target", "vote", "claim", "death", "kind", "text")
                )
                if signature not in seen:
                    seen.add(signature)
                    records.append(record)
            for item in value.values():
                if len(records) >= 12:
                    return
                if isinstance(item, (Mapping, list, tuple)):
                    self._extract_public_records(item, records, depth=depth + 1, seen=seen)
            return
        if isinstance(value, (list, tuple)):
            for item in value[:6]:
                self._extract_public_records(item, records, depth=depth + 1, seen=seen)
                if len(records) >= 12:
                    return
            return
        if isinstance(value, str):
            text = self._compact_text(value, 160)
            if text and self._is_signal_text(text):
                signature = ("text", text)
                if signature not in seen:
                    seen.add(signature)
                    records.append({"text": text})

    def _mapping_to_public_record(self, value: Mapping[str, Any]) -> dict[str, Any] | None:
        record: dict[str, Any] = {}
        field_map = {
            "speaker": ("speaker", "speaker_id", "player", "player_id", "name"),
            "voter": ("voter", "voter_id"),
            "target": ("target", "target_id", "vote_target", "accused"),
            "vote": ("vote", "votes", "voted_for", "vote_for"),
            "claim": ("claim", "public_role", "revealed_role", "identity"),
            "death": ("death", "dead", "result"),
            "kind": ("kind", "action", "event"),
            "text": ("text", "message", "content", "reason"),
            "sheriff": ("sheriff", "badge", "sheriff_id"),
        }
        for field, keys in field_map.items():
            compact = self._first_compact(value, keys)
            if compact:
                record[field] = compact
        # 只含 player/name/alive/status 的玩家状态节点不是证据。status 只有在
        # 明确表示死亡时才保留；普通 alive 列表不应给目标制造分数。
        meaningful = {"text", "voter", "target", "vote", "claim", "death", "kind", "sheriff"}
        if not any(key in record for key in meaningful):
            status = str(value.get("status") or "").lower()
            if status not in {"dead", "killed", "deceased", "死亡", "出局", "被杀"}:
                return None
            record["death"] = self._compact_text(status, 32)
        record["source"] = "claim" if "text" in record and record.get("speaker") else "fact" if any(key in record for key in ("death", "claim", "vote", "sheriff")) else "inference"
        return record

    def _public_record_text(self, value: Any) -> str:
        if isinstance(value, Mapping):
            parts: list[str] = []
            for key in ("speaker", "voter", "player", "target", "vote", "claim", "death", "kind", "text", "sheriff", "source"):
                compact = self._first_compact(value, (key,))
                if compact:
                    if key == "text" and parts:
                        parts.append(f"text={compact}")
                    else:
                        parts.append(f"{key}={compact}")
            return self._compact_text(" ".join(parts), 160)
        if isinstance(value, str):
            return self._compact_text(value, 160)
        return self._compact_text(self._compact_value(value), 160)

    @staticmethod
    def _dedupe_preserve_order(items: list[str]) -> list[str]:
        deduped: list[str] = []
        seen: set[str] = set()
        for item in items:
            if item and item not in seen:
                seen.add(item)
                deduped.append(item)
        return deduped

    def _first_compact(self, value: Mapping[str, Any], keys: tuple[str, ...]) -> str:
        for key in keys:
            if key not in value:
                continue
            compact = self._compact_value(value.get(key))
            if compact:
                return compact
        return ""

    def _collect_target_ids(self, request: Mapping[str, Any]) -> list[str]:
        target_ids: list[str] = []
        seen: set[str] = set()
        raw_actions = request.get("allowed_actions")
        if not isinstance(raw_actions, list):
            return target_ids
        for allowed in raw_actions:
            if not isinstance(allowed, Mapping):
                continue
            raw_target_ids = allowed.get("target_ids")
            if not isinstance(raw_target_ids, list):
                continue
            for target_id in raw_target_ids:
                if target_id is None:
                    continue
                text = str(target_id)
                if text not in seen:
                    seen.add(text)
                    target_ids.append(text)
        return target_ids

    def _extract_memory_lines(self, value: Any, *, depth: int = 0) -> list[str]:
        lines: list[str] = []
        self._extract_memory_lines_into(value, lines, depth=depth)
        return lines[:10]

    def _extract_memory_lines_into(self, value: Any, lines: list[str], *, depth: int) -> None:
        if len(lines) >= 10 or depth > 2:
            return
        if isinstance(value, Mapping):
            parts: list[str] = []
            for key, item in value.items():
                key_text = str(key)
                if key_text in _MEMORY_KEYS or key_text.endswith("_id"):
                    compact = self._compact_value(item)
                    if compact:
                        parts.append(f"{key_text}={compact}")
                if isinstance(item, (Mapping, list, tuple)):
                    self._extract_memory_lines_into(item, lines, depth=depth + 1)
                elif isinstance(item, str) and self._is_signal_text(item):
                    compact = self._compact_text(item)
                    if compact:
                        parts.append(compact)
            if parts:
                line = self._compact_text(" ".join(parts))
                if line:
                    lines.append(line)
            return
        if isinstance(value, (list, tuple)):
            for item in value[:5]:
                self._extract_memory_lines_into(item, lines, depth=depth + 1)
                if len(lines) >= 10:
                    return
            return
        if isinstance(value, str):
            text = self._compact_text(value)
            if text and self._is_signal_text(text):
                lines.append(text)

    def _is_signal_text(self, text: str) -> bool:
        # 状态同步里的孤立字符串不能仅因含中文就成为证据；发言由
        # _dialogue_to_public_records 单独接收并保留。
        return bool(_SIGNAL_TEXT.search(text) or _PLAYER_ID_TEXT.search(text))

    @staticmethod
    def _compact_text(text: str, limit: int = 160) -> str:
        return re.sub(r"\s+", " ", str(text)).strip()[:limit]

    def _compact_value(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, str):
            return self._compact_text(value, 80)
        try:
            dumped = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        except TypeError:
            dumped = repr(value)
        return self._compact_text(dumped, 120)

    @staticmethod
    def _truncate_text(text: str, limit: int) -> str:
        return str(text)[: max(0, int(limit))]

    @staticmethod
    def _nonnegative_int(value: object, *, fallback: int | None) -> int | None:
        if isinstance(value, bool):
            return fallback
        try:
            normalized = int(value)
        except (TypeError, ValueError):
            return fallback
        return normalized if normalized >= 0 else fallback

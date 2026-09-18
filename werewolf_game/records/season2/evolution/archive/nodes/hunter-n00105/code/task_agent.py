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
_PUBLIC_MEMORY_MAX_ROUNDS = 10
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

        # 猎人需要先把本决策点已经依法可见的发言压缩进短记忆；模型仍可按需
        # 使用受限工具读取原文，但不会因此把完整历史自动拼进 prompt。
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
        """把当前包中已经可见的本轮发言压成 claim 记录。

        这是记忆同步而不是额外取数：不调用工具、不读取外部状态，也不把原文
        无限累积。身份和验人信息始终带有 claim 标记，不能升级为系统事实。
        """
        if self.profile.role != "hunter":
            return
        tool_context = turn_packet.get("tool_context")
        dialogue = tool_context.get("current_round_dialogue") if isinstance(tool_context, Mapping) else None
        records = self._dialogue_to_public_records(dialogue)
        if not records:
            return
        game = turn_packet.get("game") if isinstance(turn_packet.get("game"), Mapping) else {}
        request = turn_packet.get("request") if isinstance(turn_packet.get("request"), Mapping) else {}
        round_part = self._first_compact(game, ("round", "day", "turn")) or self._first_compact(request, ("round", "day", "turn"))
        phase_part = self._first_compact(game, ("public_phase", "phase")) or self._first_compact(request, ("phase",))
        entry = {
            "round": round_part,
            "phase": phase_part,
            "records": records[-_PUBLIC_MEMORY_MAX_RECORDS_PER_ENTRY:],
        }
        entry["summary"] = self._format_public_memory_entry(round_part, phase_part, entry["records"], entry)
        if not entry["summary"]:
            return
        if self._recent_public_events and self._public_memory_key(self._recent_public_events[-1]) == self._public_memory_key(entry):
            self._merge_public_memory_entry(self._recent_public_events[-1], entry)
        else:
            self._recent_public_events.append(entry)

    def _dialogue_to_public_records(self, dialogue: Any) -> list[dict[str, Any]]:
        if not isinstance(dialogue, list):
            return []
        records: list[dict[str, Any]] = []
        seen: set[tuple[str, ...]] = set()
        for item in dialogue:
            if isinstance(item, Mapping):
                speaker = self._first_compact(item, ("speaker", "speaker_id", "player", "player_id", "name"))
                text = self._first_compact(item, ("text", "message", "content", "reason"))
            elif isinstance(item, str):
                speaker, text = "", self._compact_text(item, 160)
            else:
                continue
            if not text:
                continue
            record: dict[str, Any] = {"text": text, "source": "claim", "evidence_type": "player_statement"}
            if speaker:
                record["speaker"] = speaker
            signals = self._extract_claim_signals(text, speaker=speaker)
            record.update(signals)
            signature = tuple(str(record.get(key, "")) for key in ("speaker", "text", "target", "vote_direction", "accusation", "defense", "investigation_claim"))
            if signature not in seen:
                seen.add(signature)
                records.append(record)
        return records[-12:]

    def _extract_claim_signals(self, text: str, *, speaker: str = "") -> dict[str, str]:
        """提取启发式关系；结果是玩家声称，不是身份或结算事实。"""
        compact = self._compact_text(text, 160)
        tokens = re.findall(r"(?<![A-Za-z0-9_])[a-zA-Z]*\\d+[a-zA-Z]*", compact)
        others = [token for token in tokens if token.lower() != str(speaker).lower()]
        # 发言常含“我和队友都压 p9”；对指控/投票取末个玩家编号，
        # 查验声明则单独保留前几个编号，避免把发言者误当目标。
        target = others[-1] if others else (tokens[-1] if tokens else "")
        lower = compact.lower()
        negated = bool(re.search(r"(?:别|不要|不|勿|别去|不要去)\\s*(?:投|出|票|冲|点)?\\s*" + re.escape(target), compact, re.IGNORECASE)) if target else False
        signals: dict[str, str] = {}
        identity_words = re.findall(r"(?:预言家|女巫|猎人|守卫|神职|平民|好人|狼人)", compact, re.IGNORECASE)
        if identity_words:
            signals["identity_claim"] = self._compact_text("声称" + "/".join(self._dedupe_preserve_order(identity_words)), 48)
        if others and re.search(r"金水|查杀|验人|验了|验过|查验|验证", compact, re.IGNORECASE):
            investigation = "金水" if "金水" in compact else ("查杀" if "查杀" in compact else "验人")
            signals["investigation_claim"] = ",".join(f"{item}:{investigation}" for item in others[:4])
        if target and re.search(r"投|出|票|归票|冲票", compact, re.IGNORECASE) and not negated:
            signals["vote_direction"] = target
        if target and re.search(r"狼|怀疑|可疑|假|冲票|带票|带节奏|对跳|不信", compact, re.IGNORECASE) and not negated:
            signals["accusation"] = target
        if target and (negated or re.search(r"金水|好人|可信|保|别动|别投|不要出|站边", compact, re.IGNORECASE)):
            signals["defense"] = target
        return signals

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
            lines.append("重点怀疑：" + "；".join(pressure["top_evidence"][:2]))
        if pressure.get("protected"):
            lines.append("可信声明(claim)：" + ",".join(pressure["protected"][:4]) + "（仅玩家声称）")
        if pressure.get("defended"):
            lines.append("被保护目标：" + ",".join(pressure["defended"][:4]) + "（仍需复核）")
        if pressure.get("collaborations"): 
            lines.append("协同带票对：" + "；".join(pressure["collaborations"][:2]))
        if dialogue_preview:
            lines.append("当前对话末段：" + dialogue_preview)

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
            if ranking.get("pass_recommended"):
                lines.append("建议：证据不足或误伤链风险高，可跳过")
            elif ranking.get("best_target"):
                lines.append(f"建议开枪：{ranking['best_target']}（模型仍须复核）")
        elif mode == "day_vote":
            if pressure.get("top_names"):
                lines.append("生前重点审计：" + "、".join(pressure["top_names"]))
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

    def _rank_public_pressure(self, memory_entries: list[dict[str, Any]], public_state: Mapping[str, Any]) -> dict[str, Any]:
        """按关系而非被点名次数排序：攻击者和协同票型是主要信号。"""
        scores: dict[str, dict[str, Any]] = {}
        conflicts: list[str] = []
        vote_chain: list[str] = []
        accusations: dict[str, set[str]] = {}
        round_targets: dict[tuple[str, str], set[str]] = {}
        protected: set[str] = set()
        defended: set[str] = set()
        seer_claimers: set[str] = set()
        entries = memory_entries[-10:]
        for entry in entries:
            label = self._public_memory_label(entry)
            records = entry.get("records") if isinstance(entry.get("records"), list) else []
            for record in records[:_PUBLIC_MEMORY_MAX_RECORDS_PER_ENTRY]:
                speaker = self._first_compact(record, ("speaker", "voter", "player"))
                target = self._first_compact(record, ("target", "vote", "vote_direction", "accusation", "defense"))
                accusation = self._first_compact(record, ("accusation",))
                defense = self._first_compact(record, ("defense",))
                investigation = self._first_compact(record, ("investigation_claim",))
                text = self._public_record_text(record)
                if speaker and target and target != speaker:
                    vote_chain.append(self._truncate_text(f"{speaker}->{target}", 28))
                if speaker and accusation and accusation != speaker:
                    accusations.setdefault(accusation, set()).add(speaker)
                    round_targets.setdefault((label, accusation), set()).add(speaker)
                    bucket = scores.setdefault(speaker, {"score": 0, "evidence": []})
                    bucket["score"] += 1
                    if len(bucket["evidence"]) < 2:
                        bucket["evidence"].append(f"{label}指向{accusation}")
                if speaker and defense and defense != speaker:
                    defended.add(defense)
                    # 单次保护不定性；多轮同向软保由协同检查加分。
                    round_targets.setdefault((label + ":def", defense), set()).add(speaker)
                if investigation:
                    for claim in investigation.split(","):
                        protected_id, _, claim_kind = claim.partition(":")
                        if claim_kind == "金水" and protected_id:
                            protected.add(protected_id)
                    if "预言家" in text or "查验" in text:
                        seer_claimers.add(speaker)
                if text and any(word in text for word in ("对跳", "冲票", "带票", "带节奏", "矛盾", "反水")):
                    conflicts.append(self._truncate_text(f"{label} {text}", 86))
        # 只有独立指控者提供轻微压力；不能让“狼指谁谁像狼”主导排序。
        for target, speakers in accusations.items():
            if target in protected:
                for speaker in speakers:
                    scores.setdefault(speaker, {"score": 0, "evidence": []})["score"] += 2
            elif len(speakers) >= 2:
                for speaker in speakers:
                    scores.setdefault(speaker, {"score": 0, "evidence": []})["score"] += 1
        # 同一轮两人压一个目标，且在晚局或跨轮重复，才视为协同带票。
        pair_targets: dict[tuple[str, str], set[str]] = {}
        for (label, target), speakers in round_targets.items():
            if label.endswith(":def") or len(speakers) < 2:
                continue
            ordered = sorted(speakers)
            for index, left in enumerate(ordered):
                for right in ordered[index + 1:]:
                    pair_targets.setdefault((left, right), set()).add(target)
        collaborations: list[str] = []
        for (left, right), targets in pair_targets.items():
            if not targets:
                continue
            for target in targets:
                labels = sum(1 for (label, item), speakers in round_targets.items() if item == target and not label.endswith(":def") and left in speakers and right in speakers)
                if labels >= 2 or (self._looks_late(public_state, entries) and target in protected):
                    for name in (left, right):
                        scores.setdefault(name, {"score": 0, "evidence": []})["score"] += 2
                    collaborations.append(f"{left},{right}同压{target}")
        top = sorted(scores.items(), key=lambda item: (-int(item[1]["score"]), str(item[0])))
        return {
            "conflicts": self._dedupe_preserve_order(conflicts)[:2],
            "vote_chain": "；".join(self._dedupe_preserve_order(vote_chain)[:4]),
            "top_names": [name for name, bucket in top[:3] if int(bucket["score"]) > 0],
            "top_evidence": [f"{name}:{' / '.join(bucket['evidence'][:2])}" for name, bucket in top[:3] if int(bucket["score"]) > 0],
            "protected": sorted(protected)[:4],
            "defended": sorted(defended - protected)[:4],
            "collaborations": self._dedupe_preserve_order(collaborations)[:3],
            "seer_claimers": sorted(seer_claimers)[:3],
        }

    def _looks_late(self, public_state: Mapping[str, Any], entries: list[dict[str, Any]]) -> bool:
        alive = public_state.get("alive_players", public_state.get("alive"))
        if isinstance(alive, list) and len(alive) <= 5:
            return True
        for entry in reversed(entries[-2:]):
            number = self._first_compact(entry, ("round", "day"))
            try:
                if int(number) >= 5:
                    return True
            except (TypeError, ValueError):
                pass
        return False

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
            return {
                "legal_targets": [],
                "recent_memory": recent_memory,
                "target_ranking": [],
                "pass_recommended": True,
                "best_target": None,
            }

        target_ranking: list[dict[str, Any]] = []
        for target_id in legal_targets:
            score, reasons = self._score_hunter_target(target_id, memory_entries, public_state)
            target_ranking.append(
                {
                    "target_id": target_id,
                    "score": score,
                    "reason": "；".join(reasons) if reasons else "证据不足",
                }
            )
        target_ranking.sort(key=lambda item: (-int(item["score"]), str(item["target_id"])))
        best = target_ranking[0] if target_ranking else None
        second = target_ranking[1] if len(target_ranking) > 1 else None
        best_score = int(best["score"]) if best else 0
        second_score = int(second["score"]) if second else 0
        pressure = self._rank_public_pressure(memory_entries, public_state)
        urgent_signal = bool(pressure.get("collaborations"))
        few_targets = len(legal_targets) <= 2
        late = self._looks_late(public_state, memory_entries)
        pass_recommended = best is None or best_score < 4 or (best_score - second_score < 2 and best_score < 6)
        # 残局或出现协同压可信声明时，不因分差小就把最后窗口让掉；仍由模型
        # 在合法目标中作最终选择，不在本地强制开枪。
        if best is not None and (late or urgent_signal or few_targets) and best_score >= 2:
            pass_recommended = False
        return {
            "legal_targets": legal_targets,
            "recent_memory": recent_memory,
            "target_ranking": target_ranking,
            "pass_recommended": pass_recommended,
            "best_target": best["target_id"] if best and not pass_recommended else None,
        }

    def _score_hunter_target(self, target_id: str, memory_entries: list[dict[str, Any]], public_state: Mapping[str, Any]) -> tuple[int, list[str]]:
        target = self._compact_text(target_id, 32)
        if not target:
            return 0, ["证据不足"]
        score = 0
        reasons: list[str] = []
        protected: set[str] = set()
        directed_by: set[str] = set()
        attack_rounds: set[str] = set()
        for entry in memory_entries[-10:]:
            label = self._public_memory_label(entry)
            records = entry.get("records") if isinstance(entry.get("records"), list) else []
            for record in records[:_PUBLIC_MEMORY_MAX_RECORDS_PER_ENTRY]:
                speaker = self._first_compact(record, ("speaker", "voter", "player"))
                accusation = self._first_compact(record, ("accusation",))
                vote = self._first_compact(record, ("vote_direction", "vote"))
                investigation = self._first_compact(record, ("investigation_claim",))
                defense = self._first_compact(record, ("defense",))
                if investigation:
                    for claim in investigation.split(","):
                        claimed_target, _, claim_kind = claim.partition(":")
                        if claim_kind == "金水" and claimed_target:
                            protected.add(claimed_target)
                if speaker and accusation == target and speaker != target:
                    directed_by.add(speaker)
                    attack_rounds.add(label)
                if speaker and vote == target and speaker != target:
                    directed_by.add(speaker)
                    attack_rounds.add(label)
                if speaker and defense == target and speaker != target:
                    # 保护本身不是目标狼分，留给关系评分和模型复核。
                    pass
        if target in protected:
            score -= 3
            reasons.append("被预言家声称金水(claim)，有限降权")
        # 被攻击只给很轻的压力；攻击者是否在攻击可信链才是重点。
        score += min(2, len(directed_by))
        if directed_by:
            reasons.append(f"被{len(directed_by)}名玩家指向，需复核来源")
        if len(attack_rounds) >= 2:
            score += 1
            reasons.append("跨轮被持续施压")
        # 目标若自己反复指向金水/公开链，目标本身更可疑。
        for entry in memory_entries[-10:]:
            label = self._public_memory_label(entry)
            records = entry.get("records") if isinstance(entry.get("records"), list) else []
            for record in records[:_PUBLIC_MEMORY_MAX_RECORDS_PER_ENTRY]:
                speaker = self._first_compact(record, ("speaker", "voter", "player"))
                accusation = self._first_compact(record, ("accusation",))
                vote = self._first_compact(record, ("vote_direction", "vote"))
                if speaker != target:
                    continue
                aimed = accusation or vote
                if aimed and aimed in protected:
                    score += 2
                    if len(reasons) < 3:
                        reasons.append(f"{label}攻击金水声明{aimed}")
        # 协同关系的狼分归给带票者，而不是被集中攻击的对象。
        pressure = self._rank_public_pressure(memory_entries, public_state)
        for collaboration in pressure.get("collaborations") or []:
            pair = collaboration.split("同压", 1)[0]
            if target in {name.strip() for name in pair.split(",")}:
                score += 2
                if len(reasons) < 3:
                    reasons.append("晚局协同带票者")
                break
        if self._looks_late(public_state, memory_entries) and len(directed_by) >= 2:
            score += 1
            if len(reasons) < 3:
                reasons.append("晚局多人同向带票，优先审计攻击者")
        if score <= 0 and not reasons:
            reasons.append("未见目标主动造成的强关联")
        return max(score, 0), reasons[:3]

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
                self._first_compact(record, (key,)) for key in ("speaker", "voter", "player", "target", "vote", "vote_direction", "claim", "identity_claim", "investigation_claim", "accusation", "defense", "death", "kind", "source", "text")
            )
            if signature in seen:
                continue
            seen.add(signature)
            deduped.append(dict(record))
        # 当前轮新来的发言比旧同步更有决策价值，保留该轮最后几条结构化记录。
        base["records"] = deduped[-_PUBLIC_MEMORY_MAX_RECORDS_PER_ENTRY:]
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
                    record.get(key, "") for key in ("speaker", "voter", "player", "target", "vote", "vote_direction", "claim", "identity_claim", "investigation_claim", "accusation", "defense", "death", "kind", "source", "text")
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
            "claim": ("claim", "public_role", "role", "identity"),
            "death": ("death", "dead", "result", "status"),
            "kind": ("kind", "action", "event"),
            "text": ("text", "message", "reason"),
        }
        for field, keys in field_map.items():
            compact = self._first_compact(value, keys)
            if compact:
                record[field] = compact
        if not record:
            return None
        return record

    def _public_record_text(self, value: Any) -> str:
        if isinstance(value, Mapping):
            parts: list[str] = []
            for key in ("speaker", "voter", "player", "target", "vote", "vote_direction", "claim", "identity_claim", "investigation_claim", "accusation", "defense", "death", "kind", "source", "text"):
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
        return bool(_SIGNAL_TEXT.search(text) or _PLAYER_ID_TEXT.search(text) or _CHINESE_CHARACTER.search(text))

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

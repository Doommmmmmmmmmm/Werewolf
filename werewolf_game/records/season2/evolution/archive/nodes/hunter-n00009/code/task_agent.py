"""最小可运行的角色 Task-Agent。

每次行动从一个新的模型会话开始。Agent 只维护受限的局内紧凑记忆，当前轮对话只能
通过唯一的受限工具按需读取。这里不包含策略进化、外部检索或其他 Harness。
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
        self._game_memory: dict[str, Any] = {
            "game_id": None,
            "last_round_key": None,
            "recent_rounds": [],
            "clue_bank": {
                "confirmed": [],
                "claims": [],
                "votes": [],
                "deaths": [],
                "conflicts": [],
            },
            "current_controversy": "",
            "latest_public_state": "",
        }

    async def observe(self, sync_packet: dict[str, Any]) -> None:
        """接收公开同步并维护受限的局内记忆。"""

        self._update_game_memory(sync_packet)

    async def decide(self, turn_packet: dict[str, Any]) -> dict[str, Any]:
        if turn_packet["self"]["player_id"] != self.player_id:
            raise ValueError("行动包被发送给了错误玩家")
        private = turn_packet["private_information"]
        if str(private.get("role")) != self.profile.role:
            raise ValueError("Task-Agent 的角色档案与行动包身份不一致")

        self._update_game_memory(turn_packet)
        tool_context = turn_packet.get("tool_context") or {}
        current_dialogue = tool_context.get("current_round_dialogue") or []
        memory_summary = self._build_memory_summary(turn_packet, current_dialogue)
        speech_hint = self._build_action_hint(turn_packet["request"], memory_summary, current_dialogue)

        system = self._system_prompt(private, memory_summary)
        prompt = {
            "game": self._compact_value(turn_packet["game"], max_depth=2, max_items=8, max_chars=220),
            "public_rules": self._compact_value(turn_packet["public_rules"], max_depth=2, max_items=8, max_chars=320),
            "self": self._compact_value(turn_packet["self"], max_depth=2, max_items=8, max_chars=220),
            "private_information": self._compact_value(private, max_depth=2, max_items=8, max_chars=220),
            "public_state": self._compact_value(turn_packet["public_state"], max_depth=2, max_items=10, max_chars=520),
            "request": self._compact_value(turn_packet["request"], max_depth=2, max_items=8, max_chars=320),
            "history_policy": {
                "default_context": "current_state_with_compact_memory",
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
                "instruction": self._turn_instruction(turn_packet["request"], feedback, memory_summary, speech_hint),
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

    def _system_prompt(self, private: Mapping[str, Any], memory_summary: str = "") -> str:
        role_task = self.profile.task
        if memory_summary:
            role_task = f"{role_task}\n\n【局内紧凑记忆】\n{memory_summary}"
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

    def _turn_instruction(
        self,
        request: Mapping[str, Any],
        feedback: str,
        memory_summary: str,
        action_hint: str,
    ) -> str:
        action_contract = render_action_contract(request)
        if action_hint:
            action_contract = f"{action_contract}\n\n【猎人专用决策提示】\n{action_hint}"
        if memory_summary:
            action_contract = f"{action_contract}\n\n【当前紧凑记忆】\n{memory_summary}"
        return render_prompt(
            "player_turn_instruction.txt",
            action_contract=action_contract,
            validation_feedback=(f"上一次输出未通过校验：{feedback}" if feedback else ""),
        )

    def _reset_game_memory(self, game_id: str | None) -> None:
        self._game_memory = {
            "game_id": game_id,
            "last_round_key": None,
            "recent_rounds": [],
            "clue_bank": {
                "confirmed": [],
                "claims": [],
                "votes": [],
                "deaths": [],
                "conflicts": [],
            },
            "current_controversy": "",
            "latest_public_state": "",
        }

    def _update_game_memory(self, packet: Mapping[str, Any]) -> None:
        game_id = self._extract_game_id(packet)
        if self._game_memory.get("game_id") != game_id:
            self._reset_game_memory(game_id)

        round_key = self._extract_round_key(packet)
        phase = self._extract_phase(packet)
        public_root = self._public_root(packet)
        summary_bits: list[str] = []
        if public_root is not None:
            state_summary = self._compact_value(public_root, max_depth=1, max_items=8, max_chars=240)
            self._game_memory["latest_public_state"] = state_summary
            summary_bits.append(f"state={state_summary}")
        for label, keys in (
            ("confirmed", {"confirmed", "confirmed_roles", "revealed", "revealed_roles", "flip", "flipped"}),
            ("claims", {"claim", "claims", "claimed", "role_claim", "role_claims"}),
            ("votes", {"vote", "votes", "ballot", "ballots", "voting"}),
            ("deaths", {"death", "deaths", "dead", "eliminated", "elimination"}),
            ("conflicts", {"accusation", "accusations", "conflict", "contradiction", "suspect", "suspicion", "pressure", "challenge"}),
        ):
            snippets = self._find_signal_snippets(public_root, keys)
            if not snippets:
                continue
            joined = "；".join(snippets[:2])
            self._remember_clues(label, snippets)
            summary_bits.append(f"{label}={joined}")
            if label in {"conflicts", "votes", "deaths"} and joined:
                self._game_memory["current_controversy"] = f"{label}={joined}"

        round_summary = f"r={round_key} phase={phase}"
        if summary_bits:
            round_summary = f"{round_summary} | " + " | ".join(summary_bits[:4])
        recent_rounds = self._game_memory["recent_rounds"]
        if recent_rounds and recent_rounds[-1]["round_key"] == round_key:
            recent_rounds[-1] = {"round_key": round_key, "summary": round_summary}
        else:
            recent_rounds.append({"round_key": round_key, "summary": round_summary})
        del recent_rounds[:-6]
        self._game_memory["last_round_key"] = round_key
        if not self._game_memory["current_controversy"] and summary_bits:
            self._game_memory["current_controversy"] = summary_bits[-1]

    def _build_memory_summary(self, turn_packet: Mapping[str, Any], current_dialogue: Any) -> str:
        public_root = self._public_root(turn_packet)
        round_key = self._extract_round_key(turn_packet)
        phase = self._extract_phase(turn_packet)
        current_dialogue_summary = self._compact_value(current_dialogue, max_depth=1, max_items=6, max_chars=260)
        recent_rounds = [item["summary"] for item in self._game_memory.get("recent_rounds", [])[-3:]]
        clue_bank = self._game_memory.get("clue_bank", {})
        clue_parts: list[str] = []
        for label in ("confirmed", "claims", "votes", "deaths", "conflicts"):
            entries = clue_bank.get(label) or []
            if entries:
                clue_parts.append(f"{label}={';'.join(entries[-2:])}")
        if public_root is not None:
            state_summary = self._compact_value(public_root, max_depth=1, max_items=8, max_chars=240)
        else:
            state_summary = self._game_memory.get("latest_public_state") or ""
        parts = [
            f"game={self._game_memory.get('game_id') or self._extract_game_id(turn_packet)}",
            f"round={round_key}",
            f"phase={phase}",
        ]
        if state_summary:
            parts.append(f"state={state_summary}")
        if recent_rounds:
            parts.append("recent=" + " || ".join(recent_rounds))
        if clue_parts:
            parts.append("clues=" + " || ".join(clue_parts[:3]))
        current_controversy = self._game_memory.get("current_controversy")
        if current_controversy:
            parts.append(f"controversy={current_controversy}")
        if current_dialogue_summary:
            parts.append(f"dialogue={current_dialogue_summary}")
        return self._truncate_text("\n".join(parts), min(1200, self.max_prompt_chars // 4))

    def _build_action_hint(
        self,
        request: Mapping[str, Any],
        memory_summary: str,
        current_dialogue: Any,
    ) -> str:
        allowed_actions = request.get("allowed_actions")
        allowed = [item for item in allowed_actions if isinstance(item, Mapping)] if isinstance(allowed_actions, list) else []
        kinds = {str(item.get("kind") or "") for item in allowed}
        has_speak = "speak" in kinds
        has_last_words = "last_words" in kinds
        target_ids: list[str] = []
        for item in allowed:
            ids = item.get("target_ids")
            if isinstance(ids, list):
                for target_id in ids:
                    target = str(target_id)
                    if target and target not in target_ids:
                        target_ids.append(target)
        if has_speak:
            evidence = self._select_evidence_line(memory_summary, current_dialogue)
            controversy = self._extract_controversy(memory_summary)
            parts = [
                f"当前争议：{controversy or '从最近票型/矛盾里选一个具体对象'}。",
                f"具体证据：{evidence or '从最近发言、投票或死亡链里挑 1 条最直接的证据'}。",
                "明确判断：只能给出支持/怀疑/待观察中的一个态度，先点名对象再下判断。",
                "禁止使用固定模板，例如“先听一圈/我先观望/带走最像狼的人”。",
            ]
            return " ".join(parts)
        if not has_last_words:
            return ""
        candidate_ids = target_ids or self._extract_player_ids(memory_summary + "\n" + self._compact_value(current_dialogue, max_depth=1, max_items=8, max_chars=320))
        if not candidate_ids:
            return "最后遗言优先写成低噪音、可复用的判断；证据不足时直接说明不点名。"
        scored = self._score_hunter_targets(candidate_ids, memory_summary, current_dialogue)
        ranked = sorted(scored.items(), key=lambda item: (-item[1][0], item[0]))
        lines = ["最后遗言先按评分器选最匹配的对象，再决定是否点名。", "评分优先级：票型/发言矛盾 > 公开身份声明 > 对关键位的直接威胁。"]
        for target_id, (score, reasons) in ranked[:4]:
            reason_text = "、".join(reasons[:3]) if reasons else "证据弱"
            lines.append(f"{target_id}:{score}({reason_text})")
        best_score = ranked[0][1][0] if ranked else 0
        if best_score < 2:
            lines.append("证据不足时优先写跳过型遗言：不点名、只说明依据不足。")
        elif ranked:
            lines.append(f"优先聚焦：{ranked[0][0]}；如果文本里必须点名，只点最强候选。")
        return " ".join(lines)

    def _score_hunter_targets(
        self,
        target_ids: list[str],
        memory_summary: str,
        current_dialogue: Any,
    ) -> dict[str, tuple[int, list[str]]]:
        text_sources = [memory_summary, self._compact_value(current_dialogue, max_depth=1, max_items=8, max_chars=320)]
        scores: dict[str, tuple[int, list[str]]] = {}
        for target_id in target_ids:
            score = 0
            reasons: list[str] = []
            for source in text_sources:
                if not source:
                    continue
                count = source.count(target_id)
                if count:
                    score += min(3, count)
                    if f"{target_id}=" in source or f"{target_id}:" in source:
                        reasons.append("被多次点名")
                if target_id in source and any(token in source for token in ("票", "投票", "矛盾", "冲突", "质疑", "怀疑", "指控")):
                    score += 2
                    reasons.append("票型/矛盾")
                if target_id in source and any(token in source for token in ("claim", "claimed", "声称", "自称", "跳身份", "预言家", "女巫", "守卫", "警长")):
                    score += 2
                    reasons.append("公开身份线索")
                if target_id in source and any(token in source for token in ("威胁", "带走", "刀", "夜刀", "关键位", "神职")):
                    score += 2
                    reasons.append("关键位威胁")
                if target_id in source and any(token in source for token in ("支持", "站边", "帮", "护", "保")):
                    score -= 1
                    reasons.append("有人在保")
            if score <= 0:
                reasons.append("证据弱")
            scores[target_id] = (score, reasons)
        return scores

    def _select_evidence_line(self, memory_summary: str, current_dialogue: Any) -> str:
        candidates = self._extract_evidence_lines(memory_summary)
        if not candidates:
            candidates = self._extract_evidence_lines(self._compact_value(current_dialogue, max_depth=1, max_items=6, max_chars=240))
        return candidates[0] if candidates else ""

    def _extract_controversy(self, memory_summary: str) -> str:
        for label in ("controversy=", "conflicts=", "votes=", "claims="):
            index = memory_summary.find(label)
            if index >= 0:
                tail = memory_summary[index + len(label):]
                return tail.split("\n", 1)[0].split(" || ", 1)[0].strip()
        return ""

    def _extract_evidence_lines(self, text: str) -> list[str]:
        if not text:
            return []
        lines: list[str] = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if any(token in stripped for token in ("votes=", "conflicts=", "deaths=", "claims=", "confirmed=", "票型", "争议", "死亡", "身份")):
                lines.append(stripped)
        return lines[:3]

    def _extract_player_ids(self, text: str) -> list[str]:
        if not text:
            return []
        candidates: list[str] = []
        for match in re.findall(r"\b(?:[pP]\d+|玩家\d+)\b", text):
            if match not in candidates:
                candidates.append(match)
        return candidates[:8]

    def _remember_clues(self, label: str, snippets: list[str]) -> None:
        bucket = self._game_memory.get("clue_bank", {}).get(label)
        if bucket is None:
            return
        for snippet in snippets[:3]:
            if snippet and snippet not in bucket:
                bucket.append(snippet)
        del bucket[:-6]

    def _public_root(self, packet: Mapping[str, Any]) -> Mapping[str, Any] | None:
        public_state = packet.get("public_state")
        if isinstance(public_state, Mapping):
            return public_state
        if isinstance(packet, Mapping):
            return {key: value for key, value in packet.items() if key != "private_information"}
        return None

    def _extract_game_id(self, packet: Mapping[str, Any]) -> str | None:
        candidates = [
            packet.get("game_id"),
            packet.get("game", {}).get("game_id") if isinstance(packet.get("game"), Mapping) else None,
            packet.get("public_state", {}).get("game_id") if isinstance(packet.get("public_state"), Mapping) else None,
        ]
        for candidate in candidates:
            if candidate not in (None, ""):
                return str(candidate)
        return None

    def _extract_round_key(self, packet: Mapping[str, Any]) -> str:
        game = packet.get("game") if isinstance(packet.get("game"), Mapping) else {}
        public_state = packet.get("public_state") if isinstance(packet.get("public_state"), Mapping) else {}
        round_value = (
            game.get("round")
            or packet.get("round")
            or public_state.get("round")
            or public_state.get("round_number")
            or public_state.get("day")
            or "?"
        )
        return str(round_value)

    def _extract_phase(self, packet: Mapping[str, Any]) -> str:
        game = packet.get("game") if isinstance(packet.get("game"), Mapping) else {}
        public_state = packet.get("public_state") if isinstance(packet.get("public_state"), Mapping) else {}
        phase_value = (
            game.get("public_phase")
            or game.get("phase")
            or packet.get("phase")
            or public_state.get("phase")
            or public_state.get("public_phase")
            or "unknown"
        )
        return str(phase_value)

    def _find_signal_snippets(self, node: Any, wanted_keys: set[str], *, max_depth: int = 3) -> list[str]:
        snippets: list[str] = []

        def walk(value: Any, depth: int) -> None:
            if len(snippets) >= 3 or depth > max_depth:
                return
            if isinstance(value, Mapping):
                for key, child in value.items():
                    key_text = str(key).lower()
                    if key_text in wanted_keys or any(token in key_text for token in wanted_keys):
                        compact = self._compact_value(child, max_depth=1, max_items=4, max_chars=120)
                        if compact and compact not in snippets:
                            snippets.append(compact)
                    if isinstance(child, (Mapping, list, tuple)):
                        walk(child, depth + 1)
            elif isinstance(value, (list, tuple)):
                for item in value[:4]:
                    if len(snippets) >= 3:
                        break
                    walk(item, depth + 1)

        walk(node, 0)
        return snippets

    def _compact_value(
        self,
        value: Any,
        *,
        max_depth: int = 2,
        max_items: int = 6,
        max_chars: int = 180,
    ) -> str:
        text = self._compact_value_inner(value, max_depth=max_depth, max_items=max_items)
        return self._truncate_text(text, max_chars)

    def _compact_value_inner(self, value: Any, *, max_depth: int, max_items: int) -> str:
        if value is None:
            return "null"
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, str):
            return self._truncate_text(" ".join(value.split()), 120)
        if max_depth <= 0:
            return self._truncate_text(type(value).__name__, 60)
        if isinstance(value, Mapping):
            parts: list[str] = []
            for index, (key, child) in enumerate(value.items()):
                if index >= max_items:
                    parts.append("...")
                    break
                parts.append(f"{key}:{self._compact_value_inner(child, max_depth=max_depth - 1, max_items=max_items)}")
            return "{" + ",".join(parts) + "}"
        if isinstance(value, (list, tuple, set)):
            parts = []
            for index, item in enumerate(value):
                if index >= max_items:
                    parts.append("...")
                    break
                parts.append(self._compact_value_inner(item, max_depth=max_depth - 1, max_items=max_items))
            return "[" + ",".join(parts) + "]"
        return self._truncate_text(repr(value), 120)

    @staticmethod
    def _truncate_text(text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        if max_chars <= 1:
            return text[:max_chars]
        return text[: max_chars - 1] + "…"

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

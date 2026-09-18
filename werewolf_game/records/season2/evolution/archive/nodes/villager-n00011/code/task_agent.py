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
_PLAYER_TOKEN = re.compile(r"\b(?:p|P)\d+\b")

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
        self._working_memory: dict[str, Any] = {
            "recent_public_events": [],
            "claim_log": [],
            "vote_log": [],
            "suspicion_scores": {},
        }

    async def observe(self, sync_packet: dict[str, Any]) -> None:
        """接收公开同步并压缩成轻量工作记忆。"""

        self._update_working_memory(sync_packet)

    async def decide(self, turn_packet: dict[str, Any]) -> dict[str, Any]:
        if turn_packet["self"]["player_id"] != self.player_id:
            raise ValueError("行动包被发送给了错误玩家")
        private = turn_packet["private_information"]
        if str(private.get("role")) != self.profile.role:
            raise ValueError("Task-Agent 的角色档案与行动包身份不一致")

        tool_context = turn_packet.get("tool_context") or {}
        current_dialogue = tool_context.get("current_round_dialogue") or []
        decision_aid = self._build_decision_aid(turn_packet, current_dialogue)
        system = self._system_prompt(private)
        prompt = {
            "game": turn_packet["game"],
            "public_rules": turn_packet["public_rules"],
            "self": turn_packet["self"],
            "private_information": private,
            "public_state": turn_packet["public_state"],
            "request": turn_packet["request"],
            "decision_aid": decision_aid,
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
                "instruction": self._turn_instruction(
                    turn_packet["request"], feedback, decision_aid
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
        base_prompt = render_prompt(
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
        return base_prompt + "\n\n" + self._villager_policy_block()

    @staticmethod
    def _turn_instruction(
        request: Mapping[str, Any], feedback: str, decision_aid: Mapping[str, Any]
    ) -> str:
        action_kinds = sorted(
            {
                str(item.get("kind"))
                for item in request.get("allowed_actions", [])
                if isinstance(item, Mapping) and item.get("kind")
            }
        )
        action_hint = "、".join(action_kinds) if action_kinds else "未知"
        memory_text = json.dumps(decision_aid, ensure_ascii=False, separators=(",", ":"))
        base = render_prompt(
            "player_turn_instruction.txt",
            action_contract=render_action_contract(request),
            validation_feedback=(f"上一次输出未通过校验：{feedback}" if feedback else ""),
        )
        return (
            base
            + "\n\n【平民决策清单】\n"
            + "1. 先写系统事实，再写玩家声明，再写自己的推测；不要把推测写成事实。\n"
            + "2. 发言必须点名当前局面里的具体人、具体票型或具体话术，避免空泛重复。\n"
            + "3. 投票或其他目标类行动前，至少比较两个候选，再说明为什么选当前目标。\n"
            + "4. 如果信息不足，给出当前怀疑顺序和下一步观察点，不要只说‘没信息’或‘先听’。\n"
            + f"【本次可见工作记忆】{memory_text}\n"
            + f"【本次允许行动】{action_hint}"
        )

    def _build_decision_aid(
        self, turn_packet: Mapping[str, Any], current_dialogue: list[Any]
    ) -> dict[str, Any]:
        public_summary = self._summarize_public_packet(turn_packet.get("public_state"))
        dialogue_summary = self._summarize_dialogue(current_dialogue)
        return {
            "public_facts": public_summary,
            "working_memory": self._working_memory_snapshot(),
            "current_dialogue": dialogue_summary,
            "comparison_frame": [
                "先确认事实，再分辨声明与推测",
                "比较至少两个候选",
                "优先使用票型、死亡和公开身份冲突",
            ],
        }

    def _working_memory_snapshot(self) -> dict[str, Any]:
        scores: dict[str, int] = self._working_memory.get("suspicion_scores", {})
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return {
            "recent_public_events": list(self._working_memory.get("recent_public_events", []))[-4:],
            "claim_log": list(self._working_memory.get("claim_log", []))[-4:],
            "vote_log": list(self._working_memory.get("vote_log", []))[-4:],
            "suspicion_order": [name for name, _score in ranked[:4]],
        }

    def _update_working_memory(self, sync_packet: Mapping[str, Any]) -> None:
        summary = self._summarize_public_packet(sync_packet)
        if not summary:
            return
        event_line = self._format_summary_line(summary)
        if event_line:
            recent = self._working_memory.setdefault("recent_public_events", [])
            recent.append(event_line)
            del recent[:-4]
        self._merge_events("claim_log", summary.get("claims", []), limit=4)
        self._merge_events("vote_log", summary.get("votes", []), limit=4)
        self._update_suspicion_scores(summary)

    def _update_suspicion_scores(self, summary: Mapping[str, Any]) -> None:
        scores: dict[str, int] = self._working_memory.setdefault("suspicion_scores", {})
        for claim in summary.get("claims", []):
            player, claim_text = self._split_claim(claim)
            if not player:
                continue
            delta = 2 if self._contains_role_like_word(claim_text) else 1
            scores[player] = scores.get(player, 0) + delta
        for vote in summary.get("votes", []):
            target = vote.split("->", 1)[-1].strip()
            if target:
                scores[target] = scores.get(target, 0) + 1
        for sheriff in summary.get("sheriff", []):
            holder = self._extract_head(sheriff)
            if holder:
                scores.setdefault(holder, 0)

    def _merge_events(self, key: str, items: list[str], *, limit: int) -> None:
        if not items:
            return
        bucket = self._working_memory.setdefault(key, [])
        for item in items:
            if item and item not in bucket:
                bucket.append(item)
        del bucket[:-limit]

    def _summarize_public_packet(self, packet: object) -> dict[str, Any]:
        if not isinstance(packet, Mapping):
            return {}
        summary: dict[str, Any] = {
            "round": None,
            "phase": None,
            "deaths": [],
            "sheriff": [],
            "claims": [],
            "votes": [],
        }
        self._walk_public_packet(packet, summary)
        summary["deaths"] = self._dedupe_texts(summary["deaths"])[:4]
        summary["sheriff"] = self._dedupe_texts(summary["sheriff"])[:2]
        summary["claims"] = self._dedupe_texts(summary["claims"])[:4]
        summary["votes"] = self._dedupe_texts(summary["votes"])[:4]
        return summary

    def _walk_public_packet(self, node: object, summary: dict[str, Any]) -> None:
        if isinstance(node, Mapping):
            lower = {str(key).lower(): key for key in node.keys()}
            if summary["round"] is None:
                round_value = self._first_matching_value(node, lower, ("round", "day", "turn"))
                if round_value:
                    summary["round"] = round_value
            if summary["phase"] is None:
                phase_value = self._first_matching_value(
                    node, lower, ("phase", "public_phase", "stage")
                )
                if phase_value:
                    summary["phase"] = phase_value

            for key in ("death", "dead", "deaths", "eliminated", "elimination", "out"):
                if key in lower:
                    self._append_event_labels(summary["deaths"], node[lower[key]], kind="death")
            for key in ("sheriff", "police", "badge", "captain", "leader"):
                if key in lower:
                    self._append_event_labels(summary["sheriff"], node[lower[key]], kind="sheriff")
            for key in ("claim", "claims", "identity", "role", "reveal", "revealed"):
                if key in lower:
                    self._append_event_labels(summary["claims"], node[lower[key]], kind="claim")
            for key in ("vote", "votes", "ballot", "voting"):
                if key in lower:
                    self._append_event_labels(summary["votes"], node[lower[key]], kind="vote")

            for value in node.values():
                self._walk_public_packet(value, summary)
        elif isinstance(node, list):
            for item in node:
                self._walk_public_packet(item, summary)

    def _append_event_labels(self, bucket: list[str], value: object, *, kind: str) -> None:
        if kind == "vote":
            labels = self._extract_vote_labels(value)
        elif kind == "claim":
            labels = self._extract_claim_labels(value)
        elif kind == "sheriff":
            labels = self._extract_sheriff_labels(value)
        else:
            labels = self._extract_death_labels(value)
        for label in labels:
            if label and label not in bucket:
                bucket.append(label)

    def _extract_death_labels(self, value: object) -> list[str]:
        labels: list[str] = []
        if isinstance(value, Mapping):
            player = self._first_key_text(value, ("player_id", "player", "id", "name", "target_id", "target"))
            reason = self._first_key_text(value, ("reason", "cause", "result"))
            if player:
                labels.append(player if not reason else f"{player}({reason})")
            elif reason:
                labels.append(reason)
        elif isinstance(value, list):
            for item in value:
                labels.extend(self._extract_death_labels(item))
        elif isinstance(value, str):
            text = self._compact_text(value, 80)
            if text:
                labels.append(text)
        return labels

    def _extract_sheriff_labels(self, value: object) -> list[str]:
        labels: list[str] = []
        if isinstance(value, Mapping):
            holder = self._first_key_text(value, ("player_id", "player", "id", "name", "sheriff_id", "badge_holder"))
            if holder:
                labels.append(f"{holder}(警长)")
            elif any(self._compact_text(item, 20) for item in value.values()):
                labels.append(self._compact_text(value, 60))
        elif isinstance(value, list):
            for item in value:
                labels.extend(self._extract_sheriff_labels(item))
        elif isinstance(value, str):
            text = self._compact_text(value, 80)
            if text:
                labels.append(text)
        return labels

    def _extract_claim_labels(self, value: object) -> list[str]:
        labels: list[str] = []
        if isinstance(value, Mapping):
            player = self._first_key_text(value, ("player_id", "player", "id", "name", "speaker", "speaker_id"))
            claim = self._first_key_text(
                value,
                (
                    "role",
                    "claim",
                    "identity",
                    "self_claim",
                    "claimed_role",
                    "description",
                    "text",
                ),
            )
            if player and claim:
                labels.append(f"{player}:{claim}")
            elif player:
                labels.append(player)
            elif claim:
                labels.append(claim)
        elif isinstance(value, list):
            for item in value:
                labels.extend(self._extract_claim_labels(item))
        elif isinstance(value, str):
            text = self._compact_text(value, 80)
            if text:
                labels.append(text)
        return labels

    def _extract_vote_labels(self, value: object) -> list[str]:
        labels: list[str] = []
        if isinstance(value, Mapping):
            voter = self._first_key_text(
                value, ("voter_id", "voter", "player_id", "player", "id", "name", "speaker")
            )
            target = self._first_key_text(value, ("target_id", "target", "candidate", "vote_target"))
            if voter and target:
                labels.append(f"{voter}->{target}")
            elif target:
                labels.append(target)
            elif voter:
                labels.append(voter)
        elif isinstance(value, list):
            for item in value:
                labels.extend(self._extract_vote_labels(item))
        elif isinstance(value, str):
            text = self._compact_text(value, 80)
            if text:
                labels.append(text)
        return labels

    def _format_summary_line(self, summary: Mapping[str, Any]) -> str:
        parts: list[str] = []
        round_value = summary.get("round")
        phase_value = summary.get("phase")
        if round_value is not None:
            parts.append(f"R{self._compact_text(round_value, 12)}")
        if phase_value is not None:
            parts.append(self._compact_text(phase_value, 16))
        if summary.get("deaths"):
            parts.append("死亡:" + "、".join(summary["deaths"]))
        if summary.get("sheriff"):
            parts.append("警长:" + "、".join(summary["sheriff"]))
        if summary.get("claims"):
            parts.append("声明:" + "；".join(summary["claims"]))
        if summary.get("votes"):
            parts.append("票型:" + "；".join(summary["votes"]))
        return " | ".join(parts)

    def _summarize_dialogue(self, dialogue: object) -> list[str]:
        if not isinstance(dialogue, list):
            return []
        lines: list[str] = []
        for item in dialogue[-8:]:
            if isinstance(item, Mapping):
                speaker = self._first_key_text(item, ("player_id", "player", "speaker_id", "speaker", "name", "id"))
                text = self._first_key_text(item, ("text", "content", "message", "say"))
                if speaker and text:
                    lines.append(f"{speaker}:{self._compact_text(text, 40)}")
                else:
                    compact = self._compact_text(item, 60)
                    if compact:
                        lines.append(compact)
            else:
                compact = self._compact_text(item, 60)
                if compact:
                    lines.append(compact)
        return lines[:4]

    def _first_matching_value(
        self, node: Mapping[str, Any], lower: Mapping[str, str], keys: tuple[str, ...]
    ) -> str | None:
        for key in keys:
            original = lower.get(key)
            if original is None:
                continue
            text = self._compact_text(node[original], 24)
            if text:
                return text
        return None

    def _first_key_text(self, node: Mapping[str, Any], keys: tuple[str, ...]) -> str:
        for key in keys:
            if key not in node:
                continue
            text = self._compact_text(node[key], 48)
            if text:
                return text
        for value in node.values():
            if isinstance(value, Mapping):
                text = self._first_key_text(value, keys)
                if text:
                    return text
        return ""

    def _split_claim(self, claim: str) -> tuple[str, str]:
        if ":" not in claim:
            return claim, ""
        head, tail = claim.split(":", 1)
        return head.strip(), tail.strip()

    def _extract_head(self, text: str) -> str:
        if not text:
            return ""
        if "(" in text:
            return text.split("(", 1)[0].strip()
        if ":" in text:
            return text.split(":", 1)[0].strip()
        return text.strip()

    def _contains_role_like_word(self, text: str) -> bool:
        if not text:
            return False
        keywords = (
            "预言家",
            "女巫",
            "猎人",
            "守卫",
            "警长",
            "狼人",
            "好人",
            "平民",
            "民",
            "神",
            "身份",
            "claim",
            "seer",
            "witch",
            "hunter",
            "guard",
            "sheriff",
            "werewolf",
            "villager",
        )
        lowered = text.lower()
        return any(word.lower() in lowered or word in text for word in keywords)

    def _dedupe_texts(self, items: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for item in items:
            text = self._compact_text(item, 80)
            if not text or text in seen:
                continue
            seen.add(text)
            result.append(text)
        return result

    def _compact_text(self, value: object, limit: int) -> str:
        if value is None:
            return ""
        if isinstance(value, bool):
            text = "true" if value else "false"
        elif isinstance(value, (str, int, float)):
            text = str(value)
        elif isinstance(value, Mapping):
            for key in (
                "player_id",
                "player",
                "speaker_id",
                "speaker",
                "id",
                "name",
                "target_id",
                "target",
                "role",
                "claim",
                "identity",
                "text",
                "content",
            ):
                if key in value:
                    text = self._compact_text(value[key], limit)
                    if text:
                        return text
            text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        elif isinstance(value, list):
            parts = [self._compact_text(item, max(8, limit // 2)) for item in value[:4]]
            parts = [part for part in parts if part]
            text = "、".join(parts)
        else:
            text = str(value)
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) > limit:
            return text[: max(0, limit - 1)] + "…"
        return text

    def _villager_policy_block(self) -> str:
        return (
            "【平民专用原则】\n"
            "- 先列事实，再列玩家声明，再列自己的推测。\n"
            "- 只把可见信息当事实，推测必须明确标注为推测。\n"
            "- 投票前至少比较两个候选，说明为什么当前目标更优先。\n"
            "- 发言要点名具体人、具体票型或具体话术，避免空泛重复。\n"
            "- 如果暂时不确定，就给出当前怀疑顺序和下一步观察点。"
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

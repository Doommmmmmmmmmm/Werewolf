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

CURRENT_ROUND_DIALOGUE_TOOL_NAME = "read_current_round_dialogue"
DEFAULT_MAX_TOOL_CALLS_PER_DECISION = 5
DEFAULT_MAX_TOOL_RESULT_TOKENS = 1000
DEFAULT_MAX_PROMPT_CHARS = 12000
# 这是 Task-Agent 层的“纠错重试”次数：首次模型调用之外，最多再请求两次。
# 传输层的 HTTP/网络重试仍由 ModelClient 的 MODEL_MAX_RETRIES 单独控制。
DEFAULT_MAX_DECISION_RETRIES = 2
_SPEECH_ACTION_KINDS = frozenset({"speak", "last_words"})
_PASS_LIKE_ACTION_KINDS = frozenset({"pass", "skip", "wait", "idle", "noop", "none"})
DEFAULT_PUBLIC_MEMORY_SIZE = 12


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
        self._public_memory: deque[str] = deque(maxlen=DEFAULT_PUBLIC_MEMORY_SIZE)
        self._model_token_usage: dict[str, int] = {
            "successful_response_count": 0,
            "api_attempt_count": 0,
            "reported_usage_response_count": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }

    async def observe(self, sync_packet: dict[str, Any]) -> None:
        """接收公开同步，并压缩成有限长的短记忆。"""

        if not isinstance(sync_packet, Mapping):
            return
        for event in self._extract_public_events(sync_packet):
            if event:
                self._public_memory.append(event)

    def _recent_memory(self, limit: int = 5) -> list[str]:
        if limit <= 0:
            return []
        return list(self._public_memory)[-limit:]

    def _decision_brief(
        self,
        turn_packet: Mapping[str, Any],
        current_dialogue: list[Any],
    ) -> dict[str, Any]:
        request = turn_packet["request"]
        allowed_actions = [
            item for item in request.get("allowed_actions", []) if isinstance(item, Mapping)
        ]
        phase = self._request_phase_label(turn_packet, request, allowed_actions)
        recent_memory = self._recent_memory(5)
        brief: dict[str, Any] = {
            "phase": phase,
            "recent_public_signals": recent_memory,
            "current_round_dialogue_preview": self._compact_dialogue(current_dialogue, limit=3),
        }
        mode = self._request_mode(allowed_actions)
        if mode == "hunter_reaction":
            ranking = self._hunter_target_ranking(allowed_actions, recent_memory)
            brief.update(
                {
                    "decision_mode": "hunter_reaction",
                    "focus": "先比较合法 target_ids 的票型、改口、身份声明和死亡链条；证据不足时优先跳过。",
                    "target_ranking": ranking["target_ranking"],
                    "skip_option": ranking["skip_option"],
                    "skip_condition": ranking["skip_condition"],
                }
            )
        elif mode == "last_words":
            brief.update(
                {
                    "decision_mode": "last_words",
                    "focus": "收束到已公开的票型、身份矛盾和死亡顺序，不要泛化找狼。",
                    "current_controversy": self._recent_controversy(recent_memory),
                    "key_contradiction": self._key_contradiction(recent_memory),
                }
            )
        elif mode == "speak":
            brief.update(
                {
                    "decision_mode": "speak",
                    "focus": "回应当前争议，优先点出最近票型变化与身份声明矛盾。",
                    "current_controversy": self._recent_controversy(recent_memory),
                    "key_contradiction": self._key_contradiction(recent_memory),
                }
            )
        else:
            brief.update(
                {
                    "decision_mode": mode,
                    "focus": "遵守当前阶段的合法行动范围，并尽量保持公开信息一致。",
                }
            )
        return brief

    @staticmethod
    def _request_mode(allowed_actions: list[Mapping[str, Any]]) -> str:
        kinds = [str(item.get("kind") or "") for item in allowed_actions if item.get("kind")]
        if "hunter_reaction" in kinds:
            return "hunter_reaction"
        if "last_words" in kinds:
            return "last_words"
        if "speak" in kinds:
            return "speak"
        if any(item.get("target_ids") for item in allowed_actions):
            return "hunter_reaction"
        return kinds[0] if kinds else "unknown"

    def _request_phase_label(
        self,
        turn_packet: Mapping[str, Any],
        request: Mapping[str, Any],
        allowed_actions: list[Mapping[str, Any]],
    ) -> str:
        game = turn_packet.get("game") or {}
        phase = game.get("public_phase") or game.get("phase")
        if phase:
            return str(phase)
        mode = self._request_mode(allowed_actions)
        request_phase = request.get("phase")
        if request_phase:
            return str(request_phase)
        return mode

    @staticmethod
    def _compact_text(value: object, limit: int = 48) -> str:
        text = str(value).strip()
        if len(text) <= limit:
            return text
        return text[: max(0, limit - 1)].rstrip() + "…"

    def _compact_dialogue(self, current_dialogue: list[Any], limit: int = 3) -> list[str]:
        if not isinstance(current_dialogue, list) or limit <= 0:
            return []
        preview: list[str] = []
        for item in current_dialogue[-limit:]:
            if isinstance(item, Mapping):
                speaker = str(
                    item.get("speaker")
                    or item.get("player_id")
                    or item.get("from_player_id")
                    or item.get("from")
                    or item.get("name")
                    or "?"
                )
                text = item.get("text", item.get("content", item.get("message", "")))
                compact = self._compact_text(text, limit=48)
                if compact:
                    preview.append(f"{speaker}: {compact}")
            elif isinstance(item, str):
                compact = self._compact_text(item, limit=48)
                if compact:
                    preview.append(compact)
        return preview

    def _recent_controversy(self, recent_memory: list[str]) -> str:
        if not recent_memory:
            return "暂无足够公开记忆"
        return "；".join(recent_memory[-3:])

    def _key_contradiction(self, recent_memory: list[str]) -> str:
        for line in reversed(recent_memory):
            if any(keyword in line for keyword in ("改口", "矛盾", "对跳", "反水", "自相矛盾", "投票", "死亡", "警长")):
                return line
        return recent_memory[-1] if recent_memory else "暂无明确矛盾"

    def _hunter_target_ranking(
        self,
        allowed_actions: list[Mapping[str, Any]],
        recent_memory: list[str],
    ) -> dict[str, Any]:
        target_ids: list[str] = []
        for action in allowed_actions:
            raw_targets = action.get("target_ids")
            if not isinstance(raw_targets, list):
                continue
            for target in raw_targets:
                target_id = str(target)
                if target_id and target_id not in target_ids:
                    target_ids.append(target_id)
        ranking: list[dict[str, Any]] = []
        for target_id in target_ids:
            score, reasons = self._score_hunter_target(target_id, recent_memory)
            ranking.append(
                {
                    "target_id": target_id,
                    "score": round(score, 2),
                    "reasons": reasons,
                }
            )
        ranking.sort(key=lambda item: item["score"], reverse=True)
        skip_option = self._pass_like_action_kind(allowed_actions)
        best_score = ranking[0]["score"] if ranking else 0.0
        if skip_option:
            skip_condition = "若没有明确矛盾、票型偏差或身份冲突，优先跳过。"
        else:
            skip_condition = "没有跳过动作时，只选分数最高且风险最低的目标。"
        if best_score < 1.0:
            skip_condition = "证据不足时直接跳过；若不能跳过，再选最弱风险目标。"
        return {
            "target_ranking": ranking[:4],
            "skip_option": skip_option,
            "skip_condition": skip_condition,
        }

    def _score_hunter_target(self, target_id: str, recent_memory: list[str]) -> tuple[float, list[str]]:
        pattern = re.compile(rf"(?<![A-Za-z0-9]){re.escape(target_id)}(?![A-Za-z0-9])")
        score = 0.0
        reasons: list[str] = []
        for line in recent_memory:
            if not pattern.search(line):
                continue
            line_score = 0.2
            if any(keyword in line for keyword in ("改口", "矛盾", "对跳", "反水", "带节奏", "卖队", "假", "骗", "冲票", "踩", "查杀", "自相矛盾")):
                line_score += 1.3
                reasons.append(self._compact_text(line, limit=60))
            if any(keyword in line for keyword in ("投票", "票型", "站边", "出局", "死亡", "身份", "警长", "发言", "表态")):
                line_score += 0.4
            if any(keyword in line for keyword in ("可信", "一致", "解释清楚", "对得上", "稳定", "金水")):
                line_score -= 0.5
            if "狼人" in line:
                line_score += 0.8
            if "好人" in line:
                line_score -= 0.2
            score += line_score
        if not reasons:
            reasons.append("公开记忆里缺少直接矛盾，只能低置信度判断")
        return score, reasons[:3]

    @staticmethod
    def _pass_like_action_kind(allowed_actions: list[Mapping[str, Any]]) -> str:
        for action in allowed_actions:
            kind = str(action.get("kind") or "").lower()
            if kind in _PASS_LIKE_ACTION_KINDS:
                return kind
        return ""

    def _extract_public_events(self, sync_packet: Mapping[str, Any]) -> list[str]:
        events: list[str] = []
        seen: set[str] = set()
        default_round = sync_packet.get("round")
        default_phase = sync_packet.get("phase") or sync_packet.get("public_phase")

        def add_event(item: object) -> None:
            event = self._summarize_public_event(item, default_round=default_round, default_phase=default_phase)
            if not event or event in seen:
                return
            seen.add(event)
            events.append(event)

        for key in ("events", "public_events", "log", "logs", "history", "records", "timeline"):
            value = sync_packet.get(key)
            if isinstance(value, list):
                for item in value:
                    add_event(item)
                    if len(events) >= 8:
                        return events
        for key in ("public_state", "state", "game"):
            value = sync_packet.get(key)
            if value is not None:
                add_event(value)
                if len(events) >= 8:
                    return events

        def walk(item: object, depth: int) -> None:
            if len(events) >= 8 or depth > 2:
                return
            if isinstance(item, Mapping):
                if self._looks_like_public_event(item):
                    add_event(item)
                else:
                    for value in item.values():
                        if isinstance(value, (Mapping, list, tuple)):
                            walk(value, depth + 1)
                return
            if isinstance(item, (list, tuple)):
                for value in item:
                    walk(value, depth + 1)
                    if len(events) >= 8:
                        return

        walk(sync_packet, 0)
        return events

    @staticmethod
    def _looks_like_public_event(item: Mapping[str, Any]) -> bool:
        return any(
            key in item
            for key in (
                "speaker",
                "player_id",
                "from_player_id",
                "target_id",
                "vote",
                "votes",
                "kind",
                "action",
                "type",
                "text",
                "content",
                "message",
                "role",
                "death",
                "died",
                "alive",
                "status",
            )
        )

    @staticmethod
    def _summarize_public_event(
        item: object,
        *,
        default_round: object = None,
        default_phase: object = None,
    ) -> str | None:
        if isinstance(item, str):
            text = item.strip()
            if not text:
                return None
            return TaskAgent._compact_text(text, limit=72)
        if not isinstance(item, Mapping):
            return None

        round_value = item.get("round", default_round)
        phase_value = item.get("phase", item.get("public_phase", default_phase))
        prefix_parts: list[str] = []
        if round_value is not None:
            prefix_parts.append(f"r{round_value}")
        if phase_value is not None:
            prefix_parts.append(str(phase_value))
        prefix = "[" + "/".join(prefix_parts) + "] " if prefix_parts else ""

        actor = str(
            item.get("speaker")
            or item.get("player_id")
            or item.get("from_player_id")
            or item.get("from")
            or item.get("actor")
            or item.get("source")
            or ""
        ).strip()
        target = str(item.get("target_id") or item.get("target") or item.get("votee") or "").strip()
        kind = str(item.get("kind") or item.get("action") or item.get("type") or "").strip().lower()
        text = str(item.get("text") or item.get("content") or item.get("message") or "").strip()
        role = str(item.get("role") or item.get("revealed_role") or item.get("identity") or "").strip()

        parts: list[str] = []
        if actor:
            parts.append(actor)
        if kind in {"speak", "speech", "say", "dialogue", "statement"} or text:
            if text:
                parts.append(f"发言:{TaskAgent._compact_text(text, limit=36)}")
        elif kind in {"vote", "voting"} or item.get("vote") is not None:
            if target:
                parts.append(f"投票→{target}")
            elif item.get("vote") is not None:
                parts.append(f"投票→{TaskAgent._compact_text(item.get('vote'), limit=24)}")
        elif kind in {"death", "died", "eliminate", "eliminated", "kill", "killed"} or item.get("death") is not None:
            if target:
                parts.append(f"死亡→{target}")
            else:
                parts.append("发生死亡")
        elif kind in {"reveal", "identity", "open_role", "claim"} or role:
            claim = role or text or "身份公开"
            parts.append(f"身份:{TaskAgent._compact_text(claim, limit=28)}")
        else:
            for key in ("vote", "votes", "result", "status", "event"):
                value = item.get(key)
                if value is not None:
                    parts.append(f"{key}:{TaskAgent._compact_text(value, limit=28)}")
                    break
            if not parts and text:
                parts.append(TaskAgent._compact_text(text, limit=36))

        if not parts:
            return None
        if prefix:
            return prefix + " ".join(parts)
        return " ".join(parts)

    async def decide(self, turn_packet: dict[str, Any]) -> dict[str, Any]:
        if turn_packet["self"]["player_id"] != self.player_id:
            raise ValueError("行动包被发送给了错误玩家")
        private = turn_packet["private_information"]
        if str(private.get("role")) != self.profile.role:
            raise ValueError("Task-Agent 的角色档案与行动包身份不一致")

        system = self._system_prompt(private)

        tool_context = turn_packet.get("tool_context") or {}
        current_dialogue = tool_context.get("current_round_dialogue") or []
        prompt = {
            "game": turn_packet["game"],
            "public_rules": turn_packet["public_rules"],
            "self": turn_packet["self"],
            "private_information": private,
            "public_state": turn_packet["public_state"],
            "request": turn_packet["request"],
            "memory_brief": self._recent_memory(5),
            "decision_brief": self._decision_brief(turn_packet, current_dialogue),
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

    @staticmethod
    def _nonnegative_int(value: object, *, fallback: int | None) -> int | None:
        if isinstance(value, bool):
            return fallback
        try:
            normalized = int(value)
        except (TypeError, ValueError):
            return fallback
        return normalized if normalized >= 0 else fallback

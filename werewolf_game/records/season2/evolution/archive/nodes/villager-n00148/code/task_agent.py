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
    """把本次 ``allowed_actions`` 转成模型容易遵守的最终 JSON 契约。"""

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
            "players": {},
            "verified_facts": [],
            "recent_public_events": [],
            "claim_log": [],
            "vote_log": [],
            "public_reveals": [],
            "death_log": [],
            "target_pressure": {},  # 兼容旧快照；新证据不再依赖它累计
            "suspicion_scores": {},
            "evidence_ledger": {},
            "seen_event_keys": [],
            "vote_matrix": {},
            "stance_timeline": [],
            "raw_public_events": [],
            "alive_ids": [],
            "last_public_round": None,
            "late_mode": False,
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
            + "1. 先写 verified facts，再写声明，再写推测；score 只是压缩索引，不是事实。\n"
            + "2. 目标必须来自本次 allowed_actions 且当前存活；speech/pass 不要臆造 target。\n"
            + "3. 查验对象已死亡时，只评估时间线与收益，不把死后查验自动当作可落地查杀。\n"
            + "4. 不要求守卫公开证明未公开的夜间行动；对跳要比较完整验人链、警徽、票型和时间线。\n"
            + "5. 目标行动前必须比较两个合法候选，分别列证据和不确定性，再作决定。\n"
            + "6. 发言点名具体票型/时间线/声明冲突；信息不足也要给排序和备选解释。\n"
            + f"【本次可见工作记忆】{memory_text}\n"
            + f"【本次允许行动】{action_hint}"
        )

    def _build_decision_aid(
        self, turn_packet: Mapping[str, Any], current_dialogue: list[Any]
    ) -> dict[str, Any]:
        public_summary = self._summarize_public_packet(turn_packet.get("public_state"))
        self._refresh_late_mode(public_summary)
        candidate_table = self._build_candidate_table(turn_packet, public_summary)
        focus_players = [str(item.get("player_id") or "") for item in candidate_table]
        dialogue_summary = self._summarize_dialogue(current_dialogue, focus_players)
        working_memory = self._working_memory_snapshot(
            {row["player_id"] for row in candidate_table}
        )
        verified_info = self._format_verified_facts()
        suspect_list = [
            {
                "player_id": row["player_id"],
                "score": row["score"],
                "evidence_reasons": row.get("evidence_reasons", []),
                "why": row["why"],
            }
            for row in candidate_table[:5]
        ]
        return {
            "state": {
                "round": public_summary.get("round"),
                "phase": public_summary.get("phase"),
                "alive_count": public_summary.get("alive_count"),
                "late_mode": self._working_memory.get("late_mode", False),
            },
            "verified_info": verified_info,
            "suspect_list": suspect_list,
            "candidate_table": candidate_table,
            "working_memory": working_memory,
            "current_dialogue": dialogue_summary,
            "comparison_frame": [
                "先确认事实，再分辨声明与推测",
                "比较至少两个候选",
                "优先使用票型、死亡、公开翻牌和改口冲突",
            ],
        }

    def _working_memory_snapshot(self, legal_ids: set[str] | None = None) -> dict[str, Any]:
        scores: dict[str, int] = self._working_memory.get("suspicion_scores", {})
        if legal_ids is not None:
            scores = {name: score for name, score in scores.items() if name in legal_ids}
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        players = self._working_memory.get("players", {})
        top_players = [name for name, _score in ranked[:5]]
        player_cards: dict[str, Any] = {}
        for player_id in top_players:
            player_cards[player_id] = self._compact_player_card(player_id, players.get(player_id, {}))
        return {
            "late_mode": self._working_memory.get("late_mode", False),
            "last_public_round": self._working_memory.get("last_public_round"),
            "verified_facts": list(self._working_memory.get("verified_facts", []))[-8:],
            "suspicion_order": [name for name, _score in ranked[:5]],
            "player_cards": player_cards,
            "vote_matrix": dict(list(self._working_memory.get("vote_matrix", {}).items())[-6:]),
            "stance_timeline": list(self._working_memory.get("stance_timeline", []))[-8:],
            "evidence_note": "score 是每类证据封顶后的压缩索引；必须以 evidence_reasons 和原始时间线核对。",
        }

    def _update_working_memory(self, sync_packet: Mapping[str, Any]) -> None:
        summary = self._summarize_public_packet(sync_packet)
        if not summary:
            return
        # 必须在摄取本批事件之前切换模式，否则进入残局的第一批票会用旧权重。
        self._refresh_late_mode(summary)
        round_value = summary.get("round")
        if round_value is not None:
            self._working_memory["last_public_round"] = round_value
        alive_ids = summary.get("alive_ids")
        if alive_ids:
            self._working_memory["alive_ids"] = list(alive_ids)
        self._ingest_summary(summary)
        event_line = self._format_summary_line(summary)
        if event_line:
            recent = self._working_memory.setdefault("recent_public_events", [])
            if event_line not in recent:
                recent.append(event_line)
            del recent[:-6]
        self._recompute_suspicion_scores()

    def _ingest_summary(self, summary: Mapping[str, Any]) -> None:
        round_value = summary.get("round")
        phase_value = summary.get("phase")
        streams = (
            ("death", "deaths", self._ingest_death_event),
            ("reveal", "public_reveals", self._ingest_public_reveal_event),
            ("sheriff", "sheriff", self._ingest_sheriff_event),
            ("claim", "claims", self._ingest_claim_event),
            ("vote", "votes", self._ingest_vote_event),
        )
        seen = self._working_memory.setdefault("seen_event_keys", [])
        for event_type, bucket, ingest in streams:
            for event in summary.get(bucket, []):
                event_round = event.get("round", round_value) if isinstance(event, Mapping) else round_value
                event_phase = event.get("phase", phase_value) if isinstance(event, Mapping) else phase_value
                key = self._event_key(event_type, event, event_round, event_phase)
                if not key or key in seen:
                    continue
                seen.append(key)
                del seen[:-256]
                ingest(event, event_round, event_phase)

    def _event_key(self, event_type: str, event: Mapping[str, Any], round_value: Any, phase_value: Any) -> str:
        subject = event.get("player_id") or event.get("voter_id") or event.get("holder") or ""
        target = event.get("target_id") or ""
        text = self._normalize_event_text(event.get("text") or event.get("claim_text"))
        return "|".join((event_type, self._compact_text(round_value, 16), self._compact_text(phase_value, 20), self._compact_text(subject, 24), self._compact_text(target, 24), text))

    def _recompute_suspicion_scores(self) -> None:
        """从有界、分类型账本生成压缩排序；score 只是索引，不是事实。"""
        ledger: dict[str, Any] = self._working_memory.get("evidence_ledger", {})
        scores: dict[str, int] = {}
        for player_id in self._known_player_ids():
            buckets = ledger.get(player_id, {})
            total = 0
            for category, entries in buckets.items():
                if not isinstance(entries, list):
                    continue
                # 每类证据的总贡献封顶，且同一轮最多保留一个贡献。
                by_round: dict[str, int] = {}
                for entry in entries:
                    if not isinstance(entry, Mapping):
                        continue
                    round_key = str(entry.get("round") or "?")
                    value = int(entry.get("weight", 0))
                    by_round[round_key] = max(by_round.get(round_key, -99), value)
                category_cap = {
                    "vote_evidence": 6, "claim_evidence": 3,
                    "contradiction_evidence": 8, "revealed_alignment": -8,
                    "death_order_evidence": 5,
                }.get(category, 4)
                subtotal = sum(by_round.values())
                total += max(-abs(category_cap), min(abs(category_cap), subtotal))
            scores[player_id] = max(-12, min(12, total))
        self._working_memory["suspicion_scores"] = scores

    def _add_evidence(
        self, player_id: str, category: str, reason: str, round_value: Any,
        weight: int, *, source_key: str = ""
    ) -> None:
        if not player_id:
            return
        ledger = self._working_memory.setdefault("evidence_ledger", {})
        buckets = ledger.setdefault(player_id, {})
        entries = buckets.setdefault(category, [])
        entry = {
            "round": self._compact_text(round_value, 12) or "?",
            "weight": int(weight),
            "reason": self._compact_text(reason, 72),
            "source_key": self._compact_text(source_key, 40),
        }
        key = json.dumps(entry, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if not any(json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":")) == key for item in entries if isinstance(item, Mapping)):
            entries.append(entry)
        del entries[:-12]

    def _refresh_late_mode(self, public_summary: Mapping[str, Any]) -> None:
        alive_count = public_summary.get("alive_count")
        late_mode = False
        if isinstance(alive_count, int):
            late_mode = alive_count <= 8
        elif isinstance(alive_count, str) and alive_count.isdigit():
            late_mode = int(alive_count) <= 8
        self._working_memory["late_mode"] = late_mode

    def _legal_target_ids(self, request: Mapping[str, Any], public_summary: Mapping[str, Any]) -> list[str]:
        ids: list[str] = []
        for allowed in request.get("allowed_actions", []):
            if not isinstance(allowed, Mapping):
                continue
            targets = allowed.get("target_ids")
            if isinstance(targets, list):
                ids.extend(str(item) for item in targets if item)
        ids = self._dedupe_preserve_order(ids)
        alive = {str(item) for item in public_summary.get("alive_ids", []) if item}
        if not alive:
            alive = {str(item) for item in self._working_memory.get("alive_ids", []) if item}
        if alive:
            ids = [item for item in ids if item in alive]
        return [item for item in ids if item != self.player_id]

    def _build_candidate_table(
        self, turn_packet: Mapping[str, Any], public_summary: Mapping[str, Any]
    ) -> list[dict[str, Any]]:
        legal_ids = self._legal_target_ids(turn_packet.get("request", {}), public_summary)
        if not legal_ids:
            return []
        scores: dict[str, int] = self._working_memory.get("suspicion_scores", {})
        players: dict[str, dict[str, Any]] = self._working_memory.get("players", {})
        ranked = sorted(legal_ids, key=lambda pid: (-scores.get(pid, 0), pid))
        return [self._make_candidate_row(pid, scores.get(pid, 0), players.get(pid, {})) for pid in ranked[:4]]

    def _make_candidate_row(
        self, player_id: str, score: int, record: Mapping[str, Any]
    ) -> dict[str, Any]:
        recent_claims = self._tail_texts(record.get("claims", []), "claim_text")
        recent_votes = self._tail_texts(record.get("votes", []), "target_id")
        recent_reveals = self._tail_texts(record.get("public_reveals", []), "text")
        recent_deaths = self._tail_texts(record.get("death_info", []), "text")
        contradictions = self._tail_texts(record.get("contradictions", []), "text")
        verified_alignment = self._tail_texts(record.get("verified_alignment", []), "text")
        current_stance = self._derive_current_stance(record)
        why = self._candidate_reason(score, recent_claims, recent_votes, contradictions, recent_reveals, recent_deaths)
        return {
            "player_id": player_id,
            "score": score,
            "evidence_reasons": self._evidence_reasons(player_id),
            "current_stance": current_stance,
            "recent_votes": recent_votes,
            "recent_claims": recent_claims,
            "verified_alignment": verified_alignment,
            "contradictions": contradictions,
            "death_info": recent_deaths,
            "public_reveals": recent_reveals,
            "last_seen_round": record.get("last_seen_round"),
            "priority": self._priority_label(score),
            "why": why,
        }

    def _compact_player_card(self, player_id: str, record: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "claims": self._tail_texts(record.get("claims", []), "claim_text"),
            "votes": self._tail_texts(record.get("votes", []), "target_id"),
            "public_reveals": self._tail_texts(record.get("public_reveals", []), "text"),
            "death_info": self._tail_texts(record.get("death_info", []), "text"),
            "contradictions": self._tail_texts(record.get("contradictions", []), "text"),
            "verified_alignment": self._tail_texts(record.get("verified_alignment", []), "text"),
            "last_seen_round": record.get("last_seen_round"),
        }

    def _candidate_reason(
        self,
        score: int,
        recent_claims: list[str],
        recent_votes: list[str],
        contradictions: list[str],
        recent_reveals: list[str],
        recent_deaths: list[str],
    ) -> str:
        parts: list[str] = [f"分数{score}"]
        if contradictions:
            parts.append(f"矛盾{len(contradictions)}")
        if recent_reveals:
            parts.append("有公开翻牌")
        if recent_deaths:
            parts.append("有死亡信息")
        if recent_votes:
            parts.append(f"票型{recent_votes[-1]}")
        if recent_claims:
            parts.append(f"声明{recent_claims[-1]}")
        return "；".join(parts)

    def _evidence_reasons(self, player_id: str, limit: int = 3) -> list[str]:
        ledger = self._working_memory.get("evidence_ledger", {}).get(player_id, {})
        rows: list[tuple[str, str]] = []
        for category, entries in ledger.items():
            for entry in entries[-4:] if isinstance(entries, list) else []:
                if isinstance(entry, Mapping) and entry.get("reason"):
                    reason = f"{category}@R{entry.get('round') or '?'}: {entry['reason']}"
                    rows.append((str(entry.get("round") or "?"), reason))
        return [reason for _round, reason in rows[-limit:]]

    def _priority_label(self, score: int) -> str:
        if score >= 7:
            return "高"
        if score >= 3:
            return "中"
        return "低"

    def _derive_current_stance(self, record: Mapping[str, Any]) -> str:
        claims = record.get("claims", [])
        votes = record.get("votes", [])
        if claims:
            latest = claims[-1]
            target_id = self._compact_text(latest.get("target_id"), 20)
            role_hint = self._compact_text(latest.get("role_hint"), 20)
            if target_id and role_hint:
                return f"{target_id}:{role_hint}"
            if target_id:
                return target_id
            if role_hint:
                return role_hint
        if votes:
            target_id = self._compact_text(votes[-1].get("target_id"), 20)
            if target_id:
                return f"vote->{target_id}"
        return "unknown"

    def _tail_texts(self, items: list[Any], key: str, limit: int = 2) -> list[str]:
        texts: list[str] = []
        for item in items[-limit:]:
            if isinstance(item, Mapping):
                value = item.get(key)
                if key == "text" and not value:
                    value = item.get("claim_text") or item.get("role_hint") or item.get("reason")
                text = self._compact_text(value, 48)
                if text:
                    texts.append(text)
            else:
                text = self._compact_text(item, 48)
                if text:
                    texts.append(text)
        return texts

    def _format_verified_facts(self) -> list[str]:
        return list(self._working_memory.get("verified_facts", []))[-8:]

    def _update_player_record(self, player_id: str, round_value: Any, phase_value: Any) -> dict[str, Any]:
        players: dict[str, dict[str, Any]] = self._working_memory.setdefault("players", {})
        record = players.setdefault(
            player_id,
            {
                "claims": [],
                "votes": [],
                "public_reveals": [],
                "death_info": [],
                "contradictions": [],
                "verified_alignment": [],
                "vote_mismatches": [],
                "last_seen_round": None,
                "last_seen_phase": None,
                "target_mentions": [],
                "role_claims": [],
                "last_vote_target": "",
            },
        )
        if round_value is not None:
            record["last_seen_round"] = self._compact_text(round_value, 24)
        if phase_value is not None:
            record["last_seen_phase"] = self._compact_text(phase_value, 24)
        return record

    def _ingest_claim_event(self, claim: Mapping[str, Any], round_value: Any, phase_value: Any) -> None:
        speaker = self._compact_text(claim.get("player_id"), 24)
        if not speaker:
            return
        record = self._update_player_record(speaker, round_value, phase_value)
        claim_text = self._compact_text(claim.get("claim_text"), 80)
        target_id = self._compact_text(claim.get("target_id"), 24)
        role_hint = self._compact_text(claim.get("role_hint"), 24)
        previous_role = self._latest_claim_role(record)
        previous_target = self._last_claim_target(record)
        status = "initial_claim" if not record.get("claims") else "continued_claim"
        temporal_relation = "after_previous" if record.get("claims") else "first_seen"
        if record.get("death_info"):
            status = "post_death_claim"
            temporal_relation = "after_death"
        elif previous_role and role_hint and previous_role != role_hint:
            status = "changed_claim"
            temporal_relation = "changed_from_previous"
        elif previous_target and target_id and previous_target != target_id:
            temporal_relation = "changed_target"
        if role_hint in {"守卫", "女巫"} and not claim.get("public_action_proof"):
            status = "unverifiable_private_action_claim"
            temporal_relation = "private_action_not_publicly_verifiable"
        entry = {
            "round": self._compact_text(round_value, 12),
            "phase": self._compact_text(phase_value, 16),
            "claim_text": claim_text,
            "target_id": target_id,
            "role_hint": role_hint,
            "claim_status": status,
            "temporal_relation": temporal_relation,
            "stance": self._compact_text(claim.get("stance") or claim.get("position"), 16),
            "text": claim_text,
        }
        record["claims"].append(entry)
        del record["claims"][:-4]
        if target_id:
            record["target_mentions"].append(target_id)
            del record["target_mentions"][:-4]
        if role_hint:
            record["role_claims"].append(role_hint)
            del record["role_claims"][:-4]
        if claim_text:
            log = self._working_memory.setdefault("claim_log", [])
            log.append(f"{speaker}:{claim_text}")
            del log[:-6]
        if target_id:
            self._add_evidence(target_id, "claim_evidence", f"{speaker} 指向 {target_id}: {claim_text}", round_value, self._claim_pressure_weight(), source_key=str(claim.get("source_key") or ""))
        stance = entry.get("stance")
        if stance:
            timeline = self._working_memory.setdefault("stance_timeline", [])
            timeline.append({"round": entry["round"], "phase": entry["phase"], "speaker": speaker, "stance": stance, "target": target_id})
            del timeline[:-24]
        if status == "changed_claim" or (previous_target and target_id and previous_target != target_id):
            self._add_evidence(speaker, "contradiction_evidence", "前后声明或指向发生变化", round_value, 3)
        self._add_claim_contradictions(speaker, record, role_hint, claim_text)
        self._maybe_register_verified_alignment(speaker, record)

    def _ingest_vote_event(self, vote: Mapping[str, Any], round_value: Any, phase_value: Any) -> None:
        voter = self._compact_text(vote.get("player_id"), 24)
        target_id = self._compact_text(vote.get("target_id"), 24)
        if not voter or not target_id:
            return
        record = self._update_player_record(voter, round_value, phase_value)
        entry = {
            "round": self._compact_text(round_value, 12),
            "phase": self._compact_text(phase_value, 16),
            "target_id": target_id,
            "text": f"{voter}->{target_id}",
        }
        record["votes"].append(entry)
        del record["votes"][:-4]
        record["last_vote_target"] = target_id
        self._add_evidence(target_id, "vote_evidence", f"{voter} 显式投票给 {target_id}", round_value, self._vote_pressure_weight(), source_key=str(vote.get("source_key") or ""))
        matrix = self._working_memory.setdefault("vote_matrix", {})
        round_key = self._compact_text(round_value, 12) or "?"
        matrix.setdefault(round_key, []).append({"voter": voter, "target": target_id, "phase": entry["phase"]})
        del matrix[round_key][:-32]
        while len(matrix) > 12:
            matrix.pop(next(iter(matrix)))
        vote_log = self._working_memory.setdefault("vote_log", [])
        vote_log.append(entry["text"])
        del vote_log[:-8]
        last_claim_target = self._last_claim_target(record)
        if last_claim_target and last_claim_target != target_id:
            mismatch = {
                "round": entry["round"],
                "phase": entry["phase"],
                "text": f"投票{target_id}与先前指向{last_claim_target}不一致",
            }
            record["vote_mismatches"].append(mismatch)
            del record["vote_mismatches"][:-4]
            self._add_evidence(voter, "contradiction_evidence", mismatch["text"], round_value, 2)

    def _ingest_public_reveal_event(
        self, reveal: Mapping[str, Any], round_value: Any, phase_value: Any
    ) -> None:
        player_id = self._compact_text(reveal.get("player_id"), 24)
        if not player_id:
            return
        record = self._update_player_record(player_id, round_value, phase_value)
        role_hint = self._compact_text(reveal.get("role_hint"), 24)
        text = self._compact_text(reveal.get("text"), 80)
        entry = {
            "round": self._compact_text(round_value, 12),
            "phase": self._compact_text(phase_value, 16),
            "role_hint": role_hint,
            "text": text or (f"{player_id}:{role_hint}" if role_hint else player_id),
        }
        record["public_reveals"].append(entry)
        del record["public_reveals"][:-4]
        if entry["text"]:
            reveals = self._working_memory.setdefault("public_reveals", [])
            reveals.append(entry["text"])
            del reveals[:-8]
        if role_hint:
            self._add_evidence(player_id, "revealed_alignment", f"公开翻牌/身份：{role_hint}", round_value, self._reveal_pressure_weight(role_hint), source_key=str(reveal.get("source_key") or ""))
        self._maybe_register_verified_alignment(player_id, record)
        self._append_verified_fact(entry["text"])

    def _ingest_death_event(self, death: Mapping[str, Any], round_value: Any, phase_value: Any) -> None:
        player_id = self._compact_text(death.get("player_id"), 24)
        if not player_id:
            return
        record = self._update_player_record(player_id, round_value, phase_value)
        role_hint = self._compact_text(death.get("role_hint"), 24)
        reason = self._compact_text(death.get("reason"), 40)
        text = self._compact_text(death.get("text"), 80)
        entry = {
            "round": self._compact_text(round_value, 12),
            "phase": self._compact_text(phase_value, 16),
            "reason": reason,
            "role_hint": role_hint,
            "text": text or (f"{player_id}({reason})" if reason else player_id),
        }
        record["death_info"].append(entry)
        del record["death_info"][:-4]
        death_log = self._working_memory.setdefault("death_log", [])
        death_log.append(entry["text"])
        del death_log[:-8]
        if role_hint:
            self._add_evidence(player_id, "death_order_evidence", f"死亡顺序信息：{role_hint}", round_value, self._death_pressure_weight(role_hint), source_key=str(death.get("source_key") or ""))
        self._maybe_register_verified_alignment(player_id, record)
        self._append_verified_fact(entry["text"])

    def _ingest_sheriff_event(self, sheriff: Mapping[str, Any], round_value: Any, phase_value: Any) -> None:
        holder = self._compact_text(sheriff.get("player_id"), 24)
        if not holder:
            return
        record = self._update_player_record(holder, round_value, phase_value)
        text = self._compact_text(sheriff.get("text"), 80) or f"{holder}(警长)"
        entry = {
            "round": self._compact_text(round_value, 12),
            "phase": self._compact_text(phase_value, 16),
            "text": text,
        }
        record["public_reveals"].append(entry)
        del record["public_reveals"][:-4]
        self._append_verified_fact(text)

    def _maybe_register_verified_alignment(self, player_id: str, record: Mapping[str, Any]) -> None:
        if not player_id:
            return
        claim_role = self._latest_claim_role(record)
        revealed_role = self._latest_reveal_role(record)
        if claim_role and revealed_role and claim_role == revealed_role:
            note = f"{player_id}:{claim_role} 已被公开信息验证"
            aligned = record.setdefault("verified_alignment", [])
            if note not in [self._compact_text(item.get("text"), 80) for item in aligned if isinstance(item, Mapping)]:
                aligned.append({"text": note})
                del aligned[:-4]
                self._add_evidence(player_id, "revealed_alignment", note, record.get("last_seen_round"), -2)
        if player_id and any(
            self._compact_text(item.get("text"), 48) == f"{player_id} 警长信息已确认"
            for item in record.get("verified_alignment", [])
            if isinstance(item, Mapping)
        ):
            return
        if "警长" in self._latest_public_label(record):
            aligned = record.setdefault("verified_alignment", [])
            note = f"{player_id} 警长信息已确认"
            aligned.append({"text": note})
            del aligned[:-4]
            self._add_evidence(player_id, "revealed_alignment", note, record.get("last_seen_round"), -2)

    def _add_claim_contradictions(
        self, speaker: str, record: Mapping[str, Any], role_hint: str, claim_text: str
    ) -> None:
        if not role_hint:
            return
        role_claims = [self._compact_text(item, 24) for item in record.get("role_claims", [])]
        unique_claims = [item for item in role_claims if item]
        if len(set(unique_claims)) > 1:
            contradiction = {
                "text": f"{speaker} 自称出现冲突：{ ' / '.join(sorted(set(unique_claims))) }",
            }
            contradictions = record.setdefault("contradictions", [])
            contradictions.append(contradiction)
            del contradictions[:-4]
            self._add_evidence(speaker, "contradiction_evidence", contradiction["text"], record.get("last_seen_round"), 3)
        if claim_text and self._claim_implies_alignment_conflict(role_hint, claim_text):
            contradiction = {
                "text": f"{speaker} 的自述与公开信息可能冲突：{claim_text}",
            }
            contradictions = record.setdefault("contradictions", [])
            contradictions.append(contradiction)
            del contradictions[:-4]
            self._add_evidence(speaker, "contradiction_evidence", contradiction["text"], record.get("last_seen_round"), 2)

    def _claim_implies_alignment_conflict(self, role_hint: str, claim_text: str) -> bool:
        del role_hint, claim_text
        return False

    def _append_verified_fact(self, fact: str) -> None:
        fact = self._compact_text(fact, 80)
        if not fact:
            return
        facts = self._working_memory.setdefault("verified_facts", [])
        if fact not in facts:
            facts.append(fact)
        del facts[:-10]

    def _last_claim_target(self, record: Mapping[str, Any]) -> str:
        claims = record.get("claims", [])
        for item in reversed(claims):
            if isinstance(item, Mapping):
                target = self._compact_text(item.get("target_id"), 24)
                if target:
                    return target
        return ""

    def _latest_claim_role(self, record: Mapping[str, Any]) -> str:
        claims = record.get("claims", [])
        for item in reversed(claims):
            if isinstance(item, Mapping):
                role = self._compact_text(item.get("role_hint"), 24)
                if role:
                    return role
        return ""

    def _latest_public_label(self, record: Mapping[str, Any]) -> str:
        reveals = record.get("public_reveals", [])
        for item in reversed(reveals):
            if isinstance(item, Mapping):
                text = self._compact_text(item.get("text"), 48)
                if text:
                    return text
        deaths = record.get("death_info", [])
        for item in reversed(deaths):
            if isinstance(item, Mapping):
                text = self._compact_text(item.get("text"), 48)
                if text:
                    return text
        return ""

    def _latest_reveal_role(self, record: Mapping[str, Any]) -> str:
        reveals = record.get("public_reveals", [])
        for item in reversed(reveals):
            if isinstance(item, Mapping):
                role = self._compact_text(item.get("role_hint"), 24)
                if role:
                    return role
        deaths = record.get("death_info", [])
        for item in reversed(deaths):
            if isinstance(item, Mapping):
                role = self._compact_text(item.get("role_hint"), 24)
                if role:
                    return role
        return ""

    def _known_player_ids(self) -> set[str]:
        known: set[str] = set(self._working_memory.get("players", {}).keys())
        for bucket_key in ("claim_log", "vote_log", "public_reveals", "death_log", "verified_facts"):
            for item in self._working_memory.get(bucket_key, []):
                known.update(self._extract_player_ids(item))
        known.update(self._working_memory.get("target_pressure", {}).keys())
        known.discard("")
        return known

    def _claim_pressure_weight(self) -> int:
        return 3 if self._working_memory.get("late_mode") else 2

    def _vote_pressure_weight(self) -> int:
        return 2 if self._working_memory.get("late_mode") else 1

    def _reveal_pressure_weight(self, role_hint: str) -> int:
        if self._is_wolf_role(role_hint):
            return 8 if self._working_memory.get("late_mode") else 6
        if self._is_good_role(role_hint):
            return -5 if self._working_memory.get("late_mode") else -4
        return 2 if self._working_memory.get("late_mode") else 1

    def _death_pressure_weight(self, role_hint: str) -> int:
        if self._is_wolf_role(role_hint):
            return 8 if self._working_memory.get("late_mode") else 6
        if self._is_good_role(role_hint):
            return -5 if self._working_memory.get("late_mode") else -4
        return 2 if self._working_memory.get("late_mode") else 1

    def _is_wolf_role(self, role_text: str) -> bool:
        lowered = role_text.lower()
        return any(
            needle in lowered or needle in role_text
            for needle in ("狼人", "wolf", "werewolf", "狼")
        )

    def _is_good_role(self, role_text: str) -> bool:
        lowered = role_text.lower()
        return any(
            needle in lowered or needle in role_text
            for needle in ("好人", "平民", "villager", "民", "猎人", "女巫", "守卫", "预言家", "警长")
        )

    def _update_suspicion_scores(self, summary: Mapping[str, Any]) -> None:
        del summary
        self._recompute_suspicion_scores()

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
        root_lower = {str(k).lower(): k for k in packet}
        summary: dict[str, Any] = {
            "round": self._first_matching_value(packet, root_lower, ("round", "day", "turn")),
            "phase": self._first_matching_value(packet, root_lower, ("phase", "public_phase", "stage")),
            "alive_count": None, "alive_ids": [], "deaths": [], "sheriff": [],
            "claims": [], "votes": [], "public_reveals": [],
        }
        # game/public_state 是受信任的状态包装层；其余未知字段不下钻。
        for wrapper in ("game", "public_state"):
            original = root_lower.get(wrapper)
            child = packet.get(original) if original is not None else None
            if isinstance(child, Mapping):
                child_lower = {str(k).lower(): k for k in child}
                if summary["round"] is None:
                    summary["round"] = self._first_matching_value(child, child_lower, ("round", "day", "turn"))
                if summary["phase"] is None:
                    summary["phase"] = self._first_matching_value(child, child_lower, ("phase", "public_phase", "stage"))
        self._walk_public_packet(packet, summary, "root")
        # 先把批次的时间上下文写入事件，再去重；否则完整同步在不同轮次
        # 可能因缺少 round/phase 而被错误合并。
        for key in ("deaths", "sheriff", "claims", "votes", "public_reveals"):
            for event in summary[key]:
                if isinstance(event, Mapping):
                    event.setdefault("round", summary["round"])
                    event.setdefault("phase", summary["phase"])
            summary[key] = self._dedupe_events(summary[key])[:12]
        summary["alive_count"] = self._extract_alive_count(packet)
        summary["alive_ids"] = self._extract_alive_ids(packet)
        return summary

    def _walk_public_packet(self, node: object, summary: dict[str, Any], source: str = "root") -> None:
        """只沿着有公开事件语义的容器走；不扫描 players/assignment/description 等状态。"""
        if not isinstance(node, Mapping):
            if isinstance(node, list):
                for index, item in enumerate(node[:64]):
                    self._walk_public_packet(item, summary, f"{source}[{index}]")
            return
        lower = {str(key).lower(): key for key in node.keys()}
        aliases = {
            "death": ("death", "deaths", "dead", "eliminated", "elimination"),
            "sheriff": ("sheriff_event", "sheriff", "badge_event", "badge", "captain"),
            "claim": ("speech", "speeches", "statement", "statements", "claims", "public_claims", "public_speech", "public_statements", "claim_event", "claim_events"),
            "reveal": ("reveal", "reveals", "public_reveal", "public_reveals", "flip", "flips", "role_reveal"),
            "vote": ("vote", "votes", "public_vote", "public_votes", "ballot", "ballots"),
        }
        for kind, keys in aliases.items():
            for key in keys:
                original = lower.get(key)
                if original is not None:
                    self._append_event_labels(summary["deaths" if kind == "death" else "sheriff" if kind == "sheriff" else "claims" if kind == "claim" else "public_reveals" if kind == "reveal" else "votes"], node[original], kind=kind, source_key=f"{source}.{key}")
        # 事件总线允许有限递归；普通字段和未知嵌套对象绝不作为事件来源。
        for key in ("public_events", "public_event_log", "event_log", "events"):
            original = lower.get(key)
            if original is not None:
                self._walk_explicit_event_bus(node[original], summary, f"{source}.{key}")
        original = lower.get("public_state")
        if original is not None and isinstance(node[original], Mapping):
            self._walk_public_packet(node[original], summary, f"{source}.public_state")

    def _walk_explicit_event_bus(self, value: object, summary: dict[str, Any], source: str) -> None:
        """读取带 event_type/type/kind 的事件总线；未知类型只被忽略。"""
        if isinstance(value, list):
            for index, item in enumerate(value[:64]):
                self._walk_explicit_event_bus(item, summary, f"{source}[{index}]")
            return
        if not isinstance(value, Mapping):
            return
        raw_type = self._first_key_text(value, ("event_type", "type", "kind" )).lower()
        type_aliases = {
            "death": "death", "death_notice": "death", "elimination": "death", "dead": "death",
            "reveal": "reveal", "role_reveal": "reveal", "flip": "reveal",
            "claim": "claim", "speech": "claim", "public_speech": "claim", "statement": "claim",
            "vote": "vote", "day_vote": "vote", "public_vote": "vote", "ballot": "vote",
            "sheriff": "sheriff", "badge": "sheriff", "badge_event": "sheriff", "sheriff_transfer": "sheriff",
        }
        kind = type_aliases.get(raw_type)
        if not kind:
            return
        bucket_name = {"death": "deaths", "reveal": "public_reveals", "claim": "claims", "vote": "votes", "sheriff": "sheriff"}[kind]
        self._append_event_labels(summary[bucket_name], value, kind=kind, source_key=source)

    def _append_event_labels(self, bucket: list[dict[str, Any]], value: object, *, kind: str, source_key: str = "") -> None:
        extractor = {"vote": self._extract_vote_events, "claim": self._extract_claim_events, "sheriff": self._extract_sheriff_events, "reveal": self._extract_reveal_events, "death": self._extract_death_events}[kind]
        for label in extractor(value):
            if label:
                label["source_key"] = source_key
                label["event_type"] = kind
                if label not in bucket:
                    bucket.append(label)

    def _extract_death_events(self, value: object) -> list[dict[str, Any]]:
        """解析显式死亡记录；纯字符串不是可信的事件记录。"""
        events: list[dict[str, Any]] = []
        if isinstance(value, Mapping):
            player = self._first_key_text(value, ("player_id", "dead_player_id", "victim_id", "eliminated_player_id"))
            reason = self._first_key_text(value, ("reason", "cause"))
            role_hint = self._first_key_text(value, ("revealed_role", "role_hint", "role"))
            text = self._first_key_text(value, ("text", "content"))
            if player:
                events.append({
                    "player_id": player,
                    "reason": reason,
                    "role_hint": self._normalize_role_hint(role_hint),
                    "text": text or (f"{player}({reason})" if reason else player),
                })
        elif isinstance(value, list):
            for item in value:
                events.extend(self._extract_death_events(item))
        return events

    def _extract_sheriff_events(self, value: object) -> list[dict[str, Any]]:
        """只接受带 holder/player 字段的警徽事件，不从文本猜 holder。"""
        events: list[dict[str, Any]] = []
        if isinstance(value, Mapping):
            holder = self._first_key_text(value, ("sheriff_id", "badge_holder", "holder_id", "player_id"))
            text = self._first_key_text(value, ("text", "content"))
            if holder:
                events.append({"player_id": holder, "text": text or f"{holder}(警长)"})
        elif isinstance(value, list):
            for item in value:
                events.extend(self._extract_sheriff_events(item))
        return events

    def _extract_claim_events(self, value: object) -> list[dict[str, Any]]:
        """声明必须有显式 speaker 和 claim/text 字段；不从自然语言抽 target/role。"""
        events: list[dict[str, Any]] = []
        if isinstance(value, Mapping):
            player = self._first_key_text(value, ("speaker_id", "speaker", "player_id"))
            claim = self._first_key_text(value, ("claim", "claim_text", "self_claim", "statement", "claimed_role", "text", "message", "content"))
            target = self._first_key_text(value, ("target_id", "accused_id", "pointed_player"))
            role_hint = self._first_key_text(value, ("role_hint", "claimed_role", "role_claim"))
            stance = self._first_key_text(value, ("stance", "position", "supports", "opposes"))
            if player and claim:
                events.append({
                    "player_id": player,
                    "claim_text": claim,
                    "target_id": target,
                    "role_hint": self._normalize_role_hint(role_hint),
                    "stance": stance,
                    "text": claim,
                })
        elif isinstance(value, list):
            for item in value:
                events.extend(self._extract_claim_events(item))
        return events

    def _extract_vote_events(self, value: object) -> list[dict[str, Any]]:
        """只有显式 voter + target/abstain 才进入票型，声明文本不会升级为 vote。"""
        events: list[dict[str, Any]] = []
        if isinstance(value, Mapping):
            voter = self._first_key_text(value, ("voter_id", "voter", "player_id"))
            target = self._first_key_text(value, ("target_id", "vote_target", "candidate", "choice_id"))
            abstain = value.get("abstain", value.get("pass", value.get("skipped")))
            text = self._first_key_text(value, ("text", "content", "vote_text"))
            if voter and target:
                events.append({"player_id": voter, "target_id": target, "text": text or f"{voter}->{target}"})
            elif voter and abstain is True:
                events.append({"player_id": voter, "target_id": "pass", "text": text or f"{voter}->pass"})
        elif isinstance(value, list):
            for item in value:
                events.extend(self._extract_vote_events(item))
        return events

    def _extract_reveal_events(self, value: object) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        if isinstance(value, Mapping):
            player = self._first_key_text(value, ("player_id", "revealed_player_id"))
            role_hint = self._first_key_text(value, ("role", "revealed_role", "role_hint", "alignment"))
            text = self._first_key_text(value, ("text", "content"))
            if player and role_hint:
                events.append({
                    "player_id": player,
                    "role_hint": self._normalize_role_hint(role_hint),
                    "text": text or f"{player}:{role_hint}",
                })
        elif isinstance(value, list):
            for item in value:
                events.extend(self._extract_reveal_events(item))
        return events

    def _dedupe_events(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        seen: set[str] = set()
        result: list[dict[str, Any]] = []
        for item in items:
            event_type = self._compact_text(item.get("event_type"), 16)
            subject = self._compact_text(item.get("player_id"), 24)
            target = self._compact_text(item.get("target_id"), 24)
            text = self._normalize_event_text(item.get("text") or item.get("claim_text"))
            key = "|".join((event_type, self._compact_text(item.get("round"), 12), self._compact_text(item.get("phase"), 16), subject, target, text))
            if not text and not subject:
                continue
            if key in seen:
                continue
            seen.add(key)
            result.append(item)
        return result

    def _format_summary_line(self, summary: Mapping[str, Any]) -> str:
        parts: list[str] = []
        round_value = summary.get("round")
        phase_value = summary.get("phase")
        if round_value is not None:
            parts.append(f"R{self._compact_text(round_value, 12)}")
        if phase_value is not None:
            parts.append(self._compact_text(phase_value, 16))
        if summary.get("deaths"):
            parts.append("死亡:" + "、".join(self._event_texts(summary["deaths"], 3)))
        if summary.get("sheriff"):
            parts.append("警长:" + "、".join(self._event_texts(summary["sheriff"], 2)))
        if summary.get("public_reveals"):
            parts.append("翻牌:" + "、".join(self._event_texts(summary["public_reveals"], 3)))
        if summary.get("claims"):
            parts.append("声明:" + "；".join(self._event_texts(summary["claims"], 3)))
        if summary.get("votes"):
            parts.append("票型:" + "；".join(self._event_texts(summary["votes"], 3)))
        return " | ".join(parts)

    def _event_texts(self, items: list[dict[str, Any]], limit: int) -> list[str]:
        return [self._compact_text(item.get("text"), 40) for item in items[:limit] if self._compact_text(item.get("text"), 40)]

    def _summarize_dialogue(self, dialogue: object, focus_players: list[str]) -> list[str]:
        if not isinstance(dialogue, list):
            return []
        focus = {player for player in focus_players if player}
        lines: list[str] = []
        fallback: list[str] = []
        for item in dialogue[-12:]:
            if isinstance(item, Mapping):
                speaker = self._first_key_text(item, ("player_id", "player", "speaker_id", "speaker", "name", "id"))
                text = self._first_key_text(item, ("text", "content", "message", "say"))
                if speaker and text:
                    line = f"{speaker}:{self._compact_text(text, 48)}"
                else:
                    line = self._compact_text(item, 70)
            else:
                line = self._compact_text(item, 70)
            if not line:
                continue
            fallback.append(line)
            if self._dialogue_matches_focus(line, focus):
                lines.append(line)
        if not lines:
            lines = fallback
        return lines[:5]

    def _dialogue_matches_focus(self, line: str, focus: set[str]) -> bool:
        if not focus:
            return True
        lowered = line.lower()
        if any(player.lower() in lowered for player in focus):
            return True
        return any(keyword in line for keyword in ("票", "查杀", "身份", "翻牌", "警长", "出", "跟票"))

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
        # 事件字段必须在当前记录这一层且为标量；不从嵌套对象、列表或摘要文本猜主体。
        lowered = {str(key).lower(): key for key in node}
        for key in keys:
            original = lowered.get(key.lower())
            if original is None:
                continue
            value = node[original]
            if isinstance(value, (Mapping, list, tuple, set)):
                continue
            text = self._compact_text(value, 48)
            if text:
                return text
        return ""

    def _split_claim(self, claim: str) -> tuple[str, str]:
        if ":" not in claim:
            return claim, ""
        head, tail = claim.split(":", 1)
        return head.strip(), tail.strip()

    def _split_vote(self, text: str) -> tuple[str, str]:
        if "->" in text:
            left, right = text.split("->", 1)
            return left.strip(), right.strip()
        if "投" in text and self._first_player_from_text(text):
            voter = self._first_player_from_text(text)
            target = self._first_target_from_text(text)
            return voter, target
        return "", ""

    def _extract_primary_target(self, speaker: str, text: str) -> str:
        if not text:
            return ""
        ids = self._extract_player_ids(text)
        for player_id in ids:
            if player_id != speaker:
                return player_id
        return ""

    def _first_player_from_text(self, text: str) -> str:
        ids = self._extract_player_ids(text)
        return ids[0] if ids else ""

    def _first_target_from_text(self, text: str) -> str:
        ids = self._extract_player_ids(text)
        if len(ids) >= 2:
            return ids[1]
        return ids[0] if ids else ""

    def _extract_player_ids(self, value: object) -> list[str]:
        if value is None:
            return []
        if isinstance(value, Mapping):
            found: list[str] = []
            for item in value.values():
                found.extend(self._extract_player_ids(item))
            return self._dedupe_preserve_order(found)
        if isinstance(value, list):
            found: list[str] = []
            for item in value:
                found.extend(self._extract_player_ids(item))
            return self._dedupe_preserve_order(found)
        text = self._compact_text(value, 120)
        if not text:
            return []
        return self._dedupe_preserve_order(_PLAYER_TOKEN.findall(text))

    def _extract_alive_ids(self, packet: object) -> list[str]:
        if not isinstance(packet, Mapping):
            return []
        for key in ("alive_ids", "living_ids", "alive_players", "living_players"):
            value = self._find_value_by_key(packet, key)
            if isinstance(value, list):
                return self._dedupe_preserve_order([str(item) for item in value if self._looks_like_player_id(str(item))])
            if isinstance(value, Mapping):
                return self._dedupe_preserve_order([str(item) for item in value.keys() if self._looks_like_player_id(str(item))])
        return []

    def _extract_alive_count(self, packet: object) -> int | None:
        if not isinstance(packet, Mapping):
            return None
        for key in (
            "alive_count",
            "living_count",
            "alive_ids",
            "living_ids",
            "survivor_count",
            "remaining_count",
            "players_alive",
            "alive_players_count",
        ):
            value = self._find_value_by_key(packet, key)
            if isinstance(value, int) and value >= 0:
                return value
            if isinstance(value, str) and value.isdigit():
                return int(value)
            if isinstance(value, list):
                ids = self._extract_player_ids(value)
                if ids:
                    return len(ids)
        return None

    def _find_value_by_key(self, node: object, wanted_key: str) -> object | None:
        if isinstance(node, Mapping):
            for key, value in node.items():
                if str(key).lower() == wanted_key.lower():
                    return value
                nested = self._find_value_by_key(value, wanted_key)
                if nested is not None:
                    return nested
        elif isinstance(node, list):
            for item in node:
                nested = self._find_value_by_key(item, wanted_key)
                if nested is not None:
                    return nested
        return None

    def _normalize_role_hint(self, text: str) -> str:
        if not text:
            return ""
        lowered = text.lower()
        candidates = (
            ("狼人", "狼人"),
            ("werewolf", "狼人"),
            ("wolf", "狼人"),
            ("好人", "好人"),
            ("平民", "平民"),
            ("villager", "平民"),
            ("民", "平民"),
            ("预言家", "预言家"),
            ("seer", "预言家"),
            ("女巫", "女巫"),
            ("witch", "女巫"),
            ("猎人", "猎人"),
            ("hunter", "猎人"),
            ("守卫", "守卫"),
            ("guard", "守卫"),
            ("警长", "警长"),
            ("sheriff", "警长"),
        )
        for needle, normalized in candidates:
            if needle.lower() in lowered or needle in text:
                return normalized
        return ""

    def _looks_like_player_id(self, text: str) -> bool:
        return bool(text and _PLAYER_TOKEN.fullmatch(text))

    def _claim_has_accusation(self, claim_text: str) -> bool:
        keywords = (
            "查杀",
            "狼",
            "出",
            "投",
            "票",
            "怀疑",
            "认狼",
            "打",
            "冲",
            "跟票",
        )
        return any(keyword in claim_text for keyword in keywords)

    def _dedupe_preserve_order(self, items: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for item in items:
            if item and item not in seen:
                seen.add(item)
                result.append(item)
        return result

    def _contains_role_like_word(self, text: str) -> bool:
        if not text:
            return False
        return bool(self._normalize_role_hint(text))

    def _split_claim_target_text(self, claim_text: str, speaker: str) -> str:
        if not claim_text:
            return ""
        if self._claim_has_accusation(claim_text):
            target = self._extract_primary_target(speaker, claim_text)
            if target:
                return target
        return ""

    def _tail_text(self, value: object, limit: int = 80) -> str:
        return self._compact_text(value, limit)

    def _normalize_event_text(self, value: object) -> str:
        text = self._compact_text(value, 120)
        return re.sub(r"\s+", " ", text).strip().casefold()

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
            "- 证据优先级：显式翻牌/死亡/票型/警徽，其次是有时间线的声明冲突，最后才是话术风格。\n"
            "- score 只是有界索引，必须回看 evidence_reasons；事实、声明、推测分开。\n"
            "- 只在当前合法存活目标中排序；死亡玩家只能作为历史时间线证据。\n"
            "- 死后查验不自动等于铁狼；不要求守卫证明未公开的夜间行动。\n"
            "- 对跳比较验人链、警徽、票型、死亡顺序，并保留至少一个备选解释。"
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

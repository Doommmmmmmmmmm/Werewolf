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

_DEATH_KEYWORDS = ("dead", "death", "died", "kill", "eliminat", "lynch", "死亡", "出局")
_VOTE_KEYWORDS = ("vote", "ballot", "tally", "poll", "票", "归票")
_CLAIM_KEYWORDS = ("claim", "claimed", "claiming", "identity", "role", "身份", "跳", "自称")
_SHERIFF_KEYWORDS = ("sheriff", "chief", "mayor", "警长", "警徽")
_SPEECH_KEYWORDS = ("speech", "dialogue", "message", "content", "text", "utterance", "say", "发言", "说")
_PLAYER_LABEL_KEYS = (
    "player_id",
    "playerid",
    "player",
    "name",
    "nickname",
    "nick",
    "speaker_id",
    "speaker",
    "author",
    "user",
    "target_id",
    "targetid",
    "target",
    "voter_id",
    "voterid",
    "voter",
    "from",
    "source",
    "holder",
)

CURRENT_ROUND_DIALOGUE_TOOL: dict[str, Any] = {
    "name": CURRENT_ROUND_DIALOGUE_TOOL_NAME,
    "description": "读取当前昼夜轮次中、当前玩家依法可见的已发生发言。",
    "parameters": {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    },
}


def _short_text(value: object, limit: int = 120) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        text = "true" if value else "false"
    elif isinstance(value, (int, float)):
        text = str(value)
    else:
        text = str(value)
    text = " ".join(text.split()).strip()
    if not text:
        return None
    return text[:limit]


def _append_unique(items: list[str], value: str | None, *, limit: int) -> None:
    if not value:
        return
    value = value.strip()
    if not value:
        return
    if value in items:
        items.remove(value)
    items.append(value)
    if len(items) > limit:
        del items[:-limit]


def _first_scalar(mapping: Mapping[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        if key in mapping:
            text = _short_text(mapping.get(key), 80)
            if text:
                return text
    return None


def _unique_strings(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result


def _split_record(record: str, separator: str) -> tuple[str | None, str | None]:
    if separator not in record:
        return None, None
    left, right = record.split(separator, 1)
    left = left.strip()
    right = right.strip()
    return (left or None), (right or None)


def _collect_compact_records(value: Any, kind: str, *, depth: int = 0) -> list[str]:
    if depth > 4 or value is None:
        return []
    records: list[str] = []
    if isinstance(value, Mapping):
        if kind == "vote":
            voter = _first_scalar(value, ("voter_id", "voterid", "voter", "speaker_id", "speaker", "player_id", "player", "from", "source"))
            target = _first_scalar(value, ("target_id", "targetid", "target", "vote_target", "choice", "to"))
            if voter and target:
                records.append(f"{voter}->{target}")
                return records
        elif kind == "claim":
            player = _first_scalar(value, ("player_id", "playerid", "player", "speaker_id", "speaker", "author", "name"))
            claim = _first_scalar(value, ("claimed_role", "claim", "role", "identity", "claimed_identity", "self_claim"))
            if player and claim:
                records.append(f"{player}: {claim}")
                return records
        elif kind == "death":
            player = _first_scalar(value, ("player_id", "playerid", "player", "target_id", "targetid", "target", "victim", "dead", "name"))
            if player:
                records.append(player)
                return records
        elif kind == "sheriff":
            player = _first_scalar(value, ("sheriff_id", "sheriffid", "player_id", "player", "holder", "name"))
            if player:
                records.append(player)
                return records
        elif kind == "statement":
            speaker = _first_scalar(value, ("speaker_id", "speaker", "player_id", "player", "author", "name", "user"))
            text = _first_scalar(value, ("text", "content", "message", "dialogue", "utterance", "speech"))
            if speaker and text:
                records.append(f"{speaker}: {text}")
                return records
            if text:
                records.append(text)
                return records

        for nested in value.values():
            records.extend(_collect_compact_records(nested, kind, depth=depth + 1))
        return records

    if isinstance(value, (list, tuple, set)):
        for item in list(value)[:8]:
            records.extend(_collect_compact_records(item, kind, depth=depth + 1))
        return records

    text = _short_text(value, 160)
    if text:
        records.append(text)
    return records


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
        self._public_memory: dict[str, Any] = {
            "last_round": None,
            "last_phase": None,
            "current_sheriff": None,
            "sheriff_history": [],
            "dead_players": [],
            "recent_vote_records": [],
            "recent_claim_records": [],
            "recent_public_statements": [],
            "claims_by_player": {},
        }

    async def observe(self, sync_packet: dict[str, Any]) -> None:
        """接收公开同步，并维护一层极轻量的公共记忆。"""

        facts = self._extract_public_facts(sync_packet)
        self._merge_public_memory(facts)

    async def decide(self, turn_packet: dict[str, Any]) -> dict[str, Any]:
        if turn_packet["self"]["player_id"] != self.player_id:
            raise ValueError("行动包被发送给了错误玩家")
        private = turn_packet["private_information"]
        if str(private.get("role")) != self.profile.role:
            raise ValueError("Task-Agent 的角色档案与行动包身份不一致")

        system = self._system_prompt(private)
        current_round_dialogue = turn_packet.get("tool_context", {}).get("current_round_dialogue") or []
        compact_public_state = self._extract_public_facts(turn_packet.get("public_state"))
        memory_summary = self._memory_snapshot()
        analysis_hints = self._build_analysis_hints(compact_public_state, current_round_dialogue)
        prompt = {
            "turn_meta": {
                "round": turn_packet["game"].get("round"),
                "phase": turn_packet["game"].get("public_phase", turn_packet["game"].get("phase")),
            },
            "request": {
                "request_id": turn_packet["request"].get("request_id"),
                "player_id": turn_packet["request"].get("player_id"),
                "phase": turn_packet["request"].get("phase"),
                "channel": turn_packet["request"].get("channel"),
                "allowed_actions": turn_packet["request"].get("allowed_actions"),
            },
            "public_state": compact_public_state,
            "memory_summary": memory_summary,
            "analysis_hints": analysis_hints,
            "current_round_dialogue": self._extract_public_facts(current_round_dialogue),
            "history_policy": {
                "default_context": "compact_public_state_and_memory_summary",
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

    def _system_prompt(self, private: Mapping[str, Any]) -> str:
        base = render_prompt(
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
        role_name = str(private.get("role"))
        if "villager" in role_name.lower() or role_name == "平民":
            base += (
                "\n\n【平民决策模板】\n"
                "先分事实、玩家声明和猜测。每轮至少比较两个候选。\n"
                "发言必须点到具体人名，并引用至少一个具体公共事实：票型、死亡顺序、身份声明或当前轮公开发言。\n"
                "如果出现新死亡或新身份声明，先更新上一轮判断，再给出本轮结论。\n"
                "投票前必须说明为什么不是另一个候选；不要连续多轮使用同一套空泛话术。\n"
            )
        return base

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

    def _merge_public_memory(self, facts: dict[str, Any]) -> None:
        memory = self._public_memory
        if facts.get("round") is not None:
            memory["last_round"] = facts["round"]
        if facts.get("phase") is not None:
            memory["last_phase"] = facts["phase"]

        for player in facts.get("dead_players", []):
            _append_unique(memory["dead_players"], player, limit=12)
        for record in facts.get("sheriff_records", []):
            sheriff = self._extract_label_from_record(record)
            if sheriff:
                if sheriff != memory.get("current_sheriff"):
                    _append_unique(memory["sheriff_history"], sheriff, limit=6)
                memory["current_sheriff"] = sheriff
        for record in facts.get("vote_records", []):
            _append_unique(memory["recent_vote_records"], record, limit=8)
        for record in facts.get("claim_records", []):
            _append_unique(memory["recent_claim_records"], record, limit=8)
            player, claim = _split_record(record, ":")
            if player and claim:
                claims_by_player = memory["claims_by_player"].setdefault(player, [])
                _append_unique(claims_by_player, claim, limit=4)
        for record in facts.get("statement_records", []):
            _append_unique(memory["recent_public_statements"], record, limit=8)

    def _memory_snapshot(self) -> dict[str, Any]:
        memory = self._public_memory
        recent_claim_players: list[str] = []
        for record in memory["recent_claim_records"][-6:]:
            player, _ = _split_record(record, ":")
            if player and player not in recent_claim_players:
                recent_claim_players.append(player)
        if not recent_claim_players:
            recent_claim_players = [
                player for player in sorted(memory["claims_by_player"].keys())[:4]
            ]
        claims_by_player = {
            player: list(memory["claims_by_player"].get(player, []))
            for player in recent_claim_players
            if memory["claims_by_player"].get(player)
        }
        return {
            "last_round": memory["last_round"],
            "last_phase": memory["last_phase"],
            "current_sheriff": memory["current_sheriff"],
            "sheriff_history": memory["sheriff_history"][-4:],
            "dead_players": memory["dead_players"][-8:],
            "recent_vote_records": memory["recent_vote_records"][-6:],
            "recent_claim_records": memory["recent_claim_records"][-6:],
            "recent_public_statements": memory["recent_public_statements"][-5:],
            "claims_by_player": claims_by_player,
        }

    def _build_analysis_hints(
        self,
        current_public_facts: dict[str, Any],
        current_round_dialogue: Any,
    ) -> dict[str, Any]:
        memory = self._memory_snapshot()
        dialogue_facts = self._extract_public_facts(current_round_dialogue)
        combined_vote_records = _unique_strings(
            list(memory["recent_vote_records"])
            + list(current_public_facts.get("vote_records", []))
            + list(dialogue_facts.get("vote_records", []))
        )
        combined_claim_records = _unique_strings(
            list(memory["recent_claim_records"])
            + list(current_public_facts.get("claim_records", []))
            + list(dialogue_facts.get("claim_records", []))
        )
        combined_statement_records = _unique_strings(
            list(memory["recent_public_statements"])
            + list(current_public_facts.get("statement_records", []))
            + list(dialogue_facts.get("statement_records", []))
        )
        dead_players = _unique_strings(
            list(memory["dead_players"]) + list(current_public_facts.get("dead_players", []))
        )
        suspicious, trusted, contradictions, vote_pressure = self._rank_candidates(
            dead_players=dead_players,
            current_sheriff=memory["current_sheriff"],
            vote_records=combined_vote_records,
            claim_records=combined_claim_records,
            statement_records=combined_statement_records,
        )
        latest_key_vote = combined_vote_records[-1] if combined_vote_records else None
        latest_public_claim = combined_claim_records[-1] if combined_claim_records else None
        return {
            "current_public_facts": {
                "round": current_public_facts.get("round"),
                "phase": current_public_facts.get("phase"),
                "current_sheriff": memory["current_sheriff"],
                "dead_players": current_public_facts.get("dead_players", [])[-4:],
                "sheriff_records": current_public_facts.get("sheriff_records", [])[-2:],
                "vote_records": current_public_facts.get("vote_records", [])[-4:],
                "claim_records": current_public_facts.get("claim_records", [])[-4:],
                "statement_records": current_public_facts.get("statement_records", [])[-4:],
            },
            "latest_key_vote": latest_key_vote,
            "latest_public_claim": latest_public_claim,
            "suspicious_candidates": suspicious,
            "trusted_candidates": trusted,
            "recent_contradictions": contradictions,
            "vote_pressure": vote_pressure,
        }

    def _rank_candidates(
        self,
        *,
        dead_players: list[str],
        current_sheriff: str | None,
        vote_records: list[str],
        claim_records: list[str],
        statement_records: list[str],
    ) -> tuple[list[str], list[str], list[str], dict[str, int]]:
        dead_set = set(dead_players)
        claim_map: dict[str, list[str]] = {}
        vote_pressure: dict[str, int] = {}
        statement_mentions: dict[str, int] = {}
        contradictions: list[str] = []

        for record in claim_records:
            player, claim = _split_record(record, ":")
            if player and claim:
                claims = claim_map.setdefault(player, [])
                if claim not in claims:
                    claims.append(claim)
        for record in vote_records:
            voter, target = _split_record(record, "->")
            if target:
                vote_pressure[target] = vote_pressure.get(target, 0) + 1
            if voter:
                statement_mentions[voter] = statement_mentions.get(voter, 0) + 1
        for record in statement_records:
            speaker, text = _split_record(record, ":")
            if speaker and text:
                statement_mentions[speaker] = statement_mentions.get(speaker, 0) + 1

        suspicious_scores: dict[str, int] = {}
        trusted_scores: dict[str, int] = {}

        def add_score(scores: dict[str, int], player: str, delta: int) -> None:
            if not player or player in dead_set:
                return
            scores[player] = scores.get(player, 0) + delta

        for player, claims in claim_map.items():
            unique_claims = _unique_strings(claims)
            if len(unique_claims) >= 2:
                add_score(suspicious_scores, player, 3 + len(unique_claims) - 1)
                contradictions.append(f"{player}: {' / '.join(unique_claims[:3])}")
            elif len(unique_claims) == 1:
                add_score(trusted_scores, player, 1)

        for target, count in vote_pressure.items():
            add_score(suspicious_scores, target, min(3, count))

        for player, count in statement_mentions.items():
            if player in claim_map and len(_unique_strings(claim_map[player])) == 1:
                add_score(trusted_scores, player, min(2, count))

        if current_sheriff:
            add_score(trusted_scores, current_sheriff, 1)

        suspicious = [
            player
            for player, _score in sorted(
                suspicious_scores.items(), key=lambda item: (-item[1], item[0])
            )[:3]
        ]
        trusted = [
            player
            for player, _score in sorted(
                trusted_scores.items(), key=lambda item: (-item[1], item[0])
            )[:2]
        ]
        if not suspicious:
            suspicious = [
                player
                for player, _count in sorted(
                    vote_pressure.items(), key=lambda item: (-item[1], item[0])
                )[:3]
                if player not in dead_set
            ]
        if not trusted:
            trusted = [
                player
                for player, claims in sorted(claim_map.items(), key=lambda item: (len(item[1]), item[0]))
                if player not in dead_set and len(_unique_strings(claims)) == 1
            ][:2]
        return suspicious, trusted, contradictions[:4], dict(
            sorted(vote_pressure.items(), key=lambda item: (-item[1], item[0]))[:5]
        )

    def _extract_public_facts(self, node: Any) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "round": None,
            "phase": None,
            "dead_players": [],
            "sheriff_records": [],
            "vote_records": [],
            "claim_records": [],
            "statement_records": [],
        }

        def walk(value: Any, depth: int = 0) -> None:
            if depth > 4 or value is None:
                return
            if isinstance(value, Mapping):
                lower_items = [(str(key).lower(), item) for key, item in value.items()]
                for key, item in lower_items:
                    if summary["round"] is None and (
                        key == "round" or key.endswith("round") or key in {"day", "night"}
                    ):
                        text = _short_text(item, 32)
                        if text:
                            summary["round"] = text
                    if summary["phase"] is None and (
                        "phase" in key or key == "stage" or key == "turn"
                    ):
                        text = _short_text(item, 32)
                        if text:
                            summary["phase"] = text

                    if any(token in key for token in _DEATH_KEYWORDS):
                        for record in _collect_compact_records(item, "death", depth=depth + 1):
                            _append_unique(summary["dead_players"], record, limit=12)
                    elif any(token in key for token in _SHERIFF_KEYWORDS):
                        for record in _collect_compact_records(item, "sheriff", depth=depth + 1):
                            _append_unique(summary["sheriff_records"], record, limit=6)
                    elif any(token in key for token in _VOTE_KEYWORDS):
                        for record in _collect_compact_records(item, "vote", depth=depth + 1):
                            _append_unique(summary["vote_records"], record, limit=8)
                    elif any(token in key for token in _CLAIM_KEYWORDS):
                        for record in _collect_compact_records(item, "claim", depth=depth + 1):
                            _append_unique(summary["claim_records"], record, limit=8)
                    elif any(token in key for token in _SPEECH_KEYWORDS):
                        for record in _collect_compact_records(item, "statement", depth=depth + 1):
                            _append_unique(summary["statement_records"], record, limit=8)

                speaker = _first_scalar(value, (
                    "speaker",
                    "speaker_id",
                    "author",
                    "user",
                    "player",
                    "player_id",
                    "name",
                ))
                text = _first_scalar(value, (
                    "text",
                    "content",
                    "message",
                    "dialogue",
                    "utterance",
                    "speech",
                ))
                if speaker and text:
                    _append_unique(summary["statement_records"], f"{speaker}: {text}", limit=8)

                event_kind = _first_scalar(value, ("kind", "type", "event"))
                if event_kind:
                    event_kind_l = event_kind.lower()
                    if any(token in event_kind_l for token in _DEATH_KEYWORDS):
                        note = _short_text(event_kind, 80)
                        if note:
                            _append_unique(summary["dead_players"], note, limit=12)
                    elif any(token in event_kind_l for token in _VOTE_KEYWORDS):
                        note = _short_text(event_kind, 80)
                        if note:
                            _append_unique(summary["vote_records"], note, limit=8)
                    elif any(token in event_kind_l for token in _CLAIM_KEYWORDS):
                        note = _short_text(event_kind, 80)
                        if note:
                            _append_unique(summary["claim_records"], note, limit=8)
                    elif any(token in event_kind_l for token in _SPEECH_KEYWORDS):
                        note = _short_text(event_kind, 80)
                        if note:
                            _append_unique(summary["statement_records"], note, limit=8)

                for item in value.values():
                    walk(item, depth + 1)
                return
            if isinstance(value, (list, tuple, set)):
                for item in list(value)[:8]:
                    walk(item, depth + 1)
                return
            text = _short_text(value, 80)
            if text and depth <= 1:
                _append_unique(summary["statement_records"], text, limit=8)

        walk(node)
        return summary

    @staticmethod
    def _extract_label_from_record(record: str) -> str | None:
        record = record.strip()
        if not record:
            return None
        if record.startswith("sheriff="):
            value = record.split("=", 1)[1].strip()
            return value or None
        if "->" in record:
            _, target = _split_record(record, "->")
            return target
        if ":" in record:
            player, _ = _split_record(record, ":")
            return player
        return record

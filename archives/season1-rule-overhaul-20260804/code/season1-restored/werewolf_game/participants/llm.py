"""使用外部大模型的玩家适配器（Season 1 单步版本）。"""

from __future__ import annotations

import asyncio
import inspect
import json
import re
from typing import Any

from .base import Participant
from ..llm.coordinator import ModelRequestCoordinator
from ..prompts import RoleStrategyStore, render_prompt


_CHINESE_CHARACTER = re.compile(r"[\u3400-\u9fff]")
_LATIN_LETTER = re.compile(r"[A-Za-z]")


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
    if action["kind"] in {"speak", "last_words"}:
        text = str(action.get("text", "")).strip()
        if not text:
            return f"{action['kind']} 必须提供非空 text"
        if len(text) > int(allowed["max_chars"]):
            return f"发言超过 max_chars={allowed['max_chars']}"
        if allowed.get("require_chinese") and not _CHINESE_CHARACTER.search(text):
            return "发言必须包含中文"
        if not allowed.get("allow_latin_letters", True) and _LATIN_LETTER.search(text):
            return "发言不能包含英文字母"
    target_ids = allowed.get("target_ids")
    if target_ids:
        if action.get("target_id") not in target_ids:
            return "target_id 必须是 target_ids 中的一项"
    elif action.get("target_id"):
        return "该行动不能提供 target_id"
    return None


class LlmParticipant(Participant):
    """每名 LLM 玩家有独立短期私密笔记和公开状态缓存。"""

    def __init__(
        self,
        *,
        player_id: str,
        model_client: Any,
        persona: str = "",
        strategy_store: RoleStrategyStore | None = None,
        request_coordinator: ModelRequestCoordinator | None = None,
        max_tokens: int = 900,
        max_decision_retries: int = 1,
    ) -> None:
        super().__init__(player_id)
        if not hasattr(model_client, "complete_json"):
            raise ValueError("LlmParticipant 需要具有 complete_json 的模型客户端")
        self.model_client = model_client
        self.persona = persona
        self.strategy_store = strategy_store or RoleStrategyStore()
        self.request_coordinator = request_coordinator
        self.max_tokens = max_tokens
        self.max_decision_retries = max_decision_retries
        self.private_notes: list[str] = []
        self.latest_public_sync: dict[str, Any] | None = None
        self._model_token_usage: dict[str, int] = {
            "successful_response_count": 0,
            "api_attempt_count": 0,
            "reported_usage_response_count": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }

    async def observe(self, sync_packet: dict[str, Any]) -> None:
        self.latest_public_sync = {
            "game": sync_packet["game"],
            "public_rules": sync_packet["public_rules"],
            "public_state": sync_packet["public_state"],
            "public_events": sync_packet["public_events"],
        }

    async def decide(self, turn_packet: dict[str, Any]) -> dict[str, Any]:
        if turn_packet["self"]["player_id"] != self.player_id:
            raise ValueError("行动包被发送给了错误玩家")

        private = turn_packet["private_information"]
        role_profile = self.strategy_store.profile(str(private["role"]))
        system = render_prompt(
            "player_system.txt",
            player_id=self.player_id,
            role=private["role"],
            team=private["team"],
            persona=self.persona,
            role_base=role_profile.base,
            role_strategy=role_profile.strategy,
        )
        prompt = {
            "game": turn_packet["game"],
            "public_rules": turn_packet["public_rules"],
            "self": turn_packet["self"],
            "private_information": private,
            "public_state": turn_packet["public_state"],
            "visible_events": turn_packet["visible_events"],
            "synchronized_public_state": self.latest_public_sync,
            "private_memory_notes": self.private_notes,
            "request": turn_packet["request"],
        }
        feedback = ""
        for _attempt in range(self.max_decision_retries + 1):
            user_content: dict[str, Any] = {
                "instruction": render_prompt(
                    "player_turn_instruction.txt",
                    validation_feedback=(
                        f"上一次输出未通过校验：{feedback}" if feedback else ""
                    ),
                ),
                "packet": prompt,
            }
            raw = await self._complete_json(
                system=system,
                messages=[
                    {
                        "role": "user",
                        "content": json.dumps(user_content, ensure_ascii=False),
                    }
                ],
            )
            self._record_token_usage(raw)
            action = normalize_decision(raw, turn_packet["request"])
            error = decision_error(action, turn_packet["request"])
            if error is None:
                self._remember(raw.get("private_note", raw.get("memory_note")))
                return action
            feedback = error
        raise ValueError("模型多次返回非法行动")

    async def _complete_json(self, **kwargs: Any) -> dict[str, Any]:
        if self.request_coordinator is not None:
            return await self.request_coordinator.complete_json(
                self.model_client, max_tokens=self.max_tokens, **kwargs
            )
        complete_json = self.model_client.complete_json
        if inspect.iscoroutinefunction(complete_json):
            return await complete_json(max_tokens=self.max_tokens, **kwargs)
        return await asyncio.to_thread(complete_json, max_tokens=self.max_tokens, **kwargs)

    def _remember(self, note: object) -> None:
        if not isinstance(note, str):
            return
        normalized = " ".join(note.strip().split())[:300]
        if not normalized:
            return
        self.private_notes.append(normalized)
        if len(self.private_notes) > 20:
            self.private_notes.pop(0)

    def agent_manifest(self) -> dict[str, Any]:
        """返回 Season 1 单步玩家标识，不保存 prompt 或密钥。"""

        return {
            "agent_type": "single_step_player",
            "player_id": self.player_id,
        }

    def model_token_usage_snapshot(self) -> dict[str, int]:
        """返回当前游戏的非敏感模型用量汇总。"""

        return dict(self._model_token_usage)

    def _record_token_usage(self, response: dict[str, Any]) -> None:
        self._model_token_usage["successful_response_count"] += 1
        attempts = self._nonnegative_int(getattr(response, "api_attempts", 1), fallback=1)
        self._model_token_usage["api_attempt_count"] += max(1, attempts)

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

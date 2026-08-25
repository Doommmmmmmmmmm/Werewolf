"""使用外部大模型的玩家适配器。"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
import inspect
import json
import re
from typing import Any

from .base import Participant
from ..harness import HarnessRuntime, HarnessSpec
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
        harness_spec: HarnessSpec | Mapping[str, Any] | None = None,
        harness_specs: Mapping[str, HarnessSpec | Mapping[str, Any]] | None = None,
        harness_variants: Mapping[str, Sequence[HarnessSpec | Mapping[str, Any]]] | None = None,
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
        resolved_harnesses: dict[str, HarnessSpec] = {}
        if harness_specs is not None:
            for role, value in harness_specs.items():
                resolved_harnesses[str(role)] = HarnessSpec.from_mapping(value, role=str(role))
        if harness_spec is not None:
            one = HarnessSpec.from_mapping(harness_spec)
            resolved_harnesses[one.role] = one
        self.harness_specs = resolved_harnesses
        resolved_variants: dict[str, tuple[HarnessSpec, ...]] = {}
        if harness_variants is not None:
            for role, values in harness_variants.items():
                normalized_role = str(role)
                variants = tuple(
                    HarnessSpec.from_mapping(value, role=normalized_role)
                    for value in values
                )
                if variants:
                    resolved_variants[normalized_role] = variants
                    self.harness_specs.setdefault(normalized_role, variants[0])
        self.harness_variants = resolved_variants
        self._harness_runtimes: dict[str, HarnessRuntime] = {}
        self.private_notes: list[str] = []
        self.latest_public_sync: dict[str, Any] | None = None
        self._harness_trace: list[dict[str, Any]] = []
        # 只累计 API 返回的 usage 数字；不保留 prompt、原始响应或密钥。
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
        harness_runtime = self._runtime_for_role(str(private["role"]), role_profile)
        if harness_runtime is None:
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
        else:
            system = render_prompt(
                "task_agent_system.txt",
                player_id=self.player_id,
                role=private["role"],
                team=private["team"],
                persona=self.persona,
                role_base=role_profile.base,
                role_strategy=harness_runtime.strategy_for_prompt(),
                harness_spec=json.dumps(
                    harness_runtime.spec.as_dict(), ensure_ascii=False, indent=2
                ),
            )
            prompt = harness_runtime.build_context(
                turn_packet,
                private_notes=self.private_notes,
                latest_public_sync=self.latest_public_sync,
            )
            task_context = prompt.get("task_agent_context", {})
            if isinstance(task_context, Mapping):
                self._harness_trace.append(
                    {
                        "player_id": self.player_id,
                        "role": str(private.get("role") or ""),
                        "harness_id": harness_runtime.spec.harness_id,
                        "harness_fingerprint": harness_runtime.spec.fingerprint,
                        "phase": str(turn_packet.get("game", {}).get("phase") or ""),
                        "replan_required": bool(task_context.get("replan_required")),
                        "replan_reason": str(task_context.get("replan_reason") or ""),
                        "selected_card_ids": [
                            str(item.get("card_id"))
                            for item in task_context.get("selected_cards", [])
                            if isinstance(item, Mapping)
                        ],
                    }
                )
                if len(self._harness_trace) > 300:
                    self._harness_trace.pop(0)
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
                if harness_runtime is not None:
                    harness_runtime.update_belief_board(
                        raw.get("belief_update", raw.get("belief_board_update"))
                    )
                self._remember(
                    raw.get("private_note", raw.get("memory_note")),
                    harness_runtime=harness_runtime,
                )
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

    def _runtime_for_role(self, role: str, profile: Any) -> HarnessRuntime | None:
        variants = self.harness_variants.get(role, ())
        if variants:
            # 同一 player_id 在不同请求中稳定选择同一变体；不同座位则自然获得姿态差异。
            import hashlib

            slot = int.from_bytes(
                hashlib.sha256(self.player_id.encode("utf-8")).digest()[:4], "big"
            ) % len(variants)
            spec = variants[slot]
        else:
            spec = self.harness_specs.get(role)
        if spec is None:
            return None
        runtime = self._harness_runtimes.get(role)
        if runtime is None:
            runtime = HarnessRuntime(spec, profile)
            self._harness_runtimes[role] = runtime
        return runtime

    def _remember(self, note: object, *, harness_runtime: HarnessRuntime | None = None) -> None:
        if harness_runtime is not None:
            self.private_notes = harness_runtime.remember(self.private_notes, note)
            return
        if not isinstance(note, str):
            return
        normalized = " ".join(note.strip().split())[:300]
        if not normalized:
            return
        self.private_notes.append(normalized)
        if len(self.private_notes) > 20:
            self.private_notes.pop(0)

    def agent_manifest(self) -> dict[str, Any]:
        """供完整记录保存实际加载的 Harness 版本；不包含 API 密钥或 prompt 正文。"""

        return {
            "agent_type": "task_agent_harness" if self.harness_specs else "single_step_player",
            "player_id": self.player_id,
            "harnesses": {
                role: spec.manifest() if hasattr(spec, "manifest") else {
                    "role": spec.role,
                    "harness_id": spec.harness_id,
                    "harness_fingerprint": spec.fingerprint,
                    "version": spec.version,
                    "parent_id": spec.parent_id,
                    "source_type": spec.source_type,
                    "card_ids": [card.card_id for card in spec.cards],
                }
                for role, spec in self.harness_specs.items()
            },
            "variant_pools": {
                role: [spec.manifest() for spec in variants]
                for role, variants in self.harness_variants.items()
            },
            "assignment_rule": "player_id 的稳定哈希选择角色候选池中的一个 Harness",
        }

    def harness_catalog(self) -> dict[str, dict[str, Any]]:
        """返回当前参与者加载的完整 Harness 定义，供全盘记录去重保存。"""

        specs = list(self.harness_specs.values())
        for variants in self.harness_variants.values():
            specs.extend(variants)
        return {spec.fingerprint: spec.as_dict() for spec in specs}

    def harness_trace_snapshot(self) -> list[dict[str, Any]]:
        """返回不含发言正文和私密笔记的重规划轨迹。"""

        return [dict(item) for item in self._harness_trace]

    def model_token_usage_snapshot(self) -> dict[str, int]:
        """返回本玩家在当前游戏中的非敏感模型用量汇总。"""

        return dict(self._model_token_usage)

    def _record_token_usage(self, response: dict[str, Any]) -> None:
        """记录一次成功返回的模型调用及服务端实际报告的 usage。"""

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


class TaskAgentParticipant(LlmParticipant):
    """显式命名的 Season 2 玩家适配器。

    Harness 通过父类的 ``harness_spec`` / ``harness_specs`` 参数注入；保留一个独立
    类型名，便于实验记录、工厂和未来人类玩家适配器区分旧的单步 LLM 玩家。
    """

    pass

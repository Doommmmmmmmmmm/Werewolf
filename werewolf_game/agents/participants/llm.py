"""把角色 Task-Agent 接入游戏参与者接口。"""

from __future__ import annotations

from typing import Any

import asyncio
import inspect

from .base import Participant
from ...prompts import RoleProfileStore
from ..task_agent import (
    DEFAULT_MAX_DECISION_RETRIES,
    DEFAULT_MAX_TOOL_CALLS_PER_DECISION,
    DEFAULT_MAX_TOOL_RESULT_TOKENS,
    DEFAULT_MAX_PROMPT_CHARS,
    TaskAgent,
    decision_error,
    normalize_decision,
)


class TaskModelBoundary:
    """Task-Agent 模型请求的外部长度边界。

    Prompt 内容（包括可进化的角色策略）可以自由变化，但每次请求的 system 与
    messages 总字符数由运行时限制。这里使用字符作为无 tokenizer 时的保守预算单位。
    """

    def __init__(self, client: Any, *, max_prompt_chars: int = 12000) -> None:
        self._client = client
        self.max_prompt_chars = max(1, int(max_prompt_chars))

    async def complete_json(self, **kwargs: Any) -> Any:
        prompt_chars = len(str(kwargs.get("system") or ""))
        for message in kwargs.get("messages") or []:
            if isinstance(message, dict):
                prompt_chars += len(str(message.get("content") or ""))
        if prompt_chars > self.max_prompt_chars:
            raise ValueError(
                f"Task-Agent prompt 超过外部上限：{prompt_chars}>{self.max_prompt_chars} 字符"
            )
        # 底层工具循环追加工具结果后也必须继续使用同一硬上限。
        kwargs["max_prompt_chars"] = self.max_prompt_chars
        complete = self._client.complete_json
        if inspect.iscoroutinefunction(complete):
            return await complete(**kwargs)
        return await asyncio.to_thread(complete, **kwargs)

class TaskAgentParticipant(Participant):
    """一个玩家参与者，按行动包身份懒加载对应的角色 Task-Agent。"""

    def __init__(
        self,
        *,
        player_id: str,
        model_client: Any,
        persona: str = "",
        profile_store: RoleProfileStore | None = None,
        request_coordinator: Any = None,
        max_tokens: int = 900,
        max_decision_retries: int = DEFAULT_MAX_DECISION_RETRIES,
        max_tool_calls_per_decision: int = DEFAULT_MAX_TOOL_CALLS_PER_DECISION,
        max_tool_result_tokens: int = DEFAULT_MAX_TOOL_RESULT_TOKENS,
        max_prompt_chars: int = DEFAULT_MAX_PROMPT_CHARS,
    ) -> None:
        super().__init__(player_id)
        # 所有角色（包括 Season 2 候选）共享同一个外部模型边界。
        self.model_client = TaskModelBoundary(model_client, max_prompt_chars=max_prompt_chars)
        self.persona = persona
        self.profile_store = profile_store or RoleProfileStore()
        self.request_coordinator = request_coordinator
        self.max_tokens = max_tokens
        self.max_decision_retries = max_decision_retries
        self.max_tool_calls_per_decision = max_tool_calls_per_decision
        self.max_tool_result_tokens = max_tool_result_tokens
        self.max_prompt_chars = max_prompt_chars
        self._agents: dict[str, TaskAgent] = {}

    async def observe(self, sync_packet: dict[str, Any]) -> None:
        del sync_packet
        for agent in self._agents.values():
            # Keep the participant interface compatible, but do not persist
            # the public sync in the minimum stateless baseline.
            await agent.observe({})

    async def decide(self, turn_packet: dict[str, Any]) -> dict[str, Any]:
        if turn_packet["self"]["player_id"] != self.player_id:
            raise ValueError("行动包被发送给了错误玩家")
        role = str(turn_packet["private_information"]["role"])
        agent = self._agent_for(role)
        return await agent.decide(turn_packet)

    def agent_manifest(self) -> dict[str, Any]:
        return {
            "agent_type": "task_agent_participant",
            "player_id": self.player_id,
            "roles": {
                role: agent.agent_manifest() for role, agent in self._agents.items()
            },
            "conversation_mode": "new_session_per_decision",
            "max_decision_retries": self.max_decision_retries,
            "max_tool_calls_per_decision": self.max_tool_calls_per_decision,
            "max_tool_result_tokens": self.max_tool_result_tokens,
            "max_prompt_chars": self.max_prompt_chars,
        }

    def model_token_usage_snapshot(self) -> dict[str, int]:
        fields = (
            "successful_response_count",
            "api_attempt_count",
            "reported_usage_response_count",
            "input_tokens",
            "output_tokens",
            "total_tokens",
        )
        totals = {field: 0 for field in fields}
        for agent in self._agents.values():
            snapshot = agent.model_token_usage_snapshot()
            for field in fields:
                totals[field] += int(snapshot.get(field, 0) or 0)
        return totals

    def _agent_for(self, role: str) -> TaskAgent:
        agent = self._agents.get(role)
        if agent is not None:
            return agent
        agent = TaskAgent(
            player_id=self.player_id,
            profile=self.profile_store.profile(role),
            model_client=self.model_client,
            persona=self.persona,
            request_coordinator=self.request_coordinator,
            max_tokens=self.max_tokens,
            max_decision_retries=self.max_decision_retries,
            max_tool_calls_per_decision=self.max_tool_calls_per_decision,
            max_tool_result_tokens=self.max_tool_result_tokens,
            max_prompt_chars=self.max_prompt_chars,
        )
        self._agents[role] = agent
        return agent


class LlmParticipant(TaskAgentParticipant):
    """兼容旧入口名称的基础 LLM 玩家。"""

    pass


__all__ = [
    "LlmParticipant",
    "TaskAgentParticipant",
    "decision_error",
    "normalize_decision",
]

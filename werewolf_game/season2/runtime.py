"""在同一局中按角色加载不同进化节点。"""

from __future__ import annotations

import importlib.util
import sys
from typing import Any

from ..agents.participants.base import Participant
from ..agents.participants.llm import TaskModelBoundary
from ..agents.task_agent import DEFAULT_MAX_PROMPT_CHARS
from ..prompts import RoleProfile, RoleProfileStore
from .archive import EvolutionArchive


class CandidateModuleLoader:
    """加载经过静态检查的候选 ``task_agent.py``。"""

    def __init__(self, archive: EvolutionArchive) -> None:
        self.archive = archive
        self._classes: dict[str, type[Any]] = {}

    def task_agent_class(self, node_id: str) -> type[Any]:
        cached = self._classes.get(node_id)
        if cached is not None:
            return cached
        # 每个节点按 code hash 使用独立模块名，避免 Python 模块缓存串到其他候选。
        node = self.archive.node(node_id)
        path = self.archive.node_code_directory(node_id) / "task_agent.py"
        # 候选代码使用 agents 包名加载，使其相对导入能够访问稳定 core 包。
        module_name = f"werewolf_game.agents._season2_{node.role}_{node.code_hash[:16]}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"无法加载候选模块：{path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(module_name, None)
            raise
        candidate = getattr(module, "TaskAgent", None)
        if not isinstance(candidate, type):
            raise ValueError(f"候选 {node_id} 没有导出 TaskAgent 类")
        self._classes[node_id] = candidate
        return candidate


class VersionedTaskAgentParticipant(Participant):
    """玩家在收到身份后，加载该角色本局选中的候选节点。"""

    def __init__(
        self,
        *,
        player_id: str,
        model_client: Any,
        role_nodes: dict[str, str],
        archive: EvolutionArchive,
        module_loader: CandidateModuleLoader | None = None,
        profile_store: RoleProfileStore | None = None,
        request_coordinator: Any = None,
        max_tokens: int = 900,
        max_decision_retries: int = 2,
        max_tool_calls_per_decision: int = 1,
        max_tool_result_tokens: int = 800,
        max_prompt_chars: int = DEFAULT_MAX_PROMPT_CHARS,
    ) -> None:
        super().__init__(player_id)
        self.model_client = TaskModelBoundary(model_client, max_prompt_chars=max_prompt_chars)
        self.role_nodes = dict(role_nodes)
        self.archive = archive
        self.module_loader = module_loader or CandidateModuleLoader(archive)
        self.profile_store = profile_store or RoleProfileStore()
        self.request_coordinator = request_coordinator
        self.max_tokens = max_tokens
        self.max_decision_retries = max_decision_retries
        self.max_tool_calls_per_decision = max_tool_calls_per_decision
        self.max_tool_result_tokens = max_tool_result_tokens
        self.max_prompt_chars = max_prompt_chars
        self._agents: dict[str, Any] = {}

    async def observe(self, sync_packet: dict[str, Any]) -> None:
        # 版本候选可以自行决定是否维护状态；基线 Task-Agent 会忽略跨行动同步。
        for agent in self._agents.values():
            await agent.observe(sync_packet)

    async def decide(self, turn_packet: dict[str, Any]) -> dict[str, Any]:
        # 身份由 GameEngine 放入私有行动包，参与者据此选择唯一对应的角色节点。
        role = str(turn_packet["private_information"]["role"])
        return await self._agent_for(role).decide(turn_packet)

    def agent_manifest(self) -> dict[str, Any]:
        return {
            "agent_type": "season2_versioned_task_agent",
            "player_id": self.player_id,
            "role_nodes": dict(self.role_nodes),
            "loaded_roles": {
                role: agent.agent_manifest()
                for role, agent in self._agents.items()
                if hasattr(agent, "agent_manifest")
            },
        }

    def model_token_usage_snapshot(self) -> dict[str, int]:
        totals: dict[str, int] = {}
        for agent in self._agents.values():
            if not hasattr(agent, "model_token_usage_snapshot"):
                continue
            for key, value in agent.model_token_usage_snapshot().items():
                if key.startswith("prompt_"):
                    # Prompt 预算由共享的 TaskModelBoundary 统一统计，避免
                    # 同一请求同时被候选 Agent 和边界层重复计数。
                    continue
                totals[key] = totals.get(key, 0) + int(value or 0)
        for key, value in self.model_client.prompt_budget_snapshot().items():
            totals[key] = totals.get(key, 0) + int(value or 0)
        return totals

    def _agent_for(self, role: str) -> Any:
        if role in self._agents:
            return self._agents[role]
        try:
            node_id = self.role_nodes[role]
        except KeyError as error:
            raise ValueError(f"本局没有为角色 {role} 分配候选节点") from error
        node = self.archive.node(node_id)
        if node.role != role:
            raise ValueError(f"节点 {node_id} 属于 {node.role}，不能分配给 {role}")
        # 固定 base 规则仍从主项目读取，候选只替换可进化的 task/代码部分。
        base_profile = self.profile_store.profile(role)
        task = (
            self.archive.node_code_directory(node_id) / "task.md"
        ).read_text(encoding="utf-8").strip()
        profile = RoleProfile(role=role, base=base_profile.base, task=task)
        agent_class = self.module_loader.task_agent_class(node_id)
        agent = agent_class(
            player_id=self.player_id,
            profile=profile,
            model_client=self.model_client,
            request_coordinator=self.request_coordinator,
            max_tokens=self.max_tokens,
            max_decision_retries=self.max_decision_retries,
            max_tool_calls_per_decision=self.max_tool_calls_per_decision,
            max_tool_result_tokens=self.max_tool_result_tokens,
            max_prompt_chars=self.max_prompt_chars,
        )
        if not callable(getattr(agent, "decide", None)):
            raise ValueError(f"候选 {node_id} 的 TaskAgent 没有 decide 方法")
        self._agents[role] = agent
        return agent

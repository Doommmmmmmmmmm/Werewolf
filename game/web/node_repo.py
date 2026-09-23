"""已归档进化节点的只读仓库。

Season 2 结束后，它的 archive 被整体冻结到 `archives/season2-<日期>/`。归档内容
仍然可读，但**不能再用 `EvolutionArchive` 去读**：该类在打开时会比对
`season2.json` 与 manifest 里冻结的配置，而冻结的是绝对路径，文件一搬就必然不一致，
校验会直接拒绝加载。那个校验的目的是防止不同实验条件静默混用，挡的是"继续拿它跑
进化"，不挡读数据。

因此本模块直接读归档里的纯数据文件（`manifest.json`、`nodes/<id>/node.json`、
`summary/output/node_ratings.json`），并用 importlib 按目录加载节点代码。
只读、不做冻结校验、不写入归档。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

from ..agents.participants.base import Participant
from ..agents.participants.llm import TaskModelBoundary
from ..agents.task_agent import DEFAULT_MAX_PROMPT_CHARS
from ..prompts import RoleProfile, RoleProfileStore

# 节点代码里可用的文件，与 Season 2 的 Pi 白名单一致。
NODE_CODE_FILES = ("task_agent.py", "task.md")


@dataclass(frozen=True)
class ArchivedNode:
    """归档中的一个进化节点。"""

    node_id: str
    role: str
    status: str
    depth: int
    parent_id: str | None
    code_directory: Path

    @property
    def is_retained(self) -> bool:
        return self.status == "retained"


class ArchivedNodeRepository:
    """按角色读取归档节点，并加载其 Task-Agent 实现。

    ``archive_root`` 指向归档里的 archive 目录（含 `manifest.json` 与 `nodes/`）；
    ``ratings_path`` 指向预计算的强度表，缺失时强度相关口径不可用。
    """

    def __init__(
        self,
        archive_root: str | Path,
        *,
        ratings_path: str | Path | None = None,
    ) -> None:
        self.archive_root = Path(archive_root)
        manifest_path = self.archive_root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"找不到归档 manifest：{manifest_path}")
        self._manifest: dict[str, Any] = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
        self._ratings: dict[str, Any] | None = None
        if ratings_path is not None:
            path = Path(ratings_path)
            if path.is_file():
                try:
                    value = json.loads(path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    value = None
                if isinstance(value, dict):
                    self._ratings = value
        self._classes: dict[str, type[Any]] = {}

    # ------------------------------------------------------------------
    # 节点查询。
    # ------------------------------------------------------------------

    @property
    def roles(self) -> list[str]:
        return sorted(self._manifest.get("roles") or {})

    def _node_from_dict(self, value: dict[str, Any]) -> ArchivedNode:
        node_id = str(value["node_id"])
        return ArchivedNode(
            node_id=node_id,
            role=str(value["role"]),
            status=str(value.get("status") or ""),
            depth=int(value.get("depth") or 0),
            parent_id=value.get("parent_id"),
            code_directory=self.archive_root / "nodes" / node_id / "code",
        )

    def node(self, node_id: str) -> ArchivedNode:
        nodes = self._manifest.get("nodes") or {}
        try:
            return self._node_from_dict(nodes[str(node_id)])
        except KeyError as error:
            raise KeyError(f"归档中不存在节点：{node_id}") from error

    def nodes_for_role(self, role: str) -> list[ArchivedNode]:
        nodes = (self._manifest.get("nodes") or {}).values()
        return [
            self._node_from_dict(value)
            for value in nodes
            if value.get("role") == role
        ]

    def retained_nodes(self, role: str) -> list[ArchivedNode]:
        return [node for node in self.nodes_for_role(role) if node.is_retained]

    # ------------------------------------------------------------------
    # 强度表。
    # ------------------------------------------------------------------

    @property
    def ratings(self) -> dict[str, Any] | None:
        return self._ratings

    def rating_entries(self, model_key: str) -> dict[str, dict[str, Any]]:
        """返回某个强度模型的 ``node_id -> entry`` 映射；缺失时为空。"""

        models = (self._ratings or {}).get("models") or {}
        entries = (models.get(model_key) or {}).get("nodes")
        if not isinstance(entries, list):
            return {}
        return {
            str(item["node_id"]): item
            for item in entries
            if isinstance(item, dict) and item.get("node_id")
        }

    # ------------------------------------------------------------------
    # 代码加载。
    # ------------------------------------------------------------------

    def evaluation_summary(self, node_id: str) -> dict[str, Any] | None:
        """读取节点的评测汇总（20 局初筛结果）。

        不使用 node.json 里的 `evaluation_summary_path`——那是归档前的绝对路径，
        搬动后已失效。改为按归档布局推导：
        `<归档>/records/evolution/evaluations/<role>/<node_id>/summary.json`。
        """

        try:
            node = self.node(node_id)
        except KeyError:
            return None
        # archive_root = <归档>/records/evolution/archive
        path = (
            self.archive_root.parent
            / "evaluations"
            / node.role
            / node_id
            / "summary.json"
        )
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return value if isinstance(value, dict) else None

    def task_md(self, node_id: str) -> str:
        return (self.node(node_id).code_directory / "task.md").read_text(
            encoding="utf-8"
        ).strip()

    def task_agent_class(self, node_id: str) -> type[Any]:
        """按节点目录加载 ``task_agent.py`` 并返回其 ``TaskAgent`` 类。

        与 Season 2 的 `CandidateModuleLoader` 同样按内容哈希生成唯一模块名，
        避免不同节点互相串用 Python 模块缓存。
        """

        cached = self._classes.get(node_id)
        if cached is not None:
            return cached
        node = self.node(node_id)
        path = node.code_directory / "task_agent.py"
        if not path.is_file():
            raise FileNotFoundError(f"节点 {node_id} 缺少 task_agent.py：{path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        # 使用 agents 包名加载，使候选代码的相对导入能访问稳定的游戏内核。
        module_name = f"game.agents._archived_{node.role}_{digest}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"无法加载归档节点模块：{path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(module_name, None)
            raise
        candidate = getattr(module, "TaskAgent", None)
        if not isinstance(candidate, type):
            raise ValueError(f"归档节点 {node_id} 没有导出 TaskAgent 类")
        self._classes[node_id] = candidate
        return candidate


class ArchivedNodeParticipant(Participant):
    """用归档节点的 Task-Agent 实现来占一个席位。

    与 Season 2 的 `VersionedTaskAgentParticipant` 语义一致：角色固定规则
    (`base.md`) 仍从主项目的 `RoleProfileStore` 读取，只有可进化的 `task.md`
    和 `task_agent.py` 来自归档节点。身份由引擎在行动包里给出，据此选择节点。
    """

    def __init__(
        self,
        *,
        player_id: str,
        model_client: Any,
        role_nodes: Mapping[str, str],
        repository: ArchivedNodeRepository,
        profile_store: RoleProfileStore | None = None,
        request_coordinator: Any = None,
        max_tokens: int = 900,
        max_decision_retries: int = 2,
        max_tool_calls_per_decision: int = 1,
        max_tool_result_tokens: int = 800,
        max_prompt_chars: int = DEFAULT_MAX_PROMPT_CHARS,
    ) -> None:
        super().__init__(player_id)
        self.model_client = TaskModelBoundary(
            model_client, max_prompt_chars=max_prompt_chars
        )
        self.role_nodes = dict(role_nodes)
        self.repository = repository
        self.profile_store = profile_store or RoleProfileStore()
        self.request_coordinator = request_coordinator
        self.max_tokens = max_tokens
        self.max_decision_retries = max_decision_retries
        self.max_tool_calls_per_decision = max_tool_calls_per_decision
        self.max_tool_result_tokens = max_tool_result_tokens
        self.max_prompt_chars = max_prompt_chars
        self._agents: dict[str, Any] = {}

    async def observe(self, sync_packet: dict[str, Any]) -> None:
        for agent in self._agents.values():
            await agent.observe(sync_packet)

    async def decide(self, turn_packet: dict[str, Any]) -> dict[str, Any]:
        role = str(turn_packet["private_information"]["role"])
        return await self._agent_for(role).decide(turn_packet)

    def agent_manifest(self) -> dict[str, Any]:
        return {
            "agent_type": "archived_node_participant",
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
                # Prompt 预算由共享的 TaskModelBoundary 统一统计，避免重复计数。
                if key.startswith("prompt_"):
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
            raise ValueError(f"本局没有为角色 {role} 分配节点") from error
        node = self.repository.node(node_id)
        if node.role != role:
            raise ValueError(f"节点 {node_id} 属于 {node.role}，不能分配给 {role}")
        base_profile = self.profile_store.profile(role)
        profile = RoleProfile(
            role=role,
            base=base_profile.base,
            task=self.repository.task_md(node_id),
        )
        agent_class = self.repository.task_agent_class(node_id)
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
            raise ValueError(f"归档节点 {node_id} 的 TaskAgent 没有 decide 方法")
        self._agents[role] = agent
        return agent

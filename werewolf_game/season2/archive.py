"""角色候选池和进化树的持久化实现。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import StrEnum
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Iterable

from ..prompts import PROMPT_DIRECTORY
from .config import Season2Config


class NodeStatus(StrEnum):
    """进化节点的唯一三种状态。"""
    PENDING = "pending"
    RETAINED = "retained"
    DISCARDED = "discarded"


@dataclass
class EvolutionNode:
    """节点元数据；代码本体保存在同名目录的 code/ 下。"""
    node_id: str
    role: str
    status: str
    parent_id: str | None
    depth: int
    children: list[str]
    created_at: str
    code_hash: str
    patch_path: str | None = None
    diagnosis_path: str | None = None
    evaluation_summary_path: str | None = None
    status_reason: str = ""

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "EvolutionNode":
        return cls(
            node_id=str(value["node_id"]),
            role=str(value["role"]),
            status=str(value["status"]),
            parent_id=(str(value["parent_id"]) if value.get("parent_id") else None),
            depth=int(value["depth"]),
            children=[str(item) for item in value.get("children", ())],
            created_at=str(value["created_at"]),
            code_hash=str(value.get("code_hash", "")),
            patch_path=(str(value["patch_path"]) if value.get("patch_path") else None),
            diagnosis_path=(str(value["diagnosis_path"]) if value.get("diagnosis_path") else None),
            evaluation_summary_path=(str(value["evaluation_summary_path"]) if value.get("evaluation_summary_path") else None),
            status_reason=str(value.get("status_reason", "")),
        )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: object) -> None:
    # 先写临时文件再替换，避免进程中断时 manifest 只写了一半。
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def snapshot_hash(directory: Path, files: Iterable[str] = ("task_agent.py", "task.md")) -> str:
    # hash 同时绑定文件名和内容，用来识别候选版本并构造隔离模块名。
    digest = hashlib.sha256()
    for relative in sorted(files):
        path = directory / relative
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def core_snapshot_hash() -> str:
    """计算稳定游戏内核的指纹，排除缓存文件后用于赛季完整性校验。"""

    core_directory = Path(__file__).resolve().parents[1] / "core"
    digest = hashlib.sha256()
    for path in sorted(core_directory.rglob("*.py")):
        relative = path.relative_to(core_directory).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


class EvolutionArchive:
    """一个进程安全写入、可恢复的轻量 archive。

    当前调度器按单进程运行；每次变更都会原子替换 manifest，因此中断不会留下半个 JSON。
    """

    SCHEMA_VERSION = 1

    def __init__(self, config: Season2Config) -> None:
        self.config = config
        self.root = config.paths.archive_root
        self.manifest_path = self.root / "manifest.json"
        self.nodes_root = self.root / "nodes"
        self._manifest: dict[str, Any] = {}
        if self.manifest_path.exists():
            self._manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            self._validate_manifest()

    def initialize(self) -> None:
        """为所有角色建立不可变 base 节点；重复调用不会覆盖已有节点。"""

        if self._manifest:
            return
        self.root.mkdir(parents=True, exist_ok=True)
        self._manifest = {
            "schema_version": self.SCHEMA_VERSION,
            "created_at": _now(),
            "config": self.config.manifest(),
            "next_node_sequence": 1,
            "core_snapshot_hash": core_snapshot_hash(),
            "roles": {},
            "nodes": {},
        }
        # 每个角色拥有独立 base 目录，即使初始代码相同，后续也能独立进化。
        source_agent = Path(__file__).resolve().parents[1] / "agents" / "task_agent.py"
        for role in self.config.evolution.roles:
            node_id = f"{role}-base"
            directory = self.node_code_directory(node_id)
            directory.mkdir(parents=True, exist_ok=False)
            shutil.copy2(source_agent, directory / "task_agent.py")
            shutil.copy2(
                PROMPT_DIRECTORY / "roles" / role / "task.md", directory / "task.md"
            )
            node = EvolutionNode(
                node_id=node_id,
                role=role,
                status=NodeStatus.RETAINED,
                parent_id=None,
                depth=0,
                children=[],
                created_at=_now(),
                code_hash=snapshot_hash(directory, self.config.pi.allowed_files),
                status_reason="base",
            )
            self._manifest["nodes"][node_id] = asdict(node)
            self._manifest["roles"][role] = {
                "base_node_id": node_id,
                "node_ids": [node_id],
                "generated_candidate_count": 0,
                "successful_evolution_count": 0,
            }
            _atomic_json(self.node_directory(node_id) / "node.json", asdict(node))
        self._save()

    def node(self, node_id: str) -> EvolutionNode:
        try:
            return EvolutionNode.from_dict(self._manifest["nodes"][node_id])
        except KeyError as error:
            raise KeyError(f"不存在进化节点：{node_id}") from error

    def nodes_for_role(self, role: str) -> list[EvolutionNode]:
        role_entry = self._manifest["roles"].get(role)
        if not role_entry:
            raise KeyError(f"archive 中不存在角色：{role}")
        return [self.node(node_id) for node_id in role_entry["node_ids"]]

    def retained_nodes(self, role: str) -> list[EvolutionNode]:
        return [
            node for node in self.nodes_for_role(role)
            if node.status == NodeStatus.RETAINED
        ]

    def pending_nodes(self, role: str) -> list[EvolutionNode]:
        return [
            node for node in self.nodes_for_role(role)
            if node.status == NodeStatus.PENDING
        ]

    def role_generated_candidate_count(self, role: str) -> int:
        """返回该角色已经创建的全部非 base 候选数。"""

        return int(self._manifest["roles"][role]["generated_candidate_count"])

    def role_successful_evolution_count(self, role: str) -> int:
        """返回该角色从 pending 成功转为 retained 的节点数。"""

        return int(self._manifest["roles"][role]["successful_evolution_count"])

    def role_pending_count(self, role: str) -> int:
        """返回该角色尚未完成评测的候选数，用于预留成功进化容量。"""

        return sum(node.status == NodeStatus.PENDING for node in self.nodes_for_role(role))

    def eligible_nodes(self, role: str) -> list[EvolutionNode]:
        # 待评估节点必须先完成评测；保留节点只有在各类上限未达到时才能继续生子。
        result: list[EvolutionNode] = []
        # retained 扩展按整批创建子节点，因此角色剩余进化额度也必须容纳整批。
        generated_count = self.role_generated_candidate_count(role)
        generated_limit_reached = (
            generated_count + self.config.evolution.children_per_expansion
            > self.config.evolution.max_generated_candidates_per_role
        )
        # pending 最终都有可能成为 retained，因此创建新批次时要为它们预留
        # 成功进化名额，避免后续评测结果超过角色成功进化上限。
        success_capacity_reached = (
            self.role_successful_evolution_count(role)
            + self.role_pending_count(role)
            + self.config.evolution.children_per_expansion
            > self.config.evolution.max_successful_evolutions_per_role
        )
        for node in self.nodes_for_role(role):
            if node.status == NodeStatus.PENDING:
                result.append(node)
            elif (
                node.status == NodeStatus.RETAINED
                and not generated_limit_reached
                and not success_capacity_reached
                and node.depth < self.config.evolution.max_tree_depth
                # 一次扩展会生成整批子节点；必须预留完整批次，避免扩展到一半超出上限。
                and (
                    len(node.children) + self.config.evolution.children_per_expansion
                    <= self.config.evolution.max_children_per_node
                )
            ):
                result.append(node)
        return result

    def create_child(self, parent_id: str, *, diagnosis_path: Path | None = None) -> EvolutionNode:
        parent = self.node(parent_id)
        if parent.status != NodeStatus.RETAINED:
            raise ValueError("只有保留节点可以生成子节点")
        # 这里校验的是“单个子节点”的容量。调度器在选择父节点时已经确保
        # 能容纳完整的 children_per_expansion 批次；批量创建过程中前几个
        # 子节点会逐步占用容量，因此不能再次套用整批容量条件。
        if (
            self.role_generated_candidate_count(parent.role)
            >= self.config.evolution.max_generated_candidates_per_role
            or self.role_successful_evolution_count(parent.role)
            + self.role_pending_count(parent.role)
            >= self.config.evolution.max_successful_evolutions_per_role
            or parent.depth >= self.config.evolution.max_tree_depth
            or len(parent.children) >= self.config.evolution.max_children_per_node
        ):
            raise ValueError(f"父节点 {parent_id} 已达到进化上限")

        # 子节点复制完整父目录，而不是只复制最后一次修改，天然继承整条祖先路径。
        sequence = int(self._manifest["next_node_sequence"])
        self._manifest["next_node_sequence"] = sequence + 1
        node_id = f"{parent.role}-n{sequence:05d}"
        child_directory = self.node_code_directory(node_id)
        child_directory.parent.mkdir(parents=True, exist_ok=False)
        shutil.copytree(self.node_code_directory(parent_id), child_directory)
        node = EvolutionNode(
            node_id=node_id,
            role=parent.role,
            status=NodeStatus.PENDING,
            parent_id=parent_id,
            depth=parent.depth + 1,
            children=[],
            created_at=_now(),
            code_hash=snapshot_hash(child_directory),
            diagnosis_path=str(diagnosis_path) if diagnosis_path else None,
        )
        parent.children.append(node_id)
        self._manifest["nodes"][parent_id] = asdict(parent)
        self._manifest["nodes"][node_id] = asdict(node)
        role_entry = self._manifest["roles"][parent.role]
        role_entry["node_ids"].append(node_id)
        role_entry["generated_candidate_count"] = (
            int(role_entry["generated_candidate_count"]) + 1
        )
        self._write_node(parent)
        self._write_node(node)
        self._save()
        return node

    def finalize_child_code(self, node_id: str) -> EvolutionNode:
        # 只在真正生成 patch 时加载，普通游戏运行不承担该模块的导入成本。
        import difflib

        # Pi 修改的是候选快照；这里生成可审计 patch，并重新计算最终代码指纹。
        node = self.node(node_id)
        if not node.parent_id:
            raise ValueError("base 节点没有父代 patch")
        parent_dir = self.node_code_directory(node.parent_id)
        child_dir = self.node_code_directory(node_id)
        patch_lines: list[str] = []
        for filename in self.config.pi.allowed_files:
            before_path = parent_dir / filename
            after_path = child_dir / filename
            before = before_path.read_text(encoding="utf-8").splitlines(keepends=True)
            after = after_path.read_text(encoding="utf-8").splitlines(keepends=True)
            patch_lines.extend(
                difflib.unified_diff(
                    before,
                    after,
                    fromfile=f"{node.parent_id}/{filename}",
                    tofile=f"{node_id}/{filename}",
                )
            )
        patch_path = self.node_directory(node_id) / "ancestor.patch"
        patch_path.write_text("".join(patch_lines), encoding="utf-8")
        node.patch_path = str(patch_path)
        node.code_hash = snapshot_hash(child_dir, self.config.pi.allowed_files)
        self._update_node(node)
        return node

    def set_status(
        self,
        node_id: str,
        status: NodeStatus,
        *,
        reason: str,
        evaluation_summary_path: Path | None = None,
    ) -> EvolutionNode:
        # 状态只能由 pending 单向流向 retained/discarded，避免已评估节点被静默改写。
        node = self.node(node_id)
        # base 是每个角色永远保留的起点；评测结果只能作为它的附加资料，
        # 不能让一次失败评测破坏后续进化所需的基线。
        if node.node_id == f"{node.role}-base" and status == NodeStatus.DISCARDED:
            raise ValueError("base 节点不可修改为 discarded")
        if node.status != NodeStatus.PENDING:
            raise ValueError("只有待评估节点可以转为保留或舍弃")
        if status not in {NodeStatus.RETAINED, NodeStatus.DISCARDED}:
            raise ValueError("待评估节点只能转为保留或舍弃")
        if status == NodeStatus.RETAINED:
            role_entry = self._manifest["roles"][node.role]
            successful_count = int(role_entry["successful_evolution_count"])
            if successful_count >= self.config.evolution.max_successful_evolutions_per_role:
                raise ValueError(f"角色 {node.role} 已达到成功进化上限")
            role_entry["successful_evolution_count"] = successful_count + 1
        node.status = status
        node.status_reason = str(reason)
        if evaluation_summary_path:
            node.evaluation_summary_path = str(evaluation_summary_path)
        self._update_node(node)
        return node

    def attach_evaluation_summary(self, node_id: str, evaluation_summary_path: Path) -> EvolutionNode:
        """给 retained 节点绑定一份补充评测汇总，供后续诊断使用。"""

        node = self.node(node_id)
        if node.status != NodeStatus.RETAINED:
            raise ValueError("只有 retained 节点可以绑定补充评测汇总")
        if not evaluation_summary_path.exists():
            raise FileNotFoundError(evaluation_summary_path)
        node.evaluation_summary_path = str(evaluation_summary_path)
        self._update_node(node)
        return node

    def lineage(self, node_id: str) -> list[EvolutionNode]:
        # 返回 base -> 当前节点的顺序，供 Meta-Agent 理解本分支如何逐步形成。
        path: list[EvolutionNode] = []
        current: EvolutionNode | None = self.node(node_id)
        while current is not None:
            path.append(current)
            current = self.node(current.parent_id) if current.parent_id else None
        return list(reversed(path))

    def tree_manifest(self, role: str | None = None) -> dict[str, Any]:
        if role is None:
            node_ids = list(self._manifest["nodes"])
            roles = self._manifest["roles"]
        else:
            node_ids = list(self._manifest["roles"][role]["node_ids"])
            roles = {role: self._manifest["roles"][role]}
        return {
            "roles": roles,
            "nodes": {node_id: self._manifest["nodes"][node_id] for node_id in node_ids},
        }

    def node_directory(self, node_id: str) -> Path:
        return self.nodes_root / node_id

    def node_code_directory(self, node_id: str) -> Path:
        return self.node_directory(node_id) / "code"

    def _update_node(self, node: EvolutionNode) -> None:
        self._manifest["nodes"][node.node_id] = asdict(node)
        self._write_node(node)
        self._save()

    def _write_node(self, node: EvolutionNode) -> None:
        _atomic_json(self.node_directory(node.node_id) / "node.json", asdict(node))

    def _save(self) -> None:
        _atomic_json(self.manifest_path, self._manifest)

    def _validate_manifest(self) -> None:
        # archive 一旦初始化，角色集合和全部配置即被冻结，防止实验条件混淆。
        if int(self._manifest.get("schema_version", -1)) != self.SCHEMA_VERSION:
            raise ValueError("不支持的 Season 2 archive schema")
        expected_roles = set(self.config.evolution.roles)
        actual_roles = set(self._manifest.get("roles", {}))
        if expected_roles != actual_roles:
            raise ValueError("配置角色集合与已有 archive 不一致，不能静默混用")
        if self._manifest.get("config") != self.config.manifest():
            raise ValueError(
                "外置配置与已有 archive 的冻结配置不一致；请恢复原配置或使用新的 archive_root"
            )
        expected_core_hash = self._manifest.get("core_snapshot_hash")
        if not expected_core_hash:
            raise ValueError("已有 archive 缺少 core_snapshot_hash，无法确认游戏规则未被修改")
        actual_core_hash = core_snapshot_hash()
        if expected_core_hash != actual_core_hash:
            raise ValueError(
                "Game core 已发生变化；为避免混用不同规则，当前 archive 被拒绝继续使用"
            )

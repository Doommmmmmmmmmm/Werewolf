"""Season 2 单步进化状态机。"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from .archive import EvolutionArchive, NodeStatus
from .code_agent import PiCodeAgent
from .config import Season2Config
from .evaluator import EvolutionEvaluator
from .meta_agent import MetaAgent
from .scheduler import EvolutionScheduler
from .validation import validate_candidate_files, validate_candidate_source
from .runtime import CandidateModuleLoader
from .validation import validate_task_agent_class
from .smoke import smoke_test_candidate
from .resources import MetaResourceCatalog
from .progress import report, report_error


class EvolutionManager:
    def __init__(
        self,
        *,
        config: Season2Config,
        archive: EvolutionArchive,
        meta_agent: MetaAgent,
        code_agent: PiCodeAgent,
        evaluator: EvolutionEvaluator,
    ) -> None:
        self.config = config
        self.archive = archive
        self.meta_agent = meta_agent
        self.code_agent = code_agent
        self.evaluator = evaluator

    def initialize(self) -> None:
        # 初始化只创建 base；若 archive 已存在，配置校验在 EvolutionArchive 构造时完成。
        self.archive.initialize()
        for role in self.config.evolution.roles:
            base = self.archive.node(f"{role}-base")
            directory = self.archive.node_code_directory(base.node_id)
            validate_candidate_files(directory, self.config.pi.allowed_files)
            validate_candidate_source(directory)

    async def step(self) -> dict[str, Any]:
        # 一个 step 代表一次成功进化：只有新节点被评为 retained 才结束。
        # 如果候选被舍弃，则继续寻找下一个可处理节点；所有中间状态都会落盘。
        self.initialize()
        recovered = self._find_unfinished_batch()
        if recovered is None:
            timestamp = datetime.now(timezone.utc).isoformat()
            suffix = hashlib.sha256(
                f"{timestamp}:{id(self)}".encode("utf-8")
            ).hexdigest()[:8]
            operation_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + suffix
            operation_directory = self.config.paths.operation_root / operation_id
            operation_directory.mkdir(parents=True, exist_ok=False)
        else:
            operation_id, operation_directory, recovered_result = recovered
            report(
                "恢复未完成 step",
                operation_id=operation_id,
                parent=recovered_result.get("parent_id"),
                pending=sum(
                    branch.get("child_status") == NodeStatus.PENDING
                    for branch in recovered_result.get("branches", [])
                ),
            )
        history: list[dict[str, Any]] = []
        attempted_nodes: set[str] = set()
        scheduler = EvolutionScheduler(self.archive, self.config)

        # 上次已经完成了父节点扩展，但进程在子节点评测期间退出：
        # 恢复同一个 operation 和同一批子节点，不重新走角色选择器，也不重复生成分支。
        if recovered is not None:
            parent_id = str(recovered_result.get("parent_id", ""))
            attempted_nodes.add(parent_id)
            branches = [dict(branch) for branch in recovered_result.get("branches", [])]
            replay_items = [
                {"game_index": branch.get("replay_game_index")}
                for branch in branches
            ]
            resumed = await self._evaluate_pending_branches(
                operation_directory=operation_directory,
                operation_id=operation_id,
                parent_id=parent_id,
                branches=branches,
                replay_items=replay_items,
                history=history,
            )
            if resumed is not None:
                return resumed
            # 批次中的候选已全部得到 discarded 结论；本 step 才允许重新调度其他角色。
            self._write(
                operation_directory / "result.json",
                {
                    "action": "batch_discarded",
                    "operation_id": operation_id,
                    "parent_id": parent_id,
                    "branches": branches,
                    "history": history,
                },
            )

        while True:
            node = scheduler.select(exclude_node_ids=attempted_nodes)
            if node is None:
                action = "stopped" if not history else "no_evolution"
                result = {
                    "action": action,
                    "operation_id": operation_id,
                    "reason": "no_eligible_nodes" if action == "stopped" else "all_attempts_discarded",
                    "history": history,
                }
                self._write(operation_directory / "result.json", result)
                report("step 未产生新保留节点", attempts=len(history))
                return result
            attempted_nodes.add(node.node_id)
            report("step 选择节点", node=node.node_id, role=node.role, node_status=node.status)
            self._write(operation_directory / "selection.json", {
                "operation_id": operation_id,
                "selected_node": vars(node),
                "config_path": str(self.config.source_path),
                "attempt": len(history) + 1,
            })

            if node.status == NodeStatus.PENDING:
                summary, summary_path = await self.evaluator.evaluate(node.node_id)
                status = (
                    NodeStatus.DISCARDED
                    if summary["recommended_status"] == NodeStatus.DISCARDED
                    else NodeStatus.RETAINED
                )
                updated = self.archive.set_status(
                    node.node_id,
                    status,
                    reason=";".join(str(item) for item in summary["status_reasons"]),
                    evaluation_summary_path=summary_path,
                )
                item = {
                    "type": "evaluated",
                    "node_id": node.node_id,
                    "status": updated.status,
                    "summary_path": str(summary_path),
                    "summary": summary,
                }
                history.append(item)
                self._write(operation_directory / "progress.json", {"history": history})
                if updated.status == NodeStatus.RETAINED and node.parent_id:
                    result = {"action": "evolved", "operation_id": operation_id, **item, "history": history}
                    self._write(operation_directory / "result.json", result)
                    report("step 产生新保留节点", node=node.node_id)
                    return result
                continue

            # retained 节点可能还没有评测；先补做一次，确保 Meta-Agent 至少看到一局回放。
            if not node.evaluation_summary_path or not Path(node.evaluation_summary_path).exists():
                bootstrap_summary, bootstrap_path = await self.evaluator.evaluate(node.node_id)
                self.archive.attach_evaluation_summary(node.node_id, bootstrap_path)
                self._write(operation_directory / "bootstrap-evaluation.json", {
                    "summary_path": str(bootstrap_path), "summary": bootstrap_summary,
                })

            branch_count = self.config.evolution.children_per_expansion
            replay_catalog = MetaResourceCatalog(
                config=self.config, archive=self.archive, node_id=node.node_id
            )
            try:
                replay_items = replay_catalog.select_random_replay_items(branch_count)
            except Exception as error:
                item = {
                    "type": "expansion_blocked",
                    "parent_id": node.node_id,
                    "reason": f"replay_sampling_failed:{type(error).__name__}:{error}",
                }
                history.append(item)
                self._write(operation_directory / "progress.json", {"history": history})
                continue

            branches: list[dict[str, Any]] = []
            for branch_index, replay_item in enumerate(replay_items):
                branch_directory = operation_directory / f"branch{branch_index}"
                branch_directory.mkdir(parents=True, exist_ok=False)
                diagnosis_path = branch_directory / "diagnosis.json"
                child = self.archive.create_child(node.node_id, diagnosis_path=diagnosis_path)
                try:
                    report("开始生成分支", parent=node.node_id, branch=f"{branch_index + 1}/{branch_count}", child=child.node_id)
                    # 每个分支使用独立的 Meta-Agent 调用和独立的 code agent 工作目录。
                    diagnosis = await self.meta_agent.diagnose(
                        node_id=node.node_id,
                        output_path=diagnosis_path,
                        required_replay_item=replay_item,
                    )
                    code_result = await self.code_agent.apply(
                        workspace=self.archive.node_code_directory(child.node_id),
                        role=child.role,
                        diagnosis=diagnosis,
                        operation_directory=branch_directory,
                    )
                    child = self.archive.finalize_child_code(child.node_id)
                    candidate_class = CandidateModuleLoader(self.archive).task_agent_class(child.node_id)
                    validate_task_agent_class(candidate_class)
                    smoke_result: dict[str, Any] | None = None
                    if self.config.pi.candidate_smoke_test:
                        smoke_result = await smoke_test_candidate(
                            config=self.config,
                            archive=self.archive,
                            focus_node_id=child.node_id,
                        )
                        self._write(branch_directory / "candidate-smoke.json", smoke_result)
                    branches.append(
                        {
                            "branch_index": branch_index,
                            "replay_game_index": replay_item.get("game_index"),
                            "child_id": child.node_id,
                            "child_status": child.status,
                            "diagnosis_path": str(diagnosis_path),
                            "patch_path": child.patch_path,
                            "code_agent_returncode": code_result.returncode,
                            "candidate_smoke": smoke_result,
                        }
                    )
                    report("分支完成", child=child.node_id, status=child.status)
                except Exception as error:
                    report_error("分支失败", child=child.node_id, error=type(error).__name__)
                    child = self.archive.set_status(
                        child.node_id,
                        NodeStatus.DISCARDED,
                        reason=f"code_agent_failure:{type(error).__name__}:{error}",
                    )
                    branches.append(
                        {
                            "branch_index": branch_index,
                            "replay_game_index": replay_item.get("game_index"),
                            "child_id": child.node_id,
                            "child_status": child.status,
                            "diagnosis_path": str(diagnosis_path),
                            "error": f"{type(error).__name__}: {error}",
                        }
                    )
            # 子节点在同一个 step 内立即评测；只有 retained 才结束 step。
            resumed = await self._evaluate_pending_branches(
                operation_directory=operation_directory,
                operation_id=operation_id,
                parent_id=node.node_id,
                branches=branches,
                replay_items=replay_items,
                history=history,
            )
            if resumed is not None:
                return resumed
            # 本批次没有产生保留节点；回到 while，寻找下一位父节点。

    async def run(self, steps: int) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for _ in range(max(0, int(steps))):
            result = await self.step()
            results.append(result)
            if result["action"] == "stopped":
                break
        return results

    def _find_unfinished_batch(self) -> tuple[str, Path, dict[str, Any]] | None:
        """查找上次已生成、但尚未完成评测的进化批次。

        这里只恢复 ``generated_batch``。只有 selection.json 的中断表示尚未产生
        任何候选，重新进入调度即可；不会把它误认为某个角色的未完成进化。
        """

        root = self.config.paths.operation_root
        if not root.exists():
            return None
        for directory in sorted(path for path in root.iterdir() if path.is_dir()):
            result_path = directory / "result.json"
            if not result_path.exists():
                continue
            try:
                result = json.loads(result_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if result.get("action") != "generated_batch":
                continue
            branches = result.get("branches")
            if not isinstance(branches, list) or not result.get("parent_id"):
                continue
            if not any(
                branch.get("child_status") == NodeStatus.PENDING
                for branch in branches
                if isinstance(branch, dict)
            ):
                continue
            return str(result.get("operation_id") or directory.name), directory, result
        return None

    async def _evaluate_pending_branches(
        self,
        *,
        operation_directory: Path,
        operation_id: str,
        parent_id: str,
        branches: list[dict[str, Any]],
        replay_items: list[dict[str, Any]],
        history: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """评测一个已生成批次中的 pending 子节点，并支持逐个中断恢复。"""

        for branch in branches:
            if branch.get("child_status") != NodeStatus.PENDING:
                continue
            child_id = str(branch["child_id"])
            summary, summary_path = await self.evaluator.evaluate(child_id)
            status = (
                NodeStatus.DISCARDED
                if summary["recommended_status"] == NodeStatus.DISCARDED
                else NodeStatus.RETAINED
            )
            updated = self.archive.set_status(
                child_id,
                status,
                reason=";".join(str(item) for item in summary["status_reasons"]),
                evaluation_summary_path=summary_path,
            )
            branch["child_status"] = updated.status
            branch["evaluation_summary_path"] = str(summary_path)
            branch["evaluation_summary"] = summary
            history.append({"type": "branch", **branch})
            self._write(operation_directory / "progress.json", {"history": history})
            # 每完成一个分支就更新批次快照，避免进程中断后重复评测已经完成的子节点。
            self._write(
                operation_directory / "result.json",
                {
                    "action": "generated_batch",
                    "operation_id": operation_id,
                    "parent_id": parent_id,
                    "branch_count": len(branches),
                    "replay_game_indices": [item.get("game_index") for item in replay_items],
                    "branches": branches,
                    "history": history,
                },
            )
            report("子节点评测完成", child=child_id, status=updated.status)
            if updated.status == NodeStatus.RETAINED:
                result = {
                    "action": "evolved",
                    "operation_id": operation_id,
                    "parent_id": parent_id,
                    "branch_count": len(branches),
                    "replay_game_indices": [item.get("game_index") for item in replay_items],
                    "branches": branches,
                    "retained_child_id": child_id,
                    "history": history,
                }
                self._write(operation_directory / "result.json", result)
                report("step 完成", parent=parent_id, retained_child=child_id)
                return result
        return None

    @staticmethod
    def _write(path: Path, value: object) -> None:
        path.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

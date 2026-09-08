"""待评估节点的真实对局执行与宽松筛选。"""

from __future__ import annotations

import asyncio
from collections import Counter
from contextlib import suppress
import hashlib
import json
from pathlib import Path
import random
from typing import Any

from ..core.constants import team_for_role
from ..core.engine import GameEngine
from ..recording.records import GAMES_PER_ROUND, RoundGameRecordStore
from ..core.rules import create_rules_for_player_count
from ..core.runner import GameRunner
from .archive import EvolutionArchive, EvolutionNode
from .config import Season2Config
from .runtime import CandidateModuleLoader, VersionedTaskAgentParticipant
from .progress import report, report_error


class EvolutionEvaluator:
    def __init__(
        self,
        *,
        config: Season2Config,
        archive: EvolutionArchive,
        model_client: Any,
        request_coordinator: Any,
    ) -> None:
        self.config = config
        self.archive = archive
        self.model_client = model_client
        self.request_coordinator = request_coordinator
        self.loader = CandidateModuleLoader(archive)
        rules = create_rules_for_player_count(
            config.evaluation.player_count, config.evaluation.optional_roles
        )
        missing = sorted(set(config.evolution.roles) - set(rules.role_deck))
        if missing:
            raise ValueError(
                "以下进化角色不在评测牌组中：" + "、".join(missing)
                + "；请调整 evolution.roles 或 evaluation.optional_roles"
            )
        uncovered = sorted(set(rules.role_deck) - set(config.evolution.roles))
        if uncovered:
            raise ValueError(
                "评测牌组中的角色没有候选池：" + "、".join(uncovered)
                + "；请加入 evolution.roles"
            )

    async def evaluate(self, node_id: str) -> tuple[dict[str, Any], Path]:
        # 评测目录按角色/节点隔离；逐局结果落盘后可从中断处继续，不覆盖已完成对局。
        node = self.archive.node(node_id)
        evaluation_directory = self.config.paths.evaluation_root / node.role / node.node_id
        evaluation_directory.mkdir(parents=True, exist_ok=True)
        assignment_path = evaluation_directory / "evaluation-config.json"
        if not assignment_path.exists():
            assignment_path.write_text(
                json.dumps(
                    {
                        "node": vars(node),
                        "config": self.config.manifest()["evaluation"],
                        "model": self._model_manifest(),
                    },
                    ensure_ascii=False,
                    indent=2,
                ) + "\n",
                encoding="utf-8",
            )

        results: list[dict[str, Any]] = []
        pending: list[int] = []
        reused_records: set[str] = set()
        count = self.config.evaluation.games_per_pending_node
        report("开始评测", node=node_id, role=node.role, total=count)
        for game_index in range(count):
            result_path = evaluation_directory / "results" / f"game{game_index}.json"
            if result_path.exists():
                result = json.loads(result_path.read_text(encoding="utf-8"))
                results.append(result)
                continue
            recovered = self._recover_completed_record(node, evaluation_directory, game_index)
            if recovered is not None:
                result_path.parent.mkdir(parents=True, exist_ok=True)
                result_path.write_text(json.dumps(recovered, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                results.append(recovered)
                continue
            planned = self._planned_assignment(node, game_index)
            reused = self._find_reusable_record(
                node,
                planned,
                evaluation_directory=evaluation_directory,
                used_records=reused_records,
            )
            if reused is not None:
                result_path.parent.mkdir(parents=True, exist_ok=True)
                result_path.write_text(
                    json.dumps(reused, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                results.append(reused)
                report(
                    "复用匹配配置对局",
                    node=node_id,
                    game=f"{game_index + 1}/{count}",
                    source=reused.get("reused_from_record"),
                )
                continue
            pending.append(game_index)

        report("评测进度已恢复", node=node_id, completed=len(results), pending=len(pending))

        # game_concurrency 控制同时运行的 GameRunner 数量，模型请求仍受全局 coordinator 限制。
        semaphore = asyncio.Semaphore(self.config.evaluation.game_concurrency)
        remaining = set(pending)

        async def run(index: int) -> dict[str, Any]:
            async with semaphore:
                report("开始对局", node=node_id, game=f"{index + 1}/{count}")
                return await self._run_game(node, evaluation_directory, index)

        async def heartbeat() -> None:
            # 单局可能持续数分钟；心跳只报告状态，不触碰游戏上下文或记录。
            while remaining:
                await asyncio.sleep(30)
                if remaining:
                    report(
                        "评测仍在运行",
                        node=node_id,
                        active=",".join(str(index + 1) for index in sorted(remaining)),
                        completed=count - len(remaining),
                        total=count,
                    )

        tasks = [asyncio.create_task(run(index)) for index in pending]
        heartbeat_task = asyncio.create_task(heartbeat()) if pending else None
        try:
            for task in asyncio.as_completed(tasks):
                result = await task
                results.append(result)
                remaining.discard(int(result["game_index"]))
                report(
                    "对局完成",
                    node=node_id,
                    game=f"{int(result['game_index']) + 1}/{count}",
                    winner=result.get("winner") or "none",
                    completed=len(results),
                    total=count,
                )
        finally:
            remaining.clear()
            if heartbeat_task is not None:
                heartbeat_task.cancel()
                with suppress(asyncio.CancelledError):
                    await heartbeat_task

        results.sort(key=lambda item: int(item["game_index"]))
        summary = self._summary(node, results)
        summary_path = evaluation_directory / "summary.json"
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        report(
            "评测汇总完成",
            node=node_id,
            completed=len(results),
            wins=summary.get("win_count"),
            status=summary.get("recommended_status"),
        )
        return summary, summary_path

    def _planned_assignment(self, node: EvolutionNode, game_index: int) -> dict[str, Any]:
        """计算一局在真正运行前会使用的候选组合。"""

        selection_seed = f"{self.config.evolution.random_seed}:{node.node_id}:game{game_index}:selection"
        randomizer = random.Random(
            int(hashlib.sha256(selection_seed.encode("utf-8")).hexdigest(), 16)
        )
        role_nodes: dict[str, str] = {}
        for role in self.config.evolution.roles:
            if role == node.role:
                role_nodes[role] = node.node_id
            else:
                candidates = self.archive.retained_nodes(role)
                if not candidates:
                    raise RuntimeError(f"角色 {role} 的候选池为空")
                role_nodes[role] = randomizer.choice(candidates).node_id
        # 发牌/对局 seed 由实际候选组合决定，而不是焦点节点名称决定；因此
        # 两次抽到完全相同的组合时，才有可能逐局复用同一结果。
        config_key = json.dumps(
            {
                "role_nodes": role_nodes,
                "player_count": self.config.evaluation.player_count,
                "optional_roles": list(self.config.evaluation.optional_roles),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        seed_text = (
            f"{self.config.evolution.random_seed}:"
            f"{hashlib.sha256(config_key.encode('utf-8')).hexdigest()[:16]}:game{game_index}"
        )
        signature = self._evaluation_signature(role_nodes, seed_text)
        return {
            "game_index": game_index,
            "game_id": f"s2-{node.node_id}-game{game_index}",
            "seed": seed_text,
            "focus_node": node.node_id,
            "role_nodes": role_nodes,
            "evaluation_signature": signature,
        }

    def _evaluation_signature(self, role_nodes: dict[str, str], seed: str) -> str:
        value = {
            "role_nodes": role_nodes,
            "seed": seed,
            "player_count": self.config.evaluation.player_count,
            "optional_roles": list(self.config.evaluation.optional_roles),
            "model": self._model_manifest(),
        }
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def _find_reusable_record(
        self,
        node: EvolutionNode,
        planned: dict[str, Any],
        *,
        evaluation_directory: Path,
        used_records: set[str],
    ) -> dict[str, Any] | None:
        """按候选组合和全局局号逐局复用，不按空位顺序错位填充。"""

        for result_path in self.config.paths.evaluation_root.glob("**/results/game*.json"):
            if evaluation_directory in result_path.parents:
                continue
            try:
                source_result = json.loads(result_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            # 复用必须保持“局号对局号”的位置映射。不能因为配置相同，
            # 就把历史第 2 局挪来填充当前第 3 局；否则前面的局会被反复
            # 研究，而后面的局永远失去对应的历史样本。
            source_game_index = source_result.get("game_index")
            if source_game_index is None:
                # 老记录没有可靠的全局局号，无法证明它与目标局位置一致，
                # 宁可重新运行，也不做可能错位的复用。
                continue
            try:
                if int(source_game_index) != int(planned.get("game_index", -1)):
                    continue
            except (TypeError, ValueError):
                continue
            record_path = str(source_result.get("record_path") or "")
            if not record_path or record_path in used_records:
                continue
            if source_result.get("role_nodes") != planned.get("role_nodes"):
                continue
            source_signature = source_result.get("evaluation_signature")
            if source_signature:
                if source_signature != planned.get("evaluation_signature"):
                    continue
            elif source_result.get("seed") != planned.get("seed"):
                # 旧记录没有签名时至少要求 seed 也完全一致。
                continue
            record_file = Path(record_path)
            if not record_file.exists() or not record_file.is_file():
                continue
            try:
                record = json.loads(record_file.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            snapshot = record.get("final_snapshot") or {}
            public_state = snapshot.get("public_state") or {}
            if public_state.get("status") != "finished" or public_state.get("winner") not in {"wolf", "village"}:
                continue
            used_records.add(record_path)
            return self._project_record_for_focus(
                node=node,
                assignment=planned,
                record=record,
                source_record_path=record_file,
            )
        return None

    def _project_record_for_focus(
        self,
        *,
        node: EvolutionNode,
        assignment: dict[str, Any],
        record: dict[str, Any],
        source_record_path: Path,
    ) -> dict[str, Any]:
        """从已有完整回放重算目标焦点角色的错误和资源指标。"""

        snapshot = record.get("final_snapshot") or {}
        public_state = snapshot.get("public_state") or {}
        audit_state = snapshot.get("audit_state") or {}
        roles = audit_state.get("roles") if isinstance(audit_state, dict) else {}
        focus_players = {
            player_id
            for player_id, role in (roles.items() if isinstance(roles, dict) else ())
            if role == node.role
        }
        events = record.get("events") or []
        runner_events = record.get("runner_events") or []
        focus_errors = [
            item for item in runner_events
            if isinstance(item, dict)
            and item.get("event_class") == "RUNNER_ERROR"
            and str(item.get("player_id", "")) in focus_players
        ]
        focus_fallbacks = [
            item for item in runner_events
            if isinstance(item, dict)
            and item.get("type") == "FALLBACK_ACTION"
            and str(item.get("player_id", "")) in focus_players
        ]
        focus_decisions = sum(
            1 for event in events
            if isinstance(event, dict)
            and event.get("type") == "ACTION_SUBMITTED"
            and str((event.get("payload") or {}).get("player_id", "")) in focus_players
        )
        return {
            **assignment,
            "complete": True,
            "winner": public_state.get("winner"),
            "focus_won": public_state.get("winner") == team_for_role(node.role),
            "decision_count": sum(
                1 for event in events
                if isinstance(event, dict) and event.get("type") == "ACTION_SUBMITTED"
            ),
            "fallback_count": sum(
                1 for item in runner_events
                if isinstance(item, dict) and item.get("type") == "FALLBACK_ACTION"
            ),
            "fallback_rate": 0.0,
            "error_counts": dict(
                Counter(str(item.get("type", "unknown")) for item in runner_events
                if isinstance(item, dict) and item.get("event_class") == "RUNNER_ERROR"
            )),
            "errors": [
                item for item in runner_events
                if isinstance(item, dict) and item.get("event_class") == "RUNNER_ERROR"
            ],
            "focus_player_ids": sorted(focus_players),
            "focus_decision_count": focus_decisions,
            "focus_fallback_count": len(focus_fallbacks),
            "focus_fallback_rate": len(focus_fallbacks) / focus_decisions if focus_decisions else 0.0,
            "focus_error_counts": dict(Counter(str(item.get("type", "unknown")) for item in focus_errors)),
            "focus_errors": focus_errors,
            "record_path": str(source_record_path),
            "reused_from_record": str(source_record_path),
            "evaluation_signature": assignment.get("evaluation_signature"),
            "model_token_usage": record.get("model_token_usage"),
        }

    def _recover_completed_record(
        self, node: EvolutionNode, evaluation_directory: Path, game_index: int
    ) -> dict[str, Any] | None:
        """从已完成但尚未写入结果 JSON 的回放恢复结果。"""

        path = (
            evaluation_directory
            / "games"
            / f"round{game_index // GAMES_PER_ROUND}"
            / "log"
            / f"full-game{game_index % GAMES_PER_ROUND}.json"
        )
        if not path.exists():
            return None
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        snapshot = record.get("final_snapshot") or {}
        public_state = snapshot.get("public_state") or {}
        winner = public_state.get("winner")
        if public_state.get("status") != "finished" or winner not in {"wolf", "village"}:
            return None
        metadata = record.get("metadata") or {}
        assignment = metadata.get("assignment") or {}
        if not assignment:
            assignment_path = evaluation_directory / "assignments" / f"game{game_index}.json"
            if assignment_path.exists():
                assignment = json.loads(assignment_path.read_text(encoding="utf-8"))
        events = record.get("events") or []
        audit_state = (record.get("final_snapshot") or {}).get("audit_state") or {}
        roles = audit_state.get("roles") if isinstance(audit_state, dict) else {}
        focus_players = {
            player_id for player_id, role in (roles.items() if isinstance(roles, dict) else ())
            if role == node.role
        }
        runner_events = record.get("runner_events") or []
        focus_errors = [
            item for item in runner_events
            if isinstance(item, dict)
            and item.get("event_class") == "RUNNER_ERROR"
            and str(item.get("player_id", "")) in focus_players
        ]
        focus_fallbacks = [
            item for item in runner_events
            if isinstance(item, dict)
            and item.get("type") == "FALLBACK_ACTION"
            and str(item.get("player_id", "")) in focus_players
        ]
        focus_decisions = sum(
            1 for event in events
            if isinstance(event, dict)
            and event.get("type") == "ACTION_SUBMITTED"
            and str((event.get("payload") or {}).get("player_id", "")) in focus_players
        )
        return {
            "game_index": game_index,
            "game_id": assignment.get("game_id", metadata.get("game_id")),
            "seed": assignment.get("seed", metadata.get("seed")),
            "focus_node": assignment.get("focus_node", node.node_id),
            "role_nodes": assignment.get("role_nodes", {}),
            "complete": True,
            "winner": winner,
            "focus_won": winner == team_for_role(node.role),
            "decision_count": sum(1 for event in events if event.get("type") == "ACTION_SUBMITTED"),
            "fallback_count": 0,
            "fallback_rate": 0.0,
            "error_counts": {},
            "errors": [],
            "focus_player_ids": sorted(focus_players),
            "focus_decision_count": focus_decisions,
            "focus_fallback_count": len(focus_fallbacks),
            "focus_fallback_rate": len(focus_fallbacks) / focus_decisions if focus_decisions else 0.0,
            "focus_error_counts": dict(Counter(str(item.get("type", "unknown")) for item in focus_errors)),
            "focus_errors": focus_errors,
            "record_path": str(path),
            "model_token_usage": record.get("model_token_usage"),
            "recovered_from_record": True,
        }

    async def _run_game(
        self, node: EvolutionNode, evaluation_directory: Path, game_index: int
    ) -> dict[str, Any]:
        # 种子绑定节点和 game 编号，保证重跑时角色分配和引擎发牌可复现。
        assignment = self._planned_assignment(node, game_index)
        seed_text = str(assignment["seed"])
        role_nodes = dict(assignment["role_nodes"])

        rules = create_rules_for_player_count(
            self.config.evaluation.player_count,
            self.config.evaluation.optional_roles,
        )
        players = [
            {"id": f"p{index}", "name": f"Player {index}"}
            for index in range(1, self.config.evaluation.player_count + 1)
        ]
        participants = {
            player["id"]: VersionedTaskAgentParticipant(
                player_id=player["id"],
                model_client=self.model_client,
                role_nodes=role_nodes,
                archive=self.archive,
                module_loader=self.loader,
                request_coordinator=self.request_coordinator,
                max_decision_retries=self.config.evaluation.max_decision_retries,
                max_tool_calls_per_decision=self.config.evaluation.max_tool_calls_per_decision,
                max_tool_result_tokens=self.config.evaluation.max_tool_result_tokens,
                max_prompt_chars=self.config.evaluation.max_prompt_chars,
            )
            for player in players
        }
        game_id = f"s2-{node.node_id}-game{game_index}"
        record_store = RoundGameRecordStore(
            evaluation_directory / "games",
            # 每 10 局切分一个记录 round；节点评测结果仍按全局 game 编号保存。
            round_index=game_index // GAMES_PER_ROUND,
            game_index=game_index % GAMES_PER_ROUND,
        )
        assignment = {
            **assignment,
            "game_id": game_id,
        }
        assignment_path = evaluation_directory / "assignments" / f"game{game_index}.json"
        assignment_path.parent.mkdir(parents=True, exist_ok=True)
        assignment_path.write_text(
            json.dumps(assignment, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        try:
            # GameEngine/Runner 负责规则、可见性、昼夜流程和记录；评测器只收集结果。
            engine = GameEngine(
                game_id=game_id,
                players=players,
                rules=rules,
                seed=seed_text,
            )
            report = await GameRunner(
                engine=engine,
                participants=participants,
                record_store=record_store,
                decision_timeout_seconds=self.config.evaluation.decision_timeout_seconds,
            ).run()
            winner = report["public_state"].get("winner")
            errors = report.get("errors", [])
            role_assignments = engine.role_assignments()
            focus_players = {
                player_id
                for player_id, role in role_assignments.items()
                if role == node.role
            }
            focus_errors = [
                item for item in errors
                if str(item.get("player_id", "")) in focus_players
            ]
            focus_error_counts = dict(
                Counter(str(item.get("type", "unknown")) for item in focus_errors)
            )
            focus_decisions = sum(
                int(report.get("decision_count_by_player", {}).get(player_id, 0) or 0)
                for player_id in focus_players
            )
            focus_fallbacks = sum(
                int(report.get("fallback_count_by_player", {}).get(player_id, 0) or 0)
                for player_id in focus_players
            )
            result = {
                **assignment,
                "complete": winner in {"wolf", "village"},
                "winner": winner,
                "focus_won": winner == team_for_role(node.role),
                "decision_count": report.get("decision_count", 0),
                "fallback_count": report.get("fallback_count", 0),
                "fallback_rate": report.get("fallback_rate", 0.0),
                "error_counts": dict(Counter(str(item.get("type", "unknown")) for item in errors)),
                "errors": errors,
                "focus_player_ids": sorted(focus_players),
                "focus_decision_count": focus_decisions,
                "focus_fallback_count": focus_fallbacks,
                "focus_fallback_rate": focus_fallbacks / focus_decisions if focus_decisions else 0.0,
                "focus_error_counts": focus_error_counts,
                "focus_errors": focus_errors,
                "evaluation_signature": assignment.get("evaluation_signature"),
                "record_path": report.get("record_path"),
                "model_token_usage": report.get("model_token_usage"),
            }
        except Exception as error:
            report_error("对局失败", node=node.node_id, game=game_index, error=type(error).__name__)
            result = {
                **assignment,
                "complete": False,
                "winner": None,
                "focus_won": False,
                "error": f"{type(error).__name__}: {error}",
                "decision_count": 0,
                "fallback_count": 0,
                "fallback_rate": 0.0,
                "error_counts": {"game_exception": 1},
                "errors": [],
                "record_path": None,
            }
        result_path = evaluation_directory / "results" / f"game{game_index}.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return result

    def _summary(self, node: EvolutionNode, results: list[dict[str, Any]]) -> dict[str, Any]:
        # 筛选只淘汰明显不可用的候选，不要求统计上证明优于父节点。
        complete = [item for item in results if item.get("complete")]
        failed_games = len(results) - len(complete)
        wins = sum(bool(item.get("focus_won")) for item in complete)
        decisions = sum(
            int(item.get("focus_decision_count", item.get("decision_count", 0)) or 0)
            for item in results
        )
        fallbacks = sum(
            int(item.get("focus_fallback_count", item.get("fallback_count", 0)) or 0)
            for item in results
        )
        invalid_decisions = sum(
            int(
                item.get("focus_error_counts", item.get("error_counts", {})).get(
                    "invalid_decision", 0
                )
                or 0
            )
            + int(
                item.get("focus_error_counts", item.get("error_counts", {})).get(
                    "participant_error", 0
                )
                or 0
            )
            for item in results
        )
        fallback_rate = fallbacks / decisions if decisions else 0.0
        total_decisions = sum(int(item.get("decision_count", 0) or 0) for item in results)
        total_fallbacks = sum(int(item.get("fallback_count", 0) or 0) for item in results)
        total_invalid_decisions = sum(
            int(item.get("error_counts", {}).get("invalid_decision", 0) or 0)
            + int(item.get("error_counts", {}).get("participant_error", 0) or 0)
            for item in results
        )
        reasons: list[str] = []
        if failed_games > self.config.evaluation.max_failed_games:
            reasons.append(f"failed_games={failed_games} > {self.config.evaluation.max_failed_games}")
        if fallback_rate > self.config.evaluation.max_fallback_rate:
            reasons.append(
                f"fallback_rate={fallback_rate:.4f} > {self.config.evaluation.max_fallback_rate:.4f}"
            )
        if invalid_decisions > self.config.evaluation.max_invalid_decisions:
            reasons.append(
                f"invalid_decisions={invalid_decisions} > {self.config.evaluation.max_invalid_decisions}"
            )
        if self.config.evaluation.discard_all_losses and complete and wins == 0:
            reasons.append("all_completed_games_lost")
        return {
            "node_id": node.node_id,
            "role": node.role,
            "team": team_for_role(node.role),
            "game_count": len(results),
            "complete_game_count": len(complete),
            "failed_game_count": failed_games,
            "win_count": wins,
            "loss_count": len(complete) - wins,
            "win_rate": wins / len(complete) if complete else 0.0,
            "decision_count": decisions,
            "fallback_count": fallbacks,
            "fallback_rate": fallback_rate,
            "invalid_decision_count": invalid_decisions,
            "quality_scope": "focus_role_only",
            "total_decision_count": total_decisions,
            "total_fallback_count": total_fallbacks,
            "total_invalid_decision_count": total_invalid_decisions,
            "recommended_status": "discarded" if reasons else "retained",
            "status_reasons": reasons or ["passed_broad_filter"],
            "results": results,
            "model": self._model_manifest(),
        }

    def _model_manifest(self) -> dict[str, Any]:
        model = getattr(self.model_client, "config", None)
        if model is None:
            return {"available": False}
        return {
            "available": True,
            "model": getattr(model, "model", None),
            "protocol": getattr(model, "protocol", None),
            "reasoning_effort": getattr(model, "reasoning_effort", None),
            "enable_thinking": getattr(model, "enable_thinking", None),
            "max_output_tokens": getattr(model, "max_output_tokens", None),
        }

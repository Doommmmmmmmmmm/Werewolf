from __future__ import annotations

import asyncio
import json
from pathlib import Path
import tempfile
import unittest

from werewolf_game.season2.archive import EvolutionArchive, NodeStatus, core_snapshot_hash
from werewolf_game.season2.config import load_season2_config
from werewolf_game.season2.evolution import EvolutionManager
from werewolf_game.season2.evaluator import EvolutionEvaluator
from werewolf_game.season2.meta_agent import MetaAgent
from werewolf_game.season2.resources import MetaResourceCatalog
from werewolf_game.season2.runtime import CandidateModuleLoader
from werewolf_game.season2.scheduler import EvolutionScheduler
from werewolf_game.season2.smoke import smoke_test_candidate
from werewolf_game.season2.validation import (
    validate_candidate_files,
    validate_candidate_source,
    validate_task_agent_class,
)
from werewolf_game import GameEngine, GameRunner, create_rules_for_player_count
from werewolf_game.season2.runtime import VersionedTaskAgentParticipant


def write_config(root: Path, *, roles: list[str] | None = None) -> Path:
    config = {
        "paths": {
            "archive_root": str(root / "archive"),
            "evaluation_root": str(root / "evaluations"),
            "operation_root": str(root / "operations"),
            "external_knowledge_root": str(root / "knowledge"),
            "pi_root": "/opt/pi",
        },
        "evolution": {
            "roles": roles or ["wolf"],
            "random_seed": "test",
            "max_children_per_node": 2,
            "max_tree_depth": 3,
            "max_generated_candidates_per_role": 8,
            "max_successful_evolutions_per_role": 4,
            "children_per_expansion": 2,
            "role_scheduler": "least_evolved_random",
        },
        "evaluation": {
            "games_per_pending_node": 2,
            "player_count": 7,
            "optional_roles": [],
            "game_concurrency": 1,
            "model_max_in_flight": 1,
            "decision_timeout_seconds": 5,
            "max_decision_retries": 0,
            "max_tool_calls_per_decision": 1,
            "max_tool_result_tokens": 200,
            "max_failed_games": 0,
            "max_fallback_rate": 0.2,
            "max_invalid_decisions": 0,
            "discard_all_losses": True,
        },
        "meta": {
            "max_output_tokens": 1000,
            "max_tool_calls": 3,
            "max_tool_result_tokens": 2000,
            "max_attempts": 1,
            "required_resources": ["task", "boundaries", "random_game_replay"],
            "enabled_tools": [
                "list_resources",
                "read_evolution_tree",
                "list_skills",
                "read_skill",
            ],
        },
        "pi": {
            "command": ["./pi-test.sh"],
            "model": "",
            "timeout_seconds": 10,
            "max_attempts": 1,
            "use_bwrap": False,
            "candidate_smoke_test": False,
            "readonly_paths": [],
            "allowed_files": ["task_agent.py", "task.md"],
            "environment_allowlist": ["PATH"],
        },
    }
    path = root / "season2.json"
    path.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
    return path


class FakeMetaModel:
    def __init__(self) -> None:
        self.calls = 0

    async def complete_json(self, **kwargs):
        self.calls += 1
        tools = kwargs["tools"]
        executor = kwargs["tool_executor"]
        if tools:
            executor("list_resources", {})
        return {
            "summary": "需要让判断流程更清晰",
            "evidence": ["当前为最小基线"],
            "root_causes": ["缺少显式局势检查"],
            "modification_plan": [
                {"file": "task.md", "change": "加入局势检查", "rationale": "减少机械行动"}
            ],
            "risks": ["可能增加提示长度"],
            "code_agent_brief": "修改 task.md，加入简短局势检查。",
        }


class FakeMetaAgent:
    def __init__(self) -> None:
        self.calls = 0
        self.replay_game_indices: list[int | None] = []

    async def diagnose(
        self,
        *,
        node_id: str,
        output_path: Path,
        required_replay_item: dict | None = None,
    ):
        self.calls += 1
        self.replay_game_indices.append(
            required_replay_item.get("game_index") if required_replay_item else None
        )
        result = {
            "summary": "test",
            "evidence": [],
            "root_causes": [],
            "modification_plan": [
                {"file": "task.md", "change": "append", "rationale": "test"}
            ],
            "risks": [],
            "code_agent_brief": "append test",
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps({"diagnosis": result}), encoding="utf-8")
        return result


class FakeCodeResult:
    returncode = 0


class FakeCodeAgent:
    async def apply(self, *, workspace: Path, role: str, diagnosis, operation_directory):
        path = workspace / "task.md"
        path.write_text(path.read_text(encoding="utf-8") + "\n测试修改。\n", encoding="utf-8")
        validate_candidate_files(workspace, ("task_agent.py", "task.md"))
        validate_candidate_source(workspace)
        return FakeCodeResult()


class FakeEvaluator:
    def __init__(self, root: Path) -> None:
        self.root = root

    async def evaluate(self, node_id: str):
        path = self.root / f"{node_id}-summary.json"
        results = []
        # 为批量分支诊断提供足够的、彼此不同的可读取回放。
        for game_index in range(2):
            record_path = self.root / f"{node_id}-game{game_index}.json"
            record_path.write_text(
                json.dumps(
                    {
                        "metadata": {"game_id": f"fake-{node_id}-{game_index}"},
                        "events": [{"kind": "fake_event", "game_index": game_index}],
                        "final_snapshot": {"winner": "village"},
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            results.append(
                {
                    "game_index": game_index,
                    "complete": True,
                    "winner": "village",
                    "focus_won": True,
                    "record_path": str(record_path),
                }
            )
        summary = {
            "node_id": node_id,
            "recommended_status": "retained",
            "status_reasons": ["passed_broad_filter"],
            "results": results,
        }
        path.write_text(json.dumps(summary), encoding="utf-8")
        return summary, path


class LegalGameModel:
    async def complete_json(self, **kwargs):
        user = json.loads(kwargs["messages"][-1]["content"])
        request = user["packet"]["request"]
        allowed = request["allowed_actions"][0]
        action = {"kind": allowed["kind"]}
        if allowed["kind"] in {"speak", "last_words"}:
            action["text"] = "好"
        elif allowed.get("target_ids"):
            action["target_id"] = allowed["target_ids"][0]
        return action


class Season2EvolutionTest(unittest.TestCase):
    def test_config_archive_and_candidate_loader(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = load_season2_config(write_config(root))
            archive = EvolutionArchive(config)
            archive.initialize()
            manifest = json.loads((root / "archive" / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["core_snapshot_hash"], core_snapshot_hash())
            base = archive.node("wolf-base")
            self.assertEqual(base.status, NodeStatus.RETAINED)
            self.assertEqual(archive.retained_nodes("wolf")[0].node_id, "wolf-base")
            candidate = CandidateModuleLoader(archive).task_agent_class("wolf-base")
            validate_task_agent_class(candidate)

    def test_archive_rejects_changed_core_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = load_season2_config(write_config(root))
            archive = EvolutionArchive(config)
            archive.initialize()
            import unittest.mock

            with unittest.mock.patch(
                "werewolf_game.season2.archive.core_snapshot_hash",
                return_value="changed-core",
            ):
                with self.assertRaisesRegex(ValueError, "Game core 已发生变化"):
                    EvolutionArchive(config)

    def test_base_node_cannot_be_discarded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = load_season2_config(write_config(root))
            archive = EvolutionArchive(config)
            archive.initialize()
            with self.assertRaisesRegex(ValueError, "base 节点不可修改为 discarded"):
                archive.set_status("wolf-base", NodeStatus.DISCARDED, reason="test")
            self.assertEqual(archive.node("wolf-base").status, NodeStatus.RETAINED)

    def test_generated_and_successful_counts_are_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = load_season2_config(write_config(root))
            archive = EvolutionArchive(config)
            archive.initialize()
            first = archive.create_child("wolf-base")
            second = archive.create_child("wolf-base")
            self.assertEqual(archive.role_generated_candidate_count("wolf"), 2)
            self.assertEqual(archive.role_successful_evolution_count("wolf"), 0)
            archive.set_status(first.node_id, NodeStatus.RETAINED, reason="test")
            archive.set_status(second.node_id, NodeStatus.DISCARDED, reason="test")
            self.assertEqual(archive.role_generated_candidate_count("wolf"), 2)
            self.assertEqual(archive.role_successful_evolution_count("wolf"), 1)

    def test_scheduler_uses_successful_count_not_generated_count(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = load_season2_config(
                write_config(root, roles=["wolf", "seer"])
            )
            archive = EvolutionArchive(config)
            archive.initialize()
            wolf_child = archive.create_child("wolf-base")
            archive.set_status(wolf_child.node_id, NodeStatus.RETAINED, reason="test")
            archive.create_child("seer-base")
            archive.create_child("seer-base")
            selected = EvolutionScheduler(archive, config).select()
            self.assertIsNotNone(selected)
            self.assertEqual(selected.role, "seer")

    def test_scheduler_uses_pending_node_at_greater_depth(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = load_season2_config(write_config(root))
            archive = EvolutionArchive(config)
            archive.initialize()
            child = archive.create_child("wolf-base")
            selected = EvolutionScheduler(archive, config).select()
            self.assertIsNotNone(selected)
            self.assertEqual(selected.node_id, child.node_id)

    def test_meta_agent_uses_bounded_resource_catalog(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = load_season2_config(write_config(root))
            archive = EvolutionArchive(config)
            archive.initialize()
            output = root / "diagnosis.json"
            model = FakeMetaModel()
            result = asyncio.run(
                MetaAgent(config=config, archive=archive, model_client=model).diagnose(
                    node_id="wolf-base", output_path=output
                )
            )
            self.assertEqual(result["summary"], "需要让判断流程更清晰")
            artifact = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(artifact["tool_trace"][0]["tool"], "list_resources")

    def test_meta_agent_contract_keeps_evolution_details_out_of_required_prompt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = load_season2_config(write_config(root))
            archive = EvolutionArchive(config)
            archive.initialize()
            contract = MetaAgent(
                config=config, archive=archive, model_client=FakeMetaModel()
            )._task_agent_contract()
            for expected in (
                "最多调用 1 次受限工具",
                "最多 200 个长度单位",
                "最多 12000 个 Unicode 字符",
                "Meta-Agent 每次诊断最多调用 3 次",
                "Pi 只能修改候选工作区中的 task_agent.py, task.md",
            ):
                self.assertIn(expected, contract)
            self.assertIn("read_evolution_contract", contract)
            self.assertNotIn("每节点评测 2 局", contract)
            self.assertNotIn("树最大深度 3", contract)

    def test_evolution_contract_is_available_as_optional_tool(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_data = json.loads(write_config(root).read_text(encoding="utf-8"))
            config_data["meta"]["enabled_tools"].append("read_evolution_contract")
            config_path = root / "season2.json"
            config_path.write_text(json.dumps(config_data, ensure_ascii=False), encoding="utf-8")
            config = load_season2_config(config_path)
            archive = EvolutionArchive(config)
            archive.initialize()
            catalog = MetaResourceCatalog(config=config, archive=archive, node_id="wolf-base")
            contract = catalog.execute("read_evolution_contract", {})
            self.assertEqual(contract["current_node"]["status"], "retained")
            self.assertEqual(contract["evolution_budget"]["children_per_expansion"], 2)
            self.assertFalse(contract["evaluation"]["parent_child_comparison_required"])

    def test_external_skill_cards_are_meta_only_resources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = write_config(root)
            config_data = json.loads(config_path.read_text(encoding="utf-8"))
            skill_root = root / "knowledge"
            skill_root.mkdir(parents=True)
            (skill_root / "fake.md").write_text(
                "狼人应根据票型切换策略。", encoding="utf-8"
            )
            config_data["paths"]["external_knowledge_root"] = str(skill_root)
            config_path.write_text(json.dumps(config_data, ensure_ascii=False), encoding="utf-8")
            config = load_season2_config(config_path)
            archive = EvolutionArchive(config)
            archive.initialize()
            catalog = MetaResourceCatalog(config=config, archive=archive, node_id="wolf-base")
            resources = catalog.execute("list_resources", {})
            self.assertEqual(resources["external_knowledge"]["tool"], "list_skills/read_skill")
            listed = catalog.execute("list_skills", {})
            self.assertEqual(listed["skills"][0]["name"], "fake")
            result = catalog.execute("read_skill", {"skill_name": "fake"})
            self.assertEqual(result["name"], "fake")
            self.assertIn("票型", result["content"])
            self.assertEqual(catalog.execute("read_skill", {"skill_name": "../fake"})["error"], "skill_name 不合法")

    def test_manager_generates_then_evaluates_child(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = load_season2_config(write_config(root))
            archive = EvolutionArchive(config)
            meta_agent = FakeMetaAgent()
            manager = EvolutionManager(
                config=config,
                archive=archive,
                meta_agent=meta_agent,
                code_agent=FakeCodeAgent(),
                evaluator=FakeEvaluator(root),
            )
            generated = asyncio.run(manager.step())
            self.assertEqual(generated["action"], "evolved")
            self.assertEqual(generated["retained_child_id"], "wolf-n00001")
            self.assertEqual(meta_agent.calls, config.evolution.children_per_expansion)
            self.assertEqual(
                len(set(meta_agent.replay_game_indices)),
                config.evolution.children_per_expansion,
            )
            child_ids = ["wolf-n00001", "wolf-n00002"]
            self.assertEqual(len(child_ids), config.evolution.children_per_expansion)
            self.assertEqual(archive.node("wolf-n00001").status, NodeStatus.RETAINED)
            self.assertEqual(archive.node("wolf-n00002").status, NodeStatus.PENDING)
            evaluated = asyncio.run(manager.step())
            self.assertEqual(evaluated["action"], "evolved")
            self.assertEqual(evaluated["node_id"], "wolf-n00002")
            self.assertEqual(archive.node("wolf-n00002").status, NodeStatus.RETAINED)

    def test_versioned_candidates_complete_a_game(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            roles = ["wolf", "villager", "seer", "witch"]
            config = load_season2_config(write_config(root, roles=roles))
            archive = EvolutionArchive(config)
            archive.initialize()
            role_nodes = {role: f"{role}-base" for role in roles}
            players = [
                {"id": f"p{index}", "name": f"P{index}"}
                for index in range(1, 8)
            ]
            model = LegalGameModel()
            participants = {
                player["id"]: VersionedTaskAgentParticipant(
                    player_id=player["id"],
                    model_client=model,
                    role_nodes=role_nodes,
                    archive=archive,
                    max_decision_retries=0,
                    max_tool_calls_per_decision=0,
                )
                for player in players
            }
            report = asyncio.run(
                GameRunner(
                    engine=GameEngine(
                        game_id="candidate-smoke",
                        players=players,
                        rules=create_rules_for_player_count(7),
                        seed="candidate-smoke",
                    ),
                    participants=participants,
                    record_store=False,
                ).run()
            )
            self.assertIn(report["public_state"]["winner"], {"wolf", "village"})
            self.assertEqual(report["fallback_count"], 0)

    def test_evaluator_runs_configured_games_and_writes_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            roles = ["wolf", "villager", "seer", "witch"]
            config = load_season2_config(write_config(root, roles=roles))
            archive = EvolutionArchive(config)
            archive.initialize()
            child = archive.create_child("wolf-base")
            evaluator = EvolutionEvaluator(
                config=config,
                archive=archive,
                model_client=LegalGameModel(),
                request_coordinator=None,
            )
            summary, summary_path = asyncio.run(evaluator.evaluate(child.node_id))
            self.assertEqual(summary["game_count"], 2)
            self.assertEqual(summary["complete_game_count"], 2)
            self.assertTrue(summary_path.exists())

    def test_reuse_keeps_source_game_index_alignment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = load_season2_config(
                write_config(root, roles=["wolf", "villager", "seer", "witch"])
            )
            archive = EvolutionArchive(config)
            archive.initialize()
            child = archive.create_child("wolf-base")
            evaluator = EvolutionEvaluator(
                config=config,
                archive=archive,
                model_client=LegalGameModel(),
                request_coordinator=None,
            )
            source_dir = root / "evaluations" / "villager" / "villager-base" / "results"
            source_dir.mkdir(parents=True)
            source_record = source_dir.parent / "source-record.json"
            source_record.write_text(
                json.dumps(
                    {
                        "final_snapshot": {
                            "public_state": {"status": "finished", "winner": "village"},
                            "audit_state": {"roles": {"p1": "wolf"}},
                        },
                        "events": [],
                        "runner_events": [],
                    }
                ),
                encoding="utf-8",
            )
            planned_1 = evaluator._planned_assignment(child, 1)
            (source_dir / "game1.json").write_text(
                json.dumps(
                    {
                        **planned_1,
                        "game_index": 1,
                        "record_path": str(source_record),
                    }
                ),
                encoding="utf-8",
            )
            self.assertIsNone(
                evaluator._find_reusable_record(
                    child,
                    {**planned_1, "game_index": 3},
                    evaluation_directory=root / "evaluations" / "wolf" / child.node_id,
                    used_records=set(),
                )
            )
            reused = evaluator._find_reusable_record(
                child,
                planned_1,
                evaluation_directory=root / "evaluations" / "wolf" / child.node_id,
                used_records=set(),
            )
            self.assertIsNotNone(reused)
            self.assertEqual(reused["game_index"], 1)

    def test_candidate_smoke_uses_no_external_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            roles = ["wolf", "villager", "seer", "witch"]
            config = load_season2_config(write_config(root, roles=roles))
            archive = EvolutionArchive(config)
            archive.initialize()
            child = archive.create_child("wolf-base")
            result = asyncio.run(
                smoke_test_candidate(
                    config=config,
                    archive=archive,
                    focus_node_id=child.node_id,
                )
            )
            self.assertEqual(result["fallback_count"], 0)
            self.assertIn(result["winner"], {"wolf", "village"})


if __name__ == "__main__":
    unittest.main()

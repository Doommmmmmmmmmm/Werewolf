from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
import unittest

from werewolf_game import (
    GameEngine,
    RoundSkillVersionStore,
    SkillTestRunner,
    SkillTestScenario,
    SkillTestSource,
    create_default_rules,
)
from werewolf_game.constants import ACTION_LAST_WORDS, ACTION_PASS, ACTION_SPEAK, ROLE_WOLF
from werewolf_game.participants import ScriptedParticipant
from werewolf_game.prompts import PROMPT_DIRECTORY, RoleStrategyStore


def copied_strategy_store(directory: str) -> RoleStrategyStore:
    prompt_root = Path(directory) / "prompts"
    shutil.copytree(PROMPT_DIRECTORY / "roles", prompt_root / "roles")
    return RoleStrategyStore(prompt_root)


def deterministic_strategy(packet: dict) -> dict:
    allowed = packet["request"]["allowed_actions"]
    action = next((item for item in allowed if item["kind"] != ACTION_PASS), allowed[0])
    if action["kind"] in {ACTION_SPEAK, ACTION_LAST_WORDS}:
        return {"kind": action["kind"], "text": "我会依据公开票型继续判断。"}
    result = {"kind": action["kind"]}
    if action.get("target_ids"):
        result["target_id"] = action["target_ids"][0]
    return result


class SkillTestRunnerTest(unittest.IsolatedAsyncioTestCase):
    async def test_mixed_skill_test_uses_frozen_snapshots_without_reviewing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source_store = copied_strategy_store(temporary)
            rules = create_default_rules()
            roles = set(rules.role_deck)
            source_versions = RoundSkillVersionStore(temporary, round_index=0)
            source_versions.ensure_input(roles=roles, strategy_store=source_store)
            initial_wolf_strategy = source_store.profile(ROLE_WOLF).strategy

            for role in roles:
                source_store.replace_strategy(
                    role,
                    "# 测试用最新策略\n\n"
                    f"- LATEST-{role.upper()}-MARKER 用于验证冻结快照来源。\n"
                    "- 每次判断都应回看公开票型和反证，避免把经验当成规则。",
                )
            source_versions.capture_output(roles=roles, strategy_store=source_store)
            source_after_training = {
                role: source_store.profile(role).strategy for role in roles
            }
            factory_calls: list[tuple[str, int, RoleStrategyStore]] = []

            def game_factory(
                scenario_id: str, game_index: int, strategy_store: RoleStrategyStore
            ) -> tuple[GameEngine, dict]:
                factory_calls.append((scenario_id, game_index, strategy_store))
                players = [
                    {"id": f"p{index}", "name": f"Player {index}"}
                    for index in range(1, 8)
                ]
                engine = GameEngine(
                    game_id=f"test1-{scenario_id}-game{game_index}",
                    players=players,
                    rules=rules,
                    seed=f"skill-test-{scenario_id}-{game_index}",
                )
                participants = {
                    player["id"]: ScriptedParticipant(
                        player["id"], deterministic_strategy
                    )
                    for player in players
                }
                return engine, participants

            report = await SkillTestRunner(
                game_factory=game_factory,
                roles=rules.role_deck,
                record_directory=temporary,
                test_id="test1",
                initial_round=0,
                latest_round=0,
                decision_timeout_seconds=1,
                game_concurrency=3,
            ).run()

            self.assertFalse(report["skill_updated"])
            self.assertEqual(len(factory_calls), 20)
            self.assertEqual(source_store.profile(ROLE_WOLF).strategy, source_after_training[ROLE_WOLF])
            test_directory = Path(report["test_directory"])
            self.assertTrue((test_directory / "log" / "test-manifest.json").exists())
            self.assertTrue((test_directory / "log" / "test-summary.json").exists())
            self.assertFalse((test_directory / "review").exists())

            expected_versions = {
                "wolf-latest-vs-others-initial": {ROLE_WOLF: 1},
                "wolf-initial-vs-others-latest": {ROLE_WOLF: 0},
            }
            for scenario_id, expected in expected_versions.items():
                scenario_directory = test_directory / scenario_id
                self.assertTrue((scenario_directory / "log" / "skill-sources.json").exists())
                self.assertTrue((scenario_directory / "log" / "test-summary.json").exists())
                for game_index in range(10):
                    full_path = scenario_directory / "log" / f"full-game{game_index}.json"
                    self.assertTrue(full_path.exists())
                    record = json.loads(full_path.read_text(encoding="utf-8"))
                    self.assertEqual(record["final_snapshot"]["public_state"]["status"], "finished")
                    sources = record["metadata"]["skill_test"]["skill_sources"]
                    self.assertEqual(sources[ROLE_WOLF]["version"], expected[ROLE_WOLF])
                    for role in roles - {ROLE_WOLF}:
                        expected_version = 0 if expected[ROLE_WOLF] == 1 else 1
                        self.assertEqual(sources[role]["version"], expected_version)

            latest_store = RoleStrategyStore(
                test_directory / "wolf-latest-vs-others-initial" / "skill"
            )
            initial_store = RoleStrategyStore(
                test_directory / "wolf-initial-vs-others-latest" / "skill"
            )
            self.assertIn("LATEST-WOLF-MARKER", latest_store.profile(ROLE_WOLF).strategy)
            self.assertEqual(initial_store.profile(ROLE_WOLF).strategy, initial_wolf_strategy)

            second = await SkillTestRunner(
                game_factory=game_factory,
                roles=rules.role_deck,
                record_directory=temporary,
                test_id="test1",
                initial_round=0,
                latest_round=0,
                decision_timeout_seconds=1,
                game_concurrency=3,
            ).run()
            self.assertEqual(len(factory_calls), 20)
            self.assertTrue(all(game["reused"] for scenario in second["scenarios"] for game in scenario["games"]))

    async def test_progression_sources_and_custom_game_count_are_frozen(self) -> None:
        """支持初始、round10、round20 的四组阵营交叉对照。"""

        with tempfile.TemporaryDirectory() as temporary:
            source_store = copied_strategy_store(temporary)
            rules = create_default_rules()
            roles = set(rules.role_deck)

            initial_versions = RoundSkillVersionStore(temporary, round_index=0)
            initial_versions.ensure_input(roles=roles, strategy_store=source_store)
            initial_wolf_strategy = source_store.profile(ROLE_WOLF).strategy

            round10_versions = RoundSkillVersionStore(temporary, round_index=9)
            round10_versions.ensure_input(roles=roles, strategy_store=source_store)
            for role in roles:
                source_store.replace_strategy(
                    role,
                    f"# ROUND10-{role.upper()}\n\n"
                    "- 固定的十轮策略快照，仅用于验证历史版本在测试中被正确冻结。\n"
                    "- 行动前结合公开发言、票型、夜间结果与角色能力进行交叉判断。\n",
                )
            round10_versions.capture_output(roles=roles, strategy_store=source_store)

            round20_versions = RoundSkillVersionStore(temporary, round_index=19)
            round20_versions.ensure_input(roles=roles, strategy_store=source_store)
            for role in roles:
                source_store.replace_strategy(
                    role,
                    f"# ROUND20-{role.upper()}\n\n"
                    "- 固定的二十轮策略快照，仅用于验证历史版本在测试中被正确冻结。\n"
                    "- 行动前结合公开发言、票型、夜间结果与角色能力进行交叉判断。\n",
                )
            round20_versions.capture_output(roles=roles, strategy_store=source_store)
            current_strategies = {
                role: source_store.profile(role).strategy for role in roles
            }

            scenarios = (
                SkillTestScenario(
                    scenario_id="good-round20-vs-wolf-initial",
                    description="好人 round20 对初始狼人。",
                    wolf_source="initial",
                    other_source="round20",
                ),
                SkillTestScenario(
                    scenario_id="good-round20-vs-wolf-round10",
                    description="好人 round20 对 round10 狼人。",
                    wolf_source="round10",
                    other_source="round20",
                ),
                SkillTestScenario(
                    scenario_id="wolf-round20-vs-good-initial",
                    description="round20 狼人对初始好人。",
                    wolf_source="round20",
                    other_source="initial",
                ),
                SkillTestScenario(
                    scenario_id="wolf-round20-vs-good-round10",
                    description="round20 狼人对 round10 好人。",
                    wolf_source="round20",
                    other_source="round10",
                ),
            )
            sources = (
                SkillTestSource("initial", 0, "input"),
                SkillTestSource("round10", 9, "output"),
                SkillTestSource("round20", 19, "output"),
            )
            factory_calls: list[tuple[str, int, RoleStrategyStore]] = []

            def game_factory(
                scenario_id: str, game_index: int, strategy_store: RoleStrategyStore
            ) -> tuple[GameEngine, dict]:
                factory_calls.append((scenario_id, game_index, strategy_store))
                players = [
                    {"id": f"p{index}", "name": f"Player {index}"}
                    for index in range(1, 8)
                ]
                engine = GameEngine(
                    game_id=f"progression-{scenario_id}-game{game_index}",
                    players=players,
                    rules=rules,
                    seed=f"progression-{scenario_id}-{game_index}",
                )
                participants = {
                    player["id"]: ScriptedParticipant(
                        player["id"], deterministic_strategy
                    )
                    for player in players
                }
                return engine, participants

            runner_kwargs = {
                "game_factory": game_factory,
                "roles": rules.role_deck,
                "record_directory": temporary,
                "test_id": "test2",
                "initial_round": 0,
                "latest_round": 19,
                "decision_timeout_seconds": 1,
                "game_concurrency": 2,
                "game_count_per_scenario": 2,
                "scenarios": scenarios,
                "source_definitions": sources,
            }
            report = await SkillTestRunner(**runner_kwargs).run()

            self.assertFalse(report["skill_updated"])
            self.assertEqual(len(factory_calls), 8)
            self.assertEqual(report["game_count_per_scenario"], 2)
            self.assertEqual(
                source_store.profile(ROLE_WOLF).strategy, current_strategies[ROLE_WOLF]
            )
            test_directory = Path(report["test_directory"])
            manifest = json.loads(
                (test_directory / "log" / "test-manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(manifest["game_count_per_scenario"], 2)
            self.assertEqual(
                {item["source_id"] for item in manifest["source_definitions"]},
                {"initial", "round10", "round20"},
            )

            expected_sources = {
                "good-round20-vs-wolf-initial": (0, 19),
                "good-round20-vs-wolf-round10": (9, 19),
                "wolf-round20-vs-good-initial": (19, 0),
                "wolf-round20-vs-good-round10": (19, 9),
            }
            for scenario_id, (wolf_round, other_round) in expected_sources.items():
                scenario_directory = test_directory / scenario_id
                summary = json.loads(
                    (scenario_directory / "log" / "test-summary.json").read_text(
                        encoding="utf-8"
                    )
                )
                self.assertEqual(summary["game_count"], 2)
                self.assertTrue(summary["completed"])
                for game_index in range(2):
                    record = json.loads(
                        (
                            scenario_directory / "log" / f"full-game{game_index}.json"
                        ).read_text(encoding="utf-8")
                    )
                    source_info = record["metadata"]["skill_test"]["skill_sources"]
                    self.assertEqual(source_info[ROLE_WOLF]["source_round"], wolf_round)
                    for role in roles - {ROLE_WOLF}:
                        self.assertEqual(source_info[role]["source_round"], other_round)

            staged = RoleStrategyStore(
                test_directory / "good-round20-vs-wolf-round10" / "skill"
            )
            self.assertIn("ROUND10-WOLF", staged.profile(ROLE_WOLF).strategy)
            for role in roles - {ROLE_WOLF}:
                self.assertIn(
                    f"ROUND20-{role.upper()}", staged.profile(role).strategy
                )
            initial_staged = RoleStrategyStore(
                test_directory / "good-round20-vs-wolf-initial" / "skill"
            )
            self.assertEqual(
                initial_staged.profile(ROLE_WOLF).strategy, initial_wolf_strategy
            )

            second = await SkillTestRunner(**runner_kwargs).run()
            self.assertEqual(len(factory_calls), 8)
            self.assertTrue(
                all(
                    game["reused"]
                    for scenario in second["scenarios"]
                    for game in scenario["games"]
                )
            )

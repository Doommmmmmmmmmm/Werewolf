from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
import unittest

from werewolf_game import (
    GameEngine,
    GameRoundRunner,
    RoleStrategyReviewer,
    RoundSkillVersionStore,
    create_default_rules,
    review_completed_round,
)
from werewolf_game.constants import ACTION_LAST_WORDS, ACTION_PASS, ACTION_SPEAK, ROLE_SEER, ROLE_WOLF
from werewolf_game.llm import ModelRequestCoordinator
from werewolf_game.participants import ScriptedParticipant
from werewolf_game.prompts import PROMPT_DIRECTORY, RoleStrategyStore
from werewolf_game.records import RoundGameRecordStore


class FakeReviewClient:
    def __init__(self) -> None:
        self.requests: list[dict] = []

    async def complete_json(self, **request: object) -> dict:
        self.requests.append(request)  # type: ignore[arg-type]
        system = str(request.get("system", ""))
        if "批次阅读阶段" in system:
            payload = json.loads(request["messages"][0]["content"])  # type: ignore[index]
            return {
                "game_notes": [
                    {
                        "game_id": replay["game_id"],
                        "strengths": ["能记录该局的关键公开线索。"],
                        "weaknesses": ["仍需避免单局过度归因。"],
                        "opponent_vulnerabilities": ["对手的票型解释存在断点。"],
                        "strategy_vulnerabilities": ["策略需要更多反证检查。"],
                    }
                    for replay in payload["complete_game_replays"]
                ],
                "batch_patterns": ["先保留各局证据，再做跨局归纳。"],
            }
        return {
            "strengths": ["能在关键票型前整理公开证据。"],
            "weaknesses": ["部分判断过早依赖单局直觉。"],
            "opponent_vulnerabilities": ["对手在票型变化后解释不足。"],
            "strategy_vulnerabilities": ["缺少对失败样本的反证检查。"],
            "strategy_markdown": "# 更新后的经验策略\n\n- 先比较可复查的票型，再用新一局结果检验自己的假设。\n- 对失败样本保留反证，并在下一局根据新证据更新优先级。",
        }


def copied_strategy_store(directory: str) -> RoleStrategyStore:
    prompt_root = Path(directory) / "prompts"
    shutil.copytree(PROMPT_DIRECTORY / "roles", prompt_root / "roles")
    return RoleStrategyStore(prompt_root)


def ten_minimal_records() -> list[dict]:
    return [
        {
            "metadata": {"game_id": f"round0-game{game_index}"},
            "events": [],
            "runner_events": [],
            "final_snapshot": {"audit_state": {"roles": {}}},
        }
        for game_index in range(10)
    ]


def deterministic_strategy(packet: dict) -> dict:
    allowed = packet["request"]["allowed_actions"]
    action = next((item for item in allowed if item["kind"] != ACTION_PASS), allowed[0])
    if action["kind"] in {ACTION_SPEAK, ACTION_LAST_WORDS}:
        return {"kind": action["kind"], "text": "我会依据公开票型继续判断。"}
    result = {"kind": action["kind"]}
    if action.get("target_ids"):
        result["target_id"] = action["target_ids"][0]
    return result


class FailsFirstDecisionParticipant(ScriptedParticipant):
    """每局制造一次可恢复的 Runner 回退，用于训练质量闸门测试。"""

    def __init__(self, player_id: str) -> None:
        super().__init__(player_id, deterministic_strategy)
        self.has_failed = False

    async def decide(self, turn_packet: dict) -> dict:
        if not self.has_failed:
            self.has_failed = True
            raise TimeoutError("模拟模型行动超时")
        return await super().decide(turn_packet)


class RoleStrategyReviewerTest(unittest.IsolatedAsyncioTestCase):
    async def test_reviewer_only_receives_own_profile_and_ten_replays(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = copied_strategy_store(temporary)
            original_wolf_base = store.profile(ROLE_WOLF).base
            store.replace_strategy(
                ROLE_SEER,
                "# 仅用于隔离测试的预言家策略\n\n- SECRET-SEER-STRATEGY 不应进入狼人复盘上下文。",
            )
            client = FakeReviewClient()
            reviewer = RoleStrategyReviewer(model_client=client, strategy_store=store)

            result = await reviewer.review_role(
                round_index=0,
                role=ROLE_WOLF,
                game_records=ten_minimal_records(),
            )

            self.assertTrue(result.updated, result.error)
            self.assertEqual(store.profile(ROLE_WOLF).base, original_wolf_base)
            self.assertIn("更新后的经验策略", store.profile(ROLE_WOLF).strategy)
            payload = json.loads(client.requests[0]["messages"][0]["content"])
            self.assertEqual(payload["role"], ROLE_WOLF)
            self.assertEqual(len(payload["complete_game_replays"]), 2)
            self.assertIn("audit_replay_markdown", payload["complete_game_replays"][0])
            self.assertEqual(set(payload["own_role_markdown"]), {"base_md", "strategy_md"})
            self.assertNotIn("SECRET-SEER-STRATEGY", json.dumps(payload, ensure_ascii=False))
            self.assertNotIn("预言家固定规则", json.dumps(payload, ensure_ascii=False))
            final_payload = json.loads(client.requests[-1]["messages"][0]["content"])
            self.assertEqual(len(final_payload["batch_analyses"]), 5)

    async def test_reviewer_rejects_non_ten_game_input_without_changing_strategy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = copied_strategy_store(temporary)
            before = store.profile(ROLE_WOLF).strategy
            reviewer = RoleStrategyReviewer(
                model_client=FakeReviewClient(), strategy_store=store
            )
            result = await reviewer.review_role(
                round_index=0,
                role=ROLE_WOLF,
                game_records=ten_minimal_records()[:9],
            )
            self.assertFalse(result.updated)
            self.assertIn("10 局", result.error or "")
            self.assertEqual(store.profile(ROLE_WOLF).strategy, before)


class GameRoundRunnerTest(unittest.IsolatedAsyncioTestCase):
    async def test_ten_games_write_round_layout_then_review_per_role(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = copied_strategy_store(temporary)
            client = FakeReviewClient()
            reviewer = RoleStrategyReviewer(model_client=client, strategy_store=store)

            def game_factory(round_index: int, game_index: int) -> tuple[GameEngine, dict]:
                players = [
                    {"id": f"p{player_index}", "name": f"Player {player_index}"}
                    for player_index in range(1, 8)
                ]
                engine = GameEngine(
                    game_id=f"round{round_index}-game{game_index}",
                    players=players,
                    rules=create_default_rules(),
                    seed=f"training-{round_index}-{game_index}",
                )
                participants = {
                    player["id"]: ScriptedParticipant(
                        player["id"], deterministic_strategy
                    )
                    for player in players
                }
                return engine, participants

            report = await GameRoundRunner(
                game_factory=game_factory,
                record_directory=temporary,
                reviewer=reviewer,
                decision_timeout_seconds=1,
                game_concurrency=3,
            ).run_round(0)

            round_directory = Path(report["round_directory"])
            self.assertEqual(report["round_index"], 0)
            self.assertEqual(len(report["games"]), 10)
            self.assertTrue(all(not game["errors"] for game in report["games"]))
            for game_index in range(10):
                self.assertTrue((round_directory / "public" / f"game{game_index}.md").exists())
                self.assertTrue((round_directory / "full" / f"game{game_index}.md").exists())
                self.assertTrue((round_directory / "log" / f"public-game{game_index}.json").exists())
                full_path = round_directory / "log" / f"full-game{game_index}.json"
                self.assertTrue(full_path.exists())
                record = json.loads(full_path.read_text(encoding="utf-8"))
                self.assertEqual(record["metadata"]["training_round"], 0)
                self.assertEqual(record["metadata"]["game_index"], game_index)

            reviewed_roles = {review["role"] for review in report["reviews"]}
            self.assertEqual(reviewed_roles, {"wolf", "villager", "seer", "witch"})
            self.assertTrue(all(review["updated"] for review in report["reviews"]))
            for role in reviewed_roles:
                self.assertTrue((round_directory / "review" / f"role-{role}.md").exists())
                self.assertTrue((round_directory / "log" / f"review-{role}.json").exists())
            manifest_path = round_directory / "log" / "round-review.json"
            self.assertTrue(manifest_path.exists())
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual({review["role"] for review in manifest["reviews"]}, reviewed_roles)
            skill_manifest_path = round_directory / "log" / "skill-version.json"
            self.assertTrue(skill_manifest_path.exists())
            self.assertEqual(report["skill_version_manifest_path"], str(skill_manifest_path))
            self.assertEqual(manifest["skill_version_manifest_path"], str(skill_manifest_path))
            skill_manifest = json.loads(skill_manifest_path.read_text(encoding="utf-8"))
            for role in reviewed_roles:
                input_skill = skill_manifest["roles"][role]["input"]
                output_skill = skill_manifest["roles"][role]["output"]
                self.assertEqual(input_skill["version"], 0)
                self.assertEqual(output_skill["version"], 1)
                input_base = round_directory / input_skill["base_path"]
                output_base = round_directory / output_skill["base_path"]
                output_strategy = round_directory / output_skill["strategy_path"]
                self.assertTrue(input_base.exists())
                self.assertTrue(output_base.exists())
                self.assertEqual(
                    input_base.read_text(encoding="utf-8"),
                    output_base.read_text(encoding="utf-8"),
                )
                self.assertIn("更新后的经验策略", output_strategy.read_text(encoding="utf-8"))
            self.assertEqual(len(client.requests), len(reviewed_roles) * 6)
            batch_requests = [
                request for request in client.requests if "批次阅读阶段" in str(request["system"])
            ]
            self.assertEqual(len(batch_requests), len(reviewed_roles) * 5)
            for request in batch_requests:
                payload = json.loads(request["messages"][0]["content"])
                self.assertEqual(len(payload["complete_game_replays"]), 2)

            second_attempt = await review_completed_round(
                record_directory=temporary,
                round_index=0,
                reviewer=reviewer,
            )
            self.assertTrue(second_attempt["completed"])
            self.assertTrue(second_attempt["already_reviewed"])
            self.assertEqual(len(client.requests), len(reviewed_roles) * 6)
            self.assertEqual(RoundGameRecordStore.next_available(temporary), (1, 0))

            resumed = await GameRoundRunner(
                game_factory=game_factory,
                record_directory=temporary,
                reviewer=reviewer,
                decision_timeout_seconds=1,
                game_concurrency=3,
                resume_completed_games=True,
            ).run_round(0)
            self.assertTrue(all(game.get("reused") for game in resumed["games"]))
            self.assertFalse(resumed["review_performed"])
            self.assertGreater(resumed["quality"]["metrics"]["decision_count"], 0)
            self.assertEqual(resumed["quality"]["metrics"]["fallback_count"], 0)
            self.assertEqual(len(client.requests), len(reviewed_roles) * 6)

    async def test_incomplete_round_does_not_request_a_review_model(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = copied_strategy_store(temporary)
            client = FakeReviewClient()
            result = await review_completed_round(
                record_directory=temporary,
                round_index=0,
                reviewer=RoleStrategyReviewer(model_client=client, strategy_store=store),
            )
            self.assertFalse(result["completed"])
            self.assertFalse(result["already_reviewed"])
            self.assertEqual(len(result["missing_records"]), 10)
            self.assertEqual(client.requests, [])

    async def test_quality_gate_blocks_skill_update_after_fallbacks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = copied_strategy_store(temporary)
            client = FakeReviewClient()
            reviewer = RoleStrategyReviewer(model_client=client, strategy_store=store)
            original_strategy = store.profile(ROLE_WOLF).strategy

            def game_factory(round_index: int, game_index: int) -> tuple[GameEngine, dict]:
                player_list = [
                    {"id": f"p{player_index}", "name": f"Player {player_index}"}
                    for player_index in range(1, 8)
                ]
                engine = GameEngine(
                    game_id=f"round{round_index}-game{game_index}",
                    players=player_list,
                    rules=create_default_rules(),
                    seed=f"quality-{round_index}-{game_index}",
                )
                participants = {
                    player["id"]: (
                        FailsFirstDecisionParticipant(player["id"])
                        if player["id"] == "p1"
                        else ScriptedParticipant(player["id"], deterministic_strategy)
                    )
                    for player in player_list
                }
                return engine, participants

            result = await GameRoundRunner(
                game_factory=game_factory,
                record_directory=temporary,
                reviewer=reviewer,
                decision_timeout_seconds=1,
                game_concurrency=2,
                max_fallback_rate=0,
                max_fallbacks_per_game=0,
            ).run_round(0)

            self.assertTrue(result["review_skipped_by_quality_gate"])
            self.assertFalse(result["quality"]["skill_update_eligible"])
            self.assertGreater(result["quality"]["metrics"]["fallback_count"], 0)
            self.assertEqual(client.requests, [])
            self.assertEqual(store.profile(ROLE_WOLF).strategy, original_strategy)
            self.assertTrue(Path(result["quality_path"]).exists())
            quality = json.loads(Path(result["quality_path"]).read_text(encoding="utf-8"))
            self.assertTrue(quality["review_skipped_by_quality_gate"])

    async def test_adaptive_scheduler_expands_after_clean_games(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            def game_factory(round_index: int, game_index: int) -> tuple[GameEngine, dict]:
                player_list = [
                    {"id": f"p{player_index}", "name": f"Player {player_index}"}
                    for player_index in range(1, 8)
                ]
                engine = GameEngine(
                    game_id=f"round{round_index}-game{game_index}",
                    players=player_list,
                    rules=create_default_rules(),
                    seed=f"adaptive-{round_index}-{game_index}",
                )
                return engine, {
                    player["id"]: ScriptedParticipant(
                        player["id"], deterministic_strategy
                    )
                    for player in player_list
                }

            result = await GameRoundRunner(
                game_factory=game_factory,
                record_directory=temporary,
                game_concurrency=3,
                adaptive_concurrency=True,
                initial_game_concurrency=1,
                request_coordinator=ModelRequestCoordinator(max_in_flight=4),
            ).run_round(0)

            performance = json.loads(
                Path(result["performance_path"]).read_text(encoding="utf-8")
            )
            controls = performance["controls"]
            self.assertEqual(len(controls), 10)
            self.assertTrue(any(item["action"] == "increase" for item in controls))
            self.assertLessEqual(
                max(item["game_concurrency"] for item in controls), 3
            )


class RoundSkillVersionStoreTest(unittest.TestCase):
    def test_next_round_inherits_the_last_output_version(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = copied_strategy_store(temporary)
            initial = RoundSkillVersionStore(temporary, round_index=0)
            input_manifest = initial.ensure_input(
                roles=[ROLE_WOLF], strategy_store=store
            )
            self.assertEqual(input_manifest["roles"][ROLE_WOLF]["input"]["version"], 0)

            next_strategy = "# 狼人当前策略（经验，可能不完全正确）\n\n- 用公开票型校验同伴切割是否产生收益。\n- 每局结束后保留至少一个反证。"
            store.replace_strategy(ROLE_WOLF, next_strategy)
            output_manifest = initial.capture_output(
                roles=[ROLE_WOLF], strategy_store=store
            )
            self.assertEqual(output_manifest["roles"][ROLE_WOLF]["output"]["version"], 1)

            following = RoundSkillVersionStore(temporary, round_index=1)
            following_manifest = following.ensure_input(
                roles=[ROLE_WOLF], strategy_store=store
            )
            following_input = following_manifest["roles"][ROLE_WOLF]["input"]
            self.assertEqual(following_input["version"], 1)
            self.assertEqual(following_input["source"]["kind"], "previous_round")
            snapshot = Path(temporary) / "round1" / following_input["strategy_path"]
            self.assertEqual(snapshot.read_text(encoding="utf-8").strip(), next_strategy)

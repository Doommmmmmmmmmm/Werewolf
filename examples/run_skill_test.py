"""运行冻结 skill 的双向对抗评估，不更新任何角色 strategy.md。

默认执行 ``records/test1``：
1. 狼人 latest skill vs 其余角色 initial skill，10 局；
2. 狼人 initial skill vs 其余角色 latest skill，10 局。

可选环境变量：
  WEREWOLF_SKILL_TEST_ID（默认 test1）、WEREWOLF_RECORD_DIRECTORY、
  WEREWOLF_TEST_INITIAL_ROUND（默认 0）、WEREWOLF_TEST_LATEST_ROUND（默认自动发现）、
  WEREWOLF_TEST_GAMES_PER_SCENARIO（默认 10）、
  WEREWOLF_GAME_SEED、WEREWOLF_PLAYER_PERSONA、WEREWOLF_PLAYER_COUNT（默认 12）、
  WEREWOLF_OPTIONAL_ROLES、WEREWOLF_GAME_CONCURRENCY（默认 4）、
  WEREWOLF_DECISION_TIMEOUT_SECONDS（默认 180）、
  WEREWOLF_MODEL_MAX_IN_FLIGHT（默认 8）。
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from werewolf_game import (
    GameEngine,
    RoleStrategyStore,
    SkillTestRunner,
    create_rules_for_player_count,
)
from werewolf_game.llm import ModelClient, ModelRequestCoordinator
from werewolf_game.participants import LlmParticipant


async def main() -> None:
    test_id = os.environ.get("WEREWOLF_SKILL_TEST_ID", "test1")
    record_directory = os.environ.get("WEREWOLF_RECORD_DIRECTORY", "records")
    initial_round = int(os.environ.get("WEREWOLF_TEST_INITIAL_ROUND", "0"))
    latest_value = os.environ.get("WEREWOLF_TEST_LATEST_ROUND")
    latest_round = int(latest_value) if latest_value is not None else None
    games_per_scenario = int(
        os.environ.get("WEREWOLF_TEST_GAMES_PER_SCENARIO", "10")
    )
    game_concurrency = int(os.environ.get("WEREWOLF_GAME_CONCURRENCY", "4"))
    decision_timeout_seconds = float(
        os.environ.get("WEREWOLF_DECISION_TIMEOUT_SECONDS", "180")
    )
    model_max_in_flight = int(os.environ.get("WEREWOLF_MODEL_MAX_IN_FLIGHT", "8"))
    base_seed = os.environ.get("WEREWOLF_GAME_SEED", "skill-evaluation")
    persona = os.environ.get("WEREWOLF_PLAYER_PERSONA", "")
    player_count = int(os.environ.get("WEREWOLF_PLAYER_COUNT", "12"))
    optional_roles = tuple(
        role.strip()
        for role in os.environ.get("WEREWOLF_OPTIONAL_ROLES", "").split(",")
        if role.strip()
    )
    rules = create_rules_for_player_count(player_count, optional_roles)
    task_model_client = ModelClient.from_env(profile="task")
    request_coordinator = ModelRequestCoordinator(max_in_flight=model_max_in_flight)

    def game_factory(
        scenario_id: str, game_index: int, strategy_store: RoleStrategyStore
    ) -> tuple[GameEngine, dict]:
        players = [
            {"id": f"p{index}", "name": f"Player {index}"}
            for index in range(1, player_count + 1)
        ]
        engine = GameEngine(
            game_id=f"{test_id}-{scenario_id}-game{game_index}",
            players=players,
            rules=rules,
            seed=f"{base_seed}-{test_id}-{scenario_id}-game{game_index}",
        )
        participants = {
            player["id"]: LlmParticipant(
                player_id=player["id"],
                model_client=task_model_client,
                persona=persona,
                strategy_store=strategy_store,
                request_coordinator=request_coordinator,
            )
            for player in players
        }
        return engine, participants

    report = await SkillTestRunner(
        game_factory=game_factory,
        roles=rules.role_deck,
        record_directory=record_directory,
        test_id=test_id,
        initial_round=initial_round,
        latest_round=latest_round,
        decision_timeout_seconds=decision_timeout_seconds,
        game_concurrency=game_concurrency,
        game_count_per_scenario=games_per_scenario,
        resume_completed_games=True,
    ).run()
    print("Skill test finished:", report["test_directory"])
    print("Skill unchanged:", not report["skill_updated"])
    for scenario in report["scenarios"]:
        print(scenario["scenario_id"], "wins=", scenario["wins"])
    print("Manifest:", report["manifest_path"])
    print("Summary:", report["summary_path"])


if __name__ == "__main__":
    asyncio.run(main())

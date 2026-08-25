"""运行 round10 / round20 skill 的阵营对抗评测，不更新任何 strategy.md。

默认将训练 round 的零基编号映射为：

* 初始 skill：round0/skill/input；
* 10 轮后 skill：round9/skill/output；
* 20 轮后 skill：round19/skill/output。

四个场景各运行 20 局（共 80 局）：

1. 好人 round20 vs 狼人 initial；
2. 好人 round20 vs 狼人 round10；
3. 狼人 round20 vs 好人 initial；
4. 狼人 round20 vs 好人 round10。
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
    SkillTestScenario,
    SkillTestSource,
    create_rules_for_player_count,
)
from werewolf_game.llm import ModelClient, ModelRequestCoordinator
from werewolf_game.participants import LlmParticipant


SCENARIOS = (
    SkillTestScenario(
        scenario_id="good-round20-vs-wolf-initial",
        description="好人使用 20 轮后 skill；狼人使用初始 skill。",
        wolf_source="initial",
        other_source="round20",
    ),
    SkillTestScenario(
        scenario_id="good-round20-vs-wolf-round10",
        description="好人使用 20 轮后 skill；狼人使用 10 轮后 skill。",
        wolf_source="round10",
        other_source="round20",
    ),
    SkillTestScenario(
        scenario_id="wolf-round20-vs-good-initial",
        description="狼人使用 20 轮后 skill；好人使用初始 skill。",
        wolf_source="round20",
        other_source="initial",
    ),
    SkillTestScenario(
        scenario_id="wolf-round20-vs-good-round10",
        description="狼人使用 20 轮后 skill；好人使用 10 轮后 skill。",
        wolf_source="round20",
        other_source="round10",
    ),
)


async def main() -> None:
    test_id = os.environ.get("WEREWOLF_SKILL_TEST_ID", "test2")
    record_directory = os.environ.get("WEREWOLF_RECORD_DIRECTORY", "records")
    initial_round = int(os.environ.get("WEREWOLF_TEST_INITIAL_ROUND", "0"))
    round10 = int(os.environ.get("WEREWOLF_TEST_ROUND10", "9"))
    round20 = int(os.environ.get("WEREWOLF_TEST_ROUND20", "19"))
    games_per_scenario = int(
        os.environ.get("WEREWOLF_TEST_GAMES_PER_SCENARIO", "20")
    )
    game_concurrency = int(os.environ.get("WEREWOLF_GAME_CONCURRENCY", "2"))
    decision_timeout_seconds = float(
        os.environ.get("WEREWOLF_DECISION_TIMEOUT_SECONDS", "600")
    )
    model_max_in_flight = int(os.environ.get("WEREWOLF_MODEL_MAX_IN_FLIGHT", "3"))
    base_seed = os.environ.get("WEREWOLF_GAME_SEED", "skill-progression-evaluation")
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

    source_definitions = (
        SkillTestSource(
            source_id="initial",
            round_index=initial_round,
            stage="input",
            description="训练开始前的初始 skill。",
        ),
        SkillTestSource(
            source_id="round10",
            round_index=round10,
            stage="output",
            description="完成前 10 个训练 round 后的输出 skill。",
        ),
        SkillTestSource(
            source_id="round20",
            round_index=round20,
            stage="output",
            description="完成前 20 个训练 round 后的输出 skill。",
        ),
    )
    report = await SkillTestRunner(
        game_factory=game_factory,
        roles=rules.role_deck,
        record_directory=record_directory,
        test_id=test_id,
        initial_round=initial_round,
        latest_round=round20,
        decision_timeout_seconds=decision_timeout_seconds,
        game_concurrency=game_concurrency,
        game_count_per_scenario=games_per_scenario,
        resume_completed_games=True,
        scenarios=SCENARIOS,
        source_definitions=source_definitions,
    ).run()
    print("Skill progression test finished:", report["test_directory"])
    print("Skill unchanged:", not report["skill_updated"])
    for scenario in report["scenarios"]:
        print(scenario["scenario_id"], "wins=", scenario["wins"])
    print("Manifest:", report["manifest_path"])
    print("Summary:", report["summary_path"])


if __name__ == "__main__":
    asyncio.run(main())

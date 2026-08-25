"""运行一个 10 局训练 round，并在结尾由角色 Agent 更新 strategy.md。

运行：python3 examples/run_llm_round.py
可选环境变量：
  WEREWOLF_TRAINING_ROUND（默认 0）、WEREWOLF_RECORD_DIRECTORY、
  WEREWOLF_GAME_SEED、WEREWOLF_PLAYER_PERSONA、WEREWOLF_PLAYER_COUNT（7–12）、
  WEREWOLF_OPTIONAL_ROLES（如 guard,hunter）、WEREWOLF_GAME_CONCURRENCY（默认 1）、
  WEREWOLF_DECISION_TIMEOUT_SECONDS（默认 180）、
  WEREWOLF_MODEL_MAX_IN_FLIGHT（默认 8）、WEREWOLF_ADAPTIVE_CONCURRENCY（默认 true）、
  WEREWOLF_REVIEW_REPLAY_BATCH_SIZE（默认 2）。
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
    GameRoundRunner,
    RoleStrategyReviewer,
    create_rules_for_player_count,
)
from werewolf_game.llm import ModelClient, ModelRequestCoordinator
from werewolf_game.participants import LlmParticipant


async def main() -> None:
    round_index = int(os.environ.get("WEREWOLF_TRAINING_ROUND", "0"))
    record_directory = os.environ.get("WEREWOLF_RECORD_DIRECTORY", "records")
    game_concurrency = int(os.environ.get("WEREWOLF_GAME_CONCURRENCY", "1"))
    decision_timeout_seconds = float(
        os.environ.get("WEREWOLF_DECISION_TIMEOUT_SECONDS", "180")
    )
    model_max_in_flight = int(os.environ.get("WEREWOLF_MODEL_MAX_IN_FLIGHT", "8"))
    review_replay_batch_size = int(
        os.environ.get("WEREWOLF_REVIEW_REPLAY_BATCH_SIZE", "2")
    )
    adaptive_concurrency = (
        os.environ.get("WEREWOLF_ADAPTIVE_CONCURRENCY", "true").strip().lower()
        != "false"
    )
    initial_game_concurrency = int(
        os.environ.get("WEREWOLF_INITIAL_GAME_CONCURRENCY", "2")
    )
    min_game_concurrency = int(
        os.environ.get("WEREWOLF_MIN_GAME_CONCURRENCY", "1")
    )
    healthy_p95_latency_ms = int(
        os.environ.get("WEREWOLF_HEALTHY_P95_LATENCY_MS", "45000")
    )
    max_fallback_rate = float(os.environ.get("WEREWOLF_MAX_FALLBACK_RATE", "0.01"))
    max_fallbacks_per_game = int(
        os.environ.get("WEREWOLF_MAX_FALLBACKS_PER_GAME", "3")
    )
    allow_degraded_review = (
        os.environ.get("WEREWOLF_ALLOW_DEGRADED_REVIEW", "false").strip().lower()
        == "true"
    )
    base_seed = os.environ.get("WEREWOLF_GAME_SEED", "llm-training")
    persona = os.environ.get("WEREWOLF_PLAYER_PERSONA", "")
    player_count = int(os.environ.get("WEREWOLF_PLAYER_COUNT", "12"))
    optional_roles = tuple(
        role.strip()
        for role in os.environ.get("WEREWOLF_OPTIONAL_ROLES", "").split(",")
        if role.strip()
    )
    rules = create_rules_for_player_count(player_count, optional_roles)
    task_model_client = ModelClient.from_env(profile="task")
    meta_model_client = ModelClient.from_env(profile="meta")
    request_coordinator = ModelRequestCoordinator(max_in_flight=model_max_in_flight)
    reviewer = RoleStrategyReviewer(
        model_client=meta_model_client,
        request_coordinator=request_coordinator,
        replay_batch_size=review_replay_batch_size,
    )

    def game_factory(current_round: int, game_index: int) -> tuple[GameEngine, dict]:
        players = [
            {"id": f"p{index}", "name": f"Player {index}"}
            for index in range(1, player_count + 1)
        ]
        engine = GameEngine(
            game_id=f"round{current_round}-game{game_index}",
            players=players,
            rules=rules,
            seed=f"{base_seed}-round{current_round}-game{game_index}",
        )
        participants = {
            player["id"]: LlmParticipant(
                player_id=player["id"],
                model_client=task_model_client,
                persona=persona,
                strategy_store=reviewer.strategy_store,
                request_coordinator=request_coordinator,
            )
            for player in players
        }
        return engine, participants

    report = await GameRoundRunner(
        game_factory=game_factory,
        record_directory=record_directory,
        reviewer=reviewer,
        game_concurrency=game_concurrency,
        decision_timeout_seconds=decision_timeout_seconds,
        request_coordinator=request_coordinator,
        adaptive_concurrency=adaptive_concurrency,
        initial_game_concurrency=min(initial_game_concurrency, game_concurrency),
        min_game_concurrency=min(min_game_concurrency, game_concurrency),
        healthy_p95_latency_ms=healthy_p95_latency_ms,
        max_fallback_rate=max_fallback_rate,
        max_fallbacks_per_game=max_fallbacks_per_game,
        allow_degraded_review=allow_degraded_review,
        resume_completed_games=True,
    ).run_round(round_index)
    print("Round finished:", report["round_directory"])
    for review in report["reviews"]:
        print(review["role"], "updated=" + str(review["updated"]), review["review_markdown_path"])


if __name__ == "__main__":
    asyncio.run(main())

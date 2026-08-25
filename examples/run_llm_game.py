"""使用真实 LLM 玩家运行一局。

运行：python3 examples/run_llm_game.py
可选环境变量：
  WEREWOLF_GAME_ID、WEREWOLF_GAME_SEED、WEREWOLF_QUIET=1、WEREWOLF_PLAYER_PERSONA、
  WEREWOLF_PLAYER_COUNT（7–12）、WEREWOLF_OPTIONAL_ROLES（如 guard,hunter）、
  WEREWOLF_TRAINING_ROUND、WEREWOLF_GAME_INDEX、WEREWOLF_RECORD_DIRECTORY、
  WEREWOLF_DECISION_TIMEOUT_SECONDS（默认 180）、WEREWOLF_MODEL_MAX_IN_FLIGHT（默认 8）
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
    GameRunner,
    RoleStrategyReviewer,
    RoundSkillVersionStore,
    create_rules_for_player_count,
    review_completed_round,
)
from werewolf_game.llm import ModelClient, ModelRequestCoordinator
from werewolf_game.participants import LlmParticipant
from werewolf_game.records import RoundGameRecordStore


async def main() -> None:
    base_seed = os.environ.get("WEREWOLF_GAME_SEED", "llm-demo-seed")
    quiet = os.environ.get("WEREWOLF_QUIET") == "1"
    persona = os.environ.get("WEREWOLF_PLAYER_PERSONA", "")
    player_count = int(os.environ.get("WEREWOLF_PLAYER_COUNT", "12"))
    optional_roles = tuple(
        role.strip()
        for role in os.environ.get("WEREWOLF_OPTIONAL_ROLES", "").split(",")
        if role.strip()
    )
    decision_timeout_seconds = float(
        os.environ.get("WEREWOLF_DECISION_TIMEOUT_SECONDS", "180")
    )
    model_max_in_flight = int(os.environ.get("WEREWOLF_MODEL_MAX_IN_FLIGHT", "8"))
    rules = create_rules_for_player_count(player_count, optional_roles)
    record_directory = os.environ.get("WEREWOLF_RECORD_DIRECTORY", "records")
    requested_round = os.environ.get("WEREWOLF_TRAINING_ROUND")
    requested_game = os.environ.get("WEREWOLF_GAME_INDEX")
    if requested_round is None and requested_game is None:
        round_index, game_index = RoundGameRecordStore.next_available(record_directory)
    elif requested_round is not None and requested_game is not None:
        round_index, game_index = int(requested_round), int(requested_game)
    else:
        raise ValueError("WEREWOLF_TRAINING_ROUND 与 WEREWOLF_GAME_INDEX 必须同时设置")
    game_id = os.environ.get("WEREWOLF_GAME_ID", f"round{round_index}-game{game_index}")
    record_store = RoundGameRecordStore(
        record_directory,
        round_index=round_index,
        game_index=game_index,
    )
    players = [
        {"id": f"p{index}", "name": f"Player {index}"}
        for index in range(1, player_count + 1)
    ]
    task_model_client = ModelClient.from_env(profile="task")
    meta_model_client = ModelClient.from_env(profile="meta")
    request_coordinator = ModelRequestCoordinator(max_in_flight=model_max_in_flight)
    reviewer = RoleStrategyReviewer(
        model_client=meta_model_client, request_coordinator=request_coordinator
    )
    skill_versions = RoundSkillVersionStore(
        record_directory, round_index=round_index
    )
    # 在当前对局开始前固化本 round 实际提供给玩家的角色 skill。
    skill_versions.ensure_input(
        roles=rules.role_deck,
        strategy_store=reviewer.strategy_store,
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
    engine = GameEngine(
        game_id=game_id,
        players=players,
        rules=rules,
        # 同一基础种子下，每个稳定的 round/game 槽位仍有不同且可复现的发牌。
        seed=f"{base_seed}-round{round_index}-game{game_index}",
    )

    def print_public_events(events: list[dict]) -> None:
        if quiet:
            return
        for event in events:
            print(event["seq"], event["type"], event["payload"])

    report = await GameRunner(
        engine=engine,
        participants=participants,
        record_store=record_store,
        on_public_events=print_public_events,
        decision_timeout_seconds=decision_timeout_seconds,
    ).run()
    print("Finished:", report["public_state"])
    print("Record:", report["record_path"])
    print("Replay:", report["record_markdown_path"])
    print("Public record:", report["public_record_path"])
    print("Public replay:", report["public_record_markdown_path"])

    # 单局可以连续运行十次；只有记录恰好收齐后才会调用复盘模型。
    review_state = await review_completed_round(
        record_directory=record_directory,
        round_index=round_index,
        reviewer=reviewer,
    )
    if review_state["completed"]:
        if review_state["already_reviewed"]:
            print("Role review: this round was already reviewed.")
        else:
            print("Role review completed:", review_state["manifest_path"])
            for review in review_state["reviews"]:
                print(
                    " ",
                    review["role"],
                    "updated=" + str(review["updated"]),
                    review["review_markdown_path"],
                )
            print("Skill versions:", review_state["skill_version_manifest_path"])


if __name__ == "__main__":
    asyncio.run(main())

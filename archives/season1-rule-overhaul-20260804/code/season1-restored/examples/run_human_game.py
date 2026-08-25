"""人工玩家对抗 LLM 玩家的单局入口。

运行：

    python3 examples/run_human_game.py

默认由引擎随机发牌，人工玩家为 ``p1``。也可以在开局前固定人工身份：

    WEREWOLF_HUMAN_ROLE=wolf python3 examples/run_human_game.py

可用环境变量：
  WEREWOLF_HUMAN_PLAYER（默认 p1）、WEREWOLF_HUMAN_ROLE（留空为随机）、
  WEREWOLF_PLAYER_COUNT（默认 12）、WEREWOLF_GAME_SEED、WEREWOLF_RECORD_DIRECTORY、
  WEREWOLF_TRAINING_ROUND、WEREWOLF_GAME_INDEX、WEREWOLF_DECISION_TIMEOUT_SECONDS、
  WEREWOLF_MODEL_MAX_IN_FLIGHT、WEREWOLF_HUMAN_SHOW_EVENTS。

人工玩家的身份不会在输入时主动显示给其他玩家；指定身份只改变引擎发牌约束，
所有行动仍通过 GameEngine 校验。
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from werewolf_game import GameEngine, GameRunner, create_rules_for_player_count
from werewolf_game.llm import ModelClient, ModelRequestCoordinator
from werewolf_game.participants import HumanParticipant, LlmParticipant
from werewolf_game.records import RoundGameRecordStore
from werewolf_game.prompts import RoleStrategyStore


async def main() -> None:
    player_count = int(os.environ.get("WEREWOLF_PLAYER_COUNT", "12"))
    human_player_id = os.environ.get("WEREWOLF_HUMAN_PLAYER", "p1").strip()
    human_role = os.environ.get("WEREWOLF_HUMAN_ROLE", "").strip().lower()
    base_seed = os.environ.get("WEREWOLF_GAME_SEED", "human-demo-seed")
    record_directory = os.environ.get("WEREWOLF_RECORD_DIRECTORY", "records")
    decision_timeout = float(
        os.environ.get("WEREWOLF_DECISION_TIMEOUT_SECONDS", "180")
    )
    max_in_flight = int(os.environ.get("WEREWOLF_MODEL_MAX_IN_FLIGHT", "8"))
    show_events = int(os.environ.get("WEREWOLF_HUMAN_SHOW_EVENTS", "8"))

    if not human_player_id.startswith("p"):
        raise ValueError("WEREWOLF_HUMAN_PLAYER 必须是类似 p1 的玩家编号")
    if human_role and human_role not in {
        "wolf",
        "villager",
        "seer",
        "witch",
        "guard",
        "hunter",
        "idiot",
    }:
        raise ValueError("WEREWOLF_HUMAN_ROLE 不是当前规则支持的身份")

    optional_roles = tuple(
        role.strip()
        for role in os.environ.get("WEREWOLF_OPTIONAL_ROLES", "").split(",")
        if role.strip()
    )
    rules = create_rules_for_player_count(player_count, optional_roles)
    players = [
        {"id": f"p{index}", "name": f"Player {index}"}
        for index in range(1, player_count + 1)
    ]
    player_ids = {player["id"] for player in players}
    if human_player_id not in player_ids:
        raise ValueError(f"人工玩家 {human_player_id} 不在当前 {player_count} 人局中")

    requested_round = os.environ.get("WEREWOLF_TRAINING_ROUND")
    requested_game = os.environ.get("WEREWOLF_GAME_INDEX")
    if requested_round is None and requested_game is None:
        round_index, game_index = RoundGameRecordStore.next_available(record_directory)
    elif requested_round is not None and requested_game is not None:
        round_index, game_index = int(requested_round), int(requested_game)
    else:
        raise ValueError(
            "WEREWOLF_TRAINING_ROUND 与 WEREWOLF_GAME_INDEX 必须同时设置"
        )

    game_id = os.environ.get(
        "WEREWOLF_GAME_ID", f"human-round{round_index}-game{game_index}"
    )
    fixed_roles = {human_player_id: human_role} if human_role else None
    engine = GameEngine(
        game_id=game_id,
        players=players,
        rules=rules,
        seed=f"{base_seed}-round{round_index}-game{game_index}",
        fixed_roles=fixed_roles,
    )

    task_model_client = ModelClient.from_env(profile="task")
    coordinator = ModelRequestCoordinator(max_in_flight=max_in_flight)
    strategy_store = RoleStrategyStore()
    participants = {
        player["id"]: (
            HumanParticipant(player["id"], show_events=show_events)
            if player["id"] == human_player_id
            else LlmParticipant(
                player_id=player["id"],
                model_client=task_model_client,
                strategy_store=strategy_store,
                request_coordinator=coordinator,
            )
        )
        for player in players
    }

    def print_public_events(events: list[dict]) -> None:
        for event in events:
            event_type = event.get("type")
            payload = event.get("payload") or {}
            # 人类玩家在自己的行动提示中会看到最近公开事件；这里仅播报
            # 重要状态变化，避免终端被完整事件流淹没。
            if event_type in {
                "DAWN_ANNOUNCED",
                "PLAYER_DIED",
                "PLAYER_ELIMINATED",
                "GAME_FINISHED",
                "SHERIFF_ELECTED",
                "SHERIFF_BADGE_TRANSFERRED",
                "SHERIFF_BADGE_DESTROYED",
            }:
                print(f"[公开] {event_type}: {payload}")

    report = await GameRunner(
        engine=engine,
        participants=participants,
        record_store=RoundGameRecordStore(
            record_directory, round_index=round_index, game_index=game_index
        ),
        decision_timeout_seconds=decision_timeout,
        on_public_events=print_public_events,
    ).run()
    print("\n游戏结束：", report["public_state"])
    print("完整记录：", report["record_path"])
    print("公开记录：", report["public_record_markdown_path"])


if __name__ == "__main__":
    asyncio.run(main())

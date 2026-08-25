"""不调用模型，使用确定性策略完整运行一局。

运行：python3 examples/run_scripted_game.py
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
from werewolf_game.constants import ACTION_LAST_WORDS, ACTION_PASS, ACTION_SPEAK
from werewolf_game.participants import ScriptedParticipant
from werewolf_game.records import RoundGameRecordStore


def deterministic_strategy(packet: dict) -> dict:
    """示例策略：发言固定，其他阶段选择第一个合法非跳过行动。"""

    allowed = packet["request"]["allowed_actions"]
    action = next((item for item in allowed if item["kind"] != ACTION_PASS), allowed[0])
    if action["kind"] in {ACTION_SPEAK, ACTION_LAST_WORDS}:
        return {
            "kind": action["kind"],
            "text": "我会记录本轮信息，并在投票时给出判断。",
        }
    result = {"kind": action["kind"]}
    if action.get("target_ids"):
        result["target_id"] = action["target_ids"][0]
    return result


async def main() -> None:
    base_seed = os.environ.get("WEREWOLF_GAME_SEED", "scripted-demo-seed")
    player_count = int(os.environ.get("WEREWOLF_PLAYER_COUNT", "12"))
    optional_roles = tuple(
        role.strip()
        for role in os.environ.get("WEREWOLF_OPTIONAL_ROLES", "").split(",")
        if role.strip()
    )
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
    record_store = RoundGameRecordStore(
        record_directory,
        round_index=round_index,
        game_index=game_index,
    )
    players = [
        {"id": f"p{index}", "name": f"Player {index}"}
        for index in range(1, player_count + 1)
    ]
    engine = GameEngine(
        game_id=f"round{round_index}-game{game_index}",
        players=players,
        rules=rules,
        seed=f"{base_seed}-round{round_index}-game{game_index}",
    )
    participants = {
        player["id"]: ScriptedParticipant(player["id"], deterministic_strategy)
        for player in players
    }

    def print_public_events(events: list[dict]) -> None:
        for event in events:
            print(event["seq"], event["type"], event["payload"])

    report = await GameRunner(
        engine=engine,
        participants=participants,
        record_store=record_store,
        on_public_events=print_public_events,
    ).run()
    print("Finished:", report["public_state"])
    print("Record:", report["record_path"])
    print("Replay:", report["record_markdown_path"])
    print("Public record:", report["public_record_path"])
    print("Public replay:", report["public_record_markdown_path"])


if __name__ == "__main__":
    asyncio.run(main())

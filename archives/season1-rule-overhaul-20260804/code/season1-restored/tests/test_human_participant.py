from __future__ import annotations

import asyncio
import unittest

from werewolf_game import GameEngine, GameRunner, create_rules_for_player_count
from werewolf_game.constants import ACTION_SPEAK, ACTION_WOLF_KILL_VOTE, ROLE_WOLF
from werewolf_game.participants import HumanParticipant


class HumanParticipantTest(unittest.TestCase):
    def test_parse_short_commands(self) -> None:
        request = {
            "allowed_actions": [
                {"kind": ACTION_WOLF_KILL_VOTE, "target_ids": ["p2", "p3"]}
            ]
        }
        self.assertEqual(
            HumanParticipant.parse_input("刀 p2", request),
            {"kind": ACTION_WOLF_KILL_VOTE, "target_id": "p2"},
        )

    def test_parse_speech_and_json(self) -> None:
        request = {
            "allowed_actions": [
                {"kind": ACTION_SPEAK, "max_chars": 200, "require_chinese": True}
            ]
        }
        self.assertEqual(
            HumanParticipant.parse_input("发言 我先复盘票型。", request),
            {"kind": ACTION_SPEAK, "text": "我先复盘票型。"},
        )
        self.assertEqual(
            HumanParticipant.parse_input('{"kind":"speak","text":"我发言"}', request),
            {"kind": "speak", "text": "我发言"},
        )

    def test_local_validation_rejects_wrong_target_and_long_speech(self) -> None:
        vote_request = {
            "allowed_actions": [{"kind": "day_vote", "target_ids": ["p2"]}]
        }
        with self.assertRaises(ValueError):
            HumanParticipant._validate_local(
                {"kind": "day_vote", "target_id": "p9"}, vote_request
            )
        speech_request = {
            "allowed_actions": [
                {"kind": "speak", "max_chars": 2, "require_chinese": True}
            ]
        }
        with self.assertRaises(ValueError):
            HumanParticipant._validate_local(
                {"kind": "speak", "text": "这句话"}, speech_request
            )

    def test_decide_retries_after_invalid_input(self) -> None:
        inputs = iter(["投票 p9", "投票 p2"])
        output: list[str] = []
        participant = HumanParticipant(
            "p1", input_fn=lambda _: next(inputs), output_fn=output.append
        )
        packet = {
            "game": {"round": 1, "phase": "day_vote"},
            "public_state": {"players": []},
            "private_information": {"role": "villager", "team": "village"},
            "request": {
                "allowed_actions": [
                    {"kind": "day_vote", "target_ids": ["p2"]}
                ]
            },
            "visible_events": [],
        }
        action = asyncio.run(participant.decide(packet))
        self.assertEqual(action, {"kind": "day_vote", "target_id": "p2"})
        self.assertTrue(any("输入无效" in line for line in output))

    def test_human_input_does_not_inherit_model_timeout(self) -> None:
        async def delayed_input(_: str) -> str:
            await asyncio.sleep(0.01)
            return "发言 我会基于公开信息继续判断。"

        players = [{"id": f"p{i}", "name": f"Player {i}"} for i in range(1, 13)]
        engine = GameEngine(
            game_id="human-no-timeout",
            players=players,
            rules=create_rules_for_player_count(12),
            seed="human-no-timeout",
            fixed_roles={"p1": ROLE_WOLF},
        )
        engine.start()
        participant = HumanParticipant("p1", input_fn=delayed_input, output_fn=lambda _: None)
        runner = GameRunner(
            engine=engine,
            participants={"p1": participant},
            record_store=False,
            decision_timeout_seconds=0.001,
        )

        action = asyncio.run(
            runner._obtain_decision(engine.discussion_request("p1", "wolf"))
        )
        self.assertEqual(action["kind"], ACTION_SPEAK)
        self.assertEqual(runner.errors, [])


class FixedRoleDealingTest(unittest.TestCase):
    def _players(self) -> list[dict[str, str]]:
        return [{"id": f"p{i}", "name": f"Player {i}"} for i in range(1, 13)]

    def test_fixed_role_is_dealt_to_requested_player(self) -> None:
        engine = GameEngine(
            game_id="fixed-role-test",
            players=self._players(),
            rules=create_rules_for_player_count(12),
            seed="fixed-role-seed",
            fixed_roles={"p1": ROLE_WOLF},
        )
        engine.start()
        self.assertEqual(engine.role_assignments()["p1"], ROLE_WOLF)
        self.assertEqual(
            collections_count(engine.role_assignments().values())[ROLE_WOLF], 4
        )
        self.assertEqual(
            engine.record_metadata()["role_assignment"],
            {"mode": "specified", "fixed_roles": {"p1": ROLE_WOLF}},
        )

    def test_random_mode_has_no_fixed_roles(self) -> None:
        engine = GameEngine(
            game_id="random-role-test",
            players=self._players(),
            rules=create_rules_for_player_count(12),
            seed="random-role-seed",
        )
        engine.start()
        self.assertEqual(engine.record_metadata()["role_assignment"]["mode"], "random")
        self.assertEqual(engine.record_metadata()["role_assignment"]["fixed_roles"], {})

    def test_fixed_role_count_must_exist_in_deck(self) -> None:
        with self.assertRaisesRegex(ValueError, "当前牌堆只有"):
            GameEngine(
                game_id="too-many-fixed",
                players=self._players(),
                rules=create_rules_for_player_count(12),
                fixed_roles={"p1": "seer", "p2": "seer"},
            )


def collections_count(values: object) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:  # type: ignore[union-attr]
        counts[str(value)] = counts.get(str(value), 0) + 1
    return counts


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import stat
import tempfile
import unittest

from werewolf_game import (
    GameEngine,
    GameRunner,
    PublicRecorder,
    RuleSet,
    create_default_rules,
    create_rules_for_player_count,
)
from werewolf_game.core.constants import (
    ACTION_GUARD_PROTECT,
    ACTION_HUNTER_SHOOT,
    ACTION_LAST_WORDS,
    ACTION_PASS,
    ACTION_SHERIFF_CANDIDATE,
    ACTION_SHERIFF_SPEECH_ORDER,
    ACTION_SHERIFF_VOTE,
    ACTION_SPEAK,
    PHASE_DAY_DISCUSSION,
    PHASE_DAY_VOTE,
    PHASE_NIGHT_GUARD,
    PHASE_NIGHT_RESOLVE,
    PHASE_NIGHT_WOLF_VOTE,
    PHASE_SHERIFF_ELECTION_SPEECH,
    ROLE_GUARD,
    ROLE_HUNTER,
    ROLE_IDIOT,
    ROLE_SEER,
    ROLE_VILLAGER,
    ROLE_WOLF,
)
from werewolf_game.core.errors import RuleViolationError
from werewolf_game.agents.participants import ScriptedParticipant


def players(count: int = 7) -> list[dict[str, str]]:
    return [{"id": f"p{index}", "name": f"Player {index}"} for index in range(1, count + 1)]


def engine_for(seed: str = "test-seed") -> GameEngine:
    return GameEngine(
        game_id="test-game",
        players=players(),
        rules=create_default_rules(enable_sheriff_election=False),
        seed=seed,
    )


def deterministic_strategy(packet: dict) -> dict:
    action = next(
        (item for item in packet["request"]["allowed_actions"] if item["kind"] != ACTION_PASS),
        packet["request"]["allowed_actions"][0],
    )
    if action["kind"] in {ACTION_SPEAK, ACTION_LAST_WORDS}:
        return {"kind": action["kind"], "text": "我会记录公开信息，稍后再投票。"}
    result = {"kind": action["kind"]}
    if action.get("target_ids"):
        result["target_id"] = action["target_ids"][0]
    return result


class GameEngineTest(unittest.TestCase):
    def test_current_round_dialogue_is_role_filtered(self) -> None:
        engine = engine_for("dialogue-tool")
        engine.start()
        wolves = engine.wolf_players()
        wolf_id = wolves[0].player_id
        villager_id = next(
            player.player_id
            for player in engine.players
            if player.player_id not in {item.player_id for item in wolves}
        )
        for wolf in wolves:
            request = engine.discussion_request(wolf.player_id, "wolf")
            engine.accept_action(request, {"kind": ACTION_SPEAK, "text": "今晚先看票型"})

        wolf_dialogue = engine.current_round_dialogue_for(wolf_id)
        villager_dialogue = engine.current_round_dialogue_for(villager_id)
        self.assertEqual(len(wolf_dialogue), len(wolves))
        self.assertEqual({item["channel"] for item in wolf_dialogue}, {"wolf"})
        self.assertEqual(villager_dialogue, [])

    def test_standard_presets_cover_seven_to_twelve_players(self) -> None:
        expected = {
            7: {"wolf": 2, "seer": 1, "witch": 1, "villager": 3},
            8: {"wolf": 2, "seer": 1, "witch": 1, "guard": 1, "villager": 3},
            9: {"wolf": 3, "seer": 1, "witch": 1, "guard": 1, "villager": 3},
            10: {"wolf": 3, "seer": 1, "witch": 1, "guard": 1, "hunter": 1, "villager": 3},
            11: {
                "wolf": 3,
                "seer": 1,
                "witch": 1,
                "guard": 1,
                "hunter": 1,
                "idiot": 1,
                "villager": 3,
            },
            12: {
                "wolf": 4,
                "seer": 1,
                "witch": 1,
                "guard": 1,
                "hunter": 1,
                "villager": 4,
            },
        }
        for player_count, role_counts in expected.items():
            rules = create_rules_for_player_count(player_count)
            self.assertEqual(len(rules.role_deck), player_count)
            self.assertEqual(Counter(rules.role_deck), role_counts)
            engine = GameEngine(
                game_id=f"preset-{player_count}",
                players=players(player_count),
                rules=rules,
                seed=f"preset-{player_count}",
            )
            engine.start()
            self.assertEqual(Counter(engine.role_assignments().values()), role_counts)

        custom = create_rules_for_player_count(
            7, optional_roles=(ROLE_GUARD, ROLE_HUNTER, ROLE_IDIOT)
        )
        self.assertEqual(
            Counter(custom.role_deck),
            {"wolf": 2, "seer": 1, "witch": 1, "guard": 1, "hunter": 1, "idiot": 1},
        )
        with self.assertRaisesRegex(ValueError, "仅支持"):
            create_rules_for_player_count(6)

    def test_same_seed_and_private_views_are_isolated(self) -> None:
        first, second = engine_for("repeatable"), engine_for("repeatable")
        first.start()
        second.start()
        self.assertEqual(first.role_assignments(), second.role_assignments())
        assignments = first.role_assignments()
        wolf_id = next(player_id for player_id, role in assignments.items() if role == ROLE_WOLF)
        villager_id = next(player_id for player_id, role in assignments.items() if role == ROLE_VILLAGER)
        self.assertTrue(any(event["type"] == "WOLF_TEAM_REVEALED" for event in first.events_for(wolf_id)))
        self.assertFalse(any(event["type"] == "WOLF_TEAM_REVEALED" for event in first.events_for(villager_id)))
        self.assertNotIn("roles_revealed", first.player_view(villager_id)["public_state"])

    def test_speech_limits_and_action_schema(self) -> None:
        engine = engine_for("speech-limits")
        engine.start()
        wolf_id = next(player.player_id for player in engine.wolf_players())
        wolf_request = engine.discussion_request(wolf_id, "wolf")
        self.assertEqual(wolf_request["allowed_actions"][0]["max_chars"], 30)
        with self.assertRaisesRegex(RuleViolationError, "字数"):
            engine.accept_action(wolf_request, {"kind": ACTION_SPEAK, "text": "我" * 31})
        with self.assertRaisesRegex(RuleViolationError, "中文"):
            engine.accept_action(wolf_request, {"kind": ACTION_SPEAK, "text": "hello"})
        # 英文或协议编号可以作为中文发言的一部分；规则只要求至少含有中文，
        # 不再把 p3 / Player 3 之类的自然对局文本误判为非法。
        accepted = engine.accept_action(
            wolf_request, {"kind": ACTION_SPEAK, "text": "我会重点关注 p3 的票型。"}
        )
        self.assertEqual(accepted["text"], "我会重点关注 p3 的票型。")
        with self.assertRaisesRegex(RuleViolationError, "目标"):
            engine.accept_action(wolf_request, {"kind": ACTION_PASS, "target_id": "p1"})

        engine.enter_phase(PHASE_DAY_DISCUSSION)
        player_id = engine.alive_players()[0].player_id
        day_request = engine.discussion_request(player_id, "public")
        self.assertEqual(day_request["allowed_actions"][0]["max_chars"], 200)
        with self.assertRaisesRegex(RuleViolationError, "字数"):
            engine.accept_action(day_request, {"kind": ACTION_SPEAK, "text": "我" * 201})

    def test_wolves_can_target_a_wolf_teammate(self) -> None:
        engine = engine_for("wolf-self-target")
        engine.start()
        wolf_ids = [player.player_id for player in engine.wolf_players()]
        request = engine.wolf_vote_request(wolf_ids[0])
        kill_action = next(
            action
            for action in request["allowed_actions"]
            if action["kind"] == "wolf_kill_vote"
        )
        self.assertIn(wolf_ids[1], kill_action["target_ids"])

    def test_wolves_win_when_all_special_roles_are_dead(self) -> None:
        rules = RuleSet(
            rule_id="slaughter-side-test",
            role_deck=(ROLE_WOLF, ROLE_WOLF, ROLE_SEER, ROLE_VILLAGER, ROLE_VILLAGER),
            enable_sheriff_election=False,
            last_words_count=0,
        )
        engine = GameEngine(
            game_id="slaughter-side-test",
            players=players(5),
            rules=rules,
            seed="slaughter-side-test",
        )
        engine.start()
        seer_id = next(
            player_id
            for player_id, role in engine.role_assignments().items()
            if role == ROLE_SEER
        )
        engine.enter_phase(PHASE_DAY_VOTE)
        actions = []
        for voter in engine.day_voters():
            request = engine.day_vote_request(voter.player_id)
            raw = (
                {"kind": ACTION_PASS}
                if voter.player_id == seer_id
                else {"kind": "day_vote", "target_id": seer_id}
            )
            actions.append(
                engine.accept_action(request, raw)
            )
        engine.resolve_day_vote(actions)
        self.assertEqual(engine.public_state()["winner"], "wolf")

    def test_first_night_dead_player_can_be_elected_then_pass_badge(self) -> None:
        rules = create_rules_for_player_count(7, last_words_count=0)
        engine = GameEngine(
            game_id="first-night-sheriff-test",
            players=players(),
            rules=rules,
            seed="first-night-sheriff-test",
        )
        engine.start()
        assignments = engine.role_assignments()
        target_id = next(
            player_id for player_id, role in assignments.items() if role == ROLE_VILLAGER
        )

        engine.enter_phase(PHASE_NIGHT_WOLF_VOTE)
        wolf_actions = []
        for wolf in engine.wolf_players():
            request = engine.wolf_vote_request(wolf.player_id)
            wolf_actions.append(
                engine.accept_action(request, {"kind": "wolf_kill_vote", "target_id": target_id})
            )
        engine.resolve_wolf_vote(wolf_actions)
        engine.enter_phase(PHASE_NIGHT_RESOLVE)
        engine.resolve_night()

        self.assertTrue(next(player for player in engine.alive_players() if player.player_id == target_id))
        self.assertTrue(engine.has_pending_sheriff_election())
        engine.begin_sheriff_election()
        candidacies = []
        for player in engine.alive_players():
            request = engine.sheriff_candidacy_request(player.player_id)
            candidacies.append(
                engine.accept_action(
                    request,
                    {"kind": ACTION_SHERIFF_CANDIDATE if player.player_id == target_id else ACTION_PASS},
                )
            )
        engine.resolve_sheriff_candidacies(candidacies)
        self.assertEqual(engine.phase, PHASE_SHERIFF_ELECTION_SPEECH)
        engine.begin_sheriff_election_vote()
        election_actions = []
        for player in engine.alive_players():
            request = engine.sheriff_election_vote_request(player.player_id)
            election_actions.append(
                engine.accept_action(request, {"kind": ACTION_SHERIFF_VOTE, "target_id": target_id})
            )
        engine.resolve_sheriff_election_vote(election_actions)
        self.assertEqual(engine.sheriff_id, target_id)

        engine.reveal_deferred_first_dawn()
        self.assertFalse(engine._require_player(target_id).alive)
        self.assertTrue(engine.has_pending_sheriff_badge_resolution())
        engine.begin_sheriff_badge_resolution()
        badge_request = engine.sheriff_badge_request()
        successor_id = badge_request["allowed_actions"][0]["target_ids"][0]
        badge_action = engine.accept_action(
            badge_request,
            {"kind": "sheriff_badge_transfer", "target_id": successor_id},
        )
        engine.resolve_sheriff_badge(badge_action)
        self.assertEqual(engine.sheriff_id, successor_id)

    def test_night_deaths_have_no_last_words_and_day_eliminations_do(self) -> None:
        rules = RuleSet(
            rule_id="last-words-test",
            role_deck=(ROLE_WOLF, ROLE_WOLF, ROLE_SEER, "witch", ROLE_VILLAGER, ROLE_VILLAGER, ROLE_VILLAGER),
            enable_sheriff_election=False,
            last_words_count=3,
        )
        engine = GameEngine(
            game_id="last-words-test",
            players=players(),
            rules=rules,
            seed="last-words-test",
        )
        engine.start()
        engine.enter_phase(PHASE_NIGHT_RESOLVE)
        night_dead_ids = ["p1", "p2"]
        engine._kill_players(night_dead_ids)
        self.assertFalse(engine.has_pending_last_words())
        self.assertEqual(engine.audit_state()["last_words_eligible_ids"], [])
        self.assertEqual(
            engine.public_rule_summary()["last_words"]["eligibility"],
            "day_elimination_only",
        )

        engine.enter_phase(PHASE_DAY_VOTE)
        day_eliminated_ids = ["p3", "p4", "p5", "p6"]
        vote_actions = []
        for voter in engine.day_voters():
            request = engine.day_vote_request(voter.player_id)
            vote_actions.append(
                engine.accept_action(
                    request,
                    (
                        {"kind": ACTION_PASS}
                        if voter.player_id == day_eliminated_ids[0]
                        else {"kind": "day_vote", "target_id": day_eliminated_ids[0]}
                    ),
                )
            )
        engine.resolve_day_vote(vote_actions)
        engine._kill_players(day_eliminated_ids[1:], eligible_for_last_words=True)
        self.assertTrue(engine.has_pending_last_words())
        engine.begin_last_words()
        request = engine.last_words_request()
        with self.assertRaisesRegex(RuleViolationError, "字数"):
            engine.accept_action(request, {"kind": ACTION_LAST_WORDS, "text": "我" * 201})
        while engine.has_pending_last_words():
            request = engine.last_words_request()
            action = engine.accept_action(
                request,
                {"kind": ACTION_LAST_WORDS, "text": "这是我的遗言"},
            )
            engine.resolve_last_words(action)
        last_words = [
            event["payload"]["player_id"]
            for event in engine.public_events()
            if event["type"] == "PLAYER_LAST_WORDS"
        ]
        self.assertEqual(last_words, day_eliminated_ids[:3])

    def test_sheriff_vote_weight_and_reverse_speech_order(self) -> None:
        engine = GameEngine(
            game_id="sheriff-weight-test",
            players=players(),
            rules=create_default_rules(enable_sheriff_election=False, last_words_count=0),
            seed="sheriff-weight-test",
        )
        engine.start()
        engine.sheriff_id = "p1"
        engine.start_day()
        self.assertTrue(engine.prepare_day_speech_order())
        order_request = engine.sheriff_speech_order_request()
        order_action = engine.accept_action(
            order_request,
            {"kind": ACTION_SHERIFF_SPEECH_ORDER, "target_id": "previous"},
        )
        engine.resolve_sheriff_speech_order(order_action)
        self.assertEqual(
            [player.player_id for player in engine.day_discussion_players()][:3],
            ["p1", "p7", "p6"],
        )

        engine.enter_phase(PHASE_DAY_VOTE)
        target_a, target_b = "p6", "p7"
        actions = []
        for voter in engine.day_voters():
            request = engine.day_vote_request(voter.player_id)
            if voter.player_id in {"p1", "p2"}:
                raw = {"kind": "day_vote", "target_id": target_a}
            elif voter.player_id in {"p3", "p4"}:
                raw = {"kind": "day_vote", "target_id": target_b}
            else:
                raw = {"kind": ACTION_PASS}
            actions.append(engine.accept_action(request, raw))
        engine.resolve_day_vote(actions)
        outcome = next(
            event["payload"]
            for event in engine.audit_events()
            if event["type"] == "DAY_VOTE_RESOLVED"
        )
        self.assertEqual(outcome["target"], target_a)
        self.assertEqual(outcome["counts"][target_a], 2.5)

    def test_guard_blocks_wolf_attack_and_cannot_repeat_by_default(self) -> None:
        engine = GameEngine(
            game_id="guard-test",
            players=players(8),
            rules=create_rules_for_player_count(8, enable_sheriff_election=False),
            seed="guard-test",
        )
        engine.start()
        assignments = engine.role_assignments()
        guard_id = next(player_id for player_id, role in assignments.items() if role == ROLE_GUARD)
        protected_id = next(
            player_id
            for player_id, role in assignments.items()
            if player_id != guard_id and role != ROLE_WOLF
        )

        engine.enter_phase(PHASE_NIGHT_GUARD)
        guard_request = engine.guard_request(guard_id)
        guard_action = engine.accept_action(
            guard_request,
            {"kind": ACTION_GUARD_PROTECT, "target_id": protected_id},
        )
        engine.resolve_guard_actions([guard_action])

        engine.enter_phase(PHASE_NIGHT_WOLF_VOTE)
        wolf_actions = []
        for wolf in engine.wolf_players():
            request = engine.wolf_vote_request(wolf.player_id)
            wolf_actions.append(
                engine.accept_action(
                    request,
                    {"kind": "wolf_kill_vote", "target_id": protected_id},
                )
            )
        engine.resolve_wolf_vote(wolf_actions)
        engine.enter_phase(PHASE_NIGHT_RESOLVE)
        engine.resolve_night()

        alive_ids = {player["id"] for player in engine.public_state()["players"] if player["alive"]}
        self.assertIn(protected_id, alive_ids)
        self.assertEqual(engine.night.guarded_targets, [protected_id])
        self.assertTrue(any(event["type"] == "NO_ONE_DIED" for event in engine.public_events()))

        engine.start_day()
        engine.start_next_night()
        engine.enter_phase(PHASE_NIGHT_GUARD)
        next_request = engine.guard_request(guard_id)
        protect_action = next(
            action
            for action in next_request["allowed_actions"]
            if action["kind"] == ACTION_GUARD_PROTECT
        )
        self.assertNotIn(protected_id, protect_action["target_ids"])

    def test_idiot_survives_first_day_elimination_then_loses_vote(self) -> None:
        engine = GameEngine(
            game_id="idiot-test",
            players=players(),
            rules=create_rules_for_player_count(7, optional_roles=(ROLE_IDIOT,)),
            seed="idiot-test",
        )
        engine.start()
        idiot_id = next(
            player_id
            for player_id, role in engine.role_assignments().items()
            if role == ROLE_IDIOT
        )

        engine.enter_phase(PHASE_DAY_VOTE)
        first_vote_actions = []
        for voter in engine.day_voters():
            request = engine.day_vote_request(voter.player_id)
            target_id = idiot_id if idiot_id in request["allowed_actions"][0]["target_ids"] else request["allowed_actions"][0]["target_ids"][0]
            first_vote_actions.append(
                engine.accept_action(
                    request, {"kind": "day_vote", "target_id": target_id}
                )
            )
        engine.resolve_day_vote(first_vote_actions)

        state = engine.public_state()
        idiot_public = next(player for player in state["players"] if player["id"] == idiot_id)
        self.assertTrue(idiot_public["alive"])
        self.assertEqual(idiot_public["revealed_role"], ROLE_IDIOT)
        self.assertFalse(idiot_public["can_day_vote"])
        self.assertFalse(engine.can_day_vote(idiot_id))
        self.assertNotIn(idiot_id, [player.player_id for player in engine.day_voters()])
        with self.assertRaisesRegex(RuleViolationError, "投票权"):
            engine.day_vote_request(idiot_id)
        self.assertTrue(any(event["type"] == "IDIOT_REVEALED" for event in engine.public_events()))

        second_vote_actions = []
        for voter in engine.day_voters():
            request = engine.day_vote_request(voter.player_id)
            second_vote_actions.append(
                engine.accept_action(
                    request, {"kind": "day_vote", "target_id": idiot_id}
                )
            )
        engine.resolve_day_vote(second_vote_actions)
        second_state = engine.public_state()
        self.assertFalse(next(player for player in second_state["players"] if player["id"] == idiot_id)["alive"])

    def test_hunter_reaction_occurs_before_winner_check(self) -> None:
        engine = GameEngine(
            game_id="hunter-test",
            players=players(),
            rules=create_rules_for_player_count(
                7,
                optional_roles=(ROLE_HUNTER,),
                enable_sheriff_election=False,
            ),
            seed="hunter-test",
        )
        engine.start()
        assignments = engine.role_assignments()
        hunter_id = next(player_id for player_id, role in assignments.items() if role == ROLE_HUNTER)
        wolf_id = next(player_id for player_id, role in assignments.items() if role == ROLE_WOLF)

        engine.enter_phase(PHASE_NIGHT_WOLF_VOTE)
        wolf_actions = []
        for wolf in engine.wolf_players():
            request = engine.wolf_vote_request(wolf.player_id)
            wolf_actions.append(
                engine.accept_action(
                    request,
                    {"kind": "wolf_kill_vote", "target_id": hunter_id},
                )
            )
        engine.resolve_wolf_vote(wolf_actions)
        engine.enter_phase(PHASE_NIGHT_RESOLVE)
        engine.resolve_night()
        self.assertTrue(engine.has_pending_hunter_reactions())
        self.assertEqual(engine.status, "running")

        request = engine.hunter_shot_request()
        self.assertEqual(request["player_id"], hunter_id)
        action = engine.accept_action(
            request,
            {"kind": ACTION_HUNTER_SHOOT, "target_id": wolf_id},
        )
        engine.resolve_hunter_shot(action)

        alive_ids = {player["id"] for player in engine.public_state()["players"] if player["alive"]}
        self.assertNotIn(wolf_id, alive_ids)
        self.assertFalse(engine.has_pending_hunter_reactions())
        self.assertEqual(engine.phase, PHASE_NIGHT_RESOLVE)
        self.assertTrue(any(event["type"] == "HUNTER_SHOT_FIRED" for event in engine.public_events()))

    def test_hunter_chain_restores_the_original_phase(self) -> None:
        rules = RuleSet(
            rule_id="two-hunter-test",
            # 保留一个存活神职，避免夜间猎人连锁本身直接触发屠边，
            # 从而专门验证反应结束后会恢复到原夜间阶段。
            role_deck=(ROLE_WOLF, ROLE_HUNTER, ROLE_HUNTER, ROLE_SEER, ROLE_VILLAGER),
            enable_sheriff_election=False,
        )
        engine = GameEngine(
            game_id="hunter-chain-test",
            players=players(5),
            rules=rules,
            seed="hunter-chain-test",
        )
        engine.start()
        assignments = engine.role_assignments()
        hunter_ids = [
            player_id for player_id, role in assignments.items() if role == ROLE_HUNTER
        ]
        first_hunter, second_hunter = hunter_ids

        engine.enter_phase(PHASE_NIGHT_WOLF_VOTE)
        wolf = engine.wolf_players()[0]
        wolf_request = engine.wolf_vote_request(wolf.player_id)
        wolf_action = engine.accept_action(
            wolf_request,
            {"kind": "wolf_kill_vote", "target_id": first_hunter},
        )
        engine.resolve_wolf_vote([wolf_action])
        engine.enter_phase(PHASE_NIGHT_RESOLVE)
        engine.resolve_night()

        first_request = engine.hunter_shot_request()
        first_action = engine.accept_action(
            first_request,
            {"kind": ACTION_HUNTER_SHOOT, "target_id": second_hunter},
        )
        engine.resolve_hunter_shot(first_action)
        self.assertTrue(engine.has_pending_hunter_reactions())

        second_request = engine.hunter_shot_request()
        second_action = engine.accept_action(second_request, {"kind": ACTION_PASS})
        engine.resolve_hunter_shot(second_action)
        self.assertEqual(engine.phase, PHASE_NIGHT_RESOLVE)


class GameRunnerTest(unittest.IsolatedAsyncioTestCase):
    async def test_runner_processes_hunter_reaction_before_continuing_day(self) -> None:
        engine = GameEngine(
            game_id="runner-hunter-reaction",
            players=players(),
            rules=create_rules_for_player_count(7, optional_roles=(ROLE_HUNTER,)),
            seed="runner-hunter-reaction",
        )
        engine.start()
        assignments = engine.role_assignments()
        hunter_id = next(
            player_id for player_id, role in assignments.items() if role == ROLE_HUNTER
        )
        wolf_ids = {
            player_id for player_id, role in assignments.items() if role == ROLE_WOLF
        }
        shot_target = sorted(wolf_ids)[0]

        def hunter_scenario_strategy(packet: dict) -> dict:
            request = packet["request"]
            allowed = request["allowed_actions"]
            phase = request["phase"]
            if phase == PHASE_NIGHT_WOLF_VOTE:
                return {"kind": "wolf_kill_vote", "target_id": hunter_id}
            if phase == "night_witch":
                return {"kind": ACTION_PASS}
            if phase == "hunter_reaction":
                return {"kind": ACTION_HUNTER_SHOOT, "target_id": shot_target}
            if phase == PHASE_DAY_VOTE:
                targets = allowed[0].get("target_ids", [])
                wolf_targets = sorted(wolf_ids.intersection(targets))
                return {
                    "kind": "day_vote",
                    "target_id": wolf_targets[0] if wolf_targets else targets[0],
                }
            action = next(
                (item for item in allowed if item["kind"] != ACTION_PASS), allowed[0]
            )
            if action["kind"] in {ACTION_SPEAK, ACTION_LAST_WORDS}:
                return {"kind": action["kind"], "text": "我会依据公开票型继续判断。"}
            result = {"kind": action["kind"]}
            if action.get("target_ids"):
                result["target_id"] = action["target_ids"][0]
            return result

        participants = {
            player["id"]: ScriptedParticipant(player["id"], hunter_scenario_strategy)
            for player in players()
        }
        with tempfile.TemporaryDirectory() as temporary:
            report = await GameRunner(
                engine=engine,
                participants=participants,
                record_directory=temporary,
                decision_timeout_seconds=1,
            ).run()
            public_markdown = Path(report["public_record_markdown_path"]).read_text(
                encoding="utf-8"
            )
        self.assertEqual(report["errors"], [])
        self.assertEqual(report["public_state"]["winner"], "village")
        public_types = [event["type"] for event in engine.public_events()]
        self.assertIn("HUNTER_SHOT_FIRED", public_types)
        self.assertLess(public_types.index("HUNTER_SHOT_FIRED"), public_types.index("DAY_STARTED"))
        self.assertIn("### 猎人反应", public_markdown)
        self.assertIn("开枪带走", public_markdown)

    async def test_twelve_player_preset_completes_with_all_loops_enabled(self) -> None:
        rules = create_rules_for_player_count(12)
        engine = GameEngine(
            game_id="twelve-player-runner",
            players=players(12),
            rules=rules,
            seed="twelve-player-runner",
        )
        participants = {
            player["id"]: ScriptedParticipant(player["id"], deterministic_strategy)
            for player in players(12)
        }
        report = await GameRunner(
            engine=engine,
            participants=participants,
            record_store=False,
            decision_timeout_seconds=1,
        ).run()
        self.assertEqual(report["errors"], [])
        self.assertEqual(report["public_state"]["status"], "finished")
        self.assertEqual(Counter(engine.role_assignments().values()), Counter(rules.role_deck))
        audit_types = {event["type"] for event in engine.audit_events()}
        self.assertIn("GUARD_PHASE_RESOLVED", audit_types)

    async def test_runner_completes_and_writes_separate_public_record(self) -> None:
        engine = engine_for("runner-seed")
        participants = {
            player["id"]: ScriptedParticipant(player["id"], deterministic_strategy)
            for player in players()
        }
        with tempfile.TemporaryDirectory() as temporary:
            recorder = PublicRecorder()
            report = await GameRunner(
                engine=engine,
                participants=participants,
                decision_timeout_seconds=1,
                record_directory=temporary,
                public_recorder=recorder,
            ).run()
            self.assertEqual(report["public_state"]["status"], "finished")
            self.assertIn(report["public_state"]["winner"], {"wolf", "village"})
            self.assertEqual(report["errors"], [])
            self.assertTrue(report["record_path"])
            self.assertTrue(report["public_record_path"])
            self.assertEqual(stat.S_IMODE(Path(report["record_path"]).stat().st_mode), 0o600)
            self.assertEqual(stat.S_IMODE(Path(report["public_record_path"]).stat().st_mode), 0o600)

            record = json.loads(Path(report["record_path"]).read_text(encoding="utf-8"))
            event_types = {event["type"] for event in record["events"]}
            self.assertTrue({"ROLE_ASSIGNMENTS_CREATED", "PLAYER_SPOKE", "VOTE_CAST", "GAME_FINISHED"} <= event_types)
            self.assertTrue(any(event["type"] == "PUBLIC_STATE_SYNC" for event in record["runner_events"]))

            public_record = json.loads(Path(report["public_record_path"]).read_text(encoding="utf-8"))
            public_serialized = json.dumps(public_record, ensure_ascii=False)
            self.assertTrue(all(event["visibility"] == "public" for event in public_record["events"]))
            self.assertNotIn("seed", public_record["metadata"])
            self.assertNotIn("ROLE_ASSIGNMENTS_CREATED", public_serialized)
            self.assertNotIn("WOLF_TEAM_REVEALED", public_serialized)
            public_markdown = Path(report["public_record_markdown_path"]).read_text(encoding="utf-8")
            self.assertNotIn("### 狼人私聊", public_markdown)
            self.assertNotIn("### 夜间行动（审计）", public_markdown)
            snapshot = recorder.snapshot()
            self.assertTrue(snapshot["events"])
            self.assertTrue(all(event["visibility"] == "public" for event in snapshot["events"]))

    async def test_every_living_player_receives_public_sync_at_night_start(self) -> None:
        engine = engine_for("night-sync")
        synchronizations: list[dict] = []
        participants = {}
        for player in players():
            participant = ScriptedParticipant(player["id"], deterministic_strategy)

            async def observe(packet: dict, sink=synchronizations) -> None:
                sink.append(packet)

            participant.observe = observe  # type: ignore[method-assign]
            participants[player["id"]] = participant
        report = await GameRunner(
            engine=engine,
            participants=participants,
            record_store=False,
            decision_timeout_seconds=1,
        ).run()
        self.assertEqual(report["errors"], [])
        self.assertGreaterEqual(len(synchronizations), 7)
        for packet in synchronizations[:7]:
            self.assertEqual(packet["type"], "public_state_sync")
            self.assertEqual(packet["game"]["phase"], "night")
            self.assertNotIn("roles_revealed", packet["public_state"])
            self.assertNotIn("ROLE_ASSIGNMENTS_CREATED", json.dumps(packet, ensure_ascii=False))

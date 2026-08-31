from __future__ import annotations

import json
from pathlib import Path
import unittest
from unittest.mock import patch

from werewolf_game import (
    GameEngine,
    LlmPublicNarrator,
    create_default_rules,
)
from werewolf_game.constants import (
    ROLE_GUARD,
    ROLE_HUNTER,
    ROLE_IDIOT,
    ROLE_SEER,
    ROLE_VILLAGER,
    ROLE_WITCH,
    ROLE_WOLF,
)
from werewolf_game.errors import ModelClientError
from werewolf_game.participants import LlmParticipant
from werewolf_game.llm.client import ModelClient, ModelResponse, get_model_config


def players() -> list[dict[str, str]]:
    return [{"id": f"p{index}", "name": f"Player {index}"} for index in range(1, 8)]


class FakeModelClient:
    def __init__(self, response: dict) -> None:
        self.response = response
        self.requests: list[dict] = []

    async def complete_json(self, **request: object) -> dict:
        self.requests.append(request)
        return self.response


class SequenceModelClient:
    def __init__(self, responses: list[object]) -> None:
        self.responses = list(responses)
        self.requests: list[dict] = []

    async def complete_json(self, **request: object) -> dict:
        self.requests.append(request)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        assert isinstance(response, dict)
        return response


class LlmParticipantTest(unittest.IsolatedAsyncioTestCase):
    async def test_every_supported_role_has_base_and_task_file(self) -> None:
        role_directory = Path(__file__).resolve().parents[1] / "werewolf_game" / "prompts" / "roles"
        for role in (
            ROLE_WOLF,
            ROLE_VILLAGER,
            ROLE_SEER,
            ROLE_WITCH,
            ROLE_GUARD,
            ROLE_HUNTER,
            ROLE_IDIOT,
        ):
            base = role_directory / role / "base.md"
            task = role_directory / role / "task.md"
            self.assertTrue(base.exists(), f"缺少 {role} 的固定规则档案")
            self.assertTrue(task.exists(), f"缺少 {role} 的最小任务档案")
            self.assertGreater(len(base.read_text(encoding="utf-8").strip()), 80)
            self.assertGreater(len(task.read_text(encoding="utf-8").strip()), 80)

    async def test_player_prompt_comes_from_template_and_receives_only_own_view(self) -> None:
        engine = GameEngine(
            game_id="llm-view-test",
            players=players(),
            rules=create_default_rules(),
            seed="llm-view-seed",
        )
        engine.start()
        wolf_id = next(player_id for player_id, role in engine.role_assignments().items() if role == ROLE_WOLF)
        request = engine.discussion_request(wolf_id, "wolf")
        client = FakeModelClient(
            {"kind": "speak", "text": "今晚先观察队友的倾向。", "private_note": "已完成夜聊。"}
        )
        participant = LlmParticipant(player_id=wolf_id, model_client=client)
        decision = await participant.decide(engine.build_turn_packet(request))

        self.assertEqual(decision["kind"], "speak")
        self.assertEqual(decision["player_id"], wolf_id)
        prompted = json.loads(client.requests[0]["messages"][0]["content"])
        self.assertNotIn("roles_revealed", prompted["packet"]["public_state"])
        self.assertEqual(prompted["packet"]["public_rules"]["role_counts"]["wolf"], 2)
        self.assertNotIn("guard", prompted["packet"]["public_rules"]["implemented_roles"])
        self.assertNotIn("ROLE_ASSIGNMENTS_CREATED", json.dumps(prompted, ensure_ascii=False))
        self.assertIn("秘密身份：wolf", client.requests[0]["system"])
        self.assertIn("狼人固定规则", client.requests[0]["system"])
        self.assertIn("狼人最小任务", client.requests[0]["system"])
        self.assertEqual(
            participant.agent_manifest()["roles"][ROLE_WOLF]["role"], ROLE_WOLF
        )
        self.assertNotIn("预言家固定规则", client.requests[0]["system"])
        self.assertNotIn("女巫固定规则", client.requests[0]["system"])
        self.assertIn("只返回一个 JSON 对象", client.requests[0]["system"])
        instruction = prompted["instruction"]
        self.assertIn("本次行动的最终 JSON 契约", instruction)
        self.assertIn('{"kind":"speak","text":"好"}', instruction)
        self.assertIn("text 可以包含英文、数字和玩家编号", instruction)
        prompted_packet = prompted["packet"]
        self.assertNotIn("visible_events", prompted_packet)
        self.assertNotIn("tool_context", prompted_packet)
        self.assertEqual(
            client.requests[0]["tools"][0]["name"], "read_current_round_dialogue"
        )

    async def test_action_contract_is_specific_to_current_target_action(self) -> None:
        engine = GameEngine(
            game_id="target-contract-test",
            players=players(),
            rules=create_default_rules(enable_sheriff_election=False),
            seed="target-contract-seed",
        )
        engine.start()
        wolf_id = engine.wolf_players()[0].player_id
        request = engine.wolf_vote_request(wolf_id)
        target_id = next(
            action["target_ids"][0]
            for action in request["allowed_actions"]
            if action["kind"] == "wolf_kill_vote"
        )
        client = FakeModelClient(
            {"kind": "wolf_kill_vote", "target_id": target_id}
        )
        participant = LlmParticipant(player_id=wolf_id, model_client=client)

        decision = await participant.decide(engine.build_turn_packet(request))

        self.assertEqual(decision["target_id"], target_id)
        prompted = json.loads(client.requests[0]["messages"][0]["content"])
        instruction = prompted["instruction"]
        self.assertIn('{"kind":"wolf_kill_vote","target_id":"', instruction)
        self.assertIn(f"target_id 候选值：{target_id}", instruction)

    async def test_invalid_action_is_regenerated_twice_without_extra_tool_budget(self) -> None:
        engine = GameEngine(
            game_id="invalid-output-retry-test",
            players=players(),
            rules=create_default_rules(enable_sheriff_election=False),
            seed="invalid-output-retry-seed",
        )
        engine.start()
        wolf_id = engine.wolf_players()[0].player_id
        request = engine.discussion_request(wolf_id, "wolf")
        # 第一个输出不含中文，第二个带 p3 的中英文混合发言则应合法。
        client = SequenceModelClient(
            [
                {"kind": "speak", "text": "hello"},
                {"kind": "speak", "text": "我会重点看 p3 的票型。"},
            ]
        )
        participant = LlmParticipant(player_id=wolf_id, model_client=client)

        decision = await participant.decide(engine.build_turn_packet(request))

        self.assertEqual(decision["text"], "我会重点看 p3 的票型。")
        self.assertEqual(len(client.requests), 2)
        self.assertIsNotNone(client.requests[0]["tools"])
        self.assertIsNone(client.requests[1]["tools"])
        self.assertEqual(client.requests[1]["max_tool_calls"], 0)
        correction = json.loads(client.requests[1]["messages"][0]["content"])
        self.assertIn("发言必须包含中文", correction["instruction"])
        self.assertEqual(
            participant.agent_manifest()["max_decision_retries"], 2
        )

    async def test_model_failure_has_two_additional_generation_attempts(self) -> None:
        engine = GameEngine(
            game_id="model-failure-retry-test",
            players=players(),
            rules=create_default_rules(enable_sheriff_election=False),
            seed="model-failure-retry-seed",
        )
        engine.start()
        wolf_id = engine.wolf_players()[0].player_id
        request = engine.discussion_request(wolf_id, "wolf")
        client = SequenceModelClient(
            [
                ModelClientError("temporary model failure"),
                ModelClientError("temporary model failure"),
                {"kind": "speak", "text": "我会等更多发言。"},
            ]
        )
        participant = LlmParticipant(player_id=wolf_id, model_client=client)

        decision = await participant.decide(engine.build_turn_packet(request))

        self.assertEqual(decision["kind"], "speak")
        self.assertEqual(len(client.requests), 3)
        self.assertIsNotNone(client.requests[0]["tools"])
        self.assertTrue(all(item["tools"] is None for item in client.requests[1:]))

    async def test_player_accumulates_non_sensitive_model_token_usage(self) -> None:
        engine = GameEngine(
            game_id="token-usage-test",
            players=players(),
            rules=create_default_rules(),
            seed="token-usage-seed",
        )
        engine.start()
        wolf_id = next(
            player_id
            for player_id, role in engine.role_assignments().items()
            if role == ROLE_WOLF
        )
        request = engine.discussion_request(wolf_id, "wolf")
        response = ModelResponse(
            {"kind": "speak", "text": "今晚先观察队友的倾向。"},
            api_attempts=2,
            generic_retries=1,
            usage_limit_retries=0,
            token_usage={"input_tokens": 31, "output_tokens": 9, "total_tokens": 40},
        )
        participant = LlmParticipant(player_id=wolf_id, model_client=FakeModelClient(response))
        await participant.decide(engine.build_turn_packet(request))

        self.assertEqual(
            participant.model_token_usage_snapshot(),
            {
                "successful_response_count": 1,
                "api_attempt_count": 2,
                "reported_usage_response_count": 1,
                "input_tokens": 31,
                "output_tokens": 9,
                "total_tokens": 40,
            },
        )

    async def test_task_agent_reads_current_round_dialogue_only_via_tool(self) -> None:
        engine = GameEngine(
            game_id="tool-view-test",
            players=players(),
            rules=create_default_rules(enable_sheriff_election=False),
            seed="tool-view-seed",
        )
        engine.start()
        wolf_id = next(
            player_id
            for player_id, role in engine.role_assignments().items()
            if role == ROLE_WOLF
        )
        request = engine.discussion_request(wolf_id, "wolf")
        # Put one earlier wolf message in the engine's current-round stream.
        first = engine.wolf_players()[0]
        engine.accept_action(
            engine.discussion_request(first.player_id, "wolf"),
            {"kind": "speak", "text": "先看发言"},
        )
        if first.player_id == wolf_id:
            request = engine.discussion_request(engine.wolf_players()[1].player_id, "wolf")
            wolf_id = request["player_id"]

        config = get_model_config(
            "api",
            {
                "OPENAI_MODEL": "demo-model",
                "OPENAI_BASE_URL": "https://example.test/v1",
                "OPENAI_API_KEY": "server-key",
                "MODEL_API_MODE": "responses",
                "MODEL_MAX_RETRIES": "0",
            },
        )
        client = ModelClient(config)
        responses = [
            {
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "call-1",
                        "name": "read_current_round_dialogue",
                        "arguments": "{}",
                    }
                ]
            },
            {
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": '{"kind":"speak","text":"我先听听大家的看法。"}',
                            }
                        ],
                    }
                ]
            },
        ]
        participant = LlmParticipant(player_id=wolf_id, model_client=client)
        with patch.object(client, "_request_json", side_effect=responses) as mocked:
            decision = await participant.decide(engine.build_turn_packet(request))
        self.assertEqual(decision["kind"], "speak")
        self.assertEqual(mocked.call_count, 2)
        self.assertIn(
            "tool_results",
            json.dumps(mocked.call_args_list[1].args, ensure_ascii=False)
            + json.dumps(mocked.call_args_list[1].kwargs, ensure_ascii=False),
        )

    async def test_public_narrator_filters_secret_event_before_model_call(self) -> None:
        client = FakeModelClient({"summary": "第一夜结束，白天讨论即将开始。"})
        updates: list[dict] = []
        narrator = LlmPublicNarrator(model_client=client, on_update=updates.append)
        public_event = {
            "seq": 10,
            "type": "DAY_STARTED",
            "visibility": "public",
            "channel": "public",
            "recipients": [],
            "payload": {"round": 1},
        }
        secret_event = {
            "seq": 11,
            "type": "WOLF_TARGET_SELECTED",
            "visibility": "team",
            "channel": "wolf",
            "recipients": ["p1", "p2"],
            "payload": {"target_id": "p3"},
        }
        update = await narrator.observe(
            events=[public_event, secret_event],
            public_state={"status": "running", "phase": "day_discussion"},
        )
        self.assertEqual(update["summary"], "第一夜结束，白天讨论即将开始。")
        self.assertEqual(len(updates), 1)
        model_payload = json.loads(client.requests[0]["messages"][0]["content"])
        self.assertEqual(model_payload["public_events"], [public_event])
        self.assertNotIn("WOLF_TARGET_SELECTED", json.dumps(client.requests[0], ensure_ascii=False))

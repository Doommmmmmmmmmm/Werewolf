from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
import unittest

from werewolf_game import (
    GameEngine,
    LlmPublicNarrator,
    RoleStrategyStore,
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
from werewolf_game.participants import LlmParticipant
from werewolf_game.llm.client import ModelResponse
from werewolf_game.prompts import PROMPT_DIRECTORY


def players() -> list[dict[str, str]]:
    return [{"id": f"p{index}", "name": f"Player {index}"} for index in range(1, 8)]


class FakeModelClient:
    def __init__(self, response: dict) -> None:
        self.response = response
        self.requests: list[dict] = []

    async def complete_json(self, **request: object) -> dict:
        self.requests.append(request)
        return self.response


class LlmParticipantTest(unittest.IsolatedAsyncioTestCase):
    async def test_every_supported_role_has_base_and_strategy_files(self) -> None:
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
            strategy = role_directory / role / "strategy.md"
            self.assertTrue(base.exists(), f"缺少 {role} 的固定规则档案")
            self.assertTrue(strategy.exists(), f"缺少 {role} 的经验策略档案")
            self.assertGreater(len(base.read_text(encoding="utf-8").strip()), 80)
            self.assertGreater(len(strategy.read_text(encoding="utf-8").strip()), 80)

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
        self.assertIn("当前策略", client.requests[0]["system"])
        self.assertNotIn("预言家固定规则", client.requests[0]["system"])
        self.assertNotIn("女巫固定规则", client.requests[0]["system"])
        self.assertIn("只返回一个 JSON 对象", client.requests[0]["system"])

    async def test_player_can_use_the_same_custom_skill_store_as_reviewer(self) -> None:
        engine = GameEngine(
            game_id="custom-skill-store-test",
            players=players(),
            rules=create_default_rules(),
            seed="custom-skill-store-seed",
        )
        engine.start()
        wolf_id = next(
            player_id
            for player_id, role in engine.role_assignments().items()
            if role == ROLE_WOLF
        )
        request = engine.discussion_request(wolf_id, "wolf")
        with tempfile.TemporaryDirectory() as temporary:
            prompt_root = Path(temporary) / "prompts"
            shutil.copytree(PROMPT_DIRECTORY / "roles", prompt_root / "roles")
            store = RoleStrategyStore(prompt_root)
            store.replace_strategy(
                ROLE_WOLF,
                "# 狼人当前策略（经验，可能不完全正确）\n\n- CUSTOM-SKILL-VERSION-MARKER 用于验证玩家读取指定版本。",
            )
            client = FakeModelClient({"kind": "speak", "text": "我会比较大家的投票理由。"})
            participant = LlmParticipant(
                player_id=wolf_id,
                model_client=client,
                strategy_store=store,
            )
            await participant.decide(engine.build_turn_packet(request))
            self.assertIn("CUSTOM-SKILL-VERSION-MARKER", client.requests[0]["system"])

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

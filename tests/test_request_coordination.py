from __future__ import annotations

import asyncio
import unittest

from werewolf_game.agents.llm import ModelRequestCoordinator


class BlockingClient:
    def __init__(self) -> None:
        self.calls = 0
        self.first_started = asyncio.Event()
        self.release_first = asyncio.Event()

    async def complete_json(self, **_: object) -> dict:
        self.calls += 1
        if self.calls == 1:
            self.first_started.set()
            await self.release_first.wait()
        return {"kind": "pass"}


class ModelRequestCoordinatorTest(unittest.IsolatedAsyncioTestCase):
    async def test_cancelled_waiter_keeps_global_slot_until_underlying_call_finishes(self) -> None:
        client = BlockingClient()
        coordinator = ModelRequestCoordinator(max_in_flight=1)

        first = asyncio.create_task(
            coordinator.complete_json(client, system="system", messages=[], max_tokens=16)
        )
        await asyncio.wait_for(client.first_started.wait(), timeout=1)
        first.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await first

        second = asyncio.create_task(
            coordinator.complete_json(client, system="system", messages=[], max_tokens=16)
        )
        await asyncio.sleep(0)
        self.assertEqual(client.calls, 1)
        self.assertEqual(coordinator.health_snapshot()["in_flight"], 1)
        self.assertEqual(coordinator.health_snapshot()["queued"], 1)

        client.release_first.set()
        self.assertEqual(await asyncio.wait_for(second, timeout=1), {"kind": "pass"})
        await asyncio.wait_for(coordinator.wait_for_idle(), timeout=1)
        health = coordinator.health_snapshot()
        self.assertEqual(health["completed_requests"], 2)
        self.assertEqual(health["caller_cancelled_count"], 1)

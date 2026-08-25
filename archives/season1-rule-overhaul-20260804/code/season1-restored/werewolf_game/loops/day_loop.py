"""一整个白天的流程。"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from ..constants import PHASE_DAY_DISCUSSION, PHASE_DAY_VOTE
from ..engine import GameEngine
from ..errors import RuleViolationError


DecisionProvider = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]
ActionSubmitter = Callable[[dict[str, Any], dict[str, Any]], Awaitable[dict[str, Any]]]
AsyncHook = Callable[[], Awaitable[None]]


class DayLoop:
    """执行当前回合的白天发言与投票，不负责开启下一夜。"""

    def __init__(
        self,
        engine: GameEngine,
        decide: DecisionProvider,
        submit_action: ActionSubmitter,
        after_state_change: AsyncHook,
    ) -> None:
        self.engine = engine
        self.decide = decide
        self.submit_action = submit_action
        self.after_state_change = after_state_change

    async def run(self) -> None:
        if self.engine.phase != PHASE_DAY_DISCUSSION:
            raise RuleViolationError("DayLoop 必须从白天讨论阶段开始")
        await self._prepare_speech_order()
        await self._public_discussion()
        await self._day_vote()

    async def _prepare_speech_order(self) -> None:
        """以天亮死者为锚点；存活警长可将下一位改为上一位。"""

        if not self.engine.prepare_day_speech_order():
            await self.after_state_change()
            return
        request = self.engine.sheriff_speech_order_request()
        await self.after_state_change()
        action = await self.decide(request)
        accepted = await self.submit_action(request, action)
        await self.after_state_change()
        self.engine.resolve_sheriff_speech_order(accepted)
        await self.after_state_change()

    async def _public_discussion(self) -> None:
        """所有存活玩家按死者锚点和警长方向公开发言。"""

        for player in self.engine.day_discussion_players():
            request = self.engine.discussion_request(player.player_id, "public")
            action = await self.decide(request)
            await self.submit_action(request, action)
            await self.after_state_change()

    async def _day_vote(self) -> None:
        """所有存活玩家同时投票，随后由引擎处理平票、出局和胜负。"""

        self.engine.enter_phase(PHASE_DAY_VOTE)
        await self.after_state_change()
        requests = [
            self.engine.day_vote_request(player.player_id)
            for player in self.engine.day_voters()
        ]
        actions = await self._ask_all(requests)
        accepted = []
        for request, action in zip(requests, actions, strict=True):
            accepted.append(await self.submit_action(request, action))
            await self.after_state_change()
        self.engine.resolve_day_vote(accepted)
        await self.after_state_change()

    async def _ask_all(self, requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not requests:
            return []
        return list(await asyncio.gather(*(self.decide(request) for request in requests)))

"""首日警长竞选流程。

首夜死亡会由 Engine 暂存到本流程完成后才公布，因此候选、竞选发言和投票
都包含首夜实际死亡的玩家。平票只进行一轮候选人 PK，PK 再平则本局无警长。
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from ..constants import PHASE_SHERIFF_ELECTION_SPEECH
from ..engine import GameEngine


DecisionProvider = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]
ActionSubmitter = Callable[[dict[str, Any], dict[str, Any]], Awaitable[dict[str, Any]]]
AsyncHook = Callable[[], Awaitable[None]]


class SheriffElectionLoop:
    """执行首日上警、候选发言、投票与一次 PK。"""

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
        self.engine.begin_sheriff_election()
        await self.after_state_change()
        await self._candidacy()
        while self.engine.phase == PHASE_SHERIFF_ELECTION_SPEECH:
            await self._candidate_speeches()
            await self._vote()

    async def _candidacy(self) -> None:
        requests = [
            self.engine.sheriff_candidacy_request(player.player_id)
            for player in self.engine.alive_players()
        ]
        actions = await self._ask_all(requests)
        accepted = []
        for request, action in zip(requests, actions, strict=True):
            accepted.append(await self.submit_action(request, action))
            await self.after_state_change()
        self.engine.resolve_sheriff_candidacies(accepted)
        await self.after_state_change()

    async def _candidate_speeches(self) -> None:
        for player in self.engine.sheriff_election_speakers():
            request = self.engine.discussion_request(player.player_id, "public")
            action = await self.decide(request)
            await self.submit_action(request, action)
            await self.after_state_change()
        self.engine.begin_sheriff_election_vote()
        await self.after_state_change()

    async def _vote(self) -> None:
        requests = [
            self.engine.sheriff_election_vote_request(player.player_id)
            for player in self.engine.alive_players()
        ]
        actions = await self._ask_all(requests)
        accepted = []
        for request, action in zip(requests, actions, strict=True):
            accepted.append(await self.submit_action(request, action))
            await self.after_state_change()
        self.engine.resolve_sheriff_election_vote(accepted)
        await self.after_state_change()

    async def _ask_all(self, requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not requests:
            return []
        return list(await asyncio.gather(*(self.decide(request) for request in requests)))

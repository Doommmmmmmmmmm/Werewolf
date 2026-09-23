"""死亡警长的公开传警徽/撕警徽流程。"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from ..constants import PHASE_SHERIFF_BADGE
from ..engine import GameEngine


DecisionProvider = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]
ActionSubmitter = Callable[[dict[str, Any], dict[str, Any]], Awaitable[dict[str, Any]]]
AsyncHook = Callable[[], Awaitable[None]]


class SheriffBadgeLoop:
    """逐个处理待交接的死亡警长；当前规则通常只会有一个。"""

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
        if self.engine.phase != PHASE_SHERIFF_BADGE:
            self.engine.begin_sheriff_badge_resolution()
            await self.after_state_change()
        while self.engine.has_pending_sheriff_badge_resolution():
            request = self.engine.sheriff_badge_request()
            action = await self.decide(request)
            accepted = await self.submit_action(request, action)
            await self.after_state_change()
            self.engine.resolve_sheriff_badge(accepted)
            await self.after_state_change()

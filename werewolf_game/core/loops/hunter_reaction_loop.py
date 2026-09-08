"""死亡猎人的开枪反应流程。

它刻意不混入 DayLoop 或 NightLoop：猎人可能被夜间击杀、毒杀或白天放逐，
而且一枪打到另一名猎人时还会产生连锁反应。
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from ..engine import GameEngine


DecisionProvider = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]
ActionSubmitter = Callable[[dict[str, Any], dict[str, Any]], Awaitable[dict[str, Any]]]
AsyncHook = Callable[[], Awaitable[None]]


class HunterReactionLoop:
    """依次处理所有待响应的死亡猎人，直到队列为空。"""

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
        while self.engine.has_pending_hunter_reactions():
            request = self.engine.hunter_shot_request()
            await self.after_state_change()
            action = await self.decide(request)
            accepted = await self.submit_action(request, action)
            await self.after_state_change()
            self.engine.resolve_hunter_shot(accepted)
            await self.after_state_change()

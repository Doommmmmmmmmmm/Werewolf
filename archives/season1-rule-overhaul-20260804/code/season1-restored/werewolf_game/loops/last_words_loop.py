"""前若干名白天放逐出局玩家的公开遗言流程。"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from ..constants import PHASE_LAST_WORDS
from ..engine import GameEngine


DecisionProvider = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]
ActionSubmitter = Callable[[dict[str, Any], dict[str, Any]], Awaitable[dict[str, Any]]]
AsyncHook = Callable[[], Awaitable[None]]


class LastWordsLoop:
    """按白天放逐结算顺序，让已获得资格的玩家依次留下公开遗言。"""

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
        if self.engine.phase != PHASE_LAST_WORDS:
            self.engine.begin_last_words()
            await self.after_state_change()
        while self.engine.has_pending_last_words():
            request = self.engine.last_words_request()
            action = await self.decide(request)
            accepted = await self.submit_action(request, action)
            await self.after_state_change()
            self.engine.resolve_last_words(accepted)
            await self.after_state_change()

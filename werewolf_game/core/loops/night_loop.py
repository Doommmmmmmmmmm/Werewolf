"""一整个黑夜的流程。

按从上到下的顺序阅读，就能看到守卫、狼人私聊、狼人投票、预言家、女巫和夜晚结算；
首夜死讯会由 Runner 在警长竞选结束后才公开。
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from ..constants import (
    PHASE_NIGHT_GUARD,
    PHASE_NIGHT_RESOLVE,
    PHASE_NIGHT_SEER,
    PHASE_NIGHT_WITCH,
    PHASE_NIGHT_WOLF_DISCUSSION,
    PHASE_NIGHT_WOLF_VOTE,
    ROLE_GUARD,
    ROLE_SEER,
    ROLE_WITCH,
)
from ..engine import GameEngine
from ..errors import RuleViolationError


DecisionProvider = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]
ActionSubmitter = Callable[[dict[str, Any], dict[str, Any]], Awaitable[dict[str, Any]]]
AsyncHook = Callable[[], Awaitable[None]]


class NightLoop:
    """执行当前回合的夜间阶段，不包含下一天的 DayLoop。"""

    def __init__(
        self,
        engine: GameEngine,
        decide: DecisionProvider,
        submit_action: ActionSubmitter,
        sync_public_state: AsyncHook,
        after_state_change: AsyncHook,
    ) -> None:
        self.engine = engine
        self.decide = decide
        self.submit_action = submit_action
        self.sync_public_state = sync_public_state
        self.after_state_change = after_state_change

    async def run(self) -> None:
        """运行“当前夜晚”至死亡结果确定。

        首夜会先由 Runner 插入 SheriffElectionLoop，再公开死讯；其后才按需
        处理猎人、警徽和遗言，并决定是否开启白天。
        """

        if self.engine.phase != PHASE_NIGHT_WOLF_DISCUSSION:
            raise RuleViolationError("NightLoop 必须从狼人讨论阶段开始")

        # 每夜开始，先把当前公开状态同步给所有仍存活玩家。
        await self.sync_public_state()
        await self.after_state_change()

        await self._guard_action()
        await self._wolf_discussion()
        await self._wolf_vote()
        await self._seer_action()
        await self._witch_action()
        await self._resolve_night()

    async def _guard_action(self) -> None:
        """守卫在狼人行动前暗中选择一名保护对象。"""

        self.engine.enter_phase(PHASE_NIGHT_GUARD)
        await self.after_state_change()
        requests = [
            self.engine.guard_request(player.player_id)
            for player in self.engine.players_with_role(ROLE_GUARD)
        ]
        actions = await self._ask_all(requests)
        accepted = []
        for request, action in zip(requests, actions, strict=True):
            accepted.append(await self.submit_action(request, action))
            await self.after_state_change()
        self.engine.resolve_guard_actions(accepted)
        await self.after_state_change()

    async def _wolf_discussion(self) -> None:
        """狼人按座位依次在私聊频道发言。"""

        self.engine.enter_phase(PHASE_NIGHT_WOLF_DISCUSSION)
        await self.after_state_change()
        for wolf in self.engine.wolf_players():
            request = self.engine.discussion_request(wolf.player_id, "wolf")
            action = await self.decide(request)
            await self.submit_action(request, action)
            await self.after_state_change()

    async def _wolf_vote(self) -> None:
        """所有存活狼人同时选择刀人目标，再由引擎结算。"""

        self.engine.enter_phase(PHASE_NIGHT_WOLF_VOTE)
        await self.after_state_change()
        requests = [
            self.engine.wolf_vote_request(wolf.player_id)
            for wolf in self.engine.wolf_players()
        ]
        actions = await self._ask_all(requests)
        accepted = []
        for request, action in zip(requests, actions, strict=True):
            accepted.append(await self.submit_action(request, action))
            await self.after_state_change()
        self.engine.resolve_wolf_vote(accepted)
        await self.after_state_change()

    async def _seer_action(self) -> None:
        """预言家在私密频道查验一人；没有存活预言家时仍保留审计结算。"""

        self.engine.enter_phase(PHASE_NIGHT_SEER)
        await self.after_state_change()
        requests = [
            self.engine.seer_request(player.player_id)
            for player in self.engine.players_with_role(ROLE_SEER)
        ]
        actions = await self._ask_all(requests)
        accepted = []
        for request, action in zip(requests, actions, strict=True):
            accepted.append(await self.submit_action(request, action))
            await self.after_state_change()
        self.engine.resolve_seer_actions(accepted)
        await self.after_state_change()

    async def _witch_action(self) -> None:
        """女巫选择解药、毒药或跳过；当前规则每晚最多一种行动。"""

        self.engine.enter_phase(PHASE_NIGHT_WITCH)
        await self.after_state_change()
        requests = [
            self.engine.witch_request(player.player_id)
            for player in self.engine.players_with_role(ROLE_WITCH)
        ]
        actions = await self._ask_all(requests)
        accepted = []
        for request, action in zip(requests, actions, strict=True):
            accepted.append(await self.submit_action(request, action))
            await self.after_state_change()
        self.engine.resolve_witch_actions(accepted)
        await self.after_state_change()

    async def _resolve_night(self) -> None:
        """统一计算死亡；首夜公开天亮结果会延后到警长竞选结束。"""

        self.engine.enter_phase(PHASE_NIGHT_RESOLVE)
        await self.after_state_change()
        self.engine.resolve_night()
        await self.after_state_change()

    async def _ask_all(self, requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """同一投票/技能阶段同步向所有玩家请求行动。"""

        if not requests:
            return []
        return list(await asyncio.gather(*(self.decide(request) for request in requests)))

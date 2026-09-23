"""浏览器真人席位的参与者适配。

`GameRunner._obtain_decision` 从参与者实例读取 `decision_timeout_seconds`，
`None` 表示不超时；真人思考几分钟因此不需要修改引擎或 Runner。本类把一次
`decide` 调用挂在 `asyncio.Future` 上，由 HTTP 层在收到浏览器提交后兑现。

身份、目标合法性和结算仍然由 `GameEngine` 裁决；这里只做即时的输入反馈。
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from ..agents.participants.base import Participant
from ..agents.participants.human import HumanParticipant


class WebHumanParticipant(Participant):
    """通过 HTTP 提交行动的真人玩家。

    `decide` 把当前行动包暴露为 `pending_packet()`，然后等待 `submit()`。
    `observe` 必须立即返回：Runner 的 `_sync_night_public_state` 对 `observe`
    强制套用 Runner 级超时，不读取参与者级覆盖。
    """

    # 真人席位不设行动超时；Runner 会读取这个属性。
    decision_timeout_seconds: float | None = None

    def __init__(
        self, player_id: str, *, on_turn: Callable[[], Awaitable[None]] | None = None
    ) -> None:
        super().__init__(player_id)
        self._pending_packet: dict[str, Any] | None = None
        self._pending_future: asyncio.Future[dict[str, Any]] | None = None
        self.latest_public_sync: dict[str, Any] | None = None
        # 单调递增，供前端判断"是否有新的待办行动"。
        self.turn_revision = 0
        # 夜间阶段变化只产生 audit 可见事件，不会经 PublicRecorder 推给前端；
        # 轮到真人行动时必须单独唤醒 SSE，否则要等到下一次心跳才显示行动面板。
        self.on_turn = on_turn

    async def observe(self, sync_packet: dict[str, Any]) -> None:
        """只保存公开同步包，立即返回。"""

        self.latest_public_sync = sync_packet

    async def decide(self, turn_packet: dict[str, Any]) -> dict[str, Any]:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[dict[str, Any]] = loop.create_future()
        self._pending_packet = turn_packet
        self._pending_future = future
        self.turn_revision += 1
        if self.on_turn is not None:
            await self.on_turn()
        try:
            return await future
        finally:
            # 无论正常兑现、被取消还是异常，都不再对外暴露这次待办。
            self._pending_packet = None
            self._pending_future = None

    # ------------------------------------------------------------------
    # 供 HTTP 层读写的状态。
    # ------------------------------------------------------------------

    def pending_packet(self) -> dict[str, Any] | None:
        """返回当前待办行动包；没有待办时返回 None。"""

        if self._pending_future is None or self._pending_future.done():
            return None
        return self._pending_packet

    def pending_request_id(self) -> str | None:
        packet = self.pending_packet()
        if packet is None:
            return None
        request = packet.get("request") or {}
        request_id = request.get("request_id")
        return str(request_id) if request_id else None

    def validate(self, action: Mapping[str, Any]) -> None:
        """复用终端真人玩家的本地校验，给浏览器即时报错。

        这不替代引擎裁决，只是在明显非法时避免浪费一次行动。
        """

        packet = self.pending_packet()
        if packet is None:
            raise ValueError("当前没有待办行动")
        HumanParticipant._validate_local(dict(action), packet["request"])

    def submit(self, request_id: str, action: Mapping[str, Any]) -> None:
        """兑现一次待办行动。

        `request_id` 不匹配说明这是过期或重复提交，直接拒绝；引擎侧的
        `accept_action` 也会独立拒绝过期请求。
        """

        future = self._pending_future
        if future is None or future.done():
            raise ValueError("当前没有待办行动")
        expected = self.pending_request_id()
        if expected is not None and str(request_id) != expected:
            raise ValueError("行动请求已过期，请刷新当前待办行动")
        self.validate(action)
        future.set_result(dict(action))

    def cancel(self, reason: str = "session_closed") -> None:
        """对局被放弃时唤醒挂起的 decide，避免任务悬挂。"""

        future = self._pending_future
        if future is not None and not future.done():
            future.set_exception(RuntimeError(f"真人席位已退出：{reason}"))

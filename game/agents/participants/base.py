"""参与者抽象接口。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any


class Participant(ABC):
    """任何玩家（LLM、真人网页、脚本）都实现这个小接口。"""

    def __init__(self, player_id: str) -> None:
        if not player_id:
            raise ValueError("参与者必须具有 player_id")
        self.player_id = str(player_id)

    @abstractmethod
    async def decide(self, turn_packet: dict[str, Any]) -> dict[str, Any]:
        """收到自己的私密行动包后，返回一份结构化行动。"""

    async def observe(self, sync_packet: dict[str, Any]) -> None:
        """可选：收到夜晚公开状态同步。"""


class ScriptedParticipant(Participant):
    """测试和示例使用的确定性玩家。"""

    def __init__(
        self,
        player_id: str,
        strategy: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> None:
        super().__init__(player_id)
        self.strategy = strategy

    async def decide(self, turn_packet: dict[str, Any]) -> dict[str, Any]:
        return self.strategy(turn_packet)


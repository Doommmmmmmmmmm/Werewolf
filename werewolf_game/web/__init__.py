"""Web 端可游戏 demo。

真人通过浏览器占据一个席位，其余席位由 Season 2 进化节点驱动。本包只读消费
`season2` 的进化 archive，不修改 `core/` 和 Season 2 配置，因此不会破坏
`EvolutionArchive` 的 core 快照校验。
"""

from .participant import WebHumanParticipant
from .roster import RosterEntry, best_role_nodes
from .session import GameSession, SessionManager
from .server import WebDemoServer, serve

__all__ = [
    "WebHumanParticipant",
    "RosterEntry",
    "best_role_nodes",
    "GameSession",
    "SessionManager",
    "WebDemoServer",
    "serve",
]

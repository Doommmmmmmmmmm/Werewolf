"""Web 端可游戏 demo。

真人通过浏览器占据一个席位，其余席位由已归档赛季的进化节点驱动。节点通过
`ArchivedNodeRepository` 只读加载，不依赖进化框架本身，因此游戏包可独立运行；
没有归档时仍能开局（只是不能用进化节点当对手）。
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

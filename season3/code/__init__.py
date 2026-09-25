"""Season 2 的 Task-Agent 进化运行时。

该包只编排候选节点、Meta-Agent、外部 code agent 和评测；游戏规则仍完全由
``GameEngine`` 裁决。
"""

from .archive import EvolutionArchive, EvolutionNode, NodeStatus
from .config import Season2Config, load_season2_config
from .evolution import EvolutionManager

__all__ = [
    "EvolutionArchive",
    "EvolutionManager",
    "EvolutionNode",
    "NodeStatus",
    "Season2Config",
    "load_season2_config",
]

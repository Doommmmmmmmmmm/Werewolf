"""稳定的狼人杀游戏内核。

本包只包含规则、状态、昼夜流程和整局驱动，不应由任何赛季进化代码修改。
"""

from .engine import GameEngine
from .rules import RuleSet, create_default_rules, create_rules_for_player_count
from .runner import GameRunner

__all__ = [
    "GameEngine",
    "GameRunner",
    "RuleSet",
    "create_default_rules",
    "create_rules_for_player_count",
]

"""Web demo 的对局配置。

这些参数原本借用 Season 2 的 `Season2Config.evaluation`，但那份配置的语义是
"进化实验怎么评测节点"，而且已随赛季一起冻结归档。Demo 只需要其中与单局对局
有关的部分，因此在这里独立定义，避免游戏侧依赖已归档的实验配置。

默认值与 Season 2 的评测条件保持一致，使 demo 里的对局条件和节点被评测时的条件
相同；需要时可在入口处覆盖。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DemoConfig:
    """一局 demo 的规则与 Task-Agent 预算。"""

    # 牌组与规则
    player_count: int = 12
    optional_roles: tuple[str, ...] = ()
    # 参与进化的角色集合，决定要为哪些角色挑选节点。
    roles: tuple[str, ...] = (
        "wolf",
        "villager",
        "seer",
        "witch",
        "guard",
        "hunter",
    )
    # 运行时预算（与 Season 2 评测条件一致）
    decision_timeout_seconds: float = 180.0
    model_max_in_flight: int = 8
    max_decision_retries: int = 2
    max_tool_calls_per_decision: int = 1
    max_tool_result_tokens: int = 800
    max_prompt_chars: int = 12000

    def __post_init__(self) -> None:
        if not 7 <= self.player_count <= 12:
            raise ValueError("player_count 必须在 7 到 12 之间")
        if not self.roles:
            raise ValueError("roles 不能为空")


DEFAULT_DEMO_CONFIG = DemoConfig()

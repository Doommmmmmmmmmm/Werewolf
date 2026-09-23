"""为各角色挑选一个已归档的进化节点。

只读消费归档：Season 2 结束后它的 archive 被冻结到 `archives/season2-<日期>/`，
本模块通过 `ArchivedNodeRepository` 读取，不使用 `EvolutionArchive`
（后者会做配置冻结校验，路径一变就拒绝加载；详见 `node_repo` 的说明）。

四种口径：按 20 局评测胜率、按两套强度模型、以及全部使用 base 节点做对照。
某角色没有可用数据时回退到该角色的 base 节点，保证 demo 总能开局。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any

from .node_repo import ArchivedNodeRepository

STRATEGY_BEST_WIN_RATE = "best_win_rate"
STRATEGY_BASE = "base"
# 归档 summary 里的两套强度模型。评分从预计算结果读取，不在服务启动时重新拟合
# （拟合要扫 2358 局）。
STRATEGY_LOGISTIC_TEAM = "logistic_team"
STRATEGY_BRADLEY_TERRY = "bradley_terry_legacy"
STRATEGIES = (
    STRATEGY_BEST_WIN_RATE,
    STRATEGY_BASE,
    STRATEGY_LOGISTIC_TEAM,
    STRATEGY_BRADLEY_TERRY,
)
RATING_STRATEGIES = (STRATEGY_LOGISTIC_TEAM, STRATEGY_BRADLEY_TERRY)

ARCHIVE_DIRECTORY_ENV = "WEREWOLF_ARCHIVE_DIRECTORY"
# 归档根目录；游戏包不绑定某个特定赛季，按名称排序取最新的一个。
ARCHIVES_ROOT = Path(__file__).resolve().parents[2] / "archives"


def discover_archive_directory() -> Path | None:
    """找出一个可用的赛季归档目录。

    优先使用 ``WEREWOLF_ARCHIVE_DIRECTORY``；否则在 `archives/` 下按名称倒序
    取第一个含 `records/evolution/archive/manifest.json` 的目录。游戏包因此不
    依赖任何特定赛季，没有归档时也能正常运行（只是不能用进化节点当对手）。
    """

    configured = os.environ.get(ARCHIVE_DIRECTORY_ENV)
    if configured:
        return Path(configured)
    if not ARCHIVES_ROOT.is_dir():
        return None
    for candidate in sorted(ARCHIVES_ROOT.iterdir(), reverse=True):
        if not candidate.is_dir():
            continue
        if (candidate / "records" / "evolution" / "archive" / "manifest.json").is_file():
            return candidate
    return None


def default_repository(
    archive_directory: str | Path | None = None,
) -> ArchivedNodeRepository:
    """按约定的归档布局构造只读仓库。

    布局：``<归档>/records/evolution/archive`` 与
    ``<归档>/summary/output/node_ratings.json``。
    """

    root = Path(archive_directory) if archive_directory else discover_archive_directory()
    if root is None:
        raise FileNotFoundError(
            f"找不到进化节点归档；请设置 {ARCHIVE_DIRECTORY_ENV} 指向赛季归档目录"
        )
    return ArchivedNodeRepository(
        root / "records" / "evolution" / "archive",
        ratings_path=root / "summary" / "output" / "node_ratings.json",
    )


@dataclass(frozen=True)
class RosterEntry:
    """一个角色本局使用的节点及其评测背景，可直接展示给前端。"""

    role: str
    node_id: str
    win_rate: float | None
    game_count: int | None
    source: str
    # 以下仅在按强度模型挑选时有值。
    strength: float | None = None
    rating_games: int | None = None
    global_win_rate: float | None = None
    significant_vs_base: bool | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "node_id": self.node_id,
            "win_rate": self.win_rate,
            "game_count": self.game_count,
            "source": self.source,
            "strength": self.strength,
            "rating_games": self.rating_games,
            "global_win_rate": self.global_win_rate,
            "significant_vs_base": self.significant_vs_base,
        }


def _base_node_id(repository: ArchivedNodeRepository, role: str) -> str:
    """回退目标：该角色的 base 节点。"""

    candidate = f"{role}-base"
    try:
        repository.node(candidate)
        return candidate
    except (KeyError, ValueError):
        # 该角色没有以 -base 命名的节点；退回按 depth 查找根节点。
        pass
    for node in repository.nodes_for_role(role):
        if node.depth == 0:
            return node.node_id
    raise ValueError(f"角色 {role} 在 archive 中没有可用节点")


def load_ratings(path: Path | None = None) -> dict[str, Any] | None:
    """读取预计算的强度表；缺失或损坏时返回 None，由调用方回退。

    ``path`` 省略时使用默认归档位置。
    """

    if path is None:
        root = discover_archive_directory()
        if root is None:
            return None
        target = root / "summary" / "output" / "node_ratings.json"
    else:
        target = Path(path)
    try:
        value = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def rating_model_info(ratings: Mapping[str, Any] | None) -> dict[str, Any]:
    """两套强度模型的说明，供前端展示（含退化告警）。"""

    models = (ratings or {}).get("models") or {}
    info: dict[str, Any] = {}
    for key in RATING_STRATEGIES:
        model = models.get(key) or {}
        info[key] = {
            "available": bool(model.get("nodes")),
            "description": model.get("description"),
            "team_intercept": model.get("team_intercept"),
        }
    dataset = (ratings or {}).get("dataset") or {}
    info["dataset"] = {
        "unique_games": dataset.get("unique_games"),
        "winner_counts": dataset.get("winner_counts"),
        "node_count": dataset.get("node_count"),
    }
    return info


def _rank_role_by_rating(
    repository: ArchivedNodeRepository,
    role: str,
    model_key: str,
    ratings: Mapping[str, Any],
) -> RosterEntry | None:
    """按某个强度模型在该角色的 retained 节点里取 strength 最高者。

    强度只能在同角色内部比较：新模型把"狼人阵营天然占优"抽成了阵营基线截距，
    因此跨角色比较 strength 没有意义。
    """

    model = ((ratings.get("models") or {}).get(model_key)) or {}
    nodes = model.get("nodes")
    if not isinstance(nodes, list):
        return None
    # 只认当前 archive 里仍为 retained 的节点，避免用上已舍弃的候选。
    retained = {node.node_id for node in repository.retained_nodes(role)}
    scored: list[tuple[float, float, str, dict[str, Any]]] = []
    for entry in nodes:
        if not isinstance(entry, dict):
            continue
        if entry.get("role") != role:
            continue
        node_id = str(entry.get("node_id") or "")
        if node_id not in retained:
            continue
        strength = entry.get("strength")
        if not isinstance(strength, (int, float)):
            continue
        games = entry.get("games_played")
        scored.append(
            (
                float(strength),
                float(games) if isinstance(games, (int, float)) else 0.0,
                node_id,
                entry,
            )
        )
    if not scored:
        return None
    # 强度降序；并列时取出场局数更多的，最后按 node_id 保持确定性。
    strength, games, node_id, entry = sorted(
        scored, key=lambda item: (-item[0], -item[1], item[2])
    )[0]
    summary = repository.evaluation_summary(node_id) or {}
    win_rate = summary.get("win_rate")
    game_count = summary.get("game_count")
    return RosterEntry(
        role=role,
        node_id=node_id,
        win_rate=float(win_rate) if isinstance(win_rate, (int, float)) else None,
        game_count=int(game_count) if isinstance(game_count, int) else None,
        source=model_key,
        strength=strength,
        rating_games=int(games) or None,
        global_win_rate=(
            float(entry["global_win_rate"])
            if isinstance(entry.get("global_win_rate"), (int, float))
            else None
        ),
        significant_vs_base=entry.get("significant_vs_base"),
    )


def _rank_role(repository: ArchivedNodeRepository, role: str) -> RosterEntry:
    """在 retained 节点中取胜率最高者；胜率相同时取评测局数更多的。"""

    scored: list[tuple[float, int, str]] = []
    for node in repository.retained_nodes(role):
        summary = repository.evaluation_summary(node.node_id)
        if summary is None:
            continue
        win_rate = summary.get("win_rate")
        if not isinstance(win_rate, (int, float)):
            continue
        game_count = summary.get("game_count")
        scored.append(
            (
                float(win_rate),
                int(game_count) if isinstance(game_count, int) else 0,
                node.node_id,
            )
        )
    if not scored:
        return RosterEntry(
            role=role,
            node_id=_base_node_id(repository, role),
            win_rate=None,
            game_count=None,
            source="base_fallback",
        )
    # 胜率降序、局数降序；node_id 升序只为在完全并列时保持确定性。
    win_rate, game_count, node_id = sorted(
        scored, key=lambda item: (-item[0], -item[1], item[2])
    )[0]
    return RosterEntry(
        role=role,
        node_id=node_id,
        win_rate=win_rate,
        game_count=game_count or None,
        source="best_win_rate",
    )


def best_role_nodes(
    repository: ArchivedNodeRepository,
    roles: Sequence[str],
    *,
    overrides: Mapping[str, str] | None = None,
    strategy: str = STRATEGY_BEST_WIN_RATE,
    ratings: Mapping[str, Any] | None = None,
) -> dict[str, RosterEntry]:
    """为每个角色选出本局使用的节点。

    可用策略：

    - ``best_win_rate``：按节点自身 20 局评测的胜率取最高；
    - ``base``：全部使用 base 节点，用于"进化前 vs 进化后"对照；
    - ``logistic_team``：按 season2_summary 的 Logistic 团队模型强度取最高；
    - ``bradley_terry_legacy``：按旧的团队 Bradley-Terry 强度取最高（该模型
      已知退化为胜率的线性缩放，仅用于对照）。

    ``overrides`` 按角色强制指定节点，优先于任何策略；指定的节点必须属于该角色。
    强度模型缺失预计算结果时，自动回退到 ``best_win_rate``。
    """

    if strategy not in STRATEGIES:
        raise ValueError(f"未知的节点选择策略：{strategy}")
    normalized_overrides = {
        str(role): str(node_id) for role, node_id in (overrides or {}).items()
    }
    unknown_roles = sorted(set(normalized_overrides) - set(roles))
    if unknown_roles:
        raise ValueError(f"overrides 包含本局不参与的角色：{'、'.join(unknown_roles)}")

    roster: dict[str, RosterEntry] = {}
    for role in roles:
        override = normalized_overrides.get(role)
        if override is not None:
            node = repository.node(override)
            if node.role != role:
                raise ValueError(f"节点 {override} 属于 {node.role}，不能指定给 {role}")
            summary = repository.evaluation_summary(node.node_id) or {}
            win_rate = summary.get("win_rate")
            game_count = summary.get("game_count")
            roster[role] = RosterEntry(
                role=role,
                node_id=override,
                win_rate=float(win_rate) if isinstance(win_rate, (int, float)) else None,
                game_count=int(game_count) if isinstance(game_count, int) else None,
                source="override",
            )
            continue
        if strategy in RATING_STRATEGIES:
            table = ratings if ratings is not None else load_ratings()
            entry = (
                _rank_role_by_rating(repository, role, strategy, table)
                if table is not None
                else None
            )
            # 预计算结果缺失或该角色没有可用评分时，退回胜率口径而不是开局失败。
            roster[role] = entry if entry is not None else _rank_role(repository, role)
            continue
        if strategy == STRATEGY_BASE:
            node_id = _base_node_id(repository, role)
            summary = repository.evaluation_summary(node_id) or {}
            win_rate = summary.get("win_rate")
            game_count = summary.get("game_count")
            roster[role] = RosterEntry(
                role=role,
                node_id=node_id,
                win_rate=float(win_rate) if isinstance(win_rate, (int, float)) else None,
                game_count=int(game_count) if isinstance(game_count, int) else None,
                source="base",
            )
            continue
        roster[role] = _rank_role(repository, role)
    return roster


def role_node_map(roster: Mapping[str, RosterEntry]) -> dict[str, str]:
    """转换为 `VersionedTaskAgentParticipant` 需要的 ``role -> node_id`` 映射。"""

    return {role: entry.node_id for role, entry in roster.items()}

#!/usr/bin/env python3
"""Season 2 赛季总结：节点强度与胜率统计脚本。

数据来源
--------
唯一权威数据源是 ``werewolf_game/records/season2/evolution/evaluations/<role>/<node_id>/summary.json``。
``operations/`` 目录只是 Pi 候选生成过程的审计记录，其中嵌入的评测结果只是
summary.json 当时的一份快照，不作为统计输入，避免重复计数。

每个 summary.json 的 ``results`` 数组里，每一条记录就是一局真实对局的落地结果，
包含：

- ``role_nodes``: dict[role -> node_id]，本局 6 个角色分别用的候选节点
- ``winner``: "wolf" 或 "village"，最终获胜阵营
- ``complete``: 是否完整跑完（未完成的对局不计入统计）
- ``record_path``: 完整对局日志的绝对路径，作为全局去重的主键

同一局真实对局会被多份 summary.json 重复收录（因为一局里除 focus 节点外，
其余角色也用到了别的已保留节点，那些节点自己的 summary.json 里也会记这一局）。
因此必须按 ``record_path`` 做全局去重，否则同一局会被似然函数/胜率统计计入
多次，产生系统性偏差。

本脚本产出三类指标，写入 output/ 目录：

1. 全局胜率（global win rate）：节点出场过的全部对局（不管是否为 focus 节点）
   中，其所在阵营获胜的比例。
2. 调和胜率（对手/队友强度加权后的经验指数）：按用户指定公式
   ``global_win_rate * opponent_avg_win_rate / ally_avg_win_rate`` 计算，
   同时输出裁剪后的版本并在 ALGORITHM.md 中说明其数值稳定性问题。
3. 团队版 Bradley-Terry 实力值：把每局看作"狼人阵营 vs 好人阵营"的一次成对
   比较，用 MM（minorization-maximization，等价于 Zermelo 迭代）算法对全部
   节点的对数强度做最大似然估计，同时给出基于 Fisher 信息矩阵的近似标准误。

用法
----
    python3 compute_ratings.py

不需要参数，路径按仓库固定结构解析。运行时间：全量约 2000+ 局，纯 Python +
numpy 实现，预期数十秒内完成。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
EVOLUTION_ROOT = REPO_ROOT / "werewolf_game" / "records" / "season2" / "evolution"
ARCHIVE_MANIFEST = EVOLUTION_ROOT / "archive" / "manifest.json"
EVALUATIONS_ROOT = EVOLUTION_ROOT / "evaluations"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"

ROLES = ("wolf", "villager", "seer", "witch", "guard", "hunter")
# 12 人局固定配置：4 狼、1 女巫、1 守卫、1 猎人、1 预言家、4 村民。
# 用于团队 Bradley-Terry 里给每个角色节点分配"出场座位数"权重。
ROLE_SEAT_COUNT = {
    "wolf": 4,
    "villager": 4,
    "seer": 1,
    "witch": 1,
    "guard": 1,
    "hunter": 1,
}
WOLF_TEAM_ROLES = ("wolf",)
VILLAGE_TEAM_ROLES = ("villager", "seer", "witch", "guard", "hunter")


@dataclass
class NodeMeta:
    node_id: str
    role: str
    status: str
    parent_id: str | None
    depth: int
    created_at: str
    status_reason: str
    evaluation_summary_path: str | None
    # summary.json 顶层的 win_rate：节点作为 focus 节点被单独评测时的胜率
    # （Season 2 原本用来做筛选的"测试胜率"，只统计该节点出场并且是 focus
    # 的那一批对局，通常是 20 局左右）。与下面计算的"全局胜率"不同——全局
    # 胜率会把该节点作为其他节点评测时的队友/对手出场的对局也计入。
    test_win_rate: float | None = None
    test_game_count: int | None = None


@dataclass
class GameRecord:
    """一局去重后的真实对局记录。"""

    record_path: str
    role_nodes: dict[str, str]
    winner: str  # "wolf" | "village"


@dataclass
class NodeStats:
    node_id: str
    role: str
    status: str
    games_played: int = 0
    games_won: int = 0
    # 出场时，己方阵营队友（同阵营其他角色节点）的全局胜率列表，用于调和胜率。
    ally_win_rates: list[float] = field(default_factory=list)
    opponent_win_rates: list[float] = field(default_factory=list)


def load_manifest() -> dict[str, NodeMeta]:
    """读取 archive/manifest.json，返回 node_id -> NodeMeta。"""

    raw = json.loads(ARCHIVE_MANIFEST.read_text(encoding="utf-8"))
    nodes: dict[str, NodeMeta] = {}
    for node_id, value in raw.get("nodes", {}).items():
        nodes[node_id] = NodeMeta(
            node_id=node_id,
            role=str(value["role"]),
            status=str(value["status"]),
            parent_id=(str(value["parent_id"]) if value.get("parent_id") else None),
            depth=int(value["depth"]),
            created_at=str(value["created_at"]),
            status_reason=str(value.get("status_reason", "")),
            evaluation_summary_path=(
                str(value["evaluation_summary_path"])
                if value.get("evaluation_summary_path")
                else None
            ),
        )
    return nodes


def load_games(nodes: dict[str, NodeMeta]) -> list[GameRecord]:
    """遍历全部节点的 summary.json，按 record_path 全局去重后返回对局列表。

    顺带把每个节点 summary.json 顶层的 win_rate/game_count（Season 2 原本
    用来筛选候选的"测试胜率"）写回 NodeMeta，避免重复打开同一批文件。
    """

    seen: dict[str, GameRecord] = {}
    missing_summary: list[str] = []
    for node_id, meta in nodes.items():
        if meta.evaluation_summary_path is None:
            # pending 或在 smoke test 阶段即被淘汰的节点，从未进入完整评测。
            continue
        summary_path = Path(meta.evaluation_summary_path)
        if not summary_path.exists():
            missing_summary.append(node_id)
            continue
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            missing_summary.append(node_id)
            continue
        meta.test_win_rate = (
            float(summary["win_rate"]) if summary.get("win_rate") is not None else None
        )
        meta.test_game_count = (
            int(summary["game_count"]) if summary.get("game_count") is not None else None
        )
        for result in summary.get("results", []):
            if not result.get("complete"):
                continue
            winner = result.get("winner")
            if winner not in ("wolf", "village"):
                continue
            record_path = result.get("record_path")
            role_nodes = result.get("role_nodes")
            if not record_path or not isinstance(role_nodes, dict):
                continue
            if record_path in seen:
                continue
            if set(role_nodes) != set(ROLES):
                # 角色集合不完整的记录跳过，不参与统计（未观察到此类情况，
                # 但保留防御性检查以保证下游模型输入干净）。
                continue
            seen[record_path] = GameRecord(
                record_path=record_path,
                role_nodes={role: str(role_nodes[role]) for role in ROLES},
                winner=str(winner),
            )
    if missing_summary:
        print(
            f"[警告] {len(missing_summary)} 个节点声明了 evaluation_summary_path 但文件缺失/损坏，"
            f"已跳过：{missing_summary[:10]}{'...' if len(missing_summary) > 10 else ''}"
        )
    return list(seen.values())


def compute_global_win_rates(
    nodes: dict[str, NodeMeta], games: list[GameRecord]
) -> dict[str, NodeStats]:
    """第一步：只统计出场次数和获胜次数，得到全局胜率。"""

    stats: dict[str, NodeStats] = {
        node_id: NodeStats(node_id=node_id, role=meta.role, status=meta.status)
        for node_id, meta in nodes.items()
    }
    for game in games:
        wolf_won = game.winner == "wolf"
        for role, node_id in game.role_nodes.items():
            entry = stats.get(node_id)
            if entry is None:
                # 对局引用了 manifest 里没有的 node_id（不应发生，防御性处理）。
                entry = NodeStats(node_id=node_id, role=role, status="unknown")
                stats[node_id] = entry
            entry.games_played += 1
            node_won = wolf_won if role in WOLF_TEAM_ROLES else not wolf_won
            if node_won:
                entry.games_won += 1
    return stats


def global_win_rate(stats: NodeStats) -> float | None:
    if stats.games_played == 0:
        return None
    return stats.games_won / stats.games_played


def compute_harmonic_index(
    stats: dict[str, NodeStats], games: list[GameRecord]
) -> dict[str, dict[str, float | None]]:
    """第二步：基于全局胜率，计算队友/对手平均胜率与调和胜率指数。

    调和胜率定义（用户指定公式）：
        harmonic = global_win_rate * opponent_avg_win_rate / ally_avg_win_rate

    数值稳定性问题（详见 ALGORITHM.md 第 3 节）：
    - 该指数不是概率，取值可以任意大，不能解释为"胜率"。
    - ally_avg_win_rate 接近 0 时会发散，因此额外提供一个下限裁剪版本
      （分母裁剪到 >= 0.05）用于排序展示，避免极端值主导榜单。
    """

    global_rate: dict[str, float | None] = {
        node_id: global_win_rate(entry) for node_id, entry in stats.items()
    }

    ally_sum: dict[str, float] = {node_id: 0.0 for node_id in stats}
    ally_count: dict[str, int] = {node_id: 0 for node_id in stats}
    opp_sum: dict[str, float] = {node_id: 0.0 for node_id in stats}
    opp_count: dict[str, int] = {node_id: 0 for node_id in stats}

    # 注意：狼人阵营在本数据模型里只有 1 个角色键（wolf），4 个狼人座位共享
    # 同一个 node_id，因此狼人节点没有"异质队友节点"——它的队友就是它自己的
    # 3 个复制品，无法构成有意义的"队友强度"样本。这里把狼人节点的队友集合
    # 显式留空（ally_avg_win_rate 将为 None），而不是错误地把自身也算进队友，
    # 这一局限会在 ALGORITHM.md 中说明，调和胜率对狼人节点整体不可用。
    for game in games:
        wolf_nodes = [game.role_nodes[role] for role in WOLF_TEAM_ROLES]
        village_nodes = [game.role_nodes[role] for role in VILLAGE_TEAM_ROLES]
        for node_id in wolf_nodes:
            opponents = village_nodes
            _accumulate(node_id, [], opponents, global_rate, ally_sum, ally_count, opp_sum, opp_count)
        for node_id in village_nodes:
            allies = [n for n in village_nodes if n != node_id]
            opponents = wolf_nodes
            _accumulate(node_id, allies, opponents, global_rate, ally_sum, ally_count, opp_sum, opp_count)

    result: dict[str, dict[str, float | None]] = {}
    for node_id, entry in stats.items():
        g = global_rate[node_id]
        ally_avg = (ally_sum[node_id] / ally_count[node_id]) if ally_count[node_id] else None
        opp_avg = (opp_sum[node_id] / opp_count[node_id]) if opp_count[node_id] else None
        harmonic_raw: float | None = None
        harmonic_clipped: float | None = None
        if g is not None and ally_avg is not None and opp_avg is not None:
            if ally_avg > 0:
                harmonic_raw = g * opp_avg / ally_avg
            clipped_denominator = max(ally_avg, 0.05)
            harmonic_clipped = g * opp_avg / clipped_denominator
        result[node_id] = {
            "global_win_rate": g,
            "ally_avg_win_rate": ally_avg,
            "opponent_avg_win_rate": opp_avg,
            "harmonic_index_raw": harmonic_raw,
            "harmonic_index_clipped": harmonic_clipped,
        }
    return result


def _accumulate(
    node_id: str,
    allies: list[str],
    opponents: list[str],
    global_rate: dict[str, float | None],
    ally_sum: dict[str, float],
    ally_count: dict[str, int],
    opp_sum: dict[str, float],
    opp_count: dict[str, int],
) -> None:
    for ally in allies:
        rate = global_rate.get(ally)
        if rate is not None:
            ally_sum[node_id] += rate
            ally_count[node_id] += 1
    for opponent in opponents:
        rate = global_rate.get(opponent)
        if rate is not None:
            opp_sum[node_id] += rate
            opp_count[node_id] += 1


def fit_team_bradley_terry_legacy(
    nodes: dict[str, NodeMeta],
    games: list[GameRecord],
    *,
    max_iterations: int = 200,
    tolerance: float = 1e-9,
) -> dict[str, dict[str, float]]:
    """团队版 Bradley-Terry：MM 算法（等价于 Zermelo 迭代）求解节点强度。

    **保留状态：已知此模型在本场景下退化失效，仅用于与新模型对比，不应作为
    结论依据。** 退化的实证见 ALGORITHM.md 第 2.6 节：`total = T_wolf +
    T_village` 是 12 个座位强度的加总，变异系数只有 0.108（近似常数），使
    MM 更新式 `pi_new = Σ(赢的局 seats) / Σ(所有局 seats/total)` 退化为
    `pi ≈ total × 胜率`；实测 `bt_strength / global_win_rate` 比值均值 3.61、
    变异系数仅 0.062，即输出基本只是全局胜率的线性缩放，没有起到"剥离队友/
    对手强度混杂"的作用。这是模型选型问题，不是参数或实现细节问题。

    模型设定：
    - 每个节点 i 有正实力值 pi_i（对数尺度 beta_i = ln pi_i）。
    - 一局比赛里，狼人阵营的"团队实力"定义为
          T_wolf = sum(seat_count[role] * pi_{role_node}) for role in wolf 阵营
      好人阵营同理。狼人 4 个座位共享同一个候选节点，因此该节点的强度按座位数
      加权 4 次计入团队实力。
    - P(狼人阵营胜) = T_wolf / (T_wolf + T_village)，这是 Bradley-Terry 从
      两人对战到两支加权团队对战的标准扩展（Huang, Weng & Lin 2006）。

    求解算法：
    - 对 beta 做 MM 更新（Hunter 2004, "MM algorithms for generalized
      Bradley-Terry models"）。每轮迭代对每个节点 i：
          numerator_i   = 该节点参与并获胜的"座位次数"之和
          denominator_i = 对每一局包含该节点的比赛，seat_count_i / T_team(该局)
                          按座位数加权求和
          pi_i_new = numerator_i / denominator_i
      然后归一化（除以活跃节点的几何均值）避免整体漂移。
    - 迭代到 beta 变化的最大绝对值小于 tolerance 或达到最大迭代次数。
    """

    node_ids = sorted(nodes)
    index = {node_id: i for i, node_id in enumerate(node_ids)}
    n = len(node_ids)
    pi = np.ones(n, dtype=float)

    # 预处理每局的座位构成，避免重复解析。
    parsed_games = []
    for game in games:
        wolf_seats = [
            (index[game.role_nodes[role]], ROLE_SEAT_COUNT[role]) for role in WOLF_TEAM_ROLES
        ]
        village_seats = [
            (index[game.role_nodes[role]], ROLE_SEAT_COUNT[role]) for role in VILLAGE_TEAM_ROLES
        ]
        wolf_won = game.winner == "wolf"
        parsed_games.append((wolf_seats, village_seats, wolf_won))

    if n == 0 or not parsed_games:
        return {}

    for _iteration in range(max_iterations):
        numerator = np.zeros(n, dtype=float)
        denominator = np.zeros(n, dtype=float)
        for wolf_seats, village_seats, wolf_won in parsed_games:
            t_wolf = sum(pi[idx] * seats for idx, seats in wolf_seats)
            t_village = sum(pi[idx] * seats for idx, seats in village_seats)
            total = t_wolf + t_village
            if total <= 0:
                continue
            for idx, seats in wolf_seats:
                if wolf_won:
                    numerator[idx] += seats
                denominator[idx] += seats / total
            for idx, seats in village_seats:
                if not wolf_won:
                    numerator[idx] += seats
                denominator[idx] += seats / total

        new_pi = pi.copy()
        active = denominator > 0
        new_pi[active] = numerator[active] / denominator[active]
        # 从未出场的节点（denominator=0）强度保持不变（仍为初始值 1.0）。
        new_pi[~active] = pi[~active]

        # 归一化：只用有出场记录（active）的节点计算几何均值。若把从未出场、
        # 始终停留在初始值 1.0 的"死节点"也纳入几何均值，会把归一化基准拖向
        # 1.0，导致每轮迭代对活跃节点的尺度矫正不足，几十轮迭代后误差累积到
        # 数量级失控（曾实测放大至 1e31）。
        active_positive = new_pi[active & (new_pi > 0)]
        if len(active_positive) > 0:
            log_mean = np.mean(np.log(active_positive))
            new_pi = new_pi / math.exp(log_mean)

        beta_old = np.log(np.clip(pi, 1e-12, None))
        beta_new = np.log(np.clip(new_pi, 1e-12, None))
        max_delta = float(np.max(np.abs(beta_new - beta_old)))
        pi = new_pi
        if max_delta < tolerance:
            break

    beta = np.log(np.clip(pi, 1e-12, None))

    # Fisher 信息矩阵对角近似，用于给出每个节点 beta 的标准误。
    #
    # 正确推导（详见 ALGORITHM.md 第 2.4 节，含蒙特卡洛验证记录）：
    #     dP_wolf/dbeta_i = seats_i * pi_i * T_village / total^2
    # 单次对局的 Fisher 信息贡献是 (dP_wolf/dbeta_i)^2 / [P_wolf * (1-P_wolf)]。
    #
    # 早期实现曾错误地把这一项近似成 seats_i^2 * P_wolf * (1-P_wolf)，忽略了
    # T_village/total^2 这个团队叠加耦合项，对 seats=4 的角色系统性低估标准误
    # 约 4 倍。蒙特卡洛验证：真实抽样标准误 0.198，错误公式 0.048，本公式 0.193。
    fisher_diag = np.zeros(n, dtype=float)
    for wolf_seats, village_seats, wolf_won in parsed_games:
        del wolf_won  # Fisher 信息用的是模型隐含方差，不依赖本局实际胜负结果。
        t_wolf = sum(pi[idx] * seats for idx, seats in wolf_seats)
        t_village = sum(pi[idx] * seats for idx, seats in village_seats)
        total = t_wolf + t_village
        if total <= 0:
            continue
        p_wolf = t_wolf / total
        p_village = 1.0 - p_wolf
        variance_term = p_wolf * p_village
        if variance_term <= 0:
            continue
        total_sq = total * total
        for idx, seats in wolf_seats:
            grad = seats * pi[idx] * t_village / total_sq
            fisher_diag[idx] += (grad**2) / variance_term
        for idx, seats in village_seats:
            grad = seats * pi[idx] * t_wolf / total_sq
            fisher_diag[idx] += (grad**2) / variance_term

    with np.errstate(divide="ignore"):
        std_error = np.where(
            fisher_diag > 0, 1.0 / np.sqrt(np.clip(fisher_diag, 1e-12, None)), np.inf
        )

    games_played = np.zeros(n, dtype=float)
    for wolf_seats, village_seats, _wolf_won in parsed_games:
        for idx, _seats in wolf_seats + village_seats:
            games_played[idx] += 1

    result: dict[str, dict[str, float]] = {}
    for node_id, idx in index.items():
        result[node_id] = {
            "strength": float(pi[idx]),
            "log_strength": float(beta[idx]),
            "log_strength_std_error": float(std_error[idx]),
            "games_played": float(games_played[idx]),
        }
    return result


def fit_logistic_team_model(
    nodes: dict[str, NodeMeta],
    games: list[GameRecord],
    *,
    max_iterations: int = 200,
    tolerance: float = 1e-9,
    ridge_lambda: float = 1.0,
) -> dict[str, dict[str, float]]:
    """Logistic 团队模型（带 L2 岭惩罚的逻辑回归）求解节点强度。

    为什么不用"强度直接相加"的团队 Bradley-Terry
    --------------------------------------------
    见 `fit_team_bradley_terry_legacy` 的 docstring 和 ALGORITHM.md 第 2.6
    节：那个模型的 `P = T_A/(T_A+T_B)` 形式要求两队强度有可比的量级差异才
    能产生信息，但 12 人局的团队强度是 6 个节点求和，对单节点变化极不敏感，
    实测退化为"胜率 × 常数"。

    当前模型设定
    ------------
    改用对数尺度线性叠加（Logistic 团队模型）：

        logit P(狼胜 | 局 g) = beta_0 + Σ_{i∈狼队} w_i*beta_i
                                      - Σ_{j∈好人队} w_j*beta_j

    - `beta_i` 是节点 i 的对数强度，`w_i` 是该节点在本局的座位数
      （狼/民 4，预言家/女巫/守卫/猎人各 1）。
    - `beta_0` 是阵营基线截距，专门吸收"狼人阵营天然占优"（本赛季狼人
      胜率 74.6%）这部分系统性偏差，避免它污染各节点的强度估计。
    - 关键区别：单个节点的 `beta_i` 变化会**直接平移决策边界**，不会像
      "强度求和再取比值"那样被团队总量稀释，因此模型对单节点差异敏感。

    这本质上是以"节点出场情况"为特征、以"狼队是否获胜"为标签的逻辑回归，
    每局一个样本，设计矩阵每行在狼队节点位置取 +w、好人队节点位置取 -w。

    求解算法
    --------
    带 L2 岭惩罚的牛顿法（IRLS）。惩罚项有两个必要作用：
    1. 保证可识别性——无惩罚时整体平移 beta 不改变任何预测（所有节点同时
       +c 会被差值抵消），Hessian 奇异；岭惩罚把解锚定到最小范数解。
    2. 收缩小样本节点——出场很少的节点若无约束，beta 会被少数对局结果推向
       极端值（完全分离时趋于无穷）。

    `ridge_lambda=1.0` 是相对温和的惩罚（对照 2358 局样本量），在"允许
    节点间产生可辨识差异"和"抑制小样本发散"之间取平衡。

    标准误由带惩罚的 Hessian 的对角逆元素给出（见 2.4 节），是忽略参数间
    协方差的近似，足以做量级判断。
    """

    node_ids = sorted(nodes)
    index = {node_id: i for i, node_id in enumerate(node_ids)}
    n = len(node_ids)
    if n == 0 or not games:
        return {}

    # 构造设计矩阵：每局一行，狼队节点位置 +seats，好人队节点位置 -seats，
    # 最后一列恒为 1（阵营基线截距 beta_0）。标签 y = 狼队是否获胜。
    # 参数向量长度 n+1，最后一个分量是截距，不参与岭惩罚（截距代表阵营基线，
    # 不应该被收缩向 0）。
    design = np.zeros((len(games), n + 1), dtype=float)
    labels = np.zeros(len(games), dtype=float)
    games_played = np.zeros(n, dtype=float)
    for row, game in enumerate(games):
        for role in WOLF_TEAM_ROLES:
            idx = index[game.role_nodes[role]]
            design[row, idx] += ROLE_SEAT_COUNT[role]
            games_played[idx] += 1
        for role in VILLAGE_TEAM_ROLES:
            idx = index[game.role_nodes[role]]
            design[row, idx] -= ROLE_SEAT_COUNT[role]
            games_played[idx] += 1
        design[row, n] = 1.0
        labels[row] = 1.0 if game.winner == "wolf" else 0.0

    # L2 岭惩罚矩阵：只惩罚节点系数，不惩罚截距。
    penalty = np.full(n + 1, float(ridge_lambda))
    penalty[n] = 0.0
    penalty_matrix = np.diag(penalty)

    # 牛顿法 / IRLS 求解带惩罚的逻辑回归。
    theta = np.zeros(n + 1, dtype=float)
    for _iteration in range(max_iterations):
        logits = design @ theta
        # 数值稳定的 sigmoid。
        probabilities = np.where(
            logits >= 0,
            1.0 / (1.0 + np.exp(-np.clip(logits, -500, 500))),
            np.exp(np.clip(logits, -500, 500)) / (1.0 + np.exp(np.clip(logits, -500, 500))),
        )
        weights = np.clip(probabilities * (1.0 - probabilities), 1e-10, None)
        gradient = design.T @ (labels - probabilities) - penalty * theta
        hessian = (design.T * weights) @ design + penalty_matrix
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
        theta = theta + step
        if float(np.max(np.abs(step))) < tolerance:
            break

    beta = theta[:n]
    intercept = float(theta[n])

    # 标准误：带惩罚 Hessian 的逆的对角元素平方根。这是忽略参数间协方差的
    # 边际标准误近似；岭惩罚使 Hessian 非奇异，因此可以直接求逆。
    logits = design @ theta
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -500, 500)))
    weights = np.clip(probabilities * (1.0 - probabilities), 1e-10, None)
    hessian = (design.T * weights) @ design + penalty_matrix
    try:
        covariance = np.linalg.inv(hessian)
        variances = np.clip(np.diag(covariance)[:n], 0.0, None)
        std_error = np.sqrt(variances)
    except np.linalg.LinAlgError:
        std_error = np.full(n, np.inf)

    # 从未出场的节点没有任何观测，其 beta 完全由岭惩罚决定（恒为 0），
    # 标记为 inf 标准误以便下游识别为"无观测值"。
    std_error = np.where(games_played > 0, std_error, np.inf)

    result: dict[str, dict[str, float]] = {}
    for node_id, idx in index.items():
        result[node_id] = {
            # strength = exp(beta)，语义是"相对强度倍率"：exp(beta)=1 表示
            # 处于岭惩罚锚定的中性水平（比该水平强则 >1，弱则 <1）。
            "strength": float(math.exp(beta[idx])),
            "log_strength": float(beta[idx]),
            "log_strength_std_error": float(std_error[idx]),
            "games_played": float(games_played[idx]),
            "team_intercept": intercept,
        }
    return result


def compute_significance_vs_base(
    nodes: dict[str, NodeMeta], model: dict[str, dict[str, float]]
) -> dict[str, dict[str, float | None]]:
    """对每个节点，计算它与同角色 base 节点的强度差异是否统计显著。

    z = (beta_node - beta_base) / sqrt(se_node^2 + se_base^2)

    这是忽略参数间协方差的近似检验（详见 ALGORITHM.md 第 2.4/2.5 节），
    比严格的联合似然比检验宽松，但足够做"这个差异是否在噪声范围内"的
    量级判断。|z| > 2 大致对应 95% 置信水平下的显著差异。base 节点自身
    及从未出场的节点没有对照对象或标准误无穷大时，结果为 None。

    `model` 参数接受任一模型的输出（legacy Bradley-Terry 或 Logistic 团队
    模型），两者都使用统一的 `log_strength` / `log_strength_std_error`
    字段名，因此同一套检验逻辑可以复用。
    """

    base_by_role: dict[str, str] = {}
    for node_id, meta in nodes.items():
        if node_id.endswith("-base"):
            base_by_role[meta.role] = node_id

    result: dict[str, dict[str, float | None]] = {}
    for node_id, meta in nodes.items():
        base_id = base_by_role.get(meta.role)
        base_entry = model.get(base_id) if base_id else None
        node_entry = model.get(node_id)
        if (
            node_id == base_id
            or base_entry is None
            or node_entry is None
            or not math.isfinite(base_entry["log_strength_std_error"])
            or not math.isfinite(node_entry["log_strength_std_error"])
        ):
            result[node_id] = {
                "z_vs_base": None,
                "significant_vs_base": None,
            }
            continue
        diff = node_entry["log_strength"] - base_entry["log_strength"]
        se = math.sqrt(
            node_entry["log_strength_std_error"] ** 2
            + base_entry["log_strength_std_error"] ** 2
        )
        z = diff / se if se > 0 else float("inf")
        result[node_id] = {
            "z_vs_base": float(z),
            "significant_vs_base": bool(abs(z) > 2),
        }
    return result


MODEL_DESCRIPTIONS = {
    "bradley_terry_legacy": (
        "团队 Bradley-Terry（强度直接相加，P=T_A/(T_A+T_B)，MM 迭代求解）。"
        "【已知退化失效，仅保留用于对比】实测输出基本只是全局胜率的线性缩放"
        "（strength/global_win_rate 比值变异系数仅 0.062），因为 12 座位强度"
        "求和后 total 近似常数（变异系数 0.108），单节点差异被摊薄。"
    ),
    "logistic_team": (
        "Logistic 团队模型（对数尺度线性叠加，logit P = beta_0 + Σw·beta_狼"
        " - Σw·beta_好人，带 L2 岭惩罚的 IRLS 求解）。单节点 beta 变化直接"
        "平移决策边界，不被团队总量稀释；beta_0 吸收阵营基线偏差。"
    ),
}


def build_node_meta_table(
    nodes: dict[str, NodeMeta],
    stats: dict[str, NodeStats],
    harmonic: dict[str, dict[str, float | None]],
) -> list[dict[str, Any]]:
    """节点元信息 + 与模型无关的胜率指标（两个模型共用这张表）。"""

    rows: list[dict[str, Any]] = []
    for node_id, meta in sorted(nodes.items()):
        entry = stats.get(node_id, NodeStats(node_id=node_id, role=meta.role, status=meta.status))
        harmonic_entry = harmonic.get(node_id, {})
        rows.append(
            {
                "node_id": node_id,
                "role": meta.role,
                "status": meta.status,
                "parent_id": meta.parent_id,
                "depth": meta.depth,
                "created_at": meta.created_at,
                "status_reason": meta.status_reason,
                "test_win_rate": meta.test_win_rate,
                "test_game_count": meta.test_game_count,
                "games_played": entry.games_played,
                "games_won": entry.games_won,
                "global_win_rate": harmonic_entry.get("global_win_rate"),
                "ally_avg_win_rate": harmonic_entry.get("ally_avg_win_rate"),
                "opponent_avg_win_rate": harmonic_entry.get("opponent_avg_win_rate"),
                "harmonic_index_raw": harmonic_entry.get("harmonic_index_raw"),
                "harmonic_index_clipped": harmonic_entry.get("harmonic_index_clipped"),
            }
        )
    return rows


def build_model_table(
    nodes: dict[str, NodeMeta],
    model: dict[str, dict[str, float]],
    significance: dict[str, dict[str, float | None]],
    harmonic: dict[str, dict[str, float | None]],
) -> list[dict[str, Any]]:
    """单个模型的强度估计表（每个模型一张，结构相同便于直接对比）。

    表里同时冗余带上三种胜率（测试胜率/全局胜率/调和指数），这样看模型表
    就能直接判断"强度估计和胜率是否一致、差异在哪"，不必再回头跟 node_meta
    表按 node_id 对照。
    """

    rows: list[dict[str, Any]] = []
    for node_id, meta in sorted(nodes.items()):
        model_entry = model.get(node_id, {})
        sig_entry = significance.get(node_id, {})
        harmonic_entry = harmonic.get(node_id, {})
        rows.append(
            {
                "node_id": node_id,
                "role": meta.role,
                "status": meta.status,
                "test_win_rate": meta.test_win_rate,
                "games_played": model_entry.get("games_played"),
                "global_win_rate": harmonic_entry.get("global_win_rate"),
                "harmonic_index_clipped": harmonic_entry.get("harmonic_index_clipped"),
                "strength": model_entry.get("strength"),
                "log_strength": model_entry.get("log_strength"),
                "log_strength_std_error": model_entry.get("log_strength_std_error"),
                "z_vs_base": sig_entry.get("z_vs_base"),
                "significant_vs_base": sig_entry.get("significant_vs_base"),
            }
        )
    return rows


def _csv_lines(rows: list[dict[str, Any]], columns: list[str]) -> list[str]:
    """把一张表渲染成 CSV 行（首行是表头）。"""

    lines = [",".join(columns)]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column)
            if value is None:
                values.append("")
            elif isinstance(value, float):
                values.append("inf" if math.isinf(value) else f"{value:.6f}")
            else:
                text = str(value)
                if '"' in text:
                    text = text.replace('"', '""')
                if "," in text or '"' in text or "\n" in text:
                    text = f'"{text}"'
                values.append(text)
        lines.append(",".join(values))
    return lines


def write_outputs(
    node_meta_rows: list[dict[str, Any]],
    model_tables: dict[str, list[dict[str, Any]]],
    games: list[GameRecord],
    model_extras: dict[str, dict[str, Any]] | None = None,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    winner_counts = {"wolf": 0, "village": 0}
    for game in games:
        winner_counts[game.winner] += 1

    # 单一 JSON，内部分成若干张表：dataset / node_meta / models.<name>。
    # 两个模型各占一张独立的表，结构一致，便于直接横向对比同一节点在不同
    # 模型下的强度、标准误和显著性判定。
    document = {
        "dataset": {
            "unique_games": len(games),
            "winner_counts": winner_counts,
            "node_count": len(node_meta_rows),
        },
        "node_meta": node_meta_rows,
        "models": {
            name: {
                "description": MODEL_DESCRIPTIONS.get(name, ""),
                **(model_extras or {}).get(name, {}),
                "nodes": rows,
            }
            for name, rows in model_tables.items()
        },
    }

    json_path = OUTPUT_DIR / "node_ratings.json"
    json_path.write_text(
        json.dumps(document, indent=2, ensure_ascii=False, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    print(f"写入 {json_path}")

    meta_columns = [
        "node_id",
        "role",
        "status",
        "parent_id",
        "depth",
        "created_at",
        "status_reason",
        "test_win_rate",
        "test_game_count",
        "games_played",
        "games_won",
        "global_win_rate",
        "ally_avg_win_rate",
        "opponent_avg_win_rate",
        "harmonic_index_raw",
        "harmonic_index_clipped",
    ]
    model_columns = [
        "node_id",
        "role",
        "status",
        "test_win_rate",
        "games_played",
        "global_win_rate",
        "harmonic_index_clipped",
        "strength",
        "log_strength",
        "log_strength_std_error",
        "z_vs_base",
        "significant_vs_base",
    ]

    # 单个 CSV 文件，内部用 "# TABLE: <name>" 分隔多张表。每张表自带表头，
    # 表之间空一行。这样一个文件就能同时承载数据集概况、节点元信息和两个
    # 模型的强度估计，便于在一处对照，不需要在多个文件之间来回切换。
    sections: list[tuple[str, list[str]]] = []

    sections.append(
        (
            "dataset",
            _csv_lines(
                [
                    {
                        "unique_games": len(games),
                        "wolf_wins": winner_counts["wolf"],
                        "village_wins": winner_counts["village"],
                        "node_count": len(node_meta_rows),
                    }
                ],
                ["unique_games", "wolf_wins", "village_wins", "node_count"],
            ),
        )
    )

    sections.append(("node_meta", _csv_lines(node_meta_rows, meta_columns)))

    for name, rows in model_tables.items():
        extras = (model_extras or {}).get(name, {})
        comment_lines = [f"# {MODEL_DESCRIPTIONS.get(name, '')}"] if name in MODEL_DESCRIPTIONS else []
        for key, value in extras.items():
            comment_lines.append(f"# {key} = {value}")
        sections.append((f"model_{name}", comment_lines + _csv_lines(rows, model_columns)))

    blocks: list[str] = []
    for table_name, lines in sections:
        blocks.append(f"# TABLE: {table_name}\n" + "\n".join(lines))

    csv_path = OUTPUT_DIR / "node_ratings.csv"
    csv_path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
    print(f"写入 {csv_path}（包含 {len(sections)} 张表：" + "、".join(n for n, _ in sections) + "）")


def main() -> None:
    print(f"读取 manifest：{ARCHIVE_MANIFEST}")
    nodes = load_manifest()
    print(f"节点总数：{len(nodes)}")

    print("加载并去重全部历史对局……")
    games = load_games(nodes)
    print(f"去重后可用对局数：{len(games)}")
    if not games:
        raise SystemExit("没有可用的完整对局记录，无法计算统计量")

    print("计算全局胜率……")
    stats = compute_global_win_rates(nodes, games)

    print("计算调和胜率指数……")
    harmonic = compute_harmonic_index(stats, games)

    # 两个模型都跑，输出到独立的表中便于对比。legacy 模型已知退化失效，
    # 保留它是为了让"退化"这件事本身在数据上可复现、可对照。
    print("拟合模型 1/2：团队 Bradley-Terry（legacy，已知退化）……")
    bt_legacy = fit_team_bradley_terry_legacy(nodes, games)
    bt_significance = compute_significance_vs_base(nodes, bt_legacy)

    print("拟合模型 2/2：Logistic 团队模型……")
    logistic = fit_logistic_team_model(nodes, games)
    logistic_significance = compute_significance_vs_base(nodes, logistic)

    node_meta_rows = build_node_meta_table(nodes, stats, harmonic)
    model_tables = {
        "bradley_terry_legacy": build_model_table(
            nodes, bt_legacy, bt_significance, harmonic
        ),
        "logistic_team": build_model_table(
            nodes, logistic, logistic_significance, harmonic
        ),
    }
    intercept = next(
        (entry.get("team_intercept") for entry in logistic.values() if "team_intercept" in entry),
        None,
    )
    model_extras = {
        "logistic_team": {"team_intercept": intercept} if intercept is not None else {},
    }
    write_outputs(node_meta_rows, model_tables, games, model_extras)
    print("完成。")


if __name__ == "__main__":
    main()

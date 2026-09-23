#!/usr/bin/env python3
"""对比两个强度模型的输出，检验"是否退化为胜率线性缩放"。

判据：如果一个模型的 strength 只是 global_win_rate 的线性缩放，那么
`strength / global_win_rate` 这个比值在所有节点上会近似常数（变异系数很小），
说明模型没有提供超出原始胜率的信息，即没有真正剥离队友/对手强度的混杂。

用法：
    python3 compare_models.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
RATINGS_PATH = OUTPUT_DIR / "node_ratings.json"


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def stdev(values: list[float]) -> float:
    if len(values) < 2:
        return float("nan")
    mu = mean(values)
    return math.sqrt(sum((v - mu) ** 2 for v in values) / len(values))


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    mx, my = mean(xs), mean(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / len(xs)
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs) / len(xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys) / len(ys))
    return cov / (sx * sy) if sx > 0 and sy > 0 else float("nan")


def spearman(xs: list[float], ys: list[float]) -> float:
    def ranks(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        result = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            average_rank = (i + j) / 2.0
            for k in range(i, j + 1):
                result[order[k]] = average_rank
            i = j + 1
        return result

    return pearson(ranks(xs), ranks(ys))


def main() -> None:
    document = json.loads(RATINGS_PATH.read_text(encoding="utf-8"))
    node_meta = {row["node_id"]: row for row in document["node_meta"]}
    models = document["models"]

    print(f"数据集：{document['dataset']['unique_games']} 局，"
          f"{document['dataset']['node_count']} 个节点")
    print()

    per_model_strength: dict[str, dict[str, float]] = {}

    for name, block in models.items():
        rows = block["nodes"]
        played = [
            r
            for r in rows
            if r.get("games_played")
            and r.get("strength") is not None
            and node_meta[r["node_id"]].get("global_win_rate") is not None
        ]
        strengths = [r["strength"] for r in played]
        win_rates = [node_meta[r["node_id"]]["global_win_rate"] for r in played]
        ratios = [s / w for s, w in zip(strengths, win_rates) if w > 0]

        per_model_strength[name] = {r["node_id"]: r["strength"] for r in played}

        print(f"=== {name} ===")
        print(f"  有观测节点数: {len(played)}")
        print(f"  strength 范围: {min(strengths):.4f} ~ {max(strengths):.4f}")
        print(f"  strength/global_win_rate: mean={mean(ratios):.4f} "
              f"CV={stdev(ratios)/mean(ratios):.4f}")
        print(f"  与 global_win_rate 的 Pearson r = {pearson(strengths, win_rates):.4f}")
        print(f"  与 global_win_rate 的 Spearman rho = {spearman(strengths, win_rates):.4f}")
        sig = [r for r in rows if r.get("significant_vs_base") is True]
        checked = [r for r in rows if r.get("z_vs_base") is not None]
        print(f"  相对 base 显著的节点: {len(sig)} / {len(checked)} 个可检验节点")
        if checked:
            top = sorted(checked, key=lambda r: -abs(r["z_vs_base"]))[:5]
            print("  |z| 最大的 5 个:")
            for r in top:
                print(f"    {r['node_id']:20s} z={r['z_vs_base']:7.3f} "
                      f"strength={r['strength']:.4f} "
                      f"se={r['log_strength_std_error']:.4f}")
        if "team_intercept" in block:
            print(f"  阵营基线截距 beta_0 = {block['team_intercept']:.4f}")
        print()

    names = list(per_model_strength)
    if len(names) == 2:
        a, b = names
        shared = sorted(set(per_model_strength[a]) & set(per_model_strength[b]))
        xs = [per_model_strength[a][nid] for nid in shared]
        ys = [per_model_strength[b][nid] for nid in shared]
        print(f"=== 两模型对比（{len(shared)} 个共同节点）===")
        print(f"  strength Pearson r   = {pearson(xs, ys):.4f}")
        print(f"  strength Spearman rho = {spearman(xs, ys):.4f}")
        print("  （Spearman 接近 1 说明两模型给出的排序几乎相同；"
              "若两者都高度相关于胜率，则新模型也未提供额外信息）")


if __name__ == "__main__":
    main()

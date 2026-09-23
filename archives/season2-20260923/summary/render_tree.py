#!/usr/bin/env python3
"""按进化树父子关系渲染每个角色的节点树，标注各项胜率与强度估计。

展示的指标及其口径（容易混淆，这里统一说明）：

- 测试胜率（test_win_rate）：节点在 Season 2 评测流水线里作为 focus 节点被
  单独评测出来的胜率，来自 summary.json 顶层的 win_rate 字段，通常基于
  20 局左右。这是 Season 2 原本用来做保留/淘汰筛选的口径，只看该节点自己
  出战的这一批对局。
- 全局胜率（global_win_rate）：该节点在全部历史对局中出场过的每一局
  （不管是作为 focus 节点，还是被抽中给别的节点评测当队友/对手）都计入，
  更接近这个节点的"真实出场胜率"，样本量通常比测试胜率大。
- 调和胜率（harmonic_index_clipped）：在全局胜率基础上，按"队友越弱、
  对手越强则应上调"的经验公式做调整（global * opponent_avg / max(ally_avg,
  0.05)），数值不是概率，只是一个相对强度指数，狼人节点因为没有异质队友
  概念而始终为空（详见 ALGORITHM.md 第 3 节）。
- 强度（strength）与 z 值：来自 Logistic 团队模型（推荐口径，详见
  ALGORITHM.md 第 2.7 节）。strength=exp(beta)，1.0 表示处于岭惩罚锚定的
  中性水平；z 是该节点相对同角色 base 的差异显著性，`|z|>2` 才算统计显著，
  标 `*` 表示显著。注意：不要跨角色比较 strength 绝对值。
  另一个模型（团队 Bradley-Terry）已知退化失效，仅在 output/ 中保留用于
  对比，不在此展示——用 `python3 compare_models.py` 查看两模型对照。

用法：
    python3 render_tree.py            # 渲染全部 6 个角色
    python3 render_tree.py wolf seer  # 只渲染指定角色
    python3 render_tree.py --model bradley_terry_legacy wolf   # 指定模型
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
RATINGS_PATH = OUTPUT_DIR / "node_ratings.json"
ROLES_ORDER = ("wolf", "villager", "seer", "witch", "guard", "hunter")


DEFAULT_MODEL = "logistic_team"


def load_rows(model_name: str) -> list[dict[str, Any]]:
    """读取分区 JSON，把 node_meta 表与指定模型的表按 node_id 合并。"""

    if not RATINGS_PATH.exists():
        raise SystemExit(f"找不到 {RATINGS_PATH}，请先运行 compute_ratings.py")
    document = json.loads(RATINGS_PATH.read_text(encoding="utf-8"))
    models = document.get("models", {})
    if model_name not in models:
        available = "、".join(models) or "（无）"
        raise SystemExit(f"模型 {model_name} 不存在；可用模型：{available}")
    model_rows = {row["node_id"]: row for row in models[model_name]["nodes"]}
    merged: list[dict[str, Any]] = []
    for meta_row in document["node_meta"]:
        row = dict(meta_row)
        model_row = model_rows.get(meta_row["node_id"], {})
        row["strength"] = model_row.get("strength")
        row["z_vs_base"] = model_row.get("z_vs_base")
        row["significant_vs_base"] = model_row.get("significant_vs_base")
        merged.append(row)
    return merged


def fmt_rate(value: float | None) -> str:
    if value is None:
        return "  -  "
    return f"{value:5.1%}" if abs(value) < 10 else f"{value:5.2f}"


def fmt_index(value: float | None) -> str:
    if value is None:
        return "  -  "
    return f"{value:5.2f}"


def fmt_strength(value: float | None, played: float | None) -> str:
    # 从未出场的节点 strength 恒为 1.0（纯由岭惩罚决定），不是真实估计值，
    # 显示为 "-" 以免误读。
    if value is None or not played:
        return "  -  "
    return f"{value:5.2f}"


def fmt_z(z: float | None, significant: bool | None) -> str:
    if z is None:
        return "   -  "
    marker = "*" if significant else " "
    return f"{z:+5.2f}{marker}"


def render_role_tree(role: str, rows: list[dict[str, Any]]) -> None:
    role_rows = [r for r in rows if r["role"] == role]
    if not role_rows:
        print(f"\n=== {role}：无节点 ===")
        return
    by_id = {r["node_id"]: r for r in role_rows}
    children: dict[str | None, list[str]] = {}
    for r in role_rows:
        children.setdefault(r["parent_id"], []).append(r["node_id"])
    for parent_id in children:
        children[parent_id].sort(key=lambda node_id: by_id[node_id]["created_at"])

    total = len(role_rows)
    retained = sum(1 for r in role_rows if r["status"] == "retained")
    discarded = sum(1 for r in role_rows if r["status"] == "discarded")
    pending = sum(1 for r in role_rows if r["status"] == "pending")
    print(f"\n=== {role}（共 {total} 个节点：retained {retained} / discarded {discarded} / pending {pending}）===")
    header = (
        f"{'node_id':<40} {'status':<8} {'测试胜率':>8} {'局数':>5} "
        f"{'全局胜率':>8} {'出场':>5} {'调和':>6} {'强度':>6} {'z vs base':>10}"
    )
    print(header)
    print("-" * len(header))

    def walk(node_id: str, prefix: str, is_last: bool) -> None:
        row = by_id[node_id]
        connector = "└─ " if is_last else "├─ "
        branch = prefix + connector
        status = row["status"]
        marker = {"retained": "✓", "discarded": "✗", "pending": "…"}.get(status, "?")
        label = f"{branch}{node_id} [{marker}]"
        test_rate = fmt_rate(row.get("test_win_rate"))
        test_n = row.get("test_game_count")
        test_n_str = f"{test_n:>5d}" if test_n is not None else "  -  "
        global_rate = fmt_rate(row.get("global_win_rate"))
        played = row.get("games_played") or 0
        played_str = f"{int(played):>5d}" if played else "    0"
        harmonic = fmt_index(row.get("harmonic_index_clipped"))
        strength = fmt_strength(row.get("strength"), played)
        z_text = fmt_z(row.get("z_vs_base"), row.get("significant_vs_base"))
        print(
            f"{label:<40} {status:<8} {test_rate:>8} {test_n_str:>5} "
            f"{global_rate:>8} {played_str:>5} {harmonic:>6} {strength:>6} {z_text:>10}"
        )

        child_prefix = prefix + ("   " if is_last else "│  ")
        kids = children.get(node_id, [])
        for i, child_id in enumerate(kids):
            walk(child_id, child_prefix, i == len(kids) - 1)

    roots = children.get(None, [])
    for i, root_id in enumerate(roots):
        walk(root_id, "", i == len(roots) - 1)


def main() -> None:
    arguments = sys.argv[1:]
    model_name = DEFAULT_MODEL
    if "--model" in arguments:
        position = arguments.index("--model")
        if position + 1 >= len(arguments):
            raise SystemExit("--model 需要一个模型名参数")
        model_name = arguments[position + 1]
        arguments = arguments[:position] + arguments[position + 2 :]

    rows = load_rows(model_name)
    requested = arguments or list(ROLES_ORDER)
    print(f"强度与 z 值来自模型：{model_name}"
          + ("（推荐口径）" if model_name == DEFAULT_MODEL else "（对比用，注意其局限）"))
    for role in requested:
        if role not in ROLES_ORDER:
            print(f"[跳过] 未知角色：{role}")
            continue
        render_role_tree(role, rows)
    print("\n注：z 值后的 * 表示 |z|>2（统计显著）；强度仅在同角色内部可比。")


if __name__ == "__main__":
    main()

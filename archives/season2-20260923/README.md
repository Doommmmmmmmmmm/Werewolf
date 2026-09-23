# Season 2 归档：Task-Agent 自进化

本目录是 Season 2 的完整冻结产物。实验已结束，**只读，不再继续运行**。

## 内容

```text
archives/season2-20260923/
├── code/        Meta-Agent、进化树 archive、评测器、Pi 调用与校验
├── configs/     season2.json（冻结的实验配置）
├── examples/    run_season2_evolution.py（进化入口）
├── records/     实验产物（不进 git，只在磁盘）
│   ├── evolution/archive/       154 个节点的元数据与代码快照
│   ├── evolution/evaluations/   2358 局评测记录与逐局结果
│   └── evolution/operations/    111 次进化步骤的诊断、Pi 调用与结果
├── runtime/pi/  Pi coding agent（修改候选代码的工具）
├── skills/      30 张外部战术卡
├── summary/     两套强度模型的拟合、排名与算法说明
└── tests/       18 个针对进化框架的测试
```

## 实验结果

- **154 个节点**：111 retained、39 discarded、4 pending；最大深度 5
- **2358 局**去重对局，累计约 **1.87B token**
- 狼人阵营胜率 1758/2358；单角色焦点胜率：狼人中位 0.75，好人各角色 0.22–0.30

### 各角色最强节点

按 `summary/` 的 Logistic 团队模型强度：

| 角色 | 节点 | 强度 | 按评测胜率则是 |
| --- | --- | --- | --- |
| wolf | `wolf-n00135` | 1.162 | `wolf-n00004` (0.95) |
| villager | `villager-n00029` | 1.151 | `villager-n00046` (0.45) |
| seer | `seer-n00112` | 1.758 | `seer-n00131` (0.45) |
| witch | `witch-n00096` | 1.520 | `witch-n00095` (0.50) |
| guard | `guard-n00034` | 1.375 | `guard-n00016` (0.45) |
| hunter | `hunter-n00144` | 1.585 | `hunter-n00008` (0.35) |

强度值**只能在同角色内部比较**：模型把"狼人阵营天然占优"抽成了阵营基线截距
`beta_0 = 1.475`，不再混进节点强度。

**重要前提**：两套模型下都没有任何候选相对 base 达到统计显著（全部 `|z| < 1.5`，
95% 置信要求 `|z| > 2`；114 个可检验节点最大 `|z| = 1.71`）。所以上表只是点估计
排序，统计上还不能排除"候选与 base 真实水平相同"。

`summary/README.md` 和 `ALGORITHM.md` 有完整的算法说明与退化诊断。

## 怎么读归档节点

**归档后节点仍然可读，但不能再用 `EvolutionArchive` 去读。**

原因：`records/evolution/archive/manifest.json` 里冻结的是**绝对路径**（指向归档
前的 `werewolf_game/records/...`）。`EvolutionArchive` 打开时会比对当前配置与
manifest 里的冻结副本，路径不一致就抛「外置配置与已有 archive 的冻结配置不一致」。
该校验用于防止不同实验条件静默混用，挡的是"继续拿它跑进化"，不挡读数据。

读取请用游戏侧的只读仓库 `game/web/node_repo.py`：

```python
from game.web.roster import default_repository, best_role_nodes, load_ratings

repo = default_repository()                   # 自动发现 archives/ 下最新归档
print(repo.retained_nodes("seer"))            # 该角色所有 retained 节点
print(repo.task_agent_class("seer-n00112"))   # 按目录加载节点代码
print(repo.evaluation_summary("seer-n00112")) # 该节点的 20 局评测汇总
print(repo.rating_entries("logistic_team"))   # 强度表

# 按四种口径挑选各角色最优节点
roles = ["wolf", "villager", "seer", "witch", "guard", "hunter"]
print(best_role_nodes(repo, roles, strategy="logistic_team", ratings=load_ratings()))
```

Web demo 就是这样用的：

```bash
python3 game/examples/run_web_demo.py
# 或指定归档
WEREWOLF_ARCHIVE_DIRECTORY=archives/season2-20260923 python3 game/examples/run_web_demo.py
```

## 复现性

归档代码仍能独立算出规则内核指纹，用于确认游戏规则未被改动：

```bash
python3 -c "
import sys; sys.path.insert(0, 'archives/season2-20260923'); sys.path.insert(0, '.')
from code.archive import core_snapshot_hash
print(core_snapshot_hash())
"
# 应输出 b145c7ee778a63259e7f1c8590f369d0cb664014a5032b3bd8998cce94249168
# 与 records/evolution/archive/manifest.json 的 core_snapshot_hash 一致
```

该哈希覆盖 `game/core/` 下除 `runner.py` 以外的全部 14 个文件（只绑定相对路径与
内容，与目录位置无关）。如果哈希变了，说明规则内核被改动，归档的胜率数据不再与
当前引擎可比。

## 实验设计与结论

- 需求基线与边界：仓库根 `SEASON2.md`
- 探索质量分析、单向棘轮与自引用回路的实测结论、下一轮起点：`SEASON3.md`

其中两条对下一赛季最关键：

1. **单向棘轮**：148 次父子对比中 128 个子节点体积变大（86%），只有 3 个变小，
   中位每代 +11 KB。代码从 15.7 KB 涨到 62 KB 而强度不升，并撞上 12000 字符的
   prompt 预算墙（depth 3 累计 1252 次超限 → 1084 次 fallback → 971 次非法行动）。
2. **Meta-Agent 自引用**：74 份诊断里读历史诊断 113 次、读对局回放仅 36 次，
   外部战术卡只用了 8 次，导致分支反复收敛到记忆与解析器方案。

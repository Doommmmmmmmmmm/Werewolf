# 赛季 3：Meta-Agent 进化能力验证实验

状态：规划赛季；由项目负责人人工决定启动，不设自动化前置门槛。

---

## 1. 实验目的与核心原则

### 1.1 实验目的
**测试“将 Meta-Agent 自身纳入进化”这一机制，能否自发产生更强的 Task-Agent。**
核心是观察并检验系统在两层演进下的自组织进化能力，而不是人工干预出来的局部最优。

### 1.2 核心原则
1. **轻干预、全自动演进**：实验运行期间不加繁琐的人工审批与中途卡点，让两层 Agent 按照既定机制完全自主推演。
2. **事后统一算总账**：实验过程中不做过早的显著性检验或复杂验证；待全部预算（如若干个 Epoch、上千局）跑完后，基于全量记录统一拟合分析。
3. **底线规则绝对冻结**：游戏规则、核心裁决引擎（`game/core/`）、信息可见性与合规边界绝对不变；两层 Agent 的任何变异都必须写入版本归档，保证全链路可追溯。

---

## 2. 两层自进化流程设计（Epoch 双环机制）

整个系统采用“内层快步长、外层慢步长”的全自动 Epoch 循环：

```text
               ┌────────────────────────────────────────────────────────┐
               │              【外层：Meta-Agent 自动进化】              │
               │  输入：上一 Epoch 产生的 N 个 Task-Agent 的成败与对局汇总 │
               │  动作：LLM 自主更新 Meta-Prompt 与参数配置 (写入版本)  │
               └───────────────────────────┬────────────────────────────┘
                                           │ 生成新一代 Meta-Agent (M_k)
                                           ▼
               ┌────────────────────────────────────────────────────────┐
               │              【内层：Task-Agent 自动进化】              │
               │  在当前 Meta-Agent 指导下，自动循环跑 N 次：             │
               │  对局采样 → 诊断建议 → Pi 修改代码 → 评测留档           │
               └───────────────────────────┬────────────────────────────┘
                                           │ 完成 N 个 Task-Agent 探索
                                           ▼
                           自动进入下一个 Epoch（循环往复）
```

### 2.1 内层：Task-Agent 高频进化（每个 Epoch 跑 N 次）
- **触发频率**：每个 Epoch 执行固定步数（例如 $N=10$ 或 $15$ 次 Task 节点探索）。
- **执行逻辑**：
  1. 依据当前版本的 Meta-Agent 提示词与配置，采样对局回放；
  2. 生成诊断与修改建议，由代码生成器（Pi）修改 Task-Agent（代码或提示词）；
  3. 运行评测对局，记录阵营胜负、报错/Fallback 次数及代码体积；
  4. 结果写入 Task-Archive，直接作为下一代候选池的一部分。

### 2.2 外层：Meta-Agent 低频自主进化（每 Epoch 一次）
- **进化载体**：Meta-Agent 的“基因”表现为两份文件：
  - `meta_prompt.md`：指导诊断与方案提出的元提示词模板；
  - `meta_config.json`：控制回放采样胜负比、工具调用下限、变异算子比例等参数。
- **自动变异逻辑**：
  - 每个 Epoch 结束后，系统自动汇总这 $N$ 个 Task 节点的关键事实（各角色胜负、代码字符变化、超限/Fallback 率、高频错误模式）；
  - 调用模型进行“元自复盘”，模型根据实际表现自主更新 `meta_prompt.md` 与 `meta_config.json`；
  - 保存为新版本（如 `meta-v001`、`meta-v002`），无需人工确认，直接自动驱动下一 Epoch 的 Task 探索。

---

## 3. 基于 Season 2 实测的机制注入

为了让自动进化具备良好的基础环境，Season 3 吸收 Season 2 的实测结论作为初始状态：

### 3.1 抬高初始起点
不再从空白 base 开始，直接采用 Season 2 基于 2358 局拟合出的 Logistic 团队模型 6 角色最强节点作为 Season 3 的起点种子：
- **wolf**: `wolf-n00135`
- **villager**: `villager-n00029`
- **seer**: `seer-n00112`
- **witch**: `witch-n00096`
- **guard**: `guard-n00034`
- **hunter**: `hunter-n00144`

*（在全新的 `archive_root` 下隔离运行，不污染历史数据）*

### 3.2 遏制单向膨胀与 Prompt 墙
- 算子中明确支持“删除与重构”；
- 评测层将 Prompt 超限与 Fallback 作为高优先级缺陷指标反馈回汇总中，供 Meta-Agent 观测并自发调节代码体积。

### 3.3 纠偏回放证据利用
- 初始配置对回放阅读设立最低调用下限，确保诊断基于真实胜负对局而非空想；
- 历史诊断仅供去重与证否，引导模型自主从对局中提炼真实缺陷。

---

## 5. 运行与操作命令

实验运行所需的依赖与环境由仓库根目录下的 `.env` 提供（包含且仅包含 `TASK_*`、`META_*`、`PI_*` 三组模型配置）。

### 5.1 全新初始化（仅在清空重置或首次部署时使用）

如果需要彻底重置 Season 3 的进化树与评测数据：

```bash
# 1. 清空旧实验记录（归档树与评测数据）
rm -rf season3/records/season3/evolution/archive
rm -rf season3/records/season3/evolution/evaluations
rm -rf season3/records/season3/evolution/operations

# 2. 执行全新纯净初始化（生成 12 角色基线种子、meta-v000 与 manifest 冻结快照，不调用大模型）
python season3/run_season3_evolution.py --config season3/configs/season3.json --initialize-only
```

### 5.2 启动/恢复进化

正常推进 Season 3 自动化进化（支持随时中途停机，下次启动时会自动断点续传）：

```bash
# 推进指定步数（例如推进 12 步）
python season3/run_season3_evolution.py --config season3/configs/season3.json --steps 12

# 后台不挂断运行（推荐）
nohup python season3/run_season3_evolution.py --config season3/configs/season3.json --steps 120 > season3_evolution.log 2>&1 &
```


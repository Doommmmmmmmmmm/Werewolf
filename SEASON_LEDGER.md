# 狼人杀 Agent 赛季总账

这是人工维护的赛季级实验记录，不参与训练、对局或 skill 更新。每完成一个
round，只需补充一行「修订 / 10 局结果 / 结论 / 证据路径」；重要对抗评测则记录在
对应赛季的 test 小节。

## 当前编号口径

| 赛季 | 范围 | 状态 / 权威位置 |
| --- | --- | --- |
| Season 0 | 旧规则、round0–19 | `archives/season0-before-rule-overhaul-20260804/` |
| Season 1 | 规则重构版、round0–9 | `archives/season1-rule-overhaul-20260804/` |
| Season 2 | Task-Agent Harness | `SEASON2.md`、`records/seasons/season2/` |
| Season 3 | 可迭代 Meta-Agent Harness | `SEASON3.md`（计划） |

## 赛季 0：旧规则（已封存）

范围：round0–round19，共 200 局训练对局。原始记录、角色复盘和评测已封存于
[`archives/season0-before-rule-overhaul-20260804/`](archives/season0-before-rule-overhaul-20260804/)。

### 规则基线

- 12 人：狼人×4、平民×4、预言家、女巫、守卫、猎人；
- 狼人不能将狼队友作为刀口；
- 狼人以人数优势获胜；
- 无遗言、无警长竞选、无警徽与 1.5 票；
- 平民编造夜间信息不是明确支持的策略。

round0–19 的游戏规则未变，主要修订是各角色在每 10 局后根据复盘更新
`strategy.md`。具体 skill 输入/输出版本和角色复盘均保存在各 `roundN/skill/`、
`roundN/review/` 与 `roundN/log/` 中。

### 训练结果

| Round | 好人胜 | 狼人胜 | 备注 |
| --- | ---: | ---: | --- |
| 0 | 5 | 5 | 初始策略 |
| 1 | 2 | 8 | 策略迭代 |
| 2 | 3 | 7 | 策略迭代 |
| 3 | 7 | 3 | 策略迭代 |
| 4 | 4 | 6 | 策略迭代 |
| 5 | 5 | 5 | 策略迭代 |
| 6 | 7 | 3 | 策略迭代 |
| 7 | 2 | 8 | 策略迭代 |
| 8 | 3 | 7 | 策略迭代 |
| 9 | 4 | 6 | 完成前 10 轮 |
| 10 | 7 | 3 | 策略迭代 |
| 11 | 7 | 3 | 策略迭代 |
| 12 | 5 | 5 | 策略迭代 |
| 13 | 6 | 4 | 策略迭代 |
| 14 | 6 | 4 | 策略迭代 |
| 15 | 6 | 4 | 策略迭代 |
| 16 | 5 | 5 | 策略迭代 |
| 17 | 5 | 5 | 策略迭代 |
| 18 | 5 | 5 | 策略迭代 |
| 19 | 7 | 3 | 完成 20 轮 |

合计为好人 101 胜、狼人 99 胜。由于双方同时学习、每局发牌不同，这个训练内
胜率不能单独作为某一阵营 skill 提升的证据；应以冻结 skill 的对抗评测为准。

### test1：10 轮后的冻结对抗

证据：[test1 汇总](archives/season0-before-rule-overhaul-20260804/records/test1/log/test-summary.json)。

| 场景 | 局数 | 好人胜 | 狼人胜 | 结论 |
| --- | ---: | ---: | ---: | --- |
| 狼人 round10 vs 初始好人 | 10 | 2 | 8 | 最新狼人相对初始狼人表现更强。 |
| 初始狼人 vs 好人 round10 | 10 | 7 | 3 | 最新好人相对初始好人表现更强。 |

结论：在各自对初始对手的 10 局对照中，两边的 round10 skill 都显示出正向信号；
但样本量只有 10 局，不能据此判断长期、单调的提升趋势。

### test2：20 轮后的冻结对抗

证据：[test2 汇总](archives/season0-before-rule-overhaul-20260804/records/test2/log/test-summary.json)。
这里「round20」指完成 round19 复盘后的输出 skill；「round10」指完成 round9
复盘后的输出 skill。

| 场景 | 局数 | 好人胜 | 狼人胜 | 结论 |
| --- | ---: | ---: | ---: | --- |
| 好人 round20 vs 初始狼人 | 20 | 13 | 7 | 好人 round20 对初始狼有明显优势。 |
| 好人 round20 vs 狼人 round10 | 20 | 14 | 6 | 好人 round20 对 round10 狼仍保持优势。 |
| 狼人 round20 vs 初始好人 | 20 | 3 | 17 | 狼人 round20 对初始好人有明显优势。 |
| 狼人 round20 vs 好人 round10 | 20 | 12 | 8 | 面对 round10 好人时，狼人 round20 未显示出优势。 |

结论：20 轮后的两边都能显著击败初始对手；但狼人 round20 对 round10 好人的结果为
8/20，说明旧赛季狼人策略并非对更强好人稳定单调提升。后续应使用固定规则、更多
重复样本和阵营交叉对照继续验证，而不是只看训练内总胜率。

## 赛季 1：规则重构版（已完成并归档）

状态：已完成 round0–round9；训练、评测、复盘与 skill 均封存于
[`archives/season1-rule-overhaul-20260804/`](archives/season1-rule-overhaul-20260804/)。

### 与赛季 0 的关键差异

| 项目 | 赛季 0 | 赛季 1 |
| --- | --- | --- |
| 狼人刀口 | 不能刀狼队友 | 允许自刀与刀口做局 |
| 平民公开信息 | 不将编造夜间信息作为明确策略 | 可假跳或编造游戏内信息；不改变 Engine 事实 |
| 狼人胜利 | 人数优势 | 屠边：杀光全部神职或全部平民 |
| 遗言 | 无 | 全局最先死亡的 3 人可发表遗言 |
| 警长 | 无 | 首日死讯前竞选；一次 PK；警长 1.5 票、可调发言方向、可传或撕警徽 |
| 日间发言顺序 | 无警长/死者锚点机制 | 从最大座位号死者的下一位开始；平安夜从 1 号开始；警长可逆序 |

因此赛季 0 的训练胜率与 test1/test2 只能作为旧规则下的历史参照，不能与赛季 1
直接横向比较。

### round 记录

| Round | 本轮修订 | 好人胜 : 狼人胜 | 测试与结论 | 证据 |
| --- | --- | --- | --- | --- |
| 0 | 新赛季规则基线；角色 strategy 重置 | 3 : 7 | 六个实际角色均完成复盘更新；2 / 1402 次行动回退（0.14%），通过质量闸门 | `archives/season1-rule-overhaul-20260804/records/round0/` |
| 1 | 沿用赛季 1 规则与 round0 输出 strategy | 8 : 2 | 六个实际角色均完成复盘更新；2 / 1395 次行动回退（0.14%），通过质量闸门 | `archives/season1-rule-overhaul-20260804/records/round1/` |
| 2 | 从 round1 输出 strategy 重跑；单请求保守调度 | 7 : 3 | 六个实际角色均完成复盘更新；0 / 1331 次行动回退（0%），通过质量闸门 | `archives/season1-rule-overhaul-20260804/records/round2/` |
| 3 | 沿用 round2 输出 strategy；单请求保守调度 | 6 : 4 | 六个实际角色均完成复盘更新；0 / 1493 次行动回退（0%），通过质量闸门 | `archives/season1-rule-overhaul-20260804/records/round3/` |
| 4 | 沿用 round3 输出 strategy；单请求保守调度 | 6 : 4 | 六个实际角色均完成复盘更新；0 / 1424 次行动回退（0%），通过质量闸门 | `archives/season1-rule-overhaul-20260804/records/round4/` |
| 5 | 沿用 round4 输出 strategy；单请求保守调度 | 6 : 4 | 六个实际角色均完成复盘更新；0 / 1422 次行动回退（0%），通过质量闸门 | `archives/season1-rule-overhaul-20260804/records/round5/` |
| 6 | 沿用 round5 输出 strategy；单请求保守调度 | 7 : 3 | 六个实际角色均完成复盘更新；0 / 1424 次行动回退（0%），通过质量闸门 | `archives/season1-rule-overhaul-20260804/records/round6/` |
| 7 | 沿用 round6 输出 strategy；单请求保守调度 | 6 : 4 | 六个实际角色均完成复盘更新；0 / 1434 次行动回退（0%），通过质量闸门 | `archives/season1-rule-overhaul-20260804/records/round7/` |
| 8 | 沿用 round7 输出 strategy；单请求保守调度 | 7 : 3 | 六个实际角色均完成复盘更新；0 / 1492 次行动回退（0%），通过质量闸门 | `archives/season1-rule-overhaul-20260804/records/round8/` |
| 9 | 沿用 round8 输出 strategy；单请求保守调度 | 6 : 4 | 六个实际角色均完成复盘更新；0 / 1411 次行动回退（0%），通过质量闸门 | `archives/season1-rule-overhaul-20260804/records/round9/` |

### 冻结对抗：round9 输出 vs 初始 prompt

测试使用 12 人新规则局，各场景 11 局；只复制并读取 skill 快照，不调用复盘、
不改写任何 `strategy.md`。完整汇总见
[`archives/season1-rule-overhaul-20260804/records/skill-evaluation-latest-vs-initial-11-final/log/test-summary.json`](archives/season1-rule-overhaul-20260804/records/skill-evaluation-latest-vs-initial-11-final/log/test-summary.json)。

| 场景 | 使用 round9 输出的一方 | 胜负 | 方向性结果 |
| --- | --- | --- | --- |
| 最新狼人 vs 初始好人 | 狼人 | 狼人 4 : 好人 7 | 最新狼人胜率 36.4%，未显示出相对初始基线的正向信号。 |
| 初始狼人 vs 最新好人 | 好人 | 好人 8 : 狼人 3 | 最新好人胜率 72.7%，显示出明显的正向信号。 |

参照同规则的 round0 初始对初始训练结果（好人 3 : 狼人 7）：最新好人在固定初始
狼人对手下从 3/10 提升至 8/11；最新狼人对固定初始好人则从该基线的 7/10 降至
4/11。两组评测共 22 局均为 0 个模型错误、0 个回退动作，且来源已核验为
`archives/season1-rule-overhaul-20260804/skills/round0/input` 与
`archives/season1-rule-overhaul-20260804/skills/round9/output`。这是小样本、非配对的冻结对抗信号，
可以支持“当前好人 skill 有效提升、当前狼人 skill 尚未验证提升”的结论，但不能据此
断言长期或单调的策略质量变化。

### 历史补记规范

如需补记 Season 1 的历史说明，只更新本账本，不向封存目录写入新的训练结果；至少写明：

- 与上一 round 相比修改了什么（规则、Prompt、并发或 strategy 版本）；
- 10 局训练的好人 / 狼人胜负；
- 是否有冻结 skill 对抗 test，以及能否支持提升结论；
- 对应的 `archives/season1-rule-overhaul-20260804/records/roundN/` 或测试路径。

## 赛季 2：可迭代 Task-Agent Harness（开发中）

赛季 2 不再把每 10 局的角色复盘直接覆盖为唯一 `strategy.md`。可继承对象改为
Task-Agent Harness：它定义玩家如何筛选上下文、维护信念板、选择战术姿态、进行关键
节点规划、保存短期记忆和协调阵营。元层工作流仍由专家固定；它只负责产生、红队、
评测和选择多个 Task Harness 候选，不会改写自身。

赛季 2 的开赛前规则修订：夜间死亡者没有遗言；只有前 3 名白天被放逐出局的玩家可
发表遗言。赛季 1 已完成对局仍保留其“全局最先死亡 3 人可遗言”的历史规则，不能与
赛季 2 结果直接混合比较。

完整设计见 [SEASON2.md](SEASON2.md)。赛季 1 的规则、skill、记录和评测保持冻结；
赛季 2 初始基线固定使用赛季 1
`archives/season1-rule-overhaul-20260804/skills/round0/input` 快照。

第一版已经落地 `HarnessSpec`、`HarnessRuntime` 和固定
`TaskAgentMetaAgent`。当前只完成构造、红队、归档和局内接入，尚未把真实对抗结果
计入赛季结论；候选默认不自动晋升，实验记录位于 `records/seasons/season2/`。

## 赛季 3：可迭代 Meta-Agent Harness（计划）

赛季 3 以赛季 2 已稳定的 Task-Harness 语法和评测接口为固定基线，才允许“构建和
评测 Task Agent 的 Meta Harness”进入版本化、谱系化和密封评测。届时继承的单位是
研究、分析、候选生成、红队、选择与工具使用策略，而不是任意项目代码。

完整设计和启动前置条件见 [SEASON3.md](SEASON3.md)。

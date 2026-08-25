# 赛季 2：可迭代 Task-Agent Harness

状态：第一版已实现，尚未开始真实训练；当前先验证 Harness 构造和记录链路。

## 核心命题

赛季 2 优化的对象不是一篇会被整篇覆盖的 `strategy.md`，而是玩家在一局游戏中
读取信息、形成判断、选择战术、写入短期记忆和输出行动的 **Task-Agent Harness**。

当前系统近似于全读单轮 Agent：每次行动都把角色 base、角色 strategy、私密历史和
当前行动包一起交给一个模型，再由模型直接输出行动。策略的唯一可继承载体是角色级
`strategy.md`，并且每 10 局由复盘器整体替换。这会把“如何思考”固定在一个不可见的
单步提示词流程里，也会使四个同角色玩家天然使用同质行为。

赛季 2 将 Task Agent 表示为：

```text
Task Agent = 不可变规则与角色 base
           + 可版本化 HarnessSpec
           + 当前可见游戏状态
           + 由 HarnessSpec 选择的少量相关记忆 / 战术卡
           → 合法行动
```

这里可迭代的是 `HarnessSpec`，而不是游戏引擎、角色规则或完整历史本身。

## 赛季 2 开赛前冻结的规则基线

- 夜间死亡者（狼刀、毒药及夜间猎人连锁）没有遗言；只有前 3 名白天被放逐出局的玩家可发表公开遗言。

该规则属于不可变的 `GameEngine` 配置，不是 Task-Agent Harness、战术卡或 Meta-Agent
可以修改的对象。启动训练时会把它写入赛季 manifest，之后的候选一律在同一规则下评测。

赛季 2 的运行时模型也按职责冻结：局内 Task Agent 使用 `deepseek-v4-flash` 且
`reasoning.effort=none`；复盘和 Meta-Agent 更新使用原有 `gpt-5.5` 配置，不显式覆盖其
reasoning。两条模型配置必须随实验 manifest 记录，但 API 密钥绝不进入记录。

## Task-Agent Harness 的可变部分

每个候选 HarnessSpec 采用受限、可校验的结构化描述；候选可以组合已有模块，但不能
生成任意可执行代码。第一版至少包含以下字段。

| 模块 | 可迭代内容 | 例子 |
| --- | --- | --- |
| `context_policy` | 当前行动要读取哪些已知事实、近期事件和记忆摘要 | 白天投票只读取本轮发言、上轮票型、已公开身份和两个未决矛盾，而不是所有历史全文 |
| `belief_board` | 事实、声明、假设、置信度和待验证点的组织方式 | 将“系统确认的事实”与“玩家声称的信息”分栏；记录下一轮可证伪条件 |
| `tactic_router` | 为当前玩家选择的战术姿态及切换条件 | 狼人分配为潜伏、冲锋、倒钩或牺牲；真预查杀同伴、警徽归属变化等事件会触发重新选择 |
| `planning_policy` | 是否在关键节点进行短规划、比较方案和反事实检查 | 夜间先比较屠神 / 屠民路线；白天归票前检查“若目标翻好，谁收益最大” |
| `memory_policy` | 写入何种短期记忆、如何压缩和何时淘汰 | 只保留可验证承诺、票型转折、队友约定和未解决的身份冲突 |
| `coordination_policy` | 同阵营私聊的议程和确认方式 | 狼队每夜显式确认主刀、备刀、次日主推、分工和撤退触发条件 |
| `output_policy` | 行动前的事实核验与表达风格 | 先核验公开票型再发言；潜伏位限制主动归票频率，但在硬信息出现时允许退出潜伏 |

HarnessSpec 的结果应是“小型思考程序”，而非更多静态教条。战术以独立卡片保存，
每张卡必须具有：适用条件、行动倾向、反例、退出条件、预期可观察信号和置信度。
玩家每次只接收稳定核心、当前选中的少量卡片和相关状态摘要，不能再全量吞入所有
skill 与历史。

## 赛季 2 中保持固定的 Meta Harness

赛季 2 的元层流程由专家预先设计并冻结。它可以设计、变异、测试和选择 Task-Agent
Harness，但不会改写自己的工作流、提示词、工具权限、候选生成规则或评价标准。

```text
回放 / 冻结对抗 / 外部资料
          ↓
固定的研究员 → 回放分析员 → Harness 设计员 → 红队批评员
          ↓
候选 HarnessSpec archive → 固定评估器 → 固定选择器
          ↓
      已晋升的 Task-Agent Harness
```

这对应 HyperAgents 中“任务代理可以被元代理改造”的第一层，但刻意不让元代理在
本赛季改造自己。这样可以把 Task-Agent Harness 的收益与元层流程变化清楚地区分开。

### 第一版实现

代码中的 `TaskAgentMetaAgent` 固定执行四个阶段：

1. `replay_analysis`：按小批次阅读目标角色的完整审计回放，只携带该角色自己的
   `base.md` 和 `strategy.md`，提取证据、失败模式、反事实和不确定性；
2. `harness_design`：生成有限数量的结构化 `HarnessSpec` 候选。候选只能包含七个
   已定义模块和受限战术卡，不能生成代码；
3. `red_team_critic`：检查规则越界、信息泄漏、未经验证的结论、过长上下文和缺少
   退出条件的卡片；
4. Python 质量闸门：先用不调用模型的 `evaluate_harness` 合成行动包检查 schema、
   上下文/记忆上限、首次重规划、卡片退出条件和审计安全，再结合红队意见、证据覆盖
   与复杂度评分。所有候选都保留，只有通过两层闸门的候选才允许在显式晋升时成为
   active。

`HarnessRuntime` 在局内解释候选：限制可见事件和短期记忆长度，选择少量战术卡，
检测阶段或公开事件变化并标记 `replan_required`。它不会执行模型生成的表达式，
也不会向玩家开放网络、文件系统、shell 或审计状态。

同一角色的候选不会强制所有座位使用同一份 Harness。入口会保留通过结构闸门的候选池，
由玩家编号的稳定哈希为每个座位分配一个变体；因此同角色玩家可以具有不同默认姿态、
上下文窗口和卡片组合，同时在重放时仍可复现。

当前实现的元流程是固定的：Meta-Agent 只能构造 Task-Agent，不能修改自己的提示词、
候选 schema、评估器或工具权限。Meta-Agent 的自我演化仍留给 Season 3。

## 候选、谱系与评测

不再使用 `S0 → S1 → S2` 的单链替换。每个角色维护不可变 archive 节点：

- `incumbent`：已验证的当前冠军；
- `replay_mutation`：基于回放中可验证失败模式的变体；
- `counterfactual`：针对“若换一种战术会怎样”的反事实变体；
- `research_prior`：由外部资料转化、尚未被实证验证的假设；
- `red_team`：故意与当前主流策略相反的受控变体；
- `recombined`：仅在两张战术卡互不冲突时允许的组合变体。

选择父本时从质量、行为新颖度和不确定性组成的 Pareto 候选集中抽样，而不是只取
最新或当前最高分版本。失败候选仍保留其谱系、适用条件和反例，防止系统遗忘可在
不同对手或规则状态下重新有效的打法。

评测目标采用分阶段流程：

1. 训练局仅生成证据，不自动覆盖任何活跃版本；
2. 每个焦点角色生成有限数量的候选 HarnessSpec；
3. 第一版代码先用固定合成行动包和红队结果筛除结构上明显退化者；
4. 冻结种子、冻结对手池上的真实对抗评估作为下一步接入的可插拔评估器，不能由
   训练内自博弈结果代替；
5. 只有在胜率、合法行动率、事实一致性、跨对手稳健性和行为多样性均通过门槛时，
   才晋升为冠军或探索版本。当前 `promote` 只允许通过 schema、静态运行时和红队
   闸门的候选，尚不声称它已经在冻结对手池上优于 Season 1。

对狼人的第一阶段，焦点是让同局不同狼人获得不同的姿态卡与退出条件，同时以
“狼人最新版本 vs 冻结好人池”作为主评测，而不是只看混合自博弈总胜率。

## 外部研究与 Web Search

Web Search 是赛后研究代理的工具，不是游戏玩家的工具。它只能在候选生成阶段使用，
不能在进行中的游戏中查询资料或访问网络。

第一版代码提供 `ResearchSource` / `JsonSearchProvider` 接口，但默认 provider 为
`None`，不会产生网络请求。注入 provider 后，Meta-Agent 只把查询词、标题、URL、
摘要、获取时间和内容哈希保存为待验证来源；来源仍必须经过回放证据和红队闸门，不能
直接写入 active Harness。

研究工具必须满足：

- 通过可配置的搜索 Provider 调用；默认无配置时关闭，而不是依赖不稳定的网页抓取；
- 每条材料保存查询词、标题、URL、摘要、获取时间、内容哈希和来源可信度；
- 网页内容被视为不可信引用，永远不能充当系统指令、引擎规则或直接覆盖策略；
- 研究员只能将资料转化为带来源的“待验证假设”；红队必须给出失效条件；
- 只有经过冻结对抗验证的假设可以成为活跃战术卡。

预计支持 Brave、Tavily 或用户提供的兼容 JSON 检索端点；Provider 的凭据只从环境
变量读取，不写入记录。资料与回放证据分开保存，保证日后可以审计“某条打法来自
实战”还是“来自外部先验”。

## 不能被 Task Harness 改写的边界

- `GameEngine`、规则、信息可见性和合法行动校验；
- 各角色 `base.md`；
- 当前玩家本不应知道的私密信息；
- 在局内调用 Web、文件系统、shell 或外部代码；
- archive 中已保存的候选与评测记录；
- 模型 API 密钥、网络配置和资源上限。

## 记录与可复现性

赛季 2 与赛季 1 完全隔离。初始基线固定为赛季 1
`archives/season1-rule-overhaul-20260804/skills/round0/input` 的快照，而不是
可能已经收敛的 `round9` 输出。每次实验都必须记录：

```text
records/seasons/season2/
  season-manifest.json
  active/<role>.json                 # 仅显式晋升后的当前版本
  archive/<role>/<node-id>/harness.json
  archive/<role>/<node-id>/cards/*.md
  research/roundN/<role>/sources.json  # 无研究时也写入空快照
  experiments/<epoch>/assignment-manifest.json
  experiments/<epoch>/evaluation-summary.json
  experiments/<epoch>/pipeline-trace.json
  training/roundN/...                 # 现有公开 / 全量对局记录格式
```

当前代码对应的主要文件：

```text
werewolf_game/harness.py       # HarnessSpec、TacticalCard、HarnessRuntime
werewolf_game/meta_agent.py    # 回放分析、候选构造、红队、归档与晋升
werewolf_game/prompts/task_agent_system.txt
werewolf_game/prompts/meta_*_system.txt
examples/run_season2_round.py  # Season 2 Harness 构造 + 10 局训练入口
```

入口默认使用只读的 `archives/season1-rule-overhaul-20260804/skills/round0/input` 作为 Season 1 初始基线；可用
`WEREWOLF_SEASON2_BASELINE_SKILL_DIRECTORY` 指向另一份冻结快照。这样 Season 2 的
初始策略不会随工作区当前 `strategy.md` 漂移。

启动入口默认不自动晋升候选；确认实验结果后设置
`WEREWOLF_SEASON2_PROMOTE=true`。这样 active 版本与 archive 版本不会因为一次模型
输出而被静默覆盖。

对局记录额外写入每个玩家实际加载的 HarnessSpec、战术卡 ID、父本、选择原因和
重规划触发记录。完整审计记录中的 `agent_harness_catalog` 保存去重后的定义，
`TASK_AGENT_HARNESS_TRACE` 保存运行时触发原因；公开记录剥离这些字段。所有旧赛季
记录保持只读。

## 赛季完成条件

赛季 2 的完成不以训练内单次胜率为准，而以以下事实为准：

1. 同角色不同玩家确实会被分配并执行可区分的战术姿态；
2. 这些姿态能在公开信息变化时按退出条件重规划，而非死守首夜方案；
3. archive 中存在多条可复现、非最新父本分支；
4. 至少一个狼人 Harness 在固定好人池上的表现和行为多样性均优于赛季 1 基线；
5. 元层工作流在整个赛季保持固定。元层自我演化留给赛季 3。

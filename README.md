# Python 狼人杀 Agent 框架

这是一个以 Python 实现的狼人杀多 Agent 框架。规则由代码裁决，不需要“上帝 Agent”。

```text
GameEngine       发牌、状态、可见性、行动校验、结算、胜负
NightLoop        一个完整黑夜的行动顺序
DayLoop          一个完整白天的行动顺序
HunterReactionLoop 死亡猎人的独立开枪反应
SheriffElectionLoop 首日死讯公布前的警长竞选与一次 PK
SheriffBadgeLoop  死亡警长公开传递或撕毁警徽
LastWordsLoop     前三名白天放逐出局玩家的公开遗言
GameRunner       交替调用两个 loop 和参与者，不做策略判断
Participant      LLM、真人网页或脚本玩家的统一接口
```

## 最重要的阅读入口

如果只想理解流程，请按这个顺序看：

1. `werewolf_game/loops/night_loop.py`：守卫 → 狼人私聊 → 狼人投票 → 预言家 → 女巫 → 夜晚结算。
2. `werewolf_game/loops/sheriff_election_loop.py`：首夜死讯公布前的上警、竞选发言、投票和 PK。
3. `werewolf_game/loops/day_loop.py`：警长确定发言方向 → 公开发言 → 同步投票 → 出局结算。
4. `werewolf_game/loops/hunter_reaction_loop.py`、`sheriff_badge_loop.py`、`last_words_loop.py`：死亡后的公开反应。
5. `werewolf_game/engine.py`：每一步调用的规则、合法性和状态变化。
6. `werewolf_game/runner.py`：如何把 Agent、loop、记录串起来。

流程本身是显式的：

```text
GameRunner
  ├─ NightLoop.run()
  │    ├─ 同步公开状态给所有存活玩家
  │    ├─ 守卫暗中守护
  │    ├─ 狼人依序私聊（最多 30 字）
  │    ├─ 狼人同步投票
  │    ├─ 预言家查验
  │    ├─ 女巫行动
  │    └─ 夜晚结算
  ├─ SheriffElectionLoop.run()（仅首日；先于首夜死讯公布）
  │    └─ 上警 → 候选人按 1→12 发言 → 投票 → 一次 PK
  ├─ 公布首夜死讯
  ├─ HunterReactionLoop → SheriffBadgeLoop（按需）
  │    └─ 开枪、警徽交接
  ├─ DayLoop.run()
  │    ├─ 以天亮死者为锚点决定公开发言顺序（最多 200 字）
  │    └─ 存活玩家同步投票、出局结算
  ├─ HunterReactionLoop → SheriffBadgeLoop → LastWordsLoop（白天放逐后按需）
  │    └─ 开枪、警徽交接、前三名白天放逐者的遗言
  └─ 每次死亡反应结算完成后，若未结束则继续下一个昼夜
```

`GameEngine` 不调用模型或 UI；`NightLoop`、`DayLoop` 也不决定谁该投谁。模型只能提交结构化行动，仍要经过引擎校验。

## Prompt 不在代码里

所有发送给 Agent 的固定提示词都位于 `werewolf_game/prompts/`：

- `player_system.txt`：玩家 Agent 的系统提示词；
- `player_turn_instruction.txt`：每次行动请求的指令；
- `public_narrator_system.txt`：可选公开记录员 Agent 的提示词。
- `roles/<角色>/base.md`：不可自动改写的固定角色规则；
- `roles/<角色>/strategy.md`：可由复盘角色 Agent 更新的经验策略；玩家只会加载自己角色的这两份文件。

策略不是规则：玩家提示词明确规定，当 `strategy.md` 与固定规则、当前行动包或公开事实冲突时，固定规则优先。Python 代码通过 `RoleStrategyStore` 只允许复盘器写入 `strategy.md`，不会改写 `base.md`。

## 当前规则基线

训练与单局入口默认使用 12 人局：4 狼人、4 平民、预言家、女巫、守卫、猎人。`create_rules_for_player_count(...)` 仍支持 7–12 人标准预设：

| 人数 | 身份配置 |
| --- | --- |
| 7 | 狼人×2、预言家、女巫、平民×3 |
| 8 | 狼人×2、预言家、女巫、守卫、平民×3 |
| 9 | 狼人×3、预言家、女巫、守卫、平民×3 |
| 10 | 狼人×3、预言家、女巫、守卫、猎人、平民×3 |
| 11 | 狼人×3、预言家、女巫、守卫、猎人、白痴、平民×3 |
| 12 | 狼人×4、预言家、女巫、守卫、猎人、平民×4 |

也可以用 `optional_roles=("guard", "hunter", "idiot")` 将同数量的平民替换为额外可选身份；想要重复身份或完全非标准阵容时，直接构造 `RuleSet(role_deck=...)`。

- 狼人夜间先私聊，再共同投票刀人；刀口可包含狼队友，自刀、骗药和刀口做局按正常夜间规则结算；
- 狼人私聊最多 30 个汉字，白天公开发言最多 200 个汉字；
- 发言必须包含中文，默认拒绝英文字母；
- 平民可以在公开发言中假装神职或编造游戏内信息；这类内容始终只是玩家声明，不会产生真实夜间行动或系统确认；
- 每夜开始，所有存活玩家会收到仅包含公开状态的同步包；
- 预言家查验结果仅自己可见；女巫仅自己知道药水和当晚狼刀目标；
- 守卫每夜保护一名存活玩家，默认不能连续守同一人；守护可挡狼刀但不能挡毒药；
- 猎人死亡后可公开带走一名存活玩家或跳过，且会正确处理猎人连锁反应；
- 白痴第一次白天被放逐时公开身份、免于死亡，默认此后失去投票权；
- 夜间死亡者（狼刀、毒药及夜间猎人连锁）没有遗言；前 3 名白天被放逐出局的玩家各有一次公开遗言，最多 200 字；
- 首日先进行警长竞选，再公布首夜死讯。首夜死亡者仍可上警和投票；候选人按 1→12 发言。首次平票时平票候选人进行一次 PK，PK 再平则本局无警长；
- 警长在白天放逐投票中拥有 1.5 票。每个白天可选择从天亮死者的下一位顺序发言，或从上一位逆序发言；多名死者以座位号最大的为锚点；平安夜从 1 号开始；
- 警长死亡时可公开将警徽传给任一存活玩家，或撕毁；
- 狼人全部死亡时好人胜；狼人消灭全部神职或全部平民时立即获胜（屠边）。

规则配置在 `werewolf_game/rules.py`，而非 prompt 中。

## 运行

只需要 Python 3.11+，无第三方 Python 依赖。

```bash
python3 -m unittest discover -s tests -v
python3 examples/run_scripted_game.py
```

第二条命令使用脚本玩家跑完一局，不会请求任何外部模型。

配置好 `.env` 后可运行真实模型玩家：

```bash
python3 examples/run_llm_game.py
```

### 人工玩家

人工玩家实现为与 LLM 相同的 `Participant`，可和其余 LLM 玩家混合对局：

```bash
python3 examples/run_human_game.py
```

默认由引擎随机发牌，`p1` 是人工玩家。人工玩家也可以在开局前指定自己的身份，
其他牌仍会随机分配给其余玩家：

```bash
WEREWOLF_HUMAN_PLAYER=p1 WEREWOLF_HUMAN_ROLE=wolf \\
  python3 examples/run_human_game.py
```

`WEREWOLF_HUMAN_ROLE` 支持 `wolf`、`villager`、`seer`、`witch`、`guard`、
`hunter`、`idiot`；省略它就是随机身份模式。指定身份必须在当前牌堆中有对应牌，
不能增加或替换角色数量。身份分配模式和指定映射会写入完整记录的 metadata，
但不会进入公开记录。

人工玩家提示中会显示自己的秘密身份、队友（若为狼人）、合法行动和可见事件，
输入仍会经过本地提示校验及 `GameEngine` 的最终校验。常用输入格式为：

```text
发言 三号的查杀逻辑需要重新核对
投票 p7
刀 p10
验 p4
守 p8
救 p5
毒 p6
上警
警长投票 p3
顺序 逆序
传徽 p9
撕徽
遗言 我认为七号是狼
pass
```

也支持直接输入 JSON 对象。`WEREWOLF_HUMAN_SHOW_EVENTS` 可调整每次提示显示的
最近可见事件数量（默认 8）。

沿用原有的服务端环境变量：`OPENAI_API_KEY`、`OPENAI_BASE_URL`、`OPENAI_MODEL`，并支持 Responses、Messages、Chat Completions 三种协议。密钥不会被写入记录。

### Task / Meta 模型分流

局内的 `LlmParticipant`（包括人工局中的其余 AI 玩家和冻结 skill 评测玩家）统一使用
Task profile：默认 `WEREWOLF_TASK_MODEL=deepseek-v4-flash`，并通过
`WEREWOLF_TASK_REASONING_EFFORT=none` 在 Responses 请求中发送
`reasoning: {"effort": "none"}`；若网关使用 Messages 协议，则发送
`thinking: {"type": "disabled"}`。每个行动都独立调用该模型，不会调用 Meta 模型。

复盘器和后续负责更新 Task-Agent Harness 的 Meta-Agent 使用 Meta profile：当前
`WEREWOLF_META_MODEL=gpt-5.5`，不设置 `WEREWOLF_META_REASONING_EFFORT`，因此保留原有
服务端默认 reasoning 行为。两个 profile 复用 API 地址与密钥，但模型名和 reasoning
配置独立；不要把密钥写入 profile 变量。

### Season 2：Task-Agent Harness

Season 2 开始把“玩家如何读取上下文、形成信念、选择战术、记忆和重规划”
从单个 `strategy.md` 中拆成可版本化的 `HarnessSpec`。固定的
`TaskAgentMetaAgent` 会按以下顺序工作：

```text
完整回放 → 回放分析 → Harness 候选 → 红队批评 → 静态运行时闸门 → archive
```

候选由 `HarnessRuntime` 在对局中解释，不执行候选中的代码。每次对局记录会保存实际
加载的 Harness ID、指纹、父本、来源类型和战术卡 ID；完整记录还会保存去重后的
HarnessSpec catalog，以及每次行动是否因阶段/公开事件触发重规划的 trace；公开记录
不会暴露这些审计字段。

候选晋升前会经过 `evaluate_harness` 的确定性静态检查：它用合成行动包验证 schema、
上下文和记忆上限、首次重规划触发、战术卡退出条件以及审计字段边界；该检查不调用
模型，也不把静态通过误当成胜率提升。每个候选的静态结果、红队意见和最终分数分别
写入 `candidate-assessments.json` 与 `evaluation-summary.json`。
每次构造还会写 `pipeline-trace.json`，记录回放分析、研究、设计、红队/静态闸门和选择
阶段是否走了 fallback；异常只记录类型，不记录密钥或完整响应。

第一版尚未自动为每个候选运行冻结对手池对局；真实胜率、合法行动率和跨对手稳健性
评估会作为下一步可插拔 evaluator 接入，未完成前不会把 Season 2 的结果宣称为策略提升。

同角色的通过候选会组成一个小型变体池，按玩家编号稳定分配，避免四名狼人或多个神职
玩家机械复用同一个姿态；相同 seed 和 Harness archive 可以复现这次分配。

试运行入口：

```bash
python3 examples/run_season2_round.py
```

入口默认以只读方式加载
`archives/season1-rule-overhaul-20260804/skills/round0/input` 作为 Season 1 冻结基线，并读取
`records/seasons/season2/training/round{N-1}/log/` 的历史，生成候选并运行当前 round 的 10 局；
若冻结基线目录不存在会直接报错，不会静默读取当前工作区的迭代策略；仅对基线快照中
没有的新增角色使用对应的固定初始档案。
候选只写入 `archive`，不会自动成为 active。确认候选通过固定评估后，
再设置 `WEREWOLF_SEASON2_PROMOTE=true`。局内仍使用 `deepseek-v4-flash` +
`reasoning.effort=none`，Meta-Agent 使用 `gpt-5.5` 原配置。

它不允许 Meta-Agent 修改自己的工作流、规则、角色 `base.md` 或评估边界；
Meta-Agent Harness 的自我迭代属于 Season 3。

外部研究工具 `ResearchSource` / `JsonSearchProvider` 是可选注入项，默认关闭。即使启用，
网页内容也只作为带查询词和内容哈希的研究来源进入 Meta-Agent，不能被游戏玩家访问，
也不能绕过红队和固定评估器直接成为策略。命令行入口通过
`WEREWOLF_SEASON2_RESEARCH_ENDPOINT` 启用兼容 JSON provider，API key 只从环境变量
`WEREWOLF_SEASON2_RESEARCH_API_KEY` 读取。

模型请求的容错参数也可以通过环境变量调整：`MODEL_MAX_RETRIES` 控制普通网络或
服务端错误的重试次数（默认 2，线性退避）；`MODEL_TIMEOUT_MS` 控制一次 HTTP
模型读取超时（默认 45000，最大 180000）；`MODEL_USAGE_LIMIT_RETRIES` 控制
`usage limit` / HTTP 429 的专项即时重试次数（默认 12）。后者适合上游使用账号池的
场景：每次重发都会给上游一次重新选择可用账号的机会。批量训练入口的单次行动超时
默认是 180 秒，可用 `WEREWOLF_DECISION_TIMEOUT_SECONDS` 覆盖。

复盘会逐批阅读完整审计回放。`WEREWOLF_REVIEW_REPLAY_BATCH_SIZE` 默认为 2；如果
上游对长输入的读取较慢，可设为 `1`，以更多但更小的请求换取更高的成功率。它只影响
复盘请求，不会改变游戏规则或既有对局记录。

### 模型调度、扩缩容与质量闸门

12 人局的一次同步投票最多会同时请求 12 个 Agent；因此 `WEREWOLF_GAME_CONCURRENCY=4`
不等于只有 4 个 API 请求。真实模型入口会创建一个由全部玩家和复盘器共享的
`ModelRequestCoordinator`，默认最多 8 个在途模型请求（`WEREWOLF_MODEL_MAX_IN_FLIGHT`）。
行动超时后，底层同步 HTTP 调用若仍在运行，会继续占用这个槽位直到真正结束，避免回退后
又向 API 叠加新请求。

`run_llm_round.py` 默认启用保守的自适应调度：在 `game_concurrency` 这个上限内从 2 局
（或更低）开始；连续两局健康才加一局，发生参与者异常则将游戏并发和请求池容量减半并
冷却两局。可用以下环境变量调整：

| 环境变量 | 默认值 | 含义 |
| --- | ---: | --- |
| `WEREWOLF_GAME_CONCURRENCY` | `1` | 游戏并发上限 |
| `WEREWOLF_ADAPTIVE_CONCURRENCY` | `true` | 设为 `false` 关闭自适应 |
| `WEREWOLF_INITIAL_GAME_CONCURRENCY` | `2` | 自适应启动并发 |
| `WEREWOLF_MODEL_MAX_IN_FLIGHT` | `8` | 全局模型请求上限 |
| `WEREWOLF_HEALTHY_P95_LATENCY_MS` | `45000` | 允许扩容的 P95 延迟阈值 |
| `WEREWOLF_MAX_FALLBACK_RATE` | `0.01` | 允许自动更新 skill 的最大回退率 |
| `WEREWOLF_MAX_FALLBACKS_PER_GAME` | `3` | 单局允许的最大回退次数 |

每次模型回退都会在完整记录的 `runner_events` 中以 `FALLBACK_ACTION` 标明。若一个
round 超过上述质量阈值，系统保留全部记录和 skill 快照，但不会调用复盘器或写回
`strategy.md`；可显式设置 `WEREWOLF_ALLOW_DEGRADED_REVIEW=true` 覆盖这一保护。

可选的批量标识：

```bash
WEREWOLF_GAME_ID=trial-01 WEREWOLF_GAME_SEED=trial-seed-01 WEREWOLF_QUIET=1 \
  python3 examples/run_llm_game.py
```

选择人数和可选身份：

```bash
WEREWOLF_PLAYER_COUNT=10 python3 examples/run_scripted_game.py
WEREWOLF_PLAYER_COUNT=7 WEREWOLF_OPTIONAL_ROLES=guard,hunter \
  python3 examples/run_llm_game.py
```

## 角色策略自我更新

一个训练 round 固定为 10 局，从 `round0` 开始。运行完整 round 时，10 局会先全部结束；随后每个实际出现过的角色各自触发一次复盘，而不是让 10 个具体玩家分别复盘：

```bash
python3 examples/run_llm_round.py
WEREWOLF_TRAINING_ROUND=1 WEREWOLF_PLAYER_COUNT=10 \
  python3 examples/run_llm_round.py
```

角色复盘 Agent 只能接收以下材料：自己的 `base.md`、自己的 `strategy.md`，以及该 round 的 10 局完整审计回放。它不会收到其他角色的 Markdown、私密记忆或系统提示词。复盘会分别分析做得好与不好之处、对手策略漏洞、当前策略漏洞，然后完整替换自己的 `strategy.md`；固定规则永不自动改写。

每个 round 还会保留角色 skill 的输入和输出版本：输入版是该 round 开局前
玩家实际加载的 `base.md + strategy.md`，输出版是复盘后将供下一 round 使用的档案。
`log/skill-version.json` 按角色记录版本号、来源和 SHA-256；策略内容不变时版本号
不递增，复盘或人工改动导致内容变化时才递增。

也可以连续运行 10 次 `run_llm_game.py`：它们会依次占用同一 round 的
`game0`–`game9`，第 10 局完成后自动触发一次角色复盘。`round-review.json`
会作为幂等标记，因此再次运行不会重复改写策略。`run_scripted_game.py` 只生成
测试对局记录，不调用模型复盘。单局入口会将 `WEREWOLF_GAME_SEED` 作为基础种子，
再组合 round 与 game 编号，因此十局发牌彼此不同但仍可复现。

## 冻结 skill 对抗评测

训练完成后，可以用冻结的历史 skill 快照验证某个角色的迭代是否真正改善胜率，
而不是继续训练。默认评测名为 `test1`，每个场景各跑 10 局，使用 12 人标准局：

```bash
WEREWOLF_PLAYER_COUNT=12 WEREWOLF_GAME_CONCURRENCY=4 \
  python3 examples/run_skill_test.py
```

它固定运行以下两组对照，过程中不会创建 `RoleStrategyReviewer`、不会调用复盘模型，
也不会改写 `werewolf_game/prompts/roles/*/strategy.md`：

1. `wolf-latest-vs-others-initial`：狼人使用最新训练 round 的 `skill/output`，其余角色使用 `round0/skill/input`。
2. `wolf-initial-vs-others-latest`：狼人使用 `round0/skill/input`，其余角色使用最新训练 round 的 `skill/output`。

这里“旧版本”固定指训练开始前的 v0 输入快照，“最新版本”默认自动发现最后一个
拥有完整输出快照的 `roundN`；也可通过 `WEREWOLF_TEST_INITIAL_ROUND` 和
`WEREWOLF_TEST_LATEST_ROUND` 显式指定。`WEREWOLF_TEST_GAMES_PER_SCENARIO`
可设置每个场景的局数（默认 10），`WEREWOLF_SKILL_TEST_ID` 可更改评测名，避免覆盖
已有结果。评测会把实际提供给玩家的 base/strategy 副本及其来源、版本和 SHA-256 写入
`records/<test-id>/`，所以即使之后继续训练，也能复现该次对照。

例如 `records/test1/` 的结构为：

```text
records/
  test1/
    log/
      test-manifest.json             # 两个场景的来源快照清单
      test-summary.json              # 两组胜负汇总，skill_updated 永远为 false
    wolf-latest-vs-others-initial/
      skill/roles/<role>/{base,strategy}.md  # 只读测试快照副本
      public/game0.md ... game9.md
      full/game0.md ... game9.md
      log/full-game0.json ...
      log/test-summary.json
    wolf-initial-vs-others-latest/
      ...
```

若同一个 `test_id` 已存在且来源配置一致，入口只复用已完成的对局；若来源配置不同， 
则会拒绝覆盖，以保护既有评测数据。

### 10 / 20 轮 skill 阵营对抗

`examples/run_skill_progression_test.py` 用于检验不同训练阶段的 skill 是否真的
提升了各自阵营的胜率。训练 round 从 `round0` 开始计数，因此默认快照映射为：

- 初始：`round0/skill/input`；
- 完成 10 个训练 round 后：`round9/skill/output`；
- 完成 20 个训练 round 后：`round19/skill/output`。

默认在 `records/test2/` 写入以下四个场景，每个场景 20 局，共 80 局；整个过程
不运行复盘 Agent，也不改写正常训练的 `strategy.md`：

1. 好人 round20 vs 狼人 initial；
2. 好人 round20 vs 狼人 round10；
3. 狼人 round20 vs 好人 initial；
4. 狼人 round20 vs 好人 round10。

```bash
WEREWOLF_SKILL_TEST_ID=test2 \
WEREWOLF_PLAYER_COUNT=12 \
WEREWOLF_TEST_GAMES_PER_SCENARIO=20 \
WEREWOLF_GAME_CONCURRENCY=2 \
WEREWOLF_MODEL_MAX_IN_FLIGHT=3 \
WEREWOLF_DECISION_TIMEOUT_SECONDS=600 \
MODEL_TIMEOUT_MS=180000 MODEL_MAX_RETRIES=4 MODEL_USAGE_LIMIT_RETRIES=12 \
python3 examples/run_skill_progression_test.py
```

可用 `WEREWOLF_TEST_INITIAL_ROUND`、`WEREWOLF_TEST_ROUND10`、
`WEREWOLF_TEST_ROUND20` 改变这三份冻结快照的实际来源。每个场景都会把各角色的
来源 round、输入/输出阶段、版本号和 SHA-256 写进 `skill-sources.json`，汇总胜负
写入 `records/test2/log/test-summary.json`。

## 记录与复盘

跨赛季的规则差异、历史 test 结果与结论见 [SEASON_LEDGER.md](SEASON_LEDGER.md)。

记录按训练 round 保存，文件名稳定而不含时间戳。例如 `round0`：

```text
records/
  round0/
    public/
      game0.md ... game9.md       # 真人可展示的公开记录
    full/
      game0.md ... game9.md       # 管理员完整回放，含底牌和夜间私密行动
    review/
      role-wolf.md                # 每个角色的一次复盘
      role-seer.md
      ...
    skill/
      input/wolf/base.md          # 本 round 使用的 skill 输入版本
      input/wolf/strategy.md
      output/wolf/base.md         # 复盘后的 skill 输出版本
      output/wolf/strategy.md
      ...
    log/
      public-game0.json           # 与 public/game0.md 对应的结构化记录
      full-game0.json             # 与 full/game0.md 对应的结构化记录
      review-wolf.json            # 与 review/role-wolf.md 对应的结构化复盘
      skill-version.json          # 每个角色的输入/输出 skill 版本、哈希和来源
      round-review.json           # 本 round 已完成复盘的幂等标记
      round-quality.json          # 回退率、质量闸门与是否允许更新 skill
      round-performance.json      # 请求池健康度和每局后的自适应扩缩容决策
      ...
```

每个文件默认权限为 `600`，目录为 `700`。公开版只含公开事件；完整回放和角色复盘均应只在可信环境保存和阅读。

每个完整对局的 `full-gameN.json` 还会在顶层 `model_token_usage` 写入**玩家行动**的
模型 Token 汇总，并在 `full/gameN.md` 中显示同一张表。它优先使用上游 API 返回的
`usage`（兼容 `input/output_tokens` 与 `prompt/completion_tokens`）；不会保存 prompt 或
原始模型响应。`availability` 为 `complete` 时可作为精确用量，`partial` 表示存在未报告
或重试请求，`unavailable` 则表示该模型服务没有返回 usage，系统不会编造估算值。角色复盘
与可选公开播报的调用不计入单局统计。

公开记录员接口在 `werewolf_game/recorder.py`：

```text
Engine ──全量事件──> 审计记录（管理员）
       └─公开事件──> PublicRecorder / 真人 UI / 可选 LlmPublicNarrator
```

记录员绝不参与规则裁决，也没有审计记录的引用。`LlmPublicNarrator` 是可选展示层，默认不会调用模型；代码生成的事件记录才是事实来源。

已有审计 JSON 可重新渲染：

```bash
python3 scripts/render_replay.py records/某局.json
python3 scripts/render_replay.py --public records/某局.json
```

## Python 目录结构

```text
werewolf_game/
  engine.py          规则权威与信息可见性
  loops/
    night_loop.py    完整黑夜流程
    day_loop.py      完整白天流程
    hunter_reaction_loop.py  猎人死亡反应
  participants/      LLM / 脚本 / 未来真人玩家接口
  prompts/           独立 Prompt 模板
    roles/<role>/base.md      固定角色规则
    roles/<role>/strategy.md  可学习经验策略
  llm/               外部模型 API 客户端
    coordinator.py    全局请求池、延迟/失败统计与动态容量接口
  recorder.py        公开记录员与可选播报员
  records.py         审计和公开记录存储
  skill_versions.py  每个 round 的角色 skill 输入/输出版本快照
  replay.py          JSON 到中文 Markdown 复盘
  runner.py          只调度，不裁决
  training.py        每 10 局执行一次的训练 round 调度
  review.py          按角色复盘并更新 strategy.md
  harness.py         HarnessSpec、战术卡与安全的上下文筛选运行时
  meta_agent.py      固定 Meta-Agent：回放分析、候选构造、红队与归档
  research.py        可选的、默认关闭的安全检索 provider 接口
examples/            Python 示例
tests/               Python unittest 测试
```

## 实现说明

本目录只保留 Python 实现：`werewolf_game/`、`examples/*.py`、`tests/` 和
`scripts/render_replay.py`。运行、测试和记录格式均以这套实现为准。

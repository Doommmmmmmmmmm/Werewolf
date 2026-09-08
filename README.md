# Python 狼人杀 Agent 游戏框架

规则由 Python 引擎裁决，参与者可以是 LLM、真人或脚本，不需要“上帝 Agent”。Season 2
进化代码位于独立的 `werewolf_game/season2/` 包中，不进入 `GameEngine`，也不会改变规则裁决。

## 运行结构

代码和赛季运行资料统一放在 `werewolf_game/` 下；根目录只保留项目文档、测试、临时资料和
历史赛季归档：

```text
werewolf_game/
├── core/          稳定游戏内核：规则、状态、昼夜流程和胜负裁决
├── agents/        Task-Agent、参与者和模型客户端
├── recording/     审计记录、公开记录和复盘渲染
├── season2/       Season 2 Meta-Agent、进化树和评测器
├── prompts/       全局及角色 Prompt
├── configs/       赛季配置
├── skills/        Meta-Agent 可按需读取的外部 Skill
├── runtime/pi/    Pi coding agent 源码
├── records/       运行时产生的对局和进化结果
├── examples/      启动入口
└── scripts/       辅助脚本
```

当前只保留 canonical 模块路径；核心实现位于 `core/`，Agent 实现位于 `agents/`，记录实现
位于 `recording/`，不再维护第二套兼容入口。

```text
GameEngine       发牌、状态、可见性、行动校验、结算和胜负
NightLoop        一个完整黑夜的行动顺序
DayLoop          一个完整白天的行动顺序
*ReactionLoop    猎人、警徽和遗言等死亡后反应
GameRunner       交替调用昼夜 loop、参与者和记录层
Participant      LLM、真人和脚本玩家的统一接口
TaskAgent        按角色读取最小任务说明，按需读取本轮对话并生成一次合法行动
```

推荐阅读顺序：

1. `werewolf_game/core/loops/night_loop.py`
2. `werewolf_game/core/loops/day_loop.py`
3. `werewolf_game/core/engine.py`
4. `werewolf_game/core/runner.py`
5. `werewolf_game/agents/participants/`

`GameEngine` 不调用模型，也不判断策略。参与者只能提交结构化行动，最后由引擎校验。

## 当前规则

默认入口使用 12 人局：狼人×4、平民×4、预言家、女巫、守卫、猎人。
`create_rules_for_player_count(...)` 支持 7–12 人标准预设，也可以通过
`optional_roles` 替换平民身份。

- 狼人夜间先私聊，再共同投票刀人；允许自刀和刀口做局。
- 狼人胜利条件是屠边：全部神职或全部平民死亡；狼人全部死亡则好人胜利。
- 白天公开发言最多 200 字；狼人私聊最多 30 字；发言必须包含中文，但可包含英文、数字和
  玩家编号。
- 平民可以编造游戏内信息，但不会改变引擎事实。
- 每夜开始向所有存活玩家同步当前公开状态。
- 预言家查验、女巫药水和狼人私聊只对规则允许的玩家可见。
- 守卫默认不能连续守同一人；守护挡狼刀但不能挡毒药。
- 夜间死亡者没有遗言；前三名白天被放逐者各有一次公开遗言。
- 首日死讯公布前进行警长竞选；首夜死亡者仍可上警和投票，平票两次后无警长。
- 警长拥有 1.5 票，可以选择日间发言顺序方向；死亡时可传递或撕毁警徽。
- 猎人、白痴等角色的能力和连锁反应由引擎处理。

规则配置在 `werewolf_game/core/rules.py`，不依赖玩家 prompt。

## Prompt 与角色规则

固定 prompt 位于 `werewolf_game/prompts/`：

- `player_system.txt`：基础 LLM 玩家系统提示；
- `player_turn_instruction.txt`：单次行动指令；
- `public_narrator_system.txt`：可选公开记录员提示；
- `roles/<role>/base.md`：该角色的固定游戏规则。
- `roles/<role>/task.md`：该角色最小可运行 Task-Agent 的初始任务说明。

当前每个角色都有一个最小 Task-Agent：它读取自己的 `base.md`、`task.md` 和合法行动
包；本轮对话不会默认拼入上下文，只能通过一次受限的
`read_current_round_dialogue` 工具按需读取。Season 2 会将这一最小实现复制为各角色独立的
base 节点，再由 Meta-Agent 和外部 Pi coding agent 产生候选版本；原始文件不会被候选覆盖。
玩家的中文发言还必须保持自然、连贯、像真实对局参与者；不能因为使用模型就输出
机械模板、元话语或内部调试信息。

每次行动时，Task-Agent 会把当前阶段 `allowed_actions` 动态展开为“本次行动的最终 JSON
契约”：每种允许的 `kind` 都有合法 JSON 示例，目标行动列出本次可选 `target_id`。协议字段
本身使用英文（如 `"kind":"day_vote"`）是合法且必要的；发言文本可包含英文或 `p3` 等
编号。模型出现不可解析 JSON 或行动校验失败时，首次尝试之外最多重新生成两次；纠错调用
不额外得到对话工具，以保持每次行动最多一次工具调用的预算。

## 运行示例

只运行确定性脚本局，不调用外部模型：

```bash
python3 werewolf_game/examples/run_scripted_game.py
```

运行全 LLM 对局：

```bash
python3 werewolf_game/examples/run_llm_game.py
```

运行 Season 2 最低基线的 10 局 base（不更新策略）：

```bash
python3 werewolf_game/examples/run_base_games.py
```

该入口默认写入 `werewolf_game/records/season2/base-v2-schema-contract/`。先前的
`werewolf_game/records/season2/base/` 是没有阶段 JSON 契约、且曾禁止英文字符的旧 smoke 条件，保留审计，
但不能与新条件的 base 胜率混合。

## Season 2 进化

全部超参数集中在 [`werewolf_game/configs/season2.json`](werewolf_game/configs/season2.json)，包括进化上限、异步角色
调度、每节点评测局数、筛选阈值、Meta-Agent 工具预算和 Pi 启动方式。
配置会在 archive 初始化时冻结到 manifest；之后若修改配置，必须使用新的 `archive_root`，
避免不同实验条件静默混用。

只初始化各角色 base 和进化树，不调用模型：

```bash
python3 werewolf_game/examples/run_season2_evolution.py --initialize-only
```

推进一个异步步骤：

```bash
python3 werewolf_game/examples/run_season2_evolution.py --steps 1
```

运行过程中会即时输出节点选择、评测进度、每局完成、汇总、Meta-Agent 和 Pi 分支状态。
评测结果按 `gameN.json` 原子落盘；如果进程中断，重新执行同一命令会跳过已有结果，
从未完成的局继续。残留的半局回放会被识别并安全重跑；已经有完整回放但尚未写入结果
的局会直接从回放恢复，不重复调用模型。

一个 `step` 表示一次成功进化：只有至少一个新子节点被评为 `retained` 才结束。
如果生成或选中的候选被舍弃，当前 step 会继续处理其他可用节点；只有所有可用节点和预算
都耗尽时，step 才会以未产生进化结束。

在一个 step 内，具体会做以下工作：

- 选中保留节点：Meta-Agent 在只读资料沙箱中执行按需工具 loop，产生诊断；Pi 在候选工作区
  修改 `task_agent.py` / `task.md`，按 `children_per_expansion` 产生多个独立分支；每个分支
  通过文件白名单、AST、编译和接口检查后形成待评估子节点，并在当前 step 内开始评测；
- 选中待评估节点：运行配置指定数量的真实对局，其他角色从各自保留候选池随机抽取节点，
  再按外置阈值标记为保留或舍弃；第一个被保留的新节点会结束当前 step。

评测按单局配置复用：只有候选角色组合、游戏规则、随机种子和模型配置等评测签名完全一致，
且历史记录的全局 `game_index` 与当前目标局一致时，才可以从已有完整回放恢复。复用保持
局号对齐：如果相同配置出现在历史第 1、3、4 局，就只复用当前第 1、3、4 局，不会把历史
第 1、2、3 局顺序填入当前的空位；不会因为两个节点都是 base 就整批复用 20 局。

每次 retained 节点扩展会按 `children_per_expansion` 无放回抽取不同的完整回放，并分别
直接提供给对应的 Meta-Agent 调用；其他回放不会默认注入，只能通过 `read_game_replay`
工具按需分页读取。若 retained 节点尚无评测记录，进化步骤会先补做一次基线评测再启动
这批诊断。

初版外部狼人杀 Skill 卡片统一平铺在 `werewolf_game/skills/` 下。Meta-Agent 先通过
`list_skills` 查看名称和短描述，再通过 `read_skill` 按名称读取某一张卡片的完整内容；
它们不会直接进入局内 Task-Agent 上下文。

默认记录结构为：

```text
werewolf_game/records/season2/evolution/
  archive/       节点 manifest、完整代码快照、祖先 patch
  operations/    选择、Meta 诊断、Pi 调用和生成结果
  evaluations/   对手分配、完整游戏记录、逐局结果和筛选汇总
```

Pi 默认从项目内的 `werewolf_game/runtime/pi` 调用 `pi-test.sh`，并使用 `bwrap` 将可写范围限制为候选目录；
当前配置使用 `youdao/gpt-5.5`。Pi 的 provider 配置不进入 Git，而是放在本机忽略目录
`.local/pi/models.json`，其中网关地址可以按部署环境填写，API key 通过 `$OPENAI_API_KEY`
从环境变量读取。`werewolf_game/configs/season2.json` 的 `pi.agent_dir` 指向该目录，运行时会把它只读挂载
到 Pi 沙箱中的 `/pi-agent`。该 Pi 源码要求 Node 22.19 以上且需先安装依赖；也可以在配置中替换为
已构建的 Pi 可执行文件。当前 `pi.node_bin` 指向 `wolf` conda 环境中的 Node 22；启动脚本通过
Node 的 `tsx` loader 运行源码，不使用会创建 Unix socket 的 `tsx` CLI，适配受限运行环境。Meta-Agent
仍使用 `meta` 模型 profile，局内评测使用 `task` profile。

首次在一台新机器上运行时，创建本地配置目录和文件（内容示例见历史连通性测试所用的 provider
结构；不要把真实密钥写入文件）：

```bash
mkdir -p .local/pi
# 在 .local/pi/models.json 中配置 provider、baseUrl 和模型；apiKey 使用 "$OPENAI_API_KEY"
```

运行一名真人和其余 LLM 玩家：

```bash
python3 werewolf_game/examples/run_human_game.py
```

人工身份默认随机；指定身份示例：

```bash
WEREWOLF_HUMAN_ROLE=wolf python3 werewolf_game/examples/run_human_game.py
```

常用环境变量：

- `WEREWOLF_PLAYER_COUNT`：7–12，默认 12；
- `WEREWOLF_OPTIONAL_ROLES`：用逗号分隔的可选角色；
- `WEREWOLF_GAME_SEED`、`WEREWOLF_GAME_ID`：复现发牌和对局；
- `WEREWOLF_RECORD_DIRECTORY`：记录目录，默认 `records`；
- `WEREWOLF_ROUND`、`WEREWOLF_GAME_INDEX`：指定记录槽位；省略时自动寻找下一个；
- `WEREWOLF_DECISION_TIMEOUT_SECONDS`：单次行动超时；
- `WEREWOLF_MODEL_MAX_IN_FLIGHT`：模型请求并发上限；
- `WEREWOLF_TASK_MAX_DECISION_RETRIES`：结构化输出或行动校验失败后，额外生成次数，默认 2；
- `WEREWOLF_TASK_MAX_TOOL_CALLS`：每次行动最多读取本轮对话的次数，最低基线默认 1；
- `WEREWOLF_TASK_MAX_TOOL_RESULT_TOKENS`：单次工具返回的保守长度上限，默认 800；当前
  实现用 Unicode 字符作保守截断，不依赖模型专用 tokenizer；
- `WEREWOLF_BASE_GAME_COUNT`：`run_base_games.py` 的对局数，默认 10；
- `WEREWOLF_HUMAN_PLAYER`、`WEREWOLF_HUMAN_ROLE`：人工玩家设置。

## 模型配置

API 地址和密钥由 `.env` / 环境变量提供，不写入记录。当前基础 LLM 玩家使用 Task
profile；默认模型为 `qwen3.8-flash`，关闭 reasoning/thinking。模型客户端同时
兼容 Responses、Messages 和 Chat Completions 协议。

## 记录与复盘

每局由 `FileGameRecordStore` 同时保存：

- 管理员审计 JSON 与 Markdown；
- 面向真人的公开 JSON 与 Markdown；
- 引擎事件、运行异常、最终状态和可用的模型 token 汇总。

记录运行时才创建，不在仓库中预置历史 `records/`。`archives/` 下的 Season 0/1 内容
是只读历史，不参与当前游戏。

## 测试

```bash
python3 -m unittest discover -s tests -v
```

主目录的测试只覆盖游戏引擎、昼夜流程、参与者、模型客户端和记录边界。
测试还覆盖 Season 2 archive、调度、Meta 工具、候选动态加载、完整假模型对局和评测状态机。

## 设计文档

- [`SEASON2.md`](SEASON2.md)：Task-Agent 进化需求基线。资源在进化开始前由外部准备并
  冻结；Task-Agent 对局中不能使用 Web Search 或自行扩充资源；
- [`SEASON3.md`](SEASON3.md)：Meta-Agent 方向文档，保持原样，不由当前游戏运行时加载。

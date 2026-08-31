# Python 狼人杀 Agent 游戏框架

当前主目录只保留游戏本身：规则由 Python 引擎裁决，参与者可以是 LLM、真人或脚本，
不需要“上帝 Agent”。策略进化、赛季实验和历史对局不在主运行时中执行。

## 运行结构

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

1. `werewolf_game/loops/night_loop.py`
2. `werewolf_game/loops/day_loop.py`
3. `werewolf_game/engine.py`
4. `werewolf_game/runner.py`
5. `werewolf_game/participants/`

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

规则配置在 `werewolf_game/rules.py`，不依赖玩家 prompt。

## Prompt 与角色规则

固定 prompt 位于 `werewolf_game/prompts/`：

- `player_system.txt`：基础 LLM 玩家系统提示；
- `player_turn_instruction.txt`：单次行动指令；
- `public_narrator_system.txt`：可选公开记录员提示；
- `roles/<role>/base.md`：该角色的固定游戏规则。
- `roles/<role>/task.md`：该角色最小可运行 Task-Agent 的初始任务说明。

当前每个角色都有一个最小 Task-Agent：它读取自己的 `base.md`、`task.md` 和合法行动
包；本轮对话不会默认拼入上下文，只能通过一次受限的
`read_current_round_dialogue` 工具按需读取。尚未加入策略进化、Meta-Agent 或外部资源
收集。
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
python3 examples/run_scripted_game.py
```

运行全 LLM 对局：

```bash
python3 examples/run_llm_game.py
```

运行 Season 2 最低基线的 10 局 base（不更新策略）：

```bash
python3 examples/run_base_games.py
```

该入口默认写入 `records/season2/base-v2-schema-contract/`。先前的
`records/season2/base/` 是没有阶段 JSON 契约、且曾禁止英文字符的旧 smoke 条件，保留审计，
但不能与新条件的 base 胜率混合。

运行一名真人和其余 LLM 玩家：

```bash
python3 examples/run_human_game.py
```

人工身份默认随机；指定身份示例：

```bash
WEREWOLF_HUMAN_ROLE=wolf python3 examples/run_human_game.py
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
profile；默认模型为 `deepseek-v4-flash`，关闭 reasoning/thinking。模型客户端同时
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

## 设计文档

- [`SEASON2.md`](SEASON2.md)：Task-Agent 进化需求基线。资源在进化开始前由外部准备并
  冻结；Task-Agent 对局中不能使用 Web Search 或自行扩充资源；
- [`SEASON3.md`](SEASON3.md)：Meta-Agent 方向文档，保持原样，不由当前游戏运行时加载。

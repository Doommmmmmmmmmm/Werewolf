# 狼人杀游戏 · 怎么玩

规则由 Python 引擎裁决，玩家可以是真人、LLM 或脚本。本目录是游戏本体，不依赖
自进化实验，可独立运行。

## 1. 环境准备

只需要 **Python 3.11 或更高版本**，没有第三方依赖（模型调用用标准库 `urllib`
手写，HTTP 服务用 `asyncio`）。

```bash
python3 --version   # 需要 >= 3.11
```

## 2. 配置模型

想让 LLM 当玩家就要配模型地址和密钥。三种方式任选，**优先级从高到低**：

### 方式一：环境变量

```bash
export WEREWOLF_MODEL_BASE_URL="https://your-gateway.example.com/v1"
export WEREWOLF_MODEL_API_KEY="your-key"
export WEREWOLF_TASK_MODEL="qwen3.8-flash"
```

### 方式二：配置文件（推荐）

复制示例并填写：

```bash
cp werewolf.toml.example werewolf.toml
```

```toml
[model]
base_url = "https://your-gateway.example.com/v1"
api_key = "your-key"
api_mode = "responses"      # responses / messages / chat_completions
task_model = "qwen3.8-flash"
```

查找顺序：`WEREWOLF_CONFIG` 指定的路径 → `./werewolf.toml` → 仓库根的
`werewolf.toml` → `~/.config/werewolf/config.toml`。

### 方式三：`.env`

```bash
WEREWOLF_MODEL_BASE_URL=https://your-gateway.example.com/v1
WEREWOLF_MODEL_API_KEY=your-key
```

从当前目录逐级向上查找，因此**在任意子目录启动都能读到**。也兼容旧名
`OPENAI_BASE_URL` / `OPENAI_API_KEY`。

### 配置没生效？

```bash
python3 -c "from game.config import describe_sources; print(describe_sources())"
```

会打印实际使用的 `.env` 和配置文件路径。`werewolf.toml` 和 `.env` 都已被
`.gitignore` 排除，不会把密钥提交进版本库。

## 3. 四种玩法

全部命令在仓库根目录执行。

### 浏览器里玩（推荐）

```bash
python3 game/examples/run_web_demo.py
```

打开 http://127.0.0.1:8192 。你占一个席位，其余 11 席由已归档赛季的进化 Agent
驱动。界面提供：

- **随机模式**（引擎发牌）和**自选模式**（指定席位与身份）
- **对手强度**四挡：按评测胜率、两套强度模型、以及进化前基线做对照
- 座位状态、警徽、流式实况、私密信息面板（身份、狼队友、查验记录、药水）
- 暂停/继续、放弃重开、死亡后上帝视角、结束后完整复盘

可用环境变量：

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `WEREWOLF_WEB_HOST` | `127.0.0.1` | 监听地址 |
| `WEREWOLF_WEB_PORT` | `8192` | 端口 |
| `WEREWOLF_WEB_RECORD_DIRECTORY` | `game/records/web` | 对局记录目录 |
| `WEREWOLF_ARCHIVE_DIRECTORY` | 自动发现 | 进化节点归档目录 |

**安全提示**：服务器默认只绑 `127.0.0.1` 且**没有任何认证**。绑到对外地址
（`WEREWOLF_WEB_HOST=0.0.0.0`）意味着任何能访问该端口的人都可以抢占席位并读到
该席位的私密身份。远程访问建议用 SSH 隧道：

```bash
ssh -L 8192:127.0.0.1:8192 user@host
```

### 命令行里玩

```bash
python3 game/examples/run_human_game.py
```

在终端输入行动，支持中文简写：`发言 三号的逻辑有问题`、`投票 p7`、`刀 p10`、
`pass`，也接受完整 JSON。指定身份：

```bash
WEREWOLF_HUMAN_ROLE=wolf python3 game/examples/run_human_game.py
```

### 全 LLM 对局（观战）

```bash
python3 game/examples/run_llm_game.py
```

12 个席位都是 LLM，你只看。单局约 13–20 分钟。

### 不调模型的脚本局

```bash
python3 game/examples/run_scripted_game.py
```

确定性脚本玩家，秒级完成、零成本。用来验证安装是否正常。

## 4. 规则速览

默认 **12 人局**：狼人 ×4、平民 ×4、预言家、女巫、守卫、猎人。
支持 7–12 人（`WEREWOLF_PLAYER_COUNT`）。

- **狼人胜利**：屠边——杀光所有神职或所有平民；狼人全死则好人胜
- **夜间顺序**：守卫 → 狼人 → 预言家 → 女巫
- **狼队**：先私聊（每人 ≤30 字）再共同投票刀人，**允许自刀做局**
- **白天**：每人发言 ≤200 字且须含中文；平民可以编造信息，但不改变引擎事实
- **警长**：首夜后竞选，**1.5 票权**，可决定发言方向；平票 PK 一次仍平则无警长；
  死亡时可传递或撕毁警徽
- **遗言**：仅前 **3** 名白天被放逐者有遗言；**夜间死亡没有遗言**
- **死亡不翻牌**（`reveal_role_on_death = False`）；**终局公布全部身份**
- **守卫**：可自守，**不可连续守同一人**；守护挡狼刀但挡不住毒药

规则定义在 `game/core/rules.py`，与玩家 prompt 无关。

## 5. 对局记录

每局同时生成四份文件，默认在 `game/records/`：

| 文件 | 内容 |
| --- | --- |
| `log/full-gameN.json` | 管理员审计记录（含发牌、狼聊、私密行动） |
| `full/gameN.md` | 审计记录的可读复盘 |
| `log/public-gameN.json` | 公开记录（剥离发牌、私密信息和随机种子） |
| `public/gameN.md` | 公开复盘，可直接给别人看 |

Web demo 的记录在 `game/records/web/`。`records/` 已被 `.gitignore` 排除。

## 6. 目录结构

```text
game/
├── core/          规则引擎：状态、可见性、行动校验、昼夜流程、胜负裁决
│   └── loops/     一个完整黑夜/白天，以及猎人、警徽、遗言等死亡后反应
├── agents/        席位抽象（真人/LLM/脚本）、模型客户端、基线 Task-Agent
├── recording/     审计记录、公开记录投影、复盘渲染
├── prompts/       全局与角色 prompt（角色规则 base.md + 任务说明 task.md）
├── web/           零依赖 asyncio HTTP + SSE 服务器与单文件前端
├── examples/      四个入口脚本
└── config.py      模型接入配置解析
```

`GameEngine` 不调用模型、不判断策略。玩家只能提交结构化行动，最终由引擎校验。

推荐阅读顺序：`core/loops/night_loop.py` → `core/loops/day_loop.py` →
`core/engine.py` → `core/runner.py` → `agents/participants/`。

## 7. 测试

```bash
python3 -m unittest discover -s tests -v
```

覆盖引擎、昼夜流程、参与者、模型客户端、记录边界、配置解析和 Web demo。

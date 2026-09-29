# Python 狼人杀 Agent 框架

规则由 Python 引擎裁决，玩家可以是真人、LLM 或脚本，不需要"上帝 Agent"。
项目分成两套独立的东西：**游戏本体**稳定冻结，**自进化实验**按赛季迭代。

```text
game/                       游戏本体，可独立运行
archives/season2-.../       已结束赛季的冻结产物（只读）
tests/                      当前测试
SEASON2.md / SEASON3.md     实验设计文档
```

## 想玩游戏

看 **[`game/README.md`](game/README.md)** —— 环境准备、模型 URL 和 key 怎么配、
四种玩法、规则速览、记录在哪看。

最快上手：

```bash
# 1. 不调模型，验证安装（秒级完成）
python3 game/examples/run_scripted_game.py

# 2. 配好模型后，浏览器里对战进化 Agent
cp werewolf.toml.example werewolf.toml   # 填入 base_url 和 api_key
python3 game/examples/run_web_demo.py    # 打开 http://127.0.0.1:8192
```

游戏本体只需 Python 3.11+，**没有第三方依赖**。

## 想了解自进化实验

看 **[`archives/season2-20260923/README.md`](archives/season2-20260923/README.md)**
—— Season 2 的结果、怎么读归档节点、怎么复现。

一句话结果：154 个节点、2358 局、约 1.87B token，各角色都产出了更强的点估计，
但**没有任何候选相对 base 达到统计显著**。

设计与结论文档：

- **[`SEASON2.md`](SEASON2.md)** —— Task-Agent 进化的需求基线与安全边界
- **[`SEASON3.md`](SEASON3.md)** —— 下一赛季方向：把 Meta-Agent 本身纳入进化范围，
  以及从 Season 2 实测得到的三条关键输入（下一轮起点、单向棘轮、自引用回路）

## 两套代码的关系

依赖是单向的：**进化实验依赖游戏，游戏不依赖进化实验**。

- 游戏提供规则引擎、席位抽象和一个基线 Task-Agent；
- 进化实验把那个基线复制一份当种子，自己迭代出新版本，存进赛季 archive；
- Web demo 通过只读仓库加载归档节点当对手，没有归档时游戏仍可开局。

规则内核 `game/core/` 被赛季 archive 以哈希冻结（覆盖除 `runner.py` 外的 14 个
文件）。改动它会使已归档赛季的胜率数据不再与当前引擎可比，因此需要开新赛季而不是
改旧的。

## 测试

```bash
python3 -m unittest discover -s tests -v
```

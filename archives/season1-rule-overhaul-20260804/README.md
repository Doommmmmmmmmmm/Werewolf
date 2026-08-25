# Season 1 归档（规则重构后）

归档时间：2026-08-25。

本目录保存规则重构后的 Season 1 实验资料。项目根目录中的活动代码和记录没有被移动、删除或覆盖。

## 记录范围

- `records/round0`–`records/round9`：10 个训练 round，每个 round 10 局；包含公开记录、全盘记录、结构化日志、角色复盘和 skill 输入/输出。
- `records/skill-evaluation-*`：冻结 skill 对抗测试。
- `records/codex-*`、`records/interactive`、`records/tactic-selfkill-vs-timid-5`：Season 1 期间的人工和战术实验。
- `skills/roundN`：从对应训练记录复制的 skill 版本；`skills/active-round9` 是 Season 1 末轮输出 skill 的快照。
- `skills/worktree-at-archive`：归档时活动工作区的角色 prompt 快照，仅作取证，不作为历史 round 的唯一来源。
- `skills/prompt-templates`：归档时使用的玩家、记录员和角色复盘模板，以及角色档案。

## 代码快照

- `code/season1-restored/`：可运行恢复版本；已移除 Season 2 专属的 Harness、Meta-Agent、研究接口、Task-Agent prompt、训练入口和测试，并去掉共享调度器/玩家适配器中的 Harness 接入字段。

恢复版本保留了 Season 1 的规则引擎、昼夜 loop、人工/脚本/单步 LLM 玩家、记录、skill 复盘和冻结 skill 评测能力。未擅自回退与 Season 2 Harness 没有直接耦合的后续改动（例如模型 profile 分流和人工玩家支持）。由于项目没有可用的 Git 提交历史，`season1-restored` 是基于现存源代码的最佳努力恢复，不保证与 2026-08-11 当时的每一行源码完全一致。

归档中的原始对局 JSON 保留生成时的内部请求和游戏 ID，以维持回放可审计性；目录和文档则使用现行 Season 1 分类。

未复制 `.env`，避免把 API 密钥带入归档；`.env.example` 已保留。

恢复副本验证：在 `code/season1-restored` 中运行 `python3 -m unittest discover -s tests -v`，46 个测试全部通过。

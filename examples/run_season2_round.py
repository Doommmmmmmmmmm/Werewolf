"""运行 Season 2 的 Harness 构造与训练。

流程：

1. Meta-Agent 按角色读取上一轮完整回放，生成并红队筛选 HarnessSpec；
2. 通过显式质量闸门的候选可选择性写入 active；
3. 本轮 10 局只使用 Harness Task-Agent，不运行旧的 strategy.md 复盘器。

默认不自动晋升候选。确认候选评估结果后设置
``WEREWOLF_SEASON2_PROMOTE=true``，再让通过闸门的候选成为下一局 active。
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from werewolf_game import (
    GameEngine,
    GameRoundRunner,
    HarnessArchiveStore,
    HarnessSpec,
    FrozenRoleStrategyStore,
    MetaAgentConfig,
    RoleStrategyStore,
    TaskAgentMetaAgent,
    create_rules_for_player_count,
)
from werewolf_game.llm import ModelClient, ModelRequestCoordinator
from werewolf_game.participants import TaskAgentParticipant
from werewolf_game.research import JsonSearchProvider


def _load_history(season_root: Path, round_index: int) -> list[dict]:
    if round_index < 0:
        return []
    directory = season_root / "training" / f"round{round_index}" / "log"
    records: list[dict] = []
    for path in sorted(directory.glob("full-game*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            records.append(value)
    return records


async def main() -> None:
    round_index = int(os.environ.get("WEREWOLF_SEASON2_ROUND", "0"))
    season_root = Path(
        os.environ.get("WEREWOLF_SEASON2_RECORD_DIRECTORY", "records/seasons/season2")
    ).resolve()
    history_round = int(os.environ.get("WEREWOLF_SEASON2_HISTORY_ROUND", str(round_index - 1)))
    player_count = int(os.environ.get("WEREWOLF_PLAYER_COUNT", "12"))
    optional_roles = tuple(
        role.strip()
        for role in os.environ.get("WEREWOLF_OPTIONAL_ROLES", "guard,hunter,idiot").split(",")
        if role.strip()
    )
    rules = create_rules_for_player_count(player_count, optional_roles)
    baseline_skill = Path(
        os.environ.get(
            "WEREWOLF_SEASON2_BASELINE_SKILL_DIRECTORY",
            "archives/season1-rule-overhaul-20260804/skills/round0/input",
        )
    )
    if not baseline_skill.exists():
        raise FileNotFoundError(
            "Season 2 需要存在的冻结基线目录："
            f"{baseline_skill}；不会静默回退到当前工作区 strategy.md"
        )
    # 只允许对冻结快照中尚未出现的新角色（例如 idiot）使用固定的初始档案，
    # 不允许整个赛季因为路径拼错而读取可能已迭代的工作区策略。
    strategy_store = FrozenRoleStrategyStore(
        baseline_skill,
        fallback_store=RoleStrategyStore(),
    )
    archive = HarnessArchiveStore(season_root)
    request_coordinator = ModelRequestCoordinator(
        max_in_flight=int(os.environ.get("WEREWOLF_MODEL_MAX_IN_FLIGHT", "8"))
    )
    meta_client = ModelClient.from_env(profile="meta")
    task_client = ModelClient.from_env(profile="task")
    archive.ensure_season_manifest(
        {
            "schema_version": 1,
            "season": "season2",
            "name": "Season 2 · Task-Agent Harness",
            "rules": {
                "rule_id": rules.rule_id,
                "player_count": player_count,
                "role_deck": list(rules.role_deck),
                "night_death_last_words": False,
            },
            "task_model": "deepseek-v4-flash",
            "task_reasoning_effort": "none",
            "meta_model": "gpt-5.5",
            "meta_reasoning_effort": "provider_default",
            "task_protocol": task_client.config.protocol,
            "meta_protocol": meta_client.config.protocol,
            "model_max_in_flight": request_coordinator.max_in_flight,
            "replay_batch_size": int(
                os.environ.get("WEREWOLF_SEASON2_REPLAY_BATCH_SIZE", "2")
            ),
            "candidate_count": int(
                os.environ.get("WEREWOLF_SEASON2_CANDIDATE_COUNT", "3")
            ),
            "baseline_skill_directory": str(baseline_skill.resolve()),
            "harness_schema_version": 1,
        }
    )
    research_endpoint = os.environ.get("WEREWOLF_SEASON2_RESEARCH_ENDPOINT", "").strip()
    research_provider = (
        JsonSearchProvider(
            endpoint=research_endpoint,
            api_key=os.environ.get("WEREWOLF_SEASON2_RESEARCH_API_KEY", ""),
            provider_name=os.environ.get("WEREWOLF_SEASON2_RESEARCH_PROVIDER", "json_endpoint"),
        )
        if research_endpoint
        else None
    )
    meta_agent = TaskAgentMetaAgent(
        model_client=meta_client,
        strategy_store=strategy_store,
        archive_store=archive,
        request_coordinator=request_coordinator,
        config=MetaAgentConfig(
            replay_batch_size=int(os.environ.get("WEREWOLF_SEASON2_REPLAY_BATCH_SIZE", "2")),
            candidate_count=int(os.environ.get("WEREWOLF_SEASON2_CANDIDATE_COUNT", "3")),
            auto_promote=False,
        ),
        research_provider=research_provider,
    )

    history = _load_history(season_root, history_round)
    roles = sorted({str(role) for role in rules.role_deck})
    current_harnesses: dict[str, HarnessSpec] = {}
    harness_variants: dict[str, tuple[HarnessSpec, ...]] = {}
    promote = os.environ.get("WEREWOLF_SEASON2_PROMOTE", "false").strip().lower() == "true"
    for role in roles:
        parent = None
        active_path = archive.active_store.active_path(role)
        if active_path.exists():
            parent = archive.load_active(role, strategy_store=strategy_store)
        result = await meta_agent.construct_task_agent(
            round_index=round_index,
            role=role,
            game_records=history,
            parent=parent,
            promote=promote,
        )
        fallback = result.selected if result.selected_passed else (
            parent or HarnessSpec.baseline(role, strategy=strategy_store.profile(role).strategy)
        )
        current_harnesses[role] = fallback
        passed_variants = tuple(
            assessment.candidate
            for assessment in result.candidates
            if assessment.passed
        )
        harness_variants[role] = passed_variants or (fallback,)
        print(
            "Harness",
            role,
            result.selected.harness_id,
            "passed=" + str(result.selected_passed),
            "promoted=" + str(bool(result.promoted_path)),
        )

    record_directory = season_root / "training"
    base_seed = os.environ.get("WEREWOLF_GAME_SEED", "season2-training")
    persona = os.environ.get("WEREWOLF_PLAYER_PERSONA", "")

    def game_factory(current_round: int, game_index: int) -> tuple[GameEngine, dict]:
        players = [
            {"id": f"p{index}", "name": f"Player {index}"}
            for index in range(1, player_count + 1)
        ]
        engine = GameEngine(
            game_id=f"round{current_round}-game{game_index}",
            players=players,
            rules=rules,
            seed=f"{base_seed}-round{current_round}-game{game_index}",
        )
        participants = {
            player["id"]: TaskAgentParticipant(
                player_id=player["id"],
                model_client=task_client,
                persona=persona,
                strategy_store=strategy_store,
                harness_specs=current_harnesses,
                harness_variants=harness_variants,
                request_coordinator=request_coordinator,
            )
            for player in players
        }
        return engine, participants

    report = await GameRoundRunner(
        game_factory=game_factory,
        record_directory=record_directory,
        reviewer=None,
        strategy_store=strategy_store,
        decision_timeout_seconds=float(os.environ.get("WEREWOLF_DECISION_TIMEOUT_SECONDS", "180")),
        game_concurrency=int(os.environ.get("WEREWOLF_GAME_CONCURRENCY", "1")),
        request_coordinator=request_coordinator,
        adaptive_concurrency=os.environ.get("WEREWOLF_ADAPTIVE_CONCURRENCY", "true").lower() != "false",
        resume_completed_games=True,
    ).run_round(round_index)
    print("Season 2 round finished:", report["round_directory"])
    print("Quality:", report["quality"])


if __name__ == "__main__":
    asyncio.run(main())

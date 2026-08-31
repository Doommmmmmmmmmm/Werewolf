"""运行 Season 2 最低 Task-Agent 基线的 10 局对照实验。

每局独立新建玩家和引擎，但复用同一模型客户端与请求闸门；记录写入
``records/season2/base-v2-schema-contract/round0``，不会进行策略更新。

脚本可安全续跑：已经有完整最终记录的 game 槽位会直接从审计记录恢复到汇总，
不会覆盖或重跑；存在未完成记录的槽位则明确报错，避免把一次中断的实验悄悄替换掉。
"""

from __future__ import annotations

import asyncio
from collections import Counter
import json
import os
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from werewolf_game import GameEngine, GameRunner, create_rules_for_player_count
from werewolf_game.llm import ModelClient, ModelRequestCoordinator
from werewolf_game.participants import TaskAgentParticipant
from werewolf_game.records import RoundGameRecordStore


def completed_record_path(record_directory: str, game_index: int) -> Path:
    """返回固定 round0/gameN 审计记录的位置。"""

    return (
        Path(record_directory).resolve()
        / "round0"
        / "log"
        / f"full-game{game_index}.json"
    )


def load_completed_result(
    *, record_directory: str, game_index: int
) -> dict[str, Any] | None:
    """从已完成的审计记录恢复一个批量汇总条目。

    ``GameRunner`` 的实时 report 不会单独落盘，因此这里以 ``ACTION_SUBMITTED``
    数量重建 decision_count；该事件由引擎对每个玩家实际提交（含 fallback）的行动
    统一写入。只接受明确处于 ``finished`` 状态的记录。
    """

    path = completed_record_path(record_directory, game_index)
    if not path.exists():
        return None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"无法读取已有记录 {path}: {error}") from error

    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        raise RuntimeError(f"已有记录缺少 metadata: {path}")
    if metadata.get("round") != 0 or metadata.get("game_index") != game_index:
        raise RuntimeError(f"已有记录的 round/game 编号不匹配: {path}")

    final_snapshot = record.get("final_snapshot")
    final_state = (
        final_snapshot.get("public_state")
        if isinstance(final_snapshot, dict)
        else None
    )
    if not isinstance(final_state, dict) or final_state.get("status") != "finished":
        raise RuntimeError(
            f"{path} 已存在但不是完整结束记录；为保护审计记录，批量脚本不会覆盖它"
        )

    events = record.get("events")
    if not isinstance(events, list):
        events = []
    runner_events = record.get("runner_events")
    if not isinstance(runner_events, list):
        runner_events = []
    errors = [
        {
            key: event[key]
            for key in (
                "player_id",
                "type",
                "message",
                "exception_type",
                "status_code",
                "api_attempts",
                "generic_retries",
                "usage_limit_retries",
            )
            if key in event
        }
        for event in runner_events
        if isinstance(event, dict) and event.get("event_class") == "RUNNER_ERROR"
    ]
    decision_count = sum(
        1
        for event in events
        if isinstance(event, dict) and event.get("type") == "ACTION_SUBMITTED"
    )
    fallback_count = sum(
        1
        for event in runner_events
        if isinstance(event, dict) and event.get("type") == "FALLBACK_ACTION"
    )
    game_name = f"game{game_index}"
    round_directory = path.parents[1]
    rule_set = metadata.get("rule_set")
    return {
        "game_index": game_index,
        "winner": final_state.get("winner"),
        "round": final_state.get("round"),
        "decision_count": decision_count,
        "fallback_count": fallback_count,
        "fallback_rate": fallback_count / decision_count if decision_count else 0.0,
        "errors": errors,
        "model_token_usage": record.get("model_token_usage"),
        "record_path": str(path),
        "record_markdown_path": str(round_directory / "full" / f"{game_name}.md"),
        "public_record_path": str(round_directory / "log" / f"public-{game_name}.json"),
        "public_record_markdown_path": str(
            round_directory / "public" / f"{game_name}.md"
        ),
        "resumed_from_completed_record": True,
        "record_game_id": metadata.get("game_id"),
        "record_seed": metadata.get("seed"),
        "rule_id": rule_set.get("rule_id") if isinstance(rule_set, dict) else None,
    }


def aggregate_token_usage(results: list[dict[str, Any]]) -> dict[str, Any]:
    """仅汇总服务端实际报告的用量，并保留覆盖范围而不做臆测。"""

    fields = (
        "participant_count",
        "successful_response_count",
        "api_attempt_count",
        "reported_usage_response_count",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "unreported_api_attempt_count",
    )
    totals = {field: 0 for field in fields}
    availability_counts: Counter[str] = Counter()
    reported_games = 0
    for result in results:
        usage = result.get("model_token_usage")
        if not isinstance(usage, dict):
            availability_counts["unavailable"] += 1
            continue
        reported_games += 1
        availability_counts[str(usage.get("availability") or "unknown")] += 1
        for field in fields:
            value = usage.get(field)
            if isinstance(value, bool):
                continue
            try:
                normalized = int(value)
            except (TypeError, ValueError):
                continue
            if normalized >= 0:
                totals[field] += normalized
    return {
        "games_with_usage": reported_games,
        "availability_by_game": dict(availability_counts),
        "totals": totals,
    }


def build_summary(
    *,
    results: list[dict[str, Any]],
    player_count: int,
    optional_roles: tuple[str, ...],
    game_count: int,
    max_decision_retries: int,
    max_tool_calls: int,
    max_tool_result_tokens: int,
    max_in_flight: int,
    game_concurrency: int,
    model_client: ModelClient,
) -> dict[str, Any]:
    """构造可审计、且不包含 API 密钥或地址的实验汇总。"""

    config = model_client.config
    complete_results = [
        result
        for result in results
        if result.get("winner") in {"wolf", "village"}
    ]
    error_type_counts = Counter(
        str(error.get("type") or "unknown")
        for result in results
        for error in result.get("errors", [])
        if isinstance(error, dict)
    )
    error_message_counts = Counter(
        f"{error.get('type') or 'unknown'}: {error.get('message') or ''}"
        for result in results
        for error in result.get("errors", [])
        if isinstance(error, dict)
    )
    return {
        "experiment": "season2-base-v2-schema-contract",
        "baseline": {
            "player_count": player_count,
            "optional_roles": list(optional_roles),
            "conversation_mode": "new_session_per_decision",
            "max_decision_retries": max_decision_retries,
            "tool": "read_current_round_dialogue",
            "max_tool_calls_per_decision": max_tool_calls,
            "max_tool_result_tokens": max_tool_result_tokens,
            "game_count": game_count,
            "round": 0,
            "model": {
                "name": config.model,
                "protocol": config.protocol,
                "reasoning_effort": config.reasoning_effort,
                "enable_thinking": config.enable_thinking,
                "max_output_tokens": config.max_output_tokens,
            },
            "request_scheduling": {
                "global_max_in_flight": max_in_flight,
                "game_concurrency": game_concurrency,
            },
        },
        "results": sorted(results, key=lambda result: int(result["game_index"])),
        "completed_game_count": len(complete_results),
        "winner_counts": dict(
            Counter(result.get("winner") for result in complete_results)
        ),
        "rounds": [
            {"game_index": result["game_index"], "round": result.get("round")}
            for result in sorted(results, key=lambda result: int(result["game_index"]))
        ],
        "fallback_count": sum(
            int(result.get("fallback_count") or 0) for result in results
        ),
        "error_type_counts": dict(error_type_counts),
        "error_message_counts": dict(error_message_counts),
        "model_token_usage": aggregate_token_usage(results),
    }


async def run_one(
    *,
    game_index: int,
    record_directory: str,
    base_seed: str,
    player_count: int,
    optional_roles: tuple[str, ...],
    model_client: ModelClient,
    request_coordinator: ModelRequestCoordinator,
    decision_timeout_seconds: float,
    max_decision_retries: int,
    max_tool_calls: int,
    max_tool_result_tokens: int,
    quiet: bool,
) -> dict[str, Any]:
    rules = create_rules_for_player_count(player_count, optional_roles)
    players = [
        {"id": f"p{index}", "name": f"Player {index}"}
        for index in range(1, player_count + 1)
    ]
    participants = {
        player["id"]: TaskAgentParticipant(
            player_id=player["id"],
            model_client=model_client,
            request_coordinator=request_coordinator,
            max_decision_retries=max_decision_retries,
            max_tool_calls_per_decision=max_tool_calls,
            max_tool_result_tokens=max_tool_result_tokens,
        )
        for player in players
    }
    engine = GameEngine(
        game_id=f"round0-game{game_index}",
        players=players,
        rules=rules,
        seed=f"{base_seed}-game{game_index}",
    )
    record_store = RoundGameRecordStore(
        record_directory,
        round_index=0,
        game_index=game_index,
    )

    def print_public_events(events: list[dict[str, Any]]) -> None:
        if quiet:
            return
        for event in events:
            print(event["seq"], event["type"], event["payload"])

    report = await GameRunner(
        engine=engine,
        participants=participants,
        record_store=record_store,
        on_public_events=print_public_events,
        decision_timeout_seconds=decision_timeout_seconds,
    ).run()
    state = report["public_state"]
    return {
        "game_index": game_index,
        "winner": state.get("winner"),
        "round": state.get("round"),
        "decision_count": report.get("decision_count"),
        "fallback_count": report.get("fallback_count"),
        "fallback_rate": report.get("fallback_rate"),
        "errors": report.get("errors", []),
        "model_token_usage": report.get("model_token_usage"),
        "record_path": report.get("record_path"),
    }


async def main() -> None:
    record_directory = os.environ.get(
        "WEREWOLF_RECORD_DIRECTORY", "records/season2/base-v2-schema-contract"
    )
    base_seed = os.environ.get("WEREWOLF_GAME_SEED", "season2-base")
    player_count = int(os.environ.get("WEREWOLF_PLAYER_COUNT", "12"))
    optional_roles = tuple(
        role.strip()
        for role in os.environ.get("WEREWOLF_OPTIONAL_ROLES", "").split(",")
        if role.strip()
    )
    game_count = int(os.environ.get("WEREWOLF_BASE_GAME_COUNT", "10"))
    decision_timeout_seconds = float(
        os.environ.get("WEREWOLF_DECISION_TIMEOUT_SECONDS", "180")
    )
    max_in_flight = int(os.environ.get("WEREWOLF_MODEL_MAX_IN_FLIGHT", "4"))
    game_concurrency = max(
        1, int(os.environ.get("WEREWOLF_BASE_GAME_CONCURRENCY", "1"))
    )
    max_decision_retries = max(
        0, int(os.environ.get("WEREWOLF_TASK_MAX_DECISION_RETRIES", "2"))
    )
    max_tool_calls = int(os.environ.get("WEREWOLF_TASK_MAX_TOOL_CALLS", "1"))
    max_tool_result_tokens = int(
        os.environ.get("WEREWOLF_TASK_MAX_TOOL_RESULT_TOKENS", "800")
    )
    quiet = os.environ.get("WEREWOLF_QUIET", "1") == "1"

    model_client = ModelClient.from_env(profile="task")
    request_coordinator = ModelRequestCoordinator(max_in_flight=max_in_flight)
    results: list[dict[str, Any]] = []
    pending_game_indices: list[int] = []
    for game_index in range(game_count):
        completed = load_completed_result(
            record_directory=record_directory, game_index=game_index
        )
        if completed is not None:
            results.append(completed)
            print(
                f"game{game_index}: 已读取完整记录 "
                f"winner={completed.get('winner')} round={completed.get('round')} "
                f"fallback={completed.get('fallback_count')}"
            )
            continue
        # ``load_completed_result`` 会对存在但不完整的文件抛出异常；只有无记录的
        # 槽位才会进入新的运行队列，避免覆盖中断的审计文件。
        pending_game_indices.append(game_index)

    async def run_pending(game_index: int) -> dict[str, Any]:
        try:
            result = await run_one(
                game_index=game_index,
                record_directory=record_directory,
                base_seed=base_seed,
                player_count=player_count,
                optional_roles=optional_roles,
                model_client=model_client,
                request_coordinator=request_coordinator,
                decision_timeout_seconds=decision_timeout_seconds,
                max_decision_retries=max_decision_retries,
                max_tool_calls=max_tool_calls,
                max_tool_result_tokens=max_tool_result_tokens,
                quiet=quiet,
            )
        except Exception as error:
            result = {
                "game_index": game_index,
                "winner": None,
                "error": f"{type(error).__name__}: {error}",
            }
        return result

    if game_concurrency == 1:
        for game_index in pending_game_indices:
            result = await run_pending(game_index)
            results.append(result)
            print(
                f"game{game_index}: winner={result.get('winner')} "
                f"round={result.get('round')} fallback={result.get('fallback_count')}"
            )
    else:
        # 每个 GameRunner 可以并发推进，但全部请求仍共享同一个全局 coordinator，
        # 因而真实 API 并发不会超过 max_in_flight。这样只缩短等待时间，不改变任一
        # 单局的角色、提示词、工具预算或重试策略。
        semaphore = asyncio.Semaphore(game_concurrency)

        async def run_with_slot(game_index: int) -> dict[str, Any]:
            async with semaphore:
                return await run_pending(game_index)

        tasks = [
            asyncio.create_task(run_with_slot(game_index))
            for game_index in pending_game_indices
        ]
        for task in asyncio.as_completed(tasks):
            result = await task
            results.append(result)
            game_index = result["game_index"]
            print(
                f"game{game_index}: winner={result.get('winner')} "
                f"round={result.get('round')} fallback={result.get('fallback_count')}"
            )

    summary = build_summary(
        results=results,
        player_count=player_count,
        optional_roles=optional_roles,
        game_count=game_count,
        max_decision_retries=max_decision_retries,
        max_tool_calls=max_tool_calls,
        max_tool_result_tokens=max_tool_result_tokens,
        max_in_flight=max_in_flight,
        game_concurrency=game_concurrency,
        model_client=model_client,
    )
    summary_path = Path(record_directory).resolve() / "base-summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("Summary:", summary_path)


if __name__ == "__main__":
    asyncio.run(main())

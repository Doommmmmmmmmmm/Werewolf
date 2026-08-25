"""以十局为一个训练 round 的角色策略迭代调度器。"""

from __future__ import annotations

import asyncio
import json
import math
from pathlib import Path
from collections.abc import Callable, Mapping
from typing import Any

from .constants import (
    ROLE_GUARD,
    ROLE_HUNTER,
    ROLE_IDIOT,
    ROLE_SEER,
    ROLE_VILLAGER,
    ROLE_WITCH,
    ROLE_WOLF,
)
from .engine import GameEngine
from .llm.coordinator import ModelRequestCoordinator
from .prompts import RoleStrategyStore
from .records import (
    GAMES_PER_TRAINING_ROUND,
    RoundGameRecordStore,
    write_round_review,
    write_round_review_manifest,
    write_round_log,
)
from .review import RoleStrategyReviewer
from .runner import GameRunner
from .skill_versions import RoundSkillVersionStore


ROLE_REVIEW_ORDER = (
    ROLE_WOLF,
    ROLE_VILLAGER,
    ROLE_SEER,
    ROLE_WITCH,
    ROLE_GUARD,
    ROLE_HUNTER,
    ROLE_IDIOT,
)
GameFactory = Callable[[int, int], tuple[GameEngine, Mapping[str, Any]]]


class _AdaptiveConcurrencyController:
    """按已完成对局的健康度做保守的加性扩容、乘性降载。"""

    def __init__(
        self,
        *,
        minimum_game_concurrency: int,
        initial_game_concurrency: int,
        maximum_game_concurrency: int,
        request_coordinator: ModelRequestCoordinator | None,
        healthy_p95_latency_ms: int,
    ) -> None:
        self.minimum_game_concurrency = minimum_game_concurrency
        self.maximum_game_concurrency = maximum_game_concurrency
        self.game_concurrency = initial_game_concurrency
        self.request_coordinator = request_coordinator
        self.healthy_p95_latency_ms = healthy_p95_latency_ms
        self.clean_games = 0
        self.cooldown_games = 0
        self.maximum_request_concurrency = (
            request_coordinator.max_in_flight if request_coordinator is not None else None
        )
        self.request_concurrency = self.maximum_request_concurrency
        self.last_usage_limit_retries = 0

    async def observe(self, report: dict[str, Any]) -> dict[str, Any]:
        health = (
            self.request_coordinator.health_snapshot()
            if self.request_coordinator is not None
            else {}
        )
        previous_game_concurrency = self.game_concurrency
        previous_request_concurrency = self.request_concurrency
        fallback_count = int(report.get("fallback_count") or 0)
        error_counts = report.get("error_counts") or {}
        participant_errors = int(error_counts.get("participant_error") or 0)
        usage_limit_retries = int(health.get("total_usage_limit_retries") or 0)
        new_usage_limit_retries = max(
            0, usage_limit_retries - self.last_usage_limit_retries
        )
        self.last_usage_limit_retries = usage_limit_retries
        p95_latency_ms = health.get("p95_latency_ms")
        decision_failure = participant_errors > 0
        action = "hold"
        reason = "等待更多健康样本"

        if decision_failure:
            # 乘性降载：避免发生超时时仍继续向上游增加相同级别的压力。
            self.game_concurrency = max(
                self.minimum_game_concurrency,
                math.ceil(self.game_concurrency / 2),
            )
            if self.request_concurrency is not None:
                self.request_concurrency = max(1, math.ceil(self.request_concurrency / 2))
                assert self.request_coordinator is not None
                await self.request_coordinator.set_max_in_flight(self.request_concurrency)
            self.clean_games = 0
            self.cooldown_games = 2
            action = "decrease"
            reason = "出现参与者异常或行动回退"
        elif new_usage_limit_retries:
            # 额度重试成功说明号池在工作，但暂不扩大并发以免再次击穿可用账号。
            self.clean_games = 0
            action = "hold"
            reason = "本批出现额度重试，保持当前压力"
        elif self.cooldown_games:
            self.cooldown_games -= 1
            self.clean_games = 0
            action = "cooldown"
            reason = "降载后的冷却期"
        elif p95_latency_ms is not None and p95_latency_ms > self.healthy_p95_latency_ms:
            self.clean_games = 0
            action = "hold"
            reason = "P95 延迟仍偏高"
        elif fallback_count:
            # 仅有非法动作等非连接类回退时不降 API 并发，但不把它当作健康样本。
            self.clean_games = 0
            action = "hold"
            reason = "出现非连接类回退"
        else:
            self.clean_games += 1
            if self.clean_games >= 2:
                self.clean_games = 0
                next_game = min(
                    self.maximum_game_concurrency, self.game_concurrency + 1
                )
                next_request = self.request_concurrency
                if (
                    next_request is not None
                    and self.maximum_request_concurrency is not None
                ):
                    next_request = min(
                        self.maximum_request_concurrency, next_request + 1
                    )
                if next_game != self.game_concurrency or next_request != self.request_concurrency:
                    self.game_concurrency = next_game
                    self.request_concurrency = next_request
                    if self.request_coordinator is not None and next_request is not None:
                        await self.request_coordinator.set_max_in_flight(next_request)
                    action = "increase"
                    reason = "连续两局无回退且延迟健康"
                else:
                    action = "hold"
                    reason = "已达到并发上限"
            else:
                action = "hold"
                reason = "累计稳定样本"

        return {
            "game_index": report.get("game_index"),
            "action": action,
            "reason": reason,
            "fallback_count": fallback_count,
            "participant_error_count": participant_errors,
            "new_usage_limit_retries": new_usage_limit_retries,
            "previous_game_concurrency": previous_game_concurrency,
            "game_concurrency": self.game_concurrency,
            "previous_request_concurrency": previous_request_concurrency,
            "request_concurrency": self.request_concurrency,
            "cooldown_games_remaining": self.cooldown_games,
            "p95_latency_ms": p95_latency_ms,
            "health": _compact_health(health),
        }


def _compact_health(health: dict[str, Any]) -> dict[str, Any]:
    """控制记录只保存统计值；详细调用列表保留在最终 round 性能日志中。"""

    keys = (
        "max_in_flight",
        "in_flight",
        "queued",
        "completed_requests",
        "total_success_count",
        "total_failed_count",
        "total_timeout_count",
        "total_usage_limit_retries",
        "p50_latency_ms",
        "p95_latency_ms",
        "p95_queue_wait_ms",
        "failure_rate",
        "timeout_rate",
    )
    return {key: health.get(key) for key in keys if key in health}


def _validated_round_index(value: int) -> int:
    try:
        index = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError("round_index 必须是非负整数") from error
    if index < 0 or str(index) != str(value).strip():
        raise ValueError("round_index 必须是非负整数")
    return index


def _roles_from_deck(deck: object) -> set[str]:
    if not isinstance(deck, (list, tuple, set, frozenset)):
        return set()
    return {str(role) for role in deck if str(role) in ROLE_REVIEW_ORDER}


def _roles_from_record(record: dict[str, Any]) -> set[str]:
    deck = record.get("metadata", {}).get("rule_set", {}).get("role_deck", [])
    return _roles_from_deck(deck)


async def review_completed_round(
    *,
    record_directory: str | Path,
    round_index: int,
    reviewer: RoleStrategyReviewer,
    require_complete: bool = False,
) -> dict[str, Any]:
    """在十局记录齐全时执行一次角色复盘；已有 manifest 时不重复更新。

    单局入口可以在每局结束后调用本函数：尚未收齐十局时，默认返回
    ``completed=False``，不会请求复盘模型。批量训练入口则传入
    ``require_complete=True``，将缺少记录视为流程错误。
    """

    round_index = _validated_round_index(round_index)
    root = Path(record_directory).resolve()
    round_directory = RoundGameRecordStore.round_directory_for(root, round_index)
    log_directory = round_directory / "log"
    manifest_path = log_directory / "round-review.json"
    skill_versions = RoundSkillVersionStore(root, round_index=round_index)
    previous_manifest: dict[str, Any] | None = None
    previous_reviews: dict[str, dict[str, Any]] = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise RuntimeError(f"复盘 manifest 无法读取：{manifest_path}") from error
        if not isinstance(manifest, dict):
            raise RuntimeError(f"复盘 manifest 格式错误：{manifest_path}")
        previous_manifest = manifest
        for review in manifest.get("reviews", []):
            if not isinstance(review, dict):
                continue
            role = str(review.get("role") or "")
            if role in ROLE_REVIEW_ORDER:
                previous_reviews[role] = review

    record_paths = [
        log_directory / f"full-game{game_index}.json"
        for game_index in range(GAMES_PER_TRAINING_ROUND)
    ]
    missing = [path.name for path in record_paths if not path.exists()]
    if missing:
        if require_complete:
            raise RuntimeError(
                f"round{round_index} 尚未收齐 10 局完整记录，缺少：{'、'.join(missing)}"
            )
        return {
            "round_index": round_index,
            "completed": False,
            "already_reviewed": False,
            "manifest_path": None,
            "reviews": [],
            "missing_records": missing,
            "skill_version_manifest_path": None,
        }
    full_records = [
        json.loads(path.read_text(encoding="utf-8")) for path in record_paths
    ]
    roles_seen: set[str] = set()
    for record in full_records:
        roles_seen.update(_roles_from_record(record))
    review_roles = [role for role in ROLE_REVIEW_ORDER if role in roles_seen]
    if previous_manifest is not None and all(
        bool(previous_reviews.get(role, {}).get("updated")) for role in review_roles
    ):
        return {
            "round_index": round_index,
            "completed": True,
            "already_reviewed": True,
            "manifest_path": str(manifest_path),
            "reviews": [previous_reviews[role] for role in review_roles],
            "missing_records": [],
            "skill_version_manifest_path": previous_manifest.get(
                "skill_version_manifest_path",
                str(skill_versions.manifest_path)
                if skill_versions.manifest_path.exists()
                else None,
            ),
        }
    skill_versions.ensure_input(
        roles=roles_seen,
        strategy_store=reviewer.strategy_store,
    )

    reviews: list[dict[str, Any]] = []
    for role in review_roles:
        previous = previous_reviews.get(role)
        if previous is not None and bool(previous.get("updated")):
            reviews.append(previous)
            continue
        result = await reviewer.review_role(
            round_index=round_index,
            role=role,
            game_records=full_records,
        )
        paths = write_round_review(
            root,
            round_index=round_index,
            role=role,
            review=result.as_dict(),
            markdown=result.markdown(),
        )
        reviews.append({"role": role, **paths, **result.as_dict()})
    skill_manifest = skill_versions.capture_output(
        roles=roles_seen,
        strategy_store=reviewer.strategy_store,
        review_results={review["role"]: review for review in reviews},
        refresh_failed_review_outputs=previous_manifest is not None,
    )
    written_manifest_path = write_round_review_manifest(
        root,
        round_index=round_index,
        reviews=reviews,
        skill_version_manifest_path=str(skill_versions.manifest_path),
    )
    return {
        "round_index": round_index,
        "completed": True,
        "already_reviewed": False,
        "manifest_path": written_manifest_path,
        "reviews": reviews,
        "missing_records": [],
        "skill_version_manifest_path": str(skill_versions.manifest_path),
        "skill_versions": skill_manifest,
    }


class GameRoundRunner:
    """连续运行 10 局，并在结尾按角色生成复盘和策略更新。"""

    def __init__(
        self,
        *,
        game_factory: GameFactory,
        record_directory: str | Path = "records",
        reviewer: RoleStrategyReviewer | None = None,
        strategy_store: RoleStrategyStore | None = None,
        decision_timeout_seconds: float = 60.0,
        game_concurrency: int = 1,
        request_coordinator: ModelRequestCoordinator | None = None,
        adaptive_concurrency: bool = False,
        initial_game_concurrency: int | None = None,
        min_game_concurrency: int = 1,
        healthy_p95_latency_ms: int = 45000,
        max_fallback_rate: float = 0.01,
        max_fallbacks_per_game: int = 3,
        allow_degraded_review: bool = False,
        resume_completed_games: bool = False,
    ) -> None:
        self.game_factory = game_factory
        self.record_directory = Path(record_directory).resolve()
        self.reviewer = reviewer
        self.strategy_store = reviewer.strategy_store if reviewer is not None else (
            strategy_store or RoleStrategyStore()
        )
        self.decision_timeout_seconds = decision_timeout_seconds
        self.game_concurrency = self._validated_game_concurrency(game_concurrency)
        self.request_coordinator = request_coordinator
        self.adaptive_concurrency = bool(adaptive_concurrency)
        self.min_game_concurrency = self._validated_game_concurrency(
            min_game_concurrency
        )
        if self.min_game_concurrency > self.game_concurrency:
            raise ValueError("min_game_concurrency 不能大于 game_concurrency")
        default_initial = min(2, self.game_concurrency)
        self.initial_game_concurrency = self._validated_game_concurrency(
            default_initial
            if initial_game_concurrency is None
            else initial_game_concurrency
        )
        if not (
            self.min_game_concurrency
            <= self.initial_game_concurrency
            <= self.game_concurrency
        ):
            raise ValueError("initial_game_concurrency 必须位于最小值与上限之间")
        self.healthy_p95_latency_ms = self._positive_int(
            healthy_p95_latency_ms, "healthy_p95_latency_ms"
        )
        self.max_fallback_rate = self._validated_rate(max_fallback_rate)
        self.max_fallbacks_per_game = self._nonnegative_int(
            max_fallbacks_per_game, "max_fallbacks_per_game"
        )
        self.allow_degraded_review = bool(allow_degraded_review)
        self.resume_completed_games = bool(resume_completed_games)

    async def run_round(self, round_index: int) -> dict[str, Any]:
        """运行 ``game0`` 至 ``game9``，完成后才触发一次角色复盘。"""

        round_index = self._validated_round_index(round_index)
        reports: list[dict[str, Any]] = []
        round_roles: set[str] = set()
        skill_versions = RoundSkillVersionStore(
            self.record_directory, round_index=round_index
        )

        scheduled_games: list[tuple[int, GameEngine, Mapping[str, Any]]] = []
        for game_index in range(GAMES_PER_TRAINING_ROUND):
            completed_game = self._existing_completed_game(round_index, game_index)
            if completed_game is not None:
                if not self.resume_completed_games:
                    raise FileExistsError(
                        f"round{round_index}/game{game_index} 已完成；"
                        "如需复用已有记录，请设置 resume_completed_games=True"
                    )
                report, roles = completed_game
                reports.append(report)
                round_roles.update(roles)
                continue
            engine, participants = self.game_factory(round_index, game_index)
            round_roles.update(_roles_from_deck(engine.rules.role_deck))
            scheduled_games.append((game_index, engine, participants))

        skill_versions.ensure_input(
            roles=round_roles,
            strategy_store=self.strategy_store,
        )

        async def run_game(
            game_index: int, engine: GameEngine, participants: Mapping[str, Any]
        ) -> dict[str, Any]:
            store = RoundGameRecordStore(
                self.record_directory,
                round_index=round_index,
                game_index=game_index,
            )
            report = await GameRunner(
                engine=engine,
                participants=participants,
                record_store=store,
                decision_timeout_seconds=self.decision_timeout_seconds,
            ).run()
            record_path = report.get("record_path")
            if not record_path:
                raise RuntimeError(f"round{round_index}/game{game_index} 没有生成完整记录")
            return {"game_index": game_index, **report}

        control_events: list[dict[str, Any]] = []
        if self.adaptive_concurrency and scheduled_games:
            reports.extend(
                await self._run_games_adaptively(
                    scheduled_games, run_game, control_events
                )
            )
        elif self.game_concurrency == 1:
            for game in scheduled_games:
                reports.append(await run_game(*game))
        else:
            semaphore = asyncio.Semaphore(self.game_concurrency)

            async def run_bounded(
                game: tuple[int, GameEngine, Mapping[str, Any]]
            ) -> dict[str, Any]:
                async with semaphore:
                    return await run_game(*game)

            reports.extend(await asyncio.gather(*(run_bounded(game) for game in scheduled_games)))
        reports.sort(key=lambda report: int(report["game_index"]))

        quality = self._round_quality(round_index, reports)
        review_reports: list[dict[str, Any]] = []
        review_performed = False
        review_manifest_path: str | None = None
        review_skipped_by_quality_gate = False
        if self.reviewer is not None:
            if quality["skill_update_eligible"] or self.allow_degraded_review:
                review_state = await review_completed_round(
                    record_directory=self.record_directory,
                    round_index=round_index,
                    reviewer=self.reviewer,
                    require_complete=True,
                )
                review_reports = list(review_state["reviews"])
                review_performed = not bool(review_state["already_reviewed"])
                review_manifest_path = str(review_state["manifest_path"])
            else:
                review_skipped_by_quality_gate = True
                blocked_reason = "；".join(quality["block_reasons"])
                skill_versions.capture_output(
                    roles=round_roles,
                    strategy_store=self.strategy_store,
                    review_results={
                        role: {
                            "updated": False,
                            "error": f"质量闸门阻止自动复盘：{blocked_reason}",
                        }
                        for role in round_roles
                    },
                )
        else:
            skill_versions.capture_output(
                roles=round_roles,
                strategy_store=self.strategy_store,
            )

        final_health = (
            self.request_coordinator.health_snapshot()
            if self.request_coordinator is not None
            else None
        )
        quality["review_skipped_by_quality_gate"] = review_skipped_by_quality_gate
        quality["review_performed"] = review_performed
        quality["allow_degraded_review"] = self.allow_degraded_review
        quality["request_health"] = _compact_health(final_health or {})
        quality_path = write_round_log(
            self.record_directory,
            round_index=round_index,
            name="round-quality",
            payload=quality,
        )
        performance_path = write_round_log(
            self.record_directory,
            round_index=round_index,
            name="round-performance",
            payload={
                "schema_version": 1,
                "round_index": round_index,
                "adaptive_concurrency": self.adaptive_concurrency,
                "game_concurrency_upper_bound": self.game_concurrency,
                "initial_game_concurrency": self.initial_game_concurrency,
                "decision_timeout_seconds": self.decision_timeout_seconds,
                "controls": control_events,
                "request_health": final_health,
            },
        )
        return {
            "round_index": round_index,
            "round_directory": str(
                RoundGameRecordStore.round_directory_for(self.record_directory, round_index)
            ),
            "games": reports,
            "reviews": review_reports,
            "review_performed": review_performed,
            "review_manifest_path": review_manifest_path,
            "review_skipped_by_quality_gate": review_skipped_by_quality_gate,
            "quality": quality,
            "quality_path": quality_path,
            "performance_path": performance_path,
            "skill_version_manifest_path": str(skill_versions.manifest_path),
        }

    async def run_rounds(self, *, start_round: int = 0, count: int = 1) -> list[dict[str, Any]]:
        """从 round0 起连续执行多个十局训练 batch。"""

        start_round = self._validated_round_index(start_round)
        if int(count) < 1:
            raise ValueError("count 至少需要为 1")
        return [
            await self.run_round(round_index)
            for round_index in range(start_round, start_round + int(count))
        ]

    async def _run_games_adaptively(
        self,
        scheduled_games: list[tuple[int, GameEngine, Mapping[str, Any]]],
        run_game: Callable[[int, GameEngine, Mapping[str, Any]], Any],
        control_events: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """只在游戏边界扩缩容，避免中断已开始的昼夜 loop。"""

        controller = _AdaptiveConcurrencyController(
            minimum_game_concurrency=self.min_game_concurrency,
            initial_game_concurrency=self.initial_game_concurrency,
            maximum_game_concurrency=self.game_concurrency,
            request_coordinator=self.request_coordinator,
            healthy_p95_latency_ms=self.healthy_p95_latency_ms,
        )
        pending = list(scheduled_games)
        active: dict[asyncio.Task[dict[str, Any]], tuple[int, GameEngine, Mapping[str, Any]]] = {}
        reports: list[dict[str, Any]] = []
        while pending or active:
            while pending and len(active) < controller.game_concurrency:
                game = pending.pop(0)
                active[asyncio.create_task(run_game(*game))] = game
            if not active:
                continue
            completed, _ = await asyncio.wait(
                active, return_when=asyncio.FIRST_COMPLETED
            )
            for task in completed:
                active.pop(task)
                report = task.result()
                reports.append(report)
                control_events.append(await controller.observe(report))
        return reports

    def _round_quality(
        self, round_index: int, reports: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """将回退动作显式纳入 skill 更新资格，防止低质量对局污染策略。"""

        total_decisions = sum(int(report.get("decision_count") or 0) for report in reports)
        total_fallbacks = sum(int(report.get("fallback_count") or 0) for report in reports)
        fallback_rate = total_fallbacks / total_decisions if total_decisions else 0.0
        degraded_games: list[dict[str, Any]] = []
        for report in reports:
            fallbacks = int(report.get("fallback_count") or 0)
            if fallbacks > self.max_fallbacks_per_game:
                degraded_games.append(
                    {
                        "game_index": report.get("game_index"),
                        "fallback_count": fallbacks,
                        "decision_count": int(report.get("decision_count") or 0),
                        "reason": (
                            f"单局回退 {fallbacks} 次，超过上限 "
                            f"{self.max_fallbacks_per_game} 次"
                        ),
                    }
                )
        block_reasons = [item["reason"] for item in degraded_games]
        if fallback_rate > self.max_fallback_rate:
            block_reasons.append(
                f"round 回退率 {fallback_rate:.2%}，超过上限 {self.max_fallback_rate:.2%}"
            )
        return {
            "schema_version": 1,
            "round_index": round_index,
            "game_count": len(reports),
            "thresholds": {
                "max_fallback_rate": self.max_fallback_rate,
                "max_fallbacks_per_game": self.max_fallbacks_per_game,
            },
            "metrics": {
                "decision_count": total_decisions,
                "fallback_count": total_fallbacks,
                "fallback_rate": fallback_rate,
                "participant_error_count": sum(
                    int((report.get("error_counts") or {}).get("participant_error") or 0)
                    for report in reports
                ),
            },
            "degraded_games": degraded_games,
            "block_reasons": block_reasons,
            "skill_update_eligible": not block_reasons,
        }

    @staticmethod
    def _validated_round_index(value: int) -> int:
        return _validated_round_index(value)

    @staticmethod
    def _validated_game_concurrency(value: int) -> int:
        try:
            concurrency = int(value)
        except (TypeError, ValueError) as error:
            raise ValueError("game_concurrency 必须是正整数") from error
        if concurrency < 1 or str(concurrency) != str(value).strip():
            raise ValueError("game_concurrency 必须是正整数")
        return concurrency

    @staticmethod
    def _positive_int(value: object, name: str) -> int:
        try:
            normalized = int(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} 必须是正整数") from error
        if normalized < 1:
            raise ValueError(f"{name} 必须是正整数")
        return normalized

    @staticmethod
    def _nonnegative_int(value: object, name: str) -> int:
        try:
            normalized = int(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} 必须是非负整数") from error
        if normalized < 0:
            raise ValueError(f"{name} 必须是非负整数")
        return normalized

    @staticmethod
    def _validated_rate(value: object) -> float:
        try:
            normalized = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError("max_fallback_rate 必须在 0–1 之间") from error
        if not 0 <= normalized <= 1:
            raise ValueError("max_fallback_rate 必须在 0–1 之间")
        return normalized

    def _existing_completed_game(
        self, round_index: int, game_index: int
    ) -> tuple[dict[str, Any], set[str]] | None:
        """读取可安全复用的已完成游戏；半局记录必须人工清理后重跑。"""

        round_directory = RoundGameRecordStore.round_directory_for(
            self.record_directory, round_index
        )
        full_path = round_directory / "log" / f"full-game{game_index}.json"
        if not full_path.exists():
            return None
        try:
            record = json.loads(full_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise RuntimeError(f"已有完整记录无法读取：{full_path}") from error
        final_snapshot = record.get("final_snapshot") or {}
        public_state = final_snapshot.get("public_state") or {}
        if public_state.get("status") != "finished":
            raise RuntimeError(
                f"round{round_index}/game{game_index} 存在未完成记录；"
                "请删除该 game 的四个记录文件后再重跑"
            )
        runner_events = [
            event
            for event in record.get("runner_events", [])
            if isinstance(event, dict)
        ]
        errors = [
            event
            for event in runner_events
            if event.get("event_class") == "RUNNER_ERROR"
        ]
        error_counts: dict[str, int] = {}
        for error in errors:
            event_type = str(error.get("type") or "unknown")
            error_counts[event_type] = error_counts.get(event_type, 0) + 1
        # ``GameRunner.decision_count`` 包含公开/狼人发言和其他所有行动。
        # 回放中前者记录为 PLAYER_SPOKE / PLAYER_PASSED，后者记录为
        # ACTION_SUBMITTED，因此可在复用已完成对局时无损恢复质量闸门指标。
        decision_count = sum(
            event.get("type")
            in {"ACTION_SUBMITTED", "PLAYER_SPOKE", "PLAYER_PASSED"}
            for event in record.get("events", [])
            if isinstance(event, dict)
        )
        fallback_count = sum(
            event.get("type") == "FALLBACK_ACTION" for event in runner_events
        )
        return (
            {
                "game_index": game_index,
                "reused": True,
                "steps": None,
                "errors": errors,
                "error_counts": error_counts,
                "decision_count": decision_count,
                "fallback_count": fallback_count,
                "fallback_rate": (
                    fallback_count / decision_count if decision_count else 0.0
                ),
                "record_path": str(full_path),
                "record_markdown_path": str(round_directory / "full" / f"game{game_index}.md"),
                "public_record_path": str(
                    round_directory / "log" / f"public-game{game_index}.json"
                ),
                "public_record_markdown_path": str(
                    round_directory / "public" / f"game{game_index}.md"
                ),
                "public_state": public_state,
            },
            _roles_from_record(record),
        )

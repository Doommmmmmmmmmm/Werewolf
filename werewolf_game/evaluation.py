"""冻结角色 skill 的对抗评估流程。

训练 round 会在结束时改写 ``strategy.md``。本模块则只读取已经保存在
``records/roundN/skill`` 下的输入/输出快照，组合成临时的角色档案来运行
固定数量的对局。整个流程不会调用复盘器，也不会写入正常训练使用的 prompt
目录，因此可用于验证某一角色的 skill 迭代是否带来实际收益。
"""

from __future__ import annotations

import asyncio
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any

from .constants import ALL_ROLES, ROLE_WOLF
from .engine import GameEngine
from .prompts import RoleStrategyStore
from .records import FileGameRecordStore, GAMES_PER_TRAINING_ROUND, safe_file_part
from .runner import GameRunner


SKILL_TEST_SCHEMA_VERSION = 1
_ROUND_DIRECTORY = re.compile(r"round(\d+)$")


@dataclass(frozen=True)
class SkillSnapshot:
    """一份从训练记录中读取的、可复现的角色 skill 快照。"""

    role: str
    source_round: int
    stage: str
    version: int
    base_path: str
    strategy_path: str
    base_sha256: str
    strategy_sha256: str
    base: str
    strategy: str

    def provenance(self) -> dict[str, Any]:
        """返回可放入测试记录的来源信息，不重复存储 Markdown 内容。"""

        return {
            "role": self.role,
            "source_round": self.source_round,
            "stage": self.stage,
            "version": self.version,
            "base_path": self.base_path,
            "strategy_path": self.strategy_path,
            "base_sha256": self.base_sha256,
            "strategy_sha256": self.strategy_sha256,
        }


@dataclass(frozen=True)
class SkillTestScenario:
    """一组固定来源的角色 skill 对抗。"""

    scenario_id: str
    description: str
    wolf_source: str
    other_source: str


@dataclass(frozen=True)
class SkillTestSource:
    """一个可被多个测试场景复用的冻结 skill 来源。"""

    source_id: str
    round_index: int
    stage: str
    description: str = ""


DEFAULT_SKILL_TEST_SCENARIOS = (
    SkillTestScenario(
        scenario_id="wolf-latest-vs-others-initial",
        description="狼人使用最新输出 skill；其余角色使用初始输入 skill。",
        wolf_source="latest",
        other_source="initial",
    ),
    SkillTestScenario(
        scenario_id="wolf-initial-vs-others-latest",
        description="狼人使用初始输入 skill；其余角色使用最新输出 skill。",
        wolf_source="initial",
        other_source="latest",
    ),
)


SkillTestGameFactory = Callable[
    [str, int, RoleStrategyStore], tuple[GameEngine, Mapping[str, Any]]
]


def _sha256(content: str) -> str:
    return sha256(content.encode("utf-8")).hexdigest()


def _normalised_content(content: str) -> str:
    return str(content).strip() + "\n"


def _safe_identifier(value: str, *, label: str) -> str:
    normalized = safe_file_part(value)
    if normalized != value:
        raise ValueError(f"{label} 只能包含字母、数字、下划线和连字符")
    return normalized


def _validated_non_negative(value: int, *, label: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} 必须是非负整数") from error
    if result < 0 or str(result) != str(value).strip():
        raise ValueError(f"{label} 必须是非负整数")
    return result


def _validated_concurrency(value: int) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError("game_concurrency 必须是正整数") from error
    if result < 1 or str(result) != str(value).strip():
        raise ValueError("game_concurrency 必须是正整数")
    return result


def _validated_positive(value: int, *, label: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} 必须是正整数") from error
    if result < 1 or str(result) != str(value).strip():
        raise ValueError(f"{label} 必须是正整数")
    return result


class SkillTestGameRecordStore(FileGameRecordStore):
    """将单局评估记录写入 ``records/<test>/<scenario>/``。

    记录布局沿用训练 round 的 ``public/full/log`` 命名，但不使用 ``roundN``
    目录，也不会生成 review 或 skill 输出。这样测试数据可与正常训练数据隔离。
    """

    def __init__(
        self,
        directory: str | Path,
        *,
        test_id: str,
        scenario_id: str,
        game_index: int,
        game_count_per_scenario: int = GAMES_PER_TRAINING_ROUND,
        skill_sources: Mapping[str, Mapping[str, Any]],
    ) -> None:
        self.root_directory = Path(directory).resolve()
        self.test_id = _safe_identifier(str(test_id), label="test_id")
        self.scenario_id = _safe_identifier(str(scenario_id), label="scenario_id")
        self.game_index = _validated_non_negative(game_index, label="game_index")
        self.game_count_per_scenario = _validated_positive(
            game_count_per_scenario, label="game_count_per_scenario"
        )
        if self.game_index >= self.game_count_per_scenario:
            raise ValueError(
                "game_index 必须在 0–"
                f"{self.game_count_per_scenario - 1} 之间"
            )
        self.skill_sources = deepcopy(dict(skill_sources))
        super().__init__(
            directory=self.root_directory,
            record_id=f"{self.test_id}-{self.scenario_id}-game{self.game_index}",
        )

    @property
    def test_directory(self) -> Path:
        return self.root_directory / self.test_id

    @property
    def scenario_directory(self) -> Path:
        return self.test_directory / self.scenario_id

    @property
    def public_directory(self) -> Path:
        return self.scenario_directory / "public"

    @property
    def full_directory(self) -> Path:
        return self.scenario_directory / "full"

    @property
    def log_directory(self) -> Path:
        return self.scenario_directory / "log"

    def start(self, metadata: dict[str, Any]) -> Path:
        if self.record is not None:
            assert self.audit_json_path is not None
            return self.audit_json_path

        for path in (
            self.test_directory,
            self.scenario_directory,
            self.public_directory,
            self.full_directory,
            self.log_directory,
        ):
            path.mkdir(parents=True, exist_ok=True, mode=0o700)
            try:
                path.chmod(0o700)
            except PermissionError:
                pass

        game_name = f"game{self.game_index}"
        self.audit_json_path = self.log_directory / f"full-{game_name}.json"
        self.audit_markdown_path = self.full_directory / f"{game_name}.md"
        self.public_json_path = self.log_directory / f"public-{game_name}.json"
        self.public_markdown_path = self.public_directory / f"{game_name}.md"
        occupied_paths = (
            self.audit_json_path,
            self.audit_markdown_path,
            self.public_json_path,
            self.public_markdown_path,
        )
        if any(path.exists() for path in occupied_paths):
            raise FileExistsError(
                f"{self.test_id}/{self.scenario_id}/game{self.game_index} 已有记录；"
                "请选择新的 test_id 或显式复用已完成记录"
            )

        now = datetime.now(timezone.utc).isoformat()
        record_metadata = deepcopy(metadata)
        record_metadata["skill_test"] = {
            "test_id": self.test_id,
            "scenario_id": self.scenario_id,
            "game_index": self.game_index,
            "skill_sources": deepcopy(self.skill_sources),
        }
        self.record = {
            "schema_version": 1,
            "record_id": self.record_id,
            "created_at": now,
            "updated_at": now,
            "metadata": record_metadata,
            "events": [],
            "runner_events": [],
            "latest_snapshot": None,
            "final_snapshot": None,
        }
        self._write_all()
        return self.audit_json_path


class SkillTestRunner:
    """运行多组冻结 skill 对抗，不触发训练复盘。

    默认来源仍是 ``initial``（``initial_round/skill/input``）与 ``latest``
    （最新完整 round 的 ``skill/output``）。调用方也可用
    :class:`SkillTestSource` 显式指定任意历史 round 的输入或输出快照。
    """

    def __init__(
        self,
        *,
        game_factory: SkillTestGameFactory,
        roles: Iterable[str],
        record_directory: str | Path = "records",
        test_id: str = "test1",
        initial_round: int = 0,
        latest_round: int | None = None,
        decision_timeout_seconds: float = 60.0,
        game_concurrency: int = 1,
        game_count_per_scenario: int = GAMES_PER_TRAINING_ROUND,
        resume_completed_games: bool = True,
        scenarios: Iterable[SkillTestScenario] = DEFAULT_SKILL_TEST_SCENARIOS,
        source_definitions: Iterable[SkillTestSource] | None = None,
    ) -> None:
        self.game_factory = game_factory
        self.record_directory = Path(record_directory).resolve()
        self.test_id = _safe_identifier(str(test_id), label="test_id")
        self.initial_round = _validated_non_negative(
            initial_round, label="initial_round"
        )
        self.latest_round = (
            None
            if latest_round is None
            else _validated_non_negative(latest_round, label="latest_round")
        )
        self.decision_timeout_seconds = float(decision_timeout_seconds)
        self.game_concurrency = _validated_concurrency(game_concurrency)
        self.game_count_per_scenario = _validated_positive(
            game_count_per_scenario, label="game_count_per_scenario"
        )
        self.resume_completed_games = bool(resume_completed_games)
        self.roles = self._normalised_roles(roles)
        if ROLE_WOLF not in self.roles:
            raise ValueError("skill 对抗测试至少需要包含 wolf")
        self.scenarios = tuple(scenarios)
        if not self.scenarios:
            raise ValueError("至少需要一个测试场景")
        self.source_definitions = self._normalised_source_definitions(
            source_definitions
        )
        self._validate_scenarios()

    @property
    def test_directory(self) -> Path:
        return self.record_directory / self.test_id

    @property
    def manifest_path(self) -> Path:
        return self.test_directory / "log" / "test-manifest.json"

    @property
    def summary_path(self) -> Path:
        return self.test_directory / "log" / "test-summary.json"

    async def run(self) -> dict[str, Any]:
        """执行每个场景的指定局数，并写入测试来源和胜负汇总。"""

        source_snapshots, latest_round, resolved_sources = self._resolve_sources()
        scenario_sources = {
            scenario.scenario_id: self._sources_for_scenario(scenario, source_snapshots)
            for scenario in self.scenarios
        }
        self._write_or_validate_manifest(
            latest_round=latest_round,
            sources=resolved_sources,
            scenario_sources=scenario_sources,
        )

        scenario_reports: list[dict[str, Any]] = []
        for scenario in self.scenarios:
            scenario_skill_sources = scenario_sources[scenario.scenario_id]
            strategy_store = self._stage_scenario_skills(
                scenario, scenario_skill_sources
            )
            report = await self._run_scenario(
                scenario=scenario,
                strategy_store=strategy_store,
                sources=scenario_skill_sources,
            )
            scenario_reports.append(report)

        summary = {
            "schema_version": SKILL_TEST_SCHEMA_VERSION,
            "test_id": self.test_id,
            "initial_round": self.initial_round,
            "latest_round": latest_round,
            "game_count_per_scenario": self.game_count_per_scenario,
            "skill_updated": False,
            "source_definitions": [
                asdict(source) for source in resolved_sources
            ],
            "scenarios": scenario_reports,
        }
        self._write_json(self.summary_path, summary)
        return {
            **summary,
            "test_directory": str(self.test_directory),
            "manifest_path": str(self.manifest_path),
            "summary_path": str(self.summary_path),
        }

    def _resolve_sources(
        self,
    ) -> tuple[
        dict[str, dict[str, SkillSnapshot]], int, tuple[SkillTestSource, ...]
    ]:
        latest_round = (
            self.latest_round
            if self.latest_round is not None
            else self._find_latest_round_with_outputs()
        )
        sources = self.source_definitions or (
            SkillTestSource(
                source_id="initial",
                round_index=self.initial_round,
                stage="input",
                description="训练开始前的输入 skill。",
            ),
            SkillTestSource(
                source_id="latest",
                round_index=latest_round,
                stage="output",
                description="最新完整训练 round 的输出 skill。",
            ),
        )
        snapshots = {
            source.source_id: {
                role: self._load_snapshot(
                    round_index=source.round_index,
                    role=role,
                    stage=source.stage,
                )
                for role in self.roles
            }
            for source in sources
        }
        return snapshots, latest_round, sources

    def _find_latest_round_with_outputs(self) -> int:
        if not self.record_directory.exists():
            raise FileNotFoundError(f"找不到训练记录目录：{self.record_directory}")
        candidates: list[int] = []
        for path in self.record_directory.iterdir():
            if not path.is_dir():
                continue
            match = _ROUND_DIRECTORY.fullmatch(path.name)
            if match is None:
                continue
            index = int(match.group(1))
            manifest = path / "log" / "skill-version.json"
            if not manifest.exists():
                continue
            try:
                data = json.loads(manifest.read_text(encoding="utf-8"))
            except json.JSONDecodeError as error:
                raise RuntimeError(f"无法读取 skill 版本清单：{manifest}") from error
            roles = data.get("roles", {})
            if all(isinstance(roles.get(role, {}).get("output"), dict) for role in self.roles):
                candidates.append(index)
        if not candidates:
            needed = "、".join(self.roles)
            raise RuntimeError(f"未找到包含以下角色输出 skill 的训练 round：{needed}")
        return max(candidates)

    def _load_snapshot(
        self, *, round_index: int, role: str, stage: str
    ) -> SkillSnapshot:
        round_directory = self.record_directory / f"round{round_index}"
        manifest_path = round_directory / "log" / "skill-version.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"找不到 skill 版本清单：{manifest_path}")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise RuntimeError(f"skill 版本清单格式错误：{manifest_path}") from error
        try:
            entry = manifest["roles"][role][stage]
        except (KeyError, TypeError) as error:
            raise RuntimeError(
                f"round{round_index} 缺少 {role} 的 {stage} skill 快照"
            ) from error
        if not isinstance(entry, dict):
            raise RuntimeError(f"{role} 的 {stage} skill 快照格式错误")

        base_path = self._resolve_snapshot_path(
            round_directory, entry.get("base_path"), role=role, stage=stage
        )
        strategy_path = self._resolve_snapshot_path(
            round_directory, entry.get("strategy_path"), role=role, stage=stage
        )
        base = _normalised_content(base_path.read_text(encoding="utf-8"))
        strategy = _normalised_content(strategy_path.read_text(encoding="utf-8"))
        base_hash = str(entry.get("base_sha256") or "")
        strategy_hash = str(entry.get("strategy_sha256") or "")
        if base_hash != _sha256(base) or strategy_hash != _sha256(strategy):
            raise RuntimeError(
                f"{role} 的 {stage} skill 快照哈希与清单不一致：round{round_index}"
            )
        try:
            version = int(entry["version"])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(f"{role} 的 {stage} skill 版本格式错误") from error
        if version < 0:
            raise RuntimeError(f"{role} 的 {stage} skill 版本不能为负数")
        return SkillSnapshot(
            role=role,
            source_round=round_index,
            stage=stage,
            version=version,
            base_path=str(base_path.relative_to(self.record_directory)),
            strategy_path=str(strategy_path.relative_to(self.record_directory)),
            base_sha256=base_hash,
            strategy_sha256=strategy_hash,
            base=base,
            strategy=strategy,
        )

    @staticmethod
    def _resolve_snapshot_path(
        round_directory: Path, value: object, *, role: str, stage: str
    ) -> Path:
        if not isinstance(value, str):
            raise RuntimeError(f"{role} 的 {stage} skill 快照缺少路径")
        candidate = (round_directory / value).resolve()
        try:
            candidate.relative_to(round_directory.resolve())
        except ValueError as error:
            raise RuntimeError(f"{role} 的 {stage} skill 快照路径非法") from error
        if not candidate.exists():
            raise FileNotFoundError(f"找不到 {role} 的 {stage} skill 快照：{candidate}")
        return candidate

    def _sources_for_scenario(
        self,
        scenario: SkillTestScenario,
        source_snapshots: Mapping[str, Mapping[str, SkillSnapshot]],
    ) -> dict[str, SkillSnapshot]:
        selected: dict[str, SkillSnapshot] = {}
        for role in self.roles:
            source_kind = scenario.wolf_source if role == ROLE_WOLF else scenario.other_source
            selected[role] = source_snapshots[source_kind][role]
        return selected

    def _write_or_validate_manifest(
        self,
        *,
        latest_round: int,
        sources: Iterable[SkillTestSource],
        scenario_sources: Mapping[str, Mapping[str, SkillSnapshot]],
    ) -> None:
        payload = {
            "schema_version": SKILL_TEST_SCHEMA_VERSION,
            "test_id": self.test_id,
            "initial_round": self.initial_round,
            "latest_round": latest_round,
            "game_count_per_scenario": self.game_count_per_scenario,
            "skill_updated": False,
            "source_definitions": [asdict(source) for source in sources],
            "scenarios": [
                {
                    **asdict(scenario),
                    "skill_sources": {
                        role: snapshot.provenance()
                        for role, snapshot in sorted(
                            scenario_sources[scenario.scenario_id].items()
                        )
                    },
                }
                for scenario in self.scenarios
            ],
        }
        if self.manifest_path.exists():
            try:
                existing = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as error:
                raise RuntimeError(f"测试清单无法读取：{self.manifest_path}") from error
            if existing != payload:
                raise RuntimeError(
                    f"已有测试 {self.test_id} 的来源配置不同；"
                    "请使用新的 test_id，避免覆盖既有评估记录"
                )
            return
        self._write_json(self.manifest_path, payload)

    def _stage_scenario_skills(
        self,
        scenario: SkillTestScenario,
        sources: Mapping[str, SkillSnapshot],
    ) -> RoleStrategyStore:
        """复制快照到测试目录，让 LLM Participant 只读这组冻结档案。"""

        scenario_directory = self.test_directory / scenario.scenario_id
        role_root = scenario_directory / "skill" / "roles"
        source_manifest_path = scenario_directory / "log" / "skill-sources.json"
        source_payload = {
            "schema_version": SKILL_TEST_SCHEMA_VERSION,
            "test_id": self.test_id,
            "scenario_id": scenario.scenario_id,
            "description": scenario.description,
            "skill_updated": False,
            "roles": {
                role: snapshot.provenance() for role, snapshot in sorted(sources.items())
            },
        }
        if source_manifest_path.exists():
            try:
                existing = json.loads(source_manifest_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"测试 skill 来源清单无法读取：{source_manifest_path}"
                ) from error
            if existing != source_payload:
                raise RuntimeError(
                    f"{self.test_id}/{scenario.scenario_id} 的 skill 来源不同，"
                    "拒绝覆盖已有测试快照"
                )
        else:
            self._write_json(source_manifest_path, source_payload)

        for role, snapshot in sources.items():
            role_directory = role_root / role
            self._write_or_validate_snapshot_file(
                role_directory / "base.md", snapshot.base
            )
            self._write_or_validate_snapshot_file(
                role_directory / "strategy.md", snapshot.strategy
            )
        return RoleStrategyStore(scenario_directory / "skill")

    def _write_or_validate_snapshot_file(self, path: Path, content: str) -> None:
        normalized = _normalised_content(content)
        if path.exists():
            if _normalised_content(path.read_text(encoding="utf-8")) != normalized:
                raise RuntimeError(f"测试 skill 快照已存在且内容不同：{path}")
            return
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            path.parent.chmod(0o700)
        except PermissionError:
            pass
        FileGameRecordStore._write_atomic(path, normalized)

    async def _run_scenario(
        self,
        *,
        scenario: SkillTestScenario,
        strategy_store: RoleStrategyStore,
        sources: Mapping[str, SkillSnapshot],
    ) -> dict[str, Any]:
        reports: list[dict[str, Any]] = []
        scheduled_games: list[tuple[int, GameEngine, Mapping[str, Any]]] = []
        scenario_directory = self.test_directory / scenario.scenario_id
        provenance = {
            role: snapshot.provenance() for role, snapshot in sorted(sources.items())
        }
        for game_index in range(self.game_count_per_scenario):
            existing = self._existing_completed_game(scenario_directory, game_index)
            if existing is not None:
                if not self.resume_completed_games:
                    raise FileExistsError(
                        f"{self.test_id}/{scenario.scenario_id}/game{game_index} 已完成；"
                        "如需复用已有记录，请设置 resume_completed_games=True"
                    )
                reports.append(existing)
                continue
            engine, participants = self.game_factory(
                scenario.scenario_id, game_index, strategy_store
            )
            factory_roles = set(engine.rules.role_deck)
            if factory_roles != set(self.roles):
                expected = "、".join(self.roles)
                actual = "、".join(sorted(factory_roles))
                raise ValueError(
                    f"游戏身份集与测试快照不一致；期望 {expected}，实际 {actual}"
                )
            scheduled_games.append((game_index, engine, participants))

        async def run_game(
            game_index: int, engine: GameEngine, participants: Mapping[str, Any]
        ) -> dict[str, Any]:
            store = SkillTestGameRecordStore(
                self.record_directory,
                test_id=self.test_id,
                scenario_id=scenario.scenario_id,
                game_index=game_index,
                game_count_per_scenario=self.game_count_per_scenario,
                skill_sources=provenance,
            )
            report = await GameRunner(
                engine=engine,
                participants=participants,
                record_store=store,
                decision_timeout_seconds=self.decision_timeout_seconds,
            ).run()
            if not report.get("record_path"):
                raise RuntimeError(
                    f"{self.test_id}/{scenario.scenario_id}/game{game_index} 没有生成完整记录"
                )
            return {"game_index": game_index, **report}

        if self.game_concurrency == 1:
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
        winners = Counter(str(report["public_state"].get("winner") or "unknown") for report in reports)
        summary = {
            "scenario_id": scenario.scenario_id,
            "description": scenario.description,
            "game_count": len(reports),
            "completed": len(reports) == self.game_count_per_scenario,
            "skill_updated": False,
            "skill_sources": provenance,
            "wins": dict(sorted(winners.items())),
            "games": [
                {
                    "game_index": int(report["game_index"]),
                    "winner": report["public_state"].get("winner"),
                    "steps": report.get("steps"),
                    "reused": bool(report.get("reused")),
                    "error_count": len(report.get("errors") or []),
                    "record_path": report.get("record_path"),
                }
                for report in reports
            ],
        }
        self._write_json(scenario_directory / "log" / "test-summary.json", summary)
        return summary

    def _existing_completed_game(
        self, scenario_directory: Path, game_index: int
    ) -> dict[str, Any] | None:
        full_path = scenario_directory / "log" / f"full-game{game_index}.json"
        if not full_path.exists():
            return None
        try:
            record = json.loads(full_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise RuntimeError(f"已有测试记录无法读取：{full_path}") from error
        final_snapshot = record.get("final_snapshot")
        public_state = (
            final_snapshot.get("public_state", {})
            if isinstance(final_snapshot, Mapping)
            else {}
        )
        if public_state.get("status") != "finished":
            raise RuntimeError(
                f"{self.test_id}/{scenario_directory.name}/game{game_index} 存在未完成记录；"
                "请人工清理该局的四个记录文件后再运行"
            )
        return {
            "game_index": game_index,
            "reused": True,
            "steps": None,
            "errors": [],
            "record_path": str(full_path),
            "record_markdown_path": str(scenario_directory / "full" / f"game{game_index}.md"),
            "public_record_path": str(
                scenario_directory / "log" / f"public-game{game_index}.json"
            ),
            "public_record_markdown_path": str(
                scenario_directory / "public" / f"game{game_index}.md"
            ),
            "public_state": public_state,
        }

    def _write_json(self, path: Path, value: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            path.parent.chmod(0o700)
        except PermissionError:
            pass
        FileGameRecordStore._write_atomic(
            path, json.dumps(value, ensure_ascii=False, indent=2) + "\n"
        )

    @staticmethod
    def _normalised_roles(roles: Iterable[str]) -> tuple[str, ...]:
        normalized = tuple(sorted({str(role) for role in roles}))
        if not normalized:
            raise ValueError("测试角色不能为空")
        invalid = [role for role in normalized if role not in ALL_ROLES]
        if invalid:
            raise ValueError(f"不支持的角色 skill：{'、'.join(invalid)}")
        return normalized

    @staticmethod
    def _normalised_source_definitions(
        source_definitions: Iterable[SkillTestSource] | None,
    ) -> tuple[SkillTestSource, ...] | None:
        if source_definitions is None:
            return None
        normalized: list[SkillTestSource] = []
        seen_ids: set[str] = set()
        for source in source_definitions:
            if not isinstance(source, SkillTestSource):
                raise TypeError("source_definitions 必须由 SkillTestSource 组成")
            source_id = _safe_identifier(source.source_id, label="source_id")
            if source_id != source.source_id:
                raise ValueError("source_id 只能包含字母、数字、下划线和连字符")
            if source_id in seen_ids:
                raise ValueError(f"测试 skill 来源重复：{source_id}")
            if source.stage not in {"input", "output"}:
                raise ValueError("skill 来源 stage 只能是 input 或 output")
            normalized.append(
                SkillTestSource(
                    source_id=source_id,
                    round_index=_validated_non_negative(
                        source.round_index, label="source.round_index"
                    ),
                    stage=source.stage,
                    description=str(source.description),
                )
            )
            seen_ids.add(source_id)
        if not normalized:
            raise ValueError("source_definitions 至少需要包含一个来源")
        return tuple(normalized)

    def _validate_scenarios(self) -> None:
        known_ids: set[str] = set()
        available_sources = (
            {source.source_id for source in self.source_definitions}
            if self.source_definitions is not None
            else {"initial", "latest"}
        )
        for scenario in self.scenarios:
            _safe_identifier(scenario.scenario_id, label="scenario_id")
            if scenario.scenario_id in known_ids:
                raise ValueError(f"测试场景重复：{scenario.scenario_id}")
            known_ids.add(scenario.scenario_id)
            if scenario.wolf_source not in available_sources:
                raise ValueError(f"wolf_source 不存在：{scenario.wolf_source}")
            if scenario.other_source not in available_sources:
                raise ValueError(f"other_source 不存在：{scenario.other_source}")

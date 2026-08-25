"""Season 2 的固定 Meta-Agent：从历史回放构造受限 Task-Agent Harness。

本模块实现的是“构造任务代理的代理”，但不会让模型修改自身流程或项目代码。元流程
固定为：证据分批分析 → 候选 Harness 设计 → 红队批评 → Python 质量闸门与选择。
所有候选都写入 archive，只有显式调用 ``promote`` 才会成为 active 版本。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

from .harness import (
    HARNESS_SOURCE_TYPES,
    HarnessFileStore,
    HarnessSpec,
    default_harness_for_profile,
    evaluate_harness,
)
from .llm.coordinator import ModelRequestCoordinator
from .prompts import RoleStrategyStore, render_prompt
from .replay import render_audit_markdown
from .research import ResearchSource, call_search_provider


META_SCHEMA_VERSION = 1
ANALYSIS_KEYS = (
    "evidence",
    "strengths",
    "failure_modes",
    "opponent_patterns",
    "counterfactuals",
    "uncertainties",
)
CRITIQUE_KEYS = (
    "hard_rule_violations",
    "information_leakage_risks",
    "unsupported_claims",
    "complexity_concerns",
    "counterexamples",
    "strengths",
)


def _text(value: object, *, limit: int = 1200) -> str:
    return " ".join(str(value or "").strip().split())[:limit]


def _string_list(value: object, *, limit: int = 16, item_limit: int = 1200) -> list[str]:
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, (list, tuple)):
        values = list(value)
    else:
        values = []
    result: list[str] = []
    for item in values:
        normalized = _text(item, limit=item_limit)
        if normalized and normalized not in result:
            result.append(normalized)
        if len(result) >= limit:
            break
    return result


def _safe_json(value: object, *, depth: int = 0) -> Any:
    """把模型返回限制为可记录 JSON；不接受任意对象或代码。"""

    if depth > 7:
        raise ValueError("Meta-Agent 输出嵌套层级过深")
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Meta-Agent 输出不能包含 NaN 或无穷浮点数")
        return value
    if isinstance(value, Mapping):
        return {_text(key, limit=100): _safe_json(item, depth=depth + 1) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(item, depth=depth + 1) for item in list(value)[:80]]
    raise ValueError(f"Meta-Agent 输出包含不可记录值：{type(value).__name__}")


@dataclass(frozen=True)
class MetaAgentConfig:
    """固定元流程的预算和质量约束。"""

    replay_batch_size: int = 2
    candidate_count: int = 3
    max_cards: int = 8
    max_tokens: int = 1800
    analysis_max_tokens: int = 1400
    critic_max_tokens: int = 1100
    max_replay_chars_per_game: int = 26000
    allow_research_sources: bool = True
    research_query_count: int = 3
    research_max_results: int = 5
    auto_promote: bool = False

    def validate(self) -> None:
        if not 1 <= int(self.replay_batch_size) <= 10:
            raise ValueError("replay_batch_size 必须在 1–10 之间")
        if not 1 <= int(self.candidate_count) <= 8:
            raise ValueError("candidate_count 必须在 1–8 之间")
        if not 1 <= int(self.max_cards) <= 12:
            raise ValueError("max_cards 必须在 1–12 之间")
        if not 0 <= int(self.research_query_count) <= 8:
            raise ValueError("research_query_count 必须在 0–8 之间")
        if not 1 <= int(self.research_max_results) <= 20:
            raise ValueError("research_max_results 必须在 1–20 之间")
        for name in ("max_tokens", "analysis_max_tokens", "critic_max_tokens", "max_replay_chars_per_game"):
            if int(getattr(self, name)) < 100:
                raise ValueError(f"{name} 太小，无法完成 Meta-Agent 阶段")


@dataclass(frozen=True)
class ReplayAnalysis:
    role: str
    round_index: int
    game_ids: tuple[str, ...]
    batch_reports: tuple[dict[str, Any], ...]
    summary: dict[str, list[str]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": META_SCHEMA_VERSION,
            "role": self.role,
            "round_index": self.round_index,
            "game_ids": list(self.game_ids),
            "batch_reports": [dict(report) for report in self.batch_reports],
            "summary": {key: list(value) for key, value in self.summary.items()},
        }


@dataclass(frozen=True)
class CandidateAssessment:
    candidate: HarnessSpec
    critique: dict[str, list[str]]
    score: float
    passed: bool
    archive_path: str | None = None
    static_evaluation: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate.as_dict(),
            "critique": {key: list(value) for key, value in self.critique.items()},
            "score": self.score,
            "passed": self.passed,
            "archive_path": self.archive_path,
            "static_evaluation": dict(self.static_evaluation),
        }


@dataclass(frozen=True)
class MetaAgentResult:
    role: str
    round_index: int
    parent_id: str
    analysis: ReplayAnalysis
    candidates: tuple[CandidateAssessment, ...]
    selected: HarnessSpec
    selected_passed: bool
    promoted_path: str | None
    experiment_paths: dict[str, str]
    stage_trace: tuple[dict[str, Any], ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": META_SCHEMA_VERSION,
            "role": self.role,
            "round_index": self.round_index,
            "parent_id": self.parent_id,
            "analysis": self.analysis.as_dict(),
            "candidates": [item.as_dict() for item in self.candidates],
            "selected": self.selected.as_dict(),
            "selected_passed": self.selected_passed,
            "promoted_path": self.promoted_path,
            "experiment_paths": dict(self.experiment_paths),
            "stage_trace": [dict(item) for item in self.stage_trace],
        }


class HarnessArchiveStore:
    """Season 2 的候选谱系和实验记录存储。"""

    def __init__(self, record_directory: str | Path = "records/seasons/season2") -> None:
        self.root = Path(record_directory).resolve()
        self.archive_root = self.root / "archive"
        self.experiments_root = self.root / "experiments"
        self.active_store = HarnessFileStore(self.root)

    def archive_candidate(
        self,
        *,
        spec: HarnessSpec,
        critique: Mapping[str, Any],
        analysis: Mapping[str, Any],
        static_evaluation: Mapping[str, Any] | None = None,
        round_index: int,
        candidate_index: int,
    ) -> Path:
        spec.validate()
        path = self.archive_root / spec.role / spec.harness_id
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._write_json_immutable(path / "harness.json", spec.as_dict())
        self._write_json_immutable(
            path / "assessment.json",
            {
                "schema_version": META_SCHEMA_VERSION,
                "round_index": int(round_index),
                "candidate_index": int(candidate_index),
                "critique": _safe_json(critique),
                "static_evaluation": _safe_json(static_evaluation or {}),
                "analysis_digest": self._digest(analysis),
                "harness_fingerprint": spec.fingerprint,
            },
        )
        cards_directory = path / "cards"
        cards_directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        for card in spec.cards:
            self._write_text_immutable(cards_directory / f"{card.card_id}.md", card.markdown())
        return path

    def write_experiment(
        self,
        *,
        round_index: int,
        role: str,
        analysis: ReplayAnalysis,
        assessments: Iterable[CandidateAssessment],
        selected: HarnessSpec,
        research_sources: Iterable[Mapping[str, Any]] = (),
        stage_trace: Iterable[Mapping[str, Any]] = (),
    ) -> dict[str, str]:
        directory = self.experiments_root / f"round{int(round_index)}" / role
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        assessment_list = list(assessments)
        research_list = [_safe_json(item) for item in research_sources]
        research_path = self._write_research_snapshot(
            round_index=int(round_index),
            role=role,
            sources=research_list,
        )
        analysis_path = directory / "replay-analysis.json"
        self._write_json(analysis_path, analysis.as_dict())
        candidates_path = directory / "candidate-assessments.json"
        self._write_json(
            candidates_path,
            {"candidates": [item.as_dict() for item in assessment_list]},
        )
        evaluation_path = directory / "evaluation-summary.json"
        self._write_json(
            evaluation_path,
            {
                "schema_version": META_SCHEMA_VERSION,
                "round_index": int(round_index),
                "role": role,
                "method": "deterministic_static_gate_plus_red_team_score",
                "selected_harness_id": selected.harness_id,
                "candidates": [
                    {
                        "harness_id": item.candidate.harness_id,
                        "score": item.score,
                        "passed": item.passed,
                        "static_evaluation": dict(item.static_evaluation),
                    }
                    for item in assessment_list
                ],
            },
        )
        trace_path = directory / "pipeline-trace.json"
        self._write_json(
            trace_path,
            {
                "schema_version": META_SCHEMA_VERSION,
                "round_index": int(round_index),
                "role": role,
                "stages": [_safe_json(item) for item in stage_trace],
            },
        )
        assignment_path = directory / "assignment-manifest.json"
        self._write_json(
            assignment_path,
            {
                "schema_version": META_SCHEMA_VERSION,
                "round_index": int(round_index),
                "role": role,
                "selected_harness_id": selected.harness_id,
                "selected_fingerprint": selected.fingerprint,
                "research_sources_path": str(research_path),
                "research_source_count": len(research_list),
                "research_sources": research_list,
                "candidate_ids": [item.candidate.harness_id for item in assessment_list],
            },
        )
        return {
            "analysis_path": str(analysis_path),
            "candidate_assessments_path": str(candidates_path),
            "evaluation_summary_path": str(evaluation_path),
            "assignment_manifest_path": str(assignment_path),
            "research_sources_path": str(research_path),
            "pipeline_trace_path": str(trace_path),
        }

    def _write_research_snapshot(
        self,
        *,
        round_index: int,
        role: str,
        sources: list[Any],
    ) -> Path:
        """保存可审计研究来源；不同抓取时间不会覆盖旧快照。"""

        directory = self.root / "research" / f"round{int(round_index)}" / role
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        payload = {
            "schema_version": META_SCHEMA_VERSION,
            "round_index": int(round_index),
            "role": role,
            "sources": sources,
        }
        primary = directory / "sources.json"
        if not primary.exists():
            self._write_json_immutable(primary, payload)
            return primary
        try:
            existing = json.loads(primary.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing = None
        if existing == payload:
            return primary
        versioned = directory / f"sources-{self._digest(payload)[:12]}.json"
        self._write_json_immutable(versioned, payload)
        return versioned

    def promote(self, spec: HarnessSpec, *, reason: str = "") -> Path:
        spec.validate()
        path = self.active_store.save_active(spec)
        promotion_path = self.archive_root / spec.role / spec.harness_id / "promotion.json"
        self._write_json_immutable(
            promotion_path,
            {
                "schema_version": META_SCHEMA_VERSION,
                "harness_id": spec.harness_id,
                "role": spec.role,
                "fingerprint": spec.fingerprint,
                "reason": _text(reason, limit=2000),
                "active_path": str(path),
            },
        )
        return path

    def load_active(self, role: str, *, strategy_store: RoleStrategyStore | None = None) -> HarnessSpec:
        profile = strategy_store.profile(role) if strategy_store is not None else None
        return self.active_store.load_active(role, profile=profile)

    def ensure_season_manifest(self, manifest: Mapping[str, Any]) -> Path:
        """写入一次性的赛季配置；已有文件内容不允许静默改变。"""

        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        path = self.root / "season-manifest.json"
        value = _safe_json(dict(manifest))
        if path.exists():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise RuntimeError(f"Season 2 manifest 无法读取：{path}") from error
            if existing != value:
                raise FileExistsError(f"Season 2 manifest 已存在且内容不一致：{path}")
            return path
        self._write_json(path, value)
        return path

    @staticmethod
    def _digest(value: object) -> str:
        content = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    @staticmethod
    def _write_json(path: Path, value: object) -> None:
        temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        try:
            temporary.chmod(0o600)
        except PermissionError:
            pass
        temporary.replace(path)
        try:
            path.chmod(0o600)
        except PermissionError:
            pass

    @staticmethod
    def _write_json_immutable(path: Path, value: object) -> None:
        if path.exists():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise RuntimeError(f"archive 文件已存在但无法校验：{path}") from error
            if existing != value:
                raise FileExistsError(f"archive 节点不可覆盖且内容不一致：{path}")
            return
        HarnessArchiveStore._write_json(path, value)

    @staticmethod
    def _write_text(path: Path, value: str) -> None:
        temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        temporary.write_text(str(value).rstrip() + "\n", encoding="utf-8")
        try:
            temporary.chmod(0o600)
        except PermissionError:
            pass
        temporary.replace(path)
        try:
            path.chmod(0o600)
        except PermissionError:
            pass

    @staticmethod
    def _write_text_immutable(path: Path, value: str) -> None:
        normalized = str(value).rstrip() + "\n"
        if path.exists():
            if path.read_text(encoding="utf-8") != normalized:
                raise FileExistsError(f"archive 文件不可覆盖且内容不一致：{path}")
            return
        HarnessArchiveStore._write_text(path, normalized)


class TaskAgentMetaAgent:
    """固定的 Season 2 元流程。

    它只产生受 schema 约束的 HarnessSpec，不直接改变规则、不读取其他角色的 skill，
    也不在对局中运行。历史回放可以包含完整审计信息，但送入每次调用的角色档案始终
    只有目标角色自己的 base/strategy。
    """

    def __init__(
        self,
        *,
        model_client: Any,
        strategy_store: RoleStrategyStore | None = None,
        archive_store: HarnessArchiveStore | None = None,
        request_coordinator: ModelRequestCoordinator | None = None,
        config: MetaAgentConfig | None = None,
        research_provider: Any | None = None,
    ) -> None:
        if not hasattr(model_client, "complete_json"):
            raise ValueError("TaskAgentMetaAgent 需要具有 complete_json 的模型客户端")
        self.model_client = model_client
        self.strategy_store = strategy_store or RoleStrategyStore()
        self.archive_store = archive_store or HarnessArchiveStore()
        self.request_coordinator = request_coordinator
        self.config = config or MetaAgentConfig()
        self.config.validate()
        self.research_provider = research_provider

    async def construct_task_agent(
        self,
        *,
        round_index: int,
        role: str,
        game_records: Iterable[Mapping[str, Any]],
        parent: HarnessSpec | Mapping[str, Any] | None = None,
        research_sources: Iterable[Mapping[str, Any]] = (),
        promote: bool | None = None,
    ) -> MetaAgentResult:
        records = [dict(item) for item in game_records]
        stage_trace: list[dict[str, Any]] = [
            {
                "stage": "input",
                "status": "accepted",
                "round_index": int(round_index),
                "role": role,
                "game_count": len(records),
            }
        ]
        source_list = (
            [
                item.as_dict() if isinstance(item, ResearchSource) else dict(item)
                for item in research_sources
                if isinstance(item, ResearchSource) or isinstance(item, Mapping)
            ]
            if self.config.allow_research_sources
            else []
        )
        profile = self.strategy_store.profile(role)
        parent_spec = self._resolve_parent(role, parent, profile)
        try:
            analysis = await self._analyze_history(
                round_index=int(round_index),
                role=role,
                profile=profile,
                records=records,
                research_sources=source_list,
            )
            stage_trace.append(
                {
                    "stage": "replay_analysis",
                    "status": "completed",
                    "batch_count": len(analysis.batch_reports),
                    "evidence_count": len(analysis.summary.get("evidence", [])),
                }
            )
        except Exception as error:
            # 元层故障不能污染或覆盖现有 active；保留可审计的空分析并继续生成保守候选。
            analysis = ReplayAnalysis(
                role=role,
                round_index=int(round_index),
                game_ids=tuple(
                    str(record.get("metadata", {}).get("game_id") or f"game{index}")
                    for index, record in enumerate(records)
                ),
                batch_reports=(
                    {
                        "error": type(error).__name__,
                        "message": "回放分析阶段失败，不能把本轮历史当成有效证据。",
                    },
                ),
                summary={
                    **{key: [] for key in ANALYSIS_KEYS if key != "uncertainties"},
                    "uncertainties": [
                        "Meta-Agent 回放分析失败；本轮只能使用保守候选，禁止自动晋升。"
                    ],
                },
            )
            stage_trace.append(
                {
                    "stage": "replay_analysis",
                    "status": "fallback",
                    "error_type": type(error).__name__,
                }
            )
        if self.research_provider is not None and self.config.allow_research_sources:
            try:
                source_list.extend(await self._collect_research_sources(role, analysis))
                source_list = self._dedupe_sources(source_list)
                stage_trace.append(
                    {
                        "stage": "research",
                        "status": "completed",
                        "source_count": len(source_list),
                    }
                )
            except Exception as error:
                stage_trace.append(
                    {
                        "stage": "research",
                        "status": "failed",
                        "error_type": type(error).__name__,
                    }
                )
        try:
            candidates = await self._design_candidates(
                round_index=int(round_index),
                role=role,
                profile=profile,
                parent=parent_spec,
                analysis=analysis,
                research_sources=source_list,
            )
            stage_trace.append(
                {
                    "stage": "harness_design",
                    "status": "completed",
                    "candidate_count": len(candidates),
                }
            )
        except Exception as error:
            candidates = [
                self._fallback_candidate(parent_spec, role, int(round_index), "replay_mutation", 0)
            ]
            while len(candidates) < int(self.config.candidate_count):
                candidates.append(
                    self._diversified_fallback(
                        parent_spec,
                        role,
                        int(round_index),
                        ("counterfactual", "red_team", "recombined")[len(candidates) % 3],
                        len(candidates),
                    )
                )
            stage_trace.append(
                {
                    "stage": "harness_design",
                    "status": "fallback",
                    "error_type": type(error).__name__,
                    "candidate_count": len(candidates),
                }
            )
        assessments: list[CandidateAssessment] = []
        for index, candidate in enumerate(candidates):
            static_evaluation = evaluate_harness(candidate, profile).as_dict()
            try:
                critique = await self._critic_candidate(
                    role=role,
                    profile=profile,
                    candidate=candidate,
                    analysis=analysis,
                )
            except Exception as error:
                critique = {key: [] for key in CRITIQUE_KEYS}
                critique["complexity_concerns"] = [
                    f"红队阶段失败（{type(error).__name__}），候选不得自动晋升。"
                ]
            score, passed = self._score_candidate(
                candidate,
                critique,
                analysis,
                static_evaluation=static_evaluation,
            )
            archive_path = self.archive_store.archive_candidate(
                spec=candidate,
                critique=critique,
                analysis=analysis.as_dict(),
                static_evaluation=static_evaluation,
                round_index=int(round_index),
                candidate_index=index,
            )
            assessments.append(
                CandidateAssessment(
                    candidate=candidate,
                    critique=critique,
                    score=score,
                    passed=passed,
                    archive_path=str(archive_path),
                    static_evaluation=static_evaluation,
                )
            )
        stage_trace.append(
            {
                "stage": "red_team_and_static_gate",
                "status": "completed",
                "candidate_count": len(assessments),
                "passed_count": sum(item.passed for item in assessments),
                "static_passed_count": sum(
                    bool(item.static_evaluation.get("passed")) for item in assessments
                ),
            }
        )
        selected_assessment = self._select_assessment(assessments, parent_spec)
        should_promote = self.config.auto_promote if promote is None else bool(promote)
        promoted_path: str | None = None
        if should_promote and selected_assessment.passed:
            promoted_path = str(
                self.archive_store.promote(
                    selected_assessment.candidate,
                    reason="通过固定 Meta-Agent 的 schema、信息边界和红队质量闸门。",
                )
            )
        stage_trace.append(
            {
                "stage": "selection",
                "status": "promoted" if promoted_path else "selected",
                "selected_harness_id": selected_assessment.candidate.harness_id,
                "selected_passed": selected_assessment.passed,
                "auto_promote_requested": should_promote,
            }
        )
        experiment_paths = self.archive_store.write_experiment(
            round_index=int(round_index),
            role=role,
            analysis=analysis,
            assessments=assessments,
            selected=selected_assessment.candidate,
            research_sources=source_list,
            stage_trace=stage_trace,
        )
        return MetaAgentResult(
            role=role,
            round_index=int(round_index),
            parent_id=parent_spec.harness_id,
            analysis=analysis,
            candidates=tuple(assessments),
            selected=selected_assessment.candidate,
            selected_passed=selected_assessment.passed,
            promoted_path=promoted_path,
            experiment_paths=experiment_paths,
            stage_trace=tuple(stage_trace),
        )

    async def construct_for_roles(
        self,
        *,
        round_index: int,
        roles: Iterable[str],
        game_records: Iterable[Mapping[str, Any]],
        parents: Mapping[str, HarnessSpec | Mapping[str, Any]] | None = None,
        research_sources: Iterable[Mapping[str, Any]] = (),
        promote: bool | None = None,
    ) -> dict[str, MetaAgentResult]:
        """逐角色构造，避免把一个角色的 skill 泄露给另一个角色。"""

        records = [dict(item) for item in game_records]
        source_list = (
            [
                item.as_dict() if isinstance(item, ResearchSource) else dict(item)
                for item in research_sources
                if isinstance(item, ResearchSource) or isinstance(item, Mapping)
            ]
            if self.config.allow_research_sources
            else []
        )
        result: dict[str, MetaAgentResult] = {}
        for role in dict.fromkeys(str(item) for item in roles):
            result[role] = await self.construct_task_agent(
                round_index=round_index,
                role=role,
                game_records=records,
                parent=(parents or {}).get(role),
                research_sources=source_list,
                promote=promote,
            )
        return result

    async def _analyze_history(
        self,
        *,
        round_index: int,
        role: str,
        profile: Any,
        records: list[dict[str, Any]],
        research_sources: list[Mapping[str, Any]],
    ) -> ReplayAnalysis:
        game_ids = tuple(
            str(record.get("metadata", {}).get("game_id") or f"game{index}")
            for index, record in enumerate(records)
        )
        if not records:
            return ReplayAnalysis(
                role=role,
                round_index=round_index,
                game_ids=game_ids,
                batch_reports=(),
                summary={key: [] for key in ANALYSIS_KEYS},
            )
        reports: list[dict[str, Any]] = []
        size = int(self.config.replay_batch_size)
        for batch_index, start in enumerate(range(0, len(records), size)):
            batch = records[start : start + size]
            replays = []
            for offset, record in enumerate(batch):
                markdown = render_audit_markdown(record)
                replays.append(
                    {
                        "game_id": game_ids[start + offset],
                        "audit_replay_markdown": markdown[: int(self.config.max_replay_chars_per_game)],
                    }
                )
            payload = {
                "role": role,
                "own_role_markdown": {"base_md": profile.base, "strategy_md": profile.strategy},
                "round_index": round_index,
                "batch_index": batch_index,
                "complete_game_replays": replays,
                "approved_research_sources": self._research_payload(research_sources),
            }
            raw = await self._complete_json(
                system=render_prompt("meta_replay_analysis_system.txt", role=role),
                messages=[{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
                max_tokens=int(self.config.analysis_max_tokens),
            )
            reports.append(self._normalize_analysis_batch(raw, [item["game_id"] for item in replays]))
        summary = {key: [] for key in ANALYSIS_KEYS}
        for report in reports:
            for key in ANALYSIS_KEYS:
                summary[key].extend(_string_list(report.get(key)))
        summary = {key: self._dedupe(values, limit=24) for key, values in summary.items()}
        return ReplayAnalysis(
            role=role,
            round_index=round_index,
            game_ids=game_ids,
            batch_reports=tuple(reports),
            summary=summary,
        )

    async def _design_candidates(
        self,
        *,
        round_index: int,
        role: str,
        profile: Any,
        parent: HarnessSpec,
        analysis: ReplayAnalysis,
        research_sources: list[Mapping[str, Any]],
    ) -> list[HarnessSpec]:
        payload = {
            "role": role,
            "round_index": round_index,
            "own_role_markdown": {"base_md": profile.base, "strategy_md": profile.strategy},
            "parent_harness": parent.as_dict(),
            "replay_analysis": analysis.as_dict(),
            "approved_research_sources": self._research_payload(research_sources),
            "candidate_count": int(self.config.candidate_count),
            "max_cards": int(self.config.max_cards),
            "allowed_source_types": sorted(HARNESS_SOURCE_TYPES - {"baseline"}),
        }
        raw = await self._complete_json(
            system=render_prompt("meta_harness_design_system.txt", role=role),
            messages=[{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
            max_tokens=int(self.config.max_tokens),
        )
        raw_candidates = raw.get("candidates", raw.get("harnesses", [])) if isinstance(raw, dict) else []
        if not isinstance(raw_candidates, list):
            raw_candidates = []
        result: list[HarnessSpec] = []
        for index, item in enumerate(raw_candidates[: int(self.config.candidate_count)]):
            if not isinstance(item, Mapping):
                continue
            try:
                result.append(self._normalise_candidate(item, role=role, parent=parent, round_index=round_index, index=index))
            except (TypeError, ValueError):
                continue
        if not result:
            result.append(self._fallback_candidate(parent, role, round_index, "replay_mutation", 0))
        # 即使模型只返回一个候选，也保留少量受控变体，避免同角色玩家退化为完全同质。
        fallback_sources = ("counterfactual", "red_team", "recombined", "research_prior")
        source_index = 0
        while len(result) < int(self.config.candidate_count):
            source_type = fallback_sources[source_index % len(fallback_sources)]
            source_index += 1
            candidate = self._diversified_fallback(
                parent,
                role,
                round_index,
                source_type,
                len(result),
            )
            if all(candidate.fingerprint != existing.fingerprint for existing in result):
                result.append(candidate)
        return self._dedupe_specs(result)

    async def _critic_candidate(
        self,
        *,
        role: str,
        profile: Any,
        candidate: HarnessSpec,
        analysis: ReplayAnalysis,
    ) -> dict[str, list[str]]:
        payload = {
            "role": role,
            "own_role_markdown": {"base_md": profile.base, "strategy_md": profile.strategy},
            "candidate_harness": candidate.as_dict(),
            "replay_analysis": analysis.as_dict(),
            "fixed_boundaries": [
                "不能改写 GameEngine、规则、base.md、信息可见性或行动校验。",
                "不能执行代码、访问网络、读取文件或把其他角色 skill 当作输入。",
                "不能把未经验证的单局巧合写成硬规则。",
                "必须允许公开信息变化后重新规划。",
            ],
        }
        raw = await self._complete_json(
            system=render_prompt("meta_harness_critic_system.txt", role=role),
            messages=[{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
            max_tokens=int(self.config.critic_max_tokens),
        )
        source = raw.get("critique", raw) if isinstance(raw, dict) else {}
        if not isinstance(source, Mapping):
            source = {}
        return {key: _string_list(source.get(key), limit=12, item_limit=1000) for key in CRITIQUE_KEYS}

    async def _collect_research_sources(
        self, role: str, analysis: ReplayAnalysis
    ) -> list[dict[str, Any]]:
        """用固定、可审计的查询模板调用可选 provider。

        研究资料只作为带来源的待验证假设输入设计阶段；它不会绕过红队或直接成为
        active 策略。默认 provider 为 None，因此普通训练完全不产生网络请求。
        """

        queries: list[str] = []
        for finding in analysis.summary.get("failure_modes", []) + analysis.summary.get("counterfactuals", []):
            query = _text(f"{role} 狼人杀 策略经验 {finding}", limit=500)
            if query and query not in queries:
                queries.append(query)
            if len(queries) >= int(self.config.research_query_count):
                break
        sources: list[dict[str, Any]] = []
        for query in queries:
            try:
                found = await call_search_provider(
                    self.research_provider,
                    query,
                    max_results=int(self.config.research_max_results),
                )
            except Exception as error:
                sources.append(
                    {
                        "query": query,
                        "provider_error": type(error).__name__,
                        "summary": "研究 provider 调用失败；不得把失败当成策略证据。",
                    }
                )
                continue
            for source in found:
                item = source.as_dict() if isinstance(source, ResearchSource) else dict(source)
                item["query"] = query
                sources.append(item)
        return sources[: int(self.config.research_query_count) * int(self.config.research_max_results)]

    @staticmethod
    def _normalize_analysis_batch(raw: object, game_ids: list[str]) -> dict[str, Any]:
        source = raw.get("analysis", raw) if isinstance(raw, Mapping) else {}
        if not isinstance(source, Mapping):
            source = {}
        result = {key: _string_list(source.get(key), limit=24) for key in ANALYSIS_KEYS}
        notes = source.get("game_notes")
        if isinstance(notes, list):
            result["evidence"].extend(
                _string_list(
                    [
                        f"{item.get('game_id')}: {item.get('finding')}"
                        for item in notes
                        if isinstance(item, Mapping) and str(item.get("game_id")) in game_ids
                    ],
                    limit=24,
                )
            )
        return {key: list(dict.fromkeys(value))[:24] for key, value in result.items()}

    def _normalise_candidate(
        self,
        value: Mapping[str, Any],
        *,
        role: str,
        parent: HarnessSpec,
        round_index: int,
        index: int,
    ) -> HarnessSpec:
        raw = dict(value)
        raw["role"] = role
        raw["parent_id"] = parent.harness_id
        try:
            raw["version"] = max(parent.version + 1, int(raw.get("version", parent.version + 1)))
        except (TypeError, ValueError):
            raw["version"] = parent.version + 1
        raw["source_type"] = str(raw.get("source_type") or ("replay_mutation" if index == 0 else "counterfactual"))
        if raw["source_type"] not in HARNESS_SOURCE_TYPES - {"baseline"}:
            raw["source_type"] = "replay_mutation"
        raw["cards"] = list(raw.get("cards") or [])[: int(self.config.max_cards)]
        provisional = HarnessSpec.from_mapping(raw, role=role)
        # ID 基于内容而非模型任意命名，避免重复节点覆盖 archive。
        digest = provisional.fingerprint[:10]
        return HarnessSpec.from_mapping(
            {**provisional.as_dict(), "harness_id": f"r{int(round_index)}-{role}-{provisional.source_type}-{digest}"},
            role=role,
        )

    @staticmethod
    def _fallback_candidate(parent: HarnessSpec, role: str, round_index: int, source_type: str, index: int) -> HarnessSpec:
        data = parent.as_dict()
        data.update(
            {
                "role": role,
                "version": parent.version + 1,
                "parent_id": parent.harness_id,
                "source_type": source_type,
                "rationale": ["模型候选不可用，保留经过 schema 校验的保守变体。"],
            }
        )
        data["harness_id"] = f"r{int(round_index)}-{role}-{source_type}-fallback{index}"
        return HarnessSpec.from_mapping(data, role=role)

    @staticmethod
    def _diversified_fallback(
        parent: HarnessSpec,
        role: str,
        round_index: int,
        source_type: str,
        index: int,
    ) -> HarnessSpec:
        """生成结构相异但仍保守的候选，作为模型少返回候选时的安全兜底。"""

        data = parent.as_dict()
        data.pop("fingerprint", None)
        data.update(
            {
                "role": role,
                "version": parent.version + 1,
                "parent_id": parent.harness_id,
                "source_type": source_type,
                "rationale": [
                    f"受控 {source_type} 变体：用于同角色姿态多样性和反例验证，不代表已证实规律。"
                ],
            }
        )
        context = dict(data.get("context_policy") or {})
        router = dict(data.get("tactic_router") or {})
        planning = dict(data.get("planning_policy") or {})
        memory = dict(data.get("memory_policy") or {})
        if source_type == "counterfactual":
            planning["counterfactual_question"] = "如果我现在采取相反姿态，谁会最先受益？"
            router["default_posture"] = "先保留两种解释，再用下一条公开证据区分"
        elif source_type == "red_team":
            router["default_posture"] = "主动寻找当前主流判断的反例，但保留撤退条件"
            context["max_visible_events"] = max(12, int(context.get("max_visible_events", 28)) - 6)
        elif source_type == "recombined":
            memory["max_notes"] = max(6, int(memory.get("max_notes", 12)) - 2)
            planning["compare_alternatives"] = True
        elif source_type == "research_prior":
            router["default_posture"] = "将外部经验视为低置信度假设，等待实战证据"
            context["strategy_mode"] = "bounded"
        data["context_policy"] = context
        data["tactic_router"] = router
        data["planning_policy"] = planning
        data["memory_policy"] = memory
        data["harness_id"] = f"r{int(round_index)}-{role}-{source_type}-variant{index}"
        return HarnessSpec.from_mapping(data, role=role)

    @staticmethod
    def _resolve_parent(role: str, parent: HarnessSpec | Mapping[str, Any] | None, profile: Any) -> HarnessSpec:
        if parent is not None:
            return HarnessSpec.from_mapping(parent, role=role)
        return default_harness_for_profile(profile)

    @staticmethod
    def _dedupe_specs(specs: list[HarnessSpec]) -> list[HarnessSpec]:
        result: list[HarnessSpec] = []
        fingerprints: set[str] = set()
        for spec in specs:
            if spec.fingerprint in fingerprints:
                continue
            fingerprints.add(spec.fingerprint)
            result.append(spec)
        return result

    def _score_candidate(
        self,
        candidate: HarnessSpec,
        critique: Mapping[str, list[str]],
        analysis: ReplayAnalysis,
        *,
        static_evaluation: Mapping[str, Any] | None = None,
    ) -> tuple[float, bool]:
        hard = len(critique.get("hard_rule_violations", []))
        leakage = len(critique.get("information_leakage_risks", []))
        unsupported = len(critique.get("unsupported_claims", []))
        complexity = len(critique.get("complexity_concerns", []))
        evidence_count = len(analysis.summary.get("evidence", []))
        static_score = float((static_evaluation or {}).get("score", 0.0) or 0.0)
        static_passed = bool((static_evaluation or {}).get("passed"))
        static_violations = len((static_evaluation or {}).get("violations", []) or [])
        score = 100.0 - 35.0 * hard - 25.0 * leakage - 7.0 * unsupported - 3.0 * complexity
        score += min(12.0, evidence_count * 0.5)
        score += min(5.0, len(candidate.cards) * 0.5)
        score += min(10.0, static_score / 10.0)
        score -= min(30.0, static_violations * 10.0)
        critic_failed = any(
            "阶段失败" in item or "不得自动晋升" in item
            for values in critique.values()
            for item in values
        )
        analysis_failed = any(
            "回放分析失败" in item for item in analysis.summary.get("uncertainties", [])
        )
        passed = (
            hard == 0
            and leakage == 0
            and len(candidate.cards) > 0
            and static_passed
            and not critic_failed
            and not analysis_failed
        )
        return max(-100.0, min(120.0, score)), passed

    @staticmethod
    def _select_assessment(assessments: list[CandidateAssessment], parent: HarnessSpec) -> CandidateAssessment:
        if not assessments:
            raise ValueError("Meta-Agent 没有生成任何候选")
        passed = [item for item in assessments if item.passed]
        pool = passed or assessments
        return max(pool, key=lambda item: (item.score, item.candidate.source_type != parent.source_type, item.candidate.harness_id))

    async def _complete_json(self, *, system: str, messages: list[dict[str, Any]], max_tokens: int) -> dict[str, Any]:
        if self.request_coordinator is not None:
            return await self.request_coordinator.complete_json(
                self.model_client, system=system, messages=messages, max_tokens=max_tokens
            )
        complete_json = self.model_client.complete_json
        if inspect.iscoroutinefunction(complete_json):
            value = await complete_json(system=system, messages=messages, max_tokens=max_tokens)
        else:
            value = await asyncio.to_thread(
                complete_json, system=system, messages=messages, max_tokens=max_tokens
            )
        if not isinstance(value, dict):
            raise ValueError("Meta-Agent 模型没有返回 JSON 对象")
        return value

    @staticmethod
    def _research_payload(sources: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for source in sources:
            if not isinstance(source, Mapping):
                continue
            result.append(
                {
                    "title": _text(source.get("title"), limit=300),
                    "url": _text(source.get("url"), limit=1000),
                    "summary": _text(source.get("summary"), limit=2000),
                    "retrieved_at": _text(source.get("retrieved_at"), limit=80),
                    "content_sha256": _text(source.get("content_sha256"), limit=128),
                    "credibility": _text(source.get("credibility"), limit=80),
                    "provider": _text(source.get("provider"), limit=100),
                    "query": _text(source.get("query"), limit=500),
                    "provider_error": _text(source.get("provider_error"), limit=200),
                }
            )
        return result[:30]

    @staticmethod
    def _dedupe_sources(sources: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for source in sources:
            if not isinstance(source, Mapping):
                continue
            key = (
                _text(source.get("url"), limit=1000),
                _text(source.get("content_sha256"), limit=128),
            )
            if key in seen and any(key):
                continue
            seen.add(key)
            result.append(dict(source))
        return result[:30]

    @staticmethod
    def _dedupe(values: Iterable[str], *, limit: int) -> list[str]:
        result: list[str] = []
        for value in values:
            if value and value not in result:
                result.append(value)
            if len(result) >= limit:
                break
        return result


# 对外提供更短的语义名称；实现仍明确属于 Season 2 的固定元流程。
MetaAgent = TaskAgentMetaAgent

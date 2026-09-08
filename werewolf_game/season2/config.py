"""从外置 JSON 文件读取 Season 2 超参数。"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping

from ..core.constants import ALL_ROLES


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    # 配置采用严格的对象结构，避免把拼写错误的列表/字符串当成合法配置。
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} 必须是 JSON 对象")
    return value


def _positive(value: object, name: str, *, allow_zero: bool = False) -> int:
    # 所有额度在启动时校验；运行过程中只使用已经冻结的值。
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} 必须是整数") from error
    minimum = 0 if allow_zero else 1
    if result < minimum:
        raise ValueError(f"{name} 必须不少于 {minimum}")
    return result


@dataclass(frozen=True)
class PathConfig:
    archive_root: Path
    evaluation_root: Path
    operation_root: Path
    external_knowledge_root: Path
    pi_root: Path


@dataclass(frozen=True)
class EvolutionConfig:
    roles: tuple[str, ...]
    random_seed: str
    max_children_per_node: int
    max_tree_depth: int
    max_generated_candidates_per_role: int
    max_successful_evolutions_per_role: int
    children_per_expansion: int
    role_scheduler: str


@dataclass(frozen=True)
class EvaluationConfig:
    games_per_pending_node: int
    player_count: int
    optional_roles: tuple[str, ...]
    game_concurrency: int
    model_max_in_flight: int
    decision_timeout_seconds: float
    max_decision_retries: int
    max_tool_calls_per_decision: int
    max_tool_result_tokens: int
    max_prompt_chars: int
    max_failed_games: int
    max_fallback_rate: float
    max_invalid_decisions: int
    discard_all_losses: bool


@dataclass(frozen=True)
class MetaConfig:
    max_output_tokens: int
    max_tool_calls: int
    max_tool_result_tokens: int
    max_attempts: int
    required_resources: tuple[str, ...]
    enabled_tools: tuple[str, ...]


@dataclass(frozen=True)
class PiConfig:
    command: tuple[str, ...]
    model: str
    # Pi 的 Node 运行时。使用 wolf conda 环境中的 Node 22，避免误用系统 Node。
    node_bin: Path | None
    # Pi 的 provider 配置可能包含内部网关地址，只允许从本机未跟踪目录读取。
    agent_dir: Path | None
    timeout_seconds: float
    max_attempts: int
    use_bwrap: bool
    candidate_smoke_test: bool
    readonly_paths: tuple[Path, ...] = field(default_factory=tuple)
    allowed_files: tuple[str, ...] = ("task_agent.py", "task.md")
    environment_allowlist: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class Season2Config:
    source_path: Path
    paths: PathConfig
    evolution: EvolutionConfig
    evaluation: EvaluationConfig
    meta: MetaConfig
    pi: PiConfig

    def manifest(self) -> dict[str, Any]:
        """返回不含密钥、可直接写入实验记录的配置快照。"""

        return {
            "source_path": str(self.source_path),
            "paths": {
                key: str(value) for key, value in vars(self.paths).items()
            },
            "evolution": {
                **vars(self.evolution),
                "roles": list(self.evolution.roles),
            },
            "evaluation": {
                **vars(self.evaluation),
                "optional_roles": list(self.evaluation.optional_roles),
            },
            "meta": {
                **vars(self.meta),
                "required_resources": list(self.meta.required_resources),
                "enabled_tools": list(self.meta.enabled_tools),
            },
            "pi": {
                **vars(self.pi),
                "command": list(self.pi.command),
                "agent_dir": str(self.pi.agent_dir) if self.pi.agent_dir else None,
                "node_bin": str(self.pi.node_bin) if self.pi.node_bin else None,
                "readonly_paths": [str(path) for path in self.pi.readonly_paths],
                "allowed_files": list(self.pi.allowed_files),
                "environment_allowlist": list(self.pi.environment_allowlist),
            },
        }


def load_season2_config(path: str | Path) -> Season2Config:
    # 相对路径统一相对于配置文件解析，保证从任意当前工作目录启动都能复现。
    source = Path(path).resolve()
    data = _mapping(json.loads(source.read_text(encoding="utf-8")), "配置根节点")
    base = source.parent

    def resolve(value: object) -> Path:
        candidate = Path(str(value))
        return (candidate if candidate.is_absolute() else base / candidate).resolve()

    # 分区解析后再构造不可变 dataclass，防止下游代码随意修改配置。
    paths = _mapping(data.get("paths"), "paths")
    evolution = _mapping(data.get("evolution"), "evolution")
    evaluation = _mapping(data.get("evaluation"), "evaluation")
    meta = _mapping(data.get("meta"), "meta")
    pi = _mapping(data.get("pi"), "pi")

    roles = tuple(str(role) for role in evolution.get("roles", sorted(ALL_ROLES)))
    # 角色集合决定 archive 的根结构；未知角色必须在初始化前报错。
    unknown_roles = sorted(set(roles) - set(ALL_ROLES))
    if unknown_roles:
        raise ValueError("evolution.roles 包含未知角色：" + "、".join(unknown_roles))
    if not roles:
        raise ValueError("evolution.roles 不能为空")

    role_scheduler = str(evolution.get("role_scheduler", "least_evolved_random"))
    if role_scheduler not in {"least_evolved_random", "random"}:
        raise ValueError("role_scheduler 仅支持 least_evolved_random 或 random")

    allowed_files = tuple(str(item) for item in pi.get("allowed_files", ("task_agent.py", "task.md")))
    # Pi 只能修改工作区内的白名单文件，禁止通过绝对路径或 .. 越界。
    if not allowed_files or any(Path(item).is_absolute() or ".." in Path(item).parts for item in allowed_files):
        raise ValueError("pi.allowed_files 必须是候选工作区内的相对路径")

    config = Season2Config(
        source_path=source,
        paths=PathConfig(
            archive_root=resolve(paths.get("archive_root", "../records/season2/evolution/archive")),
            evaluation_root=resolve(paths.get("evaluation_root", "../records/season2/evolution/evaluations")),
            operation_root=resolve(paths.get("operation_root", "../records/season2/evolution/operations")),
            external_knowledge_root=resolve(paths.get("external_knowledge_root", "../resources/season2")),
            pi_root=resolve(paths.get("pi_root", "../runtime/pi")),
        ),
        evolution=EvolutionConfig(
            roles=roles,
            random_seed=str(evolution.get("random_seed", "season2-evolution")),
            max_children_per_node=_positive(evolution.get("max_children_per_node", 4), "max_children_per_node"),
            max_tree_depth=_positive(evolution.get("max_tree_depth", 8), "max_tree_depth"),
            max_generated_candidates_per_role=_positive(
                evolution.get("max_generated_candidates_per_role", 60),
                "max_generated_candidates_per_role",
            ),
            max_successful_evolutions_per_role=_positive(
                evolution.get("max_successful_evolutions_per_role", 20),
                "max_successful_evolutions_per_role",
            ),
            children_per_expansion=_positive(evolution.get("children_per_expansion", 4), "children_per_expansion"),
            role_scheduler=role_scheduler,
        ),
        evaluation=EvaluationConfig(
            games_per_pending_node=_positive(evaluation.get("games_per_pending_node", 20), "games_per_pending_node"),
            player_count=_positive(evaluation.get("player_count", 12), "player_count"),
            optional_roles=tuple(str(item) for item in evaluation.get("optional_roles", ())),
            game_concurrency=_positive(evaluation.get("game_concurrency", 1), "game_concurrency"),
            model_max_in_flight=_positive(evaluation.get("model_max_in_flight", 4), "model_max_in_flight"),
            decision_timeout_seconds=float(evaluation.get("decision_timeout_seconds", 180)),
            max_decision_retries=_positive(evaluation.get("max_decision_retries", 2), "max_decision_retries", allow_zero=True),
            max_tool_calls_per_decision=_positive(evaluation.get("max_tool_calls_per_decision", 5), "max_tool_calls_per_decision", allow_zero=True),
            max_tool_result_tokens=_positive(evaluation.get("max_tool_result_tokens", 800), "max_tool_result_tokens"),
            max_prompt_chars=_positive(evaluation.get("max_prompt_chars", 12000), "max_prompt_chars"),
            max_failed_games=_positive(evaluation.get("max_failed_games", 3), "max_failed_games", allow_zero=True),
            max_fallback_rate=float(evaluation.get("max_fallback_rate", 0.2)),
            max_invalid_decisions=_positive(evaluation.get("max_invalid_decisions", 10), "max_invalid_decisions", allow_zero=True),
            discard_all_losses=bool(evaluation.get("discard_all_losses", True)),
        ),
        meta=MetaConfig(
            max_output_tokens=_positive(meta.get("max_output_tokens", 4000), "meta.max_output_tokens"),
            max_tool_calls=_positive(meta.get("max_tool_calls", 12), "meta.max_tool_calls", allow_zero=True),
            max_tool_result_tokens=_positive(meta.get("max_tool_result_tokens", 8000), "meta.max_tool_result_tokens"),
            max_attempts=_positive(meta.get("max_attempts", 2), "meta.max_attempts"),
            required_resources=tuple(str(item) for item in meta.get("required_resources", ("task", "boundaries", "current_agent", "random_game_replay"))),
            enabled_tools=tuple(str(item) for item in meta.get("enabled_tools", ())),
        ),
        pi=PiConfig(
            command=tuple(str(item) for item in pi.get("command", ("./pi-test.sh",))),
            model=str(pi.get("model", "")),
            node_bin=(resolve(pi["node_bin"]) if pi.get("node_bin") else None),
            agent_dir=(
                resolve(pi["agent_dir"])
                if pi.get("agent_dir")
                else None
            ),
            timeout_seconds=float(pi.get("timeout_seconds", 900)),
            max_attempts=_positive(pi.get("max_attempts", 1), "pi.max_attempts"),
            use_bwrap=bool(pi.get("use_bwrap", True)),
            candidate_smoke_test=bool(pi.get("candidate_smoke_test", True)),
            readonly_paths=tuple(resolve(item) for item in pi.get("readonly_paths", ())),
            allowed_files=allowed_files,
            environment_allowlist=tuple(str(item) for item in pi.get("environment_allowlist", ())),
        ),
    )
    if not 7 <= config.evaluation.player_count <= 12:
        raise ValueError("evaluation.player_count 必须在 7–12 之间")
    if not 0 <= config.evaluation.max_fallback_rate <= 1:
        raise ValueError("max_fallback_rate 必须在 0–1 之间")
    if not config.pi.command:
        raise ValueError("pi.command 不能为空")
    if config.evolution.children_per_expansion > config.evolution.max_children_per_node:
        raise ValueError("children_per_expansion 不能大于 max_children_per_node")
    if (
        config.evolution.children_per_expansion
        > config.evaluation.games_per_pending_node
    ):
        raise ValueError(
            "children_per_expansion 不能大于 games_per_pending_node；"
            "否则无法为每个分支提供不重复的评测回放"
        )
    return config

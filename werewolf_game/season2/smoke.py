"""不调用外部模型的候选最小完整对局检查。"""

from __future__ import annotations

import json
from typing import Any, Mapping

from ..core.engine import GameEngine
from ..core.rules import create_rules_for_player_count
from ..core.runner import GameRunner
from .archive import EvolutionArchive
from .config import Season2Config
from .runtime import CandidateModuleLoader, VersionedTaskAgentParticipant


def _find_request(value: object) -> Mapping[str, Any] | None:
    # 假模型只从候选实际收到的 request 中取合法 action，避免 smoke 测试绕过契约。
    if isinstance(value, Mapping):
        if isinstance(value.get("allowed_actions"), list):
            return value
        for child in value.values():
            found = _find_request(child)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_request(child)
            if found is not None:
                return found
    elif isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return None
        return _find_request(parsed)
    return None


class ContractSmokeModel:
    async def complete_json(self, **kwargs: Any) -> dict[str, Any]:
        request = _find_request(kwargs.get("messages", []))
        if request is None:
            request = _find_request(kwargs.get("system", ""))
        if request is None:
            raise ValueError("候选没有把当前合法行动契约提供给模型")
        allowed_actions = request.get("allowed_actions")
        if not isinstance(allowed_actions, list) or not allowed_actions:
            raise ValueError("候选模型请求缺少 allowed_actions")
        allowed = allowed_actions[0]
        action: dict[str, Any] = {"kind": allowed["kind"]}
        if allowed["kind"] in {"speak", "last_words"}:
            action["text"] = "好"
        elif allowed.get("target_ids"):
            action["target_id"] = allowed["target_ids"][0]
        return action


async def smoke_test_candidate(
    *,
    config: Season2Config,
    archive: EvolutionArchive,
    focus_node_id: str,
) -> dict[str, Any]:
    # smoke 只验证候选能被加载并完成完整流程，不产生真实评测胜率。
    focus = archive.node(focus_node_id)
    role_nodes = {
        role: (
            focus_node_id
            if role == focus.role
            else archive.retained_nodes(role)[0].node_id
        )
        for role in config.evolution.roles
    }
    rules = create_rules_for_player_count(
        config.evaluation.player_count, config.evaluation.optional_roles
    )
    players = [
        {"id": f"p{index}", "name": f"P{index}"}
        for index in range(1, config.evaluation.player_count + 1)
    ]
    loader = CandidateModuleLoader(archive)
    model = ContractSmokeModel()
    participants = {
        player["id"]: VersionedTaskAgentParticipant(
            player_id=player["id"],
            model_client=model,
            role_nodes=role_nodes,
            archive=archive,
            module_loader=loader,
            max_decision_retries=0,
            max_tool_calls_per_decision=0,
        )
        for player in players
    }
    report = await GameRunner(
        engine=GameEngine(
            game_id=f"smoke-{focus_node_id}",
            players=players,
            rules=rules,
            seed=f"smoke-{focus_node_id}",
        ),
        participants=participants,
        record_store=False,
        decision_timeout_seconds=min(30.0, config.evaluation.decision_timeout_seconds),
    ).run()
    if report["fallback_count"] or report["errors"]:
        raise ValueError(
            f"候选 smoke 出现 fallback={report['fallback_count']} errors={len(report['errors'])}"
        )
    return {
        "winner": report["public_state"].get("winner"),
        "round": report["public_state"].get("round"),
        "decision_count": report["decision_count"],
        "fallback_count": report["fallback_count"],
        "role_nodes": role_nodes,
    }

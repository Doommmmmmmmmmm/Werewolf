"""异步角色与父节点选择。"""

from __future__ import annotations

import hashlib
import random

from .archive import EvolutionArchive, EvolutionNode
from .config import Season2Config


class EvolutionScheduler:
    def __init__(self, archive: EvolutionArchive, config: Season2Config) -> None:
        self.archive = archive
        self.config = config
        # 调度随机性由配置和当前进化计数决定；相同步骤可稳定复现选择结果。
        seed_material = (
            f"{config.evolution.random_seed}:"
            f"{sum(archive.role_generated_candidate_count(role) for role in config.evolution.roles)}:"
            f"{sum(archive.role_successful_evolution_count(role) for role in config.evolution.roles)}"
        )
        seed = int(hashlib.sha256(seed_material.encode("utf-8")).hexdigest(), 16)
        self.random = random.Random(seed)

    def select(self, *, exclude_node_ids: set[str] | None = None) -> EvolutionNode | None:
        # 先筛选节点，再按角色异步轮转，最后从该角色全部合格节点中随机选择。
        eligible_by_role = {
            role: self.archive.eligible_nodes(role)
            for role in self.config.evolution.roles
        }
        eligible_by_role = {
            role: nodes for role, nodes in eligible_by_role.items() if nodes
        }
        excluded = exclude_node_ids or set()
        eligible_by_role = {
            role: [node for node in nodes if node.node_id not in excluded]
            for role, nodes in eligible_by_role.items()
        }
        eligible_by_role = {
            role: nodes for role, nodes in eligible_by_role.items() if nodes
        }
        if not eligible_by_role:
            return None

        roles = list(eligible_by_role)
        if self.config.evolution.role_scheduler == "least_evolved_random":
            minimum = min(
                self.archive.role_successful_evolution_count(role) for role in roles
            )
            roles = [
                role for role in roles
                if self.archive.role_successful_evolution_count(role) == minimum
            ]
        # 调度比较的是成功保留节点数，不是已经生成但可能失败的候选数。
        role = self.random.choice(sorted(roles))
        nodes = eligible_by_role[role]
        return self.random.choice(sorted(nodes, key=lambda node: node.node_id))

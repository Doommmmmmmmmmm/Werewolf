"""Meta-Agent 可审计的只读资源目录。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
from typing import Any

from ..prompts import RoleProfileStore
from .archive import EvolutionArchive
from .config import Season2Config


TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "list_resources": {
        "name": "list_resources",
        "description": "列出本次诊断可用的只读资料及参数。",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    "read_current_agent": {
        "name": "read_current_agent",
        "description": "读取当前节点的 task_agent.py 或 task.md。",
        "parameters": {
            "type": "object",
            "properties": {"filename": {"type": "string", "enum": ["task_agent.py", "task.md"]}},
            "required": ["filename"],
            "additionalProperties": False,
        },
    },
    "read_evolution_path": {
        "name": "read_evolution_path",
        "description": "读取 base 到当前节点的节点信息和祖先 patch。",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    "read_evolution_tree": {
        "name": "read_evolution_tree",
        "description": "读取当前角色完整进化树索引。",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    "read_evolution_contract": {
        "name": "read_evolution_contract",
        "description": "按需读取本次实验的进化状态机、预算和节点筛选规则。",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    "read_evaluation_summary": {
        "name": "read_evaluation_summary",
        "description": "读取当前节点最近一次评测汇总。",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    "read_game_replay": {
        "name": "read_game_replay",
        "description": "分页读取当前节点评测中的一局完整审计回放。",
        "parameters": {
            "type": "object",
            "properties": {
                "game_index": {"type": "integer", "minimum": 0},
                "event_offset": {"type": "integer", "minimum": 0},
                "event_limit": {"type": "integer", "minimum": 1, "maximum": 200}
            },
            "required": ["game_index"],
            "additionalProperties": False,
        },
    },
    "read_other_branch": {
        "name": "read_other_branch",
        "description": "读取同角色另一节点的 patch、状态和胜负摘要。",
        "parameters": {
            "type": "object",
            "properties": {"node_id": {"type": "string"}},
            "required": ["node_id"],
            "additionalProperties": False,
        },
    },
    "read_previous_diagnosis": {
        "name": "read_previous_diagnosis",
        "description": "读取同角色某节点关联的历史诊断。",
        "parameters": {
            "type": "object",
            "properties": {"node_id": {"type": "string"}},
            "required": ["node_id"],
            "additionalProperties": False,
        },
    },
    "list_skills": {
        "name": "list_skills",
        "description": "列出所有可用 Skill 的名称和短描述，不返回完整攻略。",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    "read_skill": {
        "name": "read_skill",
        "description": "读取指定 Skill 的完整内容，包括短描述和长描述。",
        "parameters": {
            "type": "object",
            "properties": {
                "skill_name": {"type": "string", "pattern": "^[A-Za-z0-9][A-Za-z0-9._-]*$"}
            },
            "required": ["skill_name"],
            "additionalProperties": False,
        },
    },
}


class MetaResourceCatalog:
    def __init__(
        self,
        *,
        config: Season2Config,
        archive: EvolutionArchive,
        node_id: str,
        required_replay_item: dict[str, Any] | None = None,
    ) -> None:
        self.config = config
        self.archive = archive
        self.node = archive.node(node_id)
        self.profile = RoleProfileStore().profile(self.node.role)
        self.tool_trace: list[dict[str, Any]] = []
        # 分支扩展时由调用方提前选定一局，保证同一批诊断各自使用不同回放。
        self.required_replay_item = required_replay_item
        # catalog 只保存当前节点上下文；每次工具调用都会留下参数和结果，便于审计。
        unknown = set(config.meta.enabled_tools) - set(TOOL_SCHEMAS)
        if unknown:
            raise ValueError("未知 Meta-Agent 工具：" + "、".join(sorted(unknown)))

    def tools(self) -> list[dict[str, Any]]:
        # 工具是否可用由外部配置决定，模型不能在运行时自行增加工具。
        return [TOOL_SCHEMAS[name] for name in self.config.meta.enabled_tools]

    def required_resource_text(self) -> str:
        # 必读内容直接注入；可选资料不会因为存在于磁盘就自动进入模型上下文。
        chunks: list[str] = []
        for name in self.config.meta.required_resources:
            value = self._required(name)
            chunks.append(f"### {name}\n{value}")
        return "\n\n".join(chunks)

    def execute(self, name: str, arguments: dict[str, Any]) -> object:
        # 所有读取都经过统一入口，便于做权限检查、记录和后续预算统计。
        if name not in self.config.meta.enabled_tools:
            result: object = {"error": f"工具未启用：{name}"}
        else:
            handler = getattr(self, f"_tool_{name}", None)
            result = handler(arguments) if callable(handler) else {"error": f"工具未实现：{name}"}
        self.tool_trace.append({"tool": name, "arguments": arguments, "result": result})
        return result

    def _required(self, name: str) -> str:
        if name == "task":
            # task 同时包含角色目标、固定规则和候选 task.md，但不暴露其他角色资料。
            return json.dumps(
                {
                    "role": self.node.role,
                    "team_goal": "wolf 获得狼人阵营胜利；其他角色获得好人阵营胜利",
                    "role_rule": self.profile.base,
                    "task": (self.archive.node_code_directory(self.node.node_id) / "task.md").read_text(encoding="utf-8"),
                },
                ensure_ascii=False,
                indent=2,
            )
        if name == "boundaries":
            return (
                "不得修改 Game Engine、游戏规则、身份能力、胜负裁决、信息可见性、行动 schema、"
                "审计记录或模型密钥；不得新增网络、任意文件、shell、进程或隐藏状态访问。"
            )
        if name == "current_agent":
            return (self.archive.node_code_directory(self.node.node_id) / "task_agent.py").read_text(encoding="utf-8")
        if name == "random_game_replay":
            # 默认提供一局完整回放；随机种子固定，便于重现相同诊断输入。
            return json.dumps(self._random_game_replay(), ensure_ascii=False, indent=2)
        if name == "evaluation_summary":
            return json.dumps(self._evaluation_summary(), ensure_ascii=False, indent=2)
        if name == "evolution_tree":
            return json.dumps(self.archive.tree_manifest(self.node.role), ensure_ascii=False, indent=2)
        raise ValueError(f"不支持的必读资料：{name}")

    def _tool_list_resources(self, arguments: dict[str, Any]) -> object:
        del arguments
        return {
            "required_resources": list(self.config.meta.required_resources),
            "external_knowledge": {
                "description": "进化前冻结的狼人杀 Skill 卡片；通过 list_skills/read_skill 按需读取，不直接注入 Task-Agent",
                "tool": "list_skills/read_skill",
            },
            "tools": [
                {"name": name, "description": TOOL_SCHEMAS[name]["description"]}
                for name in self.config.meta.enabled_tools
            ],
        }

    def _tool_read_current_agent(self, arguments: dict[str, Any]) -> object:
        filename = str(arguments.get("filename", ""))
        if filename not in {"task_agent.py", "task.md"}:
            return {"error": "filename 不合法"}
        return {
            "node_id": self.node.node_id,
            "filename": filename,
            "content": (self.archive.node_code_directory(self.node.node_id) / filename).read_text(encoding="utf-8"),
        }

    def _tool_read_evolution_path(self, arguments: dict[str, Any]) -> object:
        del arguments
        lineage = []
        for node in self.archive.lineage(self.node.node_id):
            patch = ""
            if node.patch_path and Path(node.patch_path).exists():
                patch = Path(node.patch_path).read_text(encoding="utf-8")
            lineage.append({"node": vars(node), "patch": patch})
        return {"lineage": lineage}

    def _tool_read_evolution_tree(self, arguments: dict[str, Any]) -> object:
        del arguments
        return self.archive.tree_manifest(self.node.role)

    def _tool_read_evolution_contract(self, arguments: dict[str, Any]) -> object:
        """返回进化流程说明；它是可选资料，不自动注入诊断 prompt。"""

        del arguments
        evaluation = self.config.evaluation
        evolution = self.config.evolution
        node = self.node
        return {
            "purpose": "Meta-Agent 为当前角色设计下一代 Task-Agent；外部状态机负责评测、建树和预算执行。",
            "current_node": {
                "node_id": node.node_id,
                "role": node.role,
                "status": node.status,
                "depth": node.depth,
                "parent_id": node.parent_id,
                "children": list(node.children),
                "generated_candidate_count": self.archive.role_generated_candidate_count(node.role),
                "successful_evolution_count": self.archive.role_successful_evolution_count(node.role),
                "pending_candidate_count": self.archive.role_pending_count(node.role),
            },
            "node_states": {
                "pending": "待评估；先运行配置数量的对局，再决定 retained 或 discarded",
                "retained": "保留在候选池中；在预算允许时可以生成子节点",
                "discarded": "舍弃；不进入候选池，也不再生成子节点",
            },
            "evaluation": {
                "games_per_pending_node": evaluation.games_per_pending_node,
                "player_count": evaluation.player_count,
                "game_concurrency": evaluation.game_concurrency,
                "model_max_in_flight": evaluation.model_max_in_flight,
                "max_failed_games": evaluation.max_failed_games,
                "max_fallback_rate": evaluation.max_fallback_rate,
                "max_invalid_decisions": evaluation.max_invalid_decisions,
                "discard_all_losses": evaluation.discard_all_losses,
                "focus_role_uses_current_node": True,
                "other_roles_use_random_retained_candidates": True,
                "quality_error_metrics_scope": "focus_role_only; aggregate game metrics are retained for diagnostics",
                "parent_child_comparison_required": False,
            },
            "evolution_budget": {
                "children_per_expansion": evolution.children_per_expansion,
                "max_children_per_node": evolution.max_children_per_node,
                "max_tree_depth": evolution.max_tree_depth,
                "max_generated_candidates_per_role": evolution.max_generated_candidates_per_role,
                "max_successful_evolutions_per_role": evolution.max_successful_evolutions_per_role,
                "role_scheduler": evolution.role_scheduler,
                "candidate_pool_extra_limit": None,
            },
            "process": [
                "一个 step 以产生新的 retained 子节点为成功终止条件；不是生成 pending 就结束。",
                "一个 step 可以连续处理多个节点，直到产生 retained 子节点或所有机会耗尽。",
                "pending 节点只做评测并决定保留或舍弃。",
                "retained 节点扩展时固定创建 children_per_expansion 个子节点。",
                "每个子节点单独调用一次 Meta-Agent 和 Pi；一次扩展抽取不同的游戏回放。",
                "子节点从父节点快照继承全部祖先修改；其他分支需要通过工具主动读取。",
                "不要求子节点胜过父节点，也不保证每次修改都是真实改进。",
            ],
            "not_evolvable": [
                "Game Engine",
                "游戏规则与胜负裁决",
                "信息可见性与行动 JSON schema",
                "工具实现、模型网关、审计记录和所有外部预算",
            ],
        }

    def _tool_read_evaluation_summary(self, arguments: dict[str, Any]) -> object:
        del arguments
        return self._evaluation_summary()

    def _tool_read_game_replay(self, arguments: dict[str, Any]) -> object:
        # 回放按 event_offset/event_limit 分页；分页本身由 ModelClient 计入工具预算。
        summary = self._evaluation_summary()
        game_index = int(arguments.get("game_index", -1))
        result = next(
            (item for item in summary.get("results", ()) if int(item.get("game_index", -2)) == game_index),
            None,
        )
        if not result:
            return {"error": f"评测中没有 game{game_index}"}
        record_path = Path(str(result.get("record_path", "")))
        if not record_path.exists():
            return {"error": "完整回放文件不存在"}
        record = json.loads(record_path.read_text(encoding="utf-8"))
        events = record.get("events") if isinstance(record.get("events"), list) else []
        offset = max(0, int(arguments.get("event_offset", 0)))
        limit = min(200, max(1, int(arguments.get("event_limit", 80))))
        return {
            "metadata": record.get("metadata"),
            "final_snapshot": record.get("final_snapshot"),
            "runner_events": record.get("runner_events"),
            "event_offset": offset,
            "event_limit": limit,
            "event_total": len(events),
            "events": events[offset: offset + limit],
        }

    def _random_game_replay(self) -> dict[str, Any]:
        """读取本次诊断指定的一局完整回放。"""

        if self.required_replay_item is not None:
            return self._load_replay_item(self.required_replay_item, selection="preselected")

        summary = self._evaluation_summary()
        results = summary.get("results") if isinstance(summary, dict) else None
        candidates: list[dict[str, Any]] = []
        if isinstance(results, list):
            for item in results:
                if not isinstance(item, dict):
                    continue
                record_path = Path(str(item.get("record_path", "")))
                if record_path.exists() and record_path.is_file():
                    candidates.append(item)
        if not candidates:
            return {
                "available": False,
                "reason": "当前节点尚无可用评测回放；应先完成基线评测",
                "node_id": self.node.node_id,
            }
        seed_text = f"{self.config.evolution.random_seed}:{self.node.node_id}:required-replay"
        selected = random.Random(
            int(hashlib.sha256(seed_text.encode("utf-8")).hexdigest(), 16)
        ).choice(sorted(candidates, key=lambda item: int(item.get("game_index", 0))))
        return self._load_replay_item(selected, selection="deterministic_random", seed_text=seed_text)

    def select_random_replay_items(self, count: int) -> list[dict[str, Any]]:
        """无放回抽样指定数量的评测条目，供一次多分支扩展使用。"""

        summary = self._evaluation_summary()
        results = summary.get("results") if isinstance(summary, dict) else None
        candidates: list[dict[str, Any]] = []
        if isinstance(results, list):
            for item in results:
                if not isinstance(item, dict):
                    continue
                record_path = Path(str(item.get("record_path", "")))
                if record_path.exists() and record_path.is_file():
                    candidates.append(item)
        candidates.sort(key=lambda item: int(item.get("game_index", 0)))
        requested = max(0, int(count))
        if len(candidates) < requested:
            raise ValueError(
                f"当前节点只有 {len(candidates)} 局可用回放，无法无重复抽取 {requested} 局"
            )
        seed_text = f"{self.config.evolution.random_seed}:{self.node.node_id}:expansion-replays"
        rng = random.Random(int(hashlib.sha256(seed_text.encode("utf-8")).hexdigest(), 16))
        return rng.sample(candidates, requested)

    def _load_replay_item(
        self,
        selected: dict[str, Any],
        *,
        selection: str,
        seed_text: str | None = None,
    ) -> dict[str, Any]:
        record_path = Path(str(selected["record_path"]))
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            return {
                "available": False,
                "reason": f"随机回放读取失败：{type(error).__name__}",
                "node_id": self.node.node_id,
                "record_path": str(record_path),
            }
        return {
            "available": True,
            "node_id": self.node.node_id,
            "game_index": selected.get("game_index"),
            "selection": selection,
            **({"selection_seed": seed_text} if seed_text else {}),
            "record_path": str(record_path),
            "replay": record,
        }

    def _tool_read_other_branch(self, arguments: dict[str, Any]) -> object:
        node_id = str(arguments.get("node_id", ""))
        try:
            node = self.archive.node(node_id)
        except KeyError:
            return {"error": "节点不存在"}
        if node.role != self.node.role:
            return {"error": "只能读取当前角色的其他分支"}
        patch = Path(node.patch_path).read_text(encoding="utf-8") if node.patch_path and Path(node.patch_path).exists() else ""
        summary: object = None
        if node.evaluation_summary_path and Path(node.evaluation_summary_path).exists():
            summary = json.loads(Path(node.evaluation_summary_path).read_text(encoding="utf-8"))
        return {"node": vars(node), "patch": patch, "evaluation_summary": summary}

    def _tool_read_previous_diagnosis(self, arguments: dict[str, Any]) -> object:
        node_id = str(arguments.get("node_id", ""))
        try:
            node = self.archive.node(node_id)
        except KeyError:
            return {"error": "节点不存在"}
        if node.role != self.node.role:
            return {"error": "只能读取当前角色的诊断"}
        if not node.diagnosis_path or not Path(node.diagnosis_path).exists():
            return {"error": "该节点没有关联诊断"}
        return json.loads(Path(node.diagnosis_path).read_text(encoding="utf-8"))

    def _tool_list_skills(self, arguments: dict[str, Any]) -> object:
        del arguments
        return {"skills": [self._skill_summary(path) for path in self._skill_files()]}

    def _tool_read_skill(self, arguments: dict[str, Any]) -> object:
        skill_name = str(arguments.get("skill_name", "")).strip()
        if not skill_name or Path(skill_name).name != skill_name:
            return {"error": "skill_name 不合法"}
        path = self.config.paths.external_knowledge_root / f"{skill_name}.md"
        if not path.exists() or not path.is_file():
            return {"error": f"不存在 Skill：{skill_name}"}
        parsed = self._skill_summary(path)
        parsed["content"] = path.read_text(encoding="utf-8", errors="replace")
        return parsed

    def _skill_files(self) -> list[Path]:
        """只读取 skills 根目录中的文件，明确禁止用子目录隐藏第二套分类结构。"""

        root = self.config.paths.external_knowledge_root
        if not root.exists() or not root.is_dir():
            return []
        return sorted(
            (path for path in root.iterdir() if path.is_file() and path.suffix.lower() == ".md"),
            key=lambda path: path.name,
        )

    @staticmethod
    def _skill_summary(path: Path) -> dict[str, str]:
        content = path.read_text(encoding="utf-8", errors="replace")
        short_marker = "## 短描述"
        short_description = ""
        if short_marker in content:
            section = content.split(short_marker, 1)[1]
            section = section.split("## ", 1)[0]
            short_description = " ".join(
                line.strip() for line in section.splitlines() if line.strip()
            )
        return {
            "name": path.stem,
            "short_description": short_description,
        }

    def _evaluation_summary(self) -> dict[str, Any]:
        # 节点尚未评测时返回明确的 unavailable，而不是伪造空胜率。
        if not self.node.evaluation_summary_path:
            return {"node_id": self.node.node_id, "available": False}
        path = Path(self.node.evaluation_summary_path)
        if not path.exists():
            return {"node_id": self.node.node_id, "available": False, "missing_path": str(path)}
        return json.loads(path.read_text(encoding="utf-8"))

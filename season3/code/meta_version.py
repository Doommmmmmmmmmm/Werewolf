"""Season 3 Meta-Agent 版本化管理器。

负责管理可进化的 Meta-Agent 工作链路（包括 meta_agent.py 及 meta_system/meta_task 提示词）。
支持按版本加载运行，并由元优化器进行代码级别的完全变异。
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
import sys
from typing import Any

from .archive import EvolutionArchive
from .config import Season2Config
from .progress import report


class MetaVersionManager:
    """管理可进化的 Meta-Agent 代码与 Prompt 版本。"""

    def __init__(self, config: Season2Config, archive: EvolutionArchive, model_client: Any) -> None:
        self.config = config
        self.archive = archive
        self.model_client = model_client
        self.meta_root = self.config.paths.archive_root / "meta_versions"
        self.meta_base_dir = Path(__file__).resolve().parents[1] / "meta_base"
        self.meta_root.mkdir(parents=True, exist_ok=True)

    def current_meta_version_id(self) -> str:
        manifest = self.archive._manifest
        return manifest.get("current_meta_version", "meta-v000")

    def initialize(self) -> None:
        """确保 meta-v000 (初始种子) 存在于 archive 中。"""
        manifest = self.archive._manifest
        if "meta_versions" not in manifest:
            manifest["meta_versions"] = {}

        v0_id = "meta-v000"
        v0_dir = self.meta_root / v0_id
        if not v0_dir.exists():
            v0_dir.mkdir(parents=True, exist_ok=True)
            for file_name in ["meta_agent.py", "meta_system.txt", "meta_task.txt"]:
                src = self.meta_base_dir / file_name
                if src.exists():
                    shutil.copy2(src, v0_dir / file_name)

            version_info = {
                "version_id": v0_id,
                "parent_version_id": None,
                "epoch": 0,
                "created_at": manifest.get("created_at"),
                "rationale": "Initial seed from Season 2 base",
                "mutation_summary": "Initial baseline",
            }
            (v0_dir / "version.json").write_text(json.dumps(version_info, ensure_ascii=False, indent=2), encoding="utf-8")
            manifest["meta_versions"][v0_id] = version_info

        if "current_meta_version" not in manifest:
            manifest["current_meta_version"] = v0_id
            self.archive._save()

    def get_meta_agent(self, version_id: str | None = None) -> Any:
        """动态加载指定版本的 MetaAgent 类并实例化。"""
        if version_id is None:
            version_id = self.current_meta_version_id()

        version_dir = self.meta_root / version_id
        if not version_dir.exists():
            raise FileNotFoundError(f"Meta-Agent 版本目录不存在: {version_dir}")

        module_path = version_dir / "meta_agent.py"
        module_name = f"season3.code._meta_{version_id.replace('-', '_')}"
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"无法为 {module_path} 创建 module spec")

        module = importlib.util.module_from_spec(spec)
        module.__package__ = "season3.code"
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(module_name, None)
            raise
        meta_agent_cls = getattr(module, "MetaAgent")

        # 实例化该版本的 MetaAgent
        return meta_agent_cls(
            config=self.config,
            archive=self.archive,
            model_client=self.model_client,
            prompt_directory=version_dir,
        )

    def evolve_meta_agent(self, *, epoch: int, task_nodes_summary: list[dict[str, Any]]) -> str:
        """根据当前 Epoch 内产生的 Task-Agent 表现，自主变异出新一代 Meta-Agent。"""
        current_v_id = self.current_meta_version_id()
        current_dir = self.meta_root / current_v_id

        # 读取当前版本的代码和提示词
        current_code = (current_dir / "meta_agent.py").read_text(encoding="utf-8")
        current_system = (current_dir / "meta_system.txt").read_text(encoding="utf-8")
        current_task = (current_dir / "meta_task.txt").read_text(encoding="utf-8")

        next_seq = len(self.archive._manifest.get("meta_versions", {}))
        new_v_id = f"meta-v{next_seq:03d}"
        new_dir = self.meta_root / new_v_id
        new_dir.mkdir(parents=True, exist_ok=True)

        report("开始 Meta-Agent 自主进化", current_version=current_v_id, new_version=new_v_id, epoch=epoch)

        # 构造发给元进化大模型的提示词
        prompt = (
            f"你是负责进化 Meta-Agent（即设计与改进 Task-Agent 的元架构）的自改进系统。\n"
            f"当前处于第 {epoch} 个 Epoch，上一代使用的 Meta-Agent 版本是 {current_v_id}。\n\n"
            f"【过去一个 Epoch 产出的 Task-Agent 表现事实清单】\n"
            f"{json.dumps(task_nodes_summary, ensure_ascii=False, indent=2)}\n\n"
            f"【当前 Meta-Agent 的代码与提示词】\n"
            f"--- meta_system.txt ---\n{current_system}\n\n"
            f"--- meta_task.txt ---\n{current_task}\n\n"
            f"--- meta_agent.py (完整代码) ---\n{current_code}\n\n"
            f"【你的任务】\n"
            f"根据事实清单分析 Task-Agent 进化的阻碍（例如：代码是否严重膨胀、Prompt 字符是否超限导致 fallback、是否陷入千篇一律的记忆结构、是否忽略回放对局事实等）。\n"
            f"自主决定如何改进 Meta-Agent 的工作逻辑。你可以重构 meta_system.txt、meta_task.txt 或 meta_agent.py 中的逻辑。\n"
            f"请直接输出 JSON，包含以下字段：\n"
            f"- rationale: 本次 Meta 改动的核心原因分析\n"
            f"- mutation_summary: 修改概要\n"
            f"- new_meta_system: 新的 meta_system.txt 内容（非空）\n"
            f"- new_meta_task: 新的 meta_task.txt 内容（非空）\n"
            f"- new_meta_agent_py: 新的 meta_agent.py 内容（如果无需修改 Python 代码可填 null 或原代码）\n"
        )

        try:
            response = self.model_client.complete_json(
                system="你是专注于系统级自进化的 Meta-Architect。只输出 JSON 对象，不要输出 Markdown 围栏。",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.config.meta.max_output_tokens,
            )
        except Exception as error:
            from .code_agent import is_quota_exhausted_failure, QuotaExhaustedError
            if is_quota_exhausted_failure(error):
                raise QuotaExhaustedError(
                    f"Meta-Agent 进化模型网关配额用尽（终止进化）：{type(error).__name__}: {error}"
                ) from error
            report(f"Meta-Agent 进化模型调用失败，沿用当前版本: {error}")
            shutil.rmtree(new_dir, ignore_errors=True)
            return current_v_id

        rationale = response.get("rationale", "No rationale provided")
        mutation_summary = response.get("mutation_summary", "No summary provided")
        new_system = response.get("new_meta_system") or current_system
        new_task = response.get("new_meta_task") or current_task
        new_py = response.get("new_meta_agent_py") or current_code

        # 语法校验：若返回了 Python 代码，必须能通过 ast 语法检查
        if new_py and new_py.strip():
            import ast
            try:
                ast.parse(new_py, filename="meta_agent.py")
            except SyntaxError as err:
                report(f"Meta-Agent 生成的代码存在语法错误，回退使用上一代代码: {err}")
                new_py = current_code

        # 写入新版本文件
        (new_dir / "meta_system.txt").write_text(new_system, encoding="utf-8")
        (new_dir / "meta_task.txt").write_text(new_task, encoding="utf-8")
        (new_dir / "meta_agent.py").write_text(new_py, encoding="utf-8")

        version_info = {
            "version_id": new_v_id,
            "parent_version_id": current_v_id,
            "epoch": epoch,
            "created_at": self.archive._manifest.get("created_at"),
            "rationale": rationale,
            "mutation_summary": mutation_summary,
        }
        (new_dir / "version.json").write_text(json.dumps(version_info, ensure_ascii=False, indent=2), encoding="utf-8")

        # 更新 manifest
        manifest = self.archive._manifest
        if "meta_versions" not in manifest:
            manifest["meta_versions"] = {}
        manifest["meta_versions"][new_v_id] = version_info
        manifest["current_meta_version"] = new_v_id
        self.archive._save()

        report("Meta-Agent 进化完成，已切换为新版本", new_version=new_v_id, summary=mutation_summary)
        return new_v_id

"""初始化或推进 Season 3 双层进化（Task-Agent 与 Meta-Agent）。"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from game.agents.llm import ModelClient, ModelRequestCoordinator
from game.config import resolve_model_env
from season3.code.config import load_season2_config
from season3.code.archive import EvolutionArchive
from season3.code.code_agent import PiCodeAgent
from season3.code.evaluator import EvolutionEvaluator
from season3.code.evolution import EvolutionManager
from season3.code.meta_agent import MetaAgent
from season3.code.meta_version import MetaVersionManager
from season3.code.progress import report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 Season 3 自动化双层自进化")
    parser.add_argument(
        "--config",
        default=str(ROOT / "season3" / "configs" / "season3.json"),
        help="Season 3 外置 JSON 配置文件",
    )
    parser.add_argument("--steps", type=int, default=1, help="推进多少个 Task 进化步骤")
    parser.add_argument(
        "--initialize-only",
        action="store_true",
        help="只创建初始种子 base、meta-v000 和 manifest，不调用模型",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    config = load_season2_config(args.config)
    report("加载 Season 3 配置", config=args.config, steps=args.steps)
    archive = EvolutionArchive(config)

    # 1. initialize-only 处理
    if args.initialize_only:
        archive.initialize()
        meta_manager = MetaVersionManager(config=config, archive=archive, model_client=None)
        meta_manager.initialize()
        report("Season 3 archive 初始化完成", archive=str(archive.root))
        print(json.dumps({
            "initialized": True,
            "archive": str(archive.root),
            "current_meta_version": archive._manifest.get("current_meta_version"),
        }, ensure_ascii=False))
        return

    # 2. 模型配置解析与客户端初始化
    # Task 局内模型与 Meta 诊断模型均由 .env / 进程环境变量配置提供
    task_client = ModelClient.from_env(
        profile="task",
    )
    meta_client = ModelClient.from_env(
        profile="meta",
    )

    coordinator = ModelRequestCoordinator(
        max_in_flight=config.evaluation.model_max_in_flight
    )

    # 3. 初始化 Meta 版本管理器与进化管理器
    meta_version_mgr = MetaVersionManager(
        config=config,
        archive=archive,
        model_client=meta_client,
    )
    archive.initialize()
    meta_version_mgr.initialize()

    initial_meta = meta_version_mgr.get_meta_agent()

    manager = EvolutionManager(
        config=config,
        archive=archive,
        meta_agent=initial_meta,
        code_agent=PiCodeAgent(config),
        evaluator=EvolutionEvaluator(
            config=config,
            archive=archive,
            model_client=task_client,
            request_coordinator=coordinator,
        ),
        meta_version_manager=meta_version_mgr,
    )

    report("开始执行 Season 3 双层自进化调度", steps=args.steps)
    results = await manager.run(args.steps)
    report("Season 3 进化步骤执行完成", finished_steps=len(results))
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())

"""初始化或推进 Season 2 Task-Agent 进化。"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
GAME_ROOT = ROOT / "werewolf_game"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from werewolf_game.agents.llm import ModelClient, ModelRequestCoordinator
from werewolf_game.season2 import EvolutionArchive, EvolutionManager, load_season2_config
from werewolf_game.season2.code_agent import PiCodeAgent
from werewolf_game.season2.evaluator import EvolutionEvaluator
from werewolf_game.season2.meta_agent import MetaAgent
from werewolf_game.season2.progress import report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 Season 2 异步进化")
    parser.add_argument(
        "--config",
        default=str(GAME_ROOT / "configs" / "season2.json"),
        help="外置 JSON 配置文件",
    )
    parser.add_argument("--steps", type=int, default=1, help="推进多少个选择/评测/生成步骤")
    parser.add_argument(
        "--initialize-only",
        action="store_true",
        help="只创建各角色 base 和 manifest，不调用模型",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    config = load_season2_config(args.config)
    report("加载配置", config=args.config, steps=args.steps)
    archive = EvolutionArchive(config)
    # initialize-only 适合首次建立 archive，完全不需要 API 配置。
    if args.initialize_only:
        archive.initialize()
        report("archive 初始化完成", archive=str(archive.root))
        print(json.dumps({"initialized": True, "archive": str(archive.root)}, ensure_ascii=False))
        return

    # 局内玩家和 Meta-Agent 使用独立 profile；二者的模型/推理设置不混用。
    task_client = ModelClient.from_env(profile="task")
    meta_client = ModelClient.from_env(profile="meta")
    coordinator = ModelRequestCoordinator(
        max_in_flight=config.evaluation.model_max_in_flight
    )
    manager = EvolutionManager(
        config=config,
        archive=archive,
        meta_agent=MetaAgent(config=config, archive=archive, model_client=meta_client),
        code_agent=PiCodeAgent(config),
        evaluator=EvolutionEvaluator(
            config=config,
            archive=archive,
            model_client=task_client,
            request_coordinator=coordinator,
        ),
    )
    results = await manager.run(args.steps)
    report("运行结束", steps=len(results))
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())

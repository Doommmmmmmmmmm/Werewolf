"""Web 端可游戏 demo 入口。

运行：

    python3 werewolf_game/examples/run_web_demo.py

然后在浏览器打开 http://127.0.0.1:8192 。

真人占据一个席位，其余席位由 Season 2 进化 archive 中各角色胜率最高的 retained
节点驱动。可用环境变量：

  WEREWOLF_WEB_HOST（默认 127.0.0.1）、WEREWOLF_WEB_PORT（默认 8192）、
  WEREWOLF_WEB_RECORD_DIRECTORY（默认 werewolf_game/records/web）、
  WEREWOLF_ARCHIVE_DIRECTORY（默认 archives/season2-20260923）。

安全：服务器默认只绑定 127.0.0.1 且没有认证。绑定到对外地址意味着任何能访问该
地址的人都可以占用真人席位并读到该席位的私密身份。
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
GAME_ROOT = ROOT / "werewolf_game"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from werewolf_game.web.server import serve


async def main() -> None:
    host = os.environ.get("WEREWOLF_WEB_HOST", "127.0.0.1")
    port = int(os.environ.get("WEREWOLF_WEB_PORT", "8192"))
    record_directory = os.environ.get(
        "WEREWOLF_WEB_RECORD_DIRECTORY", str(GAME_ROOT / "records" / "web")
    )

    archive_directory = os.environ.get(
        "WEREWOLF_ARCHIVE_DIRECTORY",
        str(ROOT / "archives" / "season2-20260923"),
    )
    print(f"进化节点归档：{archive_directory}")
    print(f"对局记录目录：{record_directory}")
    await serve(
        archive_directory=archive_directory,
        host=host,
        port=port,
        record_directory=record_directory,
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n已停止。")

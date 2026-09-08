"""Season 2 运行进度输出。

进度只写标准输出，不参与游戏记录和模型上下文；每条消息立即 flush，
这样长时间评测时可以确认程序仍在运行。
"""

from __future__ import annotations

from datetime import datetime
import sys
from typing import Any


def report(message: str, **fields: Any) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    suffix = "".join(f" {key}={value}" for key, value in fields.items())
    print(f"[{timestamp}] [season2] {message}{suffix}", flush=True)


def report_error(message: str, **fields: Any) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    suffix = "".join(f" {key}={value}" for key, value in fields.items())
    print(f"[{timestamp}] [season2] ERROR {message}{suffix}", file=sys.stderr, flush=True)

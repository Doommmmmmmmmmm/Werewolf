"""小型异步工具。"""

from __future__ import annotations

import inspect
from typing import Any


async def maybe_await(value: Any) -> Any:
    """同时支持同步回调和 async 回调。"""

    if inspect.isawaitable(value):
        return await value
    return value


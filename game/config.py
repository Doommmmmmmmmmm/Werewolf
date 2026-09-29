"""模型接入配置解析。

原实现只读 `Path.cwd()/.env`，换个目录启动就读不到配置，URL 和 key 实际上不可配。
这里把"去哪里找配置"独立出来，按以下优先级解析（先找到的生效）：

1. 进程环境变量；
2. 配置文件：``WEREWOLF_CONFIG`` 指定的路径 → `./werewolf.toml` →
   仓库根的 `werewolf.toml` → `~/.config/werewolf/config.toml`；
3. `.env` 文件：``WEREWOLF_ENV_FILE`` 指定的路径 → 从当前目录逐级向上查找，
   再退到仓库根。

密钥只进内存，不写入对局记录。
"""

from __future__ import annotations

from collections.abc import Mapping
import os
from pathlib import Path
import tomllib

# 仓库根：本文件位于 <root>/game/config.py。
REPO_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH_ENV = "WEREWOLF_CONFIG"
ENV_FILE_ENV = "WEREWOLF_ENV_FILE"

# 配置文件里的键 -> 进程环境变量名。TOML 的 [model] 段与这些名字对应。
_MODEL_KEY_TO_ENV = {
    "base_url": "WEREWOLF_MODEL_BASE_URL",
    "api_key": "WEREWOLF_MODEL_API_KEY",
    "name": "WEREWOLF_MODEL_NAME",
    "api_mode": "WEREWOLF_MODEL_API_MODE",
    "api_path": "WEREWOLF_MODEL_API_PATH",
    "task_model": "WEREWOLF_TASK_MODEL",
    "meta_model": "WEREWOLF_META_MODEL",
    "max_output_tokens": "MAX_OUTPUT_TOKENS",
    "timeout_ms": "MODEL_TIMEOUT_MS",
    "max_retries": "MODEL_MAX_RETRIES",
}

# 新名 -> 兼容的旧名。新名缺失时回退到旧名，保持已有 .env 可用。
_ENV_ALIASES = {
    "TASK_MODEL": ("WEREWOLF_TASK_MODEL", "API_MODEL_NAME", "OPENAI_MODEL", "MODEL_ID"),
    "TASK_BASE_URL": ("API_MODEL_BASE_URL", "OPENAI_BASE_URL", "MODEL_BASE_URL", "WEREWOLF_MODEL_BASE_URL"),
    "TASK_API_KEY": ("API_MODEL_API_KEY", "OPENAI_API_KEY", "MODEL_API_KEY", "WEREWOLF_MODEL_API_KEY"),
    "TASK_API_MODE": ("API_MODEL_API_MODE", "MODEL_API_MODE", "WEREWOLF_MODEL_API_MODE"),
    "TASK_API_PATH": ("API_MODEL_API_PATH", "MODEL_API_PATH", "WEREWOLF_MODEL_API_PATH"),
    "TASK_REASONING_EFFORT": ("WEREWOLF_TASK_REASONING_EFFORT",),
    "META_MODEL": ("WEREWOLF_META_MODEL", "API_MODEL_NAME", "OPENAI_MODEL", "MODEL_ID"),
    "META_BASE_URL": ("WEREWOLF_META_BASE_URL", "API_MODEL_BASE_URL", "OPENAI_BASE_URL", "MODEL_BASE_URL", "WEREWOLF_MODEL_BASE_URL"),
    "META_API_KEY": ("WEREWOLF_META_API_KEY", "API_MODEL_API_KEY", "OPENAI_API_KEY", "MODEL_API_KEY", "WEREWOLF_MODEL_API_KEY"),
    "META_API_MODE": ("WEREWOLF_META_API_MODE", "API_MODEL_API_MODE", "MODEL_API_MODE", "WEREWOLF_MODEL_API_MODE"),
    "META_API_PATH": ("WEREWOLF_META_API_PATH", "API_MODEL_API_PATH", "MODEL_API_PATH", "WEREWOLF_MODEL_API_PATH"),
    "META_REASONING_EFFORT": ("WEREWOLF_META_REASONING_EFFORT",),
    "PI_API_KEY": ("PI_OPENAI_API_KEY", "OPENAI_API_KEY"),
    "PI_BASE_URL": ("PI_OPENAI_BASE_URL", "OPENAI_BASE_URL"),
    "WEREWOLF_MODEL_BASE_URL": ("API_MODEL_BASE_URL", "OPENAI_BASE_URL", "MODEL_BASE_URL"),
    "WEREWOLF_MODEL_API_KEY": ("API_MODEL_API_KEY", "OPENAI_API_KEY", "MODEL_API_KEY"),
    "WEREWOLF_MODEL_NAME": ("API_MODEL_NAME", "OPENAI_MODEL", "MODEL_ID"),
    "WEREWOLF_MODEL_API_MODE": ("API_MODEL_API_MODE", "MODEL_API_MODE"),
    "WEREWOLF_MODEL_API_PATH": ("API_MODEL_API_PATH", "MODEL_API_PATH"),
}


def _parse_dotenv(path: Path) -> dict[str, str]:
    """解析简单的 KEY=VALUE 文件；不支持多行值和变量展开。"""

    values: dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return values
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key] = value
    return values


def find_env_file(start: Path | None = None) -> Path | None:
    """定位 `.env`：显式指定 → 从 start 逐级向上 → 仓库根。"""

    explicit = os.environ.get(ENV_FILE_ENV)
    if explicit:
        path = Path(explicit)
        return path if path.is_file() else None
    current = (start or Path.cwd()).resolve()
    for directory in (current, *current.parents):
        candidate = directory / ".env"
        if candidate.is_file():
            return candidate
    fallback = REPO_ROOT / ".env"
    return fallback if fallback.is_file() else None


def find_config_file(start: Path | None = None) -> Path | None:
    """定位 TOML 配置：显式指定 → 当前目录 → 仓库根 → 用户配置目录。"""

    explicit = os.environ.get(CONFIG_PATH_ENV)
    if explicit:
        path = Path(explicit)
        return path if path.is_file() else None
    candidates = [
        (start or Path.cwd()) / "werewolf.toml",
        REPO_ROOT / "werewolf.toml",
        Path.home() / ".config" / "werewolf" / "config.toml",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _config_file_values(path: Path) -> dict[str, str]:
    """把 TOML 的 [model] 段映射成环境变量风格的键值。"""

    try:
        with path.open("rb") as handle:
            data = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ValueError(f"配置文件解析失败：{path}（{error}）") from error
    model = data.get("model")
    if not isinstance(model, Mapping):
        return {}
    values: dict[str, str] = {}
    for key, env_name in _MODEL_KEY_TO_ENV.items():
        if key in model and model[key] is not None:
            values[env_name] = str(model[key])
    return values


def resolve_model_env(
    *,
    start: Path | None = None,
    overrides: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """汇总模型接入配置，返回可直接查表的键值。

    不修改 `os.environ`，便于测试与多配置并存。优先级：进程环境 > 配置文件 >
    `.env`；``overrides`` 高于一切，供代码显式传参。
    """

    values: dict[str, str] = {}

    env_file = find_env_file(start)
    if env_file is not None:
        values.update(_parse_dotenv(env_file))

    config_file = find_config_file(start)
    if config_file is not None:
        values.update(_config_file_values(config_file))

    # 进程环境优先于文件。
    for key, value in os.environ.items():
        if value is not None and str(value).strip():
            values[key] = value

    if overrides:
        values.update({str(k): str(v) for k, v in overrides.items() if v is not None})

    # 新名缺失时用旧名补齐，保证旧 .env 仍然可用。
    for new_name, old_names in _ENV_ALIASES.items():
        if values.get(new_name):
            continue
        for old in old_names:
            if values.get(old):
                values[new_name] = values[old]
                break
    # 反向补齐：只配了新名时，让沿用旧名的读取路径也能取到。
    for new_name, old_names in _ENV_ALIASES.items():
        value = values.get(new_name)
        if not value:
            continue
        for old in old_names:
            values.setdefault(old, value)
    return values


def describe_sources(start: Path | None = None) -> dict[str, str | None]:
    """报告实际使用的配置来源，便于排查"为什么没读到 key"。"""

    env_file = find_env_file(start)
    config_file = find_config_file(start)
    return {
        "env_file": str(env_file) if env_file else None,
        "config_file": str(config_file) if config_file else None,
    }

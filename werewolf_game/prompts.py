"""从独立文本文件加载游戏玩家 prompt 和角色规则。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .core.constants import ALL_ROLES


PROMPT_DIRECTORY = Path(__file__).with_name("prompts")
ROLE_PROMPT_DIRECTORY = PROMPT_DIRECTORY / "roles"


@dataclass(frozen=True)
class RoleProfile:
    """一个角色的固定规则和最小任务说明。"""

    role: str
    base: str
    task: str


def read_prompt(filename: str, *, prompt_directory: Path | None = None) -> str:
    """读取一个 UTF-8 prompt 文件，不进行模板替换。"""

    path = (prompt_directory or PROMPT_DIRECTORY) / filename
    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError as error:
        raise FileNotFoundError(f"找不到 prompt 模板：{path}") from error


def render_prompt(filename: str, **values: object) -> str:
    """读取 UTF-8 模板并填入命名变量。"""

    return read_prompt(filename).format(**values).strip()


class RoleProfileStore:
    """只读加载角色的固定规则；策略学习不属于当前游戏运行时。"""

    def __init__(self, prompt_directory: str | Path | None = None) -> None:
        self.prompt_directory = Path(prompt_directory or PROMPT_DIRECTORY).resolve()
        self.role_directory = self.prompt_directory / "roles"

    def profile(self, role: str) -> RoleProfile:
        self._require_supported_role(role)
        base_path = self.role_directory / role / "base.md"
        task_path = self.role_directory / role / "task.md"
        try:
            base = base_path.read_text(encoding="utf-8").strip()
            task = task_path.read_text(encoding="utf-8").strip()
        except FileNotFoundError as error:
            raise FileNotFoundError(
                f"找不到 {role} 的角色规则或最小任务：{base_path} / {task_path}"
            ) from error
        return RoleProfile(role=role, base=base, task=task)

    @staticmethod
    def _require_supported_role(role: str) -> None:
        if role not in ALL_ROLES:
            raise ValueError(f"不支持的角色档案：{role}")


def load_role_profile(
    role: str, *, prompt_directory: str | Path | None = None
) -> RoleProfile:
    """读取角色固定规则。"""

    return RoleProfileStore(prompt_directory).profile(role)

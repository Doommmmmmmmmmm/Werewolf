"""从独立文本文件加载 Agent prompt 和角色知识档案。"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from .constants import ALL_ROLES


PROMPT_DIRECTORY = Path(__file__).with_name("prompts")
ROLE_PROMPT_DIRECTORY = PROMPT_DIRECTORY / "roles"


@dataclass(frozen=True)
class RoleProfile:
    """单个角色在一局游戏中可见的知识档案。

    ``base`` 是不能由复盘器改写的固定规则；``strategy`` 是允许跨批次迭代的
    经验策略。二者分开读取，避免经验文本覆盖规则裁决。
    """

    role: str
    base: str
    strategy: str


def read_prompt(filename: str, *, prompt_directory: Path | None = None) -> str:
    """读取一个 UTF-8 prompt 文件，不进行模板替换。"""

    path = (prompt_directory or PROMPT_DIRECTORY) / filename
    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError as error:
        raise FileNotFoundError(f"找不到 prompt 模板：{path}") from error


def render_prompt(filename: str, **values: object) -> str:
    """读取 UTF-8 模板并填入命名变量。

    prompt 文件是 Agent 行为说明的唯一位置，业务代码不内嵌提示词。
    """

    template = read_prompt(filename)
    return template.format(**values).strip()


class RoleStrategyStore:
    """管理角色固定规则与可更新策略，且只允许写入 strategy.md。"""

    def __init__(self, prompt_directory: str | Path | None = None) -> None:
        self.prompt_directory = Path(prompt_directory or PROMPT_DIRECTORY).resolve()
        self.role_directory = self.prompt_directory / "roles"

    def profile(self, role: str) -> RoleProfile:
        """读取某个角色自己的 base 和 strategy 两部分。"""

        self._require_supported_role(role)
        return RoleProfile(
            role=role,
            base=self._read_role_part(role, "base.md"),
            strategy=self._read_role_part(role, "strategy.md"),
        )

    def strategy_path(self, role: str) -> Path:
        self._require_supported_role(role)
        return self.role_directory / role / "strategy.md"

    def replace_strategy(self, role: str, strategy: str) -> Path:
        """原子地替换一个角色的经验策略，绝不写入 base.md。"""

        self._require_supported_role(role)
        normalized = str(strategy or "").strip()
        if len(normalized) < 40:
            raise ValueError("strategy.md 不能为空，且至少需要 40 个字符")
        if len(normalized) > 12000:
            raise ValueError("strategy.md 超过 12000 个字符上限")
        path = self.strategy_path(role)
        if not path.exists():
            raise FileNotFoundError(f"找不到角色策略文件：{path}")
        temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        temporary.write_text(normalized + "\n", encoding="utf-8")
        temporary.replace(path)
        return path

    def _read_role_part(self, role: str, filename: str) -> str:
        path = self.role_directory / role / filename
        try:
            return path.read_text(encoding="utf-8").strip()
        except FileNotFoundError as error:
            raise FileNotFoundError(f"找不到 {role} 的角色档案：{path}") from error

    @staticmethod
    def _require_supported_role(role: str) -> None:
        if role not in ALL_ROLES:
            raise ValueError(f"不支持的角色档案：{role}")


def load_role_profile(
    role: str, *, prompt_directory: str | Path | None = None
) -> RoleProfile:
    """读取角色自己的固定规则与当前经验策略。"""

    return RoleStrategyStore(prompt_directory).profile(role)

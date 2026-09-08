"""候选 Task-Agent 的文件边界与最小运行契约检查。"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from typing import Iterable


FORBIDDEN_IMPORT_ROOTS = frozenset(
    {
        "ctypes",
        "ftplib",
        "http",
        "multiprocessing",
        "os",
        "pathlib",
        "requests",
        "shutil",
        "socket",
        "subprocess",
        "sys",
        "tempfile",
        "urllib",
    }
)
FORBIDDEN_CALLS = frozenset({"__import__", "breakpoint", "compile", "eval", "exec", "input", "open"})


def validate_candidate_files(directory: Path, allowed_files: Iterable[str]) -> None:
    # 先检查实际文件集合，阻止 Pi 通过新增脚本、配置或软链接扩大修改面。
    allowed = {Path(item).as_posix() for item in allowed_files}
    actual = {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }
    unexpected = sorted(actual - allowed)
    missing = sorted(allowed - actual)
    if unexpected:
        raise ValueError("候选工作区出现未批准文件：" + "、".join(unexpected))
    if missing:
        raise ValueError("候选工作区缺少文件：" + "、".join(missing))


def validate_candidate_source(directory: Path) -> None:
    # AST 检查是第一道静态闸门；它不是完整沙箱，但能提前拒绝常见越界能力。
    source_path = directory / "task_agent.py"
    task_path = directory / "task.md"
    source = source_path.read_text(encoding="utf-8")
    task = task_path.read_text(encoding="utf-8").strip()
    if not task:
        raise ValueError("task.md 不能为空")
    tree = ast.parse(source, filename=str(source_path))
    class_names = {
        node.name for node in tree.body if isinstance(node, ast.ClassDef)
    }
    if "TaskAgent" not in class_names:
        raise ValueError("task_agent.py 必须定义 TaskAgent 类")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots = {alias.name.split(".", 1)[0] for alias in node.names}
            denied = sorted(roots & FORBIDDEN_IMPORT_ROOTS)
            if denied:
                raise ValueError("候选禁止导入：" + "、".join(denied))
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            root = node.module.split(".", 1)[0]
            if root in FORBIDDEN_IMPORT_ROOTS:
                raise ValueError(f"候选禁止导入：{root}")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in FORBIDDEN_CALLS:
                raise ValueError(f"候选禁止调用：{node.func.id}")
    # 使用内建 compile 做语法/字节码检查，不在候选目录产生 __pycache__。
    compile(source, str(source_path), "exec")


def validate_task_agent_class(candidate: type[object]) -> None:
    # 动态加载后再次检查接口，避免候选仅能编译却无法被 Participant 调用。
    signature = inspect.signature(candidate)
    required_keywords = {
        "player_id",
        "profile",
        "model_client",
        "request_coordinator",
        "max_tokens",
        "max_decision_retries",
        "max_tool_calls_per_decision",
        "max_tool_result_tokens",
        "max_prompt_chars",
    }
    parameters = signature.parameters
    if not any(item.kind == inspect.Parameter.VAR_KEYWORD for item in parameters.values()):
        missing = sorted(required_keywords - set(parameters))
        if missing:
            raise ValueError("TaskAgent 构造器缺少参数：" + "、".join(missing))
    if not callable(getattr(candidate, "decide", None)):
        raise ValueError("TaskAgent 必须实现 decide")

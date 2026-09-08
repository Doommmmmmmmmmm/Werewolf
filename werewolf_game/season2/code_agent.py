"""外部 Pi coding agent 适配与候选修改审计。"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
from typing import Any

from .config import Season2Config
from .validation import validate_candidate_files, validate_candidate_source


@dataclass(frozen=True)
class CodeAgentResult:
    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    attempts: int


class PiCodeAgent:
    def __init__(self, config: Season2Config) -> None:
        self.config = config

    async def apply(
        self,
        *,
        workspace: Path,
        role: str,
        diagnosis: dict[str, Any],
        operation_directory: Path,
    ) -> CodeAgentResult:
        # Pi 只接收诊断结果和候选目录，不直接接触主仓库或其他角色节点。
        prompt = self._prompt(role=role, diagnosis=diagnosis)
        operation_directory.mkdir(parents=True, exist_ok=True)
        (operation_directory / "code-agent-prompt.txt").write_text(prompt, encoding="utf-8")
        last_result: CodeAgentResult | None = None
        last_error: Exception | None = None
        for attempt in range(1, self.config.pi.max_attempts + 1):
            # 启动失败或返回非零时按配置重试；每次输出均单独归档。
            try:
                result = await asyncio.to_thread(self._run_once, workspace, prompt, attempt)
            except Exception as error:
                last_error = error
                (operation_directory / f"pi-attempt{attempt}.json").write_text(
                    json.dumps(
                        {"attempt": attempt, "error": f"{type(error).__name__}: {error}"},
                        ensure_ascii=False,
                        indent=2,
                    ) + "\n",
                    encoding="utf-8",
                )
                continue
            last_result = result
            (operation_directory / f"pi-attempt{attempt}.json").write_text(
                json.dumps(
                    {
                        "command": list(result.command),
                        "returncode": result.returncode,
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                    },
                    ensure_ascii=False,
                    indent=2,
                ) + "\n",
                encoding="utf-8",
            )
            if result.returncode == 0:
                validate_candidate_files(workspace, self.config.pi.allowed_files)
                validate_candidate_source(workspace)
                return result
        if last_result is None:
            assert last_error is not None
            raise RuntimeError(
                f"Pi 在 {self.config.pi.max_attempts} 次尝试中均未正常启动："
                f"{type(last_error).__name__}: {last_error}"
            ) from last_error
        assert last_result is not None
        raise RuntimeError(
            f"Pi 修改失败（{last_result.returncode}）：{last_result.stderr[-1000:]}"
        )

    def _run_once(self, workspace: Path, prompt: str, attempt: int) -> CodeAgentResult:
        # 延迟导入，避免未启用 Pi 的普通游戏/测试进程加载进程管理模块。
        import os
        import subprocess

        command = self._command(workspace, prompt)
        environment = {
            name: os.environ[name]
            for name in self.config.pi.environment_allowlist
            if name in os.environ
        }
        # 非 bwrap 模式下直接把本机 Pi 配置目录交给 Pi；目录本身仍由外部配置决定。
        if self.config.pi.agent_dir:
            environment["PI_CODING_AGENT_DIR"] = str(self.config.pi.agent_dir)
        if self.config.pi.node_bin:
            if not self.config.pi.node_bin.is_file():
                raise RuntimeError(f"Pi node_bin 不存在：{self.config.pi.node_bin}")
            environment["PI_NODE_BIN"] = str(self.config.pi.node_bin)
        completed = subprocess.run(
            command,
            cwd=workspace,
            text=True,
            capture_output=True,
            timeout=self.config.pi.timeout_seconds,
            check=False,
            env=environment,
        )
        return CodeAgentResult(
            command=tuple(command),
            returncode=completed.returncode,
            stdout=completed.stdout[-20000:],
            stderr=completed.stderr[-20000:],
            attempts=attempt,
        )

    def _command(self, workspace: Path, prompt: str) -> list[str]:
        pi = self.config.pi
        pi_root = self.config.paths.pi_root
        base_command = list(pi.command)
        executable = Path(base_command[0])
        if not executable.is_absolute():
            executable = pi_root / executable
        args = [str(executable), *base_command[1:], "--print", "--no-session", "--no-approve"]
        if pi.model:
            args.extend(("--model", pi.model))
        args.append(prompt)
        if not pi.use_bwrap:
            # 关闭 bwrap 只适合受控开发机；真实实验建议保持默认隔离。
            return args
        if pi.agent_dir is None:
            raise RuntimeError(
                "Pi 使用 bwrap 时必须配置 pi.agent_dir，以提供本机 provider 配置"
            )
        if not pi.agent_dir.is_dir():
            raise RuntimeError(f"Pi agent_dir 不存在或不是目录：{pi.agent_dir}")
        if shutil.which("bwrap") is None:
            raise RuntimeError("配置要求 bwrap，但当前环境没有 bwrap")
        if pi.node_bin is None:
            raise RuntimeError("启用 bwrap 时必须配置 pi.node_bin，以固定使用 Node 22")
        if not pi.node_bin.is_file():
            raise RuntimeError(f"Pi node_bin 不存在：{pi.node_bin}")

        try:
            relative_executable = executable.resolve().relative_to(pi_root.resolve())
        except ValueError as error:
            raise ValueError("启用 bwrap 时，Pi 可执行文件必须位于 paths.pi_root 内") from error
        sandbox_args = [f"/pi/{relative_executable.as_posix()}", *base_command[1:], "--print", "--no-session", "--no-approve"]
        if pi.model:
            sandbox_args.extend(("--model", pi.model))
        sandbox_args.append(prompt)
        node_root = pi.node_bin.resolve().parent.parent
        try:
            node_relative = pi.node_bin.resolve().relative_to(node_root)
        except ValueError as error:
            raise ValueError("pi.node_bin 必须位于其运行时根目录下") from error
        sandbox_node = Path("/node-runtime") / node_relative
        # Pi 本身没有权限系统，因此用 bwrap 显式建立只读 Pi 源码 + 可写候选目录。
        command = [
            "bwrap",
            "--die-with-parent",
            "--unshare-pid",
            "--unshare-ipc",
            "--unshare-uts",
            "--ro-bind", "/usr", "/usr",
            "--ro-bind", "/bin", "/bin",
            "--ro-bind", "/lib", "/lib",
            "--ro-bind", "/lib64", "/lib64",
            "--ro-bind", "/etc", "/etc",
            "--ro-bind", str(pi_root), "/pi",
            "--ro-bind", str(node_root), "/node-runtime",
            "--ro-bind", str(pi.agent_dir), "/pi-agent",
            "--bind", str(workspace), "/workspace",
            "--tmpfs", "/tmp",
            "--dir", "/tmp/home",
            "--proc", "/proc",
            "--dev", "/dev",
            "--setenv", "HOME", "/tmp/home",
            "--setenv", "PI_NODE_BIN", str(sandbox_node),
            "--setenv", "PATH", f"{sandbox_node.parent}:/usr/bin:/bin",
            "--setenv", "PI_CODING_AGENT_DIR", "/pi-agent",
            "--chdir", "/workspace",
        ]
        if pi.readonly_paths:
            command.extend(("--dir", "/readonly"))
        for index, path in enumerate(pi.readonly_paths):
            target = f"/readonly/{index}"
            command.extend(("--ro-bind", str(path), target))
        return [*command, "--", *sandbox_args]

    def _prompt(self, *, role: str, diagnosis: dict[str, Any]) -> str:
        # 提示再次声明边界，防止 code agent 把诊断建议误解为可以改引擎。
        allowed = "、".join(self.config.pi.allowed_files)
        return (
            f"你正在修改狼人杀角色 {role} 的 Task-Agent 候选。\n"
            f"只允许修改当前目录中的这些文件：{allowed}。不要创建其他文件。\n"
            "游戏规则、信息可见性、行动 JSON 契约和工具预算不可绕过；代码不得访问网络、"
            "文件系统、环境变量、进程或隐藏状态。稳定游戏内核 core/ 不属于候选工作区，绝对不能修改。"
            "保留 TaskAgent 类及现有构造接口。\n"
            f"Task-Agent 外部运行契约：每次决策最多调用 {self.config.evaluation.max_tool_calls_per_decision} 次受限工具，"
            f"每次工具返回最多 {self.config.evaluation.max_tool_result_tokens} 个长度单位，"
            f"每次模型请求 system 与 messages 总长度最多 {self.config.evaluation.max_prompt_chars} 个 Unicode 字符。"
            f"Task-Agent 单次模型输出最多 900 tokens，且工具预算为 {self.config.evaluation.max_tool_calls_per_decision} 次、"
            "Prompt 和策略可以自由修改，但不得通过自动拼接完整历史来绕过这些预算；需要历史时让模型主动调用受限工具。\n"
            "根据以下诊断和修改说明直接完成代码修改，并自行检查语法：\n\n"
            + json.dumps(diagnosis, ensure_ascii=False, indent=2)
        )

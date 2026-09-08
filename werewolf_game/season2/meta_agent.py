"""最低限度的 Meta-Agent 工具循环。"""

from __future__ import annotations

import asyncio
import inspect
import json
from pathlib import Path
from typing import Any

from ..core.errors import ModelClientError
from .archive import EvolutionArchive
from .config import Season2Config
from .resources import MetaResourceCatalog


PROMPT_DIRECTORY = Path(__file__).with_name("prompts")


class MetaAgent:
    def __init__(self, *, config: Season2Config, archive: EvolutionArchive, model_client: Any) -> None:
        self.config = config
        self.archive = archive
        self.model_client = model_client

    async def diagnose(
        self,
        *,
        node_id: str,
        output_path: Path,
        required_replay_item: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        # 一个诊断会话内允许模型多次调用只读工具；会话结束后只保存结构化产物。
        node = self.archive.node(node_id)
        catalog = MetaResourceCatalog(
            config=self.config,
            archive=self.archive,
            node_id=node_id,
            required_replay_item=required_replay_item,
        )
        system = (PROMPT_DIRECTORY / "meta_system.txt").read_text(encoding="utf-8").strip()
        template = (PROMPT_DIRECTORY / "meta_task.txt").read_text(encoding="utf-8")
        task = template.format(
            role=node.role,
            node_id=node.node_id,
            required_resources=catalog.required_resource_text(),
            task_agent_contract=self._task_agent_contract(),
        ).strip()
        last_error = ""
        for attempt in range(1, self.config.meta.max_attempts + 1):
            # schema 校验失败时重新生成，但不把上一次完整输出作为新的资料注入。
            content = task
            if last_error:
                content += "\n\n上一次输出未通过诊断 schema 校验：" + last_error
            try:
                result = await self._complete(
                    system=system,
                    messages=[{"role": "user", "content": content}],
                    tools=catalog.tools(),
                    tool_executor=catalog.execute,
                    max_tool_calls=self.config.meta.max_tool_calls,
                    max_tool_result_tokens=self.config.meta.max_tool_result_tokens,
                )
            except ModelClientError as error:
                last_error = str(error)
                if attempt >= self.config.meta.max_attempts:
                    raise
                continue
            error = self._validation_error(result)
            if error:
                last_error = error
                continue
            artifact = {
                "node_id": node_id,
                "role": node.role,
                "attempt": attempt,
                "required_resources": list(self.config.meta.required_resources),
                "required_replay_game_index": (
                    required_replay_item.get("game_index")
                    if required_replay_item
                    else None
                ),
                "tool_trace": catalog.tool_trace,
                "diagnosis": result,
                "model": self._model_manifest(),
                "token_usage": getattr(result, "token_usage", None),
            }
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            return result
        raise ValueError(f"Meta-Agent 未产生合法诊断：{last_error}")

    def _task_agent_contract(self) -> str:
        """把运行时和进化状态机的真实契约完整告知 Meta-Agent。

        这些信息只是诊断上下文，不是让 Meta-Agent 修改边界；真正的预算和
        游戏规则仍由外部 Runtime / Game Engine 强制执行。
        """

        evaluation = self.config.evaluation
        pi = self.config.pi
        enabled_tools = ", ".join(self.config.meta.enabled_tools) or "无"
        return (
            "Task-Agent 局内运行预算（外部 Runtime 强制）：\n"
            f"- 每次决策最多调用 {evaluation.max_tool_calls_per_decision} 次受限工具；"
            f"单次工具返回最多 {evaluation.max_tool_result_tokens} 个长度单位；"
            f"每次模型请求 system 与 messages 总长度最多 {evaluation.max_prompt_chars} 个 Unicode 字符。\n"
            "- Task-Agent 单次模型输出最多 900 tokens；"
            f"非法行动或调用失败后最多额外重试 {evaluation.max_decision_retries} 次，"
            "重试不会增加工具预算。\n"
            f"- 单次玩家决策超时 {evaluation.decision_timeout_seconds:g} 秒；"
            f"全局模型请求最多 {evaluation.model_max_in_flight} 个。\n"
            "- 进化流程、节点预算和筛选规则不是必读上下文；如需这些信息，"
            "请调用只读工具 read_evolution_contract。\n\n"
            "Meta-Agent 与 Pi 预算：\n"
            f"- Meta-Agent 每次诊断最多调用 {self.config.meta.max_tool_calls} 次只读资料工具，"
            f"单次工具返回最多 {self.config.meta.max_tool_result_tokens} 个长度单位，"
            f"最终诊断最多 {self.config.meta.max_output_tokens} tokens，"
            f"最多完整尝试 {self.config.meta.max_attempts} 次；分页和重复读取同样计入工具预算。\n"
            f"- 当前 Meta 必读资料：{', '.join(self.config.meta.required_resources)}；"
            f"启用工具白名单：{enabled_tools}。\n"
            f"- Pi 只能修改候选工作区中的 {', '.join(pi.allowed_files)}，"
            f"单次任务最长 {pi.timeout_seconds:g} 秒，失败最多尝试 {pi.max_attempts} 次；"
            "Pi 的修改对象是可进化的 Task-Agent prompt/代码，而不是固定游戏内核。\n"
            "- Game Engine、游戏规则、身份能力、胜负裁决、信息可见性、行动 JSON schema、"
            "工具实现、模型网关、审计记录和以上预算均不可修改；这些是外部边界，不要在诊断中提出绕过方案。"
        )

    def _model_manifest(self) -> dict[str, Any]:
        model = getattr(self.model_client, "config", None)
        if model is None:
            return {"available": False}
        return {
            "available": True,
            "model": getattr(model, "model", None),
            "protocol": getattr(model, "protocol", None),
            "reasoning_effort": getattr(model, "reasoning_effort", None),
            "enable_thinking": getattr(model, "enable_thinking", None),
            "max_output_tokens": getattr(model, "max_output_tokens", None),
        }

    async def _complete(self, **kwargs: Any) -> dict[str, Any]:
        # 兼容同步/异步模型客户端；具体模型和输出额度来自 meta profile/config。
        complete = self.model_client.complete_json
        if inspect.iscoroutinefunction(complete):
            return await complete(max_tokens=self.config.meta.max_output_tokens, **kwargs)
        return await asyncio.to_thread(
            complete, max_tokens=self.config.meta.max_output_tokens, **kwargs
        )

    @staticmethod
    def _validation_error(value: object) -> str | None:
        # 诊断 schema 是交给 Pi 的接口，先在 Python 层拒绝不完整的建议。
        if not isinstance(value, dict):
            return "输出不是 JSON 对象"
        for field in ("summary", "code_agent_brief"):
            if not isinstance(value.get(field), str) or not value[field].strip():
                return f"{field} 必须是非空字符串"
        for field in ("evidence", "root_causes", "modification_plan", "risks"):
            if not isinstance(value.get(field), list):
                return f"{field} 必须是数组"
        for item in value["modification_plan"]:
            if not isinstance(item, dict) or not all(
                isinstance(item.get(key), str) and item[key].strip()
                for key in ("file", "change", "rationale")
            ):
                return "modification_plan 每项必须包含 file/change/rationale"
        return None

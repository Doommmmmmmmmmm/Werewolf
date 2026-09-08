"""共享的模型请求调度与健康度观测。

一个 12 人局的同步投票阶段会同时产生 12 个模型请求；若再并发运行多局，
仅限制 ``game_concurrency`` 无法避免请求洪峰。本模块为所有玩家、复盘器和可选
播报器提供同一个全局闸门，并记录可用于训练调度的延迟与失败指标。
"""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
import inspect
import math
from time import monotonic
from typing import Any


@dataclass(frozen=True)
class ModelRequestOutcome:
    """一次底层模型调用完成后的非敏感诊断数据。"""

    request_number: int
    success: bool
    caller_cancelled: bool
    queue_wait_ms: int
    latency_ms: int
    api_attempts: int
    generic_retries: int
    usage_limit_retries: int
    error_type: str | None = None
    error_message: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "request_number": self.request_number,
            "success": self.success,
            "caller_cancelled": self.caller_cancelled,
            "queue_wait_ms": self.queue_wait_ms,
            "latency_ms": self.latency_ms,
            "api_attempts": self.api_attempts,
            "generic_retries": self.generic_retries,
            "usage_limit_retries": self.usage_limit_retries,
            "error_type": self.error_type,
            "error_message": self.error_message,
        }


class ModelRequestCoordinator:
    """限制全局在途请求，并对模型 API 健康度生成滑动窗口统计。

    对同步客户端会在工作线程中执行。上层行动超时取消等待时，底层线程不会被
    强行取消；本类会保持请求槽位，直至该线程真正结束，避免超时后的新请求继续
    放大 API 拥塞。
    """

    def __init__(self, *, max_in_flight: int = 8, history_size: int = 200) -> None:
        self._max_in_flight = self._positive_int(max_in_flight, "max_in_flight")
        self._history_size = self._positive_int(history_size, "history_size")
        self._condition = asyncio.Condition()
        self._in_flight = 0
        self._queued = 0
        self._issued = 0
        self._completed = 0
        self._total_success = 0
        self._total_failed = 0
        self._total_timeout = 0
        self._total_api_attempts = 0
        self._total_generic_retries = 0
        self._total_usage_limit_retries = 0
        self._outcomes: deque[ModelRequestOutcome] = deque(maxlen=self._history_size)

    @property
    def max_in_flight(self) -> int:
        return self._max_in_flight

    async def set_max_in_flight(self, value: int) -> int:
        """调整后续请求的并发上限；不会中断当前已在飞的请求。"""

        normalized = self._positive_int(value, "max_in_flight")
        async with self._condition:
            self._max_in_flight = normalized
            self._condition.notify_all()
        return normalized

    async def complete_json(self, model_client: Any, **kwargs: Any) -> dict[str, Any]:
        """经全局闸门调用任意兼容 ``complete_json`` 的模型客户端。"""

        if not hasattr(model_client, "complete_json"):
            raise ValueError("模型客户端必须提供 complete_json")
        queued_at = monotonic()
        await self._acquire()
        started_at = monotonic()
        self._issued += 1
        request_number = self._issued
        call_state = {"caller_cancelled": False}
        task = asyncio.create_task(self._invoke(model_client, **kwargs))
        task.add_done_callback(
            lambda completed: self._schedule_completion(
                completed,
                request_number=request_number,
                queued_at=queued_at,
                started_at=started_at,
                call_state=call_state,
            )
        )
        try:
            # shield 保证 GameRunner 的行动超时不会取消底层 to_thread 请求。
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            call_state["caller_cancelled"] = True
            raise

    async def wait_for_idle(self) -> None:
        """等待所有已获取槽位的请求结束；测试和受控关闭时可用。"""

        async with self._condition:
            while self._in_flight:
                await self._condition.wait()

    def health_snapshot(self, *, window_size: int = 50) -> dict[str, Any]:
        """返回最近窗口和累计的健康统计，可安全写入训练记录。"""

        window = max(1, int(window_size))
        outcomes = list(self._outcomes)[-window:]
        success_count = sum(item.success for item in outcomes)
        failed_count = len(outcomes) - success_count
        caller_cancelled_count = sum(item.caller_cancelled for item in outcomes)
        timeout_count = sum(
            item.error_type in {"TimeoutError", "CancelledError"}
            or (item.error_message or "").lower().find("timeout") >= 0
            for item in outcomes
        )
        api_attempts = sum(item.api_attempts for item in outcomes)
        generic_retries = sum(item.generic_retries for item in outcomes)
        usage_limit_retries = sum(item.usage_limit_retries for item in outcomes)
        latencies = sorted(item.latency_ms for item in outcomes)
        queue_waits = sorted(item.queue_wait_ms for item in outcomes)
        outcome_count = len(outcomes)
        return {
            "max_in_flight": self._max_in_flight,
            "in_flight": self._in_flight,
            "queued": self._queued,
            "history_size": outcome_count,
            "window_size": window,
            "issued_requests": self._issued,
            "completed_requests": self._completed,
            "total_success_count": self._total_success,
            "total_failed_count": self._total_failed,
            "total_timeout_count": self._total_timeout,
            "total_api_attempts": self._total_api_attempts,
            "total_generic_retries": self._total_generic_retries,
            "total_usage_limit_retries": self._total_usage_limit_retries,
            "success_count": success_count,
            "failed_count": failed_count,
            "caller_cancelled_count": caller_cancelled_count,
            "timeout_count": timeout_count,
            "failure_rate": failed_count / outcome_count if outcome_count else 0.0,
            "timeout_rate": timeout_count / outcome_count if outcome_count else 0.0,
            "api_attempts": api_attempts,
            "generic_retries": generic_retries,
            "usage_limit_retries": usage_limit_retries,
            "p50_latency_ms": self._percentile(latencies, 0.50),
            "p95_latency_ms": self._percentile(latencies, 0.95),
            "p95_queue_wait_ms": self._percentile(queue_waits, 0.95),
            "recent_outcomes": [item.as_dict() for item in outcomes],
        }

    async def _acquire(self) -> None:
        async with self._condition:
            self._queued += 1
            try:
                while self._in_flight >= self._max_in_flight:
                    await self._condition.wait()
                self._in_flight += 1
            finally:
                self._queued -= 1

    async def _release(self) -> None:
        async with self._condition:
            self._in_flight = max(0, self._in_flight - 1)
            self._condition.notify_all()

    async def _invoke(self, model_client: Any, **kwargs: Any) -> dict[str, Any]:
        complete_json = model_client.complete_json
        if inspect.iscoroutinefunction(complete_json):
            return await complete_json(**kwargs)
        value = await asyncio.to_thread(complete_json, **kwargs)
        if inspect.isawaitable(value):
            return await value
        if not isinstance(value, dict):
            raise ValueError("模型客户端没有返回 JSON 对象")
        return value

    def _schedule_completion(
        self,
        task: asyncio.Task[dict[str, Any]],
        *,
        request_number: int,
        queued_at: float,
        started_at: float,
        call_state: dict[str, bool],
    ) -> None:
        asyncio.create_task(
            self._finish_call(
                task,
                request_number=request_number,
                queued_at=queued_at,
                started_at=started_at,
                caller_cancelled=call_state["caller_cancelled"],
            )
        )

    async def _finish_call(
        self,
        task: asyncio.Task[dict[str, Any]],
        *,
        request_number: int,
        queued_at: float,
        started_at: float,
        caller_cancelled: bool,
    ) -> None:
        finished_at = monotonic()
        success = False
        error: BaseException | None = None
        result: object | None = None
        try:
            result = task.result()
            success = True
        except BaseException as caught:
            # 调用 task.result() 同时消费异常，避免上层超时后的未处理异常警告。
            error = caught
        attempts_source = result if success else error
        outcome = ModelRequestOutcome(
            request_number=request_number,
            success=success,
            caller_cancelled=caller_cancelled,
            queue_wait_ms=max(0, round((started_at - queued_at) * 1000)),
            latency_ms=max(0, round((finished_at - started_at) * 1000)),
            api_attempts=int(getattr(attempts_source, "api_attempts", 1) or 1),
            generic_retries=int(getattr(attempts_source, "generic_retries", 0) or 0),
            usage_limit_retries=int(
                getattr(attempts_source, "usage_limit_retries", 0) or 0
            ),
            error_type=type(error).__name__ if error is not None else None,
            error_message=(str(error).strip() or type(error).__name__)[:500]
            if error is not None
            else None,
        )
        self._outcomes.append(outcome)
        self._completed += 1
        self._total_success += int(outcome.success)
        self._total_failed += int(not outcome.success)
        self._total_timeout += int(
            outcome.error_type in {"TimeoutError", "CancelledError"}
            or (outcome.error_message or "").lower().find("timeout") >= 0
        )
        self._total_api_attempts += outcome.api_attempts
        self._total_generic_retries += outcome.generic_retries
        self._total_usage_limit_retries += outcome.usage_limit_retries
        await self._release()

    @staticmethod
    def _positive_int(value: object, name: str) -> int:
        try:
            normalized = int(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} 必须是正整数") from error
        if normalized < 1:
            raise ValueError(f"{name} 必须是正整数")
        return normalized

    @staticmethod
    def _percentile(values: list[int], percentile: float) -> int | None:
        if not values:
            return None
        index = max(0, min(len(values) - 1, math.ceil(len(values) * percentile) - 1))
        return values[index]

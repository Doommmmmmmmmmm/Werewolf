"""面向真人的公开记录员接口。"""

from __future__ import annotations

import asyncio
from copy import deepcopy
import inspect
import json
from typing import Any

from ..agents.llm.coordinator import ModelRequestCoordinator
from ..prompts import render_prompt
from ..core.utils import maybe_await


class PublicRecorder:
    """只接受公开事件，可直接被网页 UI 或控制台展示层消费。"""

    def __init__(self, on_update: Any = None) -> None:
        self.on_update = on_update
        self.events: list[dict[str, Any]] = []
        self.known_sequences: set[int] = set()
        self.public_state: dict[str, Any] | None = None

    async def observe(
        self, *, events: list[dict[str, Any]], public_state: dict[str, Any]
    ) -> dict[str, Any]:
        new_events = []
        for event in events:
            if event.get("visibility") != "public" or event.get("seq") in self.known_sequences:
                continue
            copied = deepcopy(event)
            self.events.append(copied)
            self.known_sequences.add(event["seq"])
            new_events.append(copied)
        self.public_state = deepcopy(public_state)
        update = {
            "events": new_events,
            "public_state": deepcopy(self.public_state),
            "event_count": len(self.events),
        }
        if self.on_update is not None:
            await maybe_await(self.on_update(update))
        return update

    def snapshot(self) -> dict[str, Any]:
        return {"events": deepcopy(self.events), "public_state": deepcopy(self.public_state)}


class LlmPublicNarrator(PublicRecorder):
    """可选模型播报员；它没有审计记录或游戏引擎的引用。"""

    def __init__(
        self,
        *,
        model_client: Any,
        request_coordinator: ModelRequestCoordinator | None = None,
        max_tokens: int = 400,
        on_update: Any = None,
    ) -> None:
        super().__init__()
        if not hasattr(model_client, "complete_json"):
            raise ValueError("LlmPublicNarrator 需要具有 complete_json 的模型客户端")
        self.model_client = model_client
        self.request_coordinator = request_coordinator
        self.max_tokens = max_tokens
        self.on_narration_update = on_update
        self.summaries: list[dict[str, Any]] = []

    async def observe(
        self, *, events: list[dict[str, Any]], public_state: dict[str, Any]
    ) -> dict[str, Any]:
        update = await super().observe(events=events, public_state=public_state)
        if not update["events"]:
            return update
        result = await self._complete_json(
            system=render_prompt("public_narrator_system.txt"),
            messages=[
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "public_events": update["events"],
                            "public_state": update["public_state"],
                        },
                        ensure_ascii=False,
                    ),
                }
            ],
        )
        summary = str(result.get("summary", "")).strip()
        if summary:
            entry = {"latest_event_seq": update["events"][-1]["seq"], "summary": summary}
            self.summaries.append(entry)
            update["summary"] = summary
        if self.on_narration_update is not None:
            await maybe_await(self.on_narration_update(update))
        return update

    async def _complete_json(self, **kwargs: Any) -> dict[str, Any]:
        if self.request_coordinator is not None:
            return await self.request_coordinator.complete_json(
                self.model_client, max_tokens=self.max_tokens, **kwargs
            )
        complete_json = self.model_client.complete_json
        if inspect.iscoroutinefunction(complete_json):
            return await complete_json(max_tokens=self.max_tokens, **kwargs)
        return await asyncio.to_thread(complete_json, max_tokens=self.max_tokens, **kwargs)

    def summaries_snapshot(self) -> list[dict[str, Any]]:
        return deepcopy(self.summaries)

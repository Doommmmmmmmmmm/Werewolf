"""带可见性标签的事件日志。"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable

from .constants import VISIBILITY_PUBLIC
from .models import Event


class EventLog:
    """事件是唯一的事实流；不同参与者只能读取自己的投影。"""

    def __init__(self) -> None:
        self._events: list[Event] = []

    @property
    def latest_sequence(self) -> int:
        return len(self._events)

    def emit(
        self,
        *,
        event_type: str,
        visibility: str,
        channel: str,
        payload: dict[str, Any] | None = None,
        recipients: Iterable[str] = (),
    ) -> Event:
        event = Event(
            seq=self.latest_sequence + 1,
            type=event_type,
            visibility=visibility,
            channel=channel,
            recipients=tuple(str(item) for item in recipients),
            payload=deepcopy(payload or {}),
        )
        self._events.append(event)
        return event

    def audit_since(self, sequence: int = 0) -> list[dict[str, Any]]:
        return [deepcopy(event.as_dict()) for event in self._events if event.seq > sequence]

    def public_since(self, sequence: int = 0) -> list[dict[str, Any]]:
        return [
            deepcopy(event.as_dict())
            for event in self._events
            if event.seq > sequence and event.visibility == VISIBILITY_PUBLIC
        ]

    def visible_to(self, player_id: str, sequence: int = 0) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for event in self._events:
            if event.seq <= sequence:
                continue
            if event.visibility == VISIBILITY_PUBLIC or player_id in event.recipients:
                result.append(deepcopy(event.as_dict()))
        return result


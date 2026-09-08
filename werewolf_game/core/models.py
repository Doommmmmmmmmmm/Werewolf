"""轻量数据结构。

这里不放流程逻辑；每个对象都可以直接转换为 JSON 友好的字典。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Player:
    player_id: str
    name: str
    seat: int
    alive: bool = True

    @classmethod
    def from_mapping(cls, value: dict[str, Any], seat: int) -> "Player":
        player_id = str(value.get("id", ""))
        name = str(value.get("name") or player_id or f"Player {seat}")
        return cls(
            player_id=player_id,
            name=name,
            seat=int(value.get("seat", seat)),
        )

    def public_view(self) -> dict[str, Any]:
        return {
            "id": self.player_id,
            "name": self.name,
            "seat": self.seat,
            "alive": self.alive,
        }


@dataclass
class NightState:
    wolf_target: str | None = None
    guarded_targets: list[str] = field(default_factory=list)
    healed_targets: list[str] = field(default_factory=list)
    poison_targets: list[str] = field(default_factory=list)
    deaths: list[dict[str, str]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "wolf_target": self.wolf_target,
            "guarded_targets": list(self.guarded_targets),
            "healed_targets": list(self.healed_targets),
            "poison_targets": list(self.poison_targets),
            "deaths": [dict(item) for item in self.deaths],
        }


@dataclass(frozen=True)
class Event:
    seq: int
    type: str
    visibility: str
    channel: str
    recipients: tuple[str, ...]
    payload: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "type": self.type,
            "visibility": self.visibility,
            "channel": self.channel,
            "recipients": list(self.recipients),
            "payload": self.payload,
        }

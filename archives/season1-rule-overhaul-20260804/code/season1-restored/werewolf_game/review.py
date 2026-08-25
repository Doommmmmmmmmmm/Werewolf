"""按角色而非按玩家执行的跨对局策略复盘。"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import inspect
import json
from typing import Any, Iterable

from .llm.coordinator import ModelRequestCoordinator
from .prompts import RoleStrategyStore, render_prompt
from .replay import render_audit_markdown


ROLE_LABELS = {
    "wolf": "狼人",
    "villager": "平民",
    "seer": "预言家",
    "witch": "女巫",
    "guard": "守卫",
    "hunter": "猎人",
    "idiot": "白痴",
}
ANALYSIS_FIELDS = (
    "strengths",
    "weaknesses",
    "opponent_vulnerabilities",
    "strategy_vulnerabilities",
)


def _string_list(value: object) -> list[str]:
    """将模型返回的分析规范成短文本列表。"""

    if isinstance(value, str):
        candidates = [value]
    elif isinstance(value, list):
        candidates = value
    else:
        candidates = []
    result = []
    for item in candidates:
        text = " ".join(str(item).strip().split())
        if text:
            result.append(text[:1000])
    return result


@dataclass(frozen=True)
class RoleReviewResult:
    """一次角色复盘的可记录结果。"""

    round_index: int
    role: str
    game_ids: tuple[str, ...]
    analysis: dict[str, list[str]]
    strategy_markdown: str | None
    updated: bool
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "round_index": self.round_index,
            "role": self.role,
            "role_label": ROLE_LABELS.get(self.role, self.role),
            "game_ids": list(self.game_ids),
            "analysis": {key: list(value) for key, value in self.analysis.items()},
            "strategy_markdown": self.strategy_markdown,
            "updated": self.updated,
            "error": self.error,
        }

    def markdown(self) -> str:
        """生成写入 round/review 的人类可读复盘。"""

        label = ROLE_LABELS.get(self.role, self.role)
        lines = [
            f"# round{self.round_index} · {label}角色复盘",
            "",
            f"复盘对局：{'、'.join(self.game_ids)}。",
            "",
        ]
        headings = {
            "strengths": "做得好的地方",
            "weaknesses": "做得不好的地方",
            "opponent_vulnerabilities": "对手策略漏洞",
            "strategy_vulnerabilities": "当前策略漏洞",
        }
        for key in ANALYSIS_FIELDS:
            lines.extend([f"## {headings[key]}", ""])
            values = self.analysis.get(key, [])
            if values:
                lines.extend(f"- {value}" for value in values)
            else:
                lines.append("- 本次未形成有效结论。")
            lines.append("")
        lines.extend(["## strategy.md 更新结果", ""])
        if self.updated and self.strategy_markdown:
            lines.extend(["已更新。", "", self.strategy_markdown.strip(), ""])
        else:
            lines.extend([f"未更新：{self.error or '模型没有给出有效策略。'}", ""])
        return "\n".join(lines)


class RoleStrategyReviewer:
    """仅给某一角色读取其自身档案与十局完整回放的复盘 Agent。

    十局完整审计回放可能超过模型的单次上下文窗口。因此先以小批次将完整
    回放交给同一角色的复盘 Agent，再让它基于这些批次证据更新一次 strategy。
    这一过程始终不会读取其他角色的角色档案。
    """

    def __init__(
        self,
        *,
        model_client: Any,
        strategy_store: RoleStrategyStore | None = None,
        request_coordinator: ModelRequestCoordinator | None = None,
        max_tokens: int = 1800,
        replay_batch_size: int = 2,
    ) -> None:
        if not hasattr(model_client, "complete_json"):
            raise ValueError("RoleStrategyReviewer 需要具有 complete_json 的模型客户端")
        self.model_client = model_client
        self.strategy_store = strategy_store or RoleStrategyStore()
        self.request_coordinator = request_coordinator
        self.max_tokens = max_tokens
        self.replay_batch_size = self._validated_replay_batch_size(replay_batch_size)

    async def review_role(
        self,
        *,
        round_index: int,
        role: str,
        game_records: Iterable[dict[str, Any]],
    ) -> RoleReviewResult:
        """复盘一类角色，并且只写回这一角色的 strategy.md。"""

        records = [dict(record) for record in game_records]
        game_ids = tuple(
            str(record.get("metadata", {}).get("game_id") or f"game{index}")
            for index, record in enumerate(records)
        )
        analysis = {key: [] for key in ANALYSIS_FIELDS}
        try:
            if len(records) != 10:
                raise ValueError("角色复盘必须恰好基于 10 局游戏记录")
            profile = self.strategy_store.profile(role)
            batch_analyses = await self._review_replay_batches(
                role=role,
                profile=profile,
                records=records,
            )
            payload = {
                "round_index": int(round_index),
                "role": role,
                "own_role_markdown": {
                    "base_md": profile.base,
                    "strategy_md": profile.strategy,
                },
                "replay_review_protocol": {
                    "complete_replays_read": len(records),
                    "batch_size": self.replay_batch_size,
                    "note": "每条 batch_analysis 均由同一角色 Agent 阅读对应的完整审计回放后产生。",
                },
                "batch_analyses": batch_analyses,
            }
            raw = await self._complete_json(
                system=render_prompt("role_review_system.txt", role=role),
                messages=[
                    {
                        "role": "user",
                        "content": json.dumps(payload, ensure_ascii=False),
                    }
                ],
            )
            source = raw.get("review", raw) if isinstance(raw, dict) else {}
            if not isinstance(source, dict):
                raise ValueError("复盘模型没有返回 JSON 对象")
            analysis = {key: _string_list(source.get(key)) for key in ANALYSIS_FIELDS}
            strategy = self._normalize_strategy(role, source.get("strategy_markdown"))
            self.strategy_store.replace_strategy(role, strategy)
            return RoleReviewResult(
                round_index=int(round_index),
                role=role,
                game_ids=game_ids,
                analysis=analysis,
                strategy_markdown=strategy,
                updated=True,
            )
        except Exception as error:
            return RoleReviewResult(
                round_index=int(round_index),
                role=role,
                game_ids=game_ids,
                analysis=analysis,
                strategy_markdown=None,
                updated=False,
                error=str(error),
            )

    async def _review_replay_batches(
        self,
        *,
        role: str,
        profile: Any,
        records: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """逐批阅读完整审计回放，避免把十局原始 JSON 塞进一次模型调用。"""

        batch_analyses: list[dict[str, Any]] = []
        for batch_index, start in enumerate(range(0, len(records), self.replay_batch_size)):
            batch_records = records[start : start + self.replay_batch_size]
            complete_replays = [
                {
                    "game_id": str(
                        record.get("metadata", {}).get("game_id")
                        or f"game{start + offset}"
                    ),
                    "audit_replay_markdown": render_audit_markdown(record),
                }
                for offset, record in enumerate(batch_records)
            ]
            payload = {
                "role": role,
                "own_role_markdown": {
                    "base_md": profile.base,
                    "strategy_md": profile.strategy,
                },
                "batch_index": batch_index,
                "complete_game_replays": complete_replays,
            }
            raw = await self._complete_json(
                system=render_prompt("role_review_batch_system.txt", role=role),
                messages=[
                    {
                        "role": "user",
                        "content": json.dumps(payload, ensure_ascii=False),
                    }
                ],
                max_tokens=min(self.max_tokens, 1000),
            )
            batch_analyses.append(
                self._normalize_batch_analysis(
                    raw,
                    expected_game_ids=[item["game_id"] for item in complete_replays],
                )
            )
        return batch_analyses

    async def _complete_json(self, **kwargs: Any) -> dict[str, Any]:
        max_tokens = kwargs.pop("max_tokens", self.max_tokens)
        if self.request_coordinator is not None:
            return await self.request_coordinator.complete_json(
                self.model_client, max_tokens=max_tokens, **kwargs
            )
        complete_json = self.model_client.complete_json
        if inspect.iscoroutinefunction(complete_json):
            return await complete_json(max_tokens=max_tokens, **kwargs)
        return await asyncio.to_thread(
            complete_json, max_tokens=max_tokens, **kwargs
        )

    @staticmethod
    def _validated_replay_batch_size(value: int) -> int:
        try:
            size = int(value)
        except (TypeError, ValueError) as error:
            raise ValueError("replay_batch_size 必须是 1–10 的整数") from error
        if size < 1 or size > 10:
            raise ValueError("replay_batch_size 必须是 1–10 的整数")
        return size

    @staticmethod
    def _normalize_batch_analysis(
        raw: object, *, expected_game_ids: list[str]
    ) -> dict[str, Any]:
        """约束批次结论，避免最终策略器接收任意模型结构。"""

        source = raw.get("batch_review", raw) if isinstance(raw, dict) else {}
        if not isinstance(source, dict):
            raise ValueError("批次复盘模型没有返回 JSON 对象")
        notes_source = source.get("game_notes")
        if not isinstance(notes_source, list):
            raise ValueError("批次复盘缺少 game_notes")
        notes: list[dict[str, Any]] = []
        for note in notes_source:
            if not isinstance(note, dict):
                continue
            game_id = str(note.get("game_id") or "")
            if game_id not in expected_game_ids:
                continue
            notes.append(
                {
                    "game_id": game_id,
                    **{
                        field: _string_list(note.get(field))
                        for field in ANALYSIS_FIELDS
                    },
                }
            )
        noted_ids = {note["game_id"] for note in notes}
        missing = [game_id for game_id in expected_game_ids if game_id not in noted_ids]
        if missing:
            raise ValueError(f"批次复盘缺少对局结论：{'、'.join(missing)}")
        return {
            "game_notes": notes,
            "batch_patterns": _string_list(source.get("batch_patterns")),
        }

    @staticmethod
    def _normalize_strategy(role: str, value: object) -> str:
        strategy = str(value or "").strip()
        if len(strategy) < 40:
            raise ValueError("复盘结果缺少有效的 strategy_markdown")
        if not strategy.startswith("#"):
            label = ROLE_LABELS.get(role, role)
            strategy = f"# {label}当前策略（经验，可能不完全正确）\n\n{strategy}"
        return strategy

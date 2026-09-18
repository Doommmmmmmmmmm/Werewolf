"""游戏调度器。

Runner 不裁决胜负、不理解身份策略；它只把 Engine、昼夜 loop、参与者和记录层接起来。
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

from .constants import PHASE_DAY_DISCUSSION, PHASE_NIGHT_WOLF_DISCUSSION
from .engine import GameEngine
from .errors import RuleViolationError
from .loops import (
    DayLoop,
    HunterReactionLoop,
    LastWordsLoop,
    NightLoop,
    SheriffBadgeLoop,
    SheriffElectionLoop,
)
from ..recording.records import FileGameRecordStore, RoundGameRecordStore
from .utils import maybe_await


class GameRunner:
    """交替运行 NightLoop 和 DayLoop 的轻量调度器。"""

    def __init__(
        self,
        *,
        engine: GameEngine,
        participants: Mapping[str, Any],
        decision_timeout_seconds: float = 60.0,
        record_store: FileGameRecordStore | bool | None = None,
        record_directory: str | None = None,
        public_recorder: Any = None,
        on_public_events: Any = None,
        on_audit_events: Any = None,
        on_participant_error: Any = None,
    ) -> None:
        self.engine = engine
        self.participants = dict(participants)
        self.decision_timeout_seconds = decision_timeout_seconds
        if record_store is False:
            self.record_store = None
        elif record_store is not None:
            self.record_store = record_store
        else:
            directory = record_directory or "records"
            round_index, game_index = RoundGameRecordStore.next_available(directory)
            self.record_store = RoundGameRecordStore(
                directory,
                round_index=round_index,
                game_index=game_index,
            )
        self.public_recorder = public_recorder
        self.on_public_events = on_public_events
        self.on_audit_events = on_audit_events
        self.on_participant_error = on_participant_error

        self.errors: list[dict[str, Any]] = []
        self.decision_count = 0
        self.fallback_count = 0
        # 除了全局汇总，还保留按玩家拆分的计数，供节点评测只评估焦点角色。
        self.decision_count_by_player: dict[str, int] = {}
        self.fallback_count_by_player: dict[str, int] = {}
        self.pending_runner_events: list[dict[str, Any]] = []
        self.last_seen_by_player: dict[str, int] = {}
        self.last_public_sync_seq_by_player: dict[str, int] = {}
        self.last_public_event_seq = 0
        self.last_audit_event_seq = 0
        self.last_recorded_audit_event_seq = 0
        self.record_paths: dict[str, str | None] = {
            "record_path": None,
            "record_markdown_path": None,
            "public_record_path": None,
            "public_record_markdown_path": None,
        }

    async def run(self, *, max_steps: int = 500) -> dict[str, Any]:
        """从创建状态一直运行到胜负已定。"""

        if self.engine.status == "created":
            self.engine.start()
            await self._flush_events()

        steps = 0
        while self.engine.status == "running":
            if steps >= max_steps:
                raise RuntimeError(f"对局超过最大步骤数 {max_steps}，可能无法结束")
            if self.engine.phase == PHASE_NIGHT_WOLF_DISCUSSION:
                night = NightLoop(
                    self.engine,
                    self._obtain_decision,
                    self._submit_action,
                    self._sync_night_public_state,
                    self._flush_events,
                )
                await night.run()
                if self.engine.has_pending_sheriff_election():
                    election = SheriffElectionLoop(
                        self.engine,
                        self._obtain_decision,
                        self._submit_action,
                        self._flush_events,
                    )
                    await election.run()
                    self.engine.reveal_deferred_first_dawn()
                    await self._flush_events()
                await self._run_post_death_reactions()
                if self.engine.status == "running":
                    self.engine.start_day()
                    await self._flush_events()
            elif self.engine.phase == PHASE_DAY_DISCUSSION:
                day = DayLoop(
                    self.engine,
                    self._obtain_decision,
                    self._submit_action,
                    self._flush_events,
                )
                await day.run()
                await self._run_post_death_reactions()
                if self.engine.status == "running":
                    self.engine.start_next_night()
                    await self._flush_events()
            else:
                raise RuntimeError(f"Runner 不知道如何从阶段 {self.engine.phase} 继续")
            steps += 1

        await self._flush_events()
        model_token_usage = self._model_token_usage()
        if model_token_usage is not None:
            self.pending_runner_events.append(
                {"type": "MODEL_TOKEN_USAGE", "usage": model_token_usage}
            )
            # 对局结束后再写入一次，将汇总同时落在 JSON 顶层和审计事件中。
            self._persist_record(model_token_usage=model_token_usage)
        return {
            "steps": steps,
            "errors": list(self.errors),
            "error_counts": {
                event_type: sum(
                    item.get("type") == event_type for item in self.errors
                )
                for event_type in sorted(
                    {str(item.get("type") or "unknown") for item in self.errors}
                )
            },
            "decision_count": self.decision_count,
            "fallback_count": self.fallback_count,
            "decision_count_by_player": dict(self.decision_count_by_player),
            "fallback_count_by_player": dict(self.fallback_count_by_player),
            "fallback_rate": (
                self.fallback_count / self.decision_count if self.decision_count else 0.0
            ),
            "model_token_usage": model_token_usage,
            **self.record_paths,
            "public_state": self.engine.public_state(),
        }

    async def _run_post_death_reactions(self) -> None:
        """按猎人开枪 → 警徽交接 → 遗言依次处理所有死亡后的公开反应。"""

        while self.engine.status == "running" and self.engine.has_pending_post_death_actions():
            if self.engine.has_pending_hunter_reactions():
                reaction = HunterReactionLoop(
                    self.engine,
                    self._obtain_decision,
                    self._submit_action,
                    self._flush_events,
                )
                await reaction.run()
                continue
            if self.engine.has_pending_sheriff_badge_resolution():
                badge = SheriffBadgeLoop(
                    self.engine,
                    self._obtain_decision,
                    self._submit_action,
                    self._flush_events,
                )
                await badge.run()
                continue
            if self.engine.has_pending_last_words():
                last_words = LastWordsLoop(
                    self.engine,
                    self._obtain_decision,
                    self._submit_action,
                    self._flush_events,
                )
                await last_words.run()
                continue
            break

    async def _obtain_decision(self, request: dict[str, Any]) -> dict[str, Any]:
        """把一个合法行动请求交给对应参与者，并处理超时/缺席。"""

        self.decision_count += 1
        player_id = request["player_id"]
        self.decision_count_by_player[player_id] = (
            self.decision_count_by_player.get(player_id, 0) + 1
        )
        participant = self.participants.get(player_id)
        if participant is None or not hasattr(participant, "decide"):
            await self._record_error(player_id, "missing_participant", "未注册参与者")
            return self._fallback_action(request, "missing_participant")

        since_sequence = self.last_seen_by_player.get(player_id, 0)
        packet = self.engine.build_turn_packet(request, since_sequence=since_sequence)
        self.last_seen_by_player[player_id] = packet["latest_event_seq"]
        try:
            value = participant.decide(packet)
            timeout_seconds = getattr(
                participant, "decision_timeout_seconds", self.decision_timeout_seconds
            )
            if timeout_seconds is None:
                raw = await maybe_await(value)
            else:
                raw = await asyncio.wait_for(
                    maybe_await(value), timeout=float(timeout_seconds)
                )
            if not isinstance(raw, dict):
                raise ValueError("参与者没有返回 JSON 对象")
            return {
                **raw,
                "request_id": request["request_id"],
                "player_id": player_id,
            }
        except Exception as error:
            await self._record_error(player_id, "participant_error", str(error), error)
            return self._fallback_action(request, "participant_error")

    async def _submit_action(
        self, request: dict[str, Any], action: dict[str, Any]
    ) -> dict[str, Any]:
        """引擎拒绝模型输出时记录原因并以保守行动回退。"""

        try:
            return self.engine.accept_action(request, action)
        except RuleViolationError as error:
            await self._record_error(
                request["player_id"], "invalid_decision", str(error), error
            )
            return self.engine.accept_action(
                request, self._fallback_action(request, "invalid_decision")
            )

    def _fallback_action(self, request: dict[str, Any], reason: str) -> dict[str, Any]:
        """生成回退动作并把它显式写入审计记录。"""

        action = self.engine.fallback_action(request)
        self.fallback_count += 1
        player_id = str(request.get("player_id", ""))
        self.fallback_count_by_player[player_id] = (
            self.fallback_count_by_player.get(player_id, 0) + 1
        )
        self.pending_runner_events.append(
            {
                "type": "FALLBACK_ACTION",
                "request_id": request.get("request_id"),
                "player_id": request.get("player_id"),
                "phase": request.get("phase"),
                "reason": reason,
                "action": dict(action),
            }
        )
        return action

    async def _sync_night_public_state(self) -> None:
        """每夜开始并发通知所有存活玩家，且同步包永远只有公开数据。"""

        public_state = self.engine.public_state()
        if public_state["status"] != "running" or public_state["phase"] != "night":
            return
        alive_players = self.engine.alive_players()

        async def notify(player: Any) -> None:
            player_id = player.player_id
            packet = self.engine.build_public_sync_packet(
                player_id,
                since_sequence=self.last_public_sync_seq_by_player.get(player_id, 0),
            )
            self.last_public_sync_seq_by_player[player_id] = packet["latest_event_seq"]
            participant = self.participants.get(player_id)
            if participant is None or not hasattr(participant, "observe"):
                return
            try:
                await asyncio.wait_for(
                    maybe_await(participant.observe(packet)),
                    timeout=self.decision_timeout_seconds,
                )
            except Exception as error:
                await self._record_error(
                    player_id, "public_state_sync_error", str(error), error
                )

        await asyncio.gather(*(notify(player) for player in alive_players))
        self.pending_runner_events.append(
            {
                "type": "PUBLIC_STATE_SYNC",
                "round": public_state["round"],
                "recipients": [player.player_id for player in alive_players],
                "latest_event_seq": self.engine.latest_event_sequence,
            }
        )

    async def _record_error(
        self,
        player_id: str,
        event_type: str,
        message: str,
        exception: Exception | None = None,
    ) -> None:
        """记录可审计的参与者异常，保留空消息异常的实际类型。"""

        normalized_message = str(message or "").strip()
        if not normalized_message and exception is not None:
            normalized_message = type(exception).__name__
        if not normalized_message:
            normalized_message = "未知参与者异常"

        error: dict[str, Any] = {
            "player_id": player_id,
            "type": event_type,
            "message": normalized_message,
        }
        if exception is not None:
            error["exception_type"] = type(exception).__name__
            status_code = getattr(exception, "status_code", None)
            if status_code is not None:
                error["status_code"] = status_code
            for field in ("api_attempts", "generic_retries", "usage_limit_retries"):
                value = getattr(exception, field, None)
                if value is not None:
                    error[field] = value
            # Task-Agent 最终放弃一份非法输出时，只将规范化后的 action 以有限长度
            # 写入管理员审计事件，方便复盘格式问题。公开记录投影不会包含 runner_events。
            details = getattr(exception, "details", None)
            if isinstance(details, Mapping):
                attempted = details.get("attempted_action")
                if isinstance(attempted, Mapping):
                    safe_attempt: dict[str, str] = {}
                    for field in ("kind", "target_id", "text"):
                        value = attempted.get(field)
                        if isinstance(value, str):
                            safe_attempt[field] = value[:1000]
                    if safe_attempt:
                        error["attempted_action"] = safe_attempt
                validation_error = details.get("validation_error")
                if isinstance(validation_error, str) and validation_error:
                    error["validation_error"] = validation_error
        self.errors.append(error)
        # type 保持原有 participant_error / invalid_decision，方便旧记录查询；
        # event_class 标识它属于 Runner 的异常事件而不是规则事件。
        self.pending_runner_events.append({"event_class": "RUNNER_ERROR", **error})
        if self.on_participant_error is not None:
            try:
                await maybe_await(self.on_participant_error(error))
            except Exception:
                # 记录回调不应阻止规则流程。
                pass

    async def _flush_events(self) -> None:
        """把新事件投影给公开记录员、回调和持久化层。"""

        needs_public = self.public_recorder is not None or self.on_public_events is not None
        public_events = (
            self.engine.public_events(self.last_public_event_seq) if needs_public else []
        )
        if public_events:
            self.last_public_event_seq = public_events[-1]["seq"]
            if self.public_recorder is not None and hasattr(self.public_recorder, "observe"):
                try:
                    await maybe_await(
                        self.public_recorder.observe(
                            events=public_events,
                            public_state=self.engine.public_state(),
                        )
                    )
                except Exception as error:
                    await self._record_error("public_recorder", "public_recorder_error", str(error))

        self._persist_record()

        if public_events and self.on_public_events is not None:
            await maybe_await(self.on_public_events(public_events))
        if self.on_audit_events is not None:
            audit_events = self.engine.audit_events(self.last_audit_event_seq)
            if audit_events:
                self.last_audit_event_seq = audit_events[-1]["seq"]
                await maybe_await(self.on_audit_events(audit_events))

    def _persist_record(self, *, model_token_usage: dict[str, Any] | None = None) -> None:
        if self.record_store is None:
            return
        if self.record_paths["record_path"] is None:
            metadata = self.engine.record_metadata()
            agent_manifests: dict[str, Any] = {}
            for player_id, participant in self.participants.items():
                manifest_method = getattr(participant, "agent_manifest", None)
                if not callable(manifest_method):
                    continue
                try:
                    manifest = manifest_method()
                except Exception:
                    continue
                if isinstance(manifest, dict):
                    # 只保存代理版本、谱系和卡片 ID；参与者自己负责不返回密钥或 prompt 正文。
                    agent_manifests[str(player_id)] = manifest
            if agent_manifests:
                metadata["agent_manifests"] = agent_manifests
            self.record_store.start(metadata)
            self.record_paths = self.record_store.paths()
        events = self.engine.audit_events(self.last_recorded_audit_event_seq)
        self.record_store.append(
            events=events,
            runner_events=self.pending_runner_events,
            snapshot=self.engine.record_snapshot(),
            model_token_usage=model_token_usage,
        )
        if events:
            self.last_recorded_audit_event_seq = events[-1]["seq"]
        self.pending_runner_events = []

    def _model_token_usage(self) -> dict[str, Any] | None:
        """汇总各 LLM 玩家报告的服务端实际 token usage。

        仅纳入玩家行动调用，不把轮末角色复盘、可选公开播报等跨游戏调用混进来。
        服务端未返回 usage 或发生内部重试时，用 coverage 明确标记不完整性。
        """

        snapshots: list[dict[str, Any]] = []
        for participant in self.participants.values():
            snapshot_method = getattr(participant, "model_token_usage_snapshot", None)
            if not callable(snapshot_method):
                continue
            try:
                snapshot = snapshot_method()
            except Exception:
                continue
            if isinstance(snapshot, dict):
                snapshots.append(snapshot)
        if not snapshots:
            return None

        fields = (
            "successful_response_count",
            "api_attempt_count",
            "reported_usage_response_count",
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "prompt_request_count",
            "prompt_over_limit_count",
            "prompt_measurement_count",
            "prompt_chars_total",
            "prompt_chars_max",
            "prompt_chars_min",
            "prompt_remaining_chars_min",
            "prompt_tool_calls",
            "prompt_turns",
        )
        totals = {field: 0 for field in fields}
        max_fields = {"prompt_chars_max"}
        min_fields = {"prompt_chars_min", "prompt_remaining_chars_min"}
        seen_min: set[str] = set()
        for snapshot in snapshots:
            for field in fields:
                value = snapshot.get(field, 0)
                if isinstance(value, bool):
                    continue
                try:
                    normalized = int(value)
                except (TypeError, ValueError):
                    continue
                if normalized < 0:
                    continue
                if field in max_fields:
                    totals[field] = max(totals[field], normalized)
                elif field in min_fields:
                    # 0 is a valid value for remaining budget, but a missing
                    # field also defaults to 0. Only use present values.
                    if field not in snapshot:
                        continue
                    if field not in seen_min:
                        totals[field] = normalized
                        seen_min.add(field)
                    else:
                        totals[field] = min(totals[field], normalized)
                else:
                    totals[field] += normalized

        reported = totals["reported_usage_response_count"]
        attempts = totals["api_attempt_count"]
        if attempts == 0:
            availability = "not_applicable"
        elif reported == 0:
            availability = "unavailable"
        elif reported >= attempts:
            availability = "complete"
        else:
            availability = "partial"
        return {
            "availability": availability,
            "participant_count": len(snapshots),
            **totals,
            "unreported_api_attempt_count": max(0, attempts - reported),
        }

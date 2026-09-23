"""一局 Web demo 的生命周期管理。

一局 = 一个 `asyncio.Task`，因为引擎没有从快照恢复的路径：进程退出即丢失进行中的
对局。落盘记录仍由 `RoundGameRecordStore` 正常生成，可事后复盘。

公开事件通过 `PublicRecorder(on_update=...)` 增量推给订阅者；该钩子本身就是
`recording/recorder.py` 为网页 UI 预留的。
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import secrets
import time
from typing import Any

from ..agents.llm import ModelClient, ModelRequestCoordinator
from ..agents.participants.base import Participant
from ..core.engine import GameEngine
from ..core.rules import create_rules_for_player_count
from ..core.runner import GameRunner
from ..recording.recorder import PublicRecorder
from ..recording.records import RoundGameRecordStore
from ..recording.replay import rounds_from_record
from .config import DEFAULT_DEMO_CONFIG, DemoConfig
from .node_repo import ArchivedNodeParticipant, ArchivedNodeRepository
from .participant import WebHumanParticipant
from .roster import (
    default_repository,
    RATING_STRATEGIES,
    RosterEntry,
    best_role_nodes,
    load_ratings,
    rating_model_info,
    role_node_map,
)

MODE_RANDOM = "random"
MODE_CHOOSE = "choose"
MODES = (MODE_RANDOM, MODE_CHOOSE)

SUPPORTED_ROLES = ("wolf", "villager", "seer", "witch", "guard", "hunter", "idiot")


@dataclass
class SessionOptions:
    """建局参数；均为非敏感值，可直接回显给前端。"""

    mode: str = MODE_RANDOM
    seat: str = "p1"
    role: str | None = None
    strategy: str = "best_win_rate"
    overrides: dict[str, str] = field(default_factory=dict)

    def normalized(self, player_count: int) -> "SessionOptions":
        mode = str(self.mode or MODE_RANDOM)
        if mode not in MODES:
            raise ValueError(f"未知模式 {mode}；可选 {'、'.join(MODES)}")
        seat = str(self.seat or "p1")
        valid_seats = {f"p{index}" for index in range(1, player_count + 1)}
        if seat not in valid_seats:
            raise ValueError(f"席位 {seat} 不在当前 {player_count} 人局中")
        role = self.role
        if mode == MODE_CHOOSE:
            if not role:
                raise ValueError("自选模式必须指定身份")
            role = str(role).lower()
            if role not in SUPPORTED_ROLES:
                raise ValueError(f"不支持的身份 {role}")
        else:
            role = None
        return SessionOptions(
            mode=mode,
            seat=seat,
            role=role,
            strategy=self.strategy,
            overrides=dict(self.overrides),
        )


class _PausableParticipant(Participant):
    """给 AI 席位加一道暂停闸门。

    只在"还没开始决策"时拦截：闸门关闭时新的 `decide` 挂起等待，已经进入模型
    调用的请求不受影响、正常跑完，因此暂停不会浪费已花掉的 token，也不会触发
    fallback 行动打乱对局。

    其余接口全部透传给被包装的参与者，保证 Runner 仍能读到
    `agent_manifest` / `model_token_usage_snapshot` / `decision_timeout_seconds`。
    """

    def __init__(self, inner: Any, *, gate: asyncio.Event) -> None:
        super().__init__(inner.player_id)
        self._inner = inner
        self._gate = gate

    async def decide(self, turn_packet: dict[str, Any]) -> dict[str, Any]:
        await self._gate.wait()
        return await self._inner.decide(turn_packet)

    async def observe(self, sync_packet: dict[str, Any]) -> None:
        # observe 不设闸门：Runner 对它强制套用超时，挂起会被记为同步错误。
        await self._inner.observe(sync_packet)

    def __getattr__(self, name: str) -> Any:
        # decision_timeout_seconds、agent_manifest、model_token_usage_snapshot 等
        # 由 Runner 通过 getattr 读取，这里统一透传。
        return getattr(self._inner, name)


class GameSession:
    """持有一局对局的运行任务、可见事件缓冲和真人席位。"""

    def __init__(
        self,
        *,
        session_id: str,
        options: SessionOptions,
        config: DemoConfig,
        repository: ArchivedNodeRepository,
        model_client: Any,
        coordinator: ModelRequestCoordinator,
        record_directory: str | Path,
        ratings: Mapping[str, Any] | None = None,
    ) -> None:
        self.session_id = session_id
        self.options = options
        self.config = config
        self.repository = repository
        self.created_at = time.time()

        player_count = config.player_count
        self.player_count = player_count
        self.players = [
            {"id": f"p{index}", "name": f"{index} 号玩家"}
            for index in range(1, player_count + 1)
        ]
        self.human_player_id = options.seat

        self.roster: dict[str, RosterEntry] = best_role_nodes(
            repository,
            config.roles,
            overrides=options.overrides,
            strategy=options.strategy,
            ratings=ratings,
        )

        rules = create_rules_for_player_count(
            player_count, config.optional_roles
        )
        fixed_roles = (
            {self.human_player_id: options.role}
            if options.mode == MODE_CHOOSE and options.role
            else None
        )
        self.engine = GameEngine(
            game_id=f"web-{session_id}",
            players=self.players,
            rules=rules,
            seed=f"web-demo-{session_id}",
            fixed_roles=fixed_roles,
        )

        # 暂停闸门：set 表示放行。暂停只拦住"尚未开始"的 AI 决策，
        # 已在途的模型请求会正常跑完，不浪费已经花掉的 token。
        self._resume_gate = asyncio.Event()
        self._resume_gate.set()
        self.paused = False

        self.human = WebHumanParticipant(
            self.human_player_id, on_turn=self._notify
        )
        role_nodes = role_node_map(self.roster)
        participants: dict[str, Any] = {}
        for player in self.players:
            player_id = player["id"]
            if player_id == self.human_player_id:
                participants[player_id] = self.human
                continue
            participants[player_id] = _PausableParticipant(
                ArchivedNodeParticipant(
                    player_id=player_id,
                    model_client=model_client,
                    role_nodes=role_nodes,
                    repository=repository,
                    request_coordinator=coordinator,
                    max_decision_retries=config.max_decision_retries,
                    max_tool_calls_per_decision=config.max_tool_calls_per_decision,
                    max_tool_result_tokens=config.max_tool_result_tokens,
                    max_prompt_chars=config.max_prompt_chars,
                ),
                gate=self._resume_gate,
            )
        self.participants = participants

        round_index, game_index = RoundGameRecordStore.next_available(
            str(record_directory)
        )
        self.record_store = RoundGameRecordStore(
            str(record_directory), round_index=round_index, game_index=game_index
        )

        # 本席位依法可见的事件缓冲：单调递增，前端按 since 增量拉取。
        # 包含公开事件 + 发给本席位的 team/private 事件（狼人私聊、查验结果等）。
        self.visible_events: list[dict[str, Any]] = []
        # 引擎事件序号游标，用于只拉取增量。
        self._event_cursor_seq = 0
        # 真人死亡时的事件边界；只有此后的事件才按观战视角放宽可见性，
        # 不回溯补发死前无权看到的狼队私聊。
        self._spectator_from_seq: int | None = None
        self.public_state: dict[str, Any] | None = None
        self.report: dict[str, Any] | None = None
        self.error: str | None = None
        self.finished = asyncio.Event()
        self._updated = asyncio.Condition()
        self._task: asyncio.Task[Any] | None = None

    # ------------------------------------------------------------------
    # 运行。
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._task is not None:
            return
        self._task = asyncio.create_task(self._run())

    async def _run(self) -> None:
        recorder = PublicRecorder(on_update=self._on_public_update)
        runner = GameRunner(
            engine=self.engine,
            participants=self.participants,
            record_store=self.record_store,
            public_recorder=recorder,
            # AI 席位使用评测配置的超时；真人席位在实例上声明不超时。
            decision_timeout_seconds=self.config.decision_timeout_seconds,
        )
        try:
            self.report = await runner.run()
        except asyncio.CancelledError:
            self.error = "对局已取消"
            raise
        except Exception as error:  # 记录失败原因，前端可展示
            self.error = f"{type(error).__name__}: {error}"
        finally:
            self.human.cancel("game_over")
            self.finished.set()
            await self._notify()

    async def _on_public_update(self, update: Mapping[str, Any]) -> None:
        """公开记录员触发时，顺带刷新本席位的可见事件。

        注意：不能只用公开事件做实况流。狼人私聊是 `VISIBILITY_TEAM`，
        预言家查验是 `VISIBILITY_PRIVATE`，`PublicRecorder` 只放行
        `visibility == "public"`，会把它们全部丢弃。真正的数据源是
        `engine.events_for(席位)`——引擎按 `visibility == public 或
        player_id in recipients` 过滤，正好是"这个席位依法能看到什么"。
        """

        state = update.get("public_state")
        if isinstance(state, Mapping):
            self.public_state = dict(state)
        self._drain_visible_events()
        await self._notify()

    def _drain_visible_events(self) -> None:
        """拉取本席位自上次游标之后的新可见事件。

        真人死亡后转为观战视角，可以看到狼队私聊等此前无权看到的内容。但
        **只对死亡之后新产生的事件生效**：不能回溯补发生前的私聊，否则等于
        追认了一份当时无权获得的信息。

        实现上按 seq 逐条判定，而不是在两个数据源之间切换——游标记录的是
        "见过的最大 seq"，切换数据源会把存活期跳过的低 seq 事件整批倒灌进来。
        """

        alive = self._human_alive()
        if alive:
            events = self.engine.events_for(
                self.human_player_id, self._event_cursor_seq
            )
        else:
            # 死亡瞬间锁定边界：此刻之后的事件才按观战视角放宽。
            if self._spectator_from_seq is None:
                self._spectator_from_seq = self._event_cursor_seq
            events = []
            scanned = self._event_cursor_seq
            for event in self.engine.audit_events(self._event_cursor_seq):
                # 无论是否放行都推进扫描位置，避免被跳过的事件反复重扫。
                scanned = max(scanned, int(event.get("seq", 0)))
                if event.get("visibility") == "audit":
                    continue
                visible = (
                    event.get("visibility") == "public"
                    or self.human_player_id in (event.get("recipients") or [])
                    # 只放宽死亡边界之后的事件。
                    or int(event.get("seq", 0)) > self._spectator_from_seq
                )
                if visible:
                    events.append(event)
            self._event_cursor_seq = scanned
        if events:
            self._event_cursor_seq = max(
                self._event_cursor_seq, int(events[-1]["seq"])
            )
            for event in events:
                self.visible_events.append(dict(event))

    async def _notify(self) -> None:
        # 任何唤醒点都先补拉一次可见事件：轮到真人行动时（on_turn）往往刚好
        # 积压了队友的狼人私聊，这些 team 事件不伴随公开事件产生。
        self._drain_visible_events()
        async with self._updated:
            self._updated.notify_all()

    async def wait_for_change(self, *, timeout: float = 25.0) -> None:
        """等待任意公开事件或待办行动变化；超时返回用于发送 SSE 心跳。"""

        try:
            async with self._updated:
                await asyncio.wait_for(self._updated.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            return

    async def pause(self) -> None:
        """暂停：不再放行新的 AI 决策；在途的模型请求继续跑完。"""

        self.paused = True
        self._resume_gate.clear()
        await self._notify()

    async def resume(self) -> None:
        self.paused = False
        self._resume_gate.set()
        await self._notify()

    async def close(self) -> None:
        # 关闭前先放行闸门，避免 AI 决策永久挂在 gate.wait() 上无法取消。
        self._resume_gate.set()
        self.human.cancel("session_closed")
        task = self._task
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

    # ------------------------------------------------------------------
    # 对外投影：只包含浏览器依法可见的数据。
    # ------------------------------------------------------------------

    def _human_alive(self) -> bool:
        """真人席位是否存活；对局结束后一律视为可以看全部身份。"""

        if self.finished.is_set():
            return False
        for player in self.engine.players:
            if player.player_id == self.human_player_id:
                return bool(player.alive)
        return False

    def roster_view(self) -> list[dict[str, Any]]:
        return [entry.as_dict() for entry in self.roster.values()]

    def describe(self) -> dict[str, Any]:
        return {
            "game_id": self.engine.game_id,
            "session_id": self.session_id,
            "seat": self.human_player_id,
            "mode": self.options.mode,
            "requested_role": self.options.role,
            "player_count": self.player_count,
            "players": [dict(player) for player in self.players],
            "roster": self.roster_view(),
            "strategy": self.options.strategy,
        }

    def state_view(self, since: int = 0) -> dict[str, Any]:
        """本席位可见状态 + 自 since 之后的可见事件 + 当前待办行动。

        事件已由引擎按席位过滤（公开 + 本席位为 recipient 的 team/private），
        待办行动包同样按席位过滤，可以原样下发；绝不包含 audit_state 或发牌表。
        """

        # 每次读取都先补拉一次：狼人私聊等 team 事件不一定伴随公开事件产生，
        # 只依赖 PublicRecorder 回调会漏掉它们。
        self._drain_visible_events()
        start = max(0, int(since))
        events = self.visible_events[start:]
        packet = self.human.pending_packet()
        return {
            "public_state": self.public_state or self.engine.public_state(),
            "events": events,
            "event_cursor": len(self.visible_events),
            "paused": self.paused,
            "human_alive": self._human_alive(),
            # 真人死亡后解锁上帝视角：只影响这一个浏览器的观看体验，
            # 存活的 AI 仍按引擎可见性行动，不改变规则公平性。
            "revealed_roles": None if self._human_alive() else self.engine.role_assignments(),
            "turn": packet,
            "turn_revision": self.human.turn_revision,
            "finished": self.finished.is_set(),
            "error": self.error,
            "result": self.result_view() if self.finished.is_set() else None,
        }

    def replay_view(self) -> dict[str, Any]:
        """对局结束后的公开复盘，按轮次整理，可直接在浏览器里展示。

        复用 `project_public_record`（剥掉种子、底牌和审计状态）和
        `rounds_from_record`（把扁平事件铺成按轮次结构）。喂进去的是公开投影，
        因此狼人私聊和私密夜间目标天然为空，不需要额外过滤。
        """

        if not self.finished.is_set():
            raise ValueError("对局尚未结束，暂无复盘")
        audit_path = (self.record_store.paths() or {}).get("record_path")
        if not audit_path or not Path(audit_path).is_file():
            raise ValueError("找不到本局记录文件")
        record = json.loads(Path(audit_path).read_text(encoding="utf-8"))
        # 复盘是上帝视角：直接用审计记录，保留狼队私聊、夜间行动和全部身份牌。
        # 对局已结束，不存在信息泄漏问题。
        return {
            "metadata": {
                "game_id": (record.get("metadata") or {}).get("game_id"),
                "players": (record.get("metadata") or {}).get("players", []),
            },
            "rounds": rounds_from_record(record),
            "roles": self.engine.role_assignments(),
            "roster": self.roster_view(),
            "final_public_state": self.engine.public_state(),
        }

    def result_view(self) -> dict[str, Any] | None:
        if not self.finished.is_set():
            return None
        public_state = self.engine.public_state()
        result: dict[str, Any] = {
            "public_state": public_state,
            "winner": public_state.get("winner"),
            "round": public_state.get("round"),
            "error": self.error,
        }
        if self.report is not None:
            result["record_path"] = self.report.get("record_path")
            result["public_record_markdown_path"] = self.report.get(
                "public_record_markdown_path"
            )
        return result


class SessionManager:
    """建局、查局和关闭；共享归档节点仓库、节点代码缓存和模型闸门。"""

    def __init__(
        self,
        *,
        config: DemoConfig | None = None,
        repository: ArchivedNodeRepository | None = None,
        record_directory: str | Path | None = None,
        model_client: Any | None = None,
        max_in_flight: int | None = None,
    ) -> None:
        self.config = config or DEFAULT_DEMO_CONFIG
        config = self.config
        # 只读归档仓库；节点代码按内容哈希缓存，多局共用。
        self.repository = repository or default_repository()
        # 强度表只读一次：文件约 280KB，且每局建局都要用。
        self._ratings = self.repository.ratings or load_ratings()
        self.model_client = model_client or ModelClient.from_env(profile="task")
        self.coordinator = ModelRequestCoordinator(
            max_in_flight=int(
                max_in_flight or config.model_max_in_flight or 8
            )
        )
        self.record_directory = Path(
            record_directory
            or os.environ.get(
                "WEREWOLF_WEB_RECORD_DIRECTORY",
                str(Path(__file__).resolve().parents[1] / "records" / "web"),
            )
        )
        self.sessions: dict[str, GameSession] = {}

    def create(self, options: SessionOptions) -> GameSession:
        normalized = options.normalized(self.config.player_count)
        session_id = secrets.token_hex(4)
        session = GameSession(
            session_id=session_id,
            options=normalized,
            config=self.config,
            repository=self.repository,
            model_client=self.model_client,
            coordinator=self.coordinator,
            record_directory=self.record_directory,
            ratings=self._ratings,
        )
        self.sessions[session_id] = session
        session.start()
        return session

    def get(self, session_id: str) -> GameSession:
        try:
            return self.sessions[str(session_id)]
        except KeyError as error:
            raise KeyError(f"对局 {session_id} 不存在或已结束") from error

    def rule_summary(self) -> dict[str, Any]:
        """开局规则说明。

        复用引擎的 `public_rule_summary()`：它是"所有玩家开局即应知道的配置"，
        不泄露发牌结果，因此可以在建局前就展示。这里用一个临时引擎实例取投影，
        不参与任何对局。
        """

        rules = create_rules_for_player_count(
            self.config.player_count, self.config.optional_roles
        )
        probe = GameEngine(
            game_id="rule-summary-probe",
            players=[
                {"id": f"p{index}", "name": f"P{index}"}
                for index in range(1, self.config.player_count + 1)
            ],
            rules=rules,
            seed="rule-summary-probe",
        )
        summary = probe.public_rule_summary()
        summary["player_count"] = self.config.player_count
        summary["max_day_speech_chars"] = rules.max_day_speech_chars
        summary["max_wolf_speech_chars"] = rules.max_wolf_speech_chars
        return summary

    def strategy_options(self) -> list[dict[str, Any]]:
        """可选的对手强度口径。

        两个强度模型来自 season2_summary 的预计算结果；缺失时不提供该选项。
        """

        labels = [
            ("best_win_rate", "按评测胜率（各角色 20 局胜率最高）"),
            ("logistic_team", "按强度模型：Logistic 团队"),
            ("bradley_terry_legacy", "按强度模型：团队 Bradley-Terry（已退化，仅对照）"),
            ("base", "进化前基线（base 节点）"),
        ]
        available = rating_model_info(self._ratings)
        options: list[dict[str, Any]] = []
        for key, label in labels:
            if key in RATING_STRATEGIES and not available.get(key, {}).get("available"):
                continue
            options.append({"strategy": key, "label": label})
        return options

    def available_roles(self) -> list[dict[str, Any]]:
        """自选模式可选身份及牌堆数量，用于前端渲染选择器。"""

        rules = create_rules_for_player_count(
            self.config.player_count, self.config.optional_roles
        )
        counts: dict[str, int] = {}
        for role in rules.role_deck:
            counts[role] = counts.get(role, 0) + 1
        return [
            {"role": role, "count": count} for role, count in sorted(counts.items())
        ]

    async def close_all(self) -> None:
        for session in list(self.sessions.values()):
            await session.close()
        self.sessions.clear()

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import tempfile
import unittest

from game import GameEngine, GameRunner, create_rules_for_player_count
from game.core.constants import ACTION_SPEAK, ROLE_WOLF
from game.web.participant import WebHumanParticipant
from game.web.roster import (
    RosterEntry,
    best_role_nodes,
    role_node_map,
)
from game.web.server import HttpError, Request, WebDemoServer


def _packet(request_id: str = "g:1:day_discussion:p1:0") -> dict:
    return {
        "game": {"round": 1, "phase": "day_discussion"},
        "public_state": {"players": []},
        "private_information": {"role": "villager", "team": "village"},
        "visible_events": [],
        "request": {
            "request_id": request_id,
            "player_id": "p1",
            "allowed_actions": [
                {"kind": ACTION_SPEAK, "max_chars": 200, "require_chinese": True}
            ],
        },
    }


class WebHumanParticipantTest(unittest.TestCase):
    def test_decide_waits_until_submit(self) -> None:
        async def scenario() -> dict:
            participant = WebHumanParticipant("p1")
            task = asyncio.create_task(participant.decide(_packet()))
            await asyncio.sleep(0)
            # 待办行动此时对 HTTP 层可见。
            self.assertIsNotNone(participant.pending_packet())
            self.assertEqual(
                participant.pending_request_id(), "g:1:day_discussion:p1:0"
            )
            participant.submit(
                "g:1:day_discussion:p1:0", {"kind": ACTION_SPEAK, "text": "我先听票型。"}
            )
            return await task

        action = asyncio.run(scenario())
        self.assertEqual(action, {"kind": ACTION_SPEAK, "text": "我先听票型。"})

    def test_pending_cleared_after_submit(self) -> None:
        async def scenario() -> WebHumanParticipant:
            participant = WebHumanParticipant("p1")
            task = asyncio.create_task(participant.decide(_packet()))
            await asyncio.sleep(0)
            participant.submit(
                "g:1:day_discussion:p1:0", {"kind": ACTION_SPEAK, "text": "好的。"}
            )
            await task
            return participant

        participant = asyncio.run(scenario())
        self.assertIsNone(participant.pending_packet())
        self.assertIsNone(participant.pending_request_id())

    def test_stale_request_id_is_rejected(self) -> None:
        async def scenario() -> None:
            participant = WebHumanParticipant("p1")
            task = asyncio.create_task(participant.decide(_packet()))
            await asyncio.sleep(0)
            with self.assertRaises(ValueError):
                participant.submit("过期的-request-id", {"kind": ACTION_SPEAK, "text": "你好。"})
            # 原待办仍然有效。
            participant.submit(
                "g:1:day_discussion:p1:0", {"kind": ACTION_SPEAK, "text": "你好。"}
            )
            await task

        asyncio.run(scenario())

    def test_duplicate_submit_is_rejected(self) -> None:
        async def scenario() -> None:
            participant = WebHumanParticipant("p1")
            task = asyncio.create_task(participant.decide(_packet()))
            await asyncio.sleep(0)
            participant.submit(
                "g:1:day_discussion:p1:0", {"kind": ACTION_SPEAK, "text": "第一次。"}
            )
            await task
            with self.assertRaises(ValueError):
                participant.submit(
                    "g:1:day_discussion:p1:0", {"kind": ACTION_SPEAK, "text": "第二次。"}
                )

        asyncio.run(scenario())

    def test_invalid_action_is_rejected_before_engine(self) -> None:
        async def scenario() -> None:
            participant = WebHumanParticipant("p1")
            task = asyncio.create_task(participant.decide(_packet()))
            await asyncio.sleep(0)
            # 发言必须包含中文；本地校验先给出反馈。
            with self.assertRaises(ValueError):
                participant.submit(
                    "g:1:day_discussion:p1:0", {"kind": ACTION_SPEAK, "text": "abc"}
                )
            participant.submit(
                "g:1:day_discussion:p1:0", {"kind": ACTION_SPEAK, "text": "中文发言。"}
            )
            await task

        asyncio.run(scenario())

    def test_observe_returns_immediately(self) -> None:
        # Runner 的 _sync_night_public_state 对 observe 强制套用 Runner 级超时，
        # 因此 observe 不能等待真人。
        participant = WebHumanParticipant("p1")
        asyncio.run(
            asyncio.wait_for(participant.observe({"public_state": {}}), timeout=0.05)
        )
        self.assertIsNotNone(participant.latest_public_sync)

    def test_web_seat_does_not_inherit_model_timeout(self) -> None:
        self.assertIsNone(WebHumanParticipant.decision_timeout_seconds)

        async def scenario() -> dict:
            players = [{"id": f"p{i}", "name": f"Player {i}"} for i in range(1, 13)]
            engine = GameEngine(
                game_id="web-no-timeout",
                players=players,
                rules=create_rules_for_player_count(12),
                seed="web-no-timeout",
                fixed_roles={"p1": ROLE_WOLF},
            )
            engine.start()
            participant = WebHumanParticipant("p1")
            runner = GameRunner(
                engine=engine,
                participants={"p1": participant},
                record_store=False,
                # 远小于真人反应时间；真人席位必须不受它影响。
                decision_timeout_seconds=0.01,
            )
            request = engine.discussion_request("p1", "wolf")
            task = asyncio.create_task(runner._obtain_decision(request))
            await asyncio.sleep(0.05)
            self.assertFalse(task.done())
            participant.submit(
                request["request_id"],
                {"kind": ACTION_SPEAK, "text": "今晚先打对面的强神。"},
            )
            action = await task
            self.assertEqual(runner.errors, [])
            return action

        action = asyncio.run(scenario())
        self.assertEqual(action["kind"], ACTION_SPEAK)

    def test_on_turn_fires_before_waiting(self) -> None:
        # 夜间阶段变化只产生 audit 事件，不会经 PublicRecorder 通知前端；
        # decide 必须主动唤醒 SSE，否则行动面板要等到心跳才出现。
        calls: list[int] = []

        async def scenario() -> None:
            async def on_turn() -> None:
                calls.append(1)

            participant = WebHumanParticipant("p1", on_turn=on_turn)
            task = asyncio.create_task(participant.decide(_packet()))
            await asyncio.sleep(0)
            self.assertEqual(calls, [1])
            participant.submit(
                "g:1:day_discussion:p1:0", {"kind": ACTION_SPEAK, "text": "好的。"}
            )
            await task

        asyncio.run(scenario())

    def test_cancel_wakes_pending_decide(self) -> None:
        async def scenario() -> None:
            participant = WebHumanParticipant("p1")
            task = asyncio.create_task(participant.decide(_packet()))
            await asyncio.sleep(0)
            participant.cancel("session_closed")
            with self.assertRaises(RuntimeError):
                await task

        asyncio.run(scenario())


class _FakeNode:
    def __init__(self, node_id: str, role: str, summary_path=None, depth: int = 1) -> None:
        self.node_id = node_id
        self.role = role
        self.evaluation_summary_path = summary_path
        self.depth = depth


class _FakeArchive:
    """只实现 roster 需要的只读接口（对应 ArchivedNodeRepository）。"""

    def __init__(self, nodes: dict[str, list[_FakeNode]]) -> None:
        self._nodes = nodes

    def evaluation_summary(self, node_id: str) -> dict | None:
        """按节点读评测汇总；归档后改为由仓库按布局推导，不再用绝对路径。"""

        for nodes in self._nodes.values():
            for node in nodes:
                if node.node_id != node_id:
                    continue
                path = node.evaluation_summary_path
                if not path:
                    return None
                try:
                    return json.loads(Path(path).read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    return None
        return None

    def retained_nodes(self, role: str) -> list[_FakeNode]:
        return list(self._nodes.get(role, []))

    def nodes_for_role(self, role: str) -> list[_FakeNode]:
        return list(self._nodes.get(role, []))

    def node(self, node_id: str) -> _FakeNode:
        for nodes in self._nodes.values():
            for node in nodes:
                if node.node_id == node_id:
                    return node
        raise KeyError(node_id)


class RosterTest(unittest.TestCase):
    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory()
        self.root = Path(self._dir.name)
        self.addCleanup(self._dir.cleanup)

    def _summary(self, name: str, win_rate: float, game_count: int = 20) -> Path:
        path = self.root / f"{name}.json"
        path.write_text(
            json.dumps({"win_rate": win_rate, "game_count": game_count}),
            encoding="utf-8",
        )
        return path

    def test_picks_highest_win_rate(self) -> None:
        archive = _FakeArchive(
            {
                "seer": [
                    _FakeNode("seer-base", "seer", self._summary("a", 0.30), depth=0),
                    _FakeNode("seer-n1", "seer", self._summary("b", 0.45)),
                    _FakeNode("seer-n2", "seer", self._summary("c", 0.40)),
                ]
            }
        )
        roster = best_role_nodes(archive, ["seer"])
        self.assertEqual(roster["seer"].node_id, "seer-n1")
        self.assertEqual(roster["seer"].win_rate, 0.45)
        self.assertEqual(role_node_map(roster), {"seer": "seer-n1"})

    def test_prefers_more_games_on_tie(self) -> None:
        archive = _FakeArchive(
            {
                "wolf": [
                    _FakeNode("wolf-n1", "wolf", self._summary("a", 0.80, 10)),
                    _FakeNode("wolf-n2", "wolf", self._summary("b", 0.80, 20)),
                ]
            }
        )
        roster = best_role_nodes(archive, ["wolf"])
        self.assertEqual(roster["wolf"].node_id, "wolf-n2")

    def test_falls_back_to_base_without_summaries(self) -> None:
        archive = _FakeArchive(
            {
                "guard": [
                    _FakeNode("guard-base", "guard", None, depth=0),
                    _FakeNode("guard-n1", "guard", None),
                ]
            }
        )
        roster = best_role_nodes(archive, ["guard"])
        self.assertEqual(roster["guard"].node_id, "guard-base")
        self.assertEqual(roster["guard"].source, "base_fallback")
        self.assertIsNone(roster["guard"].win_rate)

    def test_missing_summary_file_is_skipped(self) -> None:
        archive = _FakeArchive(
            {
                "witch": [
                    _FakeNode("witch-base", "witch", None, depth=0),
                    _FakeNode(
                        "witch-n1", "witch", self.root / "does-not-exist.json"
                    ),
                    _FakeNode("witch-n2", "witch", self._summary("w", 0.5)),
                ]
            }
        )
        roster = best_role_nodes(archive, ["witch"])
        self.assertEqual(roster["witch"].node_id, "witch-n2")

    def test_override_wins_over_strategy(self) -> None:
        archive = _FakeArchive(
            {
                "wolf": [
                    _FakeNode("wolf-base", "wolf", self._summary("a", 0.80), depth=0),
                    _FakeNode("wolf-n1", "wolf", self._summary("b", 0.95)),
                ]
            }
        )
        roster = best_role_nodes(archive, ["wolf"], overrides={"wolf": "wolf-base"})
        self.assertEqual(roster["wolf"].node_id, "wolf-base")
        self.assertEqual(roster["wolf"].source, "override")

    def test_override_rejects_role_mismatch(self) -> None:
        archive = _FakeArchive(
            {
                "wolf": [_FakeNode("wolf-base", "wolf", None, depth=0)],
                "seer": [_FakeNode("seer-base", "seer", None, depth=0)],
            }
        )
        with self.assertRaises(ValueError):
            best_role_nodes(archive, ["wolf", "seer"], overrides={"wolf": "seer-base"})

    def test_base_strategy_uses_base_nodes(self) -> None:
        archive = _FakeArchive(
            {
                "wolf": [
                    _FakeNode("wolf-base", "wolf", self._summary("a", 0.80), depth=0),
                    _FakeNode("wolf-n1", "wolf", self._summary("b", 0.95)),
                ]
            }
        )
        roster = best_role_nodes(archive, ["wolf"], strategy="base")
        self.assertEqual(roster["wolf"].node_id, "wolf-base")

    def test_unknown_strategy_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            best_role_nodes(_FakeArchive({}), [], strategy="不存在的策略")

    def test_rating_strategy_picks_highest_strength(self) -> None:
        archive = _FakeArchive(
            {
                "seer": [
                    _FakeNode("seer-base", "seer", self._summary("a", 0.30), depth=0),
                    # 胜率更高但强度更低，用于确认按强度而不是按胜率挑。
                    _FakeNode("seer-hi-wr", "seer", self._summary("b", 0.45)),
                    _FakeNode("seer-hi-str", "seer", self._summary("c", 0.35)),
                ]
            }
        )
        ratings = {
            "models": {
                "logistic_team": {
                    "nodes": [
                        {"node_id": "seer-hi-wr", "role": "seer", "strength": 1.1,
                         "games_played": 50, "global_win_rate": 0.3,
                         "significant_vs_base": False},
                        {"node_id": "seer-hi-str", "role": "seer", "strength": 1.7,
                         "games_played": 26, "global_win_rate": 0.35,
                         "significant_vs_base": False},
                    ]
                }
            }
        }
        roster = best_role_nodes(
            archive, ["seer"], strategy="logistic_team", ratings=ratings
        )
        self.assertEqual(roster["seer"].node_id, "seer-hi-str")
        self.assertEqual(roster["seer"].strength, 1.7)
        self.assertEqual(roster["seer"].source, "logistic_team")

    def test_rating_strategy_ignores_non_retained_nodes(self) -> None:
        # 强度表里可能含 discarded 节点，不能被选中。
        archive = _FakeArchive({"seer": [_FakeNode("seer-base", "seer", None, depth=0)]})
        ratings = {
            "models": {
                "logistic_team": {
                    "nodes": [
                        {"node_id": "seer-dropped", "role": "seer", "strength": 9.9,
                         "games_played": 30},
                    ]
                }
            }
        }
        roster = best_role_nodes(
            archive, ["seer"], strategy="logistic_team", ratings=ratings
        )
        self.assertEqual(roster["seer"].node_id, "seer-base")

    def test_rating_strategy_falls_back_without_ratings(self) -> None:
        # 预计算结果缺失时退回胜率口径，而不是开局失败。
        archive = _FakeArchive(
            {
                "seer": [
                    _FakeNode("seer-base", "seer", self._summary("a", 0.30), depth=0),
                    _FakeNode("seer-n1", "seer", self._summary("b", 0.45)),
                ]
            }
        )
        roster = best_role_nodes(
            archive, ["seer"], strategy="logistic_team", ratings={"models": {}}
        )
        self.assertEqual(roster["seer"].node_id, "seer-n1")


class HttpRequestTest(unittest.TestCase):
    def test_parses_path_and_query(self) -> None:
        request = Request("GET", "/api/games/abc/state?since=7", {}, b"")
        self.assertEqual(request.path, "/api/games/abc/state")
        self.assertEqual(request.query_int("since"), 7)
        self.assertEqual(request.query_int("missing", 3), 3)

    def test_json_body(self) -> None:
        body = json.dumps({"kind": "speak"}).encode("utf-8")
        self.assertEqual(Request("POST", "/x", {}, body).json(), {"kind": "speak"})

    def test_invalid_json_raises_http_error(self) -> None:
        with self.assertRaises(HttpError) as caught:
            Request("POST", "/x", {}, b"{not json").json()
        self.assertEqual(caught.exception.status, 400)

    def test_non_object_json_rejected(self) -> None:
        with self.assertRaises(HttpError):
            Request("POST", "/x", {}, b"[1,2]").json()


class FrontendPayloadFieldTest(unittest.TestCase):
    """逐事件核对前端读取的 payload 字段是否真的存在于该事件。

    之前 `voter_id` 写成 `player_id`、警徽移交用 `target_id`，都导致界面打出
    undefined / null。只检查"字段名在引擎里出现过"是不够的——`player_id`
    到处都有——必须按事件核对。
    """

    def setUp(self) -> None:
        root = Path(__file__).resolve().parents[1] / "game"
        self.engine_source = (root / "core" / "engine.py").read_text(encoding="utf-8")
        self.frontend = (root / "web" / "static" / "index.html").read_text(
            encoding="utf-8"
        )

    def test_vote_events_use_voter_id(self) -> None:
        # 引擎投票事件的投票人字段是 voter_id，不带 player_id。
        self.assertIn('"voter_id": action["player_id"]', self.engine_source)
        self.assertIn("p.voter_id", self.frontend)

    def test_badge_transfer_uses_from_to_fields(self) -> None:
        self.assertIn('"from_player_id"', self.engine_source)
        self.assertIn('"to_player_id"', self.engine_source)
        self.assertIn("p.from_player_id", self.frontend)
        self.assertIn("p.to_player_id", self.frontend)

    def test_seat_label_guards_against_null(self) -> None:
        # target_id 为 null 是合法状态（弃票），不能渲染成 "null 号"。
        self.assertIn("playerId === null", self.frontend)
        self.assertIn("弃票", self.frontend)


class FrontendEventCoverageTest(unittest.TestCase):
    """前端必须覆盖引擎实际发出的公开事件，并使用正确的 payload 字段名。

    之前上警名单不显示、天亮死讯误报平安夜，都是因为前端漏了事件类型或
    猜错了字段名；这里用源码扫描把这类回归钉住。
    """

    def setUp(self) -> None:
        root = Path(__file__).resolve().parents[1] / "game"
        self.engine_source = (root / "core" / "engine.py").read_text(encoding="utf-8")
        self.frontend = (root / "web" / "static" / "index.html").read_text(
            encoding="utf-8"
        )

    def _public_event_types(self) -> set[str]:
        import re

        types: set[str] = set()
        # 匹配 _emit("X", visibility=VISIBILITY_PUBLIC, ...)
        for match in re.finditer(
            r'_emit\(\s*"?([A-Z_]+)"?[^)]*?visibility=VISIBILITY_PUBLIC',
            self.engine_source,
            re.DOTALL,
        ):
            types.add(match.group(1))
        return types

    def test_all_public_events_are_rendered(self) -> None:
        missing = sorted(
            event
            for event in self._public_event_types()
            if f'"{event}"' not in self.frontend
        )
        self.assertEqual(
            missing, [], f"前端 describeEvent 未覆盖这些公开事件：{missing}"
        )

    def test_dawn_uses_engine_field_name(self) -> None:
        # 引擎发的是 dead_player_ids；写成 deaths/died 会静默显示成平安夜。
        self.assertIn("dead_player_ids", self.engine_source)
        self.assertIn("dead_player_ids", self.frontend)

    def test_candidate_ids_is_surfaced(self) -> None:
        # 上警名单只存在于事件里，public_state 不含该字段。
        self.assertIn("candidate_ids", self.frontend)

    def test_audit_only_event_is_not_relied_on(self) -> None:
        # DAY_VOTE_RESOLVED 是 audit 可见，前端收不到，不应依赖它。
        self.assertNotIn("DAY_VOTE_RESOLVED", self.frontend)


class VisibleEventSourceTest(unittest.TestCase):
    """实况流必须用按席位的可见事件，而不是只用公开事件。

    之前实况流只挂 PublicRecorder，它硬性只放行 visibility == "public"，
    于是狼人私聊（VISIBILITY_TEAM）和预言家查验（VISIBILITY_PRIVATE）
    永远进不了界面——玩狼人看不到队友发言就是这个原因。
    """

    def setUp(self) -> None:
        self.web = Path(__file__).resolve().parents[1] / "game" / "web"
        self.session = (self.web / "session.py").read_text(encoding="utf-8")
        self.frontend = (self.web / "static" / "index.html").read_text(encoding="utf-8")

    def test_uses_events_for_seat(self) -> None:
        # engine.events_for(player_id) 才是"该席位依法可见"的接口。
        self.assertIn("events_for(", self.session)
        self.assertIn("self.human_player_id", self.session)

    def test_does_not_rely_only_on_public_recorder(self) -> None:
        # 仍可用 PublicRecorder 做唤醒，但事件来源必须是 events_for。
        self.assertIn("visible_events", self.session)
        self.assertNotIn("self.public_events", self.session)

    def test_frontend_reads_channel_from_event_top_level(self) -> None:
        # channel 在事件顶层，不在 payload 里；写成 payload.channel 会永远判错。
        self.assertIn("event.channel", self.frontend)
        self.assertNotIn("payload.channel", self.frontend)

    def test_wolf_chat_is_visually_distinguished(self) -> None:
        # 私聊必须和公开发言区分，否则会误判队友已暴露。
        self.assertIn("wolf-chat", self.frontend)
        self.assertIn("仅狼队可见", self.frontend)

    def test_private_events_are_translated(self) -> None:
        for event in (
            "ROLE_ASSIGNED",
            "WOLF_TEAM_REVEALED",
            "SEER_RESULT",
            "WOLF_TARGET_SELECTED",
            "GUARD_ACTION_CONFIRMED",
            "WITCH_ACTION_CONFIRMED",
        ):
            self.assertIn(f'case "{event}"', self.frontend, f"{event} 未翻译")


class PauseAndGodViewTest(unittest.TestCase):
    """暂停闸门与死后上帝视角。"""

    def setUp(self) -> None:
        self.web = Path(__file__).resolve().parents[1] / "game" / "web"
        self.session = (self.web / "session.py").read_text(encoding="utf-8")
        self.server = (self.web / "server.py").read_text(encoding="utf-8")
        self.frontend = (self.web / "static" / "index.html").read_text(encoding="utf-8")

    def test_pause_gate_only_blocks_new_decisions(self) -> None:
        # 闸门必须拦在 decide 入口之前，而不是取消在途的模型请求。
        self.assertIn("_PausableParticipant", self.session)
        self.assertIn("await self._gate.wait()", self.session)
        # observe 不能设闸门：Runner 对它强制套用超时。
        self.assertIn("observe 不设闸门", self.session)

    def test_pause_gate_passes_through_runner_attributes(self) -> None:
        # Runner 通过 getattr 读 decision_timeout_seconds / agent_manifest 等。
        self.assertIn("__getattr__", self.session)

    def test_close_releases_gate(self) -> None:
        # 否则暂停中关闭对局，AI 会永久挂在 gate.wait() 上。
        self.assertIn("self._resume_gate.set()", self.session)

    def test_pause_routes_exist(self) -> None:
        for route in ("pause", "resume", "abandon"):
            self.assertIn(f'action == "{route}"', self.server)

    def test_roles_hidden_while_human_alive(self) -> None:
        # 存活期间必须是 None，出局后才给全部身份。
        self.assertIn("None if self._human_alive()", self.session)
        self.assertIn("role_assignments()", self.session)

    def test_spectator_view_does_not_backfill_history(self) -> None:
        """死后观战只放宽新事件，不回溯补发死前无权看到的狼队私聊。

        曾经的实现是在死亡时整体切换数据源，而游标只记录"见过的最大 seq"，
        于是存活期跳过的低 seq 私聊会被整批倒灌进来——等于追认了一份当时
        无权获得的信息。
        """

        self.assertIn("_spectator_from_seq", self.session)
        # 必须按 seq 逐条判定边界，而不是无条件放行所有非 audit 事件。
        self.assertIn("> self._spectator_from_seq", self.session)

    def test_spectator_scan_cursor_advances_past_skipped_events(self) -> None:
        # 被跳过的事件也要推进扫描位置，否则每次都重扫一遍。
        self.assertIn("scanned = max(scanned", self.session)

    def test_frontend_has_pause_and_god_view(self) -> None:
        self.assertIn("暂停", self.frontend)
        self.assertIn("上帝视角", self.frontend)
        self.assertIn("revealed_roles", self.frontend)


class EventDedupAndActorTest(unittest.TestCase):
    """事件去重与私密事件归属人。

    实测过的两个前端 bug：
      1. 同一条发言渲染两次——SSE 与 pollLoop 并发共享 state.cursor，
         合并两条流后同一批事件被投递两遍（实测 236 条变 472 条）；
      2. 观战时别人的守护/用药/查验都写成「你」——没按 recipients 判断主语。
    """

    def setUp(self) -> None:
        root = Path(__file__).resolve().parents[1] / "game"
        self.frontend = (root / "web" / "static" / "index.html").read_text(
            encoding="utf-8"
        )
        self.models = (root / "core" / "models.py").read_text(encoding="utf-8")

    def test_events_are_deduped_by_seq(self) -> None:
        # 引擎保证 seq 唯一，前端必须据此去重，而不是只靠游标算术。
        self.assertIn("state.seenEvents.has(seq)", self.frontend)
        self.assertIn("new Set()", self.frontend)

    def test_poll_loop_is_single_instance(self) -> None:
        # 否则每次 SSE onerror 都会多启一条循环，争用同一个游标。
        self.assertIn("state.polling", self.frontend)

    def test_sse_only_degrades_when_closed(self) -> None:
        # EventSource 自带重连；抖动时不能立刻叠加一条轮询。
        self.assertIn("EventSource.CLOSED", self.frontend)

    def test_cursor_only_moves_forward(self) -> None:
        self.assertIn("Math.max(state.cursor", self.frontend)

    def test_recipients_is_available_on_events(self) -> None:
        # actorOf 依赖事件顶层的 recipients。
        self.assertIn('"recipients": list(self.recipients)', self.models)

    def test_private_event_subject_comes_from_recipients(self) -> None:
        self.assertIn("function actorOf", self.frontend)
        self.assertIn("event.recipients", self.frontend)

    def test_no_hardcoded_second_person_in_private_events(self) -> None:
        # 出局后观战会收到别人的私密事件，主语不能写死「你」。
        for wrong in ("`🛡 你守护了", "`🧪 你${label}"):
            self.assertNotIn(wrong, self.frontend)


class ReplayAndRulesTest(unittest.TestCase):
    """复盘和规则说明的结构性检查（不跑对局）。"""

    def setUp(self) -> None:
        self.web = Path(__file__).resolve().parents[1] / "game" / "web"
        self.frontend = (self.web / "static" / "index.html").read_text(encoding="utf-8")

    def test_replay_is_god_view(self) -> None:
        # 复盘要看完整信息：用审计记录，含狼队私聊、夜间行动和身份牌。
        session = (self.web / "session.py").read_text(encoding="utf-8")
        self.assertIn("rounds_from_record", session)
        self.assertIn("role_assignments", session)
        self.assertIn("wolf_speeches", self.frontend)
        self.assertIn("night_actions", self.frontend)

    def test_witch_action_label_override_is_handled(self) -> None:
        # rounds_from_record 里 **payload 会用 witch_heal 覆盖中文 kind，
        # 前端必须自己再翻译一次，否则复盘里会冒出英文。
        self.assertIn("NIGHT_ACTION_LABELS", self.frontend)
        self.assertIn("witch_heal", self.frontend)

    def test_replay_requires_finished_game(self) -> None:
        session = (self.web / "session.py").read_text(encoding="utf-8")
        self.assertIn("对局尚未结束", session)

    def test_frontend_offers_restart_and_replay(self) -> None:
        self.assertIn("再来一局", self.frontend)
        self.assertIn("查看完整复盘", self.frontend)

    def test_rules_come_from_engine_not_hardcoded(self) -> None:
        # 规则文案必须由引擎的 public_rule_summary() 驱动。
        session = (self.web / "session.py").read_text(encoding="utf-8")
        self.assertIn("public_rule_summary", session)
        self.assertIn("state.meta.rules", self.frontend)

    def test_server_exposes_replay_route(self) -> None:
        server = (self.web / "server.py").read_text(encoding="utf-8")
        self.assertIn('action == "replay"', server)
        self.assertIn("rule_summary", server)


class StaticFileTest(unittest.TestCase):
    def test_frontend_is_shipped(self) -> None:
        index = Path(__file__).resolve().parents[1] / "game" / "web" / "static" / "index.html"
        self.assertTrue(index.is_file())
        text = index.read_text(encoding="utf-8")
        # 前端必须消费引擎给出的合法行动契约，而不是自己硬编码行动种类。
        self.assertIn("allowed_actions", text)
        self.assertIn("request_id", text)

    def test_static_path_traversal_is_blocked(self) -> None:
        server = WebDemoServer.__new__(WebDemoServer)

        async def scenario() -> None:
            with self.assertRaises(HttpError):
                await server._write_static(None, "../../../etc/passwd")

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()

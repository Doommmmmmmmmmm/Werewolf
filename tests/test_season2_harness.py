from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
import unittest

from werewolf_game import (
    GameEngine,
    GameRunner,
    HarnessArchiveStore,
    HarnessRuntime,
    HarnessSpec,
    HarnessStaticEvaluation,
    FrozenRoleStrategyStore,
    RoleStrategyStore,
    TaskAgentMetaAgent,
    create_default_rules,
    evaluate_harness,
)
from werewolf_game.constants import ACTION_LAST_WORDS, ACTION_PASS, ACTION_SPEAK, ROLE_WOLF
from werewolf_game.participants import LlmParticipant
from werewolf_game.prompts import PROMPT_DIRECTORY


class FakeMetaClient:
    def __init__(self) -> None:
        self.requests: list[dict] = []

    async def complete_json(self, **request: object) -> dict:
        self.requests.append(request)  # type: ignore[arg-type]
        system = str(request.get("system", ""))
        if "回放分析员" in system:
            return {
                "evidence": ["game0 中目标角色在新死讯后重新比较了票型。"],
                "strengths": ["能区分公开事实和玩家声明。"],
                "failure_modes": ["没有稳定记录撤退条件。"],
                "opponent_patterns": ["对手在票型变化后解释断裂。"],
                "counterfactuals": ["比较潜伏与冲锋姿态的退出时机。"],
                "uncertainties": ["样本量不足，不能下绝对结论。"],
            }
        if "Harness 设计员" in system:
            payload = json.loads(request["messages"][0]["content"])  # type: ignore[index]
            baseline = dict(payload["parent_harness"])
            baseline.pop("fingerprint", None)
            baseline["source_type"] = "replay_mutation"
            baseline["rationale"] = ["测试候选：增加新公开事件后的重新规划。"]
            baseline["cards"] = [
                {
                    "card_id": "replan-after-event",
                    "title": "新事件后重规划",
                    "trigger": "出现新死讯或票型转折",
                    "action_tendency": "先复核事实，再决定是否切换姿态。",
                    "counterexamples": ["单一发言不构成硬信息。"],
                    "exit_conditions": ["出现冲突的系统事实。"],
                    "observable_signals": ["新死讯", "投票结果"],
                    "confidence": 0.6,
                    "tags": ["replan"],
                }
            ]
            return {"candidates": [baseline]}
        return {
            "hard_rule_violations": [],
            "information_leakage_risks": [],
            "unsupported_claims": [],
            "complexity_concerns": [],
            "counterexamples": ["需要更多样本验证。"],
            "strengths": ["有明确退出条件。"],
        }


class FakeTaskClient:
    def __init__(self) -> None:
        self.requests: list[dict] = []

    async def complete_json(self, **request: object) -> dict:
        self.requests.append(request)  # type: ignore[arg-type]
        return {"kind": "speak", "text": "我会先比较公开票型。", "private_note": "记录新死讯后的重规划。"}


class DynamicTaskClient:
    async def complete_json(self, **request: object) -> dict:
        payload = json.loads(request["messages"][0]["content"])  # type: ignore[index]
        allowed = payload["packet"]["request"]["allowed_actions"]
        action = next((item for item in allowed if item["kind"] != ACTION_PASS), allowed[0])
        result = {"kind": action["kind"]}
        if action["kind"] in {ACTION_SPEAK, ACTION_LAST_WORDS}:
            result["text"] = "我会依据公开信息继续判断。"
        elif action.get("target_ids"):
            result["target_id"] = action["target_ids"][0]
        return result


def copied_store(directory: str) -> RoleStrategyStore:
    root = Path(directory) / "prompts"
    shutil.copytree(PROMPT_DIRECTORY / "roles", root / "roles")
    return RoleStrategyStore(root)


class Season2HarnessTest(unittest.IsolatedAsyncioTestCase):
    async def test_static_harness_gate_is_deterministic_and_model_free(self) -> None:
        store = RoleStrategyStore()
        profile = store.profile(ROLE_WOLF)
        first = evaluate_harness(HarnessSpec.baseline(ROLE_WOLF), profile)
        second = evaluate_harness(HarnessSpec.baseline(ROLE_WOLF), profile)
        self.assertIsInstance(first, HarnessStaticEvaluation)
        self.assertTrue(first.passed)
        self.assertEqual(first.as_dict(), second.as_dict())
        self.assertEqual(first.metrics["visible_event_count"], 28)

    async def test_runtime_limits_context_and_marks_replanning(self) -> None:
        store = RoleStrategyStore()
        profile = store.profile(ROLE_WOLF)
        spec = HarnessSpec.baseline(ROLE_WOLF, strategy=profile.strategy)
        runtime = HarnessRuntime(spec, profile)
        packet = {
            "game": {"phase": "day_discussion", "round": 1},
            "public_rules": {},
            "self": {"player_id": "p1"},
            "private_information": {"role": ROLE_WOLF},
            "public_state": {},
            "visible_events": [
                {"seq": i, "type": "PLAYER_SPOKE", "payload": {"text": "x" * 1000}}
                for i in range(80)
            ],
            "request": {"allowed_actions": []},
        }
        context = runtime.build_context(packet, private_notes=[str(i) for i in range(30)])
        self.assertLessEqual(len(context["visible_events"]), 28)
        self.assertLessEqual(len(context["task_agent_context"]["memory_notes"]), 12)
        self.assertTrue(context["task_agent_context"]["replan_required"])
        self.assertEqual(len(context["visible_events"][0]["payload"]["text"]), 260)
        runtime.update_belief_board(
            {
                "confirmed_facts": ["系统公布了首夜死讯"],
                "hypotheses": ["七号的票型需要下一轮验证"],
                "unknown_column": ["不应进入信念板"],
            }
        )
        updated = runtime.build_context(packet)["task_agent_context"]["belief_board"]
        self.assertEqual(updated["confirmed_facts"], ["系统公布了首夜死讯"])
        self.assertNotIn("unknown_column", updated)
        packet["game"] = {"phase": "day_discussion", "round": 2}
        next_context = runtime.build_context(packet)["task_agent_context"]
        self.assertTrue(next_context["replan_required"])
        self.assertEqual(next_context["replan_reason"], "游戏轮次发生变化")

    async def test_frozen_skill_store_is_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / ROLE_WOLF
            root.mkdir(parents=True)
            (root / "base.md").write_text("固定规则", encoding="utf-8")
            (root / "strategy.md").write_text("历史策略", encoding="utf-8")
            store = FrozenRoleStrategyStore(temporary)
            self.assertEqual(store.profile(ROLE_WOLF).strategy, "历史策略")
            with self.assertRaises(PermissionError):
                store.replace_strategy(ROLE_WOLF, "新策略")

    async def test_meta_agent_constructs_and_archives_candidate_without_other_skill(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = copied_store(temporary)
            archive = HarnessArchiveStore(Path(temporary) / "season2")
            client = FakeMetaClient()
            agent = TaskAgentMetaAgent(
                model_client=client,
                strategy_store=store,
                archive_store=archive,
            )
            records = [
                {
                    "metadata": {"game_id": "round0-game0"},
                    "events": [],
                    "runner_events": [],
                    "final_snapshot": {},
                }
            ]
            result = await agent.construct_task_agent(
                round_index=0,
                role=ROLE_WOLF,
                game_records=records,
                promote=True,
            )
            self.assertTrue(result.selected_passed)
            self.assertEqual(result.selected.role, ROLE_WOLF)
            self.assertGreaterEqual(len(result.candidates), 3)
            self.assertEqual(
                len({item.candidate.harness_id for item in result.candidates}),
                len(result.candidates),
            )
            self.assertIsNotNone(result.promoted_path)
            self.assertTrue(Path(result.promoted_path).exists())
            self.assertTrue(Path(result.experiment_paths["analysis_path"]).exists())
            self.assertTrue(Path(result.experiment_paths["evaluation_summary_path"]).exists())
            self.assertTrue(Path(result.experiment_paths["research_sources_path"]).exists())
            self.assertTrue(Path(result.experiment_paths["pipeline_trace_path"]).exists())
            self.assertTrue(any(item["stage"] == "selection" for item in result.stage_trace))
            self.assertTrue(Path(result.candidates[0].archive_path or "").joinpath("harness.json").exists())
            self.assertEqual(len(client.requests), 5)
            request_text = json.dumps(client.requests, ensure_ascii=False)
            self.assertNotIn("预言家固定规则", request_text)

    async def test_llm_participant_uses_harness_prompt_and_manifest(self) -> None:
        engine = GameEngine(
            game_id="season2-player-test",
            players=[{"id": f"p{i}", "name": f"P{i}"} for i in range(1, 8)],
            rules=create_default_rules(),
            seed="season2-player-seed",
        )
        engine.start()
        wolf_id = next(pid for pid, role in engine.role_assignments().items() if role == ROLE_WOLF)
        client = FakeTaskClient()
        participant = LlmParticipant(
            player_id=wolf_id,
            model_client=client,
            harness_specs={ROLE_WOLF: HarnessSpec.baseline(ROLE_WOLF)},
        )
        request = engine.discussion_request(wolf_id, "wolf")
        action = await participant.decide(engine.build_turn_packet(request))
        self.assertEqual(action["kind"], "speak")
        self.assertIn("Task-Agent Harness", client.requests[0]["system"])
        packet = json.loads(client.requests[0]["messages"][0]["content"])["packet"]
        self.assertIn("task_agent_context", packet)
        self.assertEqual(participant.agent_manifest()["agent_type"], "task_agent_harness")

    async def test_runner_persists_harness_versions_in_full_record_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            players = [{"id": f"p{i}", "name": f"P{i}"} for i in range(1, 8)]
            engine = GameEngine(
                game_id="season2-record-test",
                players=players,
                rules=create_default_rules(),
                seed="season2-record-seed",
            )
            spec = HarnessSpec.baseline(ROLE_WOLF)
            participants = {
                player["id"]: LlmParticipant(
                    player_id=player["id"],
                    model_client=DynamicTaskClient(),
                    harness_specs={ROLE_WOLF: spec},
                )
                for player in players
            }
            report = await GameRunner(
                engine=engine,
                participants=participants,
                record_directory=temporary,
                decision_timeout_seconds=2,
            ).run()
            full = json.loads(Path(report["record_path"]).read_text(encoding="utf-8"))
            public = json.loads(Path(report["public_record_path"]).read_text(encoding="utf-8"))
            self.assertIn("agent_manifests", full["metadata"])
            self.assertIn("agent_harness_catalog", full["metadata"])
            self.assertTrue(any(item.get("type") == "TASK_AGENT_HARNESS_TRACE" for item in full["runner_events"]))
            self.assertTrue(full["metadata"]["agent_manifests"])
            self.assertNotIn("agent_manifests", public["metadata"])

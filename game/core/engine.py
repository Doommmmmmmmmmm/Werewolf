"""狼人杀规则引擎。

这个类只负责状态、可见性、合法性和结算，不调用模型，也不决定一轮行动顺序。
夜晚和白天的顺序位于 loops/ 中，便于直接阅读每个 loop 的完整流程。
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import random
import re
from typing import Any, Iterable, Mapping

from .constants import (
    ACTION_DAY_VOTE,
    ACTION_GUARD_PROTECT,
    ACTION_HUNTER_SHOOT,
    ACTION_LAST_WORDS,
    ACTION_PASS,
    ACTION_SEER_INSPECT,
    ACTION_SHERIFF_BADGE_DESTROY,
    ACTION_SHERIFF_BADGE_TRANSFER,
    ACTION_SHERIFF_CANDIDATE,
    ACTION_SHERIFF_SPEECH_ORDER,
    ACTION_SHERIFF_VOTE,
    ACTION_SPEAK,
    ACTION_WITCH_HEAL,
    ACTION_WITCH_POISON,
    ACTION_WOLF_KILL_VOTE,
    CHANNEL_AUDIT,
    CHANNEL_PRIVATE,
    CHANNEL_PUBLIC,
    CHANNEL_WOLF,
    PHASE_DAY_DISCUSSION,
    PHASE_DAY_VOTE,
    PHASE_FINISHED,
    PHASE_HUNTER_REACTION,
    PHASE_LAST_WORDS,
    PHASE_NIGHT_GUARD,
    PHASE_NIGHT_RESOLVE,
    PHASE_NIGHT_SEER,
    PHASE_NIGHT_WITCH,
    PHASE_NIGHT_WOLF_DISCUSSION,
    PHASE_NIGHT_WOLF_VOTE,
    PHASE_NOT_STARTED,
    PHASE_SHERIFF_BADGE,
    PHASE_SHERIFF_CANDIDACY,
    PHASE_SHERIFF_ELECTION_SPEECH,
    PHASE_SHERIFF_ELECTION_VOTE,
    PHASE_SHERIFF_SPEECH_ORDER,
    ROLE_GUARD,
    ROLE_HUNTER,
    ROLE_IDIOT,
    ROLE_SEER,
    ROLE_VILLAGER,
    ROLE_WITCH,
    ROLE_WOLF,
    TEAM_VILLAGE,
    TEAM_WOLF,
    VISIBILITY_AUDIT,
    VISIBILITY_PRIVATE,
    VISIBILITY_PUBLIC,
    VISIBILITY_TEAM,
    public_phase_for,
    team_for_role,
)
from .errors import RuleViolationError
from .event_log import EventLog
from .models import NightState, Player
from .rules import RuleSet, create_default_rules


_CHINESE_CHARACTER = re.compile(r"[\u3400-\u9fff]")


class GameEngine:
    """规则唯一权威。

    每个公开/私密事件都从本类产生。外部 loop 只能提交结构化行动，不能直接修改
    身份、存活状态、药水或胜负。
    """

    def __init__(
        self,
        *,
        game_id: str = "game-1",
        players: list[dict[str, Any]],
        rules: RuleSet | None = None,
        seed: str = "werewolf",
        fixed_roles: Mapping[str, str] | None = None,
    ) -> None:
        self.game_id = str(game_id)
        self.seed = str(seed)
        self.rules = rules or create_default_rules()
        if len(players) != len(self.rules.role_deck):
            raise ValueError("玩家人数必须与身份牌数量相同")

        self.players = [
            Player.from_mapping(value, seat=index + 1)
            for index, value in enumerate(players)
        ]
        self.players.sort(key=lambda player: player.seat)
        player_ids = [player.player_id for player in self.players]
        if not all(player_ids) or len(set(player_ids)) != len(player_ids):
            raise ValueError("每位玩家必须具有唯一且非空的 id")

        self.fixed_roles = self._validate_fixed_roles(fixed_roles, player_ids)
        for role, requested_count in self._count_values(self.fixed_roles).items():
            available_count = self.rules.role_deck.count(role)
            if requested_count > available_count:
                raise ValueError(
                    f"fixed_roles 指定的{role}数量为 {requested_count}，"
                    f"但当前牌堆只有 {available_count} 张"
                )
        self.role_assignment_mode = "specified" if self.fixed_roles else "random"

        self._random = random.Random(self.seed)
        self.log = EventLog()
        self.status = "created"
        self.round = 0
        self.phase = PHASE_NOT_STARTED
        self.winner: str | None = None
        self.roles: dict[str, str] = {}
        self.role_state: dict[str, dict[str, Any]] = {}
        self.publicly_revealed_roles: dict[str, str] = {}
        self.night = NightState()
        self._pending_hunter_reactions: list[str] = []
        self._hunter_reaction_resume_phase: str | None = None
        self.sheriff_id: str | None = None
        self._sheriff_election_status = (
            "pending" if self.rules.enable_sheriff_election else "disabled"
        )
        self._sheriff_candidates: list[str] = []
        self._sheriff_election_runoff = False
        self._deferred_dawn_deaths: list[dict[str, str]] = []
        self._first_dawn_resolution_pending = False
        self._death_order: list[str] = []
        self._last_words_eligible_ids: list[str] = []
        self._pending_last_words: list[str] = []
        self._last_words_resume_phase: str | None = None
        self._pending_sheriff_badge_holders: list[str] = []
        self._sheriff_badge_resume_phase: str | None = None
        self._day_speech_direction = "next"
        self._day_speech_reference_seat = 1
        self._day_speech_reference_is_death = False
        self._phase_nonce = 0

    # ------------------------------------------------------------------
    # 生命周期：Runner 在每个白天/黑夜 loop 之间调用这些方法。
    # ------------------------------------------------------------------

    def start(self) -> dict[str, Any]:
        """发牌并开始第一夜。"""

        if self.status != "created":
            raise RuleViolationError("游戏已经开始")

        deck = list(self.rules.role_deck)
        # 固定身份只约束指定玩家；其余身份仍由同一个可复现随机源洗牌。
        # 先从牌堆移除固定牌，再将剩余牌随机分配给未指定玩家，避免改变
        # 规则引擎之后的可见性、行动顺序和结算逻辑。
        remaining_deck = list(deck)
        for role in self.fixed_roles.values():
            remaining_deck.remove(role)
        self._random.shuffle(remaining_deck)
        wolf_ids: list[str] = []
        remaining_players = [
            player for player in self.players if player.player_id not in self.fixed_roles
        ]
        assignment_by_player = dict(self.fixed_roles)
        assignment_by_player.update(
            {
                player.player_id: role
                for player, role in zip(remaining_players, remaining_deck, strict=True)
            }
        )
        for player in self.players:
            role = assignment_by_player[player.player_id]
            self.roles[player.player_id] = role
            self.role_state[player.player_id] = self._initial_role_state(role)
            if role == ROLE_WOLF:
                wolf_ids.append(player.player_id)

        self.status = "running"
        self.round = 1
        self.night = NightState()
        self._emit(
            "GAME_STARTED",
            visibility=VISIBILITY_PUBLIC,
            channel=CHANNEL_PUBLIC,
            payload={
                "game_id": self.game_id,
                "players": [
                    {
                        "id": player.player_id,
                        "name": player.name,
                        "seat": player.seat,
                    }
                    for player in self.players
                ],
                "round": self.round,
                "public_rules": self.public_rule_summary(),
            },
        )
        self._emit(
            "ROLE_ASSIGNMENTS_CREATED",
            visibility=VISIBILITY_AUDIT,
            channel=CHANNEL_AUDIT,
            payload={"assignments": deepcopy(self.roles)},
        )
        for player in self.players:
            role = self.roles[player.player_id]
            self._emit(
                "ROLE_ASSIGNED",
                visibility=VISIBILITY_PRIVATE,
                channel=CHANNEL_PRIVATE,
                recipients=[player.player_id],
                payload={
                    "player_id": player.player_id,
                    "role": role,
                    "team": team_for_role(role),
                },
            )
        self._emit(
            "WOLF_TEAM_REVEALED",
            visibility=VISIBILITY_TEAM,
            channel=CHANNEL_WOLF,
            recipients=wolf_ids,
            payload={"wolf_ids": wolf_ids},
        )
        self._announce_night()
        return self.public_state()

    def start_next_night(self) -> None:
        """白天 loop 结束且无人获胜时，进入下一夜。"""

        self._require_running()
        if self.has_pending_post_death_actions():
            raise RuleViolationError("必须先结算所有死亡后的公开反应")
        self.round += 1
        self.night = NightState()
        self._announce_night()

    def start_day(self) -> None:
        """夜晚结算完成且无人获胜时，进入白天。"""

        self._require_running()
        if self.has_pending_post_death_actions():
            raise RuleViolationError("必须先结算所有死亡后的公开反应")
        self._emit(
            "DAY_STARTED",
            visibility=VISIBILITY_PUBLIC,
            channel=CHANNEL_PUBLIC,
            payload={"round": self.round},
        )
        self.enter_phase(PHASE_DAY_DISCUSSION)

    def _announce_night(self) -> None:
        self._emit(
            "NIGHT_STARTED",
            visibility=VISIBILITY_PUBLIC,
            channel=CHANNEL_PUBLIC,
            payload={"round": self.round},
        )
        self.enter_phase(PHASE_NIGHT_WOLF_DISCUSSION)

    def enter_phase(self, phase: str) -> None:
        """切换内部阶段并留下审计事件。

        该方法刻意很小：阶段的执行顺序由 DayLoop / NightLoop 展开表达。
        """

        self._require_running()
        self.phase = phase
        self._phase_nonce += 1
        self._emit(
            "INTERNAL_PHASE_CHANGED",
            visibility=VISIBILITY_AUDIT,
            channel=CHANNEL_AUDIT,
            payload={"round": self.round, "phase": phase},
        )

    # ------------------------------------------------------------------
    # 状态投影：每一种读者拿到不同的信息视图。
    # ------------------------------------------------------------------

    def public_state(self) -> dict[str, Any]:
        players: list[dict[str, Any]] = []
        for player in self.players:
            view = player.public_view()
            if player.player_id in self.publicly_revealed_roles:
                view["revealed_role"] = self.publicly_revealed_roles[player.player_id]
            if player.alive and not self.can_day_vote(player.player_id):
                view["can_day_vote"] = False
            players.append(view)
        sheriff: dict[str, Any] | None = None
        if self.sheriff_id:
            sheriff_player = self._require_player(self.sheriff_id)
            sheriff = {
                "player_id": sheriff_player.player_id,
                "name": sheriff_player.name,
                "seat": sheriff_player.seat,
                "alive": sheriff_player.alive,
                "vote_weight": self.rules.sheriff_vote_weight,
            }
        result: dict[str, Any] = {
            "game_id": self.game_id,
            "status": self.status,
            "round": self.round,
            "phase": public_phase_for(self.phase),
            "winner": self.winner,
            "players": players,
            "sheriff": sheriff,
            "sheriff_election_status": self._sheriff_election_status,
            "day_speech_order": {
                "reference_seat": self._day_speech_reference_seat,
                "direction": self._day_speech_direction,
            },
        }
        if self.status == "finished" and self.rules.reveal_all_roles_at_end:
            result["roles_revealed"] = deepcopy(self.roles)
        return result

    def audit_state(self) -> dict[str, Any]:
        return {
            "game_id": self.game_id,
            "seed": self.seed,
            "status": self.status,
            "round": self.round,
            "phase": self.phase,
            "winner": self.winner,
            "players": [player.public_view() for player in self.players],
            "roles": deepcopy(self.roles),
            "role_state": deepcopy(self.role_state),
            "publicly_revealed_roles": deepcopy(self.publicly_revealed_roles),
            "pending_hunter_reactions": list(self._pending_hunter_reactions),
            "hunter_reaction_resume_phase": self._hunter_reaction_resume_phase,
            "sheriff_id": self.sheriff_id,
            "sheriff_election_status": self._sheriff_election_status,
            "sheriff_candidates": list(self._sheriff_candidates),
            "sheriff_election_runoff": self._sheriff_election_runoff,
            "deferred_dawn_deaths": deepcopy(self._deferred_dawn_deaths),
            "first_dawn_resolution_pending": self._first_dawn_resolution_pending,
            "death_order": list(self._death_order),
            "last_words_eligible_ids": list(self._last_words_eligible_ids),
            "pending_last_words": list(self._pending_last_words),
            "last_words_resume_phase": self._last_words_resume_phase,
            "pending_sheriff_badge_holders": list(self._pending_sheriff_badge_holders),
            "sheriff_badge_resume_phase": self._sheriff_badge_resume_phase,
            "day_speech_direction": self._day_speech_direction,
            "day_speech_reference_seat": self._day_speech_reference_seat,
            "day_speech_reference_is_death": self._day_speech_reference_is_death,
            "phase_nonce": self._phase_nonce,
            "night": self.night.as_dict(),
        }

    def public_rule_summary(self) -> dict[str, Any]:
        """返回所有玩家开局即应知道的配置，不泄露发牌结果。"""

        role_counts: dict[str, int] = {}
        for role in self.rules.role_deck:
            role_counts[role] = role_counts.get(role, 0) + 1
        night_order = [
            role
            for role in (ROLE_GUARD, ROLE_WOLF, ROLE_SEER, ROLE_WITCH)
            if role in role_counts
        ]
        return {
            "role_counts": role_counts,
            "implemented_roles": sorted(role_counts),
            "night_order": night_order,
            "reveal_role_on_death": self.rules.reveal_role_on_death,
            "reveal_all_roles_at_end": self.rules.reveal_all_roles_at_end,
            "wolf_win_condition": self.rules.wolf_win_condition,
            "wolves_can_target_wolves": self.rules.wolves_can_target_wolves,
            "public_speech_can_be_deceptive": True,
            "last_words": {
                "eligible_death_count": self.rules.last_words_count,
                "eligibility": (
                    "day_elimination_only"
                    if self.rules.last_words_day_elimination_only
                    else "global_first_deaths"
                ),
                "night_deaths_eligible": not self.rules.last_words_day_elimination_only,
                "max_chars": self.rules.max_day_speech_chars,
                "visibility": "public",
            },
            "sheriff": {
                "enabled": self.rules.enable_sheriff_election,
                "election_before_first_dawn": self.rules.enable_sheriff_election,
                "first_night_dead_can_participate": self.rules.enable_sheriff_election,
                "vote_weight": self.rules.sheriff_vote_weight,
                "tie_policy": "runoff_once_then_no_sheriff",
                "badge_on_death": "transfer_or_destroy",
            },
            "day_speech_order": {
                "normal_start": "after_largest_seat_among_dawn_deaths",
                "no_dawn_death_start_seat": 1,
                "sheriff_may_reverse": True,
            },
            "special_role_rules": {
                "guard_can_protect_self": self.rules.guard_can_protect_self,
                "guard_can_repeat_protect": self.rules.guard_can_repeat_protect,
                "idiot_survives_first_day_elimination": self.rules.idiot_survives_first_day_elimination,
                "idiot_can_vote_after_reveal": self.rules.idiot_can_vote_after_reveal,
            },
        }

    def player_view(self, player_id: str, since_sequence: int = 0) -> dict[str, Any]:
        player = self._require_player(player_id)
        role = self.roles.get(player_id)
        private_information: dict[str, Any] = {
            "role": role,
            "team": team_for_role(role) if role else None,
        }
        if role == ROLE_WOLF:
            private_information["wolf_teammates"] = [
                candidate.player_id
                for candidate in self.players
                if self.roles.get(candidate.player_id) == ROLE_WOLF
            ]
        elif role == ROLE_SEER:
            private_information["inspections"] = deepcopy(
                self.role_state.get(player_id, {}).get("inspections", {})
            )
        elif role == ROLE_WITCH:
            witch_state = self.role_state.get(player_id, {})
            private_information["antidote_available"] = bool(
                witch_state.get("antidote_available")
            )
            private_information["poison_available"] = bool(
                witch_state.get("poison_available")
            )
            if self.phase == PHASE_NIGHT_WITCH:
                private_information["wolf_target"] = self.night.wolf_target
        elif role == ROLE_GUARD:
            private_information["last_protected_id"] = self.role_state.get(
                player_id, {}
            ).get("last_protected_id")
        elif role == ROLE_HUNTER:
            death_shot_pending = player_id in self._pending_hunter_reactions
            private_information["shot_available"] = bool(
                self.role_state.get(player_id, {}).get("shot_available")
            ) or death_shot_pending
            private_information["death_shot_pending"] = death_shot_pending
        elif role == ROLE_IDIOT:
            idiot_state = self.role_state.get(player_id, {})
            private_information["revealed"] = bool(idiot_state.get("revealed"))
            private_information["can_day_vote"] = self.can_day_vote(player_id)

        return {
            "game_id": self.game_id,
            "round": self.round,
            "phase": public_phase_for(self.phase),
            "status": self.status,
            "self": {
                "player_id": player.player_id,
                "name": player.name,
                "seat": player.seat,
                "alive": player.alive,
            },
            "private_information": private_information,
            "public_state": self.public_state(),
            "visible_events": self.events_for(player_id, since_sequence),
            "latest_event_seq": self.latest_event_sequence,
        }

    def build_turn_packet(
        self, request: dict[str, Any], *, since_sequence: int = 0
    ) -> dict[str, Any]:
        """为一个玩家构造可发送给 LLM 或真人 UI 的行动包。"""

        player_id = str(request["player_id"])
        view = self.player_view(player_id, since_sequence)
        return {
            "version": 1,
            "game": {
                "game_id": self.game_id,
                "round": self.round,
                "phase": request["phase"],
                "public_phase": public_phase_for(self.phase),
                "status": self.status,
            },
            "public_rules": self.public_rule_summary(),
            "self": view["self"],
            "private_information": view["private_information"],
            "public_state": view["public_state"],
            "visible_events": view["visible_events"],
            # 参与者可将此隔离资源绑定到受限工具；它不会被 Task-Agent 默认
            # 拼进模型上下文。保留在行动包中是为了避免参与者直接访问引擎内部。
            "tool_context": {
                "current_round_dialogue": self.current_round_dialogue_for(player_id),
            },
            "request": deepcopy(request),
            "latest_event_seq": view["latest_event_seq"],
        }

    def build_public_sync_packet(
        self, player_id: str, *, since_sequence: int = 0
    ) -> dict[str, Any]:
        """夜晚开始时发送给每位存活玩家的纯公开同步包。"""

        player = self._require_player(player_id)
        return {
            "type": "public_state_sync",
            "game": {
                "game_id": self.game_id,
                "round": self.round,
                "phase": public_phase_for(self.phase),
                "status": self.status,
            },
            "public_rules": self.public_rule_summary(),
            "recipient": {
                "player_id": player.player_id,
                "name": player.name,
                "alive": player.alive,
            },
            "public_state": self.public_state(),
            "public_events": self.public_events(since_sequence),
            "latest_event_seq": self.latest_event_sequence,
        }

    # ------------------------------------------------------------------
    # 请求构造：每个 loop 显式选择它需要的请求类型。
    # ------------------------------------------------------------------

    def discussion_request(self, player_id: str, channel: str) -> dict[str, Any]:
        self._require_alive_player(player_id)
        if channel == CHANNEL_WOLF:
            max_chars = self.rules.max_wolf_speech_chars
        elif channel == CHANNEL_PUBLIC:
            max_chars = self.rules.max_day_speech_chars
        else:
            raise RuleViolationError("讨论只能发生在狼人私聊或公开频道")
        return self._request(
            player_id,
            channel,
            [
                {
                    "kind": ACTION_SPEAK,
                    "max_chars": max_chars,
                    "language": "zh-CN",
                    "require_chinese": self.rules.require_chinese_speech,
                },
                {"kind": ACTION_PASS},
            ],
        )

    def wolf_vote_request(self, player_id: str) -> dict[str, Any]:
        self._require_alive_player(player_id)
        if self.roles.get(player_id) != ROLE_WOLF:
            raise RuleViolationError("只有狼人可以参与狼人投票")
        targets = [
            player.player_id
            for player in self.alive_players()
            if self.rules.wolves_can_target_wolves
            or self.roles.get(player.player_id) != ROLE_WOLF
        ]
        return self._request(
            player_id,
            CHANNEL_WOLF,
            [
                {"kind": ACTION_WOLF_KILL_VOTE, "target_ids": targets},
                {"kind": ACTION_PASS},
            ],
        )

    def guard_request(self, player_id: str) -> dict[str, Any]:
        """为守卫构造夜间守护请求。"""

        self._require_alive_player(player_id)
        if self.roles.get(player_id) != ROLE_GUARD:
            raise RuleViolationError("只有守卫可以守护")
        guard_state = self.role_state[player_id]
        last_protected_id = guard_state.get("last_protected_id")
        targets = [
            player.player_id
            for player in self.alive_players()
            if (self.rules.guard_can_protect_self or player.player_id != player_id)
            and (
                self.rules.guard_can_repeat_protect
                or player.player_id != last_protected_id
            )
        ]
        return self._request(
            player_id,
            CHANNEL_PRIVATE,
            [
                {"kind": ACTION_GUARD_PROTECT, "target_ids": targets},
                {"kind": ACTION_PASS},
            ],
        )

    def seer_request(self, player_id: str) -> dict[str, Any]:
        self._require_alive_player(player_id)
        if self.roles.get(player_id) != ROLE_SEER:
            raise RuleViolationError("只有预言家可以查验")
        inspections = self.role_state[player_id].get("inspections", {})
        targets = [
            player.player_id
            for player in self.alive_players()
            if (self.rules.seer_can_inspect_self or player.player_id != player_id)
            and (
                self.rules.seer_can_repeat_inspect
                or player.player_id not in inspections
            )
        ]
        return self._request(
            player_id,
            CHANNEL_PRIVATE,
            [
                {"kind": ACTION_SEER_INSPECT, "target_ids": targets},
                {"kind": ACTION_PASS},
            ],
        )

    def witch_request(self, player_id: str) -> dict[str, Any]:
        self._require_alive_player(player_id)
        if self.roles.get(player_id) != ROLE_WITCH:
            raise RuleViolationError("只有女巫可以使用药水")
        witch_state = self.role_state[player_id]
        actions: list[dict[str, Any]] = []
        if witch_state.get("antidote_available") and self.night.wolf_target:
            target = self.night.wolf_target
            if self.rules.witch_can_self_save or target != player_id:
                actions.append({"kind": ACTION_WITCH_HEAL, "target_ids": [target]})
        if witch_state.get("poison_available"):
            targets = [
                player.player_id
                for player in self.alive_players()
                if self.rules.witch_can_poison_self or player.player_id != player_id
            ]
            actions.append({"kind": ACTION_WITCH_POISON, "target_ids": targets})
        actions.append({"kind": ACTION_PASS})
        return self._request(player_id, CHANNEL_PRIVATE, actions)

    def hunter_shot_request(self) -> dict[str, Any]:
        """为当前待结算的死亡猎人构造开枪请求。"""

        self._require_running()
        if not self._pending_hunter_reactions:
            raise RuleViolationError("没有待处理的猎人反应")
        if self.phase != PHASE_HUNTER_REACTION:
            self.enter_phase(PHASE_HUNTER_REACTION)

        player_id = self._pending_hunter_reactions[0]
        if self.roles.get(player_id) != ROLE_HUNTER:
            raise RuleViolationError("待处理反应玩家不是猎人")
        targets = [
            player.player_id
            for player in self.alive_players()
            if player.player_id != player_id
        ]
        return self._request(
            player_id,
            CHANNEL_PUBLIC,
            [
                {"kind": ACTION_HUNTER_SHOOT, "target_ids": targets},
                {"kind": ACTION_PASS},
            ],
        )

    def day_vote_request(self, player_id: str) -> dict[str, Any]:
        self._require_alive_player(player_id)
        if not self.can_day_vote(player_id):
            raise RuleViolationError("该玩家已失去白天投票权")
        targets = [
            player.player_id
            for player in self.alive_players()
            if player.player_id != player_id
        ]
        actions = [{"kind": ACTION_DAY_VOTE, "target_ids": targets}]
        if self.rules.allow_day_vote_abstain:
            actions.append({"kind": ACTION_PASS})
        return self._request(player_id, CHANNEL_PUBLIC, actions)

    def sheriff_candidacy_request(self, player_id: str) -> dict[str, Any]:
        """第一天警长竞选的上警/不上警选择。

        首夜死亡在这个阶段尚未公布也尚未结算，因此仍是合法竞选者和选民。
        """

        self._require_phase(PHASE_SHERIFF_CANDIDACY)
        self._require_alive_player(player_id)
        return self._request(
            player_id,
            CHANNEL_PUBLIC,
            [{"kind": ACTION_SHERIFF_CANDIDATE}, {"kind": ACTION_PASS}],
        )

    def sheriff_election_vote_request(self, player_id: str) -> dict[str, Any]:
        self._require_phase(PHASE_SHERIFF_ELECTION_VOTE)
        self._require_alive_player(player_id)
        actions: list[dict[str, Any]] = []
        if self._sheriff_candidates:
            actions.append(
                {
                    "kind": ACTION_SHERIFF_VOTE,
                    "target_ids": list(self._sheriff_candidates),
                }
            )
        actions.append({"kind": ACTION_PASS})
        return self._request(player_id, CHANNEL_PUBLIC, actions)

    def sheriff_speech_order_request(self) -> dict[str, Any]:
        """为存活警长请求当日常规发言的正序/逆序选择。"""

        self._require_phase(PHASE_DAY_DISCUSSION)
        if not self._has_living_sheriff():
            raise RuleViolationError("当前没有存活警长可以调整发言顺序")
        assert self.sheriff_id is not None
        self.enter_phase(PHASE_SHERIFF_SPEECH_ORDER)
        return self._request(
            self.sheriff_id,
            CHANNEL_PUBLIC,
            [
                {
                    "kind": ACTION_SHERIFF_SPEECH_ORDER,
                    "target_ids": ["next", "previous"],
                }
            ],
        )

    def sheriff_badge_request(self) -> dict[str, Any]:
        """为刚死亡的警长请求传警徽或撕警徽。"""

        self._require_phase(PHASE_SHERIFF_BADGE)
        if not self._pending_sheriff_badge_holders:
            raise RuleViolationError("没有待处理的警徽")
        player_id = self._pending_sheriff_badge_holders[0]
        targets = [player.player_id for player in self.alive_players()]
        actions: list[dict[str, Any]] = [{"kind": ACTION_SHERIFF_BADGE_DESTROY}]
        if targets:
            actions.insert(
                0,
                {"kind": ACTION_SHERIFF_BADGE_TRANSFER, "target_ids": targets},
            )
        return self._request(player_id, CHANNEL_PUBLIC, actions)

    def last_words_request(self) -> dict[str, Any]:
        """为前 ``last_words_count`` 名合格的白天放逐者请求公开遗言。"""

        self._require_phase(PHASE_LAST_WORDS)
        if not self._pending_last_words:
            raise RuleViolationError("没有待发表遗言的玩家")
        player_id = self._pending_last_words[0]
        return self._request(
            player_id,
            CHANNEL_PUBLIC,
            [
                {
                    "kind": ACTION_LAST_WORDS,
                    "max_chars": self.rules.max_day_speech_chars,
                    "language": "zh-CN",
                    "require_chinese": self.rules.require_chinese_speech,
                },
                {"kind": ACTION_PASS},
            ],
        )

    def _request(
        self,
        player_id: str,
        channel: str,
        allowed_actions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "request_id": ":".join(
                [self.game_id, str(self.round), self.phase, player_id, str(self._phase_nonce)]
            ),
            "player_id": player_id,
            "round": self.round,
            "phase": self.phase,
            "channel": channel,
            "allowed_actions": deepcopy(allowed_actions),
        }

    # ------------------------------------------------------------------
    # 行动接收与校验。
    # ------------------------------------------------------------------

    def accept_action(
        self, request: dict[str, Any], raw_action: dict[str, Any] | None
    ) -> dict[str, Any]:
        """校验一份行动，并记录它。

        对话行动立即变成可见事件；技能和投票先写入审计事件，之后由对应
        resolve_* 方法统一结算。
        """

        self._require_running()
        if request.get("phase") != self.phase or request.get("round") != self.round:
            raise RuleViolationError("行动请求已过期", request=request.get("request_id"))
        player_id = str(request.get("player_id", ""))
        if self.phase == PHASE_HUNTER_REACTION:
            self._require_pending_hunter_reactor(player_id)
        elif self.phase == PHASE_LAST_WORDS:
            self._require_pending_last_words_speaker(player_id)
        elif self.phase == PHASE_SHERIFF_BADGE:
            self._require_pending_sheriff_badge_holder(player_id)
        else:
            self._require_alive_player(player_id)
        source = raw_action or {}
        submitted_request_id = source.get("request_id")
        if submitted_request_id and submitted_request_id != request["request_id"]:
            raise RuleViolationError("request_id 与当前回合不一致")
        submitted_player_id = source.get("player_id", source.get("playerId"))
        if submitted_player_id and str(submitted_player_id) != player_id:
            raise RuleViolationError("行动不能代表其他玩家提交")

        kind = str(source.get("kind", source.get("action", "")))
        allowed = next(
            (item for item in request["allowed_actions"] if item["kind"] == kind),
            None,
        )
        if allowed is None:
            raise RuleViolationError("当前阶段不允许该行动", kind=kind, phase=self.phase)

        target_id = source.get("target_id", source.get("targetId"))
        target_ids = allowed.get("target_ids")
        if target_ids is not None:
            if target_ids and target_id not in target_ids:
                raise RuleViolationError("目标不在合法列表中", target_id=target_id)
            if not target_ids and target_id:
                raise RuleViolationError("该行动不接受目标")
        elif target_id:
            raise RuleViolationError("该行动不接受目标")

        text: str | None = None
        if kind in {ACTION_SPEAK, ACTION_LAST_WORDS}:
            text = str(source.get("text", "")).strip()
            if not text:
                raise RuleViolationError("发言不能为空")
            if len(text) > int(allowed["max_chars"]):
                raise RuleViolationError("发言超过字数限制")
            if allowed.get("require_chinese") and not _CHINESE_CHARACTER.search(text):
                raise RuleViolationError("发言必须包含中文")

        action: dict[str, Any] = {
            "request_id": request["request_id"],
            "player_id": player_id,
            "kind": kind,
        }
        if target_id:
            action["target_id"] = str(target_id)
        if text is not None:
            action["text"] = text

        if self.phase in {
            PHASE_NIGHT_WOLF_DISCUSSION,
            PHASE_DAY_DISCUSSION,
            PHASE_SHERIFF_ELECTION_SPEECH,
        }:
            self._record_speech(action, request["channel"])
        else:
            self._emit(
                "ACTION_SUBMITTED",
                visibility=VISIBILITY_AUDIT,
                channel=CHANNEL_AUDIT,
                payload={"phase": self.phase, "action": deepcopy(action)},
            )
        return deepcopy(action)

    def fallback_action(self, request: dict[str, Any]) -> dict[str, Any]:
        """模型报错或超时后的保守合法行动。"""

        pass_action = next(
            (item for item in request["allowed_actions"] if item["kind"] == ACTION_PASS),
            None,
        )
        selected = pass_action or request["allowed_actions"][0]
        action: dict[str, Any] = {"kind": selected["kind"]}
        if selected["kind"] in {ACTION_SPEAK, ACTION_LAST_WORDS}:
            action["text"] = "我暂时没有更多信息。"
        elif selected.get("target_ids"):
            action["target_id"] = selected["target_ids"][0]
        return action

    def _record_speech(self, action: dict[str, Any], channel: str) -> None:
        is_wolf_room = channel == CHANNEL_WOLF
        stage = (
            "sheriff_election"
            if self.phase == PHASE_SHERIFF_ELECTION_SPEECH
            else "day_discussion"
            if self.phase == PHASE_DAY_DISCUSSION
            else "wolf_discussion"
        )
        self._emit(
            "PLAYER_SPOKE" if action["kind"] == ACTION_SPEAK else "PLAYER_PASSED",
            visibility=VISIBILITY_TEAM if is_wolf_room else VISIBILITY_PUBLIC,
            channel=CHANNEL_WOLF if is_wolf_room else CHANNEL_PUBLIC,
            recipients=[player.player_id for player in self.wolf_players()] if is_wolf_room else [],
            payload={
                "player_id": action["player_id"],
                "stage": stage,
                **({"text": action["text"]} if action.get("text") else {}),
            },
        )

    # ------------------------------------------------------------------
    # 结算原语：NightLoop / DayLoop 在合适的位置调用。
    # ------------------------------------------------------------------

    def resolve_wolf_vote(self, actions: Iterable[dict[str, Any]]) -> None:
        self._require_phase(PHASE_NIGHT_WOLF_VOTE)
        action_list = list(actions)
        result = self._resolve_plurality(
            action_list, ACTION_WOLF_KILL_VOTE, self.rules.wolf_tie_policy
        )
        self.night.wolf_target = result["target"]
        wolf_ids = [player.player_id for player in self.wolf_players()]
        self._emit(
            "WOLF_TARGET_SELECTED",
            visibility=VISIBILITY_TEAM,
            channel=CHANNEL_WOLF,
            recipients=wolf_ids,
            payload={
                "target_id": result["target"],
                "tied_target_ids": result["tied_targets"],
                "counts": result["counts"],
            },
        )
        self._emit(
            "WOLF_VOTE_RESOLVED",
            visibility=VISIBILITY_AUDIT,
            channel=CHANNEL_AUDIT,
            payload={"actions": action_list, **result},
        )

    def resolve_guard_actions(self, actions: Iterable[dict[str, Any]]) -> None:
        """记录本夜的守护目标；守护仅阻挡狼刀，不阻挡毒药。"""

        self._require_phase(PHASE_NIGHT_GUARD)
        action_list = list(actions)
        for action in action_list:
            if action["kind"] != ACTION_GUARD_PROTECT:
                continue
            target_id = action["target_id"]
            guard_state = self.role_state[action["player_id"]]
            guard_state["last_protected_id"] = target_id
            self.night.guarded_targets.append(target_id)
            self._emit(
                "GUARD_ACTION_CONFIRMED",
                visibility=VISIBILITY_PRIVATE,
                channel=CHANNEL_PRIVATE,
                recipients=[action["player_id"]],
                payload={"target_id": target_id},
            )
        self._emit(
            "GUARD_PHASE_RESOLVED",
            visibility=VISIBILITY_AUDIT,
            channel=CHANNEL_AUDIT,
            payload={"actions": action_list, "guarded_targets": list(self.night.guarded_targets)},
        )

    def resolve_seer_actions(self, actions: Iterable[dict[str, Any]]) -> None:
        self._require_phase(PHASE_NIGHT_SEER)
        action_list = list(actions)
        for action in action_list:
            if action["kind"] != ACTION_SEER_INSPECT:
                continue
            team = team_for_role(self.roles[action["target_id"]])
            inspections = self.role_state[action["player_id"]].setdefault(
                "inspections", {}
            )
            inspections[action["target_id"]] = team
            self._emit(
                "SEER_RESULT",
                visibility=VISIBILITY_PRIVATE,
                channel=CHANNEL_PRIVATE,
                recipients=[action["player_id"]],
                payload={"target_id": action["target_id"], "team": team},
            )
        self._emit(
            "SEER_PHASE_RESOLVED",
            visibility=VISIBILITY_AUDIT,
            channel=CHANNEL_AUDIT,
            payload={"actions": action_list},
        )

    def resolve_witch_actions(self, actions: Iterable[dict[str, Any]]) -> None:
        self._require_phase(PHASE_NIGHT_WITCH)
        action_list = list(actions)
        for action in action_list:
            if action["kind"] == ACTION_PASS:
                continue
            witch_state = self.role_state[action["player_id"]]
            if action["kind"] == ACTION_WITCH_HEAL:
                witch_state["antidote_available"] = False
                self.night.healed_targets.append(action["target_id"])
            elif action["kind"] == ACTION_WITCH_POISON:
                witch_state["poison_available"] = False
                self.night.poison_targets.append(action["target_id"])
            self._emit(
                "WITCH_ACTION_CONFIRMED",
                visibility=VISIBILITY_PRIVATE,
                channel=CHANNEL_PRIVATE,
                recipients=[action["player_id"]],
                payload={
                    "kind": action["kind"],
                    "target_id": action.get("target_id"),
                },
            )
        self._emit(
            "WITCH_PHASE_RESOLVED",
            visibility=VISIBILITY_AUDIT,
            channel=CHANNEL_AUDIT,
            payload={"actions": action_list},
        )

    # ------------------------------------------------------------------
    # 首日警长竞选：首夜死亡延后至竞选结束后公布，因此死者仍能参选和投票。
    # ------------------------------------------------------------------

    def has_pending_sheriff_election(self) -> bool:
        return (
            self.rules.enable_sheriff_election
            and self._sheriff_election_status == "pending"
            and self._first_dawn_resolution_pending
        )

    def begin_sheriff_election(self) -> None:
        self._require_phase(PHASE_NIGHT_RESOLVE)
        if not self.has_pending_sheriff_election():
            raise RuleViolationError("当前没有待进行的警长竞选")
        self._emit(
            "SHERIFF_ELECTION_STARTED",
            visibility=VISIBILITY_PUBLIC,
            channel=CHANNEL_PUBLIC,
            payload={"round": self.round, "speaking_order": "seat_ascending"},
        )
        self.enter_phase(PHASE_SHERIFF_CANDIDACY)

    def resolve_sheriff_candidacies(self, actions: Iterable[dict[str, Any]]) -> None:
        self._require_phase(PHASE_SHERIFF_CANDIDACY)
        action_list = list(actions)
        candidates = {
            action["player_id"]
            for action in action_list
            if action.get("kind") == ACTION_SHERIFF_CANDIDATE
        }
        self._sheriff_candidates = [
            player.player_id for player in self.players if player.player_id in candidates
        ]
        self._emit(
            "SHERIFF_CANDIDATES_ANNOUNCED",
            visibility=VISIBILITY_PUBLIC,
            channel=CHANNEL_PUBLIC,
            payload={"candidate_ids": list(self._sheriff_candidates)},
        )
        self._emit(
            "SHERIFF_CANDIDACY_RESOLVED",
            visibility=VISIBILITY_AUDIT,
            channel=CHANNEL_AUDIT,
            payload={"actions": action_list, "candidate_ids": list(self._sheriff_candidates)},
        )
        if not self._sheriff_candidates:
            self._complete_sheriff_election_without_winner("no_candidates")
            return
        self.begin_sheriff_election_speeches()

    def begin_sheriff_election_speeches(self) -> None:
        if self.phase not in {
            PHASE_SHERIFF_CANDIDACY,
            PHASE_SHERIFF_ELECTION_VOTE,
        }:
            raise RuleViolationError("当前不能开始警长竞选发言")
        if not self._sheriff_candidates:
            raise RuleViolationError("没有警长候选人")
        self.enter_phase(PHASE_SHERIFF_ELECTION_SPEECH)
        self._emit(
            "SHERIFF_ELECTION_SPEECHES_STARTED",
            visibility=VISIBILITY_PUBLIC,
            channel=CHANNEL_PUBLIC,
            payload={
                "candidate_ids": list(self._sheriff_candidates),
                "runoff": self._sheriff_election_runoff,
                "speaking_order": "seat_ascending",
            },
        )

    def sheriff_election_speakers(self) -> list[Player]:
        return [
            player
            for player in self.players
            if player.player_id in self._sheriff_candidates and player.alive
        ]

    def begin_sheriff_election_vote(self) -> None:
        self._require_phase(PHASE_SHERIFF_ELECTION_SPEECH)
        self.enter_phase(PHASE_SHERIFF_ELECTION_VOTE)
        self._emit(
            "SHERIFF_ELECTION_VOTE_STARTED",
            visibility=VISIBILITY_PUBLIC,
            channel=CHANNEL_PUBLIC,
            payload={
                "candidate_ids": list(self._sheriff_candidates),
                "runoff": self._sheriff_election_runoff,
            },
        )

    def resolve_sheriff_election_vote(self, actions: Iterable[dict[str, Any]]) -> None:
        self._require_phase(PHASE_SHERIFF_ELECTION_VOTE)
        action_list = list(actions)
        result = self._resolve_plurality(
            action_list, ACTION_SHERIFF_VOTE, "no_elimination"
        )
        for action in action_list:
            self._emit(
                "SHERIFF_VOTE_CAST",
                visibility=VISIBILITY_PUBLIC,
                channel=CHANNEL_PUBLIC,
                payload={
                    "voter_id": action["player_id"],
                    "target_id": action.get("target_id"),
                },
            )
        self._emit(
            "SHERIFF_ELECTION_VOTE_RESOLVED",
            visibility=VISIBILITY_AUDIT,
            channel=CHANNEL_AUDIT,
            payload={"actions": action_list, **result, "runoff": self._sheriff_election_runoff},
        )
        if result["target"]:
            self.sheriff_id = result["target"]
            self._sheriff_election_status = "elected"
            self._emit(
                "SHERIFF_ELECTED",
                visibility=VISIBILITY_PUBLIC,
                channel=CHANNEL_PUBLIC,
                payload={"player_id": self.sheriff_id, "vote_weight": self.rules.sheriff_vote_weight},
            )
            self.enter_phase(PHASE_NIGHT_RESOLVE)
            return
        if result["tied_targets"] and not self._sheriff_election_runoff:
            self._sheriff_candidates = list(result["tied_targets"])
            self._sheriff_election_runoff = True
            self._emit(
                "SHERIFF_ELECTION_TIED",
                visibility=VISIBILITY_PUBLIC,
                channel=CHANNEL_PUBLIC,
                payload={"candidate_ids": list(self._sheriff_candidates), "runoff": True},
            )
            self.begin_sheriff_election_speeches()
            return
        self._complete_sheriff_election_without_winner("runoff_tie_or_no_vote")

    def _complete_sheriff_election_without_winner(self, reason: str) -> None:
        self.sheriff_id = None
        self._sheriff_election_status = "no_sheriff"
        self._emit(
            "SHERIFF_ELECTION_FAILED",
            visibility=VISIBILITY_PUBLIC,
            channel=CHANNEL_PUBLIC,
            payload={"reason": reason},
        )
        self.enter_phase(PHASE_NIGHT_RESOLVE)

    # ------------------------------------------------------------------
    # 白天发言顺序与警徽交接。
    # ------------------------------------------------------------------

    def prepare_day_speech_order(self) -> bool:
        """确定当天的死者锚点；返回是否需要向存活警长请求方向。"""

        self._require_phase(PHASE_DAY_DISCUSSION)
        dawn_deaths = [
            death["player_id"]
            for death in self.night.deaths
            if self._find_player(death["player_id"]) is not None
        ]
        if dawn_deaths:
            self._day_speech_reference_seat = max(
                self._require_player(player_id).seat for player_id in dawn_deaths
            )
            self._day_speech_reference_is_death = True
        else:
            self._day_speech_reference_seat = 1
            self._day_speech_reference_is_death = False
        self._day_speech_direction = "next"
        if self._has_living_sheriff():
            return True
        self._emit_day_speech_order_set()
        return False

    def resolve_sheriff_speech_order(self, action: dict[str, Any]) -> None:
        self._require_phase(PHASE_SHERIFF_SPEECH_ORDER)
        if not self._has_living_sheriff() or action.get("player_id") != self.sheriff_id:
            raise RuleViolationError("只有存活警长可以调整发言顺序")
        if action.get("kind") != ACTION_SHERIFF_SPEECH_ORDER:
            raise RuleViolationError("警长需要选择发言方向")
        direction = action.get("target_id")
        if direction not in {"next", "previous"}:
            raise RuleViolationError("发言方向只能是 next 或 previous")
        self._day_speech_direction = str(direction)
        self._emit_day_speech_order_set()
        self.enter_phase(PHASE_DAY_DISCUSSION)

    def day_discussion_players(self) -> list[Player]:
        """按当日死者锚点及警长方向返回存活玩家发言顺序。"""

        step = 1 if self._day_speech_direction == "next" else -1
        seats = {player.seat: player for player in self.players if player.alive}
        total = len(self.players)
        start = self._day_speech_reference_seat
        if self._day_speech_reference_is_death:
            start = ((start - 1 + step) % total) + 1
        result: list[Player] = []
        for offset in range(total):
            seat = ((start - 1 + step * offset) % total) + 1
            if seat in seats:
                result.append(seats[seat])
        return result

    def begin_sheriff_badge_resolution(self) -> bool:
        if not self._pending_sheriff_badge_holders:
            return False
        self.enter_phase(PHASE_SHERIFF_BADGE)
        return True

    def resolve_sheriff_badge(self, action: dict[str, Any]) -> None:
        self._require_phase(PHASE_SHERIFF_BADGE)
        if not self._pending_sheriff_badge_holders:
            raise RuleViolationError("没有待处理的警徽")
        holder_id = self._pending_sheriff_badge_holders[0]
        trigger_phase = self._sheriff_badge_resume_phase
        if action.get("player_id") != holder_id:
            raise RuleViolationError("只有死亡警长可以处理警徽")
        kind = action.get("kind")
        if kind not in {ACTION_SHERIFF_BADGE_TRANSFER, ACTION_SHERIFF_BADGE_DESTROY}:
            raise RuleViolationError("警长死亡后只能传递或撕毁警徽")
        self._pending_sheriff_badge_holders.pop(0)
        if kind == ACTION_SHERIFF_BADGE_TRANSFER:
            target_id = str(action.get("target_id") or "")
            target = self._find_player(target_id)
            if target is None or not target.alive:
                raise RuleViolationError("警徽只能传给存活玩家")
            self.sheriff_id = target_id
            self._emit(
                "SHERIFF_BADGE_TRANSFERRED",
                visibility=VISIBILITY_PUBLIC,
                channel=CHANNEL_PUBLIC,
                payload={
                    "from_player_id": holder_id,
                    "to_player_id": target_id,
                    "trigger_phase": trigger_phase,
                },
            )
        else:
            self.sheriff_id = None
            self._emit(
                "SHERIFF_BADGE_DESTROYED",
                visibility=VISIBILITY_PUBLIC,
                channel=CHANNEL_PUBLIC,
                payload={"player_id": holder_id, "trigger_phase": trigger_phase},
            )
        if self._pending_sheriff_badge_holders:
            return
        resume_phase = self._sheriff_badge_resume_phase
        self._sheriff_badge_resume_phase = None
        if resume_phase is None:
            raise RuntimeError("警徽交接缺少恢复阶段")
        self.enter_phase(resume_phase)
        self._finish_if_no_pending_post_death_actions()

    # ------------------------------------------------------------------
    # 遗言：仅前若干名被白天放逐的玩家依放逐结算顺序公开发言。
    # ------------------------------------------------------------------

    def begin_last_words(self) -> bool:
        if not self._pending_last_words:
            return False
        self.enter_phase(PHASE_LAST_WORDS)
        return True

    def resolve_last_words(self, action: dict[str, Any]) -> None:
        self._require_phase(PHASE_LAST_WORDS)
        if not self._pending_last_words:
            raise RuleViolationError("没有待发表遗言的玩家")
        player_id = self._pending_last_words[0]
        if action.get("player_id") != player_id:
            raise RuleViolationError("当前不是该玩家发表遗言的时机")
        if action.get("kind") not in {ACTION_LAST_WORDS, ACTION_PASS}:
            raise RuleViolationError("遗言阶段只能发言或跳过")
        self._pending_last_words.pop(0)
        death_index = self._death_order.index(player_id) + 1
        eligible_death_index = self._last_words_eligible_ids.index(player_id) + 1
        trigger_phase = self._last_words_resume_phase
        if action.get("kind") == ACTION_LAST_WORDS:
            self._emit(
                "PLAYER_LAST_WORDS",
                visibility=VISIBILITY_PUBLIC,
                channel=CHANNEL_PUBLIC,
                payload={
                    "player_id": player_id,
                    "text": str(action.get("text") or ""),
                    "death_index": death_index,
                    "eligible_death_index": eligible_death_index,
                    "trigger_phase": trigger_phase,
                },
            )
        else:
            self._emit(
                "PLAYER_LAST_WORDS_SKIPPED",
                visibility=VISIBILITY_PUBLIC,
                channel=CHANNEL_PUBLIC,
                payload={
                    "player_id": player_id,
                    "death_index": death_index,
                    "eligible_death_index": eligible_death_index,
                    "trigger_phase": trigger_phase,
                },
            )
        if self._pending_last_words:
            return
        resume_phase = self._last_words_resume_phase
        self._last_words_resume_phase = None
        if resume_phase is None:
            raise RuntimeError("遗言阶段缺少恢复阶段")
        self.enter_phase(resume_phase)
        self._finish_if_no_pending_post_death_actions()

    def resolve_night(self) -> None:
        self._require_phase(PHASE_NIGHT_RESOLVE)
        deaths: list[dict[str, str]] = []
        if (
            self.night.wolf_target
            and self.night.wolf_target not in self.night.guarded_targets
            and self.night.wolf_target not in self.night.healed_targets
        ):
            deaths.append(
                {"player_id": self.night.wolf_target, "cause": "wolf_attack"}
            )
        for target_id in self.night.poison_targets:
            if not any(death["player_id"] == target_id for death in deaths):
                deaths.append({"player_id": target_id, "cause": "witch_poison"})

        self.night.deaths = list(deaths)
        self._emit(
            "NIGHT_RESOLVED",
            visibility=VISIBILITY_AUDIT,
            channel=CHANNEL_AUDIT,
            payload=self.night.as_dict(),
        )
        if self.round == 1 and self.rules.enable_sheriff_election and self._sheriff_election_status == "pending":
            self._deferred_dawn_deaths = list(deaths)
            self._first_dawn_resolution_pending = True
            self._emit(
                "DAWN_DEFERRED_FOR_SHERIFF_ELECTION",
                visibility=VISIBILITY_AUDIT,
                channel=CHANNEL_AUDIT,
                payload={"round": self.round, "pending_deaths": list(deaths)},
            )
            return
        actual_deaths = self._kill_players([item["player_id"] for item in deaths])
        self.night.deaths = [
            next(item for item in deaths if item["player_id"] == player_id)
            for player_id in actual_deaths
        ]
        self._announce_dawn(actual_deaths)
        self._finish_if_no_pending_post_death_actions()

    def reveal_deferred_first_dawn(self) -> None:
        """警长竞选结束后，统一公布并结算首夜死亡。"""

        self._require_phase(PHASE_NIGHT_RESOLVE)
        if not self._first_dawn_resolution_pending:
            raise RuleViolationError("当前没有待公布的首夜死讯")
        deaths = list(self._deferred_dawn_deaths)
        self._deferred_dawn_deaths = []
        self._first_dawn_resolution_pending = False
        actual_deaths = self._kill_players([item["player_id"] for item in deaths])
        self.night.deaths = [
            next(item for item in deaths if item["player_id"] == player_id)
            for player_id in actual_deaths
        ]
        self._announce_dawn(actual_deaths)
        self._finish_if_no_pending_post_death_actions()

    def _announce_dawn(self, actual_deaths: list[str]) -> None:
        self._emit(
            "DAWN_ANNOUNCED",
            visibility=VISIBILITY_PUBLIC,
            channel=CHANNEL_PUBLIC,
            payload={"round": self.round, "dead_player_ids": actual_deaths},
        )
        if not actual_deaths:
            self._emit(
                "NO_ONE_DIED",
                visibility=VISIBILITY_PUBLIC,
                channel=CHANNEL_PUBLIC,
                payload={"round": self.round},
            )
        for player_id in actual_deaths:
            self._emit_player_died(player_id)

    def resolve_day_vote(self, actions: Iterable[dict[str, Any]]) -> None:
        self._require_phase(PHASE_DAY_VOTE)
        action_list = list(actions)
        vote_weights = (
            {self.sheriff_id: self.rules.sheriff_vote_weight}
            if self._has_living_sheriff() and self.sheriff_id is not None
            else None
        )
        result = self._resolve_plurality(
            action_list,
            ACTION_DAY_VOTE,
            self.rules.day_tie_policy,
            vote_weights=vote_weights,
        )
        if self.rules.reveal_individual_votes:
            for action in action_list:
                weight = (
                    vote_weights.get(action["player_id"], 1)
                    if vote_weights is not None
                    else 1
                )
                self._emit(
                    "VOTE_CAST",
                    visibility=VISIBILITY_PUBLIC,
                    channel=CHANNEL_PUBLIC,
                    payload={
                        "voter_id": action["player_id"],
                        "target_id": action.get("target_id"),
                        "weight": weight,
                    },
                )
        self._emit(
            "DAY_VOTE_RESOLVED",
            visibility=VISIBILITY_AUDIT,
            channel=CHANNEL_AUDIT,
            payload={"actions": action_list, **result},
        )
        if result["target"]:
            target_id = result["target"]
            idiot_state = self.role_state.get(target_id, {})
            if (
                self.roles.get(target_id) == ROLE_IDIOT
                and self.rules.idiot_survives_first_day_elimination
                and not idiot_state.get("revealed")
            ):
                idiot_state["revealed"] = True
                idiot_state["lost_vote_right"] = not self.rules.idiot_can_vote_after_reveal
                self.publicly_revealed_roles[target_id] = ROLE_IDIOT
                self._emit(
                    "IDIOT_REVEALED",
                    visibility=VISIBILITY_PUBLIC,
                    channel=CHANNEL_PUBLIC,
                    payload={
                        "player_id": target_id,
                        "role": ROLE_IDIOT,
                        "lost_vote_right": idiot_state["lost_vote_right"],
                    },
                )
            else:
                eliminated = self._kill_players(
                    [target_id], eligible_for_last_words=True
                )
                if eliminated:
                    payload: dict[str, Any] = {"player_id": target_id}
                    if self.rules.reveal_role_on_death:
                        payload["role"] = self._publicly_reveal_role(target_id)
                    self._emit(
                        "PLAYER_ELIMINATED",
                        visibility=VISIBILITY_PUBLIC,
                        channel=CHANNEL_PUBLIC,
                        payload=payload,
                    )
        else:
            event_type = "VOTE_TIE" if len(result["tied_targets"]) > 1 else "NO_ONE_ELIMINATED"
            self._emit(
                event_type,
                visibility=VISIBILITY_PUBLIC,
                channel=CHANNEL_PUBLIC,
                payload={"tied_target_ids": result["tied_targets"]},
            )
        self._finish_if_no_pending_post_death_actions()

    def resolve_hunter_shot(self, action: dict[str, Any]) -> None:
        """结算一名死亡猎人的开枪，并处理可能产生的连锁猎人反应。"""

        self._require_phase(PHASE_HUNTER_REACTION)
        if not self._pending_hunter_reactions:
            raise RuleViolationError("没有待结算的猎人反应")
        hunter_id = self._pending_hunter_reactions[0]
        if action.get("player_id") != hunter_id:
            raise RuleViolationError("只有当前死亡猎人可以开枪")
        if action.get("kind") not in {ACTION_HUNTER_SHOOT, ACTION_PASS}:
            raise RuleViolationError("猎人反应只能开枪或跳过")
        if action.get("kind") == ACTION_HUNTER_SHOOT:
            target_id = action.get("target_id")
            legal_targets = {
                player.player_id
                for player in self.alive_players()
                if player.player_id != hunter_id
            }
            if target_id not in legal_targets:
                raise RuleViolationError("猎人的开枪目标不合法")
        self._pending_hunter_reactions.pop(0)

        target_id = action.get("target_id")
        trigger_phase = self._hunter_reaction_resume_phase
        self.publicly_revealed_roles[hunter_id] = ROLE_HUNTER
        if action["kind"] == ACTION_HUNTER_SHOOT:
            self._emit(
                "HUNTER_SHOT_FIRED",
                visibility=VISIBILITY_PUBLIC,
                channel=CHANNEL_PUBLIC,
                payload={
                    "player_id": hunter_id,
                    "target_id": target_id,
                    "trigger_phase": trigger_phase,
                },
            )
            actual_deaths = self._kill_players([str(target_id)])
            for player_id in actual_deaths:
                self._emit_player_died(player_id)
        elif action["kind"] == ACTION_PASS:
            self._emit(
                "HUNTER_SHOT_SKIPPED",
                visibility=VISIBILITY_PUBLIC,
                channel=CHANNEL_PUBLIC,
                payload={"player_id": hunter_id, "trigger_phase": trigger_phase},
            )
            actual_deaths = []
        self._emit(
            "HUNTER_SHOT_RESOLVED",
            visibility=VISIBILITY_AUDIT,
            channel=CHANNEL_AUDIT,
            payload={
                "hunter_id": hunter_id,
                "kind": action["kind"],
                "target_id": target_id,
                "dead_player_ids": actual_deaths,
                "trigger_phase": trigger_phase,
                "pending_hunter_ids": list(self._pending_hunter_reactions),
            },
        )
        if self._pending_hunter_reactions:
            return

        resume_phase = self._hunter_reaction_resume_phase
        self._hunter_reaction_resume_phase = None
        if resume_phase is None:
            raise RuntimeError("猎人反应缺少恢复阶段")
        self.enter_phase(resume_phase)
        self._finish_if_no_pending_post_death_actions()

    # ------------------------------------------------------------------
    # 记录与查询接口。
    # ------------------------------------------------------------------

    @property
    def latest_event_sequence(self) -> int:
        return self.log.latest_sequence

    def events_for(self, player_id: str, sequence: int = 0) -> list[dict[str, Any]]:
        self._require_player(player_id)
        return self.log.visible_to(player_id, sequence)

    def public_events(self, sequence: int = 0) -> list[dict[str, Any]]:
        return self.log.public_since(sequence)

    def audit_events(self, sequence: int = 0) -> list[dict[str, Any]]:
        return self.log.audit_since(sequence)

    def current_round_dialogue_for(self, player_id: str) -> list[dict[str, Any]]:
        """返回玩家当前轮依法可见的对话投影。

        这是受限 ``read_current_round_dialogue`` 工具的唯一数据源。历史仍完整
        保存在引擎记录中，但这里只返回当前轮的发言，并先经过玩家可见性过滤；
        不暴露原始事件的 recipients、隐藏 payload 或其他审计字段。
        """

        self._require_player(player_id)
        visible_events = self.events_for(player_id, 0)
        all_events = self.audit_events(0)
        round_markers = {
            "NIGHT_STARTED",
            "DAY_STARTED",
            "SHERIFF_ELECTION_STARTED",
        }
        marker_sequences: list[int] = []
        for event in all_events:
            payload = event.get("payload") or {}
            if (
                event.get("type") in round_markers
                and payload.get("round") == self.round
            ):
                marker_sequences.append(int(event["seq"]))
        start_sequence = max(marker_sequences, default=0)
        dialogue: list[dict[str, Any]] = []
        for event in visible_events:
            if int(event.get("seq", 0)) < start_sequence:
                continue
            if event.get("type") != "PLAYER_SPOKE":
                continue
            payload = event.get("payload") or {}
            text = payload.get("text")
            if not isinstance(text, str) or not text.strip():
                continue
            dialogue.append(
                {
                    "seq": int(event["seq"]),
                    "speaker_id": str(payload.get("player_id", "")),
                    "stage": str(payload.get("stage", "")),
                    "channel": str(event.get("channel", "public")),
                    "text": text,
                }
            )
        return dialogue

    def role_assignments(self) -> dict[str, str]:
        return deepcopy(self.roles)

    def record_metadata(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "game_id": self.game_id,
            "seed": self.seed,
            "rule_set": asdict(self.rules),
            "players": [
                {"id": player.player_id, "name": player.name, "seat": player.seat}
                for player in self.players
            ],
            "role_assignment": {
                "mode": self.role_assignment_mode,
                "fixed_roles": deepcopy(self.fixed_roles),
            },
        }

    def record_snapshot(self) -> dict[str, Any]:
        return {
            "latest_event_seq": self.latest_event_sequence,
            "public_state": self.public_state(),
            "audit_state": self.audit_state(),
        }

    # ------------------------------------------------------------------
    # 内部规则辅助函数。
    # ------------------------------------------------------------------

    def alive_players(self) -> list[Player]:
        return [player for player in self.players if player.alive]

    def day_voters(self) -> list[Player]:
        """返回本轮仍拥有白天投票权的存活玩家。"""

        return [
            player
            for player in self.alive_players()
            if self.can_day_vote(player.player_id)
        ]

    def can_day_vote(self, player_id: str) -> bool:
        """白痴第一次免死后，默认继续存活但永久失去投票权。"""

        player = self._require_player(player_id)
        if not player.alive:
            return False
        if self.roles.get(player_id) != ROLE_IDIOT:
            return True
        return not bool(self.role_state.get(player_id, {}).get("lost_vote_right"))

    def has_pending_hunter_reactions(self) -> bool:
        """供 Runner 在死亡结算后决定是否进入猎人 reaction loop。"""

        return bool(self._pending_hunter_reactions)

    def has_pending_sheriff_badge_resolution(self) -> bool:
        return bool(self._pending_sheriff_badge_holders)

    def has_pending_last_words(self) -> bool:
        return bool(self._pending_last_words)

    def has_pending_post_death_actions(self) -> bool:
        """死亡后的猎人、警徽与遗言均必须在胜负判定前完成。"""

        return (
            self.has_pending_hunter_reactions()
            or self.has_pending_sheriff_badge_resolution()
            or self.has_pending_last_words()
        )

    def wolf_players(self) -> list[Player]:
        return [
            player
            for player in self.alive_players()
            if self.roles.get(player.player_id) == ROLE_WOLF
        ]

    def players_with_role(self, role: str) -> list[Player]:
        return [
            player
            for player in self.alive_players()
            if self.roles.get(player.player_id) == role
        ]

    def _initial_role_state(self, role: str) -> dict[str, Any]:
        if role == ROLE_WITCH:
            return {"antidote_available": True, "poison_available": True}
        if role == ROLE_SEER:
            return {"inspections": {}}
        if role == ROLE_GUARD:
            return {"last_protected_id": None}
        if role == ROLE_HUNTER:
            return {"shot_available": True}
        if role == ROLE_IDIOT:
            return {"revealed": False, "lost_vote_right": False}
        return {}

    @staticmethod
    def _validate_fixed_roles(
        fixed_roles: Mapping[str, str] | None, player_ids: Iterable[str]
    ) -> dict[str, str]:
        """校验人工指定身份，不让它绕过牌堆数量和角色合法性。"""

        if fixed_roles is None:
            return {}
        normalized = {str(player_id): str(role) for player_id, role in fixed_roles.items()}
        known_players = {str(player_id) for player_id in player_ids}
        unknown_players = sorted(set(normalized) - known_players)
        if unknown_players:
            raise ValueError(f"fixed_roles 包含不存在的玩家：{'、'.join(unknown_players)}")
        unknown_roles = sorted(set(normalized.values()) - set(self_role for self_role in (
            ROLE_WOLF,
            ROLE_VILLAGER,
            ROLE_SEER,
            ROLE_WITCH,
            ROLE_GUARD,
            ROLE_HUNTER,
            ROLE_IDIOT,
        )))
        if unknown_roles:
            raise ValueError(f"fixed_roles 包含不支持的身份：{'、'.join(unknown_roles)}")
        # 牌堆数量检查在实例方法中完成；这里仅返回规范化映射，调用方会传入
        # 当前 RuleSet 的牌堆以便给出清晰错误。
        return normalized

    @staticmethod
    def _count_values(values: Mapping[str, str]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for role in values.values():
            counts[role] = counts.get(role, 0) + 1
        return counts

    def _resolve_plurality(
        self,
        actions: list[dict[str, Any]],
        action_kind: str,
        tie_policy: str,
        *,
        vote_weights: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        counts: dict[str, float] = {}
        for action in actions:
            if action["kind"] != action_kind or not action.get("target_id"):
                continue
            target_id = action["target_id"]
            weight = (
                float(vote_weights.get(action["player_id"], 1))
                if vote_weights is not None
                else 1.0
            )
            counts[target_id] = counts.get(target_id, 0) + weight
        if not counts:
            return {"target": None, "tied_targets": [], "counts": counts}
        highest = max(counts.values())
        tied_targets = sorted(
            target_id for target_id, count in counts.items() if count == highest
        )
        if len(tied_targets) == 1:
            return {"target": tied_targets[0], "tied_targets": [], "counts": counts}
        if tie_policy == "random":
            return {
                "target": self._random.choice(tied_targets),
                "tied_targets": tied_targets,
                "counts": counts,
            }
        return {"target": None, "tied_targets": tied_targets, "counts": counts}

    def _kill_players(
        self,
        player_ids: Iterable[str],
        *,
        eligible_for_last_words: bool = False,
    ) -> list[str]:
        """死亡结算；默认不授予遗言，白天放逐须显式标记。"""

        killed: list[str] = []
        seen: set[str] = set()
        for player_id in player_ids:
            if player_id in seen:
                continue
            seen.add(player_id)
            player = self._find_player(player_id)
            if player is None or not player.alive:
                continue
            player.alive = False
            killed.append(player_id)
        if killed:
            self._death_order.extend(killed)
            self._queue_hunter_reactions(killed)
            self._queue_sheriff_badge_holders(killed)
            if eligible_for_last_words or not self.rules.last_words_day_elimination_only:
                self._queue_last_words(killed)
        return killed

    def _post_death_resume_phase(self) -> str:
        """死亡发生在猎人反应中时，恢复到其原昼夜结算阶段。"""

        if self.phase == PHASE_HUNTER_REACTION and self._hunter_reaction_resume_phase:
            return self._hunter_reaction_resume_phase
        return self.phase

    def _queue_hunter_reactions(self, player_ids: Iterable[str]) -> None:
        """把刚死亡、尚有子弹的猎人加入 reaction 队列。"""

        hunter_ids = [
            player_id
            for player_id in player_ids
            if self.roles.get(player_id) == ROLE_HUNTER
            and self.role_state.get(player_id, {}).get("shot_available")
        ]
        if not hunter_ids:
            return
        if self._hunter_reaction_resume_phase is None:
            self._hunter_reaction_resume_phase = self._post_death_resume_phase()
        for player_id in hunter_ids:
            self.role_state[player_id]["shot_available"] = False
            self._pending_hunter_reactions.append(player_id)
        self._emit(
            "HUNTER_REACTIONS_QUEUED",
            visibility=VISIBILITY_AUDIT,
            channel=CHANNEL_AUDIT,
            payload={"hunter_ids": hunter_ids, "resume_phase": self._hunter_reaction_resume_phase},
        )

    def _queue_sheriff_badge_holders(self, player_ids: Iterable[str]) -> None:
        if not self.sheriff_id:
            return
        killed = set(player_ids)
        if self.sheriff_id not in killed:
            return
        if self.sheriff_id in self._pending_sheriff_badge_holders:
            return
        if self._sheriff_badge_resume_phase is None:
            self._sheriff_badge_resume_phase = self._post_death_resume_phase()
        self._pending_sheriff_badge_holders.append(self.sheriff_id)
        self._emit(
            "SHERIFF_BADGE_RESOLUTION_QUEUED",
            visibility=VISIBILITY_AUDIT,
            channel=CHANNEL_AUDIT,
            payload={
                "holder_id": self.sheriff_id,
                "resume_phase": self._sheriff_badge_resume_phase,
            },
        )

    def _queue_last_words(self, player_ids: Iterable[str]) -> None:
        if self.rules.last_words_count <= 0:
            return
        queued: list[str] = []
        for player_id in player_ids:
            if len(self._last_words_eligible_ids) >= self.rules.last_words_count:
                break
            if player_id in self._last_words_eligible_ids:
                continue
            self._last_words_eligible_ids.append(player_id)
            self._pending_last_words.append(player_id)
            queued.append(player_id)
        if not queued:
            return
        if self._last_words_resume_phase is None:
            self._last_words_resume_phase = self._post_death_resume_phase()
        self._emit(
            "LAST_WORDS_QUEUED",
            visibility=VISIBILITY_AUDIT,
            channel=CHANNEL_AUDIT,
            payload={
                "player_ids": queued,
                "eligible_player_ids": list(self._last_words_eligible_ids),
                "resume_phase": self._last_words_resume_phase,
            },
        )

    def _emit_player_died(self, player_id: str) -> None:
        payload: dict[str, Any] = {"player_id": player_id}
        if self.rules.reveal_role_on_death:
            payload["role"] = self._publicly_reveal_role(player_id)
        self._emit(
            "PLAYER_DIED",
            visibility=VISIBILITY_PUBLIC,
            channel=CHANNEL_PUBLIC,
            payload=payload,
        )

    def _publicly_reveal_role(self, player_id: str) -> str:
        role = self.roles[player_id]
        self.publicly_revealed_roles[player_id] = role
        return role

    def _finish_if_no_pending_post_death_actions(self) -> bool:
        if self.has_pending_post_death_actions():
            return False
        return self._finish_if_needed()

    def _finish_if_no_pending_hunter_reactions(self) -> bool:
        """兼容旧调用名；现已同时等待警徽和遗言。"""

        return self._finish_if_no_pending_post_death_actions()

    def _finish_if_needed(self) -> bool:
        alive = self.alive_players()
        alive_wolves = sum(
            self.roles.get(player.player_id) == ROLE_WOLF for player in alive
        )
        if alive_wolves == 0:
            winner = TEAM_VILLAGE
        elif self.rules.wolf_win_condition == "slaughter_side":
            alive_villagers = sum(
                self.roles.get(player.player_id) == ROLE_VILLAGER for player in alive
            )
            alive_special_villagers = sum(
                self.roles.get(player.player_id) not in {ROLE_WOLF, ROLE_VILLAGER}
                for player in alive
            )
            initial_villagers = self.rules.role_deck.count(ROLE_VILLAGER)
            initial_special_villagers = (
                len(self.rules.role_deck)
                - self.rules.role_deck.count(ROLE_WOLF)
                - initial_villagers
            )
            if (
                (initial_villagers > 0 and alive_villagers == 0)
                or (
                    initial_special_villagers > 0
                    and alive_special_villagers == 0
                )
            ):
                winner = TEAM_WOLF
            else:
                return False
        else:
            alive_village = len(alive) - alive_wolves
            if alive_wolves >= alive_village:
                winner = TEAM_WOLF
            else:
                return False

        self.status = "finished"
        self.winner = winner
        self.phase = PHASE_FINISHED
        self._phase_nonce += 1
        payload: dict[str, Any] = {"winner": winner, "round": self.round}
        if self.rules.reveal_all_roles_at_end:
            payload["roles"] = deepcopy(self.roles)
        self._emit(
            "GAME_FINISHED",
            visibility=VISIBILITY_PUBLIC,
            channel=CHANNEL_PUBLIC,
            payload=payload,
        )
        self._emit(
            "GAME_FINISHED_AUDIT",
            visibility=VISIBILITY_AUDIT,
            channel=CHANNEL_AUDIT,
            payload={"winner": winner, "roles": deepcopy(self.roles)},
        )
        return True

    def _has_living_sheriff(self) -> bool:
        return bool(
            self.sheriff_id
            and (player := self._find_player(self.sheriff_id)) is not None
            and player.alive
        )

    def _emit_day_speech_order_set(self) -> None:
        ordered = self.day_discussion_players()
        self._emit(
            "DAY_SPEECH_ORDER_SET",
            visibility=VISIBILITY_PUBLIC,
            channel=CHANNEL_PUBLIC,
            payload={
                "round": self.round,
                "reference_seat": self._day_speech_reference_seat,
                "reference_is_dawn_death": self._day_speech_reference_is_death,
                "direction": self._day_speech_direction,
                "speaker_ids": [player.player_id for player in ordered],
                "sheriff_id": self.sheriff_id if self._has_living_sheriff() else None,
            },
        )

    def _find_player(self, player_id: str) -> Player | None:
        return next(
            (player for player in self.players if player.player_id == str(player_id)),
            None,
        )

    def _require_player(self, player_id: str) -> Player:
        player = self._find_player(player_id)
        if player is None:
            raise RuleViolationError("未知玩家", player_id=player_id)
        return player

    def _require_alive_player(self, player_id: str) -> Player:
        player = self._require_player(player_id)
        if not player.alive:
            raise RuleViolationError("死亡玩家不能行动", player_id=player_id)
        return player

    def _require_pending_hunter_reactor(self, player_id: str) -> Player:
        player = self._require_player(player_id)
        if self.roles.get(player_id) != ROLE_HUNTER:
            raise RuleViolationError("只有猎人可以触发死亡开枪")
        if not self._pending_hunter_reactions or self._pending_hunter_reactions[0] != player_id:
            raise RuleViolationError("当前不是该猎人的开枪时机")
        if player.alive:
            raise RuleViolationError("猎人必须在死亡后触发开枪")
        return player

    def _require_pending_last_words_speaker(self, player_id: str) -> Player:
        player = self._require_player(player_id)
        if not self._pending_last_words or self._pending_last_words[0] != player_id:
            raise RuleViolationError("当前不是该玩家发表遗言的时机")
        if player.alive:
            raise RuleViolationError("遗言只能由死亡玩家发表")
        return player

    def _require_pending_sheriff_badge_holder(self, player_id: str) -> Player:
        player = self._require_player(player_id)
        if (
            not self._pending_sheriff_badge_holders
            or self._pending_sheriff_badge_holders[0] != player_id
        ):
            raise RuleViolationError("当前不是该警长处理警徽的时机")
        if player.alive:
            raise RuleViolationError("警徽交接只能由死亡警长处理")
        return player

    def _require_running(self) -> None:
        if self.status != "running":
            raise RuleViolationError("游戏未处于进行中状态", status=self.status)

    def _require_phase(self, phase: str) -> None:
        self._require_running()
        if self.phase != phase:
            raise RuleViolationError("当前不在所需阶段", expected=phase, actual=self.phase)

    def _emit(
        self,
        event_type: str,
        *,
        visibility: str,
        channel: str,
        payload: dict[str, Any] | None = None,
        recipients: Iterable[str] = (),
    ) -> None:
        self.log.emit(
            event_type=event_type,
            visibility=visibility,
            channel=channel,
            payload=payload,
            recipients=recipients,
        )

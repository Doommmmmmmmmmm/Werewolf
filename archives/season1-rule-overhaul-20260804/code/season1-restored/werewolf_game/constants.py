"""全项目共用的规则常量。

常量保持为字符串，便于直接写入 JSON 记录，也让模型行动格式一目了然。
"""

TEAM_WOLF = "wolf"
TEAM_VILLAGE = "village"

ROLE_WOLF = "wolf"
ROLE_VILLAGER = "villager"
ROLE_SEER = "seer"
ROLE_WITCH = "witch"
ROLE_GUARD = "guard"
ROLE_HUNTER = "hunter"
ROLE_IDIOT = "idiot"

ACTION_SPEAK = "speak"
ACTION_PASS = "pass"
ACTION_WOLF_KILL_VOTE = "wolf_kill_vote"
ACTION_SEER_INSPECT = "seer_inspect"
ACTION_WITCH_HEAL = "witch_heal"
ACTION_WITCH_POISON = "witch_poison"
ACTION_GUARD_PROTECT = "guard_protect"
ACTION_HUNTER_SHOOT = "hunter_shoot"
ACTION_DAY_VOTE = "day_vote"
ACTION_LAST_WORDS = "last_words"
ACTION_SHERIFF_CANDIDATE = "sheriff_candidate"
ACTION_SHERIFF_VOTE = "sheriff_vote"
ACTION_SHERIFF_SPEECH_ORDER = "sheriff_speech_order"
ACTION_SHERIFF_BADGE_TRANSFER = "sheriff_badge_transfer"
ACTION_SHERIFF_BADGE_DESTROY = "sheriff_badge_destroy"

PHASE_NOT_STARTED = "not_started"
PHASE_NIGHT_WOLF_DISCUSSION = "night_wolf_discussion"
PHASE_NIGHT_GUARD = "night_guard"
PHASE_NIGHT_WOLF_VOTE = "night_wolf_vote"
PHASE_NIGHT_SEER = "night_seer"
PHASE_NIGHT_WITCH = "night_witch"
PHASE_NIGHT_RESOLVE = "night_resolve"
PHASE_DAY_DISCUSSION = "day_discussion"
PHASE_DAY_VOTE = "day_vote"
PHASE_HUNTER_REACTION = "hunter_reaction"
PHASE_LAST_WORDS = "last_words"
PHASE_SHERIFF_CANDIDACY = "sheriff_candidacy"
PHASE_SHERIFF_ELECTION_SPEECH = "sheriff_election_speech"
PHASE_SHERIFF_ELECTION_VOTE = "sheriff_election_vote"
PHASE_SHERIFF_SPEECH_ORDER = "sheriff_speech_order"
PHASE_SHERIFF_BADGE = "sheriff_badge"
PHASE_FINISHED = "finished"

VISIBILITY_PUBLIC = "public"
VISIBILITY_PRIVATE = "private"
VISIBILITY_TEAM = "team"
VISIBILITY_AUDIT = "audit"

CHANNEL_PUBLIC = "public"
CHANNEL_WOLF = "wolf"
CHANNEL_PRIVATE = "private"
CHANNEL_AUDIT = "audit"

ALL_ROLES = frozenset(
    {
        ROLE_WOLF,
        ROLE_VILLAGER,
        ROLE_SEER,
        ROLE_WITCH,
        ROLE_GUARD,
        ROLE_HUNTER,
        ROLE_IDIOT,
    }
)
OPTIONAL_ROLES = frozenset({ROLE_GUARD, ROLE_HUNTER, ROLE_IDIOT})
TIE_POLICIES = frozenset({"no_elimination", "no_kill", "random"})


def team_for_role(role: str) -> str:
    """返回身份所属阵营。"""

    return TEAM_WOLF if role == ROLE_WOLF else TEAM_VILLAGE


def public_phase_for(internal_phase: str) -> str:
    """把内部夜间子阶段投影为对玩家可见的“night”。"""

    if internal_phase.startswith("night_"):
        return "night"
    if internal_phase in {
        PHASE_SHERIFF_CANDIDACY,
        PHASE_SHERIFF_ELECTION_SPEECH,
        PHASE_SHERIFF_ELECTION_VOTE,
    }:
        return "sheriff_election"
    if internal_phase in {
        PHASE_HUNTER_REACTION,
        PHASE_LAST_WORDS,
        PHASE_SHERIFF_BADGE,
        PHASE_SHERIFF_SPEECH_ORDER,
    }:
        return "reaction"
    return internal_phase

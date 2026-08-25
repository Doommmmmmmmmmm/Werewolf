"""可配置规则集。

规则放在数据对象中，不依赖 LLM prompt，因此模型无法改变裁决逻辑。
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from .constants import (
    ALL_ROLES,
    OPTIONAL_ROLES,
    ROLE_GUARD,
    ROLE_HUNTER,
    ROLE_IDIOT,
    ROLE_SEER,
    ROLE_VILLAGER,
    ROLE_WITCH,
    ROLE_WOLF,
    TIE_POLICIES,
)


DEFAULT_ROLE_DECK = (
    ROLE_WOLF,
    ROLE_WOLF,
    ROLE_SEER,
    ROLE_WITCH,
    ROLE_VILLAGER,
    ROLE_VILLAGER,
    ROLE_VILLAGER,
)


# 标准预设。特殊身份数量会随人数逐步增加，但仍可通过 optional_roles
# 以“用一个平民换一张特殊身份”的方式做小范围定制。
ROLE_DECKS_BY_PLAYER_COUNT: dict[int, tuple[str, ...]] = {
    7: DEFAULT_ROLE_DECK,
    8: (
        ROLE_WOLF,
        ROLE_WOLF,
        ROLE_SEER,
        ROLE_WITCH,
        ROLE_GUARD,
        ROLE_VILLAGER,
        ROLE_VILLAGER,
        ROLE_VILLAGER,
    ),
    9: (
        ROLE_WOLF,
        ROLE_WOLF,
        ROLE_WOLF,
        ROLE_SEER,
        ROLE_WITCH,
        ROLE_GUARD,
        ROLE_VILLAGER,
        ROLE_VILLAGER,
        ROLE_VILLAGER,
    ),
    10: (
        ROLE_WOLF,
        ROLE_WOLF,
        ROLE_WOLF,
        ROLE_SEER,
        ROLE_WITCH,
        ROLE_GUARD,
        ROLE_HUNTER,
        ROLE_VILLAGER,
        ROLE_VILLAGER,
        ROLE_VILLAGER,
    ),
    11: (
        ROLE_WOLF,
        ROLE_WOLF,
        ROLE_WOLF,
        ROLE_SEER,
        ROLE_WITCH,
        ROLE_GUARD,
        ROLE_HUNTER,
        ROLE_IDIOT,
        ROLE_VILLAGER,
        ROLE_VILLAGER,
        ROLE_VILLAGER,
    ),
    12: (
        ROLE_WOLF,
        ROLE_WOLF,
        ROLE_WOLF,
        ROLE_WOLF,
        ROLE_SEER,
        ROLE_WITCH,
        ROLE_GUARD,
        ROLE_HUNTER,
        ROLE_VILLAGER,
        ROLE_VILLAGER,
        ROLE_VILLAGER,
        ROLE_VILLAGER,
    ),
}


@dataclass(frozen=True)
class RuleSet:
    """一局游戏的完整可变规则。"""

    rule_id: str = "default-seven-player"
    role_deck: tuple[str, ...] = DEFAULT_ROLE_DECK
    reveal_role_on_death: bool = False
    reveal_all_roles_at_end: bool = True
    reveal_individual_votes: bool = True
    allow_day_vote_abstain: bool = True
    wolf_tie_policy: str = "no_kill"
    day_tie_policy: str = "no_elimination"
    # 新赛季采用屠边规则。狼人可以将任意存活玩家（包括狼队友）作为刀口，
    # 从而让自刀、骗药和刀口做局成为真实可选策略。
    wolves_can_target_wolves: bool = True
    wolf_win_condition: str = "slaughter_side"
    last_words_count: int = 3
    # 默认只有白天放逐出局者可以占用遗言名额；设为 False 可复现旧版
    # “全局最先死亡者”遗言规则。
    last_words_day_elimination_only: bool = True
    enable_sheriff_election: bool = True
    sheriff_vote_weight: float = 1.5
    seer_can_inspect_self: bool = False
    seer_can_repeat_inspect: bool = False
    witch_can_self_save: bool = True
    witch_can_poison_self: bool = False
    guard_can_protect_self: bool = True
    guard_can_repeat_protect: bool = False
    idiot_survives_first_day_elimination: bool = True
    idiot_can_vote_after_reveal: bool = False
    max_day_speech_chars: int = 200
    max_wolf_speech_chars: int = 30
    require_chinese_speech: bool = True
    allow_latin_letters_in_speech: bool = False

    def __post_init__(self) -> None:
        if len(self.role_deck) < 3:
            raise ValueError("role_deck 至少需要三张身份牌")
        if any(role not in ALL_ROLES for role in self.role_deck):
            raise ValueError("role_deck 包含不支持的身份")
        wolves = self.role_deck.count(ROLE_WOLF)
        if wolves == 0 or wolves == len(self.role_deck):
            raise ValueError("身份牌中必须同时有狼人和好人阵营")
        if self.wolf_tie_policy not in TIE_POLICIES:
            raise ValueError("不支持的狼人平票规则")
        if self.day_tie_policy not in TIE_POLICIES:
            raise ValueError("不支持的白天平票规则")
        if self.wolf_win_condition not in {"parity", "slaughter_side"}:
            raise ValueError("不支持的狼人胜利条件")
        if self.last_words_count < 0:
            raise ValueError("遗言人数不能为负数")
        if self.sheriff_vote_weight < 1:
            raise ValueError("警长票权必须不少于一票")
        if self.max_day_speech_chars < 1 or self.max_wolf_speech_chars < 1:
            raise ValueError("发言字数上限必须为正整数")


def create_default_rules(**overrides: object) -> RuleSet:
    """返回默认 7 人局规则；可用 snake_case 字段覆盖。"""

    return replace(RuleSet(), **overrides)


def create_rules_for_player_count(
    player_count: int,
    optional_roles: tuple[str, ...] | list[str] = (),
    **overrides: object,
) -> RuleSet:
    """创建 7–12 人标准局，按需用平民替换为可选身份。

    ``optional_roles`` 仅接受守卫、猎人、白痴。已包含在该人数预设中的
    身份会被忽略；其余身份会各替换一名平民。因此 7 人局可传
    ``("guard", "hunter")``，得到 2 狼、预言家、女巫、守卫、猎人、平民。
    想使用重复身份或非标准阵容时，可直接构造 :class:`RuleSet`。
    """

    try:
        deck = list(ROLE_DECKS_BY_PLAYER_COUNT[int(player_count)])
    except (KeyError, TypeError, ValueError) as error:
        supported = "、".join(str(count) for count in sorted(ROLE_DECKS_BY_PLAYER_COUNT))
        raise ValueError(f"仅支持 {supported} 人配置") from error

    for role in optional_roles:
        if role not in OPTIONAL_ROLES:
            supported = "、".join(sorted(OPTIONAL_ROLES))
            raise ValueError(f"optional_roles 仅支持：{supported}")
        if role in deck:
            continue
        try:
            villager_index = deck.index(ROLE_VILLAGER)
        except ValueError as error:
            raise ValueError("当前预设没有可替换的平民席位") from error
        deck[villager_index] = role

    return RuleSet(
        rule_id=f"standard-{int(player_count)}-player",
        role_deck=tuple(deck),
        **overrides,
    )
